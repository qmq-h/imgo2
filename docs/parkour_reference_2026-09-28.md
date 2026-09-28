# parkour 参考项目的流程与配平（2026-09-28）：对我们 CMoE 的含义

对象：`/root/Desktop/parkour/`（ZiwenZhuang/parkour，CoRL 2023，HEAD `5e1f413`）。只读参考。
**本记录纠正了我在对话里先说过的一句话**，见 §2。

## 1. 流程：三段（Go2）／四步（Go1/A1 论文版）

| 阶段 | 任务 | 迭代数 | 地形 | 权重衔接 |
|---|---|---|---|---|
| ① 走地 | `go2` / `a1` | 2000 / 5000–20000 | `TerrainPerlin`（粗糙地）或空 `BarrierTrack` | 从零（`resume=False`） |
| ② 跑酷 oracle | `go2_field` / `a1_field` | 38000 / 20000 | `BarrierTrack`（10 行 × 40 列、10 种障碍、`curriculum=True`） | **`resume=True` + `load_run=logs/rough_go2/{走 run}`**（`go2_field_config.py:198-201`） |
| ②′ 单技能微调（论文版） | `a1_jump/leap/crawl/tilt/down` | 20000 | 单障碍 `BarrierTrack`（`options=[一种]`） | **`resume=True` + `load_run="{Your trained walking model directory}"`**（`a1_jump_config.py:107-108` 等，且每行下面都有一串注释掉的上一代 run） |
| ③ 组合 teacher | mutex（`ActorCriticClimbMutex`） | — | — | `sub_policy_paths` 按障碍 ID 列各技能 run |
| ④ 蒸馏 | `a1_distill` / `go2_distill` | 60000–80000 | 多障碍 `BarrierTrack` | `resume`/`load_run` + DAgger；`replace_encoder0` |

* 加载的是**整份算法 state_dict**（含 `log_std` 与 Adam 动量）：`on_policy_runner.py:258-275` → `ppo.py:236-243`。
* **技能任务在开源版里没有注册**（`envs/__init__.py:56-71` 只有 `go1_field/go1_distill/go2/go2_field/go2_distill/a1/a1_remote/go1_remote`）⇒ 想复现得自己加注册行；README 里的 `a1_climb` 任务不存在。
* **真实证据**：`go1_ckpts/*.zip` 里有**训练当时的 `config.json` 快照**，比仓库里的模板硬（`runner.load_run` 是 8 代 `WalkByRemote` 的链条，checkpoint 是 `model_107500.pt` ⇒ 多轮 20k 累加）。

## 2. ⚠️ 纠正：**parkour 从头到尾都没用步态塑形项**

我在对话里先说"走地阶段用 `feet_air_time 1.0` 把步态练好、后训练撤掉"——**不成立**。核验：

| 对象 | 是否有步态项 |
|---|---|
| `a1_config.py:86`（A1 **模板**） | `class scales(LeggedRobotCfg.rewards.scales)` **继承了基类** ⇒ 含 `feet_air_time = 1.0`（这一项在模板里是有的） |
| `go2_config.py:170`（go2 走地） | `class scales:` **不带基类** ⇒ **无任何步态项**（只有 `tracking_*`/`stand_still`/`dof_error*`/`exceed_*`/`energy`/`dof_vel_limits`） |
| **真实走策略**（`go1_ckpts/Oct24_16-46-16_107k_WalkByRemote...zip::config.json`） | `rewards.scales` = `{tracking_lin_vel 2.0, tracking_ang_vel 1.0, action_rate −0.1, collision −0.2, orientation −0.1, stand_still −0.6, legs_energy_substeps −1e−5, dof_acc −2.5e−7, exceed_dof_pos_limits −0.4, exceed_torque_limits_l1norm −0.4, torques −1e−5}` ⇒ **没有 `feet_air_time`** |
| 全仓 | 唯一与"gait"有关的原文是三条注释 `# penalty for walking gait, probably no need`（`a1_remote_config.py:58` 等） |

⇒ 准确说法：**他们的步态是"任务定义 + 别乱来惩罚"的副产物，不是被奖励锁住的**。

## 3. 那步态靠什么在阶段间不漂

1. **任务定义宽**：`tracking_lin_vel 2.0` + `tracking_ang_vel 1.0`，命令含侧向与 yaw ⇒ trot 是"跟任意方向速度"的自然解。
2. **"别乱来"项很强且跨阶段只增不减**（真实走 → field）：
   `action_rate −0.1`、`collision −0.2 → −10`、`orientation −0.1 → **−4.0**`、`lin_vel_z → −1.0`、`ang_vel_xy −0.05`、`exceed_* −0.4 → −0.8`、`stand_still −0.6`。
   ⇒ 拖行／跛行／晃身／蹭腿**全被重罚**。
3. **换掉的只有任务项**（速度跟踪 → 世界系跟踪 + 穿障 `penetrate_depth/volume`），姿态/平滑/限位/能耗**全保留**。
4. **权重 + `log_std` + 优化器状态整体继承** ⇒ 起步就在"已收敛的走法"附近，探索噪声不重置变大。
5. **最后一段是冻结教师的纯行为克隆**：真实 distill 的 `rewards.scales = {}`（空），teacher `@torch.no_grad()`（`actor_critic_field_mutex.py:105-137`）⇒ **那一段没有任何奖励能改步态**。
6. **部署时另存走策略**（`Deploy-Go1.md:82` 的 `--walkdir`）⇒ 他们**不要求跑酷策略保留优雅步态**。
7. 步态约束只在**事件级**触发：`sync_all_legs_cond −0.3` 按 `engaging_obstacle == jump` 掩码（`legged_robot_field.py:602-622`）、`dof_error_cond` 在非障碍段强制回默认姿态（`:637-647`）。

## 4. 配平对照：我们 v3 vs parkour field（go1，`go1_field_config.py:117-136`）

| 口径 | parkour field | 我们 CMoE v3 | 差距 |
|---|---|---|---|
| 任务项 | `tracking_world_vel 3.0` + `tracking_ang_vel 0.05` = **3.05** | `track_world_vel_xy 5.0` + `track_ang_vel_z 0.6` = **5.6** | 我们**强 1.8×** |
| 姿态 | `orientation **−4.0**`、`lin_vel_z −1.0`、`ang_vel_xy −0.05` | `flat_orientation_l2 −5.0`（**仅 flat/rough 生效**）、`lin_vel_z −2.0`、`ang_vel_xy −0.05` | 数值相仿，但**我们按地形豁免了 9/11 类** |
| 碰撞 | `collision **−10.0**` | `undesired_contacts −0.5` + `contact_forces −0.02` | 我们**弱 20~500×** |
| 平滑 | `action_rate **−0.1**`、`dof_acc −2.5e−7`、`torques −1e−5`、`delta_torques −1e−7` | `action_rate_l2 **−0.01**`、`joint_acc −5e−9`、`joint_torques_l2 −2.5e−6` | 我们**弱 10×** |
| 限位 | `exceed_dof_pos_limits −0.8`、`exceed_torque_limits_l1norm −0.8` | `joint_pos_penalty −0.1`（形式不同） | 我们偏弱 |
| 步态模式项 | **全无** | `feet_air_time 1.0`、`feet_height_body −5`、`variance −8`、`joint_mirror −1`、`feet_slide −0.05` | 我们有、他们没有 |

**读法**：拖行/跛行最容易被"**碰撞 + 平滑 + 姿态**"三项按住，而这恰好是我们最弱的三项（碰撞弱 20~500×、平滑弱 10×、姿态只覆盖 2 类地形）。我们在"奖励里加步态模式项"这条路上反复调，而他们的等价手段是**把非步态的"别乱来"项加到很重**。

## 5. 可迁移清单（按性价比）

| 优先级 | 改动 | 依据 |
|---|---|---|
| **P0** | `undesired_contacts −0.5 → −5~−10` | 对齐 `collision −10`；拖行＝小腿/大腿蹭地 |
| **P0** | `action_rate_l2 −0.01 → −0.1` | 对齐 `action_rate −0.1`（10×）；跛行的动作抖动会被罚 |
| **P1** | `flat_orientation_l2` **收回斜坡豁免**（他们 `orientation` 全地形 −4，含斜坡/台阶） | 我们豁免了 9/11 类 ⇒ 姿态基本没人管 |
| **P1** | 任务项 `track_world_vel_xy 5.0 → 2.0~3.0` | 对齐他们 3.0；把 shaping/task 拉回 PPO 的 ~7:1 |
| **P2** | 最后一段改成**冻结教师 + 纯回归（BC）**，而不是 KL/AMP 在线博弈 | 他们 distill 阶段 `rewards.scales={}`、teacher `no_grad` |
| **P2** | 接受"**两份策略**"（waking + parkour），别要求一份两全 | `Deploy-Go1.md:82` 的 `--walkdir` |
| **P3** | 步态约束改**事件级**（起跳/过沟那一刻）而不是地形级 | `sync_all_legs_cond` 按 `engaging_obstacle==jump` |
| **P3** | 微调时**继承 `log_std` 与优化器状态**（别重置） | `ppo.py:236-243` 整体加载 |

## 6. 限制

* 仓库模板 config 与**真实训练用的 config 已漂移**（例：`go1_remote_config.py:51-75` 写 `tracking_lin_vel=3/alive=1/only_positive_rewards=False`，真实是 `2.0/无 alive/True`）⇒ 照抄权重必须用 `go1_ckpts/*.zip` 里的快照，不要用模板。
* 作者的 run 路径全是机器绝对路径（`/home/zzw/...`、`/localdata_ssd/...`），**仓库里没有那些 run**，无法核对每代超参。
* `docs/` 不存在；全部机制是从代码推断的，**不是作者原话**（作者唯一的表态是"步态惩罚大概率不需要"）。
* 无 GPU，未运行任何训练；本记录的证据来自源码静态阅读 + 用 stub 掉 isaacgym 的方式实例化 cfg 取生效权重（子代理执行）+ 对 `go1_ckpts/*.zip` 内 `config.json` 的直接读取（本人核验）。

## 7. 应用到我们（用户 2026-09-28："amp 这个已经把步态练好了"）

### 7.1 先钉死一个结构差异：**parkour 只在"同契约"内链式加载，我们没有这个前提**

* `PPO.load_state_dict` 是**严格加载**（`rsl_rl/rsl_rl/algorithms/ppo.py:236-243` → `nn.Module.load_state_dict`），
  全仓唯一的 manipulator 是 `replace_encoder0`（换掉 actor 的深度编码器、保留 critic 的），**没有任何
  跨形状 pad/filter 机制**（`rsl_rl/rsl_rl/utils/ckpt_manipulator.py:15-35`）。
* 因此他们的"走 → 技能"能干净加载，是因为**走地那一代就在同一个 env/观测契约里训的**：
  `a1_field_config.py:8-33` 的活跃 `obs_components` ＝ `[proprioception(48), base_pose, robot_config,
  engaging_block, sidewall_distance]`（81 维），而**被注释掉的那块**（`obs_components=[proprioception]`＝48 维、
  带 `privileged_obs_components`）注释原文就是 "configs for training a walk policy" —— 那是**另一条线**
  （`go1_remote`/`a1_remote`，部署时替换站立策略用）。
* 佐证：`go1_ckpts/Nov02...distill*.zip` 的 `teacher_policy.sub_policy_paths[0..1]` 都是
  `WalkForward_pEnergySubsteps2e-5_rTrackVel3e+0_pYawAbs8e-1_...`，run 名里的 `rTrackVel3e+0`／`pYawAbs8e-1`
  正对应 **field 的 scales**（`go1_field_config.py:118-136`）⇒ 走地阶段与技能阶段**同契约**
  （field env + 空障碍赛道）。

⇒ **我们的 AMP 24500（45 维、平地、AMP 判别器奖励）与 CMoE（157/527 维）是跨契约**，parkour 里
**不存在**对应机制；这一步是我们自己造的桥（`install_prior` + `zero_pad_first_layer`，逐位等价 0.0、critic 48→125、
梯度可达新列 7.1，见 §docs/cmoe_trot_warmstart §18）。**在这一点上我们比 parkour 多一个能力**，
但也意味着"照抄 parkour 的链式加载"必须先有这座桥。

### 7.2 v4 提案：**去步态模式化 + 强质量项**（parkour 式后训练）

| 改动 | 现状 → 建议 | 依据 |
|---|---|---|
| 撤掉步态**模式**项 | `feet_air_time 1.0→0`、`feet_air_time_variance −8→0`、`feet_height_body −5→0`、`joint_mirror −1→0`（`feet_gait` 已 0） | parkour 全仓**零步态模式项** |
| 保留 `feet_slide` | −0.05 不变 | 属"质量类"（别打滑），他们用 `collision`+`action_rate` 起同样作用 |
| 平滑 | `action_rate_l2 −0.01 → **−0.1**` | 对齐 `action_rate −0.1`（10×） |
| 碰撞 | `undesired_contacts −0.5 → **−5~−10**`、`contact_forces −0.02 → −0.1` | 对齐 `collision −10` |
| **新增 `lazy_stop`** | 我们没有这一项 | 他们用一个"**下发了速度指令却不动**就罚"的项（`legged_robot.py:1859-1865`，`go2_field −3.0`）；**正对我们实测的拖行**（level 6：0.127 m/s vs 指令 0.633）。注意我们的 `stand_still −0.1` 因 `rel_standing_envs=0`（不给零命令）**永远不触发** |
| 限位 | 加回 `joint_pos_limits`（`gaitfree` 之外两条链现在是 **0**，被移除了） | 他们 `exceed_dof_pos_limits −0.4~−0.8` |
| 任务项 | `track_world_vel_xy_exp 5.0 → **2.5~3.0**` | 对齐 `tracking_world_vel 3.0` |
| 水平罚 | 把"障碍地形归零"改成"**障碍地形打折**"（如 −1.5~−2，平地 −5） | 他们 `orientation −4` **全局**（含斜坡/台阶） |
| 兜底（可选） | 最后一段用 AMP/walk 策略作**冻结教师做纯回归** | 他们 distill 阶段 `rewards.scales={}`、teacher `no_grad` |

**风险**：这样"步态全靠先验 + 质量项"，先验仍可能被缓慢改写（我们实测 `prior_weight_rel_drift` ≈0.0002/轮）；
所以 §7.2 最后一行（BC 兜底）与"必要时两份策略"要保留在选项里。
