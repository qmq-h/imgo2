# CMoE 训练准备：奖励设置复核与训练指令（2026-09-23）

- 分支／提交：`imgo2_CMoE` @ `6220e43`（奖励／地形部分；冒烟回执 §10 跑在 `37b9206` 上）
- 任务：`Imgo2-basemove-rough-cmoe`（训练）、`Imgo2-basemove-rough-cmoe-play`（回放／导出）
- 环境类：`Imgo2CMoERoughEnvCfg`（`CMoE_env_cfg.py`），算法栈：`rl_lab` 的 CMoE PPO
- 本文只复核**配置与入口**，未在训练机运行；所有“已验证”结论以训练机实际输出为准。

> **契约变更提示（2026-09-24）**：本文正文里出现的 `terrain=187`、`actor=637`、`expert=267`、`critic=235` 都是**该日期之前**的旧几何（17×11 @ 1.6×1.0 m）。自 2026-09-24 起 CMoE 的地形扫描按用户要求改为原版的 **77 维**（11×7 @ 0.1 m，覆盖 1.0×0.6 m），对应 **actor 527 / expert 157 / critic 125**；旧 checkpoint 与新配置不兼容（见移植记录同日的「观测契约变更」一节）。

## 1. 奖励的继承与生效规则

CMoE 的奖励以 PPO rough 配方为底，再在 `CMoE_env_cfg.py` 里追加一项并覆盖若干权重（`6220e43` 起）：

```
Imgo2CMoERoughEnvCfg            (base_move/CMoE_env_cfg.py: 追加 CMoERewardsCfg.feet_edge，覆盖权重/地形/终止/reset)
  └─ Imgo2RoughEnvCfg           (base_move/rough_env_cfg.py: 设权重、目标值、link/joint 过滤)
       └─ LocomotionVelocityRoughEnvCfg  (velocity_env_cfg.py: 定义全部 term 与函数名，权重默认 0)
```

两个容易误读的点：

1. `velocity_env_cfg.py` 里几乎所有 term 的默认权重都是 `0.0`；**只有 `rough_env_cfg.py` 与 CMoE 配置显式赋了非零权重的项才生效**。
2. `disable_zero_weight_rewards()`（`velocity_env_cfg.py:738`）会把权重仍为 `0` 的 term 置 `None`。基类 `Imgo2RoughEnvCfg.__post_init__` 只在 `self.__class__.__name__ == "Imgo2RoughEnvCfg"` 时调用它（`rough_env_cfg.py:172`），子类不会继承这一步，因此 `CMoE_env_cfg.py:151` 显式补了一次——**这是 CMoE 配置正确性的关键一行：它也是「清零即等于移除」生效的原因，改动时不要删**。

## 2. 实际生效的奖励项（17 项，自 `6220e43` 起）

> **配方已变（2026-09-23 21:15，`6220e43 add CMoE gap terrain and safety rewards`）**：相对 `37b9206` 新增 `feet_edge −1.0`、`feet_stumble −1.0`，`undesired_contacts` 由 −0.5 加重到 **−5.0**，并把五项固定步态 shaping 清零（`feet_height`、`feet_height_body`、`feet_slide`、`feet_air_time_variance`、`joint_mirror`）。生效项 19 → **17 项**，且首次引入**基座触地终止**（§4）。

下表的权重即 `RewardManager` 每步乘上后求和的值；分项以 `Episode_Reward/<term>` 进 TensorBoard。

| term | 权重 | 方向 | 语义与关键参数 |
|---|---:|---|---|
| `track_lin_vel_xy_exp` | **+1.5** | 奖励 | xy 线速度指数核跟踪，`std=0.5` |
| `track_ang_vel_z_exp` | **+0.6** | 奖励 | z 角速度指数核跟踪，`std=0.5` |
| `feet_air_time` | **+1.0** | 奖励 | 落地瞬间 `(滞空时间 − 0.5 s)`；命令范数 >0.1 才给（**保留**） |
| `base_height_l2` | **−10.0** | 惩罚 | `(base z − (0.30 + 局部地面高度))²`；地面高度取 `height_scanner_base` 射线均值 |
| `undesired_contacts` | **−5.0** | 惩罚 | 除足端外任何 link 受力 >1 N（**本提交由 −0.5 加重 10 倍**） |
| `flat_orientation_l2` | −5.0 | 惩罚 | 重力投影水平分量平方（机身水平度） |
| `lin_vel_z_l2` | −2.0 | 惩罚 | base 竖直速度平方 |
| `feet_edge` | **−1.0** | 惩罚 | **新增（CMoE 专用）**：某足足底 4×4 向下射线网格**部分命中**（`isfinite(z).any(dim=1) & ~all`，即实心与空洞交界）**且该足接触力 >1 N** 时，按足计数惩罚 |
| `feet_stumble` | **−1.0** | 惩罚 | **新增启用**：足端 `any(\|F_xy\| > 4·\|F_z\|)`（撞到竖直面／绊脚），带直立门控 |
| `stand_still` | −0.1 | 惩罚 | 命令范数 <0.05（`command_threshold` 设为 0.05）时的关节偏离 L1 |
| `joint_pos_penalty` | −0.1 | 惩罚 | 关节偏离默认位姿 L1；命令与机体速度都低于阈值时 ×5 |
| `ang_vel_xy_l2` | −0.05 | 惩罚 | 横滚／俯仰角速度平方 |
| `contact_forces` | −0.02 | 惩罚 | 足端接触力超过 100 N 的部分 |
| `action_rate_l2` | −0.01 | 惩罚 | 相邻动作差平方 |
| `joint_torques_l2` | −2.5e-6 | 惩罚 | 关节力矩平方 |
| `joint_power` | −2e-5 | 惩罚 | 关节功率绝对值和 `Σabs(τ·q̇)` |
| `joint_acc_l2` | −5e-9 | 惩罚 | 关节加速度平方 |

补充：

- 每步正向上限仍是 `1.5 + 0.6 + 1.0 = 3.1`；惩罚量级前三是 `base_height_l2`（−10）、**`undesired_contacts`（−5，新）**、`flat_orientation_l2`（−5）。原先量级最大的 `feet_air_time_variance −8` 已清零 ⇒ 配方从“**压步态均匀 + 站稳**”转为“**避免碰撞／避免踩边缘 + 站稳**”，步态自由度交给策略。
- **直立门控**：除下面点名的项外，本仓库自定义、被本配方用到的奖励项都乘 `clamp(-g_z, 0, 0.7)/0.7`（约 45° 后趋近 0）。
  **无门控**的是上游 Isaac Lab 核心的 `joint_torques_l2`、`joint_acc_l2`、`action_rate_l2`、`contact_forces`，本仓库自定义的 `joint_power`，以及**本提交新增的 `feet_edge`**（`rewards.py:455-481` 直接返回 `Σ(edge & contact)`，没有门控——与同样新增但**带**门控的 `feet_stumble` 不同）。
- `contact_forces` 取自上游核心：`Σ max(0, |F|_max − 100 N)`。
- `feet_edge` 的射线网格：`GridPatternCfg(resolution=0.04, size=(0.12, 0.12))` ⇒ 4×4=16 条/足，挂 `{ENV}/Robot/{FL,FR,RL,RR}_FOOT`，`offset z=0.5`、`max_distance=2.0`、`mesh_prim_paths=["/World/ground"]`、`update_period=0.02 s`。**全空洞（16 条全不命中）不罚**——那时该足通常也不接触。台阶等高度突变处所有射线都有命中，不受罚（这正是它区别于 base 高度扫描的用意）。
- `feet_edge` 把足端接触传感器 `body_ids` 与四个 sensor 名字**按索引配对**，但因为最终是 `edge & contact` 逐元素后求和，配对顺序不影响结果。

## 3. 被裁掉的项（权重 0 → 置 None）

- 基础无权重项：`is_terminated`、`body_lin_acc_l2`、`joint_vel_l2`、`joint_deviation_l1`、`joint_pos_limits`、`joint_vel_limits`、`wheel_vel_penalty`、`action_mirror`、`action_sync`、`applied_torque_limits`、`feet_gait`、`feet_contact`、`feet_contact_without_cmd`、`feet_distance_y_exp`、`upward`。
- **本提交新清零的五项固定步态 shaping**：`feet_height`、`feet_height_body`（原 −5.0）、`feet_slide`（原 −0.05）、`feet_air_time_variance`（原 −8.0）、`joint_mirror`（原 −1.0）。

`is_terminated` 仍为 0 ⇒ **没有终止惩罚**；但基座触地现在会**结束 episode**（§4），所以“翻倒”的代价变成丢掉该回合剩余的正奖励，而不是一笔固定罚分。

## 4. 奖励之外、同样决定训练行为的配置

- **终止**（`6220e43` 起 CMoE 覆盖）：`time_out`、`terrain_out_of_bounds`（二者 `time_out=True`）；**`illegal_contact` 被重新启用为基座触地终止**——`body_names=["base"]`、阈值 1.0 N（`CMoE_env_cfg.py:139-145`），即**摔倒压到基座就结束回合**，足端接触仍合法。`is_terminated` 权重仍是 0（无终止罚分）。训练中要同时看 `Train/mean_episode_length`（新增的终止会让它下降）与 `Episode_Reward/undesired_contacts`。
- **地形**（`6220e43` 起 CMoE 专用，`CMoE_env_cfg.py:101-116`）：在 PPO rough 的地形生成器上加入 **`gap` = `MeshGapTerrainCfg(proportion=0.20, gap_width_range=(0.10,0.16), platform_width=4.0)`**，并把原六类重配为 `pyramid_stairs 0.15 / pyramid_stairs_inv 0.10 / boxes 0.15 / random_rough 0.20 / hf_pyramid_slope 0.10 / hf_pyramid_slope_inv 0.10`（合计 1.00，Isaac Lab 另有按总和归一化）。4 m 平台使 `x,y ±1 m` 的 reset 始终离沟壑 ≥1 m。**地形比例与 PPO rough 任务不再相同**，两边的回报不可直接比。
- **4 个足端射线传感器**（`CMoESceneCfg`，`6220e43` 起）：`foot_edge_scanner_{fl,fr,rl,rr}`，只服务 `feet_edge` 奖励，**不进 observation**（观测维度不变，见 §5）。
- **课程**：只保留 `terrain_levels`（地形难度随表现升级，terrain generator `curriculum=True`）；`command_levels_lin_vel`／`command_levels_ang_vel` 置 None，即命令范围固定不爬坡。
- **命令**：`resampling_time_range=(10,10) s`、`rel_standing_envs=0.02`、`heading_command=True`（`heading_control_stiffness=0.5`）；范围 `vx ∈ [−1,1] m/s`、`vy ∈ [−0.8,0.8] m/s`、`yaw ∈ [−1.5,1.5] rad/s`。
- **时长／频率**：`dt=0.005 s`、`decimation=4` ⇒ 策略 50 Hz（`step_dt=0.02 s`）；`episode_length_s=20` ⇒ **单回合 1000 步**。
- **域随机化（训练）**：摩擦 `static 0.3–1.0 / dynamic 0.3–0.8`、restitution `0–0.5`；base 质量 `+(-1,3) kg`、其余 link `×0.7–1.3`；CoM `±0.05 m`；reset 位姿 `x,y ±1 m`、`yaw ±π`、线／角速度小幅，**`roll/pitch` 自 `6220e43` 起固定为 0**（原 ±0.3 rad，`CMoE_env_cfg.py:119-126`）；执行器增益 `×0.5–2.0`（`mode="reset"`，每次 reset 重采样）。外力／推动已显式关闭，`randomize_reset_joints` 的 scale 是 `(1.0,1.0)`，等于不随机。

## 5. 观测契约与训练超参

| 输入 | 维度 | 内容 |
|---|---:|---|
| `policy` 单帧 | 45 | 非特权本体感知（无 base 线速度、无高度扫描） |
| `policy` 历史 | 450 | 10 × 45 |
| `terrain` | 187 | 当前 height scanner（`1.6×1.0 m`、`0.1 m` ⇒ 17×11） |
| actor 总输入 | **637** | 450 + 187 |
| `critic` | **235** | `[base_lin_vel(3), policy(45), height_scan(187)]` |
| expert／gate | 267 | 当前 45 + 显式速度 3 + 状态 latent 16 + 地形 187 + 地形 latent 16 |

Runner（`CMoE_rsl_rl_cfg.py`）：`num_steps_per_env=24`、`max_iterations=2000`、`save_interval=100`、`seed=1`、`device=cuda:0`、`experiment_name=base_move_cmoe_rough`、`history_steps=10`、`logger=tensorboard`、`empirical_normalization=False`、`clamp_noise_std=True`（继承基类）。

算法：5 专家，`actor/critic_hidden_dims=[512,256,128]`、`gate_hidden_dim=128`、`num_prototypes=32`、`projection_dim=16`、`temperature=0.2`、`state/terrain_latent_dim=16`、`init_noise_std=1.0`；PPO `lr=1e-3`（adaptive，`desired_kl=0.01`）、5 epochs × 4 minibatches、`gamma=0.99`、`lam=0.95`、`entropy_coef=0.01`、`clip_param=0.2`、`max_grad_norm=1.0`、`value_loss_coef=1.0`（clipped）、`contrastive_loss_coef=1.0`。

每次迭代采集 `24 × num_envs` 步；`num_envs=4096` 时 rollout 为 98304 步，2000 轮共约 1.97e8 步。

日志目录为 `imgo2_rl/logs/cmoe/base_move_cmoe_rough/<YYYY-MM-DD_HH-MM-SS>[_run_name]/`（`train.py:99` 用相对 cwd 的 `logs/...` 求 abspath，**所以必须在 `imgo2_rl/` 下启动**）。

## 6. 训练指令（训练机执行）

所有 Isaac Lab 命令都走 `run_isaaclab.sh`（原因见根 README §4.2 的提示与启动器记录）。

### 步骤 0：同步代码 + 启动器自检

```bash
cd /path/to/Imgo2
git fetch origin
git checkout imgo2_CMoE
git pull --ff-only                     # 期望 37b9206
bash imgo2_rl/scripts/run_isaaclab.sh --check
```

自检期望输出：`torch <版本> (cuda <版本>)`、`linalg .so 存在`、`dlopen(linalg): OK`、`CPU solve: OK`、`cuda available: True`、`CUDA solve: OK`。任何一项失败先别往下走（解释器可用 `IMGO2_ISAACLAB_PYTHON` 覆盖，默认 `/opt/conda/envs/isaaclab/bin/python`）。

### 步骤 1：确认两个本地包装在同一个解释器里

```bash
PY="${IMGO2_ISAACLAB_PYTHON:-/opt/conda/envs/isaaclab/bin/python}"
$PY -m pip install -e imgo2_rl/source/imgo2_rl
$PY -m pip install -e imgo2_rl/scripts/rl_lab
bash imgo2_rl/scripts/run_isaaclab.sh -c "import imgo2_rl, rl_lab, pybullet_utils, torch; print('ok', torch.__version__)"
```

### 步骤 2：张量／TorchScript 测试（本机因无 torch 一直被跳过）

```bash
cd imgo2_rl
bash scripts/run_isaaclab.sh -m unittest discover -s tests -p 'test_cmoe_stack.py' -v
```

期望 3 项通过（前向 + gate 归一化 + 辅助损失、Storage 维度、导出器与 `act_inference` 的 TorchScript 一致性），且不再是 `skipped`。

> **注意（2026-09-23 本机实测新增）**：`test_cmoe_stack.py` 虽然只测张量，但它的 import 链会拉进 Isaac Sim：
> `rl_lab.modules.__init__` → `rl_lab.utils.__init__`（`export_deploy_cfg.py:5`）→ `isaaclab.assets` →
> `omni.log`。因此**必须用 Isaac Sim 所在的那个解释器**跑；用一个只装了 torch 的裸解释器会得到
> `ModuleNotFoundError: No module named 'omni.log'`，而测试文件会把这个导入错误伪装成
> `skipped "PyTorch unavailable: ..."`——看到 `skipped` 时先看跳过原因，**不要当成“环境缺 torch”**。
> 本沙箱就是这种情况（conda env `isaaclab` 已装 torch 2.7.0+cu128 与 pip 版 `isaacsim` 5.0，
> 但 `omni.*`／`isaacsim.core` 只有 app bootstrap 后才在 `sys.path` 上），故 3 项测试仍未执行，见 §9。

### 步骤 3：先跑冒烟（4 环境 × 2 轮），核对维度

```bash
cd imgo2_rl
bash scripts/run_isaaclab.sh scripts/rl_lab/cmoe/train.py \
  --task=Imgo2-basemove-rough-cmoe \
  --num_envs=4 --max_iterations=2 --headless 2>&1 | tee /tmp/cmoe_smoke.log
```

必须逐行核对打印的契约：`num_one_step_obs: 45`、`history_steps: 10`、`num_terrain_obs: 187`、`num_obs (total): 637`、`num_privileged_obs: 235`、`num_actions: 12`；并确认 2 轮内有 loss 输出、没有 NaN/Inf、`model_0.pt`／`model_2.pt` 落盘。**维度对不上或报错就先停，不要直接开正式训练。**

### 步骤 4：正式训练

```bash
cd imgo2_rl
bash scripts/run_isaaclab.sh scripts/rl_lab/cmoe/train.py \
  --task=Imgo2-basemove-rough-cmoe \
  --num_envs=4096 \
  --max_iterations=2000 \
  --seed=1 \
  --run_name=cmoe_rough_4096_seed1 \
  --headless 2>&1 | tee logs/cmoe_train_$(date +%F_%H%M).log
```

显存不够时只降 `--num_envs`（2048 → 1024），**不要改观测维度相关配置**；`num_envs` 只影响 rollout 规模，不改变上表维度。请在记录里写明实际卡型／显存与最终采用的环境数。

### 步骤 5：续训（可选）

```bash
cd imgo2_rl
bash scripts/run_isaaclab.sh scripts/rl_lab/cmoe/train.py \
  --task=Imgo2-basemove-rough-cmoe \
  --resume --load_run=2026-09-23_XX-XX-XX_cmoe_rough_4096_seed1 \
  --checkpoint=model_1000.pt \
  --num_envs=4096 --headless
```

`CMoEOnPolicyRunner` 的循环内保存显式传了 `iteration=it`（`cmoe_on_policy_runner.py:119`），`save()` 写 `'iter'`，`load()` 用该值续训——**不含 AMP-08 那个「循环内 `current_learning_iteration` 不更新导致 checkpoint `iter` 恒为 0」的缺陷**；续训前仍建议抽查一个中间 checkpoint 的 `iter` 字段。

### 步骤 6：回放 + 导出 `policy.pt` / `policy.onnx`

```bash
cd imgo2_rl
bash scripts/run_isaaclab.sh scripts/rl_lab/cmoe/play.py \
  --task=Imgo2-basemove-rough-cmoe-play \
  --num_envs=1 --headless \
  --checkpoint="$(pwd)/logs/cmoe/base_move_cmoe_rough/<run>/model_2000.pt"
```

导出落在 `<run>/exported/`。play 配置已关掉 policy／terrain 噪声与域随机化、固定 `vx=1.0 m/s`；导出图含双 estimator + gate + 5 个 expert actor、不含 critic，输入是 637 维 `[10 帧 policy history, 当前 terrain]`。

## 7. 训练中要看的量

- `Train/mean_reward`、`Train/mean_episode_length`：长度应逐步顶到 1000（=20 s）。`6220e43` 起**基座触地会终止**，所以长度下降不一定是坏事（可能只是摔了），要结合 `Episode_Reward/undesired_contacts`／`flat_orientation_l2` 判断是「学会了不摔」还是「频繁早退」。
- `Episode_Reward/<term>` 分项：重点 `track_lin_vel_xy_exp`、`base_height_l2`、`undesired_contacts`（`6220e43` 起加权到 −5，最可能主导惩罚）、**`feet_edge`**、**`feet_stumble`**、`feet_air_time`。
- `Policy/gate_entropy` 与 `Policy/expert_{0..4}_mean_weight`：熵应低于 `ln 5 = 1.609`（专家分化），若长期贴 1.609 = gate 均匀、若迅速趋 0 = 早期塌缩到单专家，两种情况都要记录。
- `Loss/Estimation Loss`、`Latent`、`Recons`、`Kld`（两个 estimator 各一套）与 `Loss/contrastive`。
- `Loss/value_function`、`Loss/surrogate`、`Policy/mean_noise_std`、`Perf/total_fps`。
- 数值健康：任何 NaN/Inf；最后一帧 gate 的 bootstrap（移植时已修，见移植记录）；wrapper 自带 `NaN/Inf in rewards/observations` 断言。

## 8. 待确认与限制

- 本文的奖励与超参结论来自**源码阅读**，未运行；权重来源是 PPO rough 配方 ＋ `6220e43` 的 CMoE 覆盖，**没有针对 5 专家 CMoE 调过参**（`feet_edge −1`、`undesired_contacts −5`、`feet_stumble −1` 都是首版值）。
- 未知：实际显存占用与可用 `num_envs`、5 专家＋双 estimator 相对 PPO 的每轮耗时、gate 是否真的按地形分工（5 专家未与 5 类地形人工绑定）、20% 沟壑比例是否合适。
- `is_terminated=0` 但基座触地会终止（`6220e43` 起）⇒ **摔倒的代价是丢掉该回合剩余正奖励，而不是固定罚分**；若观测到「早期频繁摔倒但回报不降」，需要考虑加终止罚分（属配方变更，需单独记录）。
- play／导出链路尚未在训练机跑过；导出契约（与 actor max diff、部署 libtorch 加载、观测顺序）三条复核未做。

## 9. 本轮实测结果（本机沙箱，无 GPU）

| 检查 | 结果 | 说明 |
|---|---|---|
| `bash imgo2_rl/scripts/run_isaaclab.sh --check` | **通过（GPU 除外）** | `python /opt/conda/envs/isaaclab/bin/python`、`torch 2.7.0+cu128 (cuda 12.8)`、`libtorch_cuda_linalg.so 存在`、`dlopen OK`、`CPU solve OK`；但 `cuda available: False` |
| `nvidia-smi -L` | 不可用 | `Failed to initialize NVML: Insufficient Permissions`——本会话看不到 GPU，故不能跑环境构造／训练／play |
| `import imgo2_rl, rl_lab` | 失败 | `ModuleNotFoundError: No module named 'omni.log'`（`isaaclab/assets/articulation/articulation.py:16`）；该 conda 环境的 `omni.*` 与 `isaacsim.core` 要等 Isaac Sim app bootstrap 才进 `sys.path` |
| `test_cmoe_stack.py` 3 项 | **仍未执行（skip）** | 同上原因；测试把它记为 `skipped "PyTorch unavailable: No module named 'omni.log'"`。手工把 `omni/kernel/py`、`isaacsim/extscache/*` 塞进 `PYTHONPATH` 可依次越过 `omni.log`、`carb`、`libcarb.so`、`omni.physics`，随后卡在 `isaacsim.core`（pip 版 `isaacsim` 是常规包，会遮蔽 exts 里的命名空间包），**不值得继续 hack**：这是机器环境差异，不是仓库缺陷 |

因此本文的**奖励与超参复核已核对到源码级**，但“测试通过／维度实测／短训练”三项仍属未验证，
必须在真正的 Isaac Lab 训练机（`run_isaaclab.sh` 能 import `imgo2_rl` 的那台）执行 §6。

训练机下一步至少执行 §6 的步骤 2–4，并把步骤 0、3 的输出回传后再决定是否开 2000 轮。

## 10. 首轮冒烟回执（2026-09-23 20:27，用户终端）

> **本节结果跑在 `37b9206` 上，早于 `6220e43`。** 那次运行不包含沟壑地形、四个足端射线传感器、`feet_edge`／`feet_stumble`／加重后的 `undesired_contacts`，也不含基座触地终止 ⇒ 这四类新东西**至今没有构造过或触发过**；观测维度未变，所以 §10.1 的契约结论仍成立，但环境构造与奖励触发必须在 `6220e43` 上重跑（§6 步骤 3）。

用户在同一容器（tfevents 里的主机名 `gpufree-container`，GPU 对用户可见、对 harness 会话不可见）执行 §6 步骤 3，**冒烟通过**；用户反馈“没有打印 log”，据产物判定为**控制台输出丢失，不是运行失败**。

产物目录 `imgo2_rl/logs/cmoe/base_move_cmoe_rough/2026-09-23_20-27-30/`：`model_0.pt`、`model_2.pt`（各 38 MB）、`params/{env,agent,deploy}.yaml`、`params/CMoE_env_cfg.py`、`events.out.tfevents.*`（2572 B）。

### 10.1 观测契约：由 checkpoint 权重直接验证（不依赖控制台）

| 权重 | 形状 | 契约 |
|---|---|---|
| `state_estimator.encoder.0.weight` | (128, 450) | 10 × 45 历史 ✓ |
| `terrain_estimator.encoder.0.weight` | (128, 187) | height scanner 187 ✓ |
| `experts.{0..4}.actor.0.weight` | (512, 267) | 专家输入 45+3+16+187+16 ✓ |
| `experts.{0..4}.critic.0.weight` | (512, 235) | critic 235 ✓ |
| `gate_projector.0.weight` | (128, 5) | 5 专家 ✓ |

`model_0.pt` 的 `iter=0`、`model_2.pt` 的 `iter=2`，且两者都带 `optimizer_state_dict`、`state_estimator_optimizer_state_dict`、`terrain_estimator_optimizer_state_dict` ⇒ **`iter` 写入正确（无 AMP-08 缺陷）**，两个 estimator 优化器状态都落盘。

### 10.2 TensorBoard 标量（22 个 tag × 2 步，全部有限、无 NaN/Inf）

- `Policy/gate_entropy` 1.574 → 1.560（`ln 5 = 1.609`，初始接近均匀，正常）；
- `Policy/mean_noise_std` ≈ 0.999（未偏离 `init_noise_std=1.0`）；
- `Loss/{value_function,surrogate,Estimation,Latent,Recons,Kld,contrastive}` 均有限；
- `Loss/Estimation Loss2 = 0.0`、`Kld Loss2 = 0.0` 而 `Latent/Recons Loss2 ≈ 0.03` —— **设计如此**：`cmoe_terrain_estimator.py:90` 硬返回 `0.0, value, value, 0.0`（地形估计器只做重建，没有速度/状态估计目标），不是丢失的 loss；
- `Perf/total_fps` 37 → 83（4 环境，含首轮预热）。

### 10.3 待观察：自适应学习率在 2 轮里都贴 1e-5 地板

两次记录 `Loss/learning_rate` 都是 **1e-5**（`adaptive` 下限），而本仓库**已知能训好的** AMP 跑里该值在 1e-5 与 1e-2 之间来回（例：`2026-09-17_17-19-04` 10000 轮 last = 1.71e-4，仅在 1622/10000 轮贴地板）。

**已定性（2026-09-23 晚，用这台机器上的 4 次运行对照）——这是小 `num_envs` 的批统计假象，不是配置缺陷。**

对照数据（`logs/cmoe/base_move_cmoe_rough/`）：

| 运行 | 环境数 | 迭代 | `Loss/learning_rate` | 结论 |
|---|---:|---:|---|---|
| `20-27-30` | 4 | 2 | 恒 1e-5 | 冒烟，minibatch 仅 24 样本 |
| `21-28-57` | 4 | 41 | 恒 1e-5 | 同上 |
| `21-48-34` | 4 | 74 | 1e-5～2.3e-5 | 同上 |
| **`20-38-55_cmoe_rough_4096_seed1`** | **4096** | **818** | **1.32e-4 ～ 5.06e-3（正常来回）** | 结论：4096 环境下调度工作正常 |

机理：`num_mini_batches=4` ⇒ minibatch 大小 = `24 × num_envs / 4` = 4 环境时仅 **24 个样本**，`4 环境 × 2 轮` 的 KL 估计噪声极大，`kl_mean > desired_kl*2` 几乎必然成立，于是每个 minibatch 都 `/1.5` 并迅速触底；4096 环境时 minibatch 有 24576 个样本，KL 估计稳定，lr 正常地在 1e-4～5e-3 之间调节。因此**判断调度是否正常必须看 4096 环境（或至少数百环境）的运行，不能看 4 环境冒烟**。

**顺带得到首个“训练确实work”的证据**（同在 `20-38-55` 运行，代码为 `37b9206`、旧地形）：`Train/mean_reward` **−0.43 → +9.54**（最高 11.18）、`Train/mean_episode_length` **12 → 1000**（顶满 20 s 回合）、`Policy/mean_noise_std` 0.99 → 0.50、`Policy/gate_entropy` 1.57 → 1.56（最低 1.18，说明专家有分化）、`Perf/total_fps` ≈ 2.6e4。该运行停在 `model_800.pt`（818 轮，未跑满 2000）。**注意这是沟壑/安全奖励之前的配方与地形**，新提交上必须重跑。

### 10.4 控制台日志丢失

`log()` 在写 TensorBoard 之后才 `print`（`cmoe_on_policy_runner.py:203`），且 22 个 tag 全部写入 ⇒ `print` 确实执行过；控制台看不到属于输出被丢弃。最可能是 **stdout 走管道（`tee`）时 Python 块缓冲，而 Isaac Sim 的收尾路径不保证 flush**（未在本机验证）。规避方式：训练命令加 `PYTHONUNBUFFERED=1`；日志不依赖控制台时可直接读 tfevents（见 §10.5）。

### 10.5 离线读 tfevents（不需要 GPU）

```bash
cd /root/Desktop/Imgo2/imgo2_rl
/opt/conda/envs/isaaclab/bin/python - <<'PY'
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
run = sorted(glob.glob("logs/cmoe/base_move_cmoe_rough/*/"))[-1]
ea = EventAccumulator(run); ea.Reload()
for t in ("Train/mean_reward", "Train/mean_episode_length", "Loss/learning_rate", "Policy/gate_entropy"):
    if t in ea.Tags()["scalars"]:
        v = [(s.step, round(s.value, 4)) for s in ea.Scalars(t)]
        print(f"{t:32s} {v[:5]} ... {v[-3:]}")
PY
```

## 11. `6220e43` 拉取与配方复核（2026-09-23 21:15）

**拉取过程**：`git fetch` 得到 `37b9206..6220e43 add CMoE gap terrain and safety rewards`。本地 README 有本会话的未提交改动、与远端改到同一批行 ⇒ `stash -u` → `merge --ff-only` → `stash pop` → 手工合并两处冲突（CMOE-01 行、维护记录表），远端的地形／安全奖励描述与本记录的冒烟回执都保留。改动文件 4 个：`README.md`、`docs/cmoe_port_2026-09-23.md`、`CMoE_env_cfg.py`(+104)、`mdp/rewards.py`(+30)。

**离线已核对（不需要 GPU，本轮新增）**：

1. `terrain_gen.MeshGapTerrainCfg` 在本机 Isaac Lab 中**存在**（`isaaclab/terrains/trimesh/mesh_terrains_cfg.py:132`），参数名 `gap_width_range`（MISSING，必填）与 `platform_width`（默认 1.0）与配置用法一致；`isaaclab.terrains.__init__` 有 `from .trimesh import *` ⇒ `terrain_gen.MeshGapTerrainCfg` 可解析，不会在 `__post_init__` 抛 `AttributeError`。
2. 地形比例 `0.15+0.10+0.15+0.20+0.10+0.10+0.20 = 1.00`；且 `terrain_generator.py:211-213 / 232-234` 会按总和归一化（`proportions /= np.sum(proportions)`）⇒ 即使不等 1 也不会错。
3. `feet_edge` 语义：`valid = isfinite(ray_hits_w[...,2])`（未命中为 inf），`valid.any(dim=1) & ~valid.all(dim=1)` = 部分命中 = 踩在“实心／空洞”交界，与接触力 >1 N 求交后按足求和 ⇒ 台阶（每束射线都有命中）与全空洞（全不命中）都不罚。
4. `illegal_contact` 取上游 `isaaclab/envs/mdp/terminations.py:153`：基座接触力 >1 N 即终止（非超时）。wrapper 把 `terminated | truncated` 的 env 记入 `_termination_ids`，并用 `obs_before_reset["critic"]` 做 bootstrap ⇒ 新终止复用已有的 terminal-observation 路径。
5. 观测维度未变（45／450／187／637／235）：4 个足端 ray caster 只服务奖励，不进 observation ⇒ §10.1 由 checkpoint 权重验证的契约对 `6220e43` 仍然适用。
6. `feet_edge` 要求 ray caster 数量与接触 body 数量一致（`contacts.shape[1] != len(edge_sensor_names)` 会显式 `ValueError`）；4 个 sensor 名字与 `.*_FOOT` 的 body 按索引配对，但因为最终是逐元素求交后求和，配对顺序不影响结果。

**必须在训练机验证（本机看不到 GPU、无法 bootstrap Isaac Sim）**：① 4 个足端 RayCaster 的 prim 绑定（`{ENV}/Robot/{FL,FR,RL,RR}_FOOT`）能否解析；② `gap` 子地形能否生成、`platform_width=4.0` 下 reset 是否确实离沟壑 ≥1 m；③ `feet_edge` 是否只在“接触 **且** 部分射线落空”时触发（在台阶与平地上应为 0）；④ 基座触地终止是否生效、且没有误杀正常步态（看 `Train/mean_episode_length` 是否异常短）；⑤ `undesired_contacts −5` 的分项量级是否会压过任务奖励；⑥ 训练命令加 `PYTHONUNBUFFERED=1` 后控制台日志是否正常出现。


## 12. 小训练 + 可视化指令（2026-09-23 机器实测环境）

**本机显示条件**：`DISPLAY=:20.0` 已设但 `/tmp/.X11-unix` 不存在，且项目记录过本机 GLX 建上下文失败（`X_GLXCreateContext ... BadValue`，见 [Gazebo 记录](gazebo_ros2_bringup_2026-09-17.md)）⇒ **不要去掉 `--headless` 直接开 GUI**，用下面的 **WebRTC 直播**（`--livestream`）或**录像**。已确认该安装带 `omni.kit.livestream.webrtc-7.0.0`／`omni.services.livestream.nvcf-7.2.0`，且 Isaac Lab 在 `livestream>=1` 时会强制 `headless` 并改用完整 GUI 体验文件 `isaaclab.python.kit`（否则会走 `isaaclab.python.headless.kit`、没有画面）。

### 12.1 只看新地形（最快，不需要策略）

```bash
cd /root/Desktop/Imgo2/imgo2_rl
PYTHONUNBUFFERED=1 bash scripts/run_isaaclab.sh scripts/tools/zero_agent.py \
  --task=Imgo2-basemove-rough-cmoe --num_envs=4 --livestream 2
```
`zero_agent.py` 只读 gym 注册里的 `env_cfg_entry_point`，所以能直接用 CMoE 的 `Imgo2CMoERoughEnvCfg`；机器人保持默认姿态，用来核对沟壑／台阶／独立台阶的布局与 `feet_edge` 射线是否合理。

### 12.2 小训练 + 3D 直播（推荐）

```bash
cd /root/Desktop/Imgo2/imgo2_rl
PYTHONUNBUFFERED=1 bash scripts/run_isaaclab.sh scripts/rl_lab/cmoe/train.py \
  --task=Imgo2-basemove-rough-cmoe \
  --num_envs=16 --max_iterations=50 --seed=1 \
  --run_name=cmoe_vis --livestream 2 2>&1 | tee logs/cmoe_vis.log
```
- `--livestream 2` = WebRTC **私有网络**（`1` 是公网，会额外用 `PUBLIC_IP` 环境变量指定对外地址）；**不要同时加 `--headless`**（直播本身已强制 headless，加了会让画面走 headless 体验文件）。
- 客户端用 NVIDIA 的 **Isaac Sim WebRTC Streaming Client** 连这台机器；信令端口默认见启动日志（`omni.services.livestream.nvcf` 启动时会打印可连地址），若容器有防火墙／NAT 需要放行对应端口。
- 环境数别开大：GUI 每 `decimation=4` 个物理步渲染一次，`--num_envs=16` 已经会明显慢于 headless（参考：4096 环境 headless 约 2.6e4 fps，4 环境 GUI/直播约 30～60 fps）。
- 50 轮 × 24 步 = 每环境 1200 个策略步（24 s 仿真），足以看到起步、过沟壑／台阶；`save_interval=100` ⇒ 只会落 `model_0.pt` 与 `model_50.pt`。

### 12.3 曲线可视化（TensorBoard，和 GPU 无关，最可靠）

```bash
cd /root/Desktop/Imgo2/imgo2_rl
PYTHONUNBUFFERED=1 /opt/conda/envs/isaaclab/bin/python -m tensorboard.main \
  --logdir logs/cmoe --port 6006 --host 127.0.0.1
# 在你自己电脑上做回环转发（不要把 6006 暴露到网络）：
#   ssh -N -L 6006:127.0.0.1:6006 <user>@<host>
# 然后浏览器打开 http://127.0.0.1:6006
```
（该环境已装 `tensorboard 2.20.0`，与 `SummaryWriter` 同源，必可用。）训练中直接看 `Train/mean_reward`、`Train/mean_episode_length`、`Episode_Reward/*`、`Loss/learning_rate`。

### 12.4 录像（直播连不上时的兜底）

```bash
cd /root/Desktop/Imgo2/imgo2_rl
PYTHONUNBUFFERED=1 bash scripts/run_isaaclab.sh scripts/rl_lab/cmoe/play.py \
  --task=Imgo2-basemove-rough-cmoe-play --num_envs=1 --headless \
  --video --video_length 400 \
  --checkpoint="$(pwd)/logs/cmoe/base_move_cmoe_rough/<run>/model_XXXX.pt"
# 产物：<run>/videos/play/*.mp4，同时导出 <run>/exported/policy.pt|onnx
```
`play.py` 的 `--video` 会自动 `enable_cameras`，走 `isaaclab.python.headless.rendering.kit`，适合本机这种没有可用显示的容器。**可直接用现成 checkpoint**：`2026-09-23_20-38-55_cmoe_rough_4096_seed1/model_800.pt`（旧地形、旧配方，818 轮、mean_reward +9.5，能走完整 20 s 回合），用来看 CMoE 的步态是否像样；要验证**新**地形上的表现仍需在新提交上重训后录。

### 12.5 直播排障（2026-09-23 实测，来自 `logs/cmoe_vis.log`）

- **`--livestream 2` 不会在主机上开窗口**：AppLauncher 自己打印 `livestream=2 has implicitly overridden the environment variable HEADLESS=0 to True`（log 第 2 行）⇒ 主机强制 headless、画面只走 WebRTC。「界面没打开」是预期行为，不是失败。
- `Failed to open [/var/run/utmp]` + `Active user not found. Using default user [kiosk]` 是**流媒体 SDK 的降级提示**：该字符串在二进制 `libNvStreamBase.so` 里（`isaacsim/extscache/omni.kit.streamsdk.plugins*/bin/`），容器没有 `utmp` 就会这样；紧接着的 `Streaming server started.`（log 第 277 行）才说明**服务已起**。
- **信令端口 49100**：`isaacsim/extscache/omni.kit.livestream.webrtc-7.0.0/config/extension.toml:29` 的 `app.livestream.port`；`--livestream 1`（公网）时 AppLauncher 还会追加 `--/app/livestream/publicEndpointAddress=$PUBLIC_IP`（同一端口）。
- **客户端**：NVIDIA 的 **Isaac Sim WebRTC Streaming Client**（本机扩展 README 仍指向 Omniverse Launcher 里的 “Kit Remote”），连 `<主机IP>:49100`。**注意 `ssh -L` 只转发 TCP，而 WebRTC 媒体是 UDP(SRTP)** ⇒ 只做端口转发一般「信令连上、画面不来」；需要客户端与主机同网段／VPN 直连并放行 TCP 49100 + UDP 媒体端口。
- 本机 `DISPLAY=:20.0` 但无 X socket，且项目记录过 GLX 建上下文失败 ⇒ **GUI 路线在这台机器上不可行**，要画面就走 §12.4 的录像。
- 附：同一 log 第 241 行有 `Warp CUDA error 36: API call is not supported in the installed CUDA driver (in function cuda_init ...)`——是警告，训练照常跑（216 steps/s）；Isaac Lab 的 RayCaster／PhysX 不依赖 Warp，但若后续用到 Warp 后端的功能要留意。

### 12.6 不依赖控制台的读数

控制台日志可能被缓冲丢掉（§10.4）。离线读 tfevents 的片段见 §10.5，或直接用 §12.3 的 TensorBoard。

## 13. `b87efa3` 拉取与新障碍课程（2026-09-23 21:57）

**拉取**：`git fetch` 得到 `6220e43..b87efa3 replicate CMoE obstacle courses for quadruped`；本轮远端未改 README、本地未提交改动不冲突 ⇒ 直接 `git merge --ff-only`（无需 stash）。顺带用 refspec 把本地 `main` 快进到 `3fbbfa0`（`git fetch origin main:main`）。

**改动**（3 文件：新增 `cmoe_terrains.py` +135、`CMoE_env_cfg.py`、`mdp/commands.py`）：

1. **地形换成纵向 +x 障碍课程**（`terrain_generator.size` 由 `(8,8)` 改 `(8,4)`，比例合计 1.00）：
   - `gap` 0.20 = `CMoETrackGapTerrainCfg(gap_width_range=(0.08,0.16), platform_length_range=(0.65,0.95), first_gap_x=1.8, num_gaps=4, spawn_x=0.75)`——沿 +x 重复 4 条横向沟壑，沟宽随难度 0.08→0.16 m；
   - `pyramid_stairs` 0.15 / `pyramid_stairs_inv` 0.10 = `CMoETrackStairsTerrainCfg`（4 级、`step_depth=0.30`、`step_height_range=(0.025,0.08)`、`stairs_start_x=2.0`，一份 ascending、一份 descending）；
   - `boxes` 0.15 = `CMoETrackStepTerrainCfg`（4 个彼此分离的台阶、`step_height_range=(0.04,0.12)`、`step_length_range=(0.18,0.30)`、`step_spacing=0.85`、`first_step_x=1.8`）；
   - `random_rough` 0.20、`hf_pyramid_slope` 0.10、`hf_pyramid_slope_inv` 0.10 沿用原生成器。
   - 每块地形的 `origin` 放在 `spawn_x=0.75` ⇒ 机器人在第一处障碍（x=1.8）之前出生。
2. **命令**（`CMoE_env_cfg.py:131-146`）：`vx ∈ (−0.3,1.0)`、`vy ∈ (−0.3,0.3)`、`yaw ∈ (−1.0,1.0)`、`heading ∈ (−1.6,1.6)`；`heading_command=True`、`rel_heading_envs=1.0`、`rel_standing_envs=0`；`forward_only_terrain_names = ("pyramid_stairs","pyramid_stairs_inv","boxes","gap")`、`forward_speed_range=(0.3,1.0)`、`forward_heading_target=0.0` ⇒ **占 70% 的障碍地形上强制沿世界 +x 前进（`vy=0`、朝向锁 0、速度 0.3–1.0 m/s）**，只有 `random_rough`／斜坡保有全向命令。
3. **reset**：`x,y` 由 ±1.0 收窄到 **±0.5**、`yaw` 由 ±π 固定为 **0**（配合 +x 课程）；roll/pitch 仍为 0。
4. **奖励与终止与 `6220e43` 完全相同**（17 项、基座触地终止、无终止罚分）。
5. **`mdp/commands.py` 是共享文件**，已核对无回归风险：`UniformThresholdVelocityCommand` 改用 `is_env_assigned_to_terrain`（按**初始分配**而不是实时位置判定），cfg 新增 `forward_only_terrain_names`／`forward_speed_range`／`forward_heading_target`（默认 `("pits",)`）；`_get_terrain_column_range` 对未登记的地形名返回 `None` ⇒ 掩码全 False ⇒ 对地形集里没有 `pits` 的 PPO／HIM 等任务**完全无效**；`is_robot_on_terrain` 仍保留在 `utils.py:72`，没有引用断裂。
6. 离线可核：`CMoETrack*TerrainCfg.function` 的签名与返回符合本版本约定——`SubTerrainBaseCfg.function` 声明为 `Callable[[float, SubTerrainBaseCfg], tuple[list[trimesh.Trimesh], np.ndarray]]`（`sub_terrain_cfg.py:66`），三个新函数正是 `(difficulty, cfg) -> (meshes, origin)`。

**新增待验证**（叠加 §11 的五项）：① 三个课程地形在 `size=(8,4)`、`num_rows/num_cols=10/20`、`border_width=20` 下能否正常生成、难度递进是否如预期；② `first_gap_x/first_step_x/stairs_start_x` 与 reset `x ±0.5`（相对 `spawn_x=0.75`）配合下机器人是否确实从障碍之前起步；③ `forward_only_terrain_names` 的四类地形掩码是否真的生效（`Episode_Reward` 与行为观察）；④ `CMoETrackGapTerrainCfg` 的 `gap_width`／`spacing` 难度插值方向（当前难度越高沟越宽、平台越短）在课程升级时是否产生可走的组合。

## 14. 正式训练启动（2026-09-23，40000 轮）

**配置改动**：`CMoE_rsl_rl_cfg.py` 的 `save_interval` 100 → **500**（`max_iterations` 不需要改文件，CLI 的 `--max_iterations` 会覆盖，见 `cmoe/train.py:78-80`）。

**同时修复了分项奖励日志（本轮新发现）**：本 Isaac Lab 版本在 `ManagerBasedRLEnv._reset_idx` 里把各 manager 的 reset 结果并进 `extras["log"]`（`manager_based_rl_env.py:369-393`，其中 `reward_manager.reset()` 产出 `Episode_Reward/<term>`、termination manager 产出 `Episode_Termination/<term>`，数值已按 `max_episode_length_s` 归一成每秒量）。同仓库的 AMP runner 读的是 `infos['log']`（`amp_on_policy_runner.py:199`），而 `CMoEOnPolicyRunner` 读的是并不存在的 `infos['episode']`（第 96 行）⇒ **CMoE 的 TensorBoard 里从来没有分项曲线**（实证：`2026-09-23_20-38-55…` 的 818 轮运行 Episode 类 tag 数 = 0，而 AMP 跑有 8 个）。已改为 `if 'log' in infos and infos['log']: ep_infos.append(infos['log'])`。**只影响日志**；`py_compile` 通过，运行中未验证。

**启动命令**（40000 轮在 4096 环境上约 42 h）：

```bash
cd /root/Desktop/Imgo2/imgo2_rl
mkdir -p logs
PYTHONUNBUFFERED=1 bash scripts/run_isaaclab.sh scripts/rl_lab/cmoe/train.py \
  --task=Imgo2-basemove-rough-cmoe \
  --num_envs=4096 --max_iterations=40000 --seed=1 \
  --run_name=cmoe_obstacle_4096_seed1 --headless 2>&1 | tee logs/cmoe_obstacle_40000.log
```

（怕 SSH 断连再套 `nohup env … > logs/cmoe_obstacle_40000.log 2>&1 &`，效果等价。）

**续训**（中断后用同一个 run 目录）：

```bash
cd /root/Desktop/Imgo2/imgo2_rl
nohup env PYTHONUNBUFFERED=1 bash scripts/run_isaaclab.sh scripts/rl_lab/cmoe/train.py \
  --task=Imgo2-basemove-rough-cmoe --num_envs=4096 \
  --resume --load_run=2026-09-23_XX-XX-XX_cmoe_obstacle_4096_seed1 --checkpoint=model_XXXXX.pt \
  --headless > logs/cmoe_resume.log 2>&1 &
```

**预算**：单 checkpoint 36.5 MB ⇒ 81 份 ≈ **2.9 GB**；按 `20-38-55` 那次 4096 环境的 ≈2.6e4 fps，每轮 ≈3.8 s ⇒ **40000 轮 ≈ 42 h**、每 500 轮 ≈ 32 min。

**判读要点（新配方）**：

- `Episode/Episode_Reward/*`：`feet_edge`、`feet_stumble`、`undesired_contacts` 是否只在障碍上出现、量级是否压过 `track_lin_vel_xy_exp`；这些是每秒归一值，可与 `20-38-55` 的分项直接对比。
- `Episode/Episode_Termination/illegal_contact`：**base 触地终止率**（新增终止的第一手证据）；`time_out` 应随训练上升。
- `Train/mean_episode_length`：应从 4 环境冒烟看到的 ~60 步升向 1000；**`Train/mean_reward` 是按回合累计值，会随回合变长而变负，不能直接与早期比较**——用 `mean_reward / mean_episode_length` 或 `Episode_Reward/*`（每秒量）判进步。实测短训练 50 轮：per-step 从 −0.059 改善到 −0.042，同时回合长度 59 → 445，说明是在进步而不是退化。
- `Loss/learning_rate`：4096 环境应落在 `1e-4 ~ 5e-3`；若长期贴 `1e-5` 才需要处理（4 环境贴地板是小批噪声，属正常）。
- `Policy/gate_entropy`（低于 1.609 且不过早趋 0）、`Policy/expert_*_mean_weight`、两个 estimator 的 Loss、`Perf/total_fps`（≈2.6e4）。

## 15. 40000 轮训练中期检查（2026-09-24 14:24，跑在 16095 轮）

运行：`logs/cmoe/base_move_cmoe_rough/2026-09-23_22-18-33_cmoe_obstacle_4096_seed1`；代码 = `b87efa3` + 本轮两处本地改动（`save_interval=500`、Episode 分项日志修复）。

**进度／速度**：已 **16095 / 40000** 轮（checkpoint 到 `model_16000.pt`），实测 **3.60 s/轮**（fps 26–27k）⇒ 20000 轮 ≈ **09-24 18:18**、40000 轮 ≈ **09-25 14:17**。

**代理端注意**：读 tfevents 必须带 `size_guidance={"scalars": 0}`，否则 `EventAccumulator` 默认每个 tag 只留 10000 点、会把进展误读成正好 10000 轮（本次即如此）。

**健康度（最近 300 轮）**：`Train/mean_reward` ≈ 21–22（每回合累计）、`Train/mean_episode_length` ≈ 942–979（**接近满 1000 步 = 20 s**）、`Policy/gate_entropy` 稳定 1.184、`Perf/total_fps` 稳定 2.66–2.71e4。终止分布（每秒归一）：`illegal_contact` **0.0735**、`time_out` **0.9266**、`terrain_out_of_bounds` 0 ⇒ **基座触地终止确实生效，且约 93% 的回合能跑满 20 s**。

**新增奖励项都在触发**（每秒归一，末值／min）：`feet_edge` −2.7e-4 / −5.3e-2、`feet_stumble` −4.3e-3 / −4.3e-2、`undesired_contacts` −1.4e-2 / −5.5e-1、`stand_still` −4.8e-3。Episode 分项日志修复生效：**17 个 `Episode_Reward/*` + 3 个 `Episode_Termination/*`**（此前 CMoE 一个都没有）。

**奖励构成**（每秒量，末值）：正向 `track_lin_vel_xy_exp` 1.111 + `track_ang_vel_z_exp` 0.475 = 1.586；惩罚合计 ≈ −0.52（`joint_pos_penalty` −0.106、`feet_air_time` **−0.098**、`contact_forces` −0.0825、`action_rate_l2` −0.0818、`ang_vel_xy_l2` −0.046、`lin_vel_z_l2` −0.044、`flat_orientation_l2` −0.026、`undesired_contacts` −0.014…）⇒ 净 ≈ +1.07/s，折合每步 0.0214，与 `mean_reward/episode_length = 0.0229` 自洽。**净回报几乎全部来自线速度跟踪**；`feet_air_time` 为负说明实际步长短于 0.5 s 阈值（策略用稳定性换步长）。

**专家分工**：末值权重 `expert_1 0.411 / expert_3 0.299 / expert_2 0.240 / expert_0 0.028 / expert_4 0.021` ⇒ gate 已分化（熵 1.59 → 1.18）但 **5 个专家里有 2 个几乎闲置**，`Loss/contrastive` 0.115 → 0.032。

**⚠️ 数值爆炸（已自愈，未污染 checkpoint）**：第 **13722–13733+** 轮起 `Loss/value_function` 连续 **48 轮 = inf**，`Train/mean_reward` 在 13739 达 **−1.52e23**（13778 −3.3e22、13819 −2.4e12），同期 `Loss/Kld Loss` 峰值 7e23、`Latent/Recons` 1e22、`action_rate_l2` −1e22。约 13833 后恢复，14000 轮起全部指标正常。**离线扫描 `model_13000/13500/14000/14500/16000` 的 `model_state_dict`：0 个非有限张量** ⇒ 没有 checkpoint 被写坏，爆炸是瞬态（原因未定位：可能是障碍地形上的 PhysX 接触尖峰或 estimator 优化器瞬时失稳，需下一轮加 `Episode_Reward` 逐项定位是哪一项爆的）。

**⚠️ 更正 §10.3 的结论**：`Loss/learning_rate` 贴 `adaptive` 下限 **1e-5 在 4096 环境同样发生**，而且持续了前 **10000 轮**（1000–9999 轮区间内恒为 1e-5），之后才缓慢回升（10k→1.5e-5、12k→1.1e-4、现在 3.4e-5，区间上限 1.1e-4）。所以「4 环境小批噪声」不是完整解释；更可能是 CMoE 在 minibatch 内更新 estimator 使 actor 输入持续漂移、KL 长期高于恢复阈值 `desired_kl/2 = 0.005`。**策略在前 10k 轮 lr 贴地板的条件下仍然从 −0.3 学到 +21.9（回合长度 11 → 956）**，所以不是阻断项，但意味着自适应调度在前半程实际失效。下一轮可对比 `schedule="fixed"`（lr 1e-3）或加 warmup——属配方变更，需单独记录。

## 16. 地形等级 4.558 的含义（2026-09-24）

曲线：`Episode/Curriculum/terrain_levels`（16259 个点）从 **3.4985**（第 0 轮）升到 **≈4.73**（第 1354 轮）后基本走平，全程区间 **[1.2387, 4.8356]**，末值 **4.5908**。

**机制（逐条核对 `isaaclab/isaaclab/terrains/terrain_importer.py` 与上游 `isaaclab_tasks/.../velocity/mdp/curriculums.py`）**：

- 这个量是**全体 4096 个环境的地形难度等级（行号）均值**（`terrain_levels_vel` 返回 `torch.mean(terrain.terrain_levels.float())`，而不是只对本次 reset 的环境求均值）。
- 等级范围 **0–9**（`num_rows=10`）；初始等级在 **0–5** 均匀抽样（`max_init_terrain_level=5`，见 `terrain_importer.py:335-341`）。
- **晋级判据**：本回合从 `env_origins` 出发走了 **> `size[0]/2` = 4.0 m**（`size=(8.0,4.0)`）。
  **降级判据**：走得 **< `|cmd_xy| × 20 s × 0.5` = 10 × |cmd_xy| m**（障碍地形上命令 0.3–1.0 m/s ⇒ 3–10 m），且 `move_down *= ~move_up` ⇒ **晋级优先**、命令为 0 的站定环境不会降级。
- **关键**：等级 ≥10（即通关最高档）时不会被封顶，而是**被随机扔回 0–9**（`terrain_importer.py:317-321`：`torch.randint_like(..., self.max_terrain_level)`）。

**结论**：

1. **⚠️ 2026-09-24 更正（本节初版结论有误）**：初版写成「随机重开期望 4.5 ⇒ 均值天然悬在 4.5 ⇒ 种群基本全部通关」，这是**把「重开分布的均值」当成了「占用率的均值」**。正确的稳态计算：
   - 均匀分布 ⇒ 4.5（这一点没错）；
   - 但「每回合都晋级 + 到顶后随机重开」**不是均匀分布**：一个环境从随机初始等级 L0 出发会依次停留 L0, L0+1, …, 9 各一个回合，所以 `P(level=k) ∝ P(L0 ≤ k) = (k+1)/10`，归一化后 `∝ (k+1)/55` ⇒ **稳态均值 = 6.0**（蒙特卡洛验证：`p_up=1, p_down=0` → 6.000）。
   - 因此观测到的 **4.56–4.84 既不是均匀、也达不到 always-promote 的 6.0** ⇒ 说明确实有**相当比例的降级／未晋级**，不能解读为「种群全部通关」。
   - 定量参照（含 recycle 的蒙特卡洛）：`p_up/p_down = 0.4/0.4 → 4.996`、`0.4/0.6 → 3.867`、`0.6/0.4 → 5.700`。要稳态 ≈4.6，需要降级率与晋级率同量级；另一条同样成立的可能是「一部分环境持续晋级（→6.0）+ 一部分被卡在低等级」的混合（如 `0.7×6.0 + 0.3×1.3 = 4.59`）——CMoE 的地形列是**逐环境固定**的，30% 环境落在 `random_rough`／斜坡列（命令为随机全向、可被 0.2 阈值清零）⇒ 混合假设在结构上很合理。**仅凭均值无法区分这两者。**
   - 早期轨迹也支持「自我平衡」而非「一路通关」：均值从 3.50（第 0 轮）**先跌到 1.24（第 241 轮）**，再爬到 4.73（约第 1350 轮）后走平——先冲进应付不了的难度、被打回来，再随策略变强回升。
2. **课程在第 ~1300 轮就已经饱和**（3.50 → 4.73 只用了一千多轮，之后 15000 轮里只在 4.58–4.84 之间抖动）⇒ 剩下的 24000 轮**边际收益来自策略质量**（速度跟踪、步态、过障稳健性），而不是地形变难。
3. 难度换算：`difficulty = (row + U(0,1)) / num_rows`（`terrain_generator.py:256`，`difficulty_range` 默认 (0,1)）⇒ 等级 4.6 对应 **difficulty ≈ 0.51**，即各障碍实际尺寸约为：沟壑宽 **0.121 m**（平台长 0.797 m）、楼梯级高 **0.053 m**、独立台阶 **0.081 m 高 / 0.241 m 长**。最高档（等级 9、difficulty≈0.95）是沟 0.156 m、楼梯 7.7 cm、台阶 11.6 cm ⇒ **最难档还没被跑到**，但 recycle 机制下均值不会再涨。
4. **注意一个继承来的宽松点**：晋级阈值 4.0 m 比降级阈值（最高 10 m）更容易达到，且晋级优先 ⇒ 课程比原意更松。要在新障碍课程上真正加难，得调 `size[0]`、命令速度上限或 `num_rows`（属配方变更，需单独记录）。

## 17. 高度维持现状（2026-09-24 14:35，16289 轮）

**反解方法（本轮已验证）**：`RewardManager.compute()` 把 `term × weight × dt` 累加进 `_episode_sums`，`reset()` 再用 `episodic_sum_avg / max_episode_length_s` 写出 `Episode_Reward/<term>` ⇒ **该值 = `weight × 时间平均(term)`**（`dt` 被约掉）。因此可用 `|Episode_Reward| / |weight|` 反解每个惩罚对应的物理量。用 `track_lin_vel_xy_exp` 交叉验证：`1.0695 / 1.5 = 0.713` = 指数核时间均值，量纲与量级都正确。

| 量 | 反解式 | 当前值 |
|---|---|---|
| base 高度误差（相对 `0.30 m + 局部地面`） | `sqrt(\|base_height_l2\| / 10)` | **RMS ≈ 1.7–2.2 cm**（分桶均值；最新单次读数 0.89 cm） |
| 机身倾角 | `sqrt(\|flat_orientation_l2\| / 5)` | **RMS ≈ 4.1°** |
| base 竖直速度（起伏） | `sqrt(\|lin_vel_z_l2\| / 2)` | **RMS ≈ 0.14 m/s** |
| 线速度跟踪核 | `track_lin_vel_xy_exp / 1.5` | 0.71（满值 1.0） |

**每 2000 轮趋势**：高度 RMS **2.82 → 1.69 → 1.81 → 2.20 → 1.73 cm**（整体改善约 40% 后走平）；倾角 **4.57° → 4.11°**（缓降）；竖直速度 **0.170 → 0.144 m/s**（持续缓降）。配合终止分布（`time_out` 0.93、`illegal_contact` 0.07、回合长度 942–979/1000）⇒ **站起来稳、不趴、不弹跳，且仍在轻微改善**。

**三条读数注意**：

1. 目标高度是**相对局部地面**（`height_scanner_base`，0.1×0.1 m、10 cm 分辨率的 3×3 射线），不是绝对 z；**沟壑上射线可能全部未命中**，此时 `base_height_l2` 走 `adjusted_target = root_z` 的兜底分支、误差记 0 ⇒ **该指标会低估间隙处的真实高度偏差**。
2. 该惩罚带**直立门控**（`clamp(-g_z,0,0.7)/0.7`）：倾角大于约 45° 的时刻贡献 0 ⇒ 它只描述「大致直立时」的高度控制，摔倒瞬间不在这个数字里（摔倒由 7% 的 `illegal_contact` 终止反映）。
3. 记录点是每次 reset 时对**当次重置环境**求均值（通常只有几个环境），单点读数噪声大；判断趋势要用分桶均值。
4. `feet_height_body` 已在 `6220e43` 被清零 ⇒ **现在没有摆动足高度的显式目标**，上面全是 *base* 高度；足端相关信号只剩 `feet_air_time`（当前为负 ⇒ 实际步长短于 0.5 s 阈值）与 `feet_stumble`。

## 18. 姿态／步态的角度化评估：现有工具与缺口（2026-09-24）

**仓库已有成套工具（无需新写算法）**：

- `imgo2_rl/scripts/tools/gait_metrics.py`（纯 numpy、不需 GPU）：`body_attitude_summary(quat_wxyz, lean_threshold_deg, dt)` → **roll/pitch 的 mean/RMS/峰峰（度）＋ 解缠后的 yaw 漂移与漂移率**；`summarize_contact()` → **占空比／滞空／支撑跑程**；`circular_phase()`／`interpolated_phase_angles()`／`circular_mean()` → **腿间相位**；`peak_to_peak_per_env()` → 关节与足端峰峰值；`landing_events()`、`build_report()` 汇总。
- `imgo2_rl/scripts/tools/eval_gait.py`：采样端（rollout → JSON，可 `--dump-timeseries`／`--dump-npz`），内部调用 `build_report()`。
- `imgo2_rl/tests/test_gait_metrics.py`：纯 numpy 回归，锁定聚合顺序、相位估计、游程覆盖、姿态欧拉角与「恒定倾斜」判定（**可在本机直跑**）。
- **参考基线**：`docs/gait_reference_baseline.json`（由 21 份真机录制动作直接算出）：`walking_summary` 含步态周期中位、**步频中位 1.67 Hz**、shank/thigh 峰峰值、足端 z 峰峰值、shank 速度峰值；`by_speed_prefix` 分速度档；`per_leg_stride_x_m` 逐腿步幅；`how_to_use` 明确判读规则——**步频远高于 1.67 Hz 而步幅远小于同速档 ⇒「步幅偏小、靠步频补速度」；duty_factor→1 ⇒ 滑行而非迈步**。`body_attitude_summary` 的 docstring 还给了姿态基线：pitch 均值都在 ±1.6° 内、前进略低头约 0.5°、roll 均值 ≈0 / RMS 0.33°。

**当前 TB 里能直接反解的姿态角度（近 4400 轮均值，16393 轮）**：整体倾斜 RMS **4.12°**、滚转+俯仰角速度 RMS **59 °/s**、base 高度误差 RMS **2.0 cm**、竖直速度 RMS 0.152 m/s、12 关节 Σ|Δq| ≈ **0.98 rad**（折合单关节 ≈ **4.7°**）、线/角速度跟踪核 0.697／0.769。**TB 里没有分轴 roll/pitch，也没有任何步态相位／占空比／步幅／抬脚高度**；且 `feet_air_time_variance`／`joint_mirror`／`feet_height_body` 已在 `6220e43` 清零 ⇒ **步态目前没有奖励或指标约束**。

**缺口**：`eval_gait.py` 目前**硬编码 AMP 栈**——`from rl_lab.wrapper import AmpVecEnvWrapper`、`AmpVecEnvWrapper(env, include_history_steps=...)`、`AMPOnPolicyRunner`（见其 118-150 行），换 CMoE task 会在包装/runner 处失败。要做 CMoE 的角度化姿态+步态报告，需要给它加一个 stack 选择（CMoE 分支：`CMoEVecEnvWrapper(env, history_steps=agent_cfg.history_steps)` + `CMoEOnPolicyRunner`），采样与 `build_report` 部分可完全复用。另外 `imgo2_rl/scripts/tools/read_tfevents.py` 已存在（本轮我手写了临时脚本，下次可直接用它）。

## 19. 「terrain level 还在上升吗」——数据回答与新诊断日志（2026-09-24 14:50，16537 轮）

**结论：是平的，近期略降。** 当前值 **4.7315**，历史最大 **4.8356**（远早于现在），近 2000/5000/8000 轮斜率分别为 **−0.0198 / −0.0141 / −0.0075（每千轮）**。每 2000 轮均值：

| 桶 | 0–2k | 2–4k | 4–6k | 6–8k | 8–10k | 10–12k | 12–14k | 14–16k | 16–18k(部分) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 均值 | 3.671 | 4.750 | 4.727 | 4.721 | 4.730 | 4.730 | 4.729 | 4.700 | 4.664 |

**为什么容易看成「还在上升」**：① 瞬时值在 **4.66–4.84** 之间抖（峰峰约 0.18），TensorBoard 默认平滑 + 短窗口会把一段上行噪声显示成趋势；② 曲线那一段「大幅上涨」发生在**前 ~1350 轮**（1.24 → 4.73），只要视图从起点开始看，整体形状就是上升的。核对方法：关掉 TB 的 smoothing，或框选最近 2000 轮。

**新增诊断日志（本轮代码变更，`py_compile` 通过，运行中未验证）**：新增仓库本地 `terrain_levels_vel_logged`（`velocity/mdp/curriculums.py`，判据与上游 `terrain_levels_vel` **逐字一致**，包括 `distance > size[0]/2` 晋级、`distance < |cmd_xy|·T·0.5` 且未晋级才降级、`update_env_origins` 负责增减与到顶重开），并把 CMoE 的 `curriculum.terrain_levels` 指向它（`CMoE_env_cfg.py`）。新增曲线：

```
Episode/Curriculum/terrain_levels/{level_mean, level_min, level_max,
    move_up_frac, move_down_frac, frozen_frac, distance_mean, command_norm_mean,
    level_pyramid_stairs, level_boxes, level_gap, level_random_rough, ...}
```

**判据**：`move_down_frac ≈ 0` 且 `level_mean` 停滞 ⇒ 属于「一部分环境被冻住」的混合情形；`move_up_frac ≈ move_down_frac` ⇒ 是真实的能力边界（升降级平衡）。`level_<terrain>` 用来对比障碍列（强制前进命令）与粗糙／斜坡列（随机全向命令）。

**生效条件**：只对**下次训练或 `--resume`** 有效；正在跑的进程用的是旧代码，曲线不会出现。风险点已离线核对：`CurriculumManager.reset` 会把 dict 返回值展开为 `Curriculum/<term>/<key>` 并把 tensor 转 `.item()`（`isaaclab/managers/curriculum_manager.py:104-112`），故返回值结构与日志键均被支持。

## 20. 回放发现：策略卡在沟壑（2026-09-24 15:0x，用户 GUI 回放 `model_16500`）

用户在 GUI 回放（`--num_envs=8`、`Imgo2-basemove-rough-cmoe-play`、`model_16500.pt`）时观察到：**机器人卡在沟壑处**（其余地形未报异常）。

**与训练数据的呼应**：这解释了两件事——① 地形等级均值自 ~1350 轮走平在 4.6（`§16`），若 20% 的沟壑列环境走不出 4 m 就会被降级／冻住，正好对应「一部分持续晋级 + 一部分卡住」的混合假设；② 该 run 的 `Episode_Reward/feet_edge` 极小（−2.7e-4）说明它很少踩到边缘——**因为它是停在沟前面，而不是踩上去**。

**首要嫌疑（配置层面，均在 `6220e43` 引入）**：

1. `feet_edge = −1.0`：判据是「足底射线网格**部分命中**（实心／空洞交界）**且该足接触**」——也就是**跨沟时必须摆出的「足踩边缘」姿态恰好被罚**。它是在惩罚过沟所需的动作本身。
2. `undesired_contacts = −5.0`（加重 10 倍）＋ `feet_stumble = −1.0`：跨沟时小腿／大腿蹭到对岸边缘会吃 −5/次，进一步抬高尝试成本。
3. **没有任何跨沟的正向激励**：唯一推力是 `track_lin_vel_xy_exp`，停在沟前只是丢掉跟踪分；奖励里没有「越过障碍／前进距离」项。
4. **沟壑课程起点偏难**：`CMoETrackGapTerrainCfg.gap_width_range=(0.08, 0.16)`，等级 0 就是 0.08 m 的沟；策略没有从「几乎无缝」开始学的阶段。
5. 回放命令固定 `vx=1.0`（训练域上限，动量最大、本应最好过）⇒ 仍卡住，更支持「奖惩结构让它不敢／不会过」而不是「沟太宽」。

**待确认的现象形态（决定修法）**：停在沟前不敢迈（→ 改奖惩）／前腿踩空掉进沟卡住（→ 改沟宽课程＋确认终止是否触发）／跨过去又退回（→ 相位与步幅问题）。

**候选修法（下次训练，属配方变更需单独立项）**：`feet_edge` 归零或改成「仅当整束射线全部落空（足完全悬空）才罚」；`undesired_contacts` 回调到 −0.5~−1；沟宽课程从 `(0.02, 0.10)` 起；加入前进距离／越障的正向项；必要时先用 `gap_width_range=(0.02,0.06)` 做一次「能否学会过沟」的专项验证。

### 20.1 已应用的验证性改动（2026-09-24，`CMoE_env_cfg.py`）

现象确认为「**其他地形跟速正常，只有沟壑完全停住不前进**」⇒ 判为奖惩结构而非能力/步态问题。按**最小改动、可归因、可回退**原则只动三项权重：

| 项 | 改前 | 改后 | 理由 |
|---|---:|---:|---|
| `feet_edge`（`CMoERewardsCfg`） | **−1.0** | **0.0** | 判据＝足底射线**部分命中**且该足接触 ＝ 跨沟必摆的「足踩边缘」姿态，等于罚过沟动作本身 |
| `feet_stumble` | −1.0 | **0.0** | `any(\|F_xy\|>4\|F_z\|)` 在足蹬对岸边沿时必然触发 |
| `undesired_contacts` | −5.0 | **−0.5** | 回到 PPO rough 原值；−5 时小腿/大腿擦对岸一次就吃掉大量回报 |

未改：地形比例、沟宽课程 `(0.08, 0.16)`、`track_*` 权重、终止、reset。`feet_edge` 归零后会被 `disable_zero_weight_rewards()` 置 None（4 个足端 RayCaster 仍在场景里，只是不再被读取）。`py_compile` 通过，**运行中未验证**。

**验证方式（resume 而非新跑，先花 ~2 h 验证假设）**：

```bash
cd /root/Desktop/Imgo2/imgo2_rl
PYTHONUNBUFFERED=1 bash scripts/run_isaaclab.sh scripts/rl_lab/cmoe/train.py \
  --task=Imgo2-basemove-rough-cmoe --num_envs=4096 \
  --resume --load_run=2026-09-23_22-18-33_cmoe_obstacle_4096_seed1 \
  --checkpoint=model_16500.pt --max_iterations=2000 \
  --run_name=cmoe_gapfix --headless 2>&1 | tee logs/cmoe_gapfix.log
```

（`--max_iterations` 在 resume 时是**额外轮数**；新 log 目录以 step 16500 起记，曲线与旧 run 分离。）

**成功判据**：① 新目录里 `Episode/Curriculum/terrain_levels/level_gap` 开始上升（我 §19 新加的逐地形列等级）；② `Episode/Curriculum/terrain_levels/distance_mean` 上升；③ `Episode_Reward/feet_edge` 消失、`undesired_contacts` 量级降约 10×；④ 回放时能迈过沟。**注意**：奖励改变后 value function 需要重新适配，`Train/mean_reward` 会先掉一段再回升，这不是退化。

**记录口径**：本 run 前 16500 轮用旧奖励、之后用新奖励（同一 run 混了两套配方，已在维护记录注明）；若验证成功，正式交付应**新开一个 run** 用最终配方从头训。

## 21. 参考：ZiwenZhuang/parkour 的 leap（跃沟）配方对比（2026-09-24）

用户拉取 `/root/Desktop/parkour`（**ZiwenZhuang/parkour**，Robot Parkour Learning，CoRL 2023；`origin=github.com/ZiwenZhuang/parkour`，HEAD `5e1f413`）。关键文件：`legged_gym/legged_gym/envs/go1/go1_leap_config.py`（leap 技能配置，a1 版同构）、`envs/base/legged_robot_field.py`（奖励实现）、`utils/terrain/barrier_track.py`（BarrierTrack 地形，leap 障碍几何）。

### 21.1 奖励对比（leap 技能 vs 我们的 CMoE）

| 项 | parkour `leap` | CMoE（现在） |
|---|---|---|
| 速度跟踪 | **`tracking_world_vel` +5.0**（σ=0.35，**世界系**） | `track_lin_vel_xy_exp` +1.5（σ=0.5，**体系**） |
| 角速度跟踪 | +0.05 | `track_ang_vel_z_exp` +0.6 |
| 前向命令 | **1.5–1.8 m/s** | 0.3–1.0 m/s（障碍列） |
| 姿态 | `orientation` **−0.1** | `flat_orientation_l2` **−5.0** |
| 碰撞 | `collision` −1.0 | `undesired_contacts` −0.5（改前 −5.0） |
| 保直行 | `yaw_abs` −0.2、`lin_pos_y` −0.4 | 靠 heading 命令 |
| 髋部姿态 | `hip_pos` **−5.0** | `joint_pos_penalty` −0.1（全关节） |
| 关节偏差 | `dof_error` −0.15 | — |
| 关节/力矩越限 | `exceed_dof_pos_limits` −0.8、`exceed_torque_limits_l1norm` −1.0 | 两项权重 0（关闭） |
| **穿透** | **`penetrate_depth` −1e-2、`penetrate_volume` −1e-4（×接触点速度）** | **无（物理洞）** |
| 足端/高度类 | **`feet_air_time`/`feet_edge`/`feet_stumble`/`base_height` 全部没有** | +1.0 / −1(已归零) / −1(已归零) / −10 |
| 平滑 | `action_rate`/`lin_vel_z`/`ang_vel_xy` 全部注释或很低 | `action_rate_l2` −0.01、`lin_vel_z_l2` −2、`ang_vel_xy_l2` −0.05 |
| 终止 | roll/pitch/**z_low −0.1**/z_high 2.0/out_of_track | time_out/越界/base 触地 |
| `only_positive_rewards` | **False**（允许负回报） | 无此机制 |
| 训练方式 | **`resume=True` 从行走策略 fine-tune**；`curriculum=False`、`max_init_terrain_level=2`；20000 轮、save 200 | 从零；20% 沟壑课程；40000 轮 |
| 沟壑尺寸 | **length 0.3–0.8 m**、depth 0.4–0.8 m、期望跃起高度 `height=0.15 m` | gap **0.08–0.16 m**（物理洞） |

### 21.2 三条决定性机制差异

1. **正向压倒一切**：`tracking_world_vel=+5` 配 1.5–1.8 m/s 命令，同时把姿态／碰撞／平滑惩罚相对大幅放松（基类 `orientation=−4`→leap 用 **−0.1**；`collision=−10`→**−1**）。我们恰好相反（跟踪 +1.5、姿态 −5、碰撞原 −5）⇒ **「停在沟前」在我们的配方里是理性的**。
2. **虚拟地形 + 穿透惩罚**（`barrier_track.py`）：`virtual_terrain=True` 时沟不是硬物理洞；`get_leap_penetration_depths()` 只在机体采样点**低于期望跃起高度 0.15 m** 时按 `(0.15 − z) × 速度` 罚。⇒ 给出**稠密、有方向**的"跳得不够高"信号，且**永远不会物理卡住**。我们相反：掉进物理洞 → 卡住 → 只能等 20 s 超时，信号既稀疏又没有方向。
3. **分技能训练**：先练行走，再 `resume` 到「只含 leap」的赛道 fine-tune 出该技能（固定难度、20000 轮）。他们并不指望多地形混合课程顺带学会过沟。

另外他们实现过 **`_reward_leap_bonous_cond`（过沟补偿）**：`(1 − exp(−err²/σ)) × engaging_mask`——**在正在跨越沟的区段把跟踪奖励反向**（过沟时速度必然偏离，不该继续扣分），权重曾用 6（当前配置里注释掉，属试验项）。这正是我们「停在沟前只丢一点点跟踪分」的对症补丁。

### 21.3 可迁移到我们配方的动作（按性价比）

- **A. 纯权重（最小改动）**：`track_lin_vel_xy_exp` 1.5 → **3~5**；`flat_orientation_l2` −5 → **−0.5~−1**；`action_rate_l2` −0.01 → **0**；障碍列 `forward_speed_range` 上限 1.0 → **1.5**（速度越高越容易越过沟）。
- **B. 过沟补偿/正向项（需少量新代码）**：用我们已有的 4 个足端 RayCaster 判定「是否正在跨越空洞（射线部分或全部落空）」，在该掩码上给正向奖励，或按他们的思路**反转跟踪奖励**，消除「停下更划算」。
- **C. 架构级（最有效，改动最大）**：把 gap 从物理洞改成**虚拟地板高度 + 穿透深度×速度惩罚**（照搬 `get_leap_penetration_depths` 的思路）。
- **D. 训练流程**：把过沟做成**独立 fine-tune**——从当前 16.5k 行走策略 resume，用「只含 gap」的赛道、固定难度、几千轮（与 §15 的结论一致：4000 轮足以判断配方好坏）。

**不可直接照抄的点**：他们跟踪的是**世界系**速度并额外用 `yaw_abs`/`lin_pos_y` 保直行，我们是**体系**速度 + heading 控制，权重数值不可平移；其地形/奖励依赖自有 `BarrierTrack` 类，无法直接搬进 Isaac Lab。

### 21.4 已照搬的权重（2026-09-24，`CMoE_env_cfg.py`）

按 §21.3 的 A 项执行，**只搬可平移的权重**（不改命令域、不改地形、不动没有对应项的机制）：

| 项 | 改前 | 改后 | 依据 |
|---|---:|---:|---|
| `track_lin_vel_xy_exp` | +1.5 | **+5.0** | parkour leap 的 `tracking_world_vel=5.0` |
| `flat_orientation_l2` | −5.0 | **−0.1** | parkour leap 的 `orientation=−0.1` |
| `lin_vel_z_l2` | −2.0 | **0.0（裁掉）** | parkour leap 无此项；且它抑制跃起所需的竖向运动 |
| `ang_vel_xy_l2` | −0.05 | **0.0（裁掉）** | parkour leap 无此项；跃起时俯仰角速度必然峰值 |
| `action_rate_l2` | −0.01 | **0.0（裁掉）** | parkour leap 已注释掉 |

**未照搬（有明确理由，写进代码注释）**：`track_ang_vel_z_exp` 保持 +0.6（他们 leap 的 yaw 命令是 0，该项近乎摆设；我们命令里有 ±1.0 yaw）、`base_height_l2` 保持 −10（他们用 `z_low` **终止**替代；直接删掉我们唯一的高度控制会重演 AMP 记录过的「贴地爬行」）、`feet_air_time` 保持 +1.0（他们 leap 无步态塑形，但这是我方唯一正向步态项）。

**效应用实测原项值重算**（每秒量，基于 16200+ 轮的原项时间均值）：

| | 正向 | 惩罚 | 净 | 惩罚/正向 |
|---|---:|---:|---:|---:|
| 旧 | +1.553 | −0.509 | +1.043 | 33% |
| **新** | **+4.071** | **−0.318** | **+3.754** | **8%** |

⇒ 净回报约 +2.7/s，跟踪项占正向的 **88%**；生效项由 15 项降为 **12 项**（`lin_vel_z_l2`/`ang_vel_xy_l2`/`action_rate_l2` 被裁，`feet_edge`/`feet_stumble` 已于 §20.1 裁掉）。`py_compile` 通过，**未运行验证**。

**遗留风险（下一步的候选）**：① `base_height_l2=−10` 现在是绝对主导惩罚，而 parkour 根本没有它（他们靠 `z_low` 终止）——**跃起本身需要短暂偏离 0.30 m 高度**，若策略仍不敢跃，下一个该放松的就是它（如 −3 或改成条件项）；② `lin_vel_z_l2` 归零后步态可能变弹，需回看高度 RMS 与竖直速度；③ 学习率仍贴 `adaptive` 下限（§15），若奖励尺度放大 5 倍后仍学不动，下一个嫌疑是 lr 调度而非奖励。

## 22. A 配方（parkour 权重）从零跑的首读：2026-09-24 15:50，253 轮

运行：`logs/cmoe/base_move_cmoe_rough/2026-09-24_15-34-14_cmoe_A_fresh`（从零、4096 环境、seed 1、`track_lin_vel_xy_exp=5.0` 等 A 项）。**这是第一次拿到课程机制的分解量**（§19 加的 `terrain_levels_vel_logged`）。

### 22.1 课程机制第一次被量化（253 轮）

| 量 | 首 | 末 | 解读 |
|---|---:|---:|---|
| `move_up_frac` | 0 | **0.910** | 91% 的回合在晋级 |
| `move_down_frac` | 0.950 | **0.085** | 降级从 95% 掉到 8.5% |
| `frozen_frac` | 0.161 | **0.005** | 「既不晋级也不降级」的冻结环境只剩 0.5% |
| `distance_mean` | 0.41 m | **9.21 m** | 平均每回合走 9.2 m（晋级阈值只需 >4 m） |
| `level_mean` / `min` / `max` | 3.50 / 0 / 6 | 3.97 / 0 / **9** | 已有环境摸到最高档 9 |

⇒ **§16 悬而未决的「4.6 是能力边界还是被冻住」有答案了：都不是「全冻住」，而是一个高晋级率 + 少数列被降级的混合**；`frozen_frac` 只有 0.5%，所以「命令为 0 导致永久冻结」不是主因。

### 22.2 逐地形列等级：沟壑是唯一还在掉的一列

| 列 | 首 | 末 |
|---|---:|---:|
| `pyramid_stairs`（上楼梯） | 3.455 | **5.118** |
| `pyramid_stairs_inv`（下楼梯） | 3.584 | **5.453** |
| `boxes`（独立台阶） | 3.554 | **4.761** |
| `hf_pyramid_slope` | 3.591 | 3.846 |
| `random_rough` | 3.475 | 3.564 |
| `hf_pyramid_slope_inv` | 3.443 | 3.076 |
| **`gap`（沟壑）** | 3.452 | **2.677**（max 3.452） |

⇒ **A 配方让台阶/楼梯列快速升级（+1.3~1.9 级），沟壑列却继续被降级**——「停在沟前」依旧存在。注意 253 轮还很早（旧 run 的总体均值在 241 轮时也曾跌到 1.24 再回升），但**「沟列落后于其它列」这个相对信号正是我们要修的东西**，需要跑到 ~1000–2000 轮再定论。

### 22.3 运动学与奖励的连带变化（253 轮）

- `Train/mean_reward` 0.14 → **58.3**（尺度约为旧配方的 4 倍，符合预期）；`mean_episode_length` 11 → **929**；`illegal_contact`（摔）**0.176**（旧 run 0–2k 桶是 0.394）；`time_out` 0.824。
- `track_lin_vel_xy_exp` = **+3.39/s ⇒ 核均值 0.678**（旧 run 0–2k 桶只有 0.466）⇒ **线速度跟踪早期明显变好**（权重 5 + 放松姿态/平滑的预期效果）。
- `track_ang_vel_z_exp` = +0.222 ⇒ 核 0.37（旧 run 0–2k 是 0.507）⇒ **偏航跟踪反而变弱**，与本轮未搬 yaw 相关项、且姿态/角速度惩罚被放松一致。
- 惩罚项同步变大：`contact_forces` −0.235（旧 −0.075）、`undesired_contacts` −0.143（旧 −0.020）、`joint_acc_l2` −0.034 ⇒ 动作更猛、冲击更大，属预期代价。
- **`Policy/mean_noise_std` 1.005 → 1.35（在涨）**，而旧 run 一路降到 0.48。可能因为 `action_rate_l2`/`ang_vel_xy_l2`/`lin_vel_z_l2` 三项已被裁掉、抖动不再被惩罚。**待观察**：若持续上涨，应补回一个小 `action_rate_l2`（≈−0.005）。

**结论**：A 对「走、跟速、回合长度」明确有效；**对沟壑本身尚未见效**（沟列是唯一被降级的一列）。下一步按 §21.3 走 B（过沟补偿/正向项），必要时 C（虚拟地形＋穿透惩罚）。

## 23. A 配方确认有效（2026-09-24 16:00，395 轮）+ 专家路由现状

### 23.1 沟壑已不再是短板（推翻 §22.2 的早期判断）

| 地形列 | 首 | 253 轮 | **395 轮** |
|---|---:|---:|---:|
| `boxes` 独立台阶 | 3.554 | 5.923 | **6.760** |
| `pyramid_stairs` / `_inv` | 3.455 / 3.584 | 6.154 / 6.568 | 6.684 / 6.763 |
| **`gap` 沟壑** | 3.452 | 4.190（曾回撤到 2.677） | **5.761** |
| `hf_pyramid_slope` / `random_rough` | 3.591 / 3.475 | 4.688 / 4.326 | 5.436 / 5.005 |
| `hf_pyramid_slope_inv` | 3.443 | 3.528 | 3.923（现为最落后一列） |
| `level_mean` / `max` | 3.498 / 6 | 4.994 / 9 | **5.783 / 9** |

- `move_up_frac` 近 100 轮均值 **0.827**、`move_down_frac` **0.143**、`frozen_frac` **0.030**、`distance_mean` **8.75 m**、`illegal_contact` 0.176→**0.147**、`mean_episode_length` **902**。
- `level_mean = 5.78` 已贴近「每回合都晋级」的理论稳态 **6.0**（§16 推导 + 蒙特卡洛）。
- **关键量化：卡住时 `level_gap=2.68` ⇒ 沟宽 0.105 m；现在 `5.76` ⇒ 沟宽 0.130 m。** 也就是说**它现在能过的沟比当初卡住它的沟还宽 2.5 cm** ⇒ 当初不是能力不够，而是「不敢迈」的决策问题；只改权重（同一具身体）就跨过了更宽的沟，印证 §20 的诊断。
- 现最落后的一列变成 `hf_pyramid_slope_inv`（3.92），不再是沟壑。

### 23.2 顺带发现：`base_height_l2`（−10）在沟上方**静默失效**

新 run 里 `base_height_l2` 的日志值 **349/400 个回合恰好 = 0.0**（旧 run 只有 75/16717）。原因是 `rewards.py:652` 的兜底分支：

```python
if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
    adjusted_target_height = asset.data.root_link_pos_w[:, 2]      # ⇒ 误差恒为 0
```

基座下方 9 条射线**全部落空（inf）时误差被记 0**。旧 run 里机器人停在沟前、射线总打在地面上，所以该惩罚有效；现在它真的跨到沟上方 ⇒ **唯一的高度惩罚恰好在最需要它的地方自动归零**。这等于无意中复刻了 parkour 的「过沟不罚高度」思路，也是本次能过沟的原因之一，但属于**侥幸**：若以后要加强过沟（B/C），高度约束必须在沟上方显式设计（parkour 是虚拟地板 0.15 m + 穿透深度×速度惩罚），不能依赖这个兜底。

### 23.3 专家路由现状（5 专家 MoE）

| 专家 | 新 run 首 → 中 → 末 | 旧 run（16.7k）首 → 末 |
|---|---|---|
| `expert_0` | 0.118 → 0.226 → **0.254**（升） | 0.220 → **0.030** |
| `expert_1` | 0.368 → 0.418 → **0.424**（主导） | 0.187 → **0.409** |
| `expert_2` | 0.160 → 0.173 → **0.187** | 0.268 → **0.241** |
| `expert_3` | 0.211 → 0.102 → **0.070**（降） | 0.171 → **0.300** |
| `expert_4` | 0.143 → 0.081 → **0.065**（降） | 0.155 → **0.020** |

- `Policy/gate_entropy`：新 run 1.515 → **1.306**（近 50 轮 1.312）；旧 run 末 **1.185**。均匀 = ln5 = **1.609**，完全塌缩 = 0 ⇒ **确有分化，但也没塌缩**。
- `Loss/contrastive`：0.115 → **0.0296**（两个 run 几乎同轨迹，旧 run 末 0.032）⇒ gate↔地形 的对齐在学。
- **两个 run 收敛到同一模式**：`expert_1` 独占 ~0.41–0.42（≈2.1× 均匀），另有 1–2 个次级，**至少 2 个专家被饿死（<7%）**。代码里**没有任何 load-balancing 项**（`entropy_coef` 只作用在动作 std 上），所以这种「富者愈富」是预期结果。
- 影响评估：当前证据是**不影响任务表现**——课程均值 5.78、沟 5.76 是在这套路由下达到的；过沟是靠奖励修好的，不是靠专家分工。但设计意图里的「5 专家各管一类地形」并未实现（移植记录也说明专家未与地形人工绑定）。
- 若要让 5 个专家真的分工，需要加负载均衡项或减少专家数；**若要看「专家是否按地形分工」，需要按地形列分组记录 gate 权重**（现在日志只有全局均值）——这是个可选的小改动。
- 同日其它待观察：`Policy/mean_noise_std` **1.527 仍在涨**（旧 run 降到 0.48）、`Loss/Kld Loss` 升到 **23.6**（旧 run 末 9.6，且旧 run 出现过 7e23 尖峰）。

## 24. 核实：「高度探测器在 20 m 高、观测被压到 ±1 ⇒ 地形信息失效」这个说法成立吗

**结论：不成立。** 20 m 只是**射线起点**高度，不是被测量；观测值是**米制的、相对基座的离地高度**。证据链（全部逐行核对）：

| 事实 | 位置 |
|---|---|
| `offset` **只加在射线起点**上（pattern 先生成 ray_starts，再加 `offset_pos`） | `sensors/ray_caster/ray_caster.py:214-220` |
| `_data.pos_w` 存的是 **prim（`{ENV}/Robot/base`）的世界位姿**，**不含** offset | 同文件 `236-251`（`self._view.get_world_poses` / `get_root_transforms`） |
| 观测 `height_scan = pos_w[:,2] − ray_hits_w[...,2] − offset(0.5)` ⇒ 量纲是米 | `envs/mdp/observations.py:289-297` |
| `max_distance` 默认 **1e6** ⇒ 20 m 起点能打到地面 | `ray_caster_cfg.py:78` |
| **先加噪、后裁剪**（故未命中 inf → −inf → clip → **−1.0**） | `managers/observation_manager.py:350-356` |

**Imgo2（base 0.30 m）的预期取值范围**：平地 ≈ **−0.20**（= 0.30 − 0 − 0.5）；5 cm 台阶 ≈ −0.25 ~ −0.20；**沟壑（9 条射线落空）→ −1.0**。⇒ 地形这一路**有结构**，而且**沟壑是最容易区分的那一类值**，不存在"平地/台阶/沟都读成 1"。

唯一的小瑕疵：`offset = 0.5` 是按约 0.5 m 高的机体（ANYmal 默认基线）调的中心，我们 base 0.30 m ⇒ 平地基线落在 **−0.20** 而不是 0（一个常数偏置，不损失信息，只占 `[-1,1]` 窗口的约 16%）。

### 24.1 但同一链路上确有一个真 bug（比上述疑虑更要紧）

`rewards.py:682` 的兜底是**整批（batch-wide）判定**：

```python
ray_hits = sensor.data.ray_hits_w[..., 2]
if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
    adjusted_target_height = asset.data.root_link_pos_w[:, 2]     # ⇒ 全场误差记 0
else:
    adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
```

4096 个环境里**只要有任意一个**的基座 9 射线落空（即它正悬在沟/洞上方），`.any()` 成立 ⇒ **所有环境**都走兜底分支、误差记 0 ⇒ **我们最强的 −10 高度惩罚被整批关掉**。

实证：旧 run（机器人停在沟前、基座从不悬空）**75/16717** 个回合的 `base_height_l2` 恰好为 0；新 A run（真的跨沟）**349/400** 恰好为 0。⇒ §23.2 里"侥幸"的表述应更精确为「**只要有一个环境在沟上方，所有环境都不罚**」，这是向量化兜底写法的缺陷，不是地形本身的问题。

**部署验证的含义**：637 = 450（10×45 历史）+ 187（地形扫描）。地形块必须**真喂**：yaw 对齐、1.6 × 1.0 m、0.1 m 网格、值 = **基座离地高度 − 0.5**（米，clip 到 [−1,1]）、**未命中 = −1**、无噪声。用常数 0 或 1 代替属分布外输入。⚠️ 修这个 bug 会改变奖励（−10 项"复活"），**不能加进正在跑的 run**。

## 25. 修复 `base_height_l2` 的整批兜底缺陷（2026-09-24）

**该不该修：应该。** 三条理由：

1. **语义与意图不符**：`else` 分支本来就是逐环境求射线均值，`if ... .any()` 却把整个 batch 切成兜底 ⇒ 一个环境悬在沟上，全场高度惩罚归零。
2. **奖励依赖其它环境的状态**（非平稳 / 信用分配被污染）：同一个状态在不同时刻拿到不同奖励，取决于**别的**环境是否在洞上方。
3. **奖励随 `num_envs` 变化**：环境数越多，「至少一个环境在洞上」的概率越接近 1 ⇒ 256 环境与 4096 环境训练的**不是同一个任务**，这与「可复现 / 可部署」的目标直接冲突。实测：旧 run（停在沟前、从不悬空）75/16717 回合为 0；真跨沟的新 run **349/400** 回合恰好为 0。

**改法**（`velocity/mdp/rewards.py` 的 `base_height_l2`，与同文件 `him_base_height` 的 masked-nanmean 写法对齐）：

```python
ray_hits = sensor.data.ray_hits_w[..., 2]                      # (N, R)
valid = ~torch.isnan(ray_hits) & ~torch.isinf(ray_hits) & (torch.abs(ray_hits) < 1e6)
valid_count = valid.sum(dim=1).clamp_min(1)
mean_hits = torch.where(valid, ray_hits, torch.zeros_like(ray_hits)).sum(dim=1) / valid_count
adjusted_target_height = torch.where(valid.any(dim=1), target_height + mean_hits,
                                    asset.data.root_link_pos_w[:, 2])
```

保留的行为：**该环境自己的射线全部落空时仍退回 `root_z`（误差 0）**——即「悬在洞上不罚高度」的原意不变，所以对已学会的过沟行为干扰很小（过沟期间该环境本就全落空）。

**离线验证**（纯 torch 复刻两种算法对照，本机可跑）：

| 场景 | 旧（整批） | 新（逐环境） |
|---|---|---|
| ① 平地 ×2（根高 0.30） | 0.0000, 0.0000 | 0.0000, 0.0000 |
| ② env0 悬空 / env1 高 5 cm | **0.0000, 0.0000** | 0.0000, **0.0025** |
| ③ env0 3/9 射线落空、地面 −0.05 | **0.0000, 0.0000** | **0.0025**, **0.0025** |

新增回归测试 `imgo2_rl/tests/test_base_height_per_env.py`（5 个用例，含「一个环境悬空不得清零其它环境」这条会**在旧实现上失败**的断言）。⚠️ 它需要 Isaac Sim 引导过的解释器（`rewards.py` 会 import isaaclab），**本机未能执行**；本机只跑通了 `py_compile` 与上面的纯 torch 对照。

**影响与生效条件**：

- **奖励会变**（−10 项在非悬空环境下"复活"，量级参考：误差 5 cm ⇒ 每步 −0.025）⇒ **不能加进正在跑的 run**；但**若之后对该 run `--resume`，会带上这个修复**，属于配方变化，需按新配方记。
- **跨任务影响**：所有使用 `base_height_l2` + 射线传感器的任务（PPO rough、HIM、AMP）都共享这个函数；平地上射线不会落空 ⇒ 对 AMP 平地任务无差异；对粗糙/台阶任务，同样修复了「一个环境落空全场归零」的问题。
- **观察判据（下一个 run）**：`Episode/Episode_Reward/base_height_l2` 不再有 ~87% 回合恰好为 0；同时确认 `level_gap` 仍能上升（即修复没有重新抑制过沟）。
- **同类隐患（本次未改，记录在案）**：`velocity/mdp/observations.py:120` 的 `amp_root_height` 用 `torch.mean(ray_hits)` 且**无 mask** ⇒ 射线落空时会得到 `root_z − inf = −inf`。AMP 跑平地故当前不触发，属潜在问题。

## 26. 用户回放观察：步态塌缩到同一「bound」（2026-09-24，待确认形态）

**观察**：回放 `model_1000.pt` 时，用户反馈"似乎塌缩到同一个 bound 步态"。

**机制（为什么会出现）**：现在的配方里**没有任何项区分 trot 与 bound**——

- `6220e43` 关掉五项步态 shaping（`feet_height`/`feet_height_body`/`feet_slide`/`feet_air_time_variance`/`joint_mirror`）；
- 搬 parkour 权重时又关掉 `lin_vel_z_l2`/`ang_vel_xy_l2`/`action_rate_l2` ⇒ **弹跳带来的俯仰振荡与冲击完全免费**；
- 唯一与步态相关的 `feet_air_time`（+1.0）日志值为 **−0.136/s（负）** ⇒ 落地时滞空时间短于 0.5 s 阈值，即"高频短步"，与 bound 的形态一致；
- 地形集里**没有平地**（最缓的是 `random_rough`），且要过 0.13 m 沟/5 cm 台阶/8 cm 独立台阶 ⇒ **对称的 bound 是这类地形的合理最优解**（爆发式、对称、易过障）。

**要注意的混淆**：`--num_envs=8` 配 `num_cols=10` 时，8 个环境都落在障碍列（上/下楼梯、独立台阶、粗糙、斜坡、沟壑），所以"看到的都是 bound"不等于"全局塌缩"。**待确认**：是①所有地形（含最缓的 random_rough）都 bound，还是②只在沟/台阶这类障碍上 bound、其余为对角步。①才需要改配方。

**候选修法（按"直接指定相位 → 只压弹跳 → 对齐真机"排序）**：

1. **直接指定 trot**：启用现成的 `feet_gait`（`velocity_env_cfg.py:551`，当前权重 0、`synced_feet_pair_names` 为空）：对角腿对 `(("FL_FOOT","RR_FOOT"),("FR_FOOT","RL_FOOT"))`、`std=√0.5`、`max_err=0.2`、权重 0.5–1.0。⚠️ 它**规定相位**，会与过沟所需的爆发式 bound 相互掣肘。
2. **只压"弹跳/砸地"、不规定相位**：恢复 `action_rate_l2 ≈ −0.005`、`ang_vel_xy_l2 ≈ −0.02`、`lin_vel_z_l2 ≈ −0.5`（顺带压住 §23.3 里 `mean_noise_std` 1.54 的上涨）。
3. **对齐真机录制步态**（最重）：走仓库现成的 AMP 判别器路线（参考 `docs/gait_reference_baseline.json` 与 AMP-05/06 的经验：45 维 actor、λ_gp 1–2、`feet_air_time` 目标约 0.2 s）。
4. **过障优先**：接受 bound，只做第 2 项（弱化弹跳幅度）。

**注意**：无论选哪条，都必须在 **77 维重训之前**定下来；否则又是一次 checkpoint 不兼容。

### 26.1 决策与实现：地形自适应步态（trot 为主、沟壑放开）

**用户 2026-09-24 明确目标**：「沟壑用 bound，其他大部分用 trot」。⇒ 因此**全局** trot 项（`feet_gait` 无掩码）是错的——它会在沟前也强行指定 trot，与跃起直接冲突。

**实现**：新增 `mdp/rewards.py::TrotWithoutGapReward(GaitReward)`，把 trot 相位塑形乘上「前方／足下无空洞」的掩码；`CMoE_env_cfg.py` 把 `feet_gait.func` 指向它并给对角腿对。

掩码 `gap_zone_mask`（任一为真即关闭 trot）：

1. `height_scanner`（77 条、前移 0.25 m 的 11×7）**最前三列（21 条）任一射线落空** ⇒ 前方 0.75 m 内无地面；
2. 任一**足端扫描器整束落空** ⇒ 该足已悬在洞口。

**为什么判据是「任一」而不是「全部」（关键）**：沟是横向条带，只会让某一列（7 条）落空，其余列会打到沟后面的平台。纯 torch 离线验证（21 条前沿射线）：

| 场景 | 任一落空（采用） | 全部落空（已弃） |
|---|---|---|
| ① 平地 | False | False |
| ② **前方 0.55 m 处一条横向沟** | **True** ✅ | **False** ❌ 漏判 |
| ③ 整个前沿条带悬空 | True | True |

另叠加的「足整束悬空」判据在 ④ 单独触发（1 号环境 True）。

**同时恢复的两项阻尼**（与掩码正交、各自有依据）：`action_rate_l2 = −0.01`、`ang_vel_xy_l2 = −0.05`（均为 PPO rough 原值），用于弱化弹跳的抖动与冲击，也顺带抑制 `mean_noise_std` 涨到 1.5+ 的趋势。**刻意未恢复 `lin_vel_z_l2`**：它直接惩罚竖直速度，会与过沟所需的爆发式跃起对抗。

**本方案不做的**：不**主动**奖励「沟上必须 bound」（策略目前本来就 bound；掩码已把 trot 约束从沟前撤走，等于把它留给策略自选）。若之后发现它反而在沟前改成 trot，再加一个 `mask=gap_zone` 的前后腿对 `GaitReward` 主动塑形 bound。

**附带工具**：`cmoe/play.py` 新增 `--scan187`（回放 2026-09-24 之前 187 维几何的旧 checkpoint，在 env cfg 构造后覆盖几何，因此不触发「必须 77 条」断言）；仓库源文件已恢复为 **77 版**（此前为回放被临时换成 187 快照——**若那时直接开重训会训成 187，已避免**）。

**待验证**：① `Episode_Reward/feet_gait` 在非沟地形应稳定为正值、在沟前窗口应为 0（掩码生效的指纹）；② 回放应看到「粗糙/台阶/斜坡为对角步、沟壑处仍为 bound/跃起」；③ `level_gap` 不应低于 A run（约 5.7）。若 trot 没出现 ⇒ 提高 `feet_gait` 权重（1.0 → 1.5~2.0）；若出现过沟变差 ⇒ 降低或把掩码窗口加大（`front_columns` 3 → 4）。

### 26.2 修复：浮点截断让「77 条射线」断言误报（2026-09-24，实测报错）

**现象**：`play.py`（以及任何会构造 env cfg 的入口）在 `load_cfg_from_registry` 处直接抛

```
RuntimeError: CMoE terrain scan must be 77 rays (11x7), got 11x6=66
```

**根因（是我写的判据错，不是传感器错）**：`patterns.grid_pattern` 用
`arange(start=-size/2, end=size/2 + 1e-9, step=resolution)` 生成网格 ⇒ **0.6 m / 0.1 m 会产出 7 条**（端点容差使其包含 +0.3）。而断言写成 `int(size/resolution) + 1`，被 IEEE754 骗到：`0.6/0.1 = 5.999999999999999` ⇒ `int()` 截断为 **5** ⇒ 报 11×6=66。**几何本身一直是 77**。

**同一缺陷还在第二处**：`TrotWithoutGapReward.__init__` 用同样的写法算 `num_x/num_y` 来决定前沿射线索引 ⇒ 若不修，`num_y` 会算成 6，**掩码会指到错误的射线**（静默错误，比崩溃更危险）。

**修法**：两处都改为镜像 `grid_pattern` 的容差写法，并在 `rewards.py` 提取共用助手：

```python
def _ray_count(span: float, resolution: float) -> int:
    """镜像 patterns.grid_pattern 的 arange(start, end + 1e-9, step)。"""
    return math.floor(span / resolution + 1.0e-9) + 1
```

`CMoE_env_cfg.py` 里用同一表达式（并补了 `import math`，此前该文件没有）。

**离线验证**（断言写法 vs `numpy.arange` 计数，逐几何对照）：

| 几何 | 断言 | arange | 一致 |
|---|---|---|---|
| (1.0, 0.6) @ 0.1（现行，目标 77） | 11×7=77 | 11×7=77 | ✅ |
| (1.6, 1.0) @ 0.1（旧 187） | 17×11=187 | 17×11=187 | ✅ |
| (1.5, 0.9) @ 0.15（备选） | 11×7=77 | 11×7=77 | ✅ |

前沿索引构造也一并核对：`num_x=11, num_y=7` ⇒ 前沿 **21 条**（行 0–6 全覆盖、列 [8,9,10]）✅。

**教训（已记入维护约定层面的认识）**：`py_compile` 通过 ≠ 配置可用——几何类断言只有在**真正实例化 env cfg**时才会执行，而本机没有 Isaac Sim 引导，所以这类错误只能在训练机上暴露。以后凡是「构造期判据」，在训练机启动时先看是否报错，再谈训练。

## 27. 地形难度提升：沟宽按体长 0.4–1.0 L（2026-09-24，用户决定）

**用户决定**：沟宽直接按**身体长度**给 **0.4 L – 1.0 L**。躯干 0.315 m ⇒ **0.126 – 0.315 m**，替代原 `gap_width_range=(0.08, 0.16)`（= 0.25–0.5 L，语义上是"地板缝"，不需要跃起）。`platform_length_range` **保持 (0.65, 0.95)**——验算显示 4 条最宽沟也放得进 8 m tile：

| 难度 | 沟宽 | /体长 | 平台 | 4 条沟位置（m） | 末沟止于 | tile 8 m |
|---:|---:|---:|---:|---|---:|---|
| 0.00 | 12.6 cm | 0.40 | 95 cm | 1.80–1.93, 2.88–3.00, 3.95–4.08, 5.03–5.15 | 5.15 m | ✅ |
| 0.62 | 24.3 cm | 0.77 | 76 cm | 1.80–2.04, 2.81–3.05, 3.81–4.06, 4.82–5.06 | 5.06 m | ✅ |
| 1.00 | 31.5 cm | 1.00 | 65 cm | 1.80–2.12, 2.77–3.08, 3.73–4.04, 4.70–5.01 | 5.01 m | ✅ |

课程仍在区间内插值：**level 0 ≈ 0.126 m**（≈ 旧配方最终学到的 0.13 m），**level 9 ≈ 0.306 m**（≈ 一个躯干长，必须跃起）。

**连带影响与调参入口**：

- `Episode/Curriculum/terrain_levels/level_gap` **初期会掉**（难度整体抬高），别当退化；判据是它能否重新爬升。
- 若 from-scratch 在沟上卡住（课程地板 0.126 m 对未训练策略偏高）⇒ 把下限降到 **0.10 m（0.32 L）**，留一点余量。
- 若"沟—平台—沟"节奏太挤（平台 0.65–0.76 m 内要落地并再加速）⇒ 抬高 `platform_length_range` 或把 `num_gaps` 4 → 3。
- 更宽的沟让 §26.1 的 trot 掩码更有意义（沟前 0.75 m 就会关掉 trot 塑形）；`feet_edge` 已归零，不存在"罚踩边缘、又要过沟"的矛盾。

**台阶改动尚未实施**（用户本轮只确认了沟宽）：提议的 `num_steps 4 → 6`、`step_height_range (0.025,0.08) → (0.03,0.10)`（上行/下行各一份）仍在待定，目的是让中等级也能看出明显上行梯段（现行 d=0.62 时单级仅 5.9 cm，低等级只有 2.5–5 cm，视觉上像缓坡）。

**待验证**：① `level_gap` 能否重新爬到 6 以上（≈0.24 m 沟）甚至接近 9（0.31 m）；② 回放应看到"真正需要跃起的沟"；③ 沟前是否仍能触发 bound（掩码生效）。

## 28. 上行台阶：专属更大列 + 做成真正的多级台阶（2026-09-24，用户要求）

**背景（先解释了"为什么回放没看见上行台阶"）**：play 的 `--num_envs=8`、`num_cols=10` 时 `terrain_types = floor(arange(8)/0.8) = [0,1,2,3,5,6,7,8]`，对应地形是 **列 0,1 上行台阶 / 列 2 下行 / 3 独立台阶 / 5 粗糙 / 6 斜坡 / 7 反斜坡 / 8 沟**——**上行台阶其实占了两列**。看不见的原因是低等级（初始等级 0–5 随机）下单级只有 **2.5–5.8 cm**（5–11° 缓坡），而下行那列是整段地面塌陷，视觉上极显眼。

**改动**（`CMoE_env_cfg.py`）：

| 项 | 改前 | 改后 |
|---|---|---|
| `pyramid_stairs`（上行）`proportion` | 0.15 | **0.20**（num_cols=20 时 3 → **4 列**；num_cols=10 时仍 2 列） |
| `pyramid_stairs`/`_inv` 的 `num_steps` | 4 | **6** |
| 两者 `step_height_range` | (0.025, 0.08) | **(0.03, 0.10)** |
| `random_rough.proportion` | 0.20 | **0.15**（让出 0.05，总比例仍 1.0） |

**新列分配**（离线按 `terrain_generator` 的 cumsum 规则复算）：

- play（`num_cols=10`）：上行 2、下行 1、独立台阶 2、粗糙 1、斜坡 1、反斜坡 1、沟 2 —— 7 类全在。
- 训练（`num_cols=20`）：上行 **4**、下行 2、独立台阶 3、粗糙 3、斜坡 2、反斜坡 2、沟 4。

**台阶几何**（6 级 × step_depth 0.30 = 1.8 m 跨度，x 从 2.0 到 3.8）：

| 难度 | 单级 | 6 级剖面 | 总升高 |
|---:|---:|---|---:|
| 0.62（当前课程） | 7.3 cm | 7.3 / 14.7 / 22.0 / 29.4 / 36.7 / **44.0** cm | 44 cm |
| 1.00（课程上限） | 10.0 cm | 10 / 20 / 30 / 40 / 50 / **60** cm | 60 cm |

**查看建议**：`--num_envs=10`（`num_cols=10`）可让 10 个列各占一个环境、一次看全 7 类地形；`--num_envs=8` 会跳过列 4 与 9。

**注意**：下行台阶也一并变成 6 级 / 3–10 cm（用户只点名上行，但两者共用同一几何参数更对称）；若希望下行维持 4 级，改回 `pyramid_stairs_inv` 的 `num_steps=4`、`step_height_range=(0.025,0.08)` 即可。

### 28.1 单级台阶定为 5–15 cm（2026-09-24，用户两次调整后的最终值）

**取值过程**：用户先要求 10–20 cm；先实现 `(0.05, 0.20)`（下限 0.05 是为保住可学的课程起点），随后用户改定为 **`step_height_range = (0.05, 0.15)`** ⇒ **最终单级 5–15 cm**。

**可行性核查（URDF）**：大腿 **0.22 m** + 小腿 **0.206 m** ⇒ 最大伸展 0.426 m；站立 0.30 m（伸展率 70%）。关节力矩上限 **23.7 N·m**、总质量 **12.70 kg**。**参考**：ZiwenZhuang/parkour 的 `jump` 障碍为 **height (0.2, 0.46) m**。

**最终剖面**（6 级 × `step_depth` 0.30 = 1.8 m 跨度，x 2.0→3.8；上下行同参数）：

| 等级 | difficulty | 单级 | 6 级剖面（cm） | 总升高 | 等效倾角 | 站上时该腿伸展率 |
|---:|---:|---:|---|---:|---:|---:|
| 0 | 0.05 | 5.5 cm | 6 / 11 / 16 / 22 / 28 / 33 | 33 cm | 10.4° | 58% |
| 3 | 0.35 | 8.5 cm | 8 / 17 / 26 / 34 / 42 / 51 | 51 cm | 15.8° | 50% |
| 5（play 上限） | 0.55 | **10.5 cm** | 11 / 21 / 32 / 42 / 52 / 63 | 63 cm | 19.3° | 46% |
| 9（训练上限） | 0.95 | **14.5 cm** | 14 / 29 / 43 / 58 / 72 / 87 | 87 cm | 25.8° | 36% |

（对照：上一版 `(0.05,0.20)` 在 level 9 是 19.2 cm/级、总 116 cm、32.7°；再早的 `(0.03,0.10)` 只有 9.7 cm/级。）
**障碍块 `boxes`** 同期改为 `step_height_range=(0.08, 0.30)`（level 9 ≈ 28.9 cm ≈ 0.92 体长）。

**需要跟进的步态问题**：台阶变成 10–20 cm 后，爬梯本质接近"跃上"，但 §26.1 的掩码只覆盖**沟壑**，台阶上 trot 塑形**仍然生效**，可能与该动作相互掣肘。三个选项：① 先观察（trot 只占 20% 权重，可能不构成阻碍）；② 把掩码扩展到"前方有**更高**的障碍"（可用前几列射线命中高度 > 基座局部地面 + 阈值来判定）；③ 干脆让台阶也放开相位。建议先按 ① 观察一次。

## 29. 步态相关奖励的最终设计（2026-09-24）+ 一个静默失效的 bug

**用户的关键判断**：「parkour 是在**已训好的行走策略**上再 fine-tune 技能，所以可以不加 feet 相关奖励；**我们是从零训，需要**」。据此确定：trot 地形上保留足端/相位塑形，仅障碍两类放开。

### 29.1 `feet_gait` 现在的实现（回答"怎么实现的"）

用的是仓库里现成的 `GaitReward`（`class` 型 term，建 RewardManager 时实例化），由我在 CMoE 里把 `func` 换成 `TrotWithoutGapReward`：

| 部件 | 内容 |
|---|---|
| 参数 | `synced_feet_pair_names`（**必须恰好 2 对 × 2 足**）、`std`(√0.5)、`max_err`(0.2)、`command_name`、`velocity_threshold`(0.5)、`command_threshold`(0.1)、`asset_cfg`、`sensor_cfg` |
| 同步项 | 对每对足：`exp(−[clip((air_i−air_j)²,max_err²) + clip((contact_i−contact_j)²,max_err²)] / std)` ⇒ 奖励**同相** |
| 反相项 | 对 4 组交叉对：`exp(−[clip((air_a−contact_b)²) + clip((contact_a−air_b)²)] / std)` ⇒ 奖励**对间反相** |
| 合成 | `sync₀ × sync₁ × async₀ × async₁ × async₂ × async₃` —— **6 个核相乘**（∈[0,1]，是一条很窄的脊：六个条件都要大致成立） |
| 门控 | 仅当 `‖cmd_xy‖>0.1` **或** `‖root_com_lin_vel_b[:2]‖>0.5`；再乘直立门控 `clamp(−g_z,0,0.7)/0.7` |
| 权重 | **+1.0**（tracking 是 +5.0 ⇒ 最多约 20% 塑形强度） |
| 掩码（我方新增） | `~no_trot_mask`，其中 mask ＝ ① 地形列 ∈ `("boxes","gap")`（静态，`__init__` 缓存）或 ② `height_scanner` 最前 3 列**任一**射线落空 或 ③ 任一足端扫描器整束落空 |

### 29.2 parkour 的做法（对照）

| parkour | 实现 | 与我们 |
|---|---|---|
| `_reward_sync_legs_cond` | **只在 engage `jump` 障碍时**，罚左右**后腿** action 差：`‖a_RR − flip(a_RL)‖₂`，肩关节符号翻转 | 同为"按地形条件生效"，但他们比较**左右腿 action 且翻转肩符号**（支持 bound/jump），我们比较**对角腿 joint_pos 不翻转**（支持 trot） |
| `_reward_sync_all_legs_cond` | 同上，比较**全部腿**（右腿组 FL,RL vs 左腿组 FR,RR），L1 范数，肩符号翻转 | 同上 |
| `_reward_sync_all_legs` | 同一比较的**无条件**版本（run 名里的 `pSyncSymLegs4e-01/6e-01` ⇒ 权重 0.4–0.6） | 我们仓库也有对应的 `action_mirror`（权重 0，对角对、无符号翻转） |
| `_reward_feet_air_time` | `Σ((air_time − 0.5) × first_contact) × (‖cmd_xy‖>0.1)`，权重 1.0；接触用 `contact_filt = contact | last_contact` 过滤 PhysX 网格接触噪声 | **公式与我们完全一致**（我们用 Isaac Lab 的 ContactSensor API，滤波由传感器内部完成） |
| leap 配方 | **不含** `feet_air_time`/任何足端塑形 | 正因他们在已训好的策略上 fine-tune（见用户判断） |

### 29.3 三条已落地的修改（含一个静默失效的 bug）

1. **bug：`joint_mirror` 被晚赋值清零**。我加的 `joint_mirror.weight = -1.0` 在 `__post_init__` 里**位于** `6220e43` 那个零权重块**之前**，而后者又写了 `joint_mirror.weight = 0.0`（Python 晚赋值生效）⇒ mirror **静默失效**。已删除该清零行并加注释说明；同时用脚本按**文件内赋值顺序**模拟出最终权重做校验（这是能抓住此类错误的检查）。
2. **`feet_air_time` 阈值 0.5 → 0.25 s**。参考基线步态周期中位 **0.6001 s（1.67 Hz）** ⇒ trot 单足摆动约 **0.3 s**，`(0.3 − 0.5)` 恒为负 ⇒ 该"奖励项"实测是 **−0.136/s**，一直在**扣分**。阈值改 0.25 后：正常的 0.3 s 摆动得正分、<0.25 s 的碎步被罚。
3. **恢复 `feet_height_body`（权重 −5.0，PPO rough 原值）但按地形豁免**（`MaskedFeetHeightBody`）：摆动足机体系高度误差（目标 −0.20 m ≈ 离地 0.10 m）；参考基线足端 z 峰峰中位 **0.090 m**，与目标同量级。障碍块/沟槽豁免——那里需要抬更高或跃起。

**仍未恢复**：`feet_air_time_variance`（−8.0）。它与相位项＋mirror 功能重叠，且是把步态约束得最死的项（原来量级最大的步态惩罚）。需要"四足步态更均匀"时再开。

### 29.4 让步态在两类地形上的分工（当前）

| 地形 | 列数(training) | 步态约束 |
|---|---:|---|
| 上行/下行台阶、平地、粗糙面、斜坡、反斜坡 | 4+2+2+2+2+1 = **13/20** | `feet_gait`(trot) + `joint_mirror`(对角) + `feet_height_body` + `feet_air_time`(0.25 s) |
| **障碍块 `boxes`、沟槽 `gap`** | 3+4 = **7/20** | **全部豁免** ⇒ 相位自由（bound/跃起） |

**待验证**：`Episode_Reward/feet_gait` 在非障碍地形应为正、在障碍地形应为 0；`joint_mirror`/`feet_height_body` 应为非零；回放应见"台阶/斜坡 trot、障碍上跃起"；`level_gap` 与 `level_boxes` 不应低于基座水平。

### 29.5 "feet_gait / joint_mirror / feet_height_body 是否重复"的分析与结论

**结论：三者都保留，它们管的维度不同且互补。**

| 项 | 管的量 | 是否可被其它项替代 |
|---|---|---|
| `feet_gait` | **接触时序/相位**：同相对（FL↔RR、FR↔RL）+ 4 组交叉对**反相** | ❌ 不可。`joint_mirror` 是**瞬时姿态**代理，无法区分 **trot 与 pronk（四足同时跳）**——pronk 时对角腿姿态也一致，mirror 照样满足；只有 `feet_gait` 的 async 项（交叉对必须反相）能排除它 |
| `joint_mirror` | **瞬时关节姿态的对角一致性**（二次罚） | 部分可替代，但有独特价值：**梯度性质互补**。`feet_gait` 是 **6 个指数核的乘积**（很窄的脊），随机策略早期六个核都≈0 ⇒ 乘积≈0 ⇒ **梯度几乎为零**，学得慢；`joint_mirror` 从第一步就有**稠密**梯度。二者一起＝"早期有稠密梯度 + 后期定时序" |
| `feet_height_body` | **抬脚高度（clearance）**：摆动足机体系高度误差（目标 −0.20 m ≈ 离地 0.10 m） | ❌ 不可替代（前两者都不约束高度）。从零训练没有它极易学到**贴地拖步**（仓库历史：AMP 工作把 PPO 步态描述为"贴地爬行"）；参考基线足端 z 峰峰中位 **0.090 m**，而台阶 5–15 cm ⇒ 必须抬起来 |

**代价可控性**：`feet_height_body` 是**双向软罚**。必须抬 15 cm 时误差 0.05 m ⇒ 罚 `0.05²×5 = 0.0125`/步，相对 `track_lin_vel 5.0` 可忽略 ⇒ 不会挡住爬梯；障碍块（最高 30 cm）已豁免。若嫌紧：目标抬到 0.15 m 或权重 −5.0 → −2.0。

### 29.6 顺带修掉的符号陷阱：`joint_mirror` 原配置会把左右 hip 拉向"扭转站姿"

> ⚠️ **本节结论已于 2026-09-24 更正（见 §29.9）**：文中"mirror 对是跨左右两侧"的前提是**错的**。
> PPO 的对是 **FR↔RL、FL↔RR（对角对）**，不是左右对；trot 里对角腿同相，加上本 URDF
> 四条腿关节轴完全相同 ⇒ **各关节同号**，不翻转符号是正确的。parkour 的 `_sync_legs_cond`
> 比较的是 **RR↔RL（左右对）** 并翻转肩关节符号，那是"轴约定左右相反"的另一类对称，不可直接类比。
> 因此"删掉 hip"的改动已撤回（hip 已按用户要求恢复）。本节保留作为**错误推理的历史留痕**，
> 只有最后一段"parkour 翻转肩符号"的事实描述仍成立。

**URDF 事实**：左右腿关节轴**完全相同**（hip `(1,0,0)`、thigh/shank `(0,1,0)`，`imgo2.urdf`）。⇒ 对称外展时**左右 hip 应反号**、thigh/shank 同号。默认站姿 `.*_hip_joint = 0.0`（`assets/imgo2.py:69`）——所以 hips 近 0 时该错误影响有限（`(0−0)²=0` 自然满足），但**任何对称外展**都会被 `(q_hip,a − q_hip,b)²` 推向"同号"这一**扭转解**。
而 `joint_mirror` 的对是**跨左右**的（FR↔RL、FL↔RR）且**无符号翻转** ⇒ hip 分量方向错误。**对照 parkour**：其 `_sync_legs_cond`/`_sync_all_legs_cond` 专门**翻转肩关节符号**，反证了跨左右比较必须有符号处理。

**修法（已落地）**：CMoE 的 `joint_mirror.params["mirror_joints"]` 限制为 **`(thigh|shank)`**（去掉 hip）。理由：trot 中 hip 外展基本不变，**前后摆动才是步态相关自由度**；髋部姿态由 `joint_pos_penalty`（全关节偏离默认位姿，−0.1）约束。（备选更忠实的做法：给 hip 加符号向量；本轮取简单的那个。）

**"trot 地形"的完整步态栈（当前）**：`feet_gait` **+1.0**（对角相位，掩码）+ `joint_mirror` **−1.0**（thigh/shank 对角一致，掩码）+ `feet_height_body` **−5.0**（抬脚 0.10 m，掩码）+ `feet_air_time` **+1.0 / 阈值 0.25 s**（全局，`‖cmd‖>0.1` 门控）。障碍块与沟槽（7/20 列）全部豁免。

### 29.7 `feet_air_time` 是否该对所有地形生效？→ 现状已是全局，且**应当保持全局**

> ⚠️ **本节"由负翻正"的解读已于 2026-09-24 更正（见 §29.9）**：展开 `Σ(last_air_time − c)`
> 可见 `c` 只是"每次落地扣一个常数"，对滞空时间的**梯度方向与 c 无关** ⇒ 0.25 与 0.5 的
> **方向一致**，差别只在 **`c·N_落地` 这一项的强度**（0.5 的"减少落地次数／拉长步幅"压力是 0.25 的 2 倍）
> 与日志偏移量。本节写的"0.5 时一直在惩罚抬脚、方向上鼓励贴地"**方向判断有误**；
> "该分项实测为负"这一事实仍成立。用户已决定回到 0.5 并加地形掩码。

**现状复核**（用户 2026-09-24 提问）：CMoE 只覆盖了 `threshold`（0.5 → **0.25**），**没有加地形掩码**；`sensor_cfg.body_names` 继承 `rough_env_cfg.py:114` 的 `.*_FOOT`（四只脚全算）。所以它**已经对所有地形生效**。

**函数语义**（`mdp/rewards.py:523`，本仓版本比上游多一道门）：
```
reward = Σ_feet (last_air_time − threshold) · first_contact      # 只在"落地那一帧"结算
reward *= (‖cmd_xy‖ > 0.1)                                       # 站立/零命令不结算
reward *= clamp(−g_b[2], 0, 0.7) / 0.7                           # 姿态倾斜超约 45° 归零
```
它是**一次性脉冲**，不是逐步项；也没有任何地形判断。

**为什么应该保持全局（而 `feet_gait`／`joint_mirror` 必须掩码）——偏好方向不同**：

| 项 | 偏好 | 与障碍块/沟槽所需行为（bound／leap）的关系 |
|---|---|---|
| `feet_gait` | 交叉对**反相**（trot） | **冲突** ⇒ 必须掩码 |
| `joint_mirror` | 对角**同相姿态** | **冲突**（leap 是对角同动） ⇒ 必须掩码 |
| `feet_air_time` | 腾空**越久越好** | **一致** ⇒ 不需要掩码，且是**顺风** |

也就是说：过沟需要更长腾空 ⇒ `last_air_time` 更大 ⇒ **结算更多**。掩掉它反而丢掉了沟壑/障碍块上唯一的正向腾空激励。而且它只在**成功落地**时结算，"飞进沟里"不结算 ⇒ 自带上限。

**另一个此前被忽略的正向作用**：阈值从 0.5 降到 0.25 不只是修 trot，还**把过沟的激励由负翻正**。粗算（步态周期 0.6 s、单足摆动 ≈0.3 s、两足同时落地）：阈值 0.5 时每次落地 `0.3−0.5 = −0.2`、每周期 −0.4 ⇒ 约 **−0.67/s**（相对 `track_lin_vel` ≈3.5/s 约 19%），即 A 跑里 `feet_air_time` 一直在**惩罚抬脚**，方向上鼓励"贴地"；阈值 0.25 后同样落地变 **+0.05/足**、约 **+0.17/s**。实测旧值 −0.136/s 与同量级（门控会再削一部分），方向一致。

**幅度对比（为什么它不能替代 `feet_gait`）**：正常 trot 结算 ≈0.10/步态周期 ⇒ 时间平均 ≈**0.17/s**，只占 `track_lin_vel`（5.0 × 核值 ≈0.7 ⇒ ≈3.5/s）的 **5%**。它是"别踩碎步"的**温和推力**，管不了相位——相位仍归 `feet_gait`。

**唯一需要留意的副作用**：「腾空越久越好」是诱导 pronk/hop 的经典来源。已有的护栏：① 命令 + 姿态双门控；② `feet_height_body`（−5.0）与 `action_rate_l2`（−0.01）塑形。注意障碍块/沟槽**豁免**了 `feet_height_body`，那里只剩速度跟踪项约束 ⇒ 这是**期望的**（跃起仍须按命令速度**向前**，原地弹跳拿不到 `track_lin_vel`）。若将来观测到全地形 hop，优先降权（1.0 → 0.5）而不是加掩码。

### 29.8 复盘：之前那个「trot 步态还行」的 PPO 用的是什么配方

**起因**：用户问「之前 ppo 我训练出来过一个 trot 步态还行的，那次的设置是怎样的」。

#### 29.8.1 先说证据边界

- **本机没有任何 PPO 训练产物**：`find /root -name "events.out.tfevents*"` 只有 AMP 与 CMoE 的 run（另有 CMoE 13 个 run）；`model_*.pt` 也只在 AMP/CMoE 目录下。所以**读不到那次的曲线、轮数、params 快照**。
- 配方可以从**源码**确定：`git log --follow -- base_move/rough_env_cfg.py` 只有两条提交（`f6acff9` 初始提交、`8dcb9d5` 目录整合），**奖励段自导入起没有改过**。`RewardsCfg` 基础类里 36 个 term **全部默认权重为 0**（逐项脚本核对），所以「生效集合」＝`rough_env_cfg.py:103–169` 显式赋非零权重的那批。数出来正好 **19 项**，与 README 里 `37b9206` 那条记录的「19 项」一致 ⇒ 互为交叉验证。
- 参考侧（6/30 的 `policy_flat.pt`）训练在另一台机的 `/root/gpufree-data/...`，`docs/ppo_policy_provenance_2026-09-19.md` §7 已记「本机没有它们的 checkpoint，无法从 checkpoint 侧复核轮数与奖励」。

#### 29.8.2 那次的 PPO 奖励设置（19 项，`Imgo2-basemove-rough-ppo`）

| term | 权重 | 语义 |
|---|---:|---|
| `track_lin_vel_xy_exp` | **+1.5** | xy 线速度指数核，`std=0.5` |
| `track_ang_vel_z_exp` | **+0.6** | z 角速度指数核，`std=0.5` |
| `feet_air_time` | **+1.0** | 落地瞬间 `(滞空 − 0.5 s)`，命令范数 >0.1 才给 |
| `base_height_l2` | **−10.0** | `(base z − (0.30 + 局部地形高))²` |
| **`feet_air_time_variance`** | **−8.0** | 四足 `clip(滞空,0.5)` 与 `clip(触地,0.5)` 的**跨足方差和**，带直立门控 |
| `flat_orientation_l2` | **−5.0** | 重力投影水平分量² |
| **`feet_height_body`** | **−5.0** | 摆动足机体系高度偏离 `−0.20 m` 的平方 |
| `lin_vel_z_l2` | **−2.0** | base 竖直速度² |
| **`joint_mirror`** | **−1.0** | 对角腿关节角一致（**对角对含 hip**，即 `FR_(hip\|thigh\|shank).* ↔ RL_...`） |
| `undesired_contacts` | −0.5 | 非足端 link 受力 >1 N |
| `stand_still` | −0.1 | 命令范数 <0.05 时的关节偏离 L1 |
| `joint_pos_penalty` | −0.1 | 关节偏离默认位姿 L1，低速时 ×5 |
| **`feet_slide`** | **−0.05** | 接触中足端水平速度平方 |
| `ang_vel_xy_l2` | −0.05 | 横滚／俯仰角速度² |
| `contact_forces` | −0.02 | 足端受力超 100 N 的部分 |
| `action_rate_l2` | −0.01 | 相邻动作差² |
| `joint_torques_l2` | −2.5e-6 | 关节力矩² |
| `joint_power` | −2e-5 | `Σ\|τ·q̇\|` |
| `joint_acc_l2` | −5e-9 | 关节加速度² |

**未开启**（权重 0 ⇒ `disable_zero_weight_rewards()` 直接置 None）：`feet_gait`、`feet_contact`、`feet_stumble`、`feet_height`、`action_mirror`、`action_sync`、`joint_deviation_l1`、`torques/vel limits`、`is_terminated`、`upward`、`feet_distance_y_exp` 等。

#### 29.8.3 ⚠️ 关键结论：**PPO 从来没开过 `feet_gait`**

全历史 pickaxe（`git log --all -S "feet_gait.weight"`）只命中一次赋值：`height_move` 里写 `= 0`；`GaitReward` 本身只在初始导入时出现。也就是说**那次 PPO 的 trot 不是相位奖励做出来的**，而是靠这批**「塑形 + 对称 + 抬脚 + 滑移」**项的组合：

```
feet_air_time_variance −8.0   ← 让步态"四足时序均匀"（量级最大的步态项）
joint_mirror          −1.0   ← 对角姿态一致
feet_height_body      −5.0   ← 强制抬脚（离地 0.10 m）
feet_air_time         +1.0   ← 别踩碎步
feet_slide            −0.05  ← 接触期别蹭地
+ lin_vel_z_l2 −2.0 / flat_orientation_l2 −5.0 / stand_still −0.1 / joint_pos_penalty −0.1
```

**而 CMoE 的 `6220e43` 恰好把其中五项清零了**（`feet_height`、`feet_height_body`、`feet_slide`、`feet_air_time_variance`、`joint_mirror`）——README 记的「步态自由度交给策略」正是 §26「所有地形塌缩到同一个 bound」的直接原因：**PPO 那套是靠这五项维系的，清掉就没**。我们这一轮已经恢复了 `joint_mirror`（−1.0，掩码）、`feet_height_body`（−5.0，掩码）、并把 `feet_air_time` 阈值 0.5 → 0.25；**仍缺 `feet_air_time_variance`（−8.0）与 `feet_slide`（−0.05）**。

#### 29.8.4 训练超参与其他设置（PPO）

| 项 | 值 |
|---|---|
| runner | `Imgo2RoughPPORunnerCfg`：`num_steps_per_env 24`、`max_iterations 2000`（flat 版 **5000**）、`save_interval 100`、`clip_actions 3` |
| 网络 | actor/critic **`[512,256,128]`**、`elu`、`init_noise_std 1.0` |
| 优化 | `lr 1e-3`、`schedule=adaptive`、`desired_kl 0.01`、`gamma 0.99`、`lam 0.95`、`clip_param 0.2`、`epochs 5`、`minibatches 4`、`entropy 0.01`、`value_coef 1.0`、`max_grad_norm 1.0` |
| 观测 | actor **45**（`ang_vel 3 + gravity 3 + cmd 3 + dof_pos 12 + dof_vel 12 + action 12`，**无历史、无高度扫描**）；critic **235**（45 + `base_lin_vel 3` + `height_scan 187`） |
| 地形 | rough = 六类 `terrain_generator`（楼梯/反向楼梯/`boxes`/`random_rough`/正反斜坡）+ `terrain_levels` 课程；flat = `terrain_type="plane"`、无扫描、无课程 |
| 命令 | `vx ∈ [−1,1]`、`vy ∈ [−0.8,0.8]`、`yaw ∈ [−1.5,1.5]`，`resampling 10 s`、`rel_standing_envs 0.02`、`heading_command` |
| 终止 | **只有 `time_out` / `terrain_out_of_bounds`**；`illegal_contact = None`（**基座触地不终止**）⇒ 与 CMoE 现配方最大的行为差异 |
| 域随机化 | 摩擦 `0.3–1.0/0.3–0.8`、restitution `0–0.5`、base 质量 `+(-1,3) kg`、其余 `×0.7–1.3`、CoM `±0.05 m`、reset `x,y ±1 m / yaw ±π / **roll,pitch ±0.3 rad**`、执行器增益 `×0.5–2.0`；外力/推挤显式关闭 |

**部署侧实测（平地上，`policy_flat.pt`）**：周期强度 **0.95**、**FL–FR 相位 +175°**（＝对角 trot），`vx=0.5` 实测 0.428 m/s、`z_mean 0.288 m`、roll/pitch `2.5°/1.3°`（见 `docs/ppo_policy_provenance_2026-09-19.md` §4）。这与「trot 步态还行」的记忆吻合，也说明它是**平地版**。

#### 29.8.5 与当前 CMoE 的逐项差异（可用于单变量回退）

| term | PPO rough | CMoE 现 | 说明 |
|---|---:|---:|---|
| `track_lin_vel_xy_exp` | 1.5 | **5.0** | A 配方加重 3.3× |
| `feet_air_time` | +1.0 @ **0.5 s** | +1.0 @ **0.25 s** | 已修（0.5 对 1.67 Hz 恒负） |
| `feet_air_time_variance` | **−8.0** | **0** | **缺** |
| `feet_slide` | **−0.05** | **0** | **缺** |
| `feet_height_body` | −5.0 | −5.0（掩码） | 已恢复 |
| `joint_mirror` | −1.0（含 hip） | −1.0（仅 thigh/shank，掩码） | 已恢复并修符号陷阱 |
| `feet_gait` | **0** | **+1.0** | CMoE 新增 |
| `flat_orientation_l2` | −5.0 | **−0.1** | 放松 50× |
| `lin_vel_z_l2` | −2.0 | **0** | 关闭 |
| `undesired_contacts` | −0.5 | −0.5 | 同 |
| 基座触地终止 | 无 | **有**（1 N） | 行为差异 |

**建议的单变量实验（若要复刻 PPO 的 trot）**：只加回 `feet_air_time_variance`，权重从 **−2.0** 起步（PPO 是 −8.0，但那时没有 `feet_gait` 相位项；两者叠加会过约束），观察 `Episode_Reward/feet_air_time_variance` 与足端时序是否变均匀，再决定是否加到 −8.0。`feet_slide` 可暂缓（它会轻微惩罚沟壑/障碍块上的落地位移）。

## 29.9 步态 shaping 定稿：照搬 PPO 三项 + 地形掩码，**不用 `feet_gait`**（2026-09-24 用户决定）

用户逐条给出决定：「倾向于也使用 `joint_mirror −1.0`（对角姿态一致，**含 hip**）、`feet_height_body −5.0`
（强制抬脚，离地 0.10 m）、`feet_air_time +1.0`（**阈值 0.5**），然后添加地形掩码」＋「**不使用 `feet_gait`**」。

### 29.9.1 落地改动

| 文件 | 改动 |
|---|---|
| `CMoE_env_cfg.py` | ① `feet_gait` 的 func/weight/params 赋值**全部删除**（保持基类 0.0 ⇒ `disable_zero_weight_rewards()` 整个移除，**不会实例化**，因此不再需要 `synced_feet_pair_names`）；② `joint_mirror.params["mirror_joints"]` 恢复为 **`(hip\|thigh\|shank)`**（＝PPO 原配置）；③ `feet_air_time` 改为 `mdp.MaskedFeetAirTime`、阈值回到 **0.5**、加 `free_terrain_names=("boxes","gap")`；④ `feet_height_body`（−5.0）与 `joint_mirror`（−1.0）掩码不变 |
| `mdp/rewards.py` | 新增 `class MaskedFeetAirTime(ManagerTermBase)`（在 `MaskedFeetHeightBody` 之后），语义＝`feet_air_time` ×（非豁免地形），与另两个 masked 版本同一套 `_terrain_type_mask` 机制 |

**为什么 `feet_gait` 可以直接"不赋值"**：`velocity_env_cfg.py:551` 的基类权重就是 `0.0`，而
`disable_zero_weight_rewards()` 在 `__post_init__` 末尾（实例化之前）执行 ⇒ 权重仍为 0 的 term 被整体置 `None`，
`ManagerTermBase` 子类**不会被构造**，所以空 `synced_feet_pair_names` 不会触发它 `__init__` 里的 `ValueError`。
`mdp.TrotWithoutGapReward` / `mdp.GaitReward` 保留在 mdp 里未启用，回退只需重新赋值。

### 29.9.2 生效配方（16 项，AST 按源文件顺序模拟所得）

| term | 权重 | 掩码 |
|---|---:|---|
| `track_lin_vel_xy_exp` | **+5.0** | — |
| `track_ang_vel_z_exp` | +0.6 | — |
| **`feet_air_time`** | **+1.0**（阈值 **0.5**） | **障碍块/沟槽豁免** |
| `base_height_l2` | −10.0 | — |
| **`feet_height_body`** | **−5.0** | **障碍块/沟槽豁免** |
| **`joint_mirror`** | **−1.0**（含 hip） | **障碍块/沟槽豁免** |
| `flat_orientation_l2` | −0.1 | — |
| `stand_still` | −0.1 | — |
| `joint_pos_penalty` | −0.1 | — |
| `ang_vel_xy_l2` | −0.05 | — |
| `contact_forces` | −0.02 | — |
| `action_rate_l2` | −0.01 | — |
| `undesired_contacts` | −0.5 | — |
| `joint_torques_l2` / `joint_power` / `joint_acc_l2` | −2.5e-6 / −2e-5 / −5e-9 | — |

清零 ⇒ 移除：`feet_gait`、`feet_air_time_variance`（−8.0）、`feet_slide`（−0.05）、`feet_height`、`feet_edge`、`feet_stumble`、`lin_vel_z_l2`、`feet_contact`、`action_mirror`/`action_sync` 等。
覆盖：**13/20 列**受 trot 塑形（楼梯／独立台阶／粗糙／斜坡／平地），**7/20 列**自由（`boxes` 3 + `gap` 4）。

### 29.9.3 两处更正（此前记录的推理有误）

1. **`joint_mirror` 的对角对不该删 hip**（原 §29.6 已作废）：PPO 的对是 **FR↔RL、FL↔RR（对角）**，
   而 trot 中**对角腿同相**且本 URDF 四腿关节轴完全相同 ⇒ 同相＝**同号**，不翻符号是对的。
   parkour 的 `_sync_legs_cond` 比的是 **RR↔RL（左右对）** 并翻肩符号，属另一类对称，"轴约定左右相反"，
   不能类比。唯一残留近似：*对称外展*是 `q_FR = −q_RL`，不是本项最小点（`q_FR = q_RL`）⇒ 该项会轻微
   抑制髋外展；因默认站姿 `hip=0` 正是最小值点、trot 中髋基本不动，影响有限。**若日后见歪斜/单侧铺腿，
   正确修法是对 hip 翻转符号，而不是删 hip。**
2. **`feet_air_time` 的阈值不改变梯度方向**（原 §29.7 已更正）：`Σ(last_air_time − c) = Σ last_air_time − c·N_落地`
   ⇒ `c` 越大，"**减少落地次数（步幅更长／步频更低）**"的压力越大、日志偏移越负，但对滞空时间的
   梯度始终是 +1。所以 0.5 与 0.25 不是"方向相反"，而是**强度差 2 倍**。回到 0.5 意味着比 0.25 更强调"别踩碎步"。

### 29.9.4 ⚠️ 已知代价：不再有任何项区分 trot 与 pronk

去掉 `feet_gait` 后，三项 shaping 的信息量是：
- `joint_mirror`（对角对）能排除 **bound**（前后腿同相）与 **pace**（左右腿同相），
- 但**排除不了 pronk**（四足同时跳时对角腿同样同相，该项照样满足），
- `feet_height_body` 只约束抬脚高度、`feet_air_time` 只偏好更长滞空 ⇒ 都不含相位信息。

**依据**：那次 PPO 同样没有 `feet_gait`，这套组合仍收敛到 trot（部署侧平地实测周期强度 **0.95**、
FL–FR 相位 **+175°**，见 §29.8.4），故先按用户决定执行。**回放判据**：若看到对称弹跳，
第一顺位恢复 `feet_gait`（`mdp.TrotWithoutGapReward`，权重 1.0），第二顺位恢复 `feet_air_time_variance`（−8.0）。

### 29.9.5 仍未恢复

`feet_air_time_variance`（−8.0，PPO 里量级最大的步态项）与 `feet_slide`（−0.05）**仍为 0**。
若想要"四足时序更均匀"，建议单变量只加 `feet_air_time_variance` 且从 **−2.0** 起步
（PPO 的 −8.0 是在没有别的相位项时用的）。

### 29.9.6 验证状态

- **已做**：`py_compile` 两个改动文件通过；用 AST 按**源文件顺序**（＝实际执行顺序）模拟
  `rough_env_cfg.__post_init__` → `CMoE_env_cfg.__post_init__` 的全部 `self.rewards.*.weight/func` 赋值，
  确认 16 项生效、`feet_gait` 无任何赋值、`joint_mirror −1.0` 不再被晚赋值覆盖；
  逐条核对三个 class-style term 的 `__call__` 签名与 `term_cfg.params` 集合 —— Isaac Lab
  `manager_base.py:362` 要求 `set(args[2:]) == set(term_params + args_with_defaults)`，三者均满足
  （`free_terrain_names` 有默认值，因此加不加进 `params` 都能通过校验）；`git diff --check` 通过。
- **未做（需训练机）**：env 构造（`MaskedFeetAirTime` 是否会因掩码/实例化抛错）、`feet_air_time` 掩码是否
  真的只在 13 列生效、`Episode_Reward/feet_air_time` 与 `joint_mirror` 的实际量级、有无 pronk/bound。
  **启动后先确认那一行**：`[INFO] num_terrain_obs: 77 / num_obs (total): 527 / num_privileged_obs: 125`，
  再看 TB 里 `Episode_Reward/*` 是否出现 `feet_air_time` / `feet_height_body` / `joint_mirror`
  （`feet_gait` **不应**出现）。

### 29.10 沟壑上把 `joint_mirror` 换成 **bound**（2026-09-24 用户要求）

用户问「能不能设置掩码，让 mirror 在沟壑变成 bound」——可以，而且比单纯豁免更有信息量：**豁免＝不给先验，
换成左右对＝把罚项变成"bound 的形状先验"**。已落地。

#### 29.10.1 三种地形行为

| 地形 | 对子 | 语义 |
|---|---|---|
| trot 地形（13/20 列） | `mirror_joints`＝**对角对**（FR↔RL、FL↔RR） | trot：对角腿同相 |
| **`gap`（6 列）** | `bound_mirror_joints`＝**左右对**（FL↔FR、RL↔RR） | **bound：前腿一对同相、后腿一对同相** |
| `boxes`（1 列） | 不计（`free_terrain_names`） | 完全自由（保持先前决定） |

两边都**含 hip**，都不翻符号（本 URDF 四条腿轴完全相同 ⇒ 同相＝同号）。

#### 29.10.2 实现要点：为什么不能直接调两次 `mdp.joint_mirror`

上游 `joint_mirror` 把解析结果缓存在 **env** 上，而且**只在第一次调用时解析**：

```python
if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
    env.joint_mirror_joints_cache = [[asset.find_joints(jn) for jn in pair] for pair in mirror_joints]
```

⇒ 在同一个环境里用两套对子连调两次，**第二次会被静默忽略**，两套都按第一套算（掩码看起来生效、
实际对子错）。所以新增 `_pairwise_joint_mirror(env, asset, pairs, cache_owner, cache_attr)`：算式、
`1/len(pairs)` 归一化与直立门控与上游**逐字一致**，只是把缓存挂在 **term 实例** 上（`_trot_pairs` /
`_bound_pairs` 两个键），并在文档里记下这个上游陷阱。`MaskedJointMirror.__call__` 现在返回

```
trot_rew · 1[非 free 且非 bound]  +  bound_rew · 1[bound]
```

`bound_mirror_joints` 为空时退化为旧行为（沟壑＝豁免）。

#### 29.10.3 与 parkour 的对应（已读源码核对）

parkour 的 `_reward_sync_all_legs_cond`（`legged_robot_field.py:602`，注释即 "A hack to force same
actuation on both front/rear legs when jump"）比较 `actions[:, 0:3]+[6:9]`（右前+右后）与
`actions[:, 3:6]+[9:12]`（左前+左后），并对两处 **shoulder 取负**，且只在 `engaging == "jump"` 时生效。
⇒ **同一机制**（"跃起障碍时强制左右对称＝bound"），差别是他们的 URDF 左右轴约定相反所以要翻肩符号，
我们四腿轴相同故不翻。

#### 29.10.4 ⚠️ 语义边界（必须知道）

本项只做**对内相等**，不做**对间反相**：

* 与 **bound 一致**（前对同相 + 后对同相）；
* 但**同样与 pronk 一致**（四足全同相时两对当然各自相等）；
* front/rear 的**反相**（bound 相对 pronk 的判别特征）**没有任何项在管**——去掉 `feet_gait` 后整个配方
  都没有相位项。所以本项的作用是"**把沟壑上的错误先验（对角同相）换掉、并给出 bound 的可达方向**"，
  不是"保证学到 bound"。回放判据仍以足端时序为准。

#### 29.10.5 验证

* **新增离线测试** `tests/test_masked_joint_mirror.py`：10 项通过。做法是从**真实 `rewards.py`** 里用 AST
  抽出 `_pairwise_joint_mirror` 与 `MaskedJointMirror` 的源码、在带桩的命名空间里 `exec`（整模块因
  `omni.log` 无法 import），再按地形构造 trot／bound 姿态判定"到底用了哪套对子"。覆盖：trot 地形用对角对、
  **沟壑换成左右对**、`boxes` 完全豁免、空 `bound_mirror_joints` 退化为豁免、直立门控归零、
  **两套对子不共享缓存**；另有 4 项 AST 断言锁住配置里的三套名单/对子（含"必须含 hip"）。
  `bound` 分支被删掉时 `test_gap_switches_to_bound_pairs` 会以 `0.0 != 2.36` 失败（**负向对照已做**）。
* `py_compile` 两个改动文件、`git diff --check` 通过。
* **未运行（需训练机）**：真实地形名解析（`is_env_assigned_to_terrain` 对 `gap`/`boxes` 的命中）、
  分项量级、以及**步态本身是否真的变成 bound**。

**想连 `boxes` 也换成 bound**：把它从 `free_terrain_names` 挪走即可（`free=()`、`bound=("boxes","gap")`），
一行改动。

### 29.11 训练拉长到 60000 轮 + 地形列重分配（`boxes` 只留 1 列）（2026-09-24 用户要求）

用户：「我希望训练拉长一点。另外 boxes，只给一个」。

#### 29.11.1 训练长度

| | 旧 | 新 |
|---|---|---|
| `Imgo2CMoERoughRunnerCfg.max_iterations` | `2000`（基类遗留的占位值，实际每次靠 CLI 覆盖；上一轮用 **40000**） | **60000** |

写进配置而不是继续靠 CLI，是为了让**配方自己**记住训练长度。时间估算（上一轮 4096 环境实测
**3.60–3.83 s/轮**）：60000 × 3.7 s ≈ **62 小时 ≈ 2.6 天**。
`save_interval=500` ⇒ 意外关机最多丢 ~30 分钟；`--resume` 可接着跑，且**此时
`--max_iterations=N` 表示"再跑 N 轮"而不是"跑到 N 轮"**（`tot_iter = 当前 + N`）。仍可用
`--max_iterations=N` 临时覆盖。

#### 29.11.2 地形列重分配

`boxes` 的 proportion **0.15 → 0.05**（`num_cols=20` 时 **3 列 → 1 列**），空出的 **0.10 全部给 `gap`**
（**0.20 → 0.30**，`num_cols=20` 时 **4 列 → 6 列**）。理由：沟壑是历史上唯一持续降级、也最吃课时的一列。

**其余列数量不变**（离线核对，`scripts/tools/check_terrain_columns.py`）：

| 地形 | 训练 `num_cols=20` | play `num_cols=10` |
|---|---:|---:|
| `pyramid_stairs` | 4 | 2 |
| `pyramid_stairs_inv` | 2 | 1 |
| **`boxes`** | **1** | **1** |
| `random_rough` | 2 | 1 |
| `hf_pyramid_slope` | 2 | 1 |
| `hf_pyramid_slope_inv` | 1 | **0**（本来如此，未被掩码引用） |
| **`gap`** | **6** | **3** |
| `flat` | 2 | 1 |
| 合计 | 20 | 10 |

⚠️ **列分配不是"比例 × 列数"四舍五入**：Isaac Lab 用
`argmin_k(index/num_cols + 0.001 < cumsum(proportion_normalized)[k])` 逐列切
（`terrain_generator.py:240`）。所以改一个 proportion 可能"看着变了、列数没变"，而**某个地形拿到 0 列时
不会有任何报错**，引用它名字的奖励掩码（`free_terrain_names` / `bound_terrain_names` /
`no_trot_terrain_names`）会**静默退化成"全都不命中"**。新增的
`scripts/tools/check_terrain_columns.py` 就是为这类"改了但没生效"准备的：它读已安装 Isaac Lab 的
`ROUGH_TERRAINS_CFG` 取基类顺序/比例、从 `CMoE_env_cfg.py` 解析任务侧覆盖（含新增键），
逐字复现切列规则，并检查**每个被掩码引用的地形名在训练/play 两套列数下都 ≥1 列**。

`boxes` 仍 >0 列（训练与 play 都是 1 列）⇒ `free_terrain_names=("boxes",)` 在两边都还有命中对象，
不会静默失效。

#### 29.11.3 验证

* `scripts/tools/check_terrain_columns.py` 实跑：训练 `boxes=1 / gap=6 / 其余不变 / 合计 20`；
  play `boxes=1 / gap=3`；**"全部掩码引用的地形名都有 ≥1 列 ✅"**（退出码 0）。
* 新增 `tests/test_check_terrain_columns.py` **5 项通过**：总数 20、`boxes` 恰 1 列（训练+play）、
  空出的份额确实落到 `gap`（6 / 3）、其余六列数量不变、所有掩码引用的名字都有列。
* `py_compile`、`git diff --check` 通过。**未运行**：真实地形生成（列数与名字映射由 Isaac Sim 在
  env 构造时决定）仍待训练机确认。

### 29.12 诊断：「地形等级都上来了」＝**课程已饱和**，不是"爬到了 9 级"

用户观察「好像地形等级都上来了」。核对运行 `2026-09-24_15-34-14_cmoe_A_fresh`（**187 维 A 配方**，
1072 轮）——观察属实，但含义要说清。

#### 29.12.1 实测（1072 轮）

| 指标 | 值 |
|---|---|
| `level_mean` | **5.9399** |
| 逐列等级 | 上行楼梯 **6.106**、反向楼梯 6.064、**沟壑 6.105**、`boxes` 5.991、反斜坡 5.886、正斜坡 5.827、`random_rough` 5.633 |
| `level_min` / `level_max` | 0 / 9 |
| `move_up_frac` / `move_down_frac` / `frozen_frac` | **0.9235** / 0.0539 / 0.0226 |
| `distance_mean` | **9.7031 m** |
| `command_norm_mean` | 0.6118 m/s |

沟壑列的轨迹：**0.613 @100 轮 → 3.690 @300 → 7.329 @500 → 5.614 @700 → 6.105 @1072**
（先冲高再回落到饱和值）——**历史上唯一持续降级的那一列已经补上**。

#### 29.12.2 为什么 5.94 就是"饱和"：理论上限恰好是 **6.000**

判据（`terrain_levels_vel_logged`，与上游逐字一致）：
* 晋级：`distance > size[0]/2` = **4.0 m**（tile 长 8 m）；
* 降级：`distance < ‖cmd_xy‖·T·0.5` = `10‖cmd‖` ≈ **6.1 m**（本题 `move_down *= ~move_up`，晋级优先）；
* 到顶：`terrain.update_env_origins` 里 `level >= max_terrain_level` 时用
  **`randint(0, max_terrain_level)`＝均匀 0–9 重开**（`terrain_importer.py:317`）。

按这套规则解等级马尔可夫链的稳态（`max_terrain_level = num_rows = 10`）：

| 情形 | 理论稳态 `level_mean` |
|---|---:|
| **全晋级（p_up=1）** | **6.000** |
| 实测率（up .9235 / down .0539 / frozen .0226） | **5.946** |
| 旧 16.7k 障碍版（up .83 / down .143 / frozen .03） | 5.824 |
| 半晋级（up .5 / down .5） | 3.429 |

⇒ 观测 **5.9399** 与"实测率对应的稳态 5.946"吻合到小数第三位、与"全晋级上限 6.000"只差 0.06。
**所以 `level_mean ≈ 6` 正是"每回合几乎都晋级"的稳态签名，而不是"难度爬满了"。**
饱和时的等级分布（理论）：`0.019 / 0.038 / 0.056 / 0.074 / 0.092 / 0.110 / 0.129 / 0.147 / 0.164 / 0.171`
⇒ **只有约 17% 的样本落在 9 级**、约 2% 在 0 级，其余散在中间——这正是"到顶就随机重开"造成的。

**为什么必然饱和**：晋级阈值 4.0 m 是固定的，而策略实测走 **9.70 m**（＝阈值的 **2.4 倍**）；
降级阈值 `10‖cmd‖` 在 ‖cmd‖≈0.61 时是 6.1 m，也被轻松跨过。⇒ 课程已经**不再有区分度**
（晋级 92% vs 降级 5%），它现在的作用只是"把环境随机撒在 0–9 级之间"。

#### 29.12.3 同一 run 的两个红旗（与课程无关，但需要盯）

* **`Policy/mean_noise_std` 1.16 → 1.42 → 1.53 → 1.56 → 1.84 → 1.94，一路在涨**（对比：旧 PPO
  run 是降到 0.50）。探索标准差持续上升通常意味着优势信号弱/价值函数跟不上；A 配方关掉了
  `action_rate`/`ang_vel_xy`/`lin_vel_z`，本轮 `a30b25d` 已把前两项按 PPO 原值恢复（`lin_vel_z` 仍为 0），
  **新 run 要重点看这条是否还涨**。
* **`Train/mean_reward` 从 ~300 轮起就基本走平**（57.5 → 58.9，中间峰值 61.2），
  `ep_len` 935/1000（`time_out` 88.4%、`illegal_contact` 11.6%）、`track_lin_vel` 核 0.70、
  `track_ang_vel` 核 0.43。⇒ 饱和之后**没有继续变好**，说明瓶颈已不在"地形难度"。

#### 29.12.4 想让课程重新有信号，四个选项（未实施，待用户决定）

1. **提高晋级阈值**（在我们自己的 `terrain_levels_vel_logged` 里改，属**有意偏离上游**）：例如
   `distance > size[0]`（8 m）或 `> 1.5·size[0]`，或像降级那样做成**随命令缩放**
   （`distance > ‖cmd‖·T·0.8`）。现在 4 m 对 20 s 回合太松。
2. **改"到顶重开"规则**：把均匀 0–9 改成"只在高段重开"（如 7–9）或干脆停在最高级
   ⇒ 直接提高**9 级样本占比**（现在只有 17%）。这需要不再调用上游 `terrain.update_env_origins`。
3. **拉宽难度区间**：让 9 级真的超出当前能力（沟 >1.0 体长、台阶 >0.15 m…）——但从实测看
   它现在连 9 级都能过，得一次加得足够多。
4. **接受饱和**：地形不再是瓶颈，收益只能从奖励/鲁棒性侧拿（对应 §29.12.3 的两条）。

#### 29.12.5 适用边界

以上全部是**187 维、A 配方（parkour 权重）**的结论。刚提交的 **77 维 + PPO 三项步态 shaping +
地形掩码**（`a30b25d`/`c413eef`）**尚未跑过**，其课程饱和点与列间差异需要在新 run 上重新观察。

### 29.13 在跑的 run `2026-09-24_17-25-50_cmoe_B_77dim_ppogait` 用的是哪一版配方（逐项核对）

用户 2026-09-24 **17:25:50** 起的 run 落在**仓库根**的 `logs/`（不是 `imgo2_rl/logs/`，取决于启动时的
CWD）——以后找 run 要两个根都看。逐项核对它自带的 `params/{env,agent}.yaml`：

**在（✅）**：
* **77 维契约**（三重证据）：`terrain_estimator.encoder=(128,**77**)`、`expert actor=(512,**157**)`、
  `critic=(512,**125**)`、`terrain_projector=(128,93)`、`gating_network=(128,157)`；
  `height_scanner.offset.pos=(0.25,0,20)`（前移生效）；checkpoint 体积 38.3 MB(187) → 30.9 MB(77)。
* 三项步态 shaping＋掩码：`joint_mirror` = **`MaskedJointMirror` −1.0**、`mirror_joints` 为对角对**含 hip**；
  `feet_air_time` = **`MaskedFeetAirTime` +1.0 @0.5**；`feet_height_body` = `MaskedFeetHeightBody` −5.0；
  三者 `free_terrain_names=(boxes, gap)`。**`feet_gait: null`、`feet_air_time_variance: null`** ✅。
* 新地形**高度**：台阶 6 级 ×(0.05,0.15)、`boxes` (0.08,0.30)、沟 (0.126,0.315)。

**不在（❌，都是 17:25 之后才提的需求）**：
| 缺口 | run 里的实际值 | 应有值 |
|---|---|---|
| 沟壑换 bound 的 mirror | 两态版：`free=(boxes,gap)`、**无 `bound_terrain_names`** | 三态（沟壑＝左右对） |
| `boxes` 列数 | `proportion 0.15` ⇒ **3 列** | 0.05 ⇒ 1 列 |
| `gap` 列数 | `proportion 0.20` ⇒ **4 列** | 0.30 ⇒ 6 列 |
| 训练长度 | `agent.yaml: max_iterations **2000**` | 60000 |

⇒ 这个 run **会跑到 2000 就停**（≈19:45，4.1 s/轮 ⇒ 全程 ≈2.3 h）。

### 29.14 该 run 的实测（1477 轮）——新步态栈的第一批真实数据

| 项 | 值 | 解读 |
|---|---|---|
| `mean_reward` | **77.2** | 与 A_fresh（58.9）不同权重、不可直接比 |
| `mean_episode_length` | **992.7/1000** | 比 A_fresh 的 935 好 |
| `Episode_Termination/illegal_contact` | **0.0356** | **比 A_fresh 的 0.116 好 3.3 倍**（摔得少了） |
| `Policy/mean_noise_std` | **0.654** | ⭐ A_fresh 是 1.94 **且一路在涨**；恢复 `action_rate/ang_vel_xy` 后这条红旗消失了 |
| `gate_entropy` | 1.443 | 均匀为 ln5=1.609 ⇒ 专家未塌缩（A_fresh 1.306） |
| `Loss/learning_rate` | **1e-05** | 又贴到 adaptive 下限（§23 的老问题），Kld 12.05 |
| `feet_air_time` | **−0.1014**/s | 掩码版（7/20 列豁免）。展开 `Σair−0.5·N落地` ⇒ 负值意味着步态周期**短于约 1 s**；量级远小于 A_fresh 的 −0.113（同阈值但全局）⇒ 障碍列（更长的滞空）被掩掉后剩下的更短 |
| `feet_height_body` | **−0.01502**/s | ÷5 ⇒ 均方误 3.0e-3 ⇒ 抬脚高度 RMS 偏差 **≈5.5 cm**（目标离地 0.10 m）⇒ 有实际塑形压力但**只占 `track_lin`(4.4) 的 0.3%**，不会挡住爬梯 |
| `joint_mirror` | **−0.0644**/s | 对角腿未完全同步（非平凡满足），量级温和 |
| 课程 | `level_mean 5.454`↑、`move_up 0.884`、`distance_mean 8.65 m`；**`level_boxes` 4.31（最低）**、**`level_hf_pyramid_slope_inv` 1.72（异常低）**、`level_gap 6.00` | `boxes` 现在最高 30 cm ⇒ 最难；**反向斜坡列只有 1.7 级**需要单独查（掉进坑里？`distance` 判据不适用？） |
| `Perf/total_fps` | 2.16e4 | 比旧 run 的 2.6e4 略低（奖励项更多＋掩码项） |

### 29.15 「学会绕开障碍」的机制（用户回放 `model_1000` 观察）

用户回放 `2026-09-24_17-25-50_cmoe_B_77dim_ppogait/model_1000.pt` 后反馈「似乎都学会绕开了」。
逐行核对代码＋日志后确认：**这个行为是被允许的，而且课程在奖励它**。四个环节：

**① 障碍在**自己 tile 内**是全宽的 ⇒ tile 内绕不开。**
`cmoe_terrains.py::_platform(x0, x1, width, ...)` 的 `width` 全部传 `cfg.size[1]`（＝4.0 m），
矩形中心 `y = 0.5*width` ⇒ 台阶/独立块/沟在 y∈[0,4] 上完全横贯。

**② 但地形网格是「行＝难度沿 +x、列＝地形类型沿 +y」。**
`terrain_generator.py::_add_sub_terrain`：`transform[0:2,-1] = (row+0.5)*size[0], (col+0.5)*size[1]`
⇒ **横向移 4 m 就进入另一种地形**。而本项目 8 类里有 **5 类根本不是"赛道"**：
`random_rough`／`hf_pyramid_slope`／`hf_pyramid_slope_inv`／`flat` 没有横贯障碍
（play 的 10 列布局：col0-1 楼梯、col2 反楼梯、col3 `boxes`、col4 粗糙、col5 斜坡、col6-8 沟、col9 平地）
⇒ **横移到这些列就能直着走**，"绕开"成立。

**③ 越界终止的量程极大，横穿好几列毫无代价。**
`terrain_out_of_bounds` 用**整张图**的尺寸：`map_width = n_rows*grid_width + 2*border = 10*8+40 = 120 m`、
`map_height = n_cols*grid_length + 2*border = 10*4+40 = 80 m`（训练 20 列 ⇒ 120 m），
`distance_buffer = 3 m` ⇒ 离地图中心 **x 57 m / y 37 m** 以内都合法 ⇒ 横穿 9 列（36 m）都不触发。
日志佐证：`Episode_Termination/terrain_out_of_bounds` **全程 0**。

**④ 课程判据直接奖励它。**
晋级判据是 `distance = ‖root_xy − origin_xy‖ > size[0]/2 = 4 m`（**欧氏距离，含横向分量**）
⇒ 横着走同样算"通过" ⇒ `distance_mean 8.65 m`、`move_up_frac 0.884`、课程饱和（§29.12 同一根因）。

**⑤ 命令侧拦不住（这是最隐蔽的一环）。**
障碍列上 `UniformThresholdVelocityCommand` 强制 `vy = 0` 且 `heading_target = 0`，**但**：
* yaw 指令是 heading 控制器**按当前误差实时生成**的（`ang_vel_z = k·(0 − yaw)`）⇒ **转身本身不怎么被罚**
  （只要它在"往回转"，指令就跟着它走）；
* 速度奖励用的是**机体系**速度 ⇒ 一边转身一边在机体系里前进，vx 跟踪照样拿高分。
日志佐证（1477 轮）：线速度跟踪核 **0.88**（好），**偏航跟踪核只有 0.46**（等价偏差 ≈0.44 rad/s ≈25°/s）
⇒ **它确实没有在朝 +x 走**。

**⑥ 附带发现：掩码按"出生列"静态判定。**
`_terrain_type_mask` 用 `terrain.terrain_types`（**出生时**登记的列，`terrain_importer.py`）
⇒ 机器人漂到别的列后，`forward_only` 命令与 trot/bound 掩码**都不会更新**（它"以为"自己还在楼梯列）。

#### 候选修法（未实施）

| | 做法 | 代价／影响 |
|---|---|---|
| **F1** | 越出本列 ⇒ **终止**（`\|root_y − origin_y\| > size[1]/2`）或按越界距离罚 | 只加一个 term；**逻辑栏杆**，不动地形几何；顺带让"按出生列的掩码"重新变得自洽 |
| **F2** | 晋级判据改为**沿 +x 的前向进度**（`root_x − origin_x > size[0]/2`），降级同理 | 改我们自己的 `terrain_levels_vel_logged`；顺带修 §29.12 的课程饱和 |
| **F3** | **物理封边**：`_platform` 的 width 4.0→~2.0 并在两侧加高墙（或所有列都改成带墙赛道） | 根治，但要重新生成地形、且 `rough/slope/flat` 也得改造；会改变已有 checkpoint 的可比性 |
| **F4** | 命令/奖励侧：障碍列 `heading_command=False`+`ang_vel_z=0`、`heading_control_stiffness` 0.5→1.5、或加"偏离中心线"罚 | 改命令语义，可能让正常行走变僵 |
| **F5** | **加日志**：`\|y−origin_y\|` 均值/最大值、出列环境比例 | 极便宜，且**不用回放也能从 TB 判断是否在绕**（本机无 GPU，这条价值最高） |

建议 **F1 + F2 + F5** 一起做（都在我们自己的代码里、可离线审），把 F3 留作根治备选。

### 29.16 用户回放确认：**横移绕开** ＋ **全地形仍是 bound**（2026-09-24）

用户回放 `cmoe_B_77dim_ppogait/model_1000.pt` 的结论：「**横移过去了**，在大部分地形；但**平地与粗糙地上
确实在走**（正常前进）；另外**都还是 bound 步态**」。

**① 「横移」证实了 §29.15 的机制**：绕开＝**侧向平移**进相邻列（`flat/rough/slope` 那几列没有横贯障碍），
而平地/粗糙列本来就没障碍所以看着是在正常走 —— 与"横移 4 m 换一列"的几何完全吻合。
⇒ 根因仍是④（课程用**含横向分量的欧氏距离**判通过）＋⑤（速度奖励在**机体系**、yaw 指令由 heading
控制器实时生成 ⇒ 转身/侧移罚得不够）＋③（越界终止量程 120 m×80 m，横穿几列无代价）。

**② bound 步态不是"绕开"的副产物**：在**平地与粗糙列**（原本没有障碍、也在正常前进）上依然是 bound
⇒ 步态问题**独立存在**，必须单独修。

**③ 为什么我们这套三项 shaping 拦不住 bound**（逐项分析）：

| 项 | 对 trot / bound 的作用 | 问题 |
|---|---|---|
| `joint_mirror −1.0` | 只做"**对角对内相等**" | 量级太小：实测 **−0.064/s**，而 `track_lin_vel` 是 **4.4/s** ⇒ 只占 **1.5%**，属"温和偏好" |
| `feet_air_time +1.0 @0.5` | 展开＝`4(1−d) − 2·N落地/T`；两种步态**每周期都是 4 次落地**，所以差别只在 `4(1−d)`＝"同时腾空的足数×时间" | **奖励"更多地同时腾空"⇒ 方向上偏袒 bound/腾跃式**（trot 典型 d=0.5、T=0.6 ⇒ −1.33/s；bound d=0.4、T=0.8 ⇒ −0.1/s） |
| `feet_height_body −5.0` | 只约束抬脚高度 | 对 trot/bound **中立** |
| **`feet_air_time_variance −8.0`** | 惩罚**四足之间**滞空/触地时长的方差 ⇒ **前/后对时序不一致（＝bound）会被罚** | **PPO 那套里量级最大的步态项，我们自 `6220e43` 清零后一直没恢复**（§29.8 已记）⇒ **这正是最该补的一项** |

另外：**35% 的列（`boxes` 3 + `gap` 4）完全豁免**步态塑形，那些列上学到的 bound 会通过共享主干／
专家路由渗到其它列。

**④ 从日志反解步频**：`feet_air_time` 实测 **−0.1014/s**，去掉 35% 豁免列的掩码因子（×1/0.65）≈ **−0.156/s**，
代入 `4(1−d) − 2/T`：d=0.5 ⇒ **T≈0.93 s**、d=0.45 ⇒ 0.85 s、d=0.6 ⇒ 1.14 s
⇒ **步频约 0.9–1.2 Hz（≈1 Hz）**，只有参考基线（0.600 s／1.67 Hz）的**一半左右**，
与"低步频、大腾空"的 bound 一致（trot 通常接近参考基线的 1.7 Hz）。

#### 待用户确认的一个判别性问题

看到的 bound 是哪种？
* **前两足一起 / 后两足一起（真 bound）** ⇒ 加 `feet_air_time_variance −8.0` 就能治（它专门罚前/后对的时序差）；
* **四足同时起落（pronk）** ⇒ `feet_air_time_variance` 与 `joint_mirror` **都满足**，
  必须用相位项 `feet_gait`（`mdp.TrotWithoutGapReward` 还留着，直接开权重即可）。

建议同时补 `eval_gait.py` 的 **CMoE 适配**，把 FL-FR 相位（≈180°＝trot、≈0°＝bound/pronk）、
步周期、集中度 R、抬脚高度**量出来**，不再靠肉眼判断。

#### 动作清单（建议先修再训，不要拿当前配方跑 60000 轮）

1. **F1** 越出本列即终止（`|root_y − origin_y| > size[1]/2`）＝逻辑栏杆 ⇒ 横移立刻变成"丢回合"；
2. **F2** 晋级判据改成**沿 +x 的前向进度**（`root_x − origin_x > size[0]/2`）⇒ 横移不再得分（顺带修 §29.12 饱和）；
3. **F5** 加日志：`|y−origin_y|` 均值/最大、**出列环境比例**（不用回放就能从 TB 判断是否在绕）；
4. **步态**：加回 `feet_air_time_variance **−8.0**`（做成掩码版，豁免 `boxes`/`gap`）；
   若确认是 pronk，再开 `feet_gait`（`TrotWithoutGapReward`，1.0）；
5. （可选）把豁免集从 `boxes+gap` 缩到只有 `gap`，让更多列受步态塑形。

### 29.17 按 parkour 修「横移绕开」：世界系速度 + 中心线/朝向罚 + **赛道侧墙**（2026-09-24）

用户看到 `/root/Desktop/parkour/` 后定调：「该参考 parkour 用全局的速度来约束了」。
把 parkour 那套**逐条读出来**（不是为了照抄参数，而是看它到底靠什么不让机器人绕开）：

| parkour | 位置 | 作用 |
|---|---|---|
| `tracking_world_vel = 5.` | `go1_leap_config.py:94` | **世界系**速度跟踪（`_reward_tracking_world_vel` 比 `commands[:, :2]` 与 `root_states[:, 7:9]`＝世界系速度）⇒ 目标方向**不随自身转动而变** |
| `lin_pos_y = -0.4` | `go1_leap_config.py:99` | `_reward_lin_pos_y = \|root_y − env_origin_y\|` ⇒ **离赛道中心线的横向距离**（L1） |
| `yaw_abs = -0.2` | `go1_leap_config.py:98` | `_reward_yaw_abs = \|yaw\|`（目标朝向 +x） |
| **`track_width=1.6`、`wall_thickness=0.04`、`wall_height=0.5`** | `utils/terrain/barrier_track.py:21-24`（源码注释画出 `\|xxxx\|track wall`） | **窄赛道 + 两侧物理墙** ⇒ 物理上绕不开 |
| `tracking_ang_vel = 0.05` | 同上 | 比我们的 0.6 小 12 倍（他们 yaw 命令近乎为零） |

**关键认识**：parkour 的"不绕开"**首先靠几何**（窄道 + 墙），速度/位置/朝向罚只是软约束。
我们的地形恰好相反：障碍在 tile 内是全宽的，但**列与列之间完全开放** ⇒ 横移 4 m 就换一种地形。

#### 已落地（三项）

1. **世界系速度跟踪**：新增 `mdp/rewards.py::track_world_vel_xy_exp`，`CMoE_env_cfg.py` 里
   `track_world_vel_xy_exp = 5.0`、**`track_lin_vel_xy_exp = 0.0`**（被取代）。
   目标速度由命令的**目标朝向**旋转到世界系：`target_w = R_z(heading_target)·(vx_b, vy_b)`
   ——障碍列 `heading_target = 0` ⇒ 目标就是纯 `+x`；普通列跟随采样到的目标朝向。
   σ² 保持我们原来的 **0.25**（parkour 的 leap 用 0.35，偏软；留作单变量）。
2. **中心线 + 朝向罚**：新增 `lin_pos_y`（`|y − env_origin_y|`）与 `yaw_abs`（`|yaw|` 归一化到 (−π,π]），
   权重 **−0.4 / −0.2**（＝parkour），**只在 `forward_only_terrain_names` 那 4 个障碍列生效**
   （普通列本来就有 ±0.3 的 `vy` 指令与随机朝向目标，不该被罚）。中心线取 `env_origins`——
   `track_*` 的 `origin` 是 `(spawn_x, 0.5*size[1], ·)`，而 tile 也以 `0.5*size[1]` 为中心摆放，
   所以这个差正是"离赛道中线多远"。
3. **赛道侧墙**：`cmoe_terrains.py` 新增 `_side_walls()`，在 `track_gap`/`track_step`/`track_stairs`
   三条赛道的 `y≈0` 与 `y≈width` 各加一堵 `0.04 m × (最高平台 + 0.5) m` 的墙（参数照 parkour）。
   ⇒ 机器人**物理上无法离开本列**。连带三件事一起好了：①"绕开"不可能；②课程用的"离出生点欧氏
   距离"重新等价于"沿 +x 的前向进度"（§29.12/§29.15 的利用点失效）；③"按出生列"的奖励掩码与
   命令掩码重新自洽（不会再漂到别的列去）。

**生效奖励 16 → 18 项**（新增 3、移除 1）：

```
track_world_vel_xy_exp +5.0   ← 新增（取代机体系版本）
lin_pos_y              −0.4   ← 新增（仅障碍列）
yaw_abs                −0.2   ← 新增（仅障碍列）
track_lin_vel_xy_exp    0     ← 移除
```

#### 验证（离线，全部可复现）

| 测试 | 内容 |
|---|---|
| `tests/test_world_vel_tracking.py`（**8 项**） | 目标速度按 `heading_target` 旋转（0°→(v,0)、90°→(0,v)）；**核心回归**：障碍列上机器人转身 90°、机体系"前进"、世界系里横移 ⇒ 世界系项给 **0.056**（机体系版本会给 **1.0**，这正是漏洞来源）；倒走 ≈0.003；直立门控归零；另有 3 项 AST 断言锁配置（世界系已启用、机体系已归零、两项软约束用的是 `forward_only_terrain_names`）。写法同 `test_masked_joint_mirror.py`：AST 抽真实源码 + 桩 exec（整模块因 `omni.log` 无法 import）。 |
| `tests/test_track_walls.py`（**6 项**） | 三段赛道在 d=0/0.5/1 各**恰好 2 堵墙**；墙贴 `y=0` 与 `y=width−0.04` 且覆盖整条 x∈[0,8]；墙顶高出最高平台 ≥0.5 m；非墙网格数量不变（gap＝`num_gaps+1`、step＝`1+num_steps`、stairs＝`1+num_steps+1`）；出生点 (0.75, 2.0) 与出生平面高度不变。用 `sys.modules` 塞桩模块后 exec 真实源码（`trimesh` 是真的）⇒ 纯几何离线可算。 |
| **测试抓到的真 bug** | 下行楼梯（`ascending=False`）在**起点**就处于 `total_height`，我最初只给上行梯按 `total_height` 加高墙 ⇒ 下行梯那段高台把墙埋掉。已修（两种排布都按 `total_height`）。 |
| **负向对照** | 把 `track_gap` 的墙去掉 ⇒ `test_two_walls_per_track` / `..._both_y_edges...` 立刻失败；还原后通过。 |
| 合计 | 相关离线测试 **37 项全通过**（含原有的 mirror 10、reward-overrides 7、terrain-columns 5），`py_compile`、`check_terrain_columns.py`（新名单也纳入校验）、`git diff --check` 通过。 |

#### 未做 / 待定

* **`feet_air_time_variance`（−8.0）仍未恢复** —— 它是针对 **bound**（前/后对时序不对称）的那一项，
  但要不要加取决于用户确认"看到的是 bound 还是 pronk"（§29.16）。pronk 的话必须改开相位项 `feet_gait`。
* parkour leap 配方里我们还缺：`penetrate_depth/volume`（穿模罚）、`hip_pos −5`、`dof_error −0.15`、
  `exceed_dof_pos_limits −0.8`、`delta_torques −1e-7`、`legs_energy_substeps`；以及他们的
  `tracking_ang_vel` 只有 0.05（我们 0.6）。这些没照搬，属**未评估**项。
* **侧墙改的是地形几何** ⇒ 2026-09-24 之前的所有 CMoE checkpoint（含 `cmoe_A_fresh`、16.7k 障碍版、
  `cmoe_B_77dim_ppogait`）与新地形**不可比**；必须从零重训。
* 侧墙只在机器人贴近边线时才会进入高度扫描视野（扫描足迹仅 1.0 m × 0.6 m，赛道宽 4 m），
  所以 77 维地形观测的**内容**变化极小；但仍是观测分布的一次变化。

### 29.18 用户改定：**不加侧墙**；**全场景只给"超前"速度**；脱离中心的惩罚**全局**生效（2026-09-24）

用户对 §29.17 的两条否决/收窄：

> 1. 侧墙不该有
> 2. 所有场景都变成只有超前的速度，都给脱离中心的惩罚

#### 29.18.1 改动

| 项 | §29.17（已作废） | 现在 |
|---|---|---|
| 赛道侧墙 | 三条 `track_*` 两侧加 `0.04×0.5 m` 墙 | **全部移除**（`git checkout` 还原 `cmoe_terrains.py`，与 `HEAD` 逐字一致；赛道保持**开放**） |
| 前向命令范围 | 只有 4 个障碍列（`forward_only_terrain_names`） | **全部 8 类地形**都进该名单 ⇒ 全场景统一"沿世界 +x 前进 0.3–1.0 m/s、朝向锁 0"，不再有全向命令的列 |
| `lin_pos_y` / `yaw_abs` | 只作用于 4 个障碍列 | `terrain_names = ()` ⇒ **全局**（"都给脱离中心的惩罚"） |

生效奖励仍是 **18 项**（`track_world_vel_xy_exp +5.0`、`lin_pos_y −0.4`、`yaw_abs −0.2`、`track_lin_vel_xy_exp 归零`）。
`yaw_abs` 一并全局化：既然全场景的 `heading_target` 都是 0，该项与命令一致（用户只点名了"脱离中心"，
如果不想全局罚朝向，把那一行的 `terrain_names=()` 换成障碍列名单即可）。

#### 29.18.2 现在的"不许绕开"靠什么

去掉了几何约束（墙）之后，只剩下**奖励端两层**：

1. **世界系速度跟踪**（`+5.0`）——目标方向是命令的目标朝向（全场景为 +x），**不随自身转动而变**，
   横移/掉头立刻体现为巨大误差；
2. **全局中心线罚**（`lin_pos_y = −0.4`，L1）——离赛道中线 4 m 就持续扣 **1.6/s**（整回合总奖励量级 ~4/s）
   ⇒ 长期"绕到邻列去跑"在收益上不可行。

**残留风险（必须知道）**：这两条都是**软约束**，不像墙那样物理禁止；且课程的晋级判据仍是
"离出生点的**欧氏**距离 > 4 m"（含横向分量）。所以理论上"一次性横移出去、再在邻列直着跑"仍有
局部收益（一次性付 ~1 s 的跟踪罚，换来一条无障碍的通道）。若回放仍见绕开，下一档最小改动是
**把晋级判据换成沿 +x 的前向进度**（§29.15 的 F2，只改我们自己的 `terrain_levels_vel_logged`）。

#### 29.18.3 验证

* 删掉 `tests/test_track_walls.py`（那是在测墙），换成 `tests/test_track_geometry.py` **5 项**：
  **断言赛道"没有沿 y 边缘的薄网格"**（＝没有侧墙，防止以后被顺手加回来）＋平台横贯整宽、
  平台数量、出生点、出生平面不变。
* `tests/test_world_vel_tracking.py` 更新为：两项软约束 `terrain_names=()` 全局 ＋
  **断言 `forward_only_terrain_names` 恰好覆盖 8 类地形全部**。
* `scripts/tools/check_terrain_columns.py` 新增**覆盖率校验**：`forward_only_terrain_names` 与
  `sub_terrains` 的键集合必须一致（漏一项那一列会**静默**退回全向命令）——实跑输出
  `✅ forward_only_terrain_names 覆盖全部 8 类地形`。
* 相关离线测试 **37 项全通过**；`py_compile`、`git diff --check` 通过。
* **未运行**（需训练机）：Isaac Lab 构造、全场景前向命令是否真的生效、以及绕开是否被抑制。

### 29.19 用户问"joint_mirror 是不是权重太低，平地也没看到 trot"——量级够了但**方向不对**

#### 29.19.1 权重确实低（用户直觉对）

在跑的 run（`cmoe_B_77dim_ppogait` @1477）实测：

| 项 | 值 | 占比 |
|---|---|---|
| `Episode_Reward/track_lin_vel_xy_exp` | 4.40/s | 100% |
| `Episode_Reward/joint_mirror` | **−0.0644/s** | **1.5%** |
| `Episode_Reward/feet_height_body` | −0.0150/s | 0.3% |

#### 29.19.2 但**加大它不是拿到 trot 的办法**（这是关键）

`joint_mirror` 的对角对比较的是"**对内相等**"：

* **trot**：`FL≈RR`、`FR≈RL` ⇒ 满足；
* **bound**（前对同相、后对同相）：对角腿**反相** ⇒ 违反（会被罚）；
* **pronk**（四足同相）：对角腿当然也相等 ⇒ **满足**。

⇒ 它**同时与 trot 和 pronk 相容**，加大权重只会更用力地压"对角相等"，而这一步 **pronk 满足得最彻底**
⇒ **有可能把步态推向 pronk，而不是 trot**。

**更值得注意的是实测数值本身**：`joint_mirror` 的时间平均只有 **0.0644**（＝weight −1 时的 term 值，
即两对之和的均值）。用参考步态的关节幅度估一下：若对角腿真的**反相半个周期**，thigh p2p≈0.45 rad、
shank p2p≈0.55 rad ⇒ `Σ(Δq²)` 的时间平均应在 **0.2–0.3** 量级，是实测值的 **3–5 倍**。
⇒ 现在这点残差说明**对角腿之间只错开一点点** —— 也就是：
**用户看到的"bound"更可能是"四足接近同相（pronk 倾向）"或"相位差很小的快 bound"**，而不是经典的前/后对分明的 bound。
这一点直接决定该加哪一项（见下）。

#### 29.19.3 根因：整个配方里**没有任何一项管"对角对之间的反相"**

| 项 | 管什么 | 能否区分 trot / bound / pronk |
|---|---|---|
| `joint_mirror −1.0` | 对角**对内相等** | trot ✓／bound ✗／**pronk ✓** |
| `feet_air_time +1.0 @0.5` | 展开 `4(1−d) − 2·N落地/T` ⇒ **奖励"同时腾空多"** | **偏袒 bound/pronk**（trot d=0.5/T=0.6 ⇒ −1.33/s；bound d=0.4/T=0.8 ⇒ −0.1/s） |
| `feet_height_body −5.0` | 抬脚高度 | 中立 |
| ~~四足时序均匀度~~ | —— | **缺**（`feet_air_time_variance` 从 `6220e43` 起一直没恢复） |

#### 29.19.4 已落地：把 `feet_air_time_variance −8.0` 加回来（掩码版）

* `mdp/rewards.py` 新增 `MaskedFeetAirTimeVariance`（复用 `_terrain_type_mask`，豁免 `boxes`/`gap`，
  与另两项一致）；`CMoE_env_cfg.py` 接线 **−8.0**（＝PPO 原值，那次 trot 配方里量级最大的步态项）。
* **生效奖励 18 → 19 项**。
* 它罚"四足之间滞空/触地时长的**方差**" ⇒ **bound（前/后对错开）会被罚**；
  ⚠️ 但 **pronk（四足完全同步）满足它** —— 所以它能不能治，取决于 §29.19.2 里那个待确认的形态。
* **`joint_mirror` 权重保持 −1.0 不动**（理由见 §29.19.2：加大它可能加深 pronk）；
  若确实想加强"稠密姿态先验"，建议最多到 **−2.0**，且不要指望它单独出 trot。

#### 29.19.5 下一步（按性价比）

1. **先客观量出相位，别再靠肉眼**：给 `scripts/tools/eval_gait.py` 加 **CMoE 适配**
   （它现在硬编码 `AmpVecEnvWrapper`/`AMPOnPolicyRunner`）。它直接给 **FL-FR 相位**（≈180°＝trot、
   ≈0°＝bound/pronk）、步周期、集中度 R、抬脚高度 —— 用现成的 `model_1000.pt` 就能测，不用重训。
2. **若确认是 pronk / 相位差很小** ⇒ 必须开相位项 `feet_gait`（`mdp.TrotWithoutGapReward`，权重 1.0）；
   `variance` 与 `mirror` 都对 pronk 无效。
3. **若确认是真 bound** ⇒ 先看这一轮 `feet_air_time_variance` 是否把它压下去（它的 `Episode_Reward` 分项
   本身就是"bound 程度"的度量）。
4. **可选的干净对照**：把 `sub_terrains` 的比例临时改成"`flat` = 1.0、其余 = 0"（一行），
   在**纯平地**上训练若干千轮 —— 平地最容易出 trot，能一眼分清"配方本身缺相位约束"还是
   "混合地形把步态带偏了"。注意这会把 `level_*` 课程变成无意义（但 `forward_only` 与掩码都还有效）。

### 29.20 地形晋级判据改造：**前向进度 + 速度跟踪门控**（2026-09-24 用户要求）

用户：「地形等级提升还是需要考虑速度跟踪效果」。据此 `mdp/curriculums.py::terrain_levels_vel_logged`
相对上游 `terrain_levels_vel` 有**两处有意偏离**（原实现是"判据逐字一致、只加日志"）。

#### 29.20.1 新判据

```python
progress = root_pos_w[:, 0] − env_origins[:, 0]            # ① 沿 +x 的前向进度（不含横向）
track_avg = _episode_tracking_average(env, env_ids, "track_world_vel_xy_exp")   # ② 本回合平均跟踪核

move_up   = (progress > size[0]/2)  &  (track_avg > 0.80)              # 4.0 m 且跟踪达标
move_down = (progress < ‖cmd_xy‖·T·0.5) | (track_avg < 0.35)           # 距离不够 或 跟踪太差
move_down *= ~move_up                                                  # 晋级优先
terrain.update_env_origins(env_ids, move_up, move_down)                # 等级增减与到顶重开不变
```

* **偏离 ①（前向进度）**：上游用 `‖Δxy‖`（欧氏距离）。实测漏洞是"**横移绕开**"照样算通过
  （§29.15/§29.16）。改用 `Δx` 后，**"绕开"从判据里彻底消失** —— 这比 §29.15 里提的"越界即终止(F1)"
  更轻，也不需要侧墙（用户已否决侧墙）。
* **偏离 ②（跟踪门控）**：只看走了多远会奖励"慢慢蹭过去"。现在要求本回合
  **平均速度跟踪核 > 0.80**（取不到分项时自动退回纯距离判据，并记 `tracking_is_used=0`）。
  阈值依据：A 配方 run 的核 ≈ **0.70**、当前 run ≈ **0.88** ⇒ 0.80 是一条"确实要跟速"的实线；
  0.35 以下视为明显失败。

#### 29.20.2 跟踪均值怎么来的（时机已核对，不是想当然）

读奖励管理器为每个分项累计的**回合和**（`RewardManager.compute` 每步 `_episode_sums[name] += term·weight·dt`）：

```
track_avg = _episode_sums[name][env_ids] / weight / (episode_length_buf[env_ids] · step_dt)
```

**能这么读的关键是顺序**（`isaaclab/envs/manager_based_rl_env.py::_reset_idx`）：
`curriculum_manager.compute()` 在**最前面**（:358）→ … → `reward_manager.reset()`（读走并清零
`_episode_sums`，~:377）→ … → `episode_length_buf[env_ids] = 0` 在**最后**（:396）
⇒ 课程判定时拿到的正是**刚结束那一回合**的和与回合长度。权重从 `get_term_cfg(name).weight` 取，
不写死数字。取不到（改名/被移除）⇒ 退回纯距离判据。

#### 29.20.3 新增日志（能在 TB 里看出门控是否在起作用）

`Episode/Curriculum/terrain_levels/` 下新增：`progress_mean`、**`tracking_mean`**、`tracking_min`、
**`tracking_pass_frac`**（>0.80 的比例）、**`tracking_fail_frac`**（<0.35 的比例）、`tracking_is_used`；
原有 `level_*`、`move_up/down/frozen_frac`、`distance_mean`、`command_norm_mean` 保留。

#### 29.20.4 预期效果（训练机验证）

* `move_up_frac` 会从 **0.884 明显下降**（跟踪门控 + 前向进度同时收紧）⇒ 课程**不再饱和**
  （§29.12 的 `level_mean≈6` 稳态会被打破）；
* 硬列（`boxes` 现在最高 30 cm）会稳定在更低的等级——这正是"在能跟速的前提下"的真实水平；
* "横移绕开"不再得分（晋级看 `Δx`）。

#### 29.20.5 验证

* 新增 `tests/test_terrain_curriculum.py` **4 项**（AST/桩 exec 真实源码；含函数体内相对导入的处理）：
  - 五个环境的判据矩阵（进度够+跟踪好⇒晋级；进度够+跟踪中等⇒冻结；跟踪差⇒降级；没走够⇒降级；
    **横向走 3 m（欧氏 4.24 > 4）但前向只 3 m ⇒ 不许晋级**——这条就是"绕开"的回归）；
  - `tracking_is_used=1/0` 与 `tracking_pass_frac=3/5`、`tracking_fail_frac=1/5`；
  - **取不到跟踪分项时退回纯距离判据**（该矩阵里 env1/env2 变成晋级，且不输出 `tracking_mean`）；
  - 晋级优先（`move_down *= ~move_up`）。
  **负向对照**：把跟踪门控那段去掉 ⇒ `test_progress_and_tracking_both_required` 立刻失败；还原后通过。
* 相关离线测试 **41 项全通过**；`py_compile`、`git diff --check` 通过。
* **未运行**（需训练机）：真实 `_episode_sums` 读取（时机由源码顺序保证，但没有真跑过）、
  阈值 0.80/0.35 是否合适、以及 `move_up_frac` 的实际下降幅度。

### 29.21 开 `feet_gait`（相位项），掩码与其余四项一致（2026-09-24 用户决定）

用户：「那就开 feet gait，同样加掩码」。

#### 29.21.1 改动

* `mdp/rewards.py::TrotWithoutGapReward` **改成纯地形掩码版**：掩码就是
  `_terrain_type_mask(free_terrain_names)`，与 `MaskedJointMirror`／`MaskedFeetHeightBody`／
  `MaskedFeetAirTime`／`MaskedFeetAirTimeVariance` **同一套**（默认豁免 `("boxes", "gap")`）。
  旧的复合掩码（地形类型 ＋ `height_scanner` 最前几列任一射线落空 ＋ 任一脚端扫描器整束落空）
  **已移除**（要回退看提交 `a30b25d`）。
* `CMoE_env_cfg.py`：`feet_gait.func = mdp.TrotWithoutGapReward`、`weight = 1.0`、
  `synced_feet_pair_names = (("FL_FOOT","RR_FOOT"), ("FR_FOOT","RL_FOOT"))`（trot 对角对，
  **必须给全**，空字符串会在 `GaitReward.__init__` 直接抛 ValueError）、
  `free_terrain_names = ("boxes", "gap")`。
* **生效奖励 19 → 20 项**。

#### 29.21.2 为什么它才是"能不能出 trot"的决定性一项

五个步态相关项（**全部**豁免 `boxes`/`gap` 共 7/20 列）：

| 项 | 权重 | trot | bound | pace | **pronk** |
|---|---:|---|---|---|---|
| `joint_mirror`（对角对内相等） | −1.0 | ✓ | ✗ | ✗ | **✓** |
| `feet_air_time`（滞空越长越好） | +1.0 | ✓ | ✓（还偏袒） | ~ | ✓（还偏袒） |
| `feet_height_body`（抬脚高度） | −5.0 | 中立 | 中立 | 中立 | 中立 |
| `feet_air_time_variance`（四足时长方差） | −8.0 | ✓ | ✗ | ✗ | **✓** |
| **`feet_gait`（对角对反相）** | **+1.0** | **✓** | ✗ | ✗ | **✗** |

⇒ **只有 `feet_gait` 的 4 个 async 核显式要求"对角对之间反相"**，因而能排除 pronk。
这也是 §29.19.2 里那个"`joint_mirror` 加大权重可能反而推向 pronk"的结论的直接对策。

**代价（要盯）**：6 核相乘是**很窄的脊** —— 随机策略早期诸核都≈0 ⇒ 乘积≈0 ⇒ **梯度≈0**，学得慢。
所以它与 `joint_mirror`（稠密二次先验、第一步就有梯度）**并用**才互补：mirror 提供早期方向，
`feet_gait` 提供最终相位约束。日志里应看到 `Episode_Reward/feet_gait` 从 0 慢慢爬升。

#### 29.21.3 验证

* 新增 `tests/test_feet_gait_mask.py` **5 项**：豁免列（`gap`/`boxes`）严格为 0、非豁免列保留基类值
  （桩基类返回 1.0 ⇒ 证明是**乘法掩码**不是替换）、不传 `free_terrain_names` 时默认 `("boxes","gap")`
  ⇒ 输出 `[1,0,0,1]`、以及 2 项配置断言（func/weight/free 名单、对角对给全＝`FL_FOOT`/`RR_FOOT`/
  `FR_FOOT`/`RL_FOOT`）。写法：AST 抽真实类源码 + 桩 `GaitReward`/`_terrain_type_mask` exec。
* `tests/test_check_reward_overrides.py` 更新：生效 **20 项**、`feet_gait` 从"必须禁用"改为
  "已启用且挂 `TrotWithoutGapReward`"。
* `check_terrain_columns.py` 把 `feet_gait.free_terrain_names` 纳入"掩码引用的地形名必须 ≥1 列"校验。
* 相关离线测试 **46 项全通过**；`py_compile`、`git diff --check` 通过。
* **未运行**（需训练机）：`feet_gait` 的实际量级与爬升速度、是否真的把步态拉回 trot。
  建议配合 `eval_gait.py` 的 CMoE 适配（量 FL-FR 相位：≈180°＝trot、≈0°＝bound/pronk）来判读。

### 29.22 「play 里看到的地形好像和配置不一致」——是**起点等级与列分配**的差异，不是配置没生效

用户反馈：表格里的台阶/`boxes` 高度是 `(0.05,0.15)` / `(0.08,0.30)`，但刚 play 时看到的"好像不是"。

#### 29.22.1 配置确实生效（三重核对）

1. 工作区源码：`pyramid_stairs`/`pyramid_stairs_inv` = `step_height_range=(0.05,0.15)`、`num_steps=6`；
   `boxes` = `step_height_range=(0.08,0.30)`、`num_steps=4`、`proportion=0.05`。
2. **在跑的 run 的训练快照**（`logs/.../2026-09-24_17-25-50_cmoe_B_77dim_ppogait/params/env.yaml`）：
   `pyramid_stairs 0.05/0.15 num_steps 6`、`pyramid_stairs_inv 0.05/0.15 num_steps 6`、`boxes 0.08/0.3`。
3. **用真实地形函数离线算出每个等级的几何**（下表），与解析式逐字吻合。

#### 29.22.2 为什么 play 里看起来"矮"：只从 0–5 级起步

* `Imgo2CMoERoughPlayEnvCfg` 里 **`max_init_terrain_level = 5`** ⇒ 初始等级在 **0–5 里随机抽**；
  单环境一局内几乎不会晋级 ⇒ **看到的基本是 0–5 级**：台阶单级 **5.5–10.5 cm**、`boxes` **9.1–18.9 cm**。
  配置区间的**上端（15 cm / 30 cm）要 level≈9** 才出现 —— 训练也是同样从 0–5 起步、靠课程涨上去的。
* play 用 **`num_cols = 10`**（训练 20）⇒ 列分配不同，且 **`hf_pyramid_slope_inv` 拿到 0 列**
  （play 里根本没有这一类地形）；默认 `--num_envs=1` 只落在 **第 0 列＝上行楼梯**，
  5–10 cm 的台阶在侧视下确实像缓坡（这也是早先"没看见上行台阶"的同一原因）。
* `--num_envs=10` 时的列：0-1 上行楼梯、2 下行楼梯、3 `boxes`、4 粗糙、5 斜坡、6-8 沟、9 **平地**
  —— 其中 col4/col9 **本来就没有障碍**（＝用户前一次看到的"平地和粗糙确实在正常走"）。
* `--scan187` 只改高度扫描器的几何，**不改地形**。

#### 29.22.3 真实几何（离线用 `track_stairs_terrain` / `track_step_terrain` 算的）

| level | difficulty | 台阶单级 | 台阶累计升高 | `boxes` 单块高 |
|---:|---:|---:|---:|---:|
| 0 | 0.05 | **5.5 cm** | 33 cm | **9.1 cm** |
| 1 | 0.15 | 6.5 cm | 39 cm | 11.3 cm |
| 2 | 0.25 | 7.5 cm | 45 cm | 13.5 cm |
| 3 | 0.35 | 8.5 cm | 51 cm | 15.7 cm |
| 4 | 0.45 | 9.5 cm | 57 cm | 17.9 cm |
| 5 | 0.55 | 10.5 cm | 63 cm | 20.1 cm |
| 6 | 0.65 | 11.5 cm | 69 cm | 22.3 cm |
| 7 | 0.75 | 12.5 cm | 75 cm | 24.5 cm |
| 8 | 0.85 | 13.5 cm | 81 cm | 26.7 cm |
| 9 | 0.95 | **14.5 cm** | 87 cm | **28.9 cm** |

（`difficulty = (row + U(0,1)) / num_rows`、`num_rows=10` ⇒ level k 的名义难度是 `(k+0.5)/10`，
所以**区间端点 0.05/0.15 与 0.08/0.30 是"连续扫描的上下界"，不是 level 0/9 的精确值**。）

#### 29.22.4 新增 `play.py --terrain_level=N`（钉死等级，用于检查指定难度）

`play.py` 新增开关：把全部环境的 `terrain_levels` 设为 N、按 `terrain_origins[level, type]` 重算
`env_origins`、**把 `terrain_levels` 课程项换成 no-op**（否则一局内仍会升降级），然后
`env.unwrapped.reset()` 让机器人按新原点重新出生。

```bash
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/rl_lab/cmoe/play.py \
  --task=Imgo2-basemove-rough-cmoe-play --checkpoint=<...> --scan187 \
  --num_envs=10 --terrain_level=9 --livestream 2
```

⚠️ **未经运行验证**（本机无 GPU）：全程 fail-soft —— 任一步失败只打印 `[WARN]`，回放照常按默认等级进行。
