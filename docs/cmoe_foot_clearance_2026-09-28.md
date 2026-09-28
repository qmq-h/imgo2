# CMoE 抬脚高度奖励 `feet_swing_clearance`（2026-09-28）

新增一项"**相对支撑面**的抬脚高度奖励"，用来补上 `-gaitfree` 链路里"脚离地多高完全没人管"的洞。
本记录写清：现象与依据、为什么不用已有的两项、语义与实现、沟壑/飞行相处理、按地形参数、
探针、测试、以及**没有验证的部分**。

## 1. 现象（为什么要加）

用户观察 + 实测：`-gaitfree` 任务链里 `feet_height_body` 与 `feet_height` **都被归零**
⇒ 奖励里**没有任何一项**对"脚离地多高"有压力：

* 实测平地足端**滞空时长从先验的 0.084 s 掉到 0.015 s**（`gait_airtime_flat`），
  形态是**蹭地拖行**（不是"抬起来落脚"）；
* `-gaitfree` 的五项手工步态 shaping 归零后，剩下的步态相关项只有 `feet_slide`（不许打滑）、
  `lin_vel_z_l2`（别蹦）与 `flat_orientation_l2`（机身水平）——**它们都不管抬脚高度**。

## 2. 为什么不能直接启用已有的两项（**必须保持它们为 0**）

| 项 | 用的是什么 z | 缺陷 |
|---|---|---|
| `mdp/rewards.py::feet_height_body` | **机体系**（`body_pos_w − root_pos_w` 再旋到机体系） | ① 可以被"**压低基座**"而不是"抬脚"满足：数值上把基座压 5 cm 就净赚约 **+0.02/步** reward；② 与机体俯仰/脚的先后位置耦合（`z_body ≈ −sinθ·Δx + cosθ·Δz`），台阶/坡上量到的不是"离地高度" |
| `mdp/rewards.py::feet_height` | 世界系（好，不受基座高度影响） | `target_height` 是**固定绝对世界高度**（基类默认 0.05，rough 里 0.08）⇒ 只在"地面在 z≈0"的瓦片上成立。台阶/箱块/横栏/混合/窄梯瓦片内高差 0~0.9 m、斜坡瓦片内升到 ~0.4 m ⇒ **会误罚**。掩码救不了：**同一瓦片内部就有高差** |

结论：要的是"**足相对当前支撑面**的高度"，而支撑面必须**实测**（不能写死世界高度），
并且要能**按地形给不同的目标高度**。这就是新项 `feet_swing_clearance`。

## 3. 新项语义（严格按用户定稿）

```text
z_ref   = mean(世界 z of 接触中的足)          # 支撑面参考；只用 contact 传感器判定
h_i     = z_foot_i_world − z_ref              # 第 i 只脚的"相对支撑面离地高度"
swing_i = 1{第 i 只脚 不 在接触}
r       = k · Σ_i swing_i · clamp(1 − |h_i − h*| / band, 0, 1)
```

* `h*`（满分目标高度）与 `band`（容差半宽）：`|h − h*| ≥ band` ⇒ 该项 **0**（带外**不加分**，
  所以"高抬腿"不刷分）；
* 参考面 `z_ref` 是**实测**的：基座整体抬/压同一个量时 `h_i` **逐个元素不变** ⇒ 旧
  `feet_height_body` 的"压身体换 reward"路径在数学上不存在；
* 纯数学放在新文件 `imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/mdp/clearance_math.py`
  （**只依赖 torch**，可离线单测）：
  * `swing_clearance_reward(foot_z_world, contact_mask, *, target, band, reference, foot_lin_vel_xy=None, tanh_mult=None)`；
  * `stance_reference(foot_z_world, contact_mask, fallback, max_hold_steps, hold_steps=None) -> (reference, valid)`。

### 3.1 与任务书"签名建议"的两处偏离（有意，已在此说明）

1. **`stance_reference` 多一个入参 `hold_steps`、多一个出参 `valid`**：
   "无接触时最多沿用 `max_hold_steps` 步"必须有状态（已经沿用了几步），而纯函数不能保存状态。
   任务书本身允许用 mask 作"无效"标记，这里就返回 `(reference, valid)`，参考无效处是 `NaN`。
   状态（`_reference` / `_hold`）由类 `FeetSwingClearance` 持有并每步更新：
   有接触 ⇒ `hold = 0`；无接触 ⇒ `hold += 1`（于是"最后一次接触之后最多沿用 `max_hold_steps` 步"）。
2. **多了一个可选的 `tanh_mult`**（与 `feet_height` 同形的速度门），**默认 `None`＝关闭**，
   CMoE 配置里也是 `None` ⇒ 实际生效的就是上面那个严格算式。留着只是备一个"慢速摆动足不给分"
   的旋钮。

### 3.2 类实现额外加的一道门（超出任务书列出的公式，repo 惯例）

`FeetSwingClearance.__call__` 在返回前乘了**直立门** `clamp(−g_z, 0, 0.7)/0.7`（与
`feet_height` / `feet_air_time` / `joint_mirror` / `track_*` 等本文件几乎所有项一致），
以及**有命令门**（`‖cmd‖ > 0.1`，任务书明确要求，与 `feet_height` 一致）。
直立门不改变"抬脚"语义，只是沿用仓库惯例，避免翻倒时给正分（基座触地本身会终止回合，
所以影响窗口本来很小）。**如果要严格按公式、不要门，去掉那一行即可**（本记录与测试都不依赖它）。

## 4. 参考面 / 飞行相 / 沟壑：必须显式处理的三种情形

| 情形 | 处理 | 依据 |
|---|---|---|
| **正常支撑** | `z_ref` ＝当前接触足世界 z 的均值 | 支撑面就是"踩着的那些脚" |
| **飞行相**（没有任何脚接触） | 沿用最近一次**有效** `z_ref` 最多 `max_hold_steps = 3` 步（= 60 ms 控制步）⇒ 躲开起跳瞬间的参考面抖动；超过 ⇒ 该项 **0** | 长滞空不再给分，防止"腾空收腿"刷分；也让真正的跃沟只吃"起跳/落地"两小段 |
| **悬空/沟壑**（基座下方无地面） | 复用 `base_height_l2` 的**射线有效性**判定：读 `env.scene.sensors["height_scanner_base"].data.ray_hits_w`，**整束全落空** ⇒ 该环境该项为 **0** | 此时"支撑面"根本不存在，不该按任何 `h*` 付钱（用户特别强调"也要考虑沟壑"） |
| **回合重置** | 本回合第一步（`episode_length_buf <= 1`）清掉 `z_ref` 缓冲与 hold 计数 | 换瓦片/换初始位姿后旧参考面无意义 |

### 4.1 一个真踩到的坑：`episode_length_buf == 0` 永远不成立

最初的写法是"`episode_length_buf == 0`（回合第一步）时清缓冲"。**这是错的**：
Isaac Lab 的 `manager_based_rl_env.py` 在**算奖励之前**就 `episode_length_buf += 1`（`:201`），
而把它清 0 的 `_reset_idx` 发生在**算完奖励之后**（`:221` / `:396`）⇒ 回合第一步算奖励时该值是
**1**（只有刚构造、还没步进过时才是 0）。用 `== 0` 会**一次也命中不了**、缓冲永远不清。
现改为 `<= 1`，并由 `tests/test_feet_swing_clearance.py::test_episode_reset_clears_the_reference_buffer`
（用 1 做输入）与 `test_reference_buffer_survives_a_mid_episode_step`（用 3 做输入）双向钉住。

## 5. 按地形给 `h*`/`band`

表在 `CMoE_env_cfg.py` 顶部的 `SWING_CLEARANCE_TERRAIN_GROUPS`（**只有地形名，没有列数**）：

| 地形（用户定的分组） | `h*` | `band` | 依据 |
|---|---|---|---|
| `flat`、`random_rough` | 0.07 | 0.05 | 与真机录制基线（足端 z 峰峰值中位 **0.090 m**）同量级，也接近 PPO 那套 `feet_height` 的 0.08 |
| `hf_pyramid_slope`、`hf_pyramid_slope_inv` | 0.07 | 0.05 | 参考面已经跟着坡面走，不需要额外抬高（抬更高只多耗功） |
| `pyramid_stairs`、`pyramid_stairs_inv`、`narrow_stairs` | 0.09 | 0.06 | 单级台阶 0.05–0.15 m，脚要抬过台阶边缘 |
| `boxes`、`gap`、`hurdle`、`mix` | 0.12 | 0.10 | **更宽是有意的**：大跨步/跃起不该被罚；带外不加分 ⇒ 也不会靠高抬腿刷分（`0.02~0.22 m` 都有分，`0.12` 是满分点） |

`clearance_terrain_params()` 把这张表摊成两张 `地形名 → 数值` 字典，并**校验它们恰好覆盖
全部 `sub_terrains` 键**：写错名字、或以后新增地形忘了归类，都在 `__post_init__` 里直接
`RuntimeError`（**不像掩码那样静默退回默认档**）。未列到的地形名在类里退回
`SWING_CLEARANCE_DEFAULT_TARGET/BAND = 0.07/0.05`（只服务 `velocity_env_cfg.py` 里那张空表的默认 RewTerm）。

## 6. 接线与生效项数

* `velocity_env_cfg.py::RewardsCfg`：新增 `feet_swing_clearance = RewTerm(func=mdp.feet_swing_clearance, weight=0.0, params={...})`
  ⇒ **默认 0，不影响任何现有任务**（`disable_zero_weight_rewards()` 会把它移除）；
* `CMoE_env_cfg.py::Imgo2CMoERoughEnvCfg.__post_init__`：
  * `weight = +0.5`（用户建议值；量级参照 `feet_air_time +1.0`，比 `track_world_vel_xy_exp 5.0`
    低一个量级 ⇒ "明确的偏好"而非主导项）；
  * `asset_cfg`/`sensor_cfg` 的 body 过滤都用 `self.foot_link_name`（`.*_FOOT`）；
  * `k=1.0`、`tanh_mult=None`、`max_hold_steps=3`、`free_terrain_names=()`（全地形生效）；
  * `feet_height`（本来就是 0）与 `feet_height_body`（**−5.0 → 0**）保持为 0，接线留着并写明原因；
  * `-gaitfree` 子类**不归零**这一项（它正是用来补 gaitfree 缺失的抬脚压力）。
* 接触判定：`contact_forces` 的 `net_forces_w_history` → 力范数 → 历史维取最大 → `> 1.0 N`
  （与 `feet_slide`、`cmoe/play.py` 的 dump 代码**逐字一致**，含 `history_length=3`、`track_air_time=True`）。
* **生效项数**（`scripts/tools/check_reward_overrides.py` 实跑）：
  `cmoe` **27 → 29**（+`feet_swing_clearance` +2 个诊断项 −`feet_height_body`）；
  `cmoe-gaitfree` **23 → 26**（同一项 + 2 个诊断项，`feet_height_body` 本来就是 0）。

## 7. 逐地形探针（观测"是否用压身体换抬脚"）

两个 **1e-6 权重**（只写 TB、不进梯度的量级）诊断项，并接进
`mdp/curriculums.py::terrain_levels_vel_logged` 的 `gait_metric_terms` ⇒ 逐列读数：

| 探针项 | 逐列 tag | 含义 |
|---|---|---|
| `diag_clearance_mean` | `gait_clearance_<地形>` | 该列**摆动足** `h_i` 的均值（m）。与奖励的两处有意不同：不带缓冲（飞行相读 0）、不做命令门/直立门/带形 ⇒ 就是原始几何量 |
| `diag_base_height` | `gait_base_height_<地形>` | 该列基座相对**局部地面**的**有符号**高度误差（m，正=偏高）；射线逻辑与 `base_height_l2` 同一套，但不平方、不乘门 ⇒ "压低基座"（负）与"抬高身体"（正）可分辨 |

**判读**：`gait_clearance_*` 涨、`gait_base_height_*` 也涨 ⇒ 抬脚靠"抬高身体"（那个方向本来就被
`base_height_l2 −10` 按住）；`gait_clearance_*` 涨而 `gait_base_height_*` 更低 ⇒ 才是"压低身体换抬脚"
（旧 `feet_height_body` 的刷分路径）。已有的 `gait_height_<地形>`（`base_height_l2` 的 `(误差)²` 反解）
仍在，两者互补（这里给符号）。

## 8. 测试（`imgo2_rl/tests/test_feet_swing_clearance.py`，41 项，无需 GPU/Isaac Lab）

* **纯函数**：`h*` 处满分、两只摆动足求和（不是求平均）、`|h−h*|=band` ⇒ 0、接触足不计分、
  全接触 ⇒ 0、`band=0` 退化为"精确相等"指示（不产生 NaN）、参考无效（NaN/Inf）⇒ 0、
  逐环境 `target`/`band`、可选速度门默认关、形状校验报错；
* **压身体不变性**：足端世界 z 与 `z_ref` 同步 `−0.05` ⇒ 输出**逐元素完全不变**
  （用一组差值精确可表示的数值钉 `torch.equal`；另有随机值 1e-6 一致的用例）；
* **`stance_reference`**：全接触＝均值、只算接触足、无接触时沿用缓冲、超过 `max_hold_steps`
  ⇒ `valid=False` 且参考 NaN、无历史 ⇒ 无效、`max_hold_steps=0` ⇒ 飞行相恒无效；
* **类本体（桩 env 跑真实类源码）**：逐地形 `h*`/`band` 生效、`free_terrain_names` 掩码、
  飞行相"沿用 3 步后置 0"、沟壑（射线全落空）置 0、命令门、直立门、**回合重置清缓冲**、
  同回合中途不清缓冲、`k` 线性缩放；
* **接线**：CMoE 链权重 > 0、`feet_height`/`feet_height_body` 仍为 0、`-gaitfree` 不归零、
  公共配置默认 0、`params` 键 ↔ 函数签名（Isaac Lab `manager_base` 的集合校验）、
  逐地形参数名都真实存在于 `sub_terrains`、分组互斥且恰好覆盖、逐档数值、
  **用生产代码里的 `clearance_terrain_params()` 做负向对照**（把 `flat` 打成 `flatt` ⇒ 必须 raise；
  新增地形忘了归类 ⇒ 必须 raise）、两条诊断项在奖励表与逐列聚合名单里都在。

**实测**（`/opt/conda/envs/isaaclab/bin/python -m unittest discover -s tests`）：
本轮全仓 **555 通过 / 8 跳过**（改前 **512 通过 / 8 跳过**；新增 41 项 + 既有审计测试 +2 项）。
`check_reward_overrides.py cmoe` = **29 项生效**、`cmoe-gaitfree` = **26 项生效**；
`check_terrain_columns.py` 末行 `全部掩码引用的地形名都有 ≥1 列 ✅`；
`check_asset_paths.py` PASS；`compileall`、`git diff --check`、tracked-ignore 干净。

## 9. 未验证项与限制（**没有在真实 Isaac 里跑过**）

1. **真实 Isaac 行为完全未验证**：本机（`gpufree-container`）无可用 GPU／Isaac Lab 运行环境，
   上面全部证据都是**离线**的（纯 torch + 源码级断言 + AST 桩 env）。没有构造过环境、没有短训、
   没有回放 ⇒ **不能声称这一项在训练里有效**。
2. **接触阈值 1.0 N 未经标定**：沿用的是 `feet_slide`／`play.py` 的口径。若足端接触力在
   本机器人/本质量下长期低于 1 N（或高于），接触判定会整体偏松/偏紧，`z_ref` 与 `swing` 都会受影响。
3. **接触判定取"历史最大"会"粘"住 15 ms**（`history_length=3` × `sim.dt=0.005`）：
   刚抬起的脚可能仍被算作支撑足，`z_ref` 因此可能短暂偏高（量级 ≈ 脚高 /(F−1)），
   摆动足计分的起点也会晚 15 ms。这是"与仓库既有接触口径一致"的有意取舍，是否需要改成
   `net_forces_w`（当前力）要用真实回放标定。
4. **`h*`/`band` 的数值没有真实标定**：0.07/0.05、0.09/0.06、0.12/0.10 是用户给的设计值
   （依据是录制基线 0.090 m 与台阶高度区间），不是调参结果；`weight = +0.5` 同理。
5. **压身体不变性的严格版用的是特殊数值**：`0.05` 不是二进制精确数，一般数值下
   `(z−d)−(z_ref−d)` 与 `z−z_ref` 会差 1 ulp（实测 ~1e-8）⇒ "逐元素完全不变"在浮点上只到
   1e-6 级（数学上成立）。测试用一组差值精确可表示的数值钉严格版 + 一组随机值钉 1e-6。
6. **类的直立门／可选速度门**：直立门是 repo 惯例（超出任务书公式，见 §3.2），速度门默认关、
   未启用、也未经真实数据检验。
7. **诊断项的读数含义**依赖 `terrain_levels_vel_logged` 的逐列聚合（`env_ids` 只是"本步刚结束
   回合的环境"，某列可能没有样本 ⇒ 该键不写），因此逐列 `gait_clearance_*` 可能有空档 —— 与
   既有 `gait_*` 探针同一行为。
8. **`feet_height_body` 归零的连带影响**：它原本在 CMoE 里权重 −5.0（掩码豁免 boxes/gap），
   现在为 0 ⇒ 这一轮配方的"步态 shaping 总量"下降（−5.0 → 0），同时 +0.5 的正项进来。
   net 变化的方向**没有真实训练证据**；旧 run 的曲线不可直接对比（配方已变 ⇒ 必须新 run）。

## 10. 待训练机做的事（验收方向）

1. 起新 run（奖励改动**不能 resume**）：`--task=Imgo2-basemove-rough-cmoe-gaitfree`；
2. 启动后确认 TB 的 `Episode_Reward/` 里**有** `feet_swing_clearance`、**没有**
   `feet_height`／`feet_height_body`；逐列探针出现 `gait_clearance_<地形>`／`gait_base_height_<地形>`；
3. 关键读数：`gait_airtime_flat` 应从 0.015 s 回升（先验 0.084、PPO 那套 0.28 上下），
   且 `gait_clearance_flat` 与 `gait_base_height_flat` **不要把基座往下压**；
4. 若 `gait_clearance_*` 长期为 0：先查接触判定（阈值 1.0 N 是否过松 ⇒ 全部算接触）与
   `height_scanner_base` 的射线是否真的在基座正下方；
5. 回放（`cmoe/play.py --dump_gait`）确认平地滞空与 duty 的实际变化。
