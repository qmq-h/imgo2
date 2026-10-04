# `mix` 受控测试场景（`Imgo2-basemove-rough-cmoe-mix-test`）— 2026-10-04

## 1. 目的与范围

用户 2026-10-04 要求新增一个**继承 play 任务**的"测试场景"任务类：

* 场景里**只有 `mix` 一种地形**；
* **速度指令只给前进**；
* **横向（y）用 PD 外环控制**；
* **heading 保持为 0**。

用途：在受控条件下评估策略通过 `mix`（复合障碍：窄走廊 + 台阶上行 + 深坑 + 高台 + 高栏 + 深坑）
的能力，**把横向漂移与航向漂移从评测里剔除**。

**这不是新策略任务**：动作空间、观测组与维度、奖励、终止、课程**一字未改**，只是换了一个场景 +
一层指令外环 ⇒ 既有 CMoE checkpoint（77 维地形 / 527 维 actor / 125 维 critic）**可直接加载**。

## 2. 改动清单

| 文件 | 改动 |
|---|---|
| `imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/mdp/mix_test_pd.py` | **新增**：纯数学（只依赖 `torch`）——`lateral_heading_pd`、`rotate_base_to_world` 与默认增益常量 |
| `.../velocity/mdp/mix_test_command.py` | **新增**：`MixTestVelocityCommand`（继承 Isaac Lab `UniformVelocityCommand`）+ `MixTestVelocityCommandCfg`（PD 增益字段） |
| `.../velocity/mdp/__init__.py` | 加 `from .mix_test_command import *` |
| `.../velocity/base_move/CMoE_env_cfg.py` | **新增** `Imgo2CMoEMixTestEnvCfg(Imgo2CMoERoughPlayEnvCfg)` |
| `.../velocity/base_move/__init__.py` | 注册 `Imgo2-basemove-rough-cmoe-mix-test`（按 `-play` 同族） |
| `imgo2_rl/scripts/tools/check_terrain_columns.py` | 加 `--task`（按任务区分）＋解析改**类作用域** |
| `imgo2_rl/tests/test_cmoe_mix_test_scene.py` | **新增** 35 项离线测试 |
| `imgo2_rl/tests/test_check_terrain_columns.py` | 新增 `TestMixTestTerrainColumns` 14 项（原判据原样保留 + 负向对照） |

## 3. 场景契约

由 `Imgo2CMoEMixTestEnvCfg.__post_init__`（继承 `Imgo2CMoERoughPlayEnvCfg`）设定：

* **地形**：`self.scene.terrain.terrain_generator.sub_terrains.clear()` 后只放一处
  `CMoETrackMixTerrainCfg(proportion=1.0, x_unit=0.02, z_unit=0.002, height_scale=1.1,
  gap_shrink_units=10.0, corridor_width=0.80, pit_depth=0.50, pattern_start_x=0.30, spawn_x=0.75)`。
  除 `proportion` 外**逐字沿用训练实例化时的值**（训练侧是 `CMoETrackMixTerrainCfg(proportion=0.10)`，
  其余字段全取类默认值）——`tests/test_cmoe_mix_test_scene.py` 会把这些**字面量**与
  `cmoe_terrains.py` 的类默认值**逐一比对**，改默认值而不同步这里就会红。
  规格原文写的是 `prop_start_x=0.30`，本仓的真实字段名是 **`pattern_start_x`**（见
  `cmoe_terrains.py::CMoETrackMixTerrainCfg`），故按真实字段名落地。
* **网格**：`num_cols = 1`、`num_rows = 20` ⇒ 20 行 × 1 列。`--terrain_level=N` 把全部环境钉在第 N 行
  ⇒ **难度 = N/20**（每 0.05 一档；`d = row/num_rows` 由 Isaac Lab 的 `terrain_generator` 决定）。
* **环境数**：`self.scene.num_envs = 20`（可被命令行 `--num_envs` 覆盖）。
* **继承来的确定性**：`pose_range` / `velocity_range` 全 0、全部域随机化事件置 `None`、观测噪声关闭
  （`observations.policy/terrain.enable_corruption = False`）、`max_init_terrain_level = 5`。
* **契约不变**：新类体里**没有**任何对 `self.observations` / `self.actions` / `self.rewards` /
  `self.terminations` / `self.curriculum` / `self.scene.height_scanner` / `terrain_generator.size` 的赋值
  （测试里用 AST 逐条断言），也没有出现 `ObsGroup` / `ObsTerm` / `ActionCfg`。

### 3.1 速度指令

| 项 | 值 |
|---|---|
| `ranges.lin_vel_x` | **`(1.0, 1.0)`** ⇒ 恒定 **1.0 m/s** 前进（2026-10-04 用户规格修正） |
| `ranges.lin_vel_y` | `(0.0, 0.0)`（横向由 PD 每步写，范围只作初值/占位） |
| `ranges.ang_vel_z` | `(0.0, 0.0)`（航向由 PD 每步写，范围只作初值/占位） |
| `ranges.heading` | `(0.0, 0.0)` ⇒ heading 目标恒 0 |
| `heading_command` | `True`（规格要求；也是 `ranges.heading` 的合法性前提） |
| `rel_heading_envs` / `rel_standing_envs` | `1.0` / `0.0` |
| `resampling_time_range` | `(1.0e9, 1.0e9)` ⇒ **不做指令重采样**，整个 episode 指令恒定 |

`self.commands.base_velocity` 是**整项替换**成 `mdp.MixTestVelocityCommandCfg`（不是逐字段覆盖）：
新命令项需要 PD 增益字段，语义与 `UniformThresholdVelocityCommandCfg` 不同，逐字段覆盖容易漏
（参见 CMOE-04 的"晚赋值覆盖"教训）。替换后 `forward_only_terrain_names` 等阈值命令专属字段不再存在，
**本命令项根本不读它们**；`export_deploy_cfg.py` 只读 `ranges`（本 cfg 有），不受影响。

## 4. 命令项设计（`mdp/mix_test_command.py`）

1. **继承 `UniformVelocityCommand`**：复用它的 `vel_command_b` / `heading_target` /
   `is_heading_env` / `is_standing_env` / `metrics` 与 `_update_metrics`，只重写
   `_resample_command` / `_update_command`。
2. **`vx_cmd` 恒定**：`super()._resample_command()` 采样**一次**（`lin_vel_x=(1.0,1.0)`），
   `_update_command` **不碰第 0 列**（测试断言 `vel_command_b[:,0]` 恒为 1.0）。
3. **横向 PD**（每步重算，PD 是时变的）：写 `vel_command_b[:,1]`。
4. **航向 PD**：写 `vel_command_b[:,2]`。
5. **写回 `vel_command_b` 与 `vel_command_w`**：奖励（`track_world_vel_xy_exp` 等）与观测读的
   仍是 `command`（＝`vel_command_b`）；`vel_command_w` 是**世界系镜像**，见 §6。
6. **纯函数化**：PD 数学在 `mdp/mix_test_pd.py`（只依赖 torch），可离线单测；命令项只负责
   把 `root_pos_w` / `root_lin_vel_w` / `heading_w` / `root_ang_vel_b` 接上去。

### 4.1 坐标系约定（**必读**）

| 变量 | 取法 | 为什么 |
|---|---|---|
| `y`（`y_local`） | **世界系**横向位置 − 本环境出生原点：`root_pos_w[:,1] − env.scene.env_origins[:,1]` | `env_origins` 是该 env 的地形瓦片出生点 ⇒ 这个差值就是**离赛道中心线的横向距离**（玩家出生在走廊中心 y=0） |
| `vy`（`vy_local`） | **世界系**横向线速度：`root_lin_vel_w[:,1]` | 与 `y` 同坐标系，也与评测奖励 `track_world_vel_xy_exp`（**世界系**，2026-09-24 起）一致 |
| `yaw` | `robot.data.heading_w` | Isaac Lab 由 **`root_quat_w`** 把机体系 +x 投到世界系后 `atan2(y, x)` ⇒ 就是偏航角，且已绕回 `(−π, π]` |
| `wz`（`wz_local`） | **机体系**偏航角速度：`root_ang_vel_b[:,2]` | 命令 `vel_command_b[:,2]` 写的就是机体系角速度；既有 `error_vel_yaw` 度量也用它 |

**备选（未采用，记录以便复核）**：把 `vy` 换成机体系 `root_lin_vel_b[:,1]`。在 yaw 已被航向 PD 压到
接近 0 时两者几乎等价；选世界系是为了与 `y`（世界系位置误差）和世界系速度奖励同口径。

### 4.2 为什么不用内置 heading 控制器（规格 ③.3 要求的说明）

`UniformVelocityCommand._update_command` 在 `heading_command=True` 时用**内置 P 控制器**
（`vel_command_b[:,2] = clip(heading_control_stiffness · heading_error, ang_vel_z[0], ang_vel_z[1])`）
写 `wz`。本任务：

* `ranges.ang_vel_z = (0,0)` ⇒ 内置控制器会把 `wz` **恒夹成 0**，航向保持根本没有输出；
* 内置控制器只有 P 项，没有 `wz` 阻尼，压不住高速下的航向振荡。

因此本命令项**不使用**内置控制器，而是每步用自己的 PD 直接写 `vel_command_b[:,2]`
（等价于"关掉内置 heading 控制器、由本 PD 直接写 `wz`"）。**cfg 里 `heading_command` 仍保持 `True`**
——规格 ② 要求它，而且 `UniformVelocityCommand.__init__` 用它做 `ranges.heading is not None` 的合法性检查。
`heading_control_stiffness` 对本命令项**无作用**（未显式设置，取基类默认 1.0）。

## 5. PD 公式、默认增益与调法

```
vy_cmd = clip( kp_y · (y_des − y)        + kd_y · (0 − vy),   −vy_max, +vy_max )
wz_cmd = clip( kp_h · (yaw_des − yaw)    + kd_h · (0 − wz),   −wz_max, +wz_max )
```

默认增益（**用户给定**，全部走 `MixTestVelocityCommandCfg` 的字段 ⇒ 可覆盖）：

| 参数 | 默认 | 单位 | 含义 |
|---|---|---|---|
| `kp_y` | **1.0** | 1/s | 横向位置比例增益（1 m 偏移 ⇒ 1 m/s 指令） |
| `kd_y` | **0.3** | — | 横向速度阻尼 |
| `vy_max` | **0.6** | m/s | 横向指令上限 |
| `y_des` | **0.0** | m | 横向目标（贴赛道中心线） |
| `kp_h` | **1.5** | 1/s | 航向比例增益（1 rad 误差 ⇒ 1.5 rad/s 指令） |
| `kd_h` | **0.3** | — | 航向阻尼 |
| `wz_max` | **1.0** | rad/s | 航向指令上限 |
| `yaw_des` | **0.0** | rad | 航向目标（始终朝世界 +x） |

### 怎么调

* **改前进速度**：`CMoE_env_cfg.py` 里 `ranges.lin_vel_x=(1.0, 1.0)` 那一行。想做速度扫描就把它改成
  区间（如 `(0.5, 1.0)`）**并**把 `resampling_time_range` 缩短（否则整回合只采样一次），
  或者干脆用 `eval_gait.py` 那类逐速度评测入口。
* **改 PD 增益**：目前写在 `MixTestVelocityCommandCfg` 的字段默认值里（`mdp/mix_test_command.py`）。
  要按 run 调，在 `Imgo2CMoEMixTestEnvCfg.__post_init__` 构造 cfg 时传参即可，例如
  `self.commands.base_velocity = mdp.MixTestVelocityCommandCfg(..., kp_y=1.5, vy_max=0.8)`。
* **调参直觉**：`kp_y` 大了会来回摆（`vy` 阻尼项 `kd_y` 要跟着加）；`vy_max` 是"允许用多大横移去纠偏"，
  它是**唯一**能防止"用一个更大的横移换回中心线"的上限；`kp_h`/`wz_max` 同理约束转向。
  参考量级：站立高度 0.30 m、走廊半宽 0.40 m，而 `vy_max=0.6 m/s` 已经接近行走速度
  （1.0 m/s）的 60%，因此**横向纠偏不能指望它救回大偏差**——本场景的前提是"横移本来就不该发生"。
* **关掉转向、只测横向**：把 `kp_h`/`kd_h` 置 0 即可（`yaw_des` 仍 0，只做阻尼）。

## 6. `vel_command_w`（**与规格的一处冲突，已按"不改设计、只报告"处理**）

规格要求"把 `vy_cmd`/`wz_cmd` 写进该 command 的 `vel_command_b`/`vel_command_w`，
**与既有 `base_velocity` 的写法一致**"。但事实是：**基类 `UniformVelocityCommand` 与本仓都没有
`vel_command_w`**（全仓 `grep -rn vel_command_w` 在本次改动前为空）。处理方式：

* `vel_command_b` 是真正生效的缓冲（奖励、观测、度量都读它）；
* `vel_command_w` 由本命令项**新增**为纯附加缓冲，每步写入"机体系指令按当前 yaw 旋转到世界系"的镜像
  （`rotate_base_to_world`，纯函数、可离线单测）；
* 它**不进入观测、不进入动作空间、不参与奖励**，因此**不影响 checkpoint 兼容性**，仅供回放/审计读。

## 7. 运行示例

回放/评测（须替换 checkpoint 绝对路径）：

```bash
cd imgo2_rl
python scripts/rl_lab/cmoe/play.py \
    --task=Imgo2-basemove-rough-cmoe-mix-test \
    --num_envs=20 --headless \
    --terrain_level=10 \
    --checkpoint="/absolute/path/to/model.pt"
```

* `--terrain_level=N` ⇒ 全部环境钉在第 N 行 ⇒ **难度 N/20**（如 N=10 ⇒ d=0.5、N=20 ⇒ d=1.0）。
* `--num_envs=20` 时网格 20 行 × 1 列刚好一格一个环境；**但初始等级仍是随机的 0–5 行**
  （见 §9 冲突 ②）。
* 也可用 `--force_expert K` 等既有 CMoE play 开关（本任务不引入新开关）。

## 8. 地形列校验：按任务区分

`scripts/tools/check_terrain_columns.py` 的原判据是"**所有被掩码引用的地形名都必须 ≥1 列**"。
本测试场景**只有 `mix`** ⇒ 掩码引用的 `boxes` / `gap` / `flat` 在该场景**不存在**，引用它们的项
（`joint_mirror`、`feet_air_time`、`feet_height_body`、`feet_air_time_variance`、`feet_gait`、
`base_height_flat_l2`）在本场景**恒为 0**——这是**预期**，不是配置错误。

处理方式（用户要求"不削弱原判据"）：

* 新增 `--task`，默认 `cmoe-rough` **走原判据、一字未放宽**；
* `--task mix-test`（＝ `Imgo2-basemove-rough-cmoe-mix-test`）走**单独分支**：
  仍强制 `sub_terrains.clear()` + 只有 mix + mix ≥1 列 + `num_cols=1` + `num_rows=20`，
  并把"掩码引用但本场景不存在"的名字**逐条打印为预期**；
* 解析从"整文件 `ast.walk`"改成**类作用域**（只走指定类的 `__post_init__`）——否则 mix-test 类里的
  `sub_terrains["mix"] = ...(proportion=1.0)` 会**静默污染**训练侧的比例解析；
* 测试对 test 任务**单独断言**，并加两条**负向对照**（把"只有 mix"丢给原判据必须仍报错、
  存在但 0 列也必须报错）证明原判据没被放宽。

实测输出（`python3 scripts/tools/check_terrain_columns.py --task mix-test`，退出码 0）：

```
=== [Imgo2-basemove-rough-cmoe-mix-test] 受控测试场景（只 mix；cfg 类 Imgo2CMoEMixTestEnvCfg）===

=== Imgo2-basemove-rough-cmoe-mix-test｜只 mix｜num_cols==类数 ⇒ 等比例（每类 1 列）（num_cols=1，共 1 列）===
  mix                      1 列
  ✅ 唯一地形 mix 有 1 列（num_cols=1、num_rows=20）
  ⚠️ 本场景**只有 mix**；掩码引用但本场景**不存在**的地形名（原任务判据**不放宽**，这些项在本场景恒为 0 属**预期**）：
    - joint_mirror.free_terrain_names → boxes：本场景无该地形 ⇒ 该项本场景恒为 0（预期，不是配置错误）
    - joint_mirror.bound_terrain_names → gap：本场景无该地形 ⇒ 该项本场景恒为 0（预期，不是配置错误）
    ... （feet_air_time / feet_height_body / feet_air_time_variance / feet_gait 的 boxes、gap，
         base_height_flat_l2.active_terrain_names 的 flat，共 11 条，逐条列出）
  ✅ `forward_only_terrain_names`（继承父类） 覆盖 mix；另含 10 个本场景不存在的地形名（...）—— 命令项
     `MixTestVelocityCommand` 根本不读这张表，且 `is_env_assigned_to_terrain` 对未登记的名字返回全 False
     ⇒ 在这里是**惰性**的，属预期

结论：test 任务只有 mix 一种地形；掩码引用的其它地形名在本场景恒为 0，属**预期** ✅
```

## 9. 未验证项与与规格的冲突

**未验证（本容器无 GPU / 无 Isaac 运行环境：`NVIDIA_VISIBLE_DEVICES=void`、`import isaaclab` 缺 `omni.log`）**：

1. **未构造 Isaac 环境**：`MixTestVelocityCommandCfg` 在真实 `CommandManager` 下的解析、
   `debug_vis=True` 在 20 环境下的 marker 创建、`vel_command_w` 这个附加缓冲在 `CommandTerm` 生命周期里
   的行为都未实测。
2. **未回放、未导出 `policy.pt`**：checkpoint 兼容性是**静态推理**（动作/观测契约未改），未用真实
   `runner.load()` 验证过。
3. **PD 增益未标定**：`1.0/0.3/0.6`（横向）与 `1.5/0.3/1.0`（航向）是用户给定的默认值，**未在仿真里
   标定**过稳定性/超调；`y`/`vy` 用世界系这一选择的影响未量化。
4. **`mix` 单列 20 行的实际生成未跑**：地形网格、20 行难度映射、`--terrain_level=N` 的实际几何
   （含 `pattern_start_x=0.30` 是否确实装得下 0.75 m 出生点）都只在源码/离线层面核对。
5. **奖励掩码在本场景恒为 0 的实际影响未测**：本场景 6 项掩码项恒 0 ⇒ 评测时 `Episode_Reward/*` 里
   这些键的读数不能与训练 run 直接对比（这只是评测读数问题，不是 bug）。
6. **一处已知的读数副作用（有意保留）**：基类 `_update_metrics` 用
   `max_command_step = resampling_time_range[1] / step_dt` 做归一化；本任务
   `resampling_time_range = (1e9, 1e9)` ⇒ `error_vel_xy` / `error_vel_yaw` 这两个**度量**会读数≈0
   （只进日志、不参与控制与奖励）。要可读的跟踪误差请用奖励分项
   `Episode_Reward/track_world_vel_xy_exp` 与 `track_ang_vel_z_exp`。

**与规格的冲突（未擅自改设计，按要求报告）**：

1. **`vel_command_w` 不存在**：规格假定它与 `vel_command_b` 一样是既有写法，实际基类与本仓都没有。
   已按 §6 作为**纯附加**缓冲实现。
2. **"`--num_envs=20` 时每个环境一行"不成立**：继承来的 `max_init_terrain_level = 5`
   （`isaaclab/terrains/terrain_importer.py:337`：`max_init_level = min(5, num_rows−1)`、
   `terrain_levels = randint(0, max_init_level+1)`）⇒ 20 个环境的初始等级是 **0–5 行随机抽（有放回）**，
   不是 20 行各一个。本类**有意不动** `max_init_terrain_level`（规格未要求，改它就是改设计）。
   要扫完 20 档请逐档传 `--terrain_level=N`；若确实想要"20 环境均匀铺满 20 行"，需要额外的确定性分配
   （Isaac Lab 的课程初始化没有这个语义），属**待决定**项。
3. **规格里的 `prop_start_x`** 实际字段名是 `pattern_start_x`（见 §3）。
4. **`check_reward_overrides.py` 未加 `cmoe-mix-test` 链**：本场景**完全不改奖励**（生效项仍是 `cmoe`
   的 28 项），加一条与 `cmoe` 逐项相同的链没有信息量，故未加（若要加，只需在 `CHAINS` 里补一条）。

## 10. 验证与实测输出

```bash
cd imgo2_rl
/opt/conda/envs/isaaclab/bin/python -m unittest discover -s tests     # 608 通过 / 10 跳过
python3 scripts/tools/check_reward_overrides.py cmoe-gaitfree          # 24 项（cmoe = 28 项，均与改前一致）
python3 scripts/tools/check_terrain_columns.py                         # 末行"全部掩码引用的地形名都有 ≥1 列 ✅"
python3 scripts/tools/check_terrain_columns.py --task mix-test         # 末行"…属预期 ✅"（退出码 0）
```

新增测试 49 项（`test_cmoe_mix_test_scene.py` 35 项 + `TestMixTestTerrainColumns` 14 项）：

* **PD 纯函数**：`y=0,yaw=0 ⇒ vy_cmd=wz_cmd=0`；`y>0 ⇒ vy_cmd<0`；`vy>0 ⇒` 负阻尼；`yaw>0 / wz>0 ⇒`
  负航向指令；**四象限**各一例（位置项与阻尼项同号叠加、反号相消，符号正确）；`vy_max`/`wz_max`
  在正负两侧都被夹住；自定义增益与 `y_des`/`yaw_des` 被真正读取；默认增益常量等于用户给定值。
* **命令项（桩 env + 真实源码）**：`_resample_command` 采样一次 vx 并把横向/航向置 0；
  `_update_command` **不改 vx**、按 PD 写 `vy`/`wz`；`vel_command_w` 旋转镜像正确；
  输出被 cfg 上限夹住；站立环境整条归零；横向误差用 `env_origins` 而非绝对世界坐标。
* **配置级**：继承 `Imgo2CMoERoughPlayEnvCfg`；`sub_terrains.clear()` 后只有 `mix`；
  `num_cols=1` / `num_rows=20` / `num_envs=20`；`lin_vel_x=(1.0,1.0)`、`lin_vel_y=(0,0)`、
  `ang_vel_z=(0,0)`、`heading=(0,0)`；`heading_command=True`、`rel_heading_envs=1.0`、
  `rel_standing_envs=0.0`；`resampling_time_range=(1e9,1e9)`；命令项用 `mdp.MixTestVelocityCommandCfg`；
  **观测/动作契约不变**；mix 地形参数与 `cmoe_terrains.py` 默认值逐一相等。
* **注册级**：`Imgo2-basemove-rough-cmoe-mix-test` 已注册、指向新 cfg、与 `-play` 任务同族。
* **工具级**：按任务区分、默认任务解析**不被 test 场景污染**、原判据负向对照仍报错、
  `MASKED_NAMES` 未被改、CLI `--task` 两种取值都返回 0、未知任务报错。
