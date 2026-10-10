# 拖曳第三类连接：刚体球铰连杆 + 20×20 场景网格（训练侧，2026-10-08）

> **2026-10-10 变更**：`extra_distance`（`mdp.post_stop_distance`，−0.1）已从奖励表**删除**（函数本体与 `post_stop_allowance_m` 形参保留、目前未接入奖励）；`min_clearance` 的阈值由 `ratio(0.25) × 连接长度` 改为 `spawn_margin(0.85) × 出生间隙`、权重由 −2.0 提到 **−5.0**。本文以下是变更**之前**的记录，现行口径见 [奖励改动记录](towing_reward_retune_2026-10-10.md) 与 README 问题表 TOW-24。

## 现象与范围

用户要求「整体就存在三种类型的环境：弹性绳、低弹性绳、刚体（可提供反驱）」，并明确：

- 刚体连接是**球铰连杆**——固定两挂点距离、允许绕挂点自由转动；
- 连杆长度 0.4–0.8 m；
- **本轮只在训练侧实现**，MuJoCo／Gazebo sim2sim 的第三场景不在本轮。

同日用户进一步指定场景布局：**20 列 × 20 行 = 400 环境**，列 = 连接类型/弹性档
（弹性绳 8 / 刚体 8 / 普通绳 4，即 4:4:2 放大），行 = 长度 0.4→0.8 m 均分 20 档；
**弹性绳只在列上做弹性区分**（4 档 k/c，每档 2 列），其余域随机化保持逐 env 随机。
第一版实现（`3afac61`）是「类型与长度逐 env 随机」，本轮被确定性网格取代。

现状（改动前）：训练侧只有 `compliant`（弹性绳，单边弹簧阻尼）与 `inextensible`
（低弹性绳，单边距离约束）两套模型，逐环境用 `SplitRopeModel` 按 0/1 掩码混合
（`rope_model_id = torch.randint(0, 2, …)`）。两套都是**单边**——只拉不推。

## 关键判断：这是新物理，不是「把绳调硬」

- 绳：`d ≤ L0`、`T ≥ 0`，松弛后不再作用；
- 连杆：`d ≡ L0`、`T ∈ ℝ`，**压缩时把机器人往前推**（反驱），永远啮合。

因此不能复用单边模型的 `max(0, J)`。`RopeSample` 的两个字段语义必须放宽：
`rope_tension` 允许负值（推力），`rope_extension` 保留符号（负值 = 压缩量）。
把 `compliant` 的 `k` 调大只会得到更硬的「拉」，永远得不到「推」。

## 修法

### 1. 物理模型（`mdp/rope_model.py`）

- 新增 `RigidLink`（方案 C）：沿用 inextensible 的「速度级约束 + 位置反馈、显式写成力」，
  但**不取正部**：

      C = d − L0
      target_rate = clamp(−β·C/dt, ±max_correction_rate)
      J = (ḋ − target_rate) / k_eff        # 有符号
      T = J / dt
      F_R = T·e,  F_cart = −F_R            # e：机器人 → 小车

  `C > 0` ⇒ `J > 0` ⇒ 拉；`C < 0` ⇒ `J < 0` ⇒ 推，`F_R` 指向机器人前方即反驱。
  `is_taut` 恒真；`rope_extension` 报有符号的 `C`。`rest_length` 可以是标量或逐环境
  `(N,)` 张量（训练里按 env 随机化）。
- 新增 `MultiRopeModel`：按**整数 id** 逐环境选择 N 个模型。
  **与旧 `SplitRopeModel` 的关键差别**：不要求各模型长度相同，因此 `rope_extension`
  也必须混合（旧实现取其中一套的几何量）。`SplitRopeModel` 保留为两模型特例
  （仍要求两套绳同长），既有调用方与测试不变。
- 注册表：`ROPE_MODELS = ("compliant", "inextensible")` **语义保持为绳**（单边性测试可
  整表套用），新增 `RIGID_MODEL = "rigid"`、`CONNECTION_MODELS` 才是三类全集；
  `make_rope_model` 接受三者。
- 新增纯几何函数 `attachment_horizontal_gap(target_distance, delta_z)`：
  `h = sqrt(target² − Δz²)`，供三类连接的 spawn 摆放使用，本身可离线测试。

### 2. 训练场景（`upper_mdp.py` / `upper_env_cfg.py` / `mdp/connection_grid.py`）

- 新增纯算术模块 `mdp/connection_grid.py`：`COLUMNS=20`、`ROWS=20`、`GRID_SIZE=400`；
  列布局 0-7 弹性绳（4 档 × 2 列）、8-15 刚体、16-19 普通绳；`row_length(row)` 为
  0.4→0.8 均分 20 档；`ELASTIC_KC = ((1000,50), (4000,100), (20000,220), (100000,460))`
  （按 ζ≈0.3 与名义折合质量配对）；`SLACK_RATIO = 0.5`；`env_spec(i)` 行优先映射
  （`列 = i % 20`、`行 = i // 20`；Isaac Sim `GridCloner` 对 400 env 用 20×20，
  `x` 随 `i // 20`、`y` 随 `i % 20`，故同一列落在同一条 x 线、同一行落在同一条 y 线）。
- `upper_mdp.__init__` 按 env index 一次算出 `rope_model_id` / `connection_length` /
  `initial_distance` / `rope_stiffness` / `rope_damping` 五个**逐 env** 张量，之后不再改写；
  三套模型都按这些张量构造（`CompliantRope`/`InextensibleRope` 现在也接受逐 env 数组，
  见 `rope.config_value`）。
- 默认 `num_envs = COLUMNS × ROWS = 400`；`--num_envs` 覆盖成非 400 整数倍只覆盖网格前缀
  （可跑冒烟，但不是平衡设计，会打印 WARN）。
- 弹性绳只按**档**不同：4 档 k/c 各自 2 列，同一档内部完全一致；刚体与普通绳的列是
  同参数重复列。域随机化（质量 5–15、摩擦 0.4–1.2、轮阻 0.008–0.032、速度、停车时间、
  12.5% 无小车）保持逐 env 随机。
- **spawn 摆放对三类统一**：目标三维挂点距 = `0.5 × L0`（绳，留松弛）或 `L`（刚体），
  由 `attachment_horizontal_gap` 解出水平间距，再按机器人**实际** spawn 位姿（含 x/y/yaw
  抖动）把小车的挂点放到机器人正后方。顺序上先按连接长度摆放、再加无小车环境的横向停放。
  这取代了第一版「只在刚体分支里摆放」的写法。
- `min_clearance` 阈值改为**逐 env** `ratio × term.connection_length`，ratio 由 0.40 调到
  **0.25**：短绳行（L0=0.4）spawn 间隙只有约 0.108 m，写死 0.32 m 会让它一开局即满额惩罚。
  0.25 下最短板（绳 target=0.2、刚体 target=0.4）的 spawn 间隙都高于阈值，保持
  「初始不生效」。

### 3. 测量台与离线工具

- `tow_drag.py`：`--rope-model` 增加 `rigid`；逐 env 混合改用 `MultiRopeModel`。
  连杆长度取**初始挂点距**（`--rope-length − --slack`，默认 0.4 m）而不是绳长——杆是刚性的，
  按绳布局留 0.4 m 松弛等于把它压缩 0.4 m，开局会猛弹。写进产物的
  `rope.model_note`／`rest_length_m` 都是实际用值。
- `scan_towing_boundary.py`：`--rope-model` 同步加 `rigid`。
- `summarize_tow.py`：`rope.model == "rigid"` 时跳过弹性诊断（没有 k/c），
  避免拿 config 里残留的 compliant 默认 `k=4000` 算出一组假的弹性指标。

## 验证

本机解释器：`/opt/conda/envs/isaaclab/bin/python3`（Python 3.11.13，有 numpy 与 torch）。

| 检查 | 结果 |
|---|---|
| `RigidLink` 双边性（拉长拉／压缩推／反驱方向／等大反向／沿杆分离被拉回、靠拢被推开／长度张量逐 env） | 实跑通过 |
| `MultiRopeModel` 三模型逐 env 选择 + 长度不同时 `rope_extension` 混合 | 实跑通过 |
| `attachment_horizontal_gap` 勾股解（`hypot(h, Δz) == target`；Δz=0 时 h=target；越界钳 0） | 实跑通过 |
| 网格形状：20 列 8/8/4、20 行等距 0.4→0.8、弹性 4 档 × 2 列、行优先映射全覆盖一次 | 实跑通过（`test_towing_connection_grid.py`） |
| 四档弹性在最坏质量（5 kg 小车）下的显式弹簧步长上限 > 5 ms；最硬档 k=1e5 用 1-D 两体积分器真跑 10 s 不发散 | 实跑通过（解析上界 + 实际积分） |
| 逐 env 张量复刻 `upper_mdp` 构造：400 env 的类型计数 160/80/160、20 个长度、**spawn 时最大 \|T\| = 0**；抽查 9 个 env 与单模型逐位一致（float32 相对误差 < 1e-6） | 独立脚本通过（已删除临时脚本） |
| spawn 摆放数值复核（float32 下 `\|d − target\| ≤ 3e-8`，含 yaw 抖动） | 独立脚本通过（已删除临时脚本） |
| 拖曳离线测试 | **266 项通过** |
| 全量 `imgo2_rl/tests` | **316 项通过、0 跳过、0 失败** |
| `compileall` 改动文件、`git diff --check`、tracked-ignore 为空 | 通过 |
| `check_asset_paths.py`、`check_model_sync.py`、`check_towing_mjcf.py` | 通过 |

## 待确认与完成条件（缺什么才能完成）

1. **Isaac Lab 实跑**：本机无 Isaac Sim 运行时（`isaaclab.utils.math` 因缺 `omni.log`
   不可导入），刚体 env 未跑过。需在训练机跑冒烟并确认：400 环境能构造、每列的
   `model_index` 与 `connection_length` 实际生效、`MultiRopeModel` 逐 env 结果与单模型
   逐位一致、以及 12.5% 无小车掩码在网格下仍正确隔离。
2. **初始落地瞬态**：机器人按 `IMGO2_CFG` 的 0.35 m 出生后下落约 8 cm，而刚性杆从第一步
   就啮合，挂点高差变化会转成一次推挤。需实测该瞬态幅度，判断是否需要为刚体 env 单独
   提高出生高度或加大 `rigid_max_correction_rate` 的取舍。文档不预设它会「温和」。
3. **停车反驱**：`post_stop_towing_force` 用的是力范数，压缩也会被惩罚——这正是要观察的
   行为，但「惩罚大小是否合理、会不会把策略逼成『贴住小车』」需要运行数据与 STOP reward
   排序一起评价。
4. **短连接的几何拥挤**：最短绳行 `L0=0.4` 时 spawn 间隙只有约 0.108 m（阈值 0.10 m），
   机器人后腿与车斗是否真的不接触、以及 `clearance_barrier`（绝对 0.20 m 警戒）在短连接
   行是否长期激活，都需要实跑数据；`min_clearance` 现已逐 env 用连接长度，但
   `clearance_barrier` 仍是绝对量，未改。
5. **MuJoCo／Gazebo 第三场景**：本轮用户明确只在训练侧实现。sim2sim 对照刚体连杆还需要
   `mjcf/scene_tow_rigid.xml`（双边约束：连杆 body + 两个 `connect` equality，或
   `range="L L"` 的限位 tendon）与 `rl_sim_mujoco.cpp` 的场景名判断，见 TOW-05。
6. **真刚度边界**：本模型是外力等效约束，稳态会留 `C ≈ T·k_eff·dt²/β` 的柔度
   （名义参数、T≈10 N 时约 7 mm），**不是无穷刚度**。要真正刚性需要 PhysX 关节级约束，
   而机器人与小车是两个独立 articulation；本轮要的是「双边性（反驱）」，不得在文档或
   结论里把它写成「理想刚体」。网格最硬档 k=1×10⁵ N/m 仍比真实 6 mm 涤纶
   （k≈6e4–3e5 N/m）软，再硬需改用约束模型或子步。
