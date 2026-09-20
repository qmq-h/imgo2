# P1/P2 小车建模与滑行标定：文件架构方案

日期：2026-09-20。状态：**设计方案，尚未实现或运行 P1/P2**。

依据：用户更新的本地 `paper_plan_imgo2.md`（SHA256 `7961bae8e6da38f8368c8436cb007c140c29e3b9ffa1b09114527123428548c9`）以及同步至 `40cb298` 的源码。该草稿按用户要求保留在本地并从 Git 跟踪中移除；本记录独立保存实施所需的架构约定。本轮只安排文件职责与接口，保留用户计划原文。

## 1. 范围与组织方式

P1 建立车体、四个被动轮和挂点；P2 在没有机器人的平地场景中赋初速度并标定滑行阻力。采用 `SimulationContext + InteractiveScene` 的独立实验入口，暂不注册 Gym 任务，不需要 reward、policy、runner 或训练配置。

代码分三层：`assets/` 管小车本身，包内 `towing/` 管可复用的场景与动力学，`scripts/towing/` 管实验流程。这样 P3/P4 可以直接复用资产、阻力和记录模块。现有 `tasks/manager_based/locomotion/velocity/` 继续负责底层 locomotion；到 P10 再新增独立上层任务目录。

## 2. 拟新增文件

**下列路径均为拟新增，当前不存在，不是可直接运行的入口。** 常规包目录配空的 `__init__.py`，其中不启动仿真、不生成文件。

```text
imgo2_rl/
├── source/imgo2_rl/imgo2_rl/
│   ├── assets/
│   │   ├── cart.py                   # 参数定义、ArticulationCfg 工厂
│   │   └── cart_usd.py               # 根据参数创建 USD 刚体、碰撞与关节
│   └── towing/
│       ├── cart_scene_cfg.py         # 平地 + 小车的独立测试场景
│       ├── resistance.py             # 轮轴阻力计算与施加
│       └── recording.py              # CSV 与运行元数据写入
├── scripts/
│   ├── towing/
│   │   └── cart_coast.py             # P1 drop / P2 coast 两种模式
│   └── tools/
│       └── summarize_cart_coast.py   # 标准库离线汇总、停止判定与 SVG 曲线
└── tests/
    └── test_cart_coast_metrics.py    # 停止/未停止/异常轨迹的指标测试
```

| 文件 | 负责 | 边界 |
|---|---|---|
| `assets/cart.py` | 车体长宽高、轮半径/宽度、轮距/轴距、各刚体质量、挂点局部坐标、初始姿态；创建新的资产配置 | 几何与质量只有一份参数源；总质量明确为车体加四轮，不把总质量再次写到车体 |
| `assets/cart_usd.py` | 方箱车体、圆柱轮、惯量、碰撞、四个 revolute joint、articulation root、挂点 frame | 无驱动目标；关节 stiffness/damping 初始为零；仿真启动后才导入/调用 USD API；可导出快照 |
| `towing/cart_scene_cfg.py` | 地面接触材质、环境间距、小车配置与出生位置 | 与现有 locomotion 的地形、随机化配置分开；时间步由入口配置并记录 |
| `towing/resistance.py` | 四轮角速度 → 每轮阻力矩；每个 physics step 写入 | 首版只做 `tau = -b * omega`；不同时启用 USD damping 和显式黏性阻力，避免重复计入 |
| `towing/recording.py` | 固定字段 CSV、完整参数 JSON、代码版本与运行状态 | 不计算动力学；先使用标准库，不引入 TensorBoard/W&B；P5 增加机器人与绳索字段 |
| `scripts/towing/cart_coast.py` | CLI、AppLauncher、场景创建、落地稳定、初速度设置、推进与清理 | 所有物理细节调用包模块；有限运行时长；先单环境验证 |
| `scripts/tools/summarize_cart_coast.py` | 读 CSV/JSON，输出停止距离、停止时间、残余速度及曲线 | 仅标准库，不导入 `imgo2_rl` 或 Isaac Lab，可在普通 Python 离线运行 |

现有 `imgo2_rl/__init__.py` 会导入任务注册，因此所有仿真入口应先启动 `AppLauncher`，再导入项目包。离线汇总直接作为脚本运行，避免意外触发 Isaac Lab 导入。

## 3. 模型源码与生成物

第一版使用程序化 USD：`cart.py` 的参数与 `cart_usd.py` 的构造逻辑共同作为源码，在 Stage 中生成模型。简单箱体与圆柱不需要网格、URDF 转换器或 CAD 文件，也无需增加机器人模型副本。

每次实验可导出该小车的 `cart.usda` 快照到运行目录，与解析后的参数一起留存；不要求先手工生成一个固定位置的 USD 才能启动。将来若需要 CAD 或跨仿真器模型，再决定外部资产的统一存放位置。

建议的 USD 结构：

```text
Cart                              # Articulation root 所在位置由运行验证确定
├── base_link                     # 刚体：车体质量/惯量/碰撞
│   └── rope_attachment           # 无质量 Xform，仅定义挂点
├── wheel_fl                      # 刚体：圆柱轴沿 y
├── wheel_fr
├── wheel_rl
├── wheel_rr
└── joints
    ├── wheel_fl_joint            # base_link ↔ wheel_fl，转轴沿 y
    ├── wheel_fr_joint
    ├── wheel_rl_joint
    └── wheel_rr_joint
```

坐标统一为 x 前、y 左、z 上。四轮关节正方向一致，按名称解析运行时索引。挂点位置相对 `base_link` 定义，P3 由它求世界位置和力臂；它不是第六个带质量的刚体。

第一版采用浮动底座、平整地面、沿 x 的初速度，记录横向偏移与 yaw。若确需严格一维约束，应作为独立实验配置记录，不能每步覆盖 y/姿态冒充自由滚动；P1 的落地检查必须保留垂向自由度。

## 4. P2 实验接口与验收

入口提供 `drop` 与 `coast` 模式。`drop` 检查重力落地、轮地接触和四个自由轮；`coast` 先静置稳定，再将小车平移速度设为 0.5 或 1.0 m/s，同时按转轴符号设置无滑动初始轮速 `omega = v0/r`，以减少“车已动、轮未转”的起始滑移。t=0 与 x0 在赋速后记录。

物理推进顺序：读取轮速 → 计算/设置阻力矩 → `scene.write_data_to_sim()` → `sim.step()` → `scene.update(dt)` → 记录新状态。后续 rope 也接在每个物理步施力阶段，不挂到低频的 command 更新中。

首版每轮相同黏性系数 b，单位 N·m·s/rad，允许 b=0 作对照。驱动力矩为零，施加的只有耗散阻力。接触摩擦负责轮地牵引，不用它代替可控的轮轴阻力；车体额外线性/角阻尼应明确设定并记录，防止隐藏阻力污染标定。

最小 CSV 字段：`time_s, cart_x_m, cart_y_m, cart_z_m, cart_vx_mps, cart_vy_mps, cart_vz_mps`、姿态、四轮 `omega_radps` 与 `resistance_nm`。运行元数据包含实际质量/惯量、轮尺寸、b、材质、时间步、求解器、初速度、停止判据、Isaac Lab/Sim 版本、代码提交号及工作区是否有修改。

**纯黏性阻力渐近衰减，停止是测量判据。** 建议首版以平移速率低于 0.02 m/s 且持续 0.5 s 判为停止，同时报告最大残余轮速；这些阈值必须进元数据。停止距离为合格低速窗口起点相对 x0 的位移。超时仍未停止时写 `stopped=false`，距离是截断观测值，不能伪报为停止距离。异常翻倒、非有限状态或明显横向运动单列为无效工况。

计划中的 0.5–2 m 仅用于 **v0=1.0 m/s** 的初版工程调试，不能直接要求 0.5 m/s 工况也满足同一区间，更不能写成真实脚轮参数。P2 应展示 b=0/低/中/高阻力的曲线，并检查耗散方向及更大 b 是否带来更短滑行距离；必要时将 dt 减半复查选定工况。

## 5. 产物与后续扩展

拟使用的输出目录（首次运行时创建）：

```text
imgo2_rl/logs/towing/cart_coast/<run_id>/
├── config.json
├── cart.usda
├── trajectory.csv
├── summary.json
└── coast.svg
```

路径从脚本位置推导，允许 `--output-dir` 覆盖。根 `.gitignore` 已忽略 `logs/` 和 `*.usda`，无需为本方案新增忽略规则；经验证的简短结论和选定离线结果放 `docs/`。将来若跟踪二进制 USD，再同步 `.gitattributes`。

- P3：在 `towing/` 增加 `rope.py`，复用挂点接口，成对施加张力及偏心力矩。
- P4–P9：增加机器人与小车组合场景、冻结底层策略的适配器，以及相应实验入口；保留 `cart_coast.py` 作为独立标定工具。拟新增 `towing/low_level_policy.py` 与 `towing/policy_cfg.py`，统一接收速度指令、输出底层动作，支持选择已有 AMP/PPO checkpoint。各策略分别保留观测顺序/缩放、历史、动作缩放、控制周期与 reset 契约，不直接假设接口相同；环境物理模块不判断算法名称。
- P10：再建立 `tasks/manager_based/towing/`，把上层 action/reward/observation 接到已有物理模块。避免提前复制 AMP/PPO runner 或把拖曳逻辑塞入公共 `velocity_env_cfg.py`。

## 6. 已完成、待实现与验证限制

**本轮已完成**：读取新版计划、核对现有包导入/任务注册/日志规则，形成文件职责方案，修正架构层面“被动轮等于无阻力”“零速必须严格等于零”的潜在歧义。无代码缺陷修复，无 P1/P2 验收结果。

**待实现/确认**：

1. 实现上述文件并在 Isaac Lab 中完成 drop；缺资产生成器、场景与运行证据。
2. 实现阻力与 coast，验证 0.5/1.0 m/s 曲线；缺仿真结果，b 暂不能标为已标定。
3. 车体尺寸、轮径、质量和挂点位置目前没有实测值；可先用显式标注的工程默认参数，实车参数到位再替换。
4. 当前机器的 Isaac Lab articulation 源码对未配置 actuator 的关节发出 warning，但静态阅读不能证明空 `actuators` 在实际运行中可用；实现时验证无驱动、非零阻力矩确实写入。禁止为消警告而把轮变成速度伺服。
5. 用户已确认使用现有 AMP/PPO 训练结果作为冻结底层：**AMP 低速速度跟踪更好，PPO 运动更好看**（用户反馈，非本轮新增测量）。P4 先支持两者切换，按相同负载工况分别记录 `policy_type`、checkpoint 路径/哈希、配置和指标，不把两种底层的结果混为同一组。P1/P2 无机器人依赖；P4 前仍需落实具体 checkpoint 与配套机器人配置，计划中的 Go2 称呼不作为自动切换机器人模型的依据。

官方 [actuator 文档](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.actuators.html) 区分驱动增益与关节摩擦参数，实际 API 以运行环境版本为准。本轮参考本地 `articulation.py` 与 `scripts/tutorials/02_scene/create_scene.py` 静态阅读，未启动 Isaac Sim、未训练、未改模型或算法。

文档验证：使用本机 `C:/Users/qmq/AppData/Local/Python/pythoncore-3.14-64/python.exe` 检查本记录 UTF-8 编码及本地链接通过；`git diff --check` 通过。计划移出索引后，本地 SHA256 与上文一致，`git check-ignore` 命中，`git ls-files -i -c --exclude-standard` 为空。工作区另有用户删除的 `research_exploration_plan.md`，本次不将该删除纳入提交；README 中其旧入口需在处理该独立删除时一并清理。
