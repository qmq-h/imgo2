# P1/P2 小车建模与拖曳任务架构

日期：2026-09-20。状态：**P1/P2 已实现并在训练机完成物理验收（见 [检查与验收记录](cart_p1_p2_checks_2026-09-20.md)）；P3 绳索力已实现、离线验证通过并在 P4 拖曳中被真实使用（见 [P3 记录](towing_p3_rope_2026-09-20.md)）；**P4 核心行为已在训练机验证（0.5 m/s 拖曳成立）**，剩余 `v_cmd=1.0 m/s`、`m_L` 包线扫描与 PPO 契约。** 当前完成项、验证与限制统一见 [实现记录](cart_p1_p2_implementation_2026-09-20.md)。

## 1. 已确定的安排

小车直接维护 URDF，不增加 xacro 或程序化 USD 构造器，运行时由 Isaac Lab 导入生成 USD。拖曳任务集中在 `tasks/manager_based/towing/`，与 `locomotion/` 同级，不另建包根级 `imgo2_rl/towing/`。

已有 AMP/PPO 训练结果作为冻结底层。用户反馈 AMP 低速跟踪更好、PPO 运动更好看；后续支持切换并分别记录结果。本轮没有新增测量。P1/P2 独立验证小车，不需要 RL。

本记录替代此前 xacro 和程序化 USD 提案。本地研究草稿 `paper_plan_imgo2.md` 按用户要求由 Git 忽略，实施所需约定以本文为准。

## 2. 文件安排

下图 P1–P4 文件现已创建；P10 文件仍为后续路径。另补充 `assets/cart_model.py` 与 `scripts/tools/cart_coast_metrics.py` 供标准库验证复用。P1/P2 仿真入口、P3 的 `mdp/rope.py`（经 P4 拖曳运行使用）与 P4 的拖曳入口都已在训练机验证；P4 剩余 `v_cmd = 1.0 m/s` 与 `m_L` 包线扫描。另补 `scripts/tools/summarize_tow.py` 供拖动判读离线复算。

```text
imgo2_description/
└── cart/
    └── cart.urdf                         # 直接维护的小车模型源

imgo2_rl/
├── source/imgo2_rl/imgo2_rl/
│   ├── assets/
│   │   └── cart.py                       # URDF 路径、导入与 ArticulationCfg
│   └── tasks/manager_based/
│       ├── locomotion/                   # 已有底层运动任务
│       └── towing/
│           ├── __init__.py               # 已建；后续任务注册
│           ├── cart_scene_cfg.py         # P1/P2 平地与小车场景
│           ├── towing_env_cfg.py         # P4 机器人与小车组合环境（已建）
│           ├── mdp/
│           │   ├── __init__.py           # 已建
│           │   ├── resistance.py         # P2 轮轴阻力
│           │   └── rope.py               # P3 绳索张力与力矩（已建，离线验证通过）
│           ├── agents/
│           │   ├── __init__.py           # 已建
│           │   └── rsl_rl_ppo_cfg.py     # P10+ 上层 PPO 训练配置
│           └── utils/
│               ├── __init__.py           # 已建
│               ├── recording.py          # P2 起：轨迹、参数与状态（P4 增拖曳记录）
│               ├── policy_cfg.py         # P4 底层 checkpoint 与接口契约（已建）
│               └── low_level_policy.py   # P4 AMP 冻结策略适配（已建；PPO 待做）
├── scripts/
│   ├── towing/
│   │   ├── cart_coast.py                 # P1 drop / P2 coast 实验入口
│   │   └── tow_drag.py                   # P4 机器人拖曳小车实验入口（已建）
│   └── tools/
│       └── summarize_cart_coast.py       # 标准库离线指标与 SVG 曲线
└── tests/
    ├── test_cart_coast_metrics.py        # P1/P2 指标与接口测试
    ├── test_towing_rope.py              # P3 绳力测试
    ├── test_towing_policy_contract.py   # P4 冻结策略契约测试
    └── test_towing_tow_drag.py          # P4 拖曳入口测试
```

`agents/` 将来训练上层策略，不重训底层 AMP/PPO。CLI 启动脚本仍放 `scripts/`，可复用任务定义在 `manager_based/towing/`；P10 根据实际 wrapper 选择现有训练入口或补专用入口，当前不复制算法 runner。

现有 `tasks/__init__.py` 通过 `import_packages` 递归发现包，并排除 `.mdp` 和 `utils`。无需改 `locomotion/__init__.py`。新初始化文件仅声明职责，不导入不存在的实现、不注册虚假的 Gym ID。仿真入口先启动 `AppLauncher`，再导入项目包。

## 3. 模型约定

`imgo2_description/cart/cart.urdf` 是几何、名义质量、质心和惯量的唯一源。第一版用 box 车体、cylinder 车轮，不需要网格。尺寸尚无实测值，实施时使用明确标注的工程默认值。

- x 前、y 左、z 上；车体与四轮共五个有质量刚体，轮轴沿 y，用四个 `continuous` joint，运行时按名称解析索引与核对转向。
- 底座浮动；禁用 cylinder→capsule 替换；关闭驱动或增益置零，URDF damping/friction 初始为零，P2 阻力由代码单独施加。
- 挂点用空 link + fixed joint 表达坐标系，不加虚假质量。解析并保存相对车体的变换，施力到车体及对应力臂，不能依赖固定关节合并后仍存在独立挂点 body。
- `assets/cart.py` 负责路径、导入与初始状态，不复制几何参数。路径由 `Path(__file__)` 推导，可用环境变量覆盖；实验质量覆盖需明确惯量处理并记录。
- 地面材质在小车场景单独配置，不改已有 locomotion 公共场景。

`check_model_sync.py` 已增加小车独立结构/物理检查，Imgo2 保留一致性/FK 检查，并继续拒绝未登记 URDF。小车不加入机器人参数对照组。MODEL-04 的跨平台排序问题已修，全模型检查通过，依据见实现记录。

## 4. P1/P2 实验

`cart_scene_cfg.py` 配置平地与小车；`cart_coast.py` 使用 `SimulationContext + InteractiveScene` 有限时长推进，先单环境。文件放入 manager_based 目录不意味着单体标定必须使用 `ManagerBasedRLEnv`。

- **drop**：验证重力落地、轮地接触、四个自由轮和姿态稳定。
- **coast**：静置稳定后设 0.5/1.0 m/s 初速度，同时按轮半径与轴向设置匹配轮速，减小起始滑移。赋速后定义 t=0、x0。
- **阻力**：每轮 `tau=-b*omega`，b 单位 N·m·s/rad，保留 b=0 对照；不重复启用 URDF damping，不用接触摩擦代替轮轴阻力，显式记录车体额外阻尼。
- **推进**：读轮速 → 算/设阻力矩 → `scene.write_data_to_sim()` → `sim.step()` → `scene.update(dt)` → 记录。P3 绳索也在物理步施力阶段接入。
- **停止**：建议平移速率低于 0.02 m/s 持续 0.5 s，并报告残余轮速，阈值入元数据。纯黏性阻力渐近衰减，不要求数学上的严格零速。
- **距离**：合格低速窗口起点相对 x0 的位移。超时写 `stopped=false`，只报告截断距离；翻倒、非有限状态和明显横向运动单列异常。

1.0 m/s 下滑行 0.5–2 m 仅是工程目标，不是真实脚轮标准，不要求 0.5 m/s 满足相同范围。至少比较零/低/中/高阻力的耗散方向和距离趋势，必要时 dt 减半复核。不能通过逐步覆盖横向位置或姿态来冒充自由滚动。

## 5. 记录与后续接口

拟输出到 `imgo2_rl/logs/towing/cart_coast/<run_id>/`，首次运行时创建，已有忽略规则覆盖。记录 `config.json`、`trajectory.csv`、`summary.json`、`coast.svg`，可保存实际导入 USD 快照。

CSV 含时间、位置、线速度、姿态、四轮角速度与阻力矩；元数据含模型哈希、实际质量/惯量、轮尺寸、b、材质、dt、求解器、初速度、停止判据、软件及代码版本。离线汇总脚本只用标准库，不导入项目包。

P4 起 `utils/low_level_policy.py` 接收速度指令并生成底层动作，`utils/policy_cfg.py` 保存各策略 checkpoint 路径/哈希、观测顺序/缩放、历史、动作映射/缩放、控制周期与 reset 契约。冻结两种底层，按相同负载工况分别记录 `policy_type` 与结果，不默认接口相同。具体 checkpoint 与机器人配置在接入前落实，P1/P2 不依赖这项信息。

## 6. 已完成与待验证

本节以下保留目录骨架阶段的验证依据；**当前状态以 [实现与验证记录](cart_p1_p2_implementation_2026-09-20.md) 为准**。

**骨架阶段已完成**：直接 URDF 与目录安排确定，建立 `manager_based/towing/` 和 `mdp/agents/utils` 初始化文件，清理冲突提案。

**P1/P2 验收已完成（2026-09-20，训练机）**：导入/落地/滑行/阻力扫描/dt 复核全部通过，空 actuator 力矩直通层已在真实 PhysX 中确认（结果与解析黏性模型吻合 1% 以内），详见 [检查与验收记录](cart_p1_p2_checks_2026-09-20.md)。CART-01 已收口；遗留的是另两条与本架构无关的缺陷登记（CART-02 入口退出码、CART-03 离线重算报错）。

**P3 已实现、仅离线验证**：`mdp/rope.py` 提供单侧弹簧阻尼张力、力对与力臂力矩，31 项标准库测试通过（含动量守恒、能量不增、松弛段不做功、numpy/torch 后端一致、`mdp` 包脱离仿真器可导入）。见 [P3 记录](towing_p3_rope_2026-09-20.md)。

**P4 已实现，核心行为已在训练机验证（0.5 m/s 拖曳成立）**：`utils/policy_cfg.py`（AMP 45 维契约，与部署 yaml 逐项交叉核对）、`utils/low_level_policy.py`（TorchScript 适配器）、`towing_env_cfg.py`（机器人+小车场景，只改 `prim_path`）、`scripts/towing/tow_drag.py`（拖曳入口），另有 57 项离线测试（P4 三个文件：契约 24 + 入口 23 + 判读 10）。**尚未做**：`v_cmd = 1.0 m/s`、质量扫描 `m_L = 5–25 kg`（§4/§5 的要求）、PPO 底层契约与按 `policy_type` 分别记录。见 [P4 上半](towing_p4_policy_contract_2026-09-20.md) 与 [P4 下半](towing_p4_tow_drag_2026-09-20.md)。

**当前待验证（P4+）**：在训练机**串行**执行拖曳入口，按 `summary.json` 判「稳定拖曳」，并确认 `--spawn-height` 与绳参数 k/c/L0；之后才是质量扫描、PPO 契约与上层策略。

验证使用 `C:/Users/qmq/AppData/Local/Python/pythoncore-3.14-64/python.exe`（Python 3.14）：四个新文件编译通过；调用本机 Isaac Lab 的实际 `import_packages` 辅助函数对隔离包做发现/导入，通过，并确认 agents 被导入、mdp/utils 按黑名单跳过。文档 UTF-8 与本地链接检查通过；`git diff --check` 通过，`git ls-files -i -c --exclude-standard` 为空。骨架聚合 SHA256 为 `38af375746e0e6ca53f3ae65c04de6889b145c2bb1ef7da37008b9932d82d743`（按排序的相对路径、NUL、文件内容依次拼接计算）。

`check_asset_paths.py` 通过：现有机器人 URDF 存在、17 个网格引用无缺失、21 份动作数据、无残留机器绝对路径；这只证明已有资产路径有效，不是小车模型验证。未运行完整任务注册、Isaac Sim、训练、部署或硬件。用户删除的 `research_exploration_plan.md` 保持原状，本轮不处理该独立删除及其既有入口。
