# P1/P2 实现与离线验证记录

日期：2026-09-20。状态：**P1/P2 实现已就绪，本机离线检查通过；Isaac Lab 物理验收待训练机执行。用户已明确：本机不运行 Isaac Lab，代码仍面向 Isaac Lab。**

## 1. 本轮实现

目录安排沿用 [架构记录](cart_p1_p2_architecture_2026-09-20.md)：任务在 `manager_based/towing/`，与 locomotion 同级。没有修改现有底层 AMP/PPO 算法或模型参数。

| 文件（相对仓库根） | 内容与状态 |
|---|---|
| `imgo2_description/cart/cart.urdf` | 直接维护的箱体 + 四轮 URDF，四个 continuous joint，挂点为无质量固定坐标系；离线检查通过 |
| `imgo2_rl/source/imgo2_rl/imgo2_rl/assets/cart_model.py` | 标准库读取、几何/拓扑/质量/惯量校验；运行端与离线端复用 |
| `imgo2_rl/source/imgo2_rl/imgo2_rl/assets/cart.py` | Isaac Lab 导入配置与无 actuator 的力矩直通层；替代后端的提交顺序测试通过，PhysX 待验证 |
| `imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/cart_scene_cfg.py` | 已编写独立平地、小车、轮接触传感器场景；未运行 |
| 同目录 `mdp/resistance.py` | 显式黏性轮阻 `tau=-b*omega`，标量/数组接口，负值与非有限 b 拒绝 |
| 同目录 `utils/recording.py` | CSV/JSON 写入、非有限数据和重复时间拒绝、不覆盖已存在目录 |
| `imgo2_rl/scripts/towing/cart_coast.py` | drop/coast、静置检查、匹配初速度/轮速、有限时长扫描、运行状态记录；语法和 `--help` 通过，物理运行待训练机 |
| `imgo2_rl/scripts/tools/cart_coast_metrics.py` | 标准库停止时间/距离、接触比例、滚动残差、异常检测及 SVG 曲线 |
| `imgo2_rl/scripts/tools/summarize_cart_coast.py` | 离线重算 CSV；失败/未知状态的残缺运行不能被重新标为有效 |
| `imgo2_rl/scripts/tools/check_cart_model.py` | 小车独立模型检查，不导入 Isaac Lab |
| `imgo2_rl/tests/test_cart_coast_metrics.py` | 22 项标准库测试，通过 |

本机标准库 Python 可直接执行的命令（从仓库根）：

```powershell
& 'C:/Users/qmq/AppData/Local/Python/pythoncore-3.14-64/python.exe' -m unittest discover -s imgo2_rl/tests -p test_cart_coast_metrics.py -v
& 'C:/Users/qmq/AppData/Local/Python/pythoncore-3.14-64/python.exe' imgo2_rl/scripts/tools/check_cart_model.py
& 'C:/Users/qmq/AppData/Local/Python/pythoncore-3.14-64/python.exe' imgo2_rl/scripts/tools/check_asset_paths.py
```

此解释器为本机已确认路径；其他机器使用其实际 Python。训练机验收命令见第 6 节，尚未在本机执行。通用离线入口可通过 `summarize_cart_coast.py --help` 查看，需输入实际存在的运行目录。

## 2. 工程参数和验收口径

参数来自 URDF，**并非实车标定**：车体 `0.5 × 0.32 × 0.1 m`、8.4 kg；四个实心轮各 0.4 kg、半径 0.08 m、宽 0.04 m；总质量 10 kg。轮轴坐标 x=±0.18、y=±0.20、z=-0.07 m，静止车体高度约 0.15 m。挂点相对车体 `(0.25,0,0)`。车体惯量 `0.07868/0.182/0.24668 kg·m²`，轮惯量由实心圆柱公式计算并校验。

模型 SHA256：`866934f60da3561a21124a4e2973bbac8552a26523bdb8d40f99c2bbb964090c`。

已编写入口的默认扫描为 v0=0.5/1.0 m/s、每轮 b=0/0.008/0.016/0.032 N·m·s/rad，物理步长 0.005 s。这些只是待运行的初值。P2 的 0.5–2 m 目标尚无实际轨迹支持。

停止判定：速度范数低于 0.02 m/s 持续 0.5 s，采用最终持续低速窗口；中途短暂停顿后再运动不会被误判。超时写 `stopped=false`、停止距离为 null，仅给观测距离。记录的力矩对应“结束于该采样时刻的上一物理步”，因此耗散检查使用上一帧轮速，避免离散时间错位。

## 3. 已发现和处理的问题

1. **空 actuator 的力矩通路（适配已测，PhysX 待验）**：阅读本机 Isaac Lab `Articulation._apply_actuator_model()` 发现，仿真力矩缓冲区在 actuator 循环内填充。空 `actuators` 时仅设目标力矩不足以保证写入。`PassiveCartArticulation.write_data_to_sim()` 在父类写入后，通过 PhysX tensor view 提交公开的 `joint_effort_target`；不添加驱动或位置/速度伺服。离线测试执行实际类体，用替代后端验证两个环境的力矩与索引在父类写零之后提交，并验证配置 actuator 时拒绝运行；这不等于真实 PhysX 验证。
2. **URDF 覆盖检查只允许机器人副本**：已增加小车独立校验，并将仓库非忽略的未跟踪 URDF 纳入枚举，保持未知模型报错，不把小车加入机器人 FK 对照组。新小车结构检查通过。
3. **离线停止和异常口径**：已实现并测试，涵盖停止/未停止、重启运动、反向速度、缺接触、翻倒、非有限值、错误力矩方向、CSV 完整性。解析指数衰减轨迹只用于指标单元测试，**不是物理仿真结果**。
4. **完整性和扫描汇总**：即使 status 被标为 completed，离线重算仍核对采样数与时间间隔，截断 CSV 不会被标有效。中断记录 failed。`calibration.json` 汇总各速度的阻力—距离趋势及 1 m/s 下 0.5–2 m 的候选 b；无效/未停止工况不进入候选。候选仅供工程调试，仍需图形检查和 dt 敏感性复核。

## 4. 验证结果和已有失败

解释器：`C:/Users/qmq/AppData/Local/Python/pythoncore-3.14-64/python.exe`，Python 3.14。

- 22 项小车/指标/接口及指纹回归测试全部通过。
- 16 个相关 Python 文件（含测试文件）按字节编译通过；入口 `--help` 通过，无 Isaac Lab 依赖。
- `check_asset_paths.py` 通过：现有机器人 17 个网格引用无缺失、21 份动作数据、新增小车路径/结构通过。
- `check_model_sync.py` **最终整体通过**：机器人轴/限位、质量/惯量/碰撞、网格、FK、三份 URDF 覆盖及小车独立结构检查通过；FK 最差恒等 RMSE 仍为 0.00214 m。
- **MODEL-04 已修复并验证**：初次运行网格指纹为 `bc421eb1cbef`，预期为 `8dc5b5995a11`，未修改版本也有此现象。根因是 `sorted(Path)` 在 Windows 按大小写折叠排序，而 Linux 区分大小写；改为 `key=lambda p: p.name` 后，同一套字节立即得到原预期 `8dc5b5995a11`。网格与预期值均未改动。新增混合大小写文件名测试和篡改字节的负向测试，确认跨平台排序且仍能检出内容变化。

本机 `nvidia-smi` 本轮确认 GPU 为 RTX 4060 Laptop、驱动 591.59，修正“当前机器没有 NVIDIA 驱动”的旧环境假设。但 `isaaclab.bat -p --version` 输出找不到 Python、`_isaac_sim/python.bat` 不存在；批处理还返回 0，不能把退出码当成环境可用。没有进入物理仿真，也没有安装或修改环境。

## 5. 待完成与同步状态

- **P1/P2 验收**：用户明确面向训练机的 Isaac Lab，本机不启动或安装该运行环境。仍需训练机完成导入、落地、轮接触、无驱动滚动、阻力扫描及 dt 敏感性验证，仿真端标为“已写，待验证”。
- P4 的具体 AMP/PPO checkpoint 和接口配置仍待接入前落实；本轮不训练底层、不运行硬件。
- 本轮 `pull --ff-only` 拉取到远端 `13eeb4e`，本地 `aff4c61` 与远端分叉，无法快进，未合并/重置工作区。已检查远端增量：read_tfevents 工具/测试、README 和研究草稿，未改小车相关实现。按用户要求，本次将 P1/P2 实现、离线检查及维护文档作为本地提交保存；尚未合并远端或推送，训练机运行前需完成同步。合并时继续保留本地研究草稿的忽略决定，不能让远端重新跟踪它。用户另有 `research_exploration_plan.md` 的删除，保留在工作区，不纳入本次提交。

## 6. 训练机短时验收（待执行）

以下从仓库根开始，在已经可运行现有 locomotion 的 Isaac Lab Python 环境中执行。入口/路径已静态确认存在，但命令没有在本机启动仿真；无需 checkpoint，也不会开始 RL 训练。

```bash
cd imgo2_rl
python scripts/tools/check_asset_paths.py
python scripts/tools/check_model_sync.py

# P1：单环境落地，3 秒，检查四轮接触、静止高度与姿态
python scripts/towing/cart_coast.py --mode drop --duration 3 --headless

# P2：先单工况跑通，再扫描
python scripts/towing/cart_coast.py --mode coast --velocities 1.0 --damping 0.016 --duration 10 --headless
python scripts/towing/cart_coast.py --mode coast --velocities 0.5 1.0 --damping 0 0.008 0.016 0.032 --duration 10 --headless

# 对选定工况减半步长，复核接触/停止距离的敏感性
python scripts/towing/cart_coast.py --mode coast --velocities 1.0 --damping 0.016 --duration 10 --dt 0.0025 --headless
```

默认每次在 `imgo2_rl/logs/towing/cart_coast/` 下创建唯一目录，控制台打印实际路径；`--output-dir` 可指定一个尚不存在的输出目录。去掉 `--headless` 可在训练机观察模型。`IMGO2_CART_URDF_PATH` 只覆盖小车，保留现有 `IMGO2_URDF_PATH` 的机器人语义。

验收读数：

- P1 看每个 case 的 `summary.json`：`valid=true`、四轮最终接触存在、静止高度接近 0.15 m；检查导入的 `runtime.json` 总质量 10 kg、四个 wheel joint、刚度/阻尼为零。空 actuators 的警告是当前被动模型预期现象，不以添加伺服消警告。
- P2 看 `trajectory.csv`、`coast.svg`、`sweep.json` 与 `calibration.json`。b=0 可以 `stopped=false`；正 b 应耗散且随 b 增大停止距离下降。`valid=true` 只说明轨迹没有触发异常，不代表已经满足目标距离或完成标定。
- 框架运行完成但数据异常时返回 2；正常完成返回 0，包括有效但未停止的零阻力对照。初始化/运行异常保留 failed 状态，不生成伪造的成功结果。
- 对非零阻力，确认速度曲线实际衰减，才能认定力矩在 PhysX 中生效；替代后端单元测试不能替代这一步。
- 离线重算：`python scripts/tools/summarize_cart_coast.py <实际的case目录>`，占位项须换成控制台输出下实际存在的子目录。
