# P4（下半）：机器人 + 小车 + 绳的拖曳场景与实验入口

日期：2026-09-20。状态：**已实现，离线验证通过；物理验收待训练机（由用户执行）**。
对应研究计划 `paper_plan_imgo2.md`「P4：接入你现有 locomotion」的运行部分，
上半（冻结策略契约与适配器）见 [P4 上半记录](towing_p4_policy_contract_2026-09-20.md)。

## 1. 本轮实现

| 文件（相对仓库根） | 内容与状态 |
|---|---|
| `.../towing/towing_env_cfg.py` | 平地 + 机器人（训练侧 `IMGO2_CFG`）+ 被动小车的场景配置；挂点常量 |
| `.../towing/utils/recording.py` | **扩充**：`TowRecorder` + `TOW_FIELDS`（沿用 `CartRecorder` 的严格校验） |
| `imgo2_rl/scripts/towing/tow_drag.py` | 拖曳入口：冻结策略按速度指令驱动机器人，经绳拖曳小车，逐物理步记录 |
| `imgo2_rl/tests/test_towing_tow_drag.py` | 16 项测试（布局算术、参数校验、记录器、接口契约） |

**没有新建任何目录**，全部落在既有目录内。

## 2. 场景与布局约定

- 机器人在前、小车在后，两者都朝 **+x**；速度指令为正即向前拖。
- 机器人挂点：base 坐标系的 `(-0.16, 0, 0)`。依据：base 碰撞箱长 0.315 m
  （`urdf/imgo2.urdf`）⇒ 后表面在 x = −0.1575 m，取 −0.16 略过后表面。
  **工程取值，没有实测挂点**；换实物尺寸只改 `ROBOT_ATTACHMENT_OFFSET_M`。
- 小车挂点：`cart.urdf` 的 `rope_attachment`，车体坐标 `(+0.25, 0, 0)`（前轴之前）。
  小车在后时该点朝向机器人 ⇒ 受力方向与拖曳一致。
- 初始挂点间距 = `L0 + slack`（默认 1.0 + 0.05 m）。`initial_cart_x()` 实现该算术，
  测试用**场景常量 + `cart.urdf` 实读挂点**反解校核：间距必须等于 `L0 + slack`
  （差一点初始张力就错了，所以这条专门锁住）。
- 机器人初始高度用显式参数 `--spawn-height`，默认 `ROBOT_SPAWN_HEIGHT_M = 0.35 m`
  （= 训练侧 `IMGO2_CFG.init_state.pos` 的 z）。README 记录过「训练多数从录制帧
  ≈0.297 m 起、play 从 0.35 m 起」可能对应**高/低两个稳定站高分支**，因此这里不写死，
  由训练机确认哪一个给出可复现的稳定拖曳。
- 关节初始角按契约的 `default_dof_pos` 给（与 `IMGO2_CFG` 一致），不使用随机初始状态。

## 3. 每物理步的顺序与力臂处理

与 P1/P2 相同（显式欧拉，力的口径一致）：

```text
按当前状态算绳力（P3 的 rope_state）与轮阻（-b·ω）
  → robot/cart.set_external_force_and_torque(...) / cart.set_joint_effort_target(...)
  → scene.write_data_to_sim() → sim.step() → scene.update(dt) → 记录
```

- **力臂交给仿真**：`set_external_force_and_torque(positions=...)` 的作用点坐标是**连杆坐标系**
  （`articulation.py` 的 `apply_forces_and_torques_at_position` 真正消费它），所以直接把挂点
  常量传进去，由 PhysX 自己算力矩，不手写 `offset×F`，也不用纠结 CoM 约定
  （实测 `base` 的惯性原点偏移 < 3 mm）。**整套调用必须用同一个 `is_global` 口径**，
  否则 Isaac Lab 会警告并改变全体的作用坐标系；这里统一用默认的连杆系。
- 力本身从世界系转到连杆系用 `quat_apply_inverse`。
- 绳数学走 `rope.py` 的**标量路径**（`num_envs = 1`），即已经被 31 项测试覆盖的那条路径。
- 策略抽帧：`decimation = round(control_dt / dt)`，契约要求 `control_dt` 是物理 `dt` 的整数倍，
  不满足直接报错。默认两者都是 0.005 s ⇒ 1:1。
- 启动时会核对：关节数、**关节名与顺序**是否与策略契约完全一致、`base`/`base_link` 唯一、
  小车四关节与四轮接触；不一致立即报错（而不是跑到一半形状不匹配）。

## 4. 运行与判据

```bash
cd imgo2_rl
bash scripts/run_isaaclab.sh scripts/towing/tow_drag.py --velocity 0.5 --duration 5 --headless
```

产物在 `imgo2_rl/logs/towing/tow_drag/<run_id>/`：`config.json`（含策略/模型哈希、绳参数、
抽帧、挂点）、`tow.csv`（逐物理步）、`summary.json`、`experiment.json`。
`--output-dir` 只能指向尚不存在的目录。

`summary.json` 的判据（计划 P4 的「稳定拖曳」）：

| 字段 | 含义 | 期望 |
|---|---|---|
| `steady_robot_vx_mps` / `steady_load_vx_mps` | 稳态窗（指令阶段后半段）内两体平均 vx | 两者接近，且接近 `v_user` |
| `steady_speed_gap_mps` | 两者之差 | 判据阈值 0.1 m/s |
| `steady_tension_n` / `steady_tension_std_n` | 稳态窗张力均值/标准差 | 张力显著为正且波动小（阈值 25% 均值） |
| `steady_distance_m` | 稳态窗绳长（挂点间距） | 略大于 `L0`（伸长 = T/k） |
| `tension_slack_fraction` | 指令阶段中 T = 0 的步数占比 | 持续松弛说明绳太松或拖曳失败 |
| `max_abs_pitch_rad` | 全程最大 |pitch| | 阈值 0.6 rad |

`failures` 为空 ⇒ `valid = true`、退出码 0；有失败项 ⇒ 退出码 2。

**退出码说明**：拖曳入口在失败路径上**显式 `os._exit(2)` 且不调用 `application.close()`** ——
这是 CART-02 的教训（Kit 关停会直接终止进程、把异常与退出码一起吞掉，实测三次尝试退出码都是 0）。
新入口因此不复制那个缺陷；`cart_coast.py` 的 CART-02 仍未修。

**操作前提：Isaac Sim 必须串行执行**（并发实例争锁会中断，见 P1/P2 检查记录 §3.5）。

## 5. 离线验证

| 验证 | 结果 |
|---|---|
| `python3 -m unittest test_towing_tow_drag` | **16 项通过** |
| 全量离线测试（11 个测试文件） | **178 项通过**（本轮之前 162 项） |
| `check_model_sync.py` / `check_asset_paths.py` / `check_cart_model.py` | 退出码均为 0 |
| `tow_drag.py --help`（只用标准库） | 退出码 0，不需要 Isaac Lab |

覆盖：初始间距算术（用场景常量与 `cart.urdf` 实读值反解）、后挂点在机体后表面之后、
小车挂点朝向机器人、参数校验、记录器的表头/非有限值/时间不递增/拒绝覆盖、以及接口契约
（`positions=` 作用点、`write_data_to_sim()` 必须早于 `sim.step()` 且晚于施力、失败路径
`os._exit(2)` 且不调用 `application.close()`、场景配置只改 `prim_path` 不碰增益/观测/奖励、
入口不导入 `velocity_env_cfg`/`amp_env_cfg`、**初始位姿必须在 `InteractiveScene(...)` 之前
写进配置**）。

**测试抓到的两个真实缺陷（都已修）**：

1. **`--damping 0` / `--slack 0` 被错拒**：第一版把这两个量放进「必须为正」的校验循环，
   于是一个合法的无阻尼绳（`c = 0`）和初始就绷直（`slack = 0`）都过不了参数校验。
   已拆成「严格为正」与「允许为 0」两组。
2. **初始位姿改晚了**：第一版在 `InteractiveScene(scene_cfg)` **之后**才写
   `robot_cfg.init_state.pos`。场景在构造时就把资产按 `init_state` 摆好，之后再改配置
   不生效 ⇒ **`--spawn-height` 会被静默忽略**（而且没有任何报错）。已改为在构造场景前写配置，
   并用测试锁住这个先后顺序。

## 6. 已知限制（本轮**未**验证的部分）

- **全部物理结论都未验证**：代码里一切 Isaac Lab / PhysX 调用只做过静态核对
  （`articulation.py` 的 API 语义、`set_external_force_and_torques_at_position` 真的被调用），
  第一次实跑可能需要小修 —— 这正是把运行交给训练机的原因。
- 绳参数 `k = 4000 N/m`、`c = 100 N·s/m`、`L0 = 1.0 m` 是 P3 记录里的**工程起点**，
  未标定；`spawn_height = 0.35 m` 同样待确认（见 §2 的高/低站高分支）。
- **质量扫描未做**：计划要求扫 `m_L = 5/10/15/20/25 kg` 来确定现有底层的 working envelope，
  本入口目前只用小车 URDF 的名义质量（10 kg，即 `m_L = 10`）。缩放质量必须同时缩放惯量
  （AGENTS.md 的「只改质量不改惯量会造成不自洽」），这属于下一个节点。
- PPO 底层的契约仍未做（架构记录 §5 要求冻结两种、分别记录 `policy_type`），
  与 AMP 的观测顺序/动作映射**不默认相同**。
- 尚未记录步态类指标（`foot_slip` 等）；那是计划 P5 的 Logger 范围。

## 7. 未做

- 未修改 locomotion 的任何代码/配置；未训练、未导出策略、未动硬件。
- 未改动 `cart_coast.py`（CART-02 仍在问题表里）。
- 未新建目录、未新增依赖。
