# P4（上半）：冻结底层策略的接口契约与适配器

日期：2026-09-20。状态：**已实现，并随 P4 下半在训练机验证通过**（0.5 m/s 拖曳成立，该契约驱动的策略跟速 102%，见 [P4 下半记录 §5.5](towing_p4_tow_drag_2026-09-20.md)）。PPO 底层契约（架构记录 §5 要求冻结两种）仍未做。
对应研究计划 `paper_plan_imgo2.md`「P4：接入你现有 locomotion」的第一步：**先完全不改
locomotion**，把已有底层当冻结策略使用。

## 1. 本轮实现

| 文件（相对仓库根） | 内容与状态 |
|---|---|
| `.../towing/utils/policy_cfg.py` | 冻结策略契约（观测顺序/缩放、动作缩放、默认姿态、控制周期、reset 契约）；**纯标准库** |
| `.../towing/utils/low_level_policy.py` | 适配器：按契约组装观测 → TorchScript 推理 → 裁剪动作 → 关节位置目标 |
| `imgo2_rl/tests/test_towing_policy_contract.py` | 20 项测试，含与部署 yaml 的逐项交叉核对 |

`utils/__init__.py` 保持不 re-export：它自己写明「重依赖留在消费模块内」，而
`low_level_policy.py` 会 import torch。

## 2. AMP 契约（45 维 actor）

观测顺序与维数（`observation_terms` / `observation_dims`）：

```text
base_ang_vel 3 (×0.25) | projected_gravity 3 (×1.0) | velocity_commands 3 (×1.0)
| joint_pos_rel 12 (×1.0) | joint_vel_rel 12 (×0.05) | last_action 12 (×1.0)   = 45
```

其余契约：`clip_obs = 100.0`、无观测历史、默认姿态 `(0, 0.87, -1.82)×4`、
动作缩放 `(0.125, 0.25, 0.25)×4`、动作裁剪 `±3`、`joint_mapping` 恒等、控制周期 0.005 s。

**动作语义按部署侧对齐**（`imgo2_deploy/src/.../rl_sdk.cpp` + `rl_sim_mujoco.cpp`）：

- `obs.actions = Forward()`，而 `Forward()` 末尾就是 `clamp(clip_actions_lower/upper)`
  ⇒ **`last_action` 是裁剪后的网络输出**，不是网络原始输出；
- 关节目标 = `default_dof_pos + action_scale * clip(action)`（部署 `ComputeOutput()` 同式）；
- 观测在拼接后整体 clip 到 ±`clip_obs` 再喂网络。

`reset` 契约：`last_action` 归零（与部署 `RL::InitRL()` 一致），并给出建议的站定步数
`reset_settle_s / control_dt`；`reset_settle_s` 初值 1.0 s，**待 P4 仿真确认**（参考部署的 FSM
也是先 GetUp/站定再交给策略，本项不沿用隐式默认）。

## 3. 离线验证

解释器 `/opt/conda/envs/isaaclab/bin/python3`（Python 3.11.13）。

| 验证 | 结果 |
|---|---|
| `python3 -m unittest test_towing_policy_contract` | **20 项通过** |
| 全量离线测试（10 个测试文件） | **162 项通过**（本轮之前 142 项） |
| `check_model_sync.py` / `check_asset_paths.py` / `check_cart_model.py` | 退出码均为 0 |

关键点：

- **与部署配置交叉核对**（防漂移）：测试直接读
  `imgo2_deploy/policy/imgo2/amp/config.yaml` 与 `base.yaml`，逐项比对
  `num_observations`、`num_of_dofs`、`model_name`、`clip_obs`、观测项顺序、观测历史、
  `action_scale`、`default_dof_pos`、动作裁剪上下界、`joint_mapping`、观测缩放（按项展开）、
  以及控制周期。任一不一致即失败 ⇒ 契约不会在两处各自漂移。pyyaml 缺失时该项跳过。
- **导出件维数自检**：适配器构造时用一批零输入做一次前向，确认导出件输入/输出形状与契约
  一致；并把「对零输入产生非有限值」也算失败。这是部署侧「三条契约」第一条在训练侧的等价物：
  维数不对要在**构造时**报错，而不是拖到拖曳跑一半才形状不匹配。
- **适配器端到端**（临时 TorchScript 假策略，本机无 GPU 走 CPU）：观测组装的顺序/缩放/裁剪、
  动作裁剪、`default + scale*action`、`last_action` 回灌进下一帧观测、`reset` 归零与站定步数、
  导出件输出维数不符时报错、导出件缺失时报错。
- **契约校验**：维数不匹配、缩放长度不符、`joint_mapping` 非置换、裁剪上下界倒置、
  非正缩放/周期、关节名重复，全部显式报错。
- **已核对真实导出件**：`imgo2_deploy/policy/imgo2/amp/policy.pt` 的 sha256 前 16 位为
  `cba59d44e387834e`，与 README AMP-05 记录一致（= 24500 轮 checkpoint 的正式导出件）。
  注意该 yaml **头部注释**仍写着 `model_5000.pt` / `0e5bd661…`，已过期；契约字段本身未变，
  这两处不一致已在测试里用哈希固定住。

## 4. 待完成（P4 下半）

- `towing/towing_env_cfg.py`：机器人 + 小车 + 绳的场景配置（沿用 P1/P2 的
  `SimulationContext + InteractiveScene` 风格，不用 `ManagerBasedRLEnv`）。
- 拖曳实验入口（`scripts/towing/` 内）：每物理步「绳力 → 轮阻 → 写仿真 → step」，
  机器人侧由 `FrozenLowLevelPolicy` 按速度指令驱动，记录 `v_R / v_L / T / d / pitch`。
- **训练机验收**（由用户执行，命令见 P4 下半记录）：先不改 locomotion，`v_cmd = 0.5 m/s`
  跑 5 s 检查 `v_R ≈ v_L` 与稳定拖曳阶段 `T(t)` 是否进入相对稳定区间；再试 `1.0 m/s`；
  然后扫 `m_L = 5/10/15/20/25 kg` 确定现有底层的 working envelope（计划 P4 的明确要求）。
  注意 P3 记录 §4.2 的提醒：**自由质点模型的收敛结论不可外推到这里**，必须以真实系统为准。
- PPO 底层的契约（架构记录 §5 要求冻结两种、分别记录 `policy_type`）留作下一个节点：
  它的观测顺序/动作映射与 AMP 不同，**不默认相同**。

## 5. 未做

- 未修改 locomotion 的任何代码或配置；未接入场景；未运行仿真或训练。
- 未标定 `reset_settle_s`、未标定绳参数 k/c/L0（P3 记录的工程起点仅供 P4 参考）。
- 未验证任何物理行为：本轮全部结论都是**离线接口层面**的。
