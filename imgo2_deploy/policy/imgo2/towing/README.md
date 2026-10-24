# towing 策略目录：已有 TorchScript，但仍未接入部署 FSM

## `policy.pt`（2026-10-24 导出，58 → 13，stateful）

- 来源 checkpoint：`logs/towing_rl_lab/towing_upper/2026-10-19_17-44-30_reward_20261019_ppo2500/model_1300.pt`
- 导出命令：`imgo2_rl/scripts/rl_lab/towing/play.py --export`（见根 README TOW-05）
- 大小：1,941,632 B
- sha256：`36313d2a132c3253602f6841dd2e7e4f71b0297d55f5c579e622636af5b0550d`
- 形态：**TorchScript**（`torch.jit.script`），不是训练态 checkpoint
- 输入：`[1, 58]` 的 `policy_frame`（`upper_policy_runtime.FRAME_TERMS`）
- 输出：`[1, 13]` 的 **clamp 到 ±1 的归一化动作**（1 维 vx 偏移 + 12 维关节残差；
  不是关节残差 rad，部署侧仍需乘 `action_scale` 并接线偏移头）
- 状态：decoder GRU + actor GRU 的 hidden 是模块内部 buffer，`forward` 每拍推进；
  导出时带 `reset()`（`@torch.jit.export`），但 C++ `Model::forward` 没有 reset 入口。

## `model_1300.pt`（训练 checkpoint 留档）

- 源：`logs/towing_rl_lab/towing_upper/2026-10-19_17-44-30_reward_20261019_ppo2500/model_1300.pt`
- 大小：10,245,431 B
- sha256：`a54780813029b2764fc651593e2b20b87bef8e060b524eba67ce1195697f9e05`
- 内容是训练态 v5 checkpoint（actor/decoder/optimizer/normalizer），**不是部署件**；
  留着是为了从同一权重重新导出/复核。

## 仍缺什么（TOW-05 未完成）

1. **`config.yaml` 仍不存在**：部署侧靠它拿 `num_observations`／观测顺序与 scale／
   `action_scale`／`rl_kp`／`joint_mapping`；当前目录没有。
2. **FSM 未接线**：`imgo2_deploy/src/imgo2_deploy` 的按键/状态类仍只有 ppo／himloco／amp，
   没有 towing 对应状态；所以这份 `policy.pt` 不会被部署链路自动加载。
3. **状态复位**：模块内部两套 GRU hidden 会跨调用保留；部署侧一个 episode 一个进程，
   或另加 reset 通道，否则下一回合会带上一回合历史。
4. **libtorch 实载未验证**：本机只验证了 `torch.jit.load` + CPU 前向 + `reset()`；
   部署自己的 libtorch 加载器还没实跑。
5. **动作接线**：`policy.pt` 输出归一化动作，部署侧必须按冻结策略的 `action_scale`
   （hip 0.125 / thigh·shank 0.25）换残差，并把前 1 维 vx 偏移按 `command_offset_vx`
   接到低层指令；这些仍由部署 FSM/config 完成。

在 1–5 完成前，`policy.pt` 是**可加载的导出件**，不是“已接入部署的策略”。
历史占位权重仍在 git 历史里（`git log -- imgo2_deploy/policy/imgo2/towing/policy.pt`）。
