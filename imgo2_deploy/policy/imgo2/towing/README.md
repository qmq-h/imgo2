# towing 策略目录：仍**没有可部署的 TorchScript**

## 2026-10-24 新增：`model_1300.pt`（训练 checkpoint 留档）

- 来源：`logs/towing_rl_lab/towing_upper/2026-10-19_17-44-30_reward_20261019_ppo2500/model_1300.pt`
- 大小：10,245,431 B
- sha256：`a54780813029b2764fc651593e2b20b87bef8e060b524eba67ce1195697f9e05`
- 内容：训练态 checkpoint（v5 契约：frame 58 / explicit 5 / latent 16 / actor 79 / 动作 13，
  含 actor/decoder/optimizer/normalizer），**不是** `play.py` 导出的 TorchScript。
- 用途：给部署/复现留一份与 2026-10-20 全列评测对应的权重；**不要**直接把它当 `policy.pt` 接进 libtorch。

## `policy.pt` 仍是旧占位，不要使用

本目录的 `policy.pt` 是 2026-09-25 按用户决定放入的**占位权重**：

- 来源：`logs/towing_rl_lab/towing_upper/2026-09-23_22-19-00/model_2000.pt`
  的原样拷贝（sha256 = `a4d800a26c1d1a50…`，9,761,119 B）；
- 内容：**训练态 checkpoint**（含 optimizer／decoder／normalizer），不是 TorchScript；
- 契约：旧契约（含质量头）时代的权重；现行 v5 会在 `load_state_dict` 前拒载；
- 本目录**没有** `config.yaml`，FSM 也**没有** towing 对应的按键/状态类，不会被部署链路自动加载。

## 仍缺什么（TOW-05）

要得到真正可部署的 towing 策略：

1. 给 `imgo2_rl/scripts/rl_lab/towing/play.py` 加导出分支（headless，内部走
   `export_policy_as_jit`；注意 recurrent actor + decoder 的 GRU hidden 与 5 维 estimate 接线）；
2. 用 v5 契约的 run 在训练机导出，产物在 checkpoint 同级 `exported/policy.pt`；
3. 拷到本目录并补 `config.yaml`，按三条契约复核：与 checkpoint 的 actor 在同一批确定性输入上
   max diff = 0、能被部署自己的 libtorch 加载、部署 interface 的观测维数与顺序和训练侧一致。

在完成上述导出前，`model_1300.pt` 只是**训练权重留档**，不是部署件。详细背景见根
`README.md` 的 TOW-05 与 `docs/towing/towing_play_2026-09-25.md` §9。
