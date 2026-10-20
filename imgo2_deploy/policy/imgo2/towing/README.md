# towing 策略目录：当前**没有可部署策略**

本目录的 `policy.pt` 是 2026-09-25 按用户决定放入的**占位权重**，不是部署件：

- 来源：`imgo2_rl/logs/towing_rl_lab/towing_upper/2026-09-23_22-19-00/model_2000.pt`
  的原样拷贝（`sha256 = a4d800a26c1d1a50…`，9,761,119 B）；
- 内容：**训练态 checkpoint**（含 optimizer／decoder／normalizer），不是 `play.py` 导出的
  TorchScript，部署侧 libtorch 加载器读不了；
- 契约：旧契约（含质量头）时代的权重；现行训练/运行时契约是 **v5**
  （frame 58 / explicit 5 / latent 16 / actor 79 / critic 73 / 动作 13），
  v5 runner 会在 `load_state_dict` 之前直接拒载；
- 本目录**没有** `config.yaml`，FSM 也**没有** towing 对应的按键/状态类，所以它不会被
  部署链路自动加载；**不要**手工把它接进去。

要得到真正可部署的 towing 策略（TOW-05，仍未完成）：

1. `imgo2_rl/scripts/rl_lab/towing/play.py` 目前**没有导出分支**（ppo／amp／himloco 的
   `play.py` 才有）。先按 AGENTS.md 的口径给它加导出（headless，内部走
   `export_policy_as_jit`，产物在 checkpoint 同级的 `exported/policy.pt` + `policy.onnx`）；
   注意 recurrent actor + decoder 的 GRU hidden 与 5 维 estimate 接线；
2. 用 v5 契约的训练 run 在训练机导出，拷到本目录并补 `config.yaml`；
3. 按三条契约复核：与 checkpoint 的 actor 在同一批确定性输入上 max diff = 0、
   能被部署自己的 libtorch 加载、部署 interface 的观测维数与顺序和训练侧一致。

历史权重保留在 git 里（`git log -- imgo2_deploy/policy/imgo2/towing/policy.pt`）；详细背景见
根 `README.md` 的 TOW-05 与 `docs/towing/towing_play_2026-09-25.md` §9。
