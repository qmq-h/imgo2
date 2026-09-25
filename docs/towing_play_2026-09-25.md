# 拖曳上层策略回放指令、结果与 `model_2000.pt` 前置核对（2026-09-25）

关联问题：TOW-03（上层训练／验收）、TOW-05（导出件与 sim2sim 闭环）。
任务 ID：`Imgo2-towing-upper-rl-lab`（`--agent=rl_lab_cfg_entry_point`）。
目标 checkpoint：`logs/towing_rl_lab/towing_upper/2026-09-23_22-19-00/model_2000.pt`（本地产物，`logs/` 被 `.gitignore` 忽略，不入库）。

## 0. 结论速览

- 已把本机工作树**快进**到 `origin/main` = `962094f`（`Add configurable cart towing scenes for MuJoCo`），本地 5 个未提交文件未受影响（见 §2）。
- 本轮**未改任何代码**；只做拉取、离线核对、训练曲线读数与回放指令。
- checkpoint 与运行期观测契约**离线核对通过**（§4）：`memory_a` 输入 56 维 = 51 维帧 + 5 维 estimate，`memory_c` 65 维，动作 3 维；`TowingVecEnvWrapper` 自身在构造时断言 `policy==51`、`decoder==6`，契约不符会立即 `ValueError` 而不是静默出错。
- **回放已由用户执行并拿到结果**（2026-09-25 15:36 启动、15:39:36 落盘，`--num_envs=1`、`--max-steps=2000` 跑满）：`err_cmd=0.339`、`err_track=0.387`、`err_low=0.065` m/s —— 下层把上层自己的 `ref` 执行到 0.065 m/s，**剩余误差仍主要来自上层积分指令本身**。完整读数与判读见 §7。回放之前的指令由 agent 给出、由用户在训练机终端执行：agent 会话进程看不到 GPU（§2），Isaac Sim 无法在该进程里启动，`play_summary.json` 是唯一的回传通道。
- **"上层输出的指令还在抖吗"（2026-09-25 追加分析）**：`ref` 是 20 Hz 限速率积分器（每步 `x∈[-0.05,+0.025]`、`yaw∈±0.05`），**高频抖动在结构上不可能**（振幅上限 0.025 m/s）；训练日志里 `action_rate`／`action_magnitude` 的"抖动"数字几乎全部由探索噪声解释（`2Σσ²=0.477` vs 实测 `E‖Δa‖²=0.429`），本轮早先的"抖动仍明显"判断据此作废。剩下的症状是**跟不上**（`err_cmd=0.339` m/s），其中"起步 1–2 s 暂态滞后"与"整段稳态缺口"尚未区分——这是唯一还缺的数据，命令见 §7.2 末。

## 1. 本次拉取

```text
git fetch origin main      → 3fbbfa0..962094f
git merge --ff-only origin/main
962094f Add configurable cart towing scenes for MuJoCo   (2026-09-24 19:48:53 +0800)
```

拉进来的 9 个文件：`README.md`、`docs/towing_sim2sim_2026-09-24.md`、`imgo2_deploy/src/imgo2_deploy/{include/rl_sim_mujoco.hpp,src/rl_sim_mujoco.cpp}`、`imgo2_description/mjcf/{cart.xml,imgo2.xml,scene_tow_compliant.xml,scene_tow_inextensible.xml}`、`imgo2_rl/scripts/tools/check_towing_mjcf.py`。

**对本次回放无影响**：该提交只动 MuJoCo／C++ 部署侧与文档，没有触碰 Isaac Lab 的 towing 任务、`play.py`、`rl_lab` runner 或 agent/env 配置。回放链路的代码版本与训练时一致（§2）。

## 2. 运行环境与限制

| 项 | 实测 |
|---|---|
| 解释器 | `/opt/conda/envs/isaaclab/bin/python`（`run_isaaclab.sh` 的默认值） |
| torch | `2.7.0+cu128`；`dlopen(libtorch_cuda_linalg.so)` OK；CPU `linalg.solve` OK |
| `imgo2_rl` / `rl_lab` | 均已**可编辑安装**指向本仓库（`/root/Desktop/Imgo2/imgo2_rl/source/imgo2_rl`、`.../scripts/rl_lab`），所以工作树的未提交改动对回放**直接生效** |
| CUDA（agent 会话进程） | `cuda available: False`。`/dev` 下**没有** `nvidia0`／`nvidiactl`／`nvidia-uvm`，环境变量 `NVIDIA_VISIBLE_DEVICES=void`，`nvidia-smi` 报 `Failed to initialize NVML: Insufficient Permissions` |
| DISPLAY | `DISPLAY=:20.0` 但 `/tmp/.X11-unix` 不存在（无 X server）；本机回放一律加 `--headless` |

这与 TOW-04 一致：CUDA 可见性**按进程求值**，agent 会话进程的观测不代表用户终端。用户终端此前实测 `run_isaaclab.sh --check` 为 `cuda available: True`／`CUDA solve: OK`，训练也是在那里跑起来的。

### 不要回退这 5 个未提交文件

`git status` 显示以下文件仍是未提交状态，且**正是 `2026-09-23_22-19-00` 这次 run 用的那份**（已用 run 目录里启动时落盘的 `params/env.yaml` 逐项对照：`reference_tracking=-5.0`、`action_rate=-0.1`、`action_magnitude=-0.05`、`min_clearance` 的 `ratio=0.4`、三项 1e-6 诊断项等与 `upper_env_cfg.py` 现值**完全一致**）：

```text
imgo2_rl/scripts/rl_lab/rl_lab/runners/towing_on_policy_runner.py   （--compact-log，仅控制台排版）
imgo2_rl/scripts/rl_lab/towing/train.py                             （--compact-log 开关）
imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/upper_env_cfg.py   （奖励定义）
imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/upper_mdp.py       （velocity_tracking_exp 改用 reference_command、两项新奖励）
imgo2_rl/tests/test_towing_upper_rl_contract.py                     （上述奖励的契约测试）
```

回放用的是**策略权重**，观测维数不受这些改动影响，所以即便回退也加载得动；但 `play_summary.json` 与终端里的 reward 分项会与训练记录对不上（`velocity_tracking_exp` 会改回比"实际速度 vs 指令"）。要看和训练同口径的分项，就保持工作树原样。

## 3. 本轮实际执行的离线核对

```text
/opt/conda/envs/isaaclab/bin/python -m pytest imgo2_rl/tests/ -q        → 297 passed（25 s）
/opt/conda/envs/isaaclab/bin/python imgo2_rl/scripts/tools/check_towing_mjcf.py   → OK
/opt/conda/envs/isaaclab/bin/python imgo2_rl/scripts/tools/check_model_sync.py    → PASS（2 份已登记 URDF 一致）
/opt/conda/envs/isaaclab/bin/python imgo2_rl/scripts/tools/check_asset_paths.py   → PASS
run_isaaclab.sh --check                                                 → 见 §2（CUDA 在本进程不可见）
```

即：拉取后的仓库在"无 GPU 也能跑的那部分"上是干净的；`--check` 的 `cuda available: False` 是本进程视角的限制，不是环境损坏。

## 4. checkpoint 与运行时契约

`torch.load(..., map_location="cpu")` 读取 `model_2000.pt`：

```text
iter = 2000
model_state_dict / decoder_state_dict / critic_normalizer_state_dict / optimizer_state_dict /
decoder_optimizer_state_dict / infos
actor.0..6      : 256→128→64→3      （动作 3 维 = [vx, vy, yaw]）
critic.0..6     : 256→128→64→1
memory_a.rnn    : weight_ih (768, 56)   ← 56 维 actor 输入（51 帧 + 5 维 estimate）
memory_c.rnn    : weight_ih (768, 65)   ← 65 维 critic 输入
std             : (3,)
```

与训练期落盘的 `params/agent.yaml`（`decoder.frame_dim: 51`、`clip_actions: 1.0`、`rnn_hidden_size: 256`）以及运行时断言 `TowingVecEnvWrapper`（`policy==51 and decoder==6`）逐项吻合。`play.py` 的 `runner.load()` 因此不会在 `load_state_dict` 处尺寸不匹配。

本次 run 的关键配置（`params/agent.yaml` / `env.yaml`）：`seed=1`、`num_envs=4096`、`num_steps_per_env=48`、`max_iterations=2000`、`save_interval=100`、`decimation=10`、`episode_length_s=10.0`（⇒ 200 控制步/回合，`step_dt=0.05 s`）、`clip_actions=1.0`。

## 5. 训练侧读数（`logs/.../2026-09-23_22-19-00/events.out.tfevents...`，读于 2026-09-25）

TensorBoard 口径（已核对本机 `isaaclab/managers/reward_manager.py:119/149`）：
`Episode_Reward/<项> = mean_env[ Σ_t (func·weight·step_dt) ] / max_episode_length_s`，`step_dt=0.05`、`max_episode_length_s=10`。所以它**不是**每步值，标题里也没有"×1e-6"这类提醒。

| 标量 | iter 0 | 100 | 1000 | 1999（末轮） | 末轮换算（每回合 199.1 步） |
|---|---|---|---|---|---|
| `Train/mean_reward` | 1.588 | 5.501 | 6.251 | **6.854** | 回合回报 |
| `Train/mean_episode_length` | 46.1 | 196.8 | 200.0 | **199.1** | 满 200 步（=10 s） |
| `Policy/mean_noise_std` | 0.504 | 0.380 | 0.281 | **0.282** | 训练期动作噪声仍 ~0.28（play 用确定性均值） |
| `Episode_Reward/tracking_velocity` | 0.117 | 0.877 | 0.903 | **0.9211** | 每步均值 **0.925/1.0**（几乎全程跟住指令） |
| `Episode_Reward/reference_tracking` | −0.0022 | −0.185 | −0.169 | **−0.140** | ‖ref−user‖² 均值 **0.028** ⇒ RMS ≈ 0.17 m/s |
| `Episode_Reward/action_magnitude` | −0.0043 | −0.029 | −0.028 | **−0.0288** | ‖a‖² 均值 0.578（**含采样噪声**，见 §7.2） |
| `Episode_Reward/action_rate` | −0.0165 | −0.0736 | −0.0416 | **−0.0428** | ‖Δa‖² 均值 0.429（**含采样噪声**，见 §7.2；不能直接读作抖动） |
| `Episode_Reward/yaw_heading` | −2.8e−5 | −6.2e−4 | −1.0e−3 | **−1.14e−3** | yaw² 均值 5.7e−4 ⇒ RMS 0.024 rad ≈ **1.4°**，不再自转 |
| `Episode_Reward/min_clearance` | −1.3e−4 | −1.4e−3 | −2.2e−4 | **−8.7e−5** | 每步缺口 4.4e−5 m ⇒ 硬最小间距几乎不触发 |
| `Episode_Reward/clearance` | −5.3e−3 | −1.5e−2 | −7.9e−3 | **−7.1e−3** | 软障碍每步 7.2e−3 |
| `Episode_Reward/collision` | 0 | −0.0163 | −0.0049 | **−0.0023** | 碰撞指示均值 4.5e−5 |
| `Episode_Reward/fall` | 0 | 0 | 0 | **0** | 全程未跌倒（与终止统计一致） |
| `Episode_Reward/stop_towing_force` | −7e−7 | −3.4e−3 | −4.8e−3 | **−3.1e−3** | 停车后归一化绳力每步 3.1e−3 |
| `Episode_Reward/extra_distance` | −2.3e−4 | −6.2e−3 | −8.2e−3 | **−6.7e−3** | 停车后越点距离每步 6.7e−2 m |
| `Episode_Termination/time_out` | 0.117 | 0.926 | 0.985 | **0.9882** | 回合几乎都由超时结束 |
| `Episode_Termination/cart_collision` | 0 | 0.0739 | 0.0152 | **0.0118** | 碰撞终止已恢复上报并降到 ~1.2% |
| `Episode_Termination/robot_fall` | 0 | 0 | 0 | **0** | 无跌倒 |
| `Episode_Reward/obs_cart_present` | 0.105 | 0.862 | 0.887 | **0.8745** | ×1e6 后为 0.8745 ≈ **7/8 = 0.875**，与"12.5% 无小车"设计吻合 |
| `Episode_Reward/obs_towing_force` | 6.9e−12 | 1.30e−6 | 1.96e−6 | **1.64e−6** | Σ‖F‖ ≈ **328 N·步/回合**（199 步平均 1.65 N） |
| `Episode_Reward/obs_towing_force_active` | 0 | 1.30e−6 | 1.96e−6 | **1.64e−6** | 与上一行几乎相等 ⇒ 有拉力时都 >1 N（松弛时为 0），不是"有拉力但被稀释" |
| `Loss/decoder_velocity` | 0.0583 | 0.0525 | 0.0089 | **0.0052** | 训练期 smooth-L1（非物理 MAE） |
| `Loss/decoder_force` | 0.0586 | 8.18 | 4.50 | **3.69** | 同上 |
| `Loss/decoder_mass` | 0 | 2.07 | 1.41 | **1.15** | 同上 |

读法上的三条限制：

1. 这些数**来自训练期的随机策略采样**（`mean_noise_std≈0.28`），不是 `play.py` 的确定性推理；`play.py` 的 docstring 已明确两者不可直接互推。
2. `Episode_Reward/<项>` 是"回合加权和 ÷ 10 s"，与"每步值"差约 5–20 倍；上表右列的换算已按 `÷(0.05×199.1)` 展开，仅用于量级判读。
3. TensorBoard 里的分项**不能**替代这三条验收项：跨环境隔离、STOP reward 排序（需 scripted 对照）、estimator 物理量精度（需 `play.py --decoder-summary`）。

## 6. play 指令

以下命令在**用户自己的终端**（有 GPU 的那个）执行，工作目录 `cd /root/Desktop/Imgo2`。

### 6.1 推荐：headless + 诊断 + 结构化摘要（一轮 10 回合）

```bash
cd /root/Desktop/Imgo2
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/rl_lab/towing/play.py \
    --task=Imgo2-towing-upper-rl-lab \
    --agent=rl_lab_cfg_entry_point \
    --num_envs=1 \
    --headless \
    --checkpoint=logs/towing_rl_lab/towing_upper/2026-09-23_22-19-00/model_2000.pt \
    --episode-log --cmd-interval=50 --cmd-summary --decoder-summary \
    --max-steps=2000
```

- `--max-steps=2000` = 2000 × `step_dt 0.05 s` = 100 s 仿真 ≈ 10 个回合（每回合 10 s），跑完自动退出。
- `--cmd-interval=50` 每 50 步（2.5 s）打一行 `user / ref / achieved / err_cmd / err_track / accel`。
- `--cmd-summary` 结束打印每环境跟踪汇总；`--decoder-summary` 打印 decoder 在**物理量**上的 MAE（m/s、N、kg）。
- 摘要默认写到 checkpoint 同级的 `logs/towing_rl_lab/towing_upper/2026-09-23_22-19-00/play_summary.json`（另有 `--summary-out=<路径>` 可改），每 30 s 也会刷新一次，因此即使中途 Ctrl+C 也有文件可读。

### 6.2 看画面（需要显示；本机无 X server，不能用）

把 `--headless` 去掉、加 `--real-time`，仍建议 `--num_envs=1`：

```bash
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/rl_lab/towing/play.py \
    --task=Imgo2-towing-upper-rl-lab --num_envs=1 --real-time \
    --checkpoint=logs/towing_rl_lab/towing_upper/2026-09-23_22-19-00/model_2000.pt \
    --episode-log
```

### 6.3 录视频（headless，落盘 mp4）

```bash
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/rl_lab/towing/play.py \
    --task=Imgo2-towing-upper-rl-lab --num_envs=1 --headless \
    --checkpoint=logs/towing_rl_lab/towing_upper/2026-09-23_22-19-00/model_2000.pt \
    --video --video_length=400 --max-steps=400
```

产物：`logs/towing_rl_lab/towing_upper/2026-09-23_22-19-00/videos/play/*.mp4`（`--video` 会自带 `--enable_cameras`；视频录满即退出）。400 步 = 20 s 仿真 = 2 个回合。

### 6.4 要统计量就把环境数开大

`--num_envs=1` 时约有 12.5% 概率抽到"无小车"环境（§5 的 `obs_cart_present=0.8745`）。要评估拖曳本身，用 `--num_envs=32`（约 28 个带小车）配合 `--cmd-summary`／`--decoder-summary`；只是想看机器人拖车的动态，用 1 个。

## 7. 回放后怎么判读

`play_summary.json` 的键（当前版本）：`checkpoint`、`num_envs`、`steps`、`decoder_vel_mae_mps`、`decoder_force_mae_N`、`decoder_mass_mae_kg_weighted`、`decoder_mass_supervised_frac`、`towing_force_gt_mean_N`、`track_steps_mean`、`err_cmd_mean`、`err_track_mean`、`err_low_mean`。

- `err_cmd = |上层积分指令 ref − 命令期望 user|`：**上层自己的责任**，越小说明 `reference_tracking` 见效。
- `err_track = |实际速度 − 命令期望|`：最终跟速误差。
- `err_low = |实际速度 − ref|`：底层执行误差。
- 判据（`play.py` 自己打印的口径）：`err_track/err_cmd ≫ 1` ⇒ 上层指令基本到位、误差主要来自底层执行；≈1 ⇒ 上层指令本身没跟上，即 2026-09-23 加 `reference_tracking` 要修的那个现象。

历史参照（**仅作量级参照，不是同 checkpoint、也不是同代码版本**）：`logs/towing_rl_lab/towing_upper/2026-09-22_22-20-01/play_summary.json` 由旧版 `play.py` 写出（键名还是 `decoder_*_normalized`），当时 `err_cmd=0.498`、`err_track=0.489`、`err_low=0.097`——`err_cmd≈err_track` 说明误差几乎全部来自"上层指令没积到位"。本次回放要看的第一件事就是 `err_cmd` 相对它是否明显下降。

### 7.1 实际结果（2026-09-25 15:36 那次回放）

命令即 §6.1（`--num_envs=1 --headless --max-steps=2000`），环境在 15:36:07 重建 USD 缓存后启动，摘要 15:39:36 写入 `<run>/play_summary.json`（`steps=2000` ⇒ 跑满 10 个回合未提前退出）。

| 指标 | 本次（`model_2000.pt`，确定性推理） | 旧 run（`2026-09-22_22-20-01/model_1000.pt`，旧 `play.py`） | 变化 |
|---|---|---|---|
| `err_cmd_mean`（上层 `ref` vs `user`） | **0.3387** | 0.4979 | −32.0% |
| `err_track_mean`（实际速度 vs `user`） | **0.3869** | 0.4886 | −20.8% |
| `err_low_mean`（实际速度 vs `ref`） | **0.0650** | 0.0972 | −33.2% |
| `err_track / err_cmd` | **1.14** | 0.98 | 误差归属从"上层"变为"仍偏上层" |
| `err_track / err_low` | **5.95** | 5.03 | 下层执行精度优势更明显 |
| `decoder_vel_mae_mps` | 0.0326 m/s | —（旧键为归一化值 0.0322） | 口径不同，不可比 |
| `decoder_force_mae_N` | 0.3558 N | —（旧键 0.0169 归一化） | 口径不同，不可比 |
| `decoder_mass_mae_kg_weighted` | **1.6236 kg** | 1.0626 kg | 口径同（物理量），略差 |
| `decoder_mass_supervised_frac` | **5.4%** | 2.6% | 仍有拉力（>1 N）的步占比很低 |
| `towing_force_gt_mean_N` | 0.0（**瞬时值**） | 0.677 | 不可比，见下 |
| `track_steps_mean` | 835 / 2000 步 | 468.9 / 1095 步 | 有速度指令的步占比 41.8% vs 42.8%，一致 |

判读：

1. **策略确实在跟指令，而且下层比上层准得多**。`err_low=0.065 m/s` 说明冻结的底层 AMP 策略几乎完全执行了上层积分出来的 `ref`；`err_track=0.387` 里 `err_cmd=0.339` 占 88%、`err_low=0.065` 只占 17%（两者是不同方向上的向量误差模长，不构成严格分解，只作量级比较；`err_track ≤ err_cmd + err_low` 成立）。**剩余瓶颈在上层动作积分，不在底层。**
2. **`reference_tracking` 方向正确但没修完**。`err_cmd` 从 0.498 降到 0.339（−32%）是实打实的改善（虽然是不同 checkpoint／不同 `play.py`，只能量级比较）；但 `ref` 平均仍比 `user` 少 0.34 m/s，与训练日志里 `reference_tracking=−0.14`（折合 RMS 0.17 m/s）同向。下一步应针对"上层加速不够"：与 `action_magnitude=−0.05` 的张力（`+x` 上限 0.5 m/s²，到 1.0 m/s 需 40 步饱和）已被 `upper_env_cfg.py` 注释标为待观察项。
3. **decoder 速度估计可用、质量估计证据薄**。速度 MAE 0.033 m/s（回放 10 回合平均）在 0.2–1.0 m/s 的指令域里够用；力 MAE 0.356 N 也是小量；但质量只在 5.4% 的步上受监督（判据 `|F_gt|>1 N`），1.62 kg 的 MAE 建立在约 108 个步上，**不足以判定质量辨识可用**。
4. **`towing_force_gt_mean_N=0.0` 不能读作"没有拉力"**。`play.py` 在摘要时刻取的是 `action_term.towing_force_b.norm().mean()` 的**瞬时值**，回合末（停车后／绳松弛）读到 0 是正常现象；要平均绳力只能看 `obs_towing_force` 或加一段累计统计。反过来说，这也说明摘要里**没有**"每回合平均绳力"这一栏，是判读空白点。
5. **单环境单次采样的统计效力有限**：`--num_envs=1` 只有 1 条环境（12.5% 概率是无小车工况，本次 `supervised_frac>0` 说明有拉力、不是纯无小车工况），10 个回合；上面所有数字都应视为"能跑、量级合理"，不能当作分布结论。要统计量请用 §6.4 的 `--num_envs=32`。
6. **仍看不到的东西**：终止原因、settle/牵引/停车三阶段的分解、加速度是否饱和、画面。这些只在终端输出里（`--episode-log`／`--cmd-interval` 打印），本次没有留档；需要时用 §6.3 录视频或把终端输出重定向到文件。

### 7.2 上层指令到底"抖不抖"：两段离线分析

背景：2026-09-23 加 `action_rate`／`action_magnitude` 两项奖励的初衷是"上层动作 ±1 饱和抖动"。用户随后问"输入 cmd 后上层输出的指令还在抖吗"。上一次回放没有留逐帧输出，所以本节只用**代码里的硬约束**与**checkpoint/TensorBoard 里的标量**回答能回答的部分，并指出剩下必须靠逐帧 trace 才能区分的情形。

**(a) `ref` 结构上不可能高频抖动。** 上层动作经过 `HierarchicalVelocityAction.process_actions`：

```text
acceleration = a * (acceleration_max if a>=0 else -acceleration_min)   # a ∈ [-1,1] 裁剪
reference_command += acceleration * upper_control_dt                   # upper_control_dt = 0.05 s
acceleration_max = (0.5, 0.5, 1.0)   acceleration_min = (-1.0, -0.5, -1.0)
reference_min = (0.0, -0.3, -1.0)    reference_max = (1.0, 0.3, 1.0)
```

即 `ref` 是**限速率积分器**：每个控制步（20 Hz）最多变化 `x: [-0.05, +0.025]`、`y: ±0.025`、`yaw: ±0.05`。哪怕上层动作每步在 ±1 之间翻转，`ref_x` 也只能做振幅 ≤0.025 m/s 的锯齿（RMS ≤0.018 m/s），相对本次实测 `err_cmd=0.339` 只有约 5%。**所以"上层指令高频抖"这个现象在结构上不成立**；真正的问题是 `ref` 追不上 `user`（滞后／缺口），不是抖。

**(b) 训练日志里的 `action_rate`／`action_magnitude` 基本是探索噪声，不是策略抖动。** 这两个奖励作用在**采样后**的动作上（`processed_actions`），而训练期动作 = 确定性均值 + 学习到的噪声 `std`。从 checkpoint 读到的 `std = (0.2975, 0.2670, 0.2805)`，若均值动作逐步不变，纯噪声就贡献：

```text
E‖Δa‖² = 2·Σσ² = 2 × 0.2385 = 0.477
E‖a‖²  ≥ Σσ²  = 0.2385
```

把 §5 的 TB 值按 `mean func = value×10/(|weight|×0.05×N)` 展开成每步值，再减去噪声项：

| iter | `E‖a‖²`（含噪声） | `E‖Δa‖²`（含噪声） | 减去噪声后的 `E‖Δmean‖²` | 判读 |
|---|---|---|---|---|
| 0 | 0.374 | 0.717 | +0.240 | 早期动作确实在变 |
| 100 | 0.590 | 0.748 | +0.271 | 仍在变 |
| 500 | 0.443 | 0.276 | **−0.201** | 变化量已低于噪声下限 |
| 1000 | 0.565 | 0.416 | **−0.061** | 同上 |
| 1500 | 0.655 | 0.475 | **−0.002** | 同上 |
| 1999 | 0.578 | 0.429 | **−0.048** | 同上 |

残差为负（动作裁剪只会让噪声项更小，最多把残差抬到 ~+0.03），结论是：**从第 500 轮起，确定性均值动作的逐步变化在估计误差内为 0**，即策略学到的是一条（分段的）常值动作曲线；之前把 `RMS‖Δa‖≈0.65/步` 读成"抖动仍明显"是**错的**——那 0.65 完全由 `σ≈0.28` 的探索噪声解释。这条已在 §8 更正，并同步进 README 维护记录。

**(c) 由此还能反推"缺口"比"暂态"更可疑，但两者尚未区分开。**
- 若上层在牵引段长期输出接近常值的动作，`E‖mean a‖²≈0.34`（`RMS‖a_mean‖≈0.58`）给出特征加速度量级：把这些动作**全部算在 x 上**（上界）也只有 `0.5×0.58≈0.29 m/s²`，把 `ref` 从 0 拉到 0.2–1.0 m/s 需要 **0.7–3.4 s**，而牵引段只有 3–5 s（`tow_start_s=1.0`、`stop_time_s∈[4,6]`）⇒ **稳态缺口**与"只能慢慢爬"一致；若均值动作还分给了 y／yaw，或停车段有反向动作拉低这个均值，则 x 分量更小、滞后更重。
- 但同一条常值动作也可能是"起步滞后"：cmd 刚给上的前 1–2 s `ref` 落后，之后追上并保持。
- 二者在 `err_cmd` 的**全程均值**上长得一样（0.339），修法却完全不同（前者要改奖励／加上前馈或放开加速度上限，后者只影响暂态）。**现有摘要不区分阶段、也没有逐帧序列，因此无法判定——这是唯一还缺的数据。**

**(d) 还要注意奖励算在采样动作上的副作用。** `action_magnitude`／`action_rate` 对采样动作求值 ⇒ 它们会**直接惩罚探索噪声本身**（每步固定贡献 `weight·Σσ²·dt`），于是策略有把 `σ` 压小的动机。`Policy/mean_noise_std` 从 0.504 降到 0.282 有自适应 KL 的作用，但这两项也贡献了一份力量。若要真正约束确定性抖动，这两项应改用**均值动作**（或对 `ref` 的变化率）计算，而不是用采样动作——属于设计改动，未实施。

**要把 (c) 定论，只需要一次逐帧回放**（`--cmd-interval=1` 已经把 `user[0]/ref[0]/achieved[0]/err_cmd/err_track/accel[0]` 每步打印出来，无需改代码）：

```bash
cd /root/Desktop/Imgo2
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/rl_lab/towing/play.py \
    --task=Imgo2-towing-upper-rl-lab --agent=rl_lab_cfg_entry_point \
    --num_envs=1 --headless \
    --checkpoint=logs/towing_rl_lab/towing_upper/2026-09-23_22-19-00/model_2000.pt \
    --cmd-interval=1 --max-steps=2000 \
    2>&1 | tee logs/towing_rl_lab/towing_upper/2026-09-23_22-19-00/play_trace.log
```

拿到这份 log 后可以离线算出：① `Δref_x` 是否始终落在 `[-0.05, +0.025]`（验证 (a)）；② `accel_x` 的符号翻转频率与 `ref_x−user_x` 随"cmd 起点后时间"的曲线（区分 (c) 的暂态／稳态）；③ 是否存在周期 ≥5 步的慢速来回摆（`action_rate` 这类逐步指标看不见的抖）。

与 §5 的对照：TB 的 `reference_tracking` 换算出 ‖ref−user‖ 的 RMS≈0.17 m/s，而 play 报的是同样两个量的 **MAE**=0.339 m/s（确定性推理、单环境）。两者差 2 倍，可能是阶段混合不同（训练均值含 12.5% 无小车环境与全部阶段）、也可能是确定性均值动作比采样动作更差 —— 这一点同样要等 trace 与多环境回放才能定论，不能只取一边下结论。



## 8. 本轮已确认 / 待修复、待确认

**本轮已确认（有实测依据）**

1. 工作树与 `origin/main` 一致（`962094f`），且拉取内容不触碰 Isaac Lab towing 回放链路。验证：`git log`／`git diff --stat`。
2. 拉取后离线检查全通过：`imgo2_rl/tests` **297 passed**、`check_towing_mjcf.py` OK、`check_model_sync.py` PASS、`check_asset_paths.py` PASS。
3. `model_2000.pt` 与运行时观测契约一致（56 维 actor / 65 维 critic / 3 维动作），`play.py` 不会尺寸不匹配。
4. `2026-09-23_22-19-00` 这次 run 的奖励配置 = 当前工作树（`params/env.yaml` 逐项对照）。
5. 该 run 已跑满 4096 环境 × 2000 轮，TensorBoard 含**全部分项奖励、三类终止原因与 `cart_present`／绳力诊断**，本节所有读数来自 `events.out.tfevents.*`（4 MB，2000 条/标量）。
6. 确定性回放**已由用户在训练机终端跑通**（2026-09-25 15:36→15:39，`--num_envs=1`、2000 步跑满）：`play_summary.json` 落盘，`err_cmd=0.339`／`err_track=0.387`／`err_low=0.065` m/s，decoder 速度 MAE 0.033 m/s。核心结论：**下层把上层 `ref` 执行到 0.065 m/s，剩余误差主要来自上层积分指令**（§7.1）。
7. **"上层指令在抖"这一判断被推翻**（§7.2）：`ref` 是 20 Hz 限速率积分器，每步变化上限 `x∈[-0.05,+0.025]`、`yaw∈±0.05`，高频抖动幅度 ≤0.025 m/s（约 `err_cmd` 的 5%）；训练日志里的 `action_rate`／`action_magnitude` 抖动数字**几乎全部由探索噪声解释**（`2Σσ²=0.477` vs 实测 `E‖Δa‖²=0.429`，`std=(0.2975, 0.2670, 0.2805)` 取自 checkpoint），从第 500 轮起确定性均值动作的逐步变化在估计误差内为 0。**本轮早先写的"抖动仍明显、需继续加权重"据此作废，不应再提权这两项奖励。**

**待修复、待确认**

1. **"暂态滞后"还是"稳态缺口"未区分**（P0，TOW-03）：`err_cmd=0.339 m/s`（`ref` 平均比 `user` 少 0.34 m/s）既可能是 cmd 刚给上前 1–2 s 的起步滞后，也可能是整段牵引都差 0.34（§7.2(c)），而 `err_low` 只有 0.065 说明底层不是瓶颈。两种成因修法完全不同（前者只影响暂态，后者要改奖励／放开 `acceleration_max`／加前馈）。**缺什么才能完成**：一次把 `--cmd-interval=1` 逐帧输出留档的回放（命令见 §7.2 末；看 `ref_x` 是否贴限速率、`accel_x` 符号翻转频率、`ref_x−user_x` 随"cmd 起点后时间"的曲线），以及分阶段的 `err_cmd`（settle／牵引／停车）统计——当前摘要不区分阶段。
2. **上层策略仍无部署导出件**（P0，TOW-05）：`towing/play.py` 只做回放，没有导出 `exported/policy.pt`（ppo／amp／himloco 的 `play.py` 才有导出分支）。按 AGENTS.md 约定导出必须走 `play.py`，需要用户决定是给 towing 的 `play.py` 加 `--export` 还是另起一次调用；recurrent actor＋decoder 的导出还要处理 GRU hidden 与 5 维 estimate 的接线，之后才能接 MuJoCo／Gazebo sim2sim。**2026-09-25 已在 `imgo2_deploy/policy/imgo2/towing/policy.pt` 放入训练态 `model_2000.pt` 的占位拷贝（§9，用户决定）——它不可被 libtorch 加载，导出件仍缺。**
3. **统计效力不足**（P1）：本次 `--num_envs=1`、10 个回合；`decoder_mass_supervised_frac` 只有 5.4%（约 108 步），质量 MAE 1.62 kg 的证据薄。需要一个 `--num_envs=32` 的回放（约 28 个带小车环境）才能给分布结论。
4. **摘要缺"阶段分解"与"终止原因"**（P1）：`play_summary.json` 只有全程平均的 `err_*` 和瞬时 `towing_force_gt_mean_N`（本次读到 0.0，不能当平均绳力）；settle／牵引／停车的分段误差与终止原因只在终端里，未留档。
5. **反抖动奖励的设计缺陷**（P1，更正 §7.2(b)）：`action_rate`／`action_magnitude` 作用在**采样动作**上，等于同时惩罚探索噪声本身（每步固定贡献 `weight·Σσ²·dt`），会给"压小 `σ`"一个与任务无关的激励；`mean_noise_std` 0.504→0.282 有一部分可能来自这里（自适应 KL 也在起作用，未分离）。要真正约束确定性抖动，应改用**均值动作**或 `ref` 的变化率计算。属设计改动，未实施。**注**：本轮早先写的"抖动仍明显、`RMS‖Δa‖≈0.65/步`"是把探索噪声当成了策略抖动，已作废。
6. **`upper_env_cfg.py`／`upper_mdp.py` 等 5 个文件的改动仍未提交**（P1）：它们承载本次 run 的奖励定义与 `--compact-log`，留在工作树里没有任何版本号可引用。是否提交、以及是否连同 `docs/` 一起提交，需要用户决定。**2026-09-25 已随本次提交一起入库**（用户决定），见 §9 与 README 维护记录。
7. **跨环境隔离与 STOP reward 排序仍未验证**（P0，TOW-03 原始验收项）：§5 的 TB 分项与 §7.1 的单环境回放只能说明"能跑、量级合理"，不能替代 scripted 对照（同一工况脚本化对比 STOP 前后绳力／间隙）与跨环境一致性检查。

## 9. 部署目录占位权重入库（2026-09-25，用户决定）

用户指示："把 2000 轮那个策略拉取到 `imgo2_deploy/policy/imgo2/towing` 下；直接先放 pt，暂时不导出也没事。"

```bash
cp logs/towing_rl_lab/towing_upper/2026-09-23_22-19-00/model_2000.pt \
   imgo2_deploy/policy/imgo2/towing/policy.pt
```

- **校验**：两处 `sha256 = a4d800a26c1d1a50eee220da44a32b51111c3ce475f28c8e25b421b32a1f6ab2`、9761119 B，`cp` 后逐字节一致；`git check-ignore -q <path>` 退出 1 ⇒ `.gitignore` 的例外 `!imgo2_deploy/policy/imgo2/**/policy.pt` 生效，文件可入库（该例外本来是为 `play.py` 导出的网络准备的）。
- **它现在不是"可部署的策略"，只是权重留档**（用户已知并接受）：
  1. 内容是**训练态 checkpoint**——`model_state_dict`／`decoder_state_dict`／`critic_normalizer_state_dict`／两个 optimizer／`iter`，不是 `export_policy_as_jit` 的 TorchScript 归档；`inference_runtime::TorchModel::load` 预期直接加载失败（**未实测**：本机无 libtorch／MuJoCo 构建）。
  2. `towing/play.py` 的导出分支为 **0 处**（`grep -c export` = 0；ppo／amp／himloco 的 `play.py` 才有 `export_policy_as_jit`）。
  3. 该目录**没有** `config.yaml`（部署侧靠它拿 `num_observations`／观测顺序与 scale／`action_scale`／`rl_kp`／`joint_mapping`），FSM 里也**没有**对应按键与状态类（现有键只有 1/2/3 = ppo／himloco／amp）。
  4. 更根本：部署运行时的接口是 `Model::forward(const std::vector<std::vector<float>>&)`（`inference_runtime.hpp`），**单输入、无内部状态**；`rl_sim_mujoco.cpp` 只在 `observations_history` 非空时拼一个 history 缓冲后调用它。装不下 decoder+actor 两个 GRU 的 hidden state ⇒ 即便导出成 TorchScript，也必须自带状态管理（或改运行时），这正是 TOW-05 里"51 维双 GRU 观测／控制链"未完成的部分。
- **要真正接入**（TOW-05 完成条件，未变）：先给 `towing/play.py` 加导出路径（headless），由训练机跑一次产出 `exported/policy.pt`，再补 `policy/imgo2/towing/config.yaml` 并**替换**本占位文件；导出后仍按三条契约复核（与 checkpoint actor 同批输入 max diff = 0、能被部署 libtorch 加载、观测维数顺序与训练侧一致）。

