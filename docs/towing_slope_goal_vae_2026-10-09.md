> **已被同日后续改动取代（2026-10-09，见 [连续坡面剖面记录](towing_profile_terrain_2026-10-09.md)）**：
> 本文描述的「每格一个恒定坡度、按行列交替正负」已被替换为**一条连续剖面**
> （每条 lane：跑道 15 m、剖面 11.25 m：平地 2.25 m → 上坡 3 m → 坡顶 0.75 m → 下坡 3 m → 平地 2.25 m，
> 坡度量级 0/5/10 由列决定）。
> 800 环境、40 列 × 20 行、长度 0.6–1.2 m、目标 10 m 与 timeout 的算法仍然有效，
> 但**坡度符号、出生旋转、跌倒判据与 timeout 用的距离**都已改变；以新记录为准。

# 牵引坡面、目标终止与 VAE 构建记录

> **2026-10-10 变更**：`extra_distance`（`mdp.post_stop_distance`，−0.1）已从奖励表**删除**（函数本体与 `post_stop_allowance_m` 形参保留、目前未接入奖励）；`min_clearance` 的阈值由 `ratio(0.25) × 连接长度` 改为「**出生间隙 − 绝对死区 `deadband_m`(0.02 m)**」（2026-10-10 当天先落成 `spawn_margin(0.85)` 相对余量、同日再微调为绝对死区；语义＝「不许比出生时更近」）、权重由 −2.0 提到 **−5.0**。本文以下是变更**之前**的记录，现行口径见 [奖励改动记录](towing_reward_retune_2026-10-10.md) 与 README 问题表 TOW-24。

日期：2026-10-09。目标基线为 main c49be10；桌面 imgo2_CMoE 分支的已有改动独立保留。

## 已修源码，待运行验证

- 连接长度由 0.4–0.8 m 改为 0.6–1.2 m、20 档。旧短绳出生间隙约 0.108 m，且受绝对 clearance barrier 持续惩罚；新最短绳名义间隙约 0.250 m，soft barrier 仍非零，不能称安全已验证。
- 地形为 40 列 × 20 行 = 800 环境；20 平地、10 个 |5°|、10 个 |10°|。每行 ±5°/±10° 各 5 列；每块连接比例 4:4:2，合计 compliant/rigid/inextensible = 320/320/160。坡度正负号随行交替，使弹性档不会永久绑定一个方向；同一长度并非各弹性档与上下坡的完全笛卡尔积。12.5% 无车随机子集保留，因此有效有车样本计数不保证严格比例。
- 使用真实闭合斜面 mesh 和 row-major 环境原点映射。每条跑道世界 X 为原点后 3 m 到前 17 m、宽 6 m，行/列间留 2 m 空隙；不是用地形标签替代几何。出生旋转、根部离地高度、挂点间距按坡面切向/法向求解；跌倒判断改坡面法向高度，增加车/机器人出界失败终止，避免走出独立 mesh。
- 正常终止为机器人沿坡面从本次出生位置前进到目标，或超时。目标定义为前进距离平面，非精确三维点；横向安全由跑道出界约束。目标默认 10 m，可改 `goal_distance_m`；摔倒/碰撞/出界仍为失败终止。目标成功为真 termination，超时标 `time_out=True`；旧 4–6 s 定时 STOP 与停车阶段奖励移除，测量台 `tow_drag.py` 的独立 STOP 实验保留。
- 速度 0.4–1.5 m/s；统一 timeout = 启动等待 + D/v_min + 余量 = 1 + 10/0.4 + 2 = 28 s。理想最低速度抵达为 26 s、最高速度约 7.667 s，含等待。距离按坡面行程而非水平投影；修改目标/最低速度/启动等待时 timeout 自动跟随。此标准保证理想行走不超时，不保证真实牵引一定完成。

## 16 维 latent 与显式 6 维

结构：57 维本体帧 → Linear128/ELU → GRU128 → μ16/logσ²16 → z16 → Linear128/ELU → 显式头 [vx, vy, m, Fx, Fy, Fz] 共 6 维。decoder 必须经过 z，没有绕过 bottleneck 的显式捷径。Actor 拼接 57 + 6 + 16 = 79 维，critic 仍 72 维，动作仍 12 维关节残差。

参考本仓 CMoE state estimator 的均值/方差、重参数化与 KL 思想。这里不是完整移植 CMoE：CMoE 的独立显式头监督速度，同时用 latent+explicit 重建下一本体帧；本轮按用户要求，latent 经 decoder 直接重建当前 6 维特权目标。只监督这 6 维不能保证 latent 独立识别摩擦/坡度，或每维有物理含义；它只需成为预测与策略可用的隐表示。是否需要下一帧辅助重建，待训练数据决定。

训练 z = μ + exp(logσ²/2)·ε，ε 为标准高斯 `randn_like`；未照搬 CMoE 源码中的均匀 `rand_like`。logσ² 裁剪到 [-10,4]。损失保留各物理单位的 SmoothL1（速度/力/质量权重 5/10/1，质量保持受交互力及有车掩码门控），加 β KL(q(z|history)||N(0,I))，β 默认 0.005、可配置。KL 原值和加权贡献均写日志；权重合理性、塌缩和可辨识性待实跑。

采集与推理用 μ，解码得到显式估计，actor 输入两者都 detach；PPO 不回传估计器，只由估计损失更新 VAE。训练保存 rollout 起始 GRU 隐状态，训练器从同一状态展开，按上一拍 done 清零；此前采集连续带状态而每个 48 步训练块从 None 起步的问题已修源码。模型更新后携带隐状态属于通常的截断 BPTT，仍需运行核对。

Checkpoint 加入 version=2、frame/explicit/latent 维数契约。旧 56 或 63 维 actor checkpoint 明确拒绝加载，应新建运行，不能续训旧 run。play 的 `last_estimate` 保持 6 维物理量，另外暴露 `last_latent`；未导出部署 policy。

## 验证依据与缺项

用户明确要求「不用本地做检验」，后续未运行编译、网络测试或 Isaac Lab。隔离 CPU PyTorch 依赖安装已取消；临时目录删除，未改原训练环境。

此前使用 Windows Python 3.11.9（`C:/Users/qmq/AppData/Local/Programs/Python/Python311/python.exe`）运行 `test_towing_slope_geometry.py` 的 4 项纯几何/超时检查通过：800 单元坡度和类型计数、闭合拓扑/朝上法线、出生挂点与目标范围、全速度范围理想到达时间。完整坡面检查入口曾因暂存区未复制 rope_model 依赖而未执行；不得标为全套通过。这些不验证地形导入、GPU 张量、接触动力学或 VAE。

训练机待执行（普通 Python 命令需使用其已有 Isaac Lab/torch 环境）：

```sh
python imgo2_rl/scripts/tools/check_towing_slope_grid.py
python -m unittest discover -s imgo2_rl/tests -p 'test_towing*.py'
```

再构造覆盖全网格的 800 环境短回放/短训练，核对上/下坡出生姿态、mesh 原点、首拍刚体误差、目标成功/timeout 的 bootstrap、失败终止；检查 VAE 前向/梯度、done 掩码、起始隐状态、6 维估计与 KL 日志。长训练与硬件运行未授权且未启动。

README 保留 TOW-08/09 为「已修，待验证」。已知残差回放单位与 clip_actions 说明问题 TOW-07、本体探索幅度和 VAE 超参数不在本轮验证结论之内。

## 提交与推送范围建议

构建结束时未提交、未推送，未运行新的本地验证；用户随后授权 commit + push，按以下范围执行，待运行验证状态不变。main 工作树共 18 个新增/修改文件（含先前 git 同步记录），应将牵引源码、地形/回合模块、训练配置、更新后的测试与 README/docs 一起提交，建议标题 `feat(towing): add slope goals and supervised VAE estimator`。README 引用的 `docs/git_sync_2026-10-08.md` 也必须入库，避免断链。推送到 main 时明确保持「已修，待训练机验证」，不宣称可用于已验证训练或部署。

桌面工作树属于 imgo2_CMoE，已有 ad04b3b 独立提交（CMoE 配置精简、4 文件，当前相对本地 origin/imgo2_CMoE 超前 1）。该提交应独立审阅并推送到自己的分支；不要合入本轮 main 牵引提交。桌面的 amp_env_cfg.py 未提交修改仅删除旧奖励说明，不必为它额外形成行为变更提交。其余 CMoE 评审、spawn 工具、牵引方案草稿和 drawio 仍保留在桌面；本轮正式结论以 main 本记录为准，先不混入 main。提交推送阶段已重新 fetch；main 与 origin/main 同步，CMoE 仅有上述 1 条待推送提交。

用户指定的 `C:/Users/qmq/Desktop/Imgo2/.towing_slope_edit` 已删除；正式改动已保存到 main 工作树，不再依赖该临时目录。
## 追加（同日）：目标制反转，恢复 STOP 相位

用户当天判定「改成目标制不对」，要求**保留**两条停车段奖励并改回三段制：

- `post_stop_towing_force`（−1.0）：停车之后绳子还绷着就扣分——"到点了就把拉力卸掉"，
  别一直拽着、也别把小车当锚；
- `post_stop_distance`（−0.1）：停车之后还往前多走的距离——"说停就停，别继续滑"。

回合结构回到：**settle（指令 0，静止稳定）→ tow（指令 = tow_speed）→ STOP（指令 0）→ 超时结束**。
因此本文件第 17 行写的「目标成功为真 termination …… 旧 4–6 s 定时 STOP 与停车阶段奖励移除」
**已被本追加取代**：`goal_reached` 不再作为终止项，改名为 `stop_reached` 且只用来置零指令
（`STOP_DISTANCE_M = 10.0 m`，语义由「目标」变「停止触发点」）；target 相关的 VAE 与观测部分不变。

### 「越过坡之后」怎么保证

用户要求「给 cmd vel 一定要在越过坡之后」。坡面出口是 `FLAT_OUT_START_M = 9.0 m`
（剖面：平地 2.25 → 上坡 3 → 坡顶 0.75 → 下坡 3 → 平地 2.25），触发点取 10.0 m，
`upper_env_cfg.__post_init__` 里有断言 `stop_distance_m > FLAT_OUT_START_M` 守着。

**触发用进度、不用时间**：最慢速度 0.4 m/s 下，光走到坡出口就要约 23 s（timeout 约 28.2 s），
固定时间阈值无法保证"越过坡"；进度触发同时也满足"走过一段时间之后"。

### 相位与时限

- `stop_time_s`（初值 `+inf`）与 `stop_origin_x` 在进度越过 `stop_distance_m` 那一拍写入；
  两条奖励按 `elapsed_s >= stop_time_s` 门控，未停车前**精确为 0**（`inf` 比较恒假）。
- **正常回合一律由 `time_out` 收尾**（用户 2026-10-09 明确确认）。中途曾加过一个
  「停车后固定窗口终止」的 `post_stop_timeout`，随后按用户要求去掉：不加停车窗口终止项，
  timeout = `settle + 到 STOP 点的坡面弧长 / 最小速度 + POST_STOP_WINDOW_S`
  = 1 + 10.0926/0.4 + 3 = **29.23 s**。副作用是「速度越快、停车尾巴越长」（0.4 m/s 3 s、
  1.5 m/s 22.5 s），该偏置已向用户说明并被接受。
- **「走没走到 STOP 点」改用统计量回答**：诊断项 `obs_stop_reached`（权重 1e-6，只进
  TensorBoard）返回 action term 的粘性标志 `_was_stopped`。读法：
  `Episode_Reward/obs_stop_reached ÷ 1e-6` = 该回合处于 STOP 相位的步数占比，
  **> 0 即「走到了」**、恒 0 即「没走到」（这些回合全部由 `time_out` 收尾）。
- **失败退出保留**：`robot_fall`（离局部坡面高度 < 0.18 m ⇒ **倒地退出**）、
  `cart_collision`、`terrain_exit`；`time_out` 是正常收尾。

### 验证与限制

全量 `imgo2_rl/tests` **424 项通过 0 失败**：契约测试改成 STOP 制（断言 `goal_reached` 不再是
DoneTerm、`stop_reached` 存在、两条奖励权重与门控、`stop_time_s`/`stop_origin_x` 的写入与复位），
几何测试把 `GOAL_DISTANCE_M` 改名为 `STOP_DISTANCE_M` 并加「STOP 点在坡面出口之后」断言。

**未验证**：本机无 Isaac Lab（`omni.kit` 不可 import），未构造环境、未实跑；STOP 触发的运行时
行为、两条奖励的实际量级、`POST_STOP_WINDOW_S = 3 s` 是否够用，以及 800 环境里平地/5°/10°
两类坡道的 STOP 是否都落在出口平地，都待训练机确认（README 问题表 TOW-13）。
