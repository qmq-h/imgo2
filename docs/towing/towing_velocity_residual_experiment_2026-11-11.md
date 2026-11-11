# 速度残差（vx offset）有效性实验计划（2026-11-11）

> 当前状态：v6 已删除 1 维 vx 偏移头，动作 **12 维纯关节残差**；正在按 50 Hz
> （`upper_control_dt=0.02`）用部署侧老 PPO 从零训练一个新的上层策略。
> 本文只记录实验计划，暂不改代码。

## 1. 为什么现在重新考虑

2026-10-24 去掉 vx 偏移头的原因是：旧 v5 actor 的偏移头一直在**系统性负偏**，
把脚本速度往下压。

旧 v5 `model1300` 的 1200 case 统计：

- `upper_offset_vx` **1200/1200 全非零**；
- tow 段中位数 **−0.139 m/s**；
- coast 段中位数 **−0.176 m/s**；
- 范围约 **−0.188…−0.011 m/s**；
- 翻车/横向问题与负偏移同时存在，但当时无法区分“偏移救了什么”和“偏移压坏了什么”。

v6 去掉后，actor 完全不能改写 vx，速度只由脚本/PD 给出。需要验证：

1. 12 维纯残差是否比 13 维（带 vx 偏移）更容易在 15°/25 kg 侧翻；
2. 速度残差是否能改善跟随/停车，还是只是持续压低速度；
3. 如果带回来，是否值得多 1 维动作、58→57 帧、79→78 actor 的契约代价。

## 2. 实验设计（等当前 50 Hz v6 run 完成后再做）

必须做**同口径 A/B**，只改速度残差这一项：

| 项 | A：当前 v6 | B：加回 vx offset |
|---|---|---|
| 动作 | 12 维关节残差 | **13 维 = 1 vx + 12 关节** |
| policy 帧 | 57 | 58 |
| actor | 78 | 79 |
| critic | 72 | 73 |
| contract | v6 | **v7**（新版本，旧 v6 不复用） |
| 低层 | 老 PPO `53a57f90` | 同一份 |
| 频率 | 50 Hz | 50 Hz |
| 地形 | 70 列/1400 env，15° 最大 | 同一份 |
| 质量 | 5–30 kg 连续 | 同一份 |
| 训练轮数 | 3000 | 3000 |
| seed | 固定 | 同一 seed |

B 的偏移头建议先保持原语义：

```text
offset = clamp(clamp(u_cmd, ±1) × 0.5, −0.2, +0.6)
loco_vx = clamp(task_vx + offset, −1.0, 1.0)
```

但训练时必须记录：

- 偏移均值/分位数（tow、coast 分开）；
- 偏移是否仍然全程负偏；
- STOP 之后偏移是否还让机器人倒退。

## 3. 评测

A/B 各跑：

1. **无载**：全 1400 env，`--no-cart-fraction 1.0`；
2. **带载**：5/10/15/20/25/30 kg 六档，`--no-cart-fraction 0`；
3. 0.5/1.0 m/s，`--stop-progress 10`，`--tow-duration 24 --coast-duration 5`；
4. 先确定性，再抽一组 `--upper-stochastic` 对照。

重点指标：

- `FALL`，特别是 **15° / 25 kg / 1.0 m/s**；
- 小车速度稳定性（`speed_tracking` 通过数与 steady MAE/std）；
- `JOINT_ERROR`（预计两版都 0，仅作对照）；
- `LANE_DEVIATION`、`IMPACT_HIGH`；
- 30 kg 不翻车；
- 偏移头统计（B 独有）。

## 4. 判定标准（先写死，避免事后解释）

B（加回 vx offset）只有在以下条件同时成立时才值得保留：

1. **15°/25 kg 的 FALL 明显低于 A**（不是只低 1–2 个 case）；
2. 小车速度通过数不低于 A，或至少 steady MAE 不退化；
3. `LANE/IMPACT/JNT` 不明显恶化；
4. 无载 15° 仍能通过；
5. 偏移头不是在“全程负偏压速度”，而是有可解释的相位依赖。

否则保留 A（v6，12 维）。

## 5. 当前不做什么

- 不改当前 50 Hz v6 训练 run；
- 不把旧 v5 的 13 维 checkpoint 直接混进 v6；
- 不把 v5 的 7/1050 FALL 当作本实验的基线（v5 没有 15°/25 kg 同口径）；
- 等当前 run 完成、看 50 Hz v6 的 15°/25 kg 侧翻率后，再决定是否启动 B。

## 6. 相关记录

- 去掉 vx 偏移头：`docs/towing/towing_reward_2026-10-19.md` / 2026-10-24 v6 提交
- upper1500 20 Hz 评测：`docs/towing/towing_upper1500_eval_2026-11-11.md`
- 老 PPO 全列：`docs/towing/towing_lowlevel_ppo_eval_2026-11-09.md`
- 50 Hz 更新：`docs/towing/control_frequency_50hz_2026-11-09.md`
