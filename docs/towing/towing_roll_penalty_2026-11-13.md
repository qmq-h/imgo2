# 连续 roll 惩罚（2026-11-13）

> 用户要求「考虑增加 roll 方向的惩罚」，随后确认在 `main` 与 `imgo2_dvx` 都加。
> 当前只改训练奖励；已启动的 v7 run 不受影响，新惩罚在下一次从零训练生效。

## 1. 公式

```text
roll_excess = relu(|roll| - deadband_rad)
roll_penalty = roll_excess²
```

- `roll`：与 `robot_fall` 同一套世界系 `euler_xyz_from_quat` 的 roll；
- `deadband_rad = 0.15`（约 8.6°）；以下精确为 0；
- 0.8 rad 的 `fall` 终止判据不变；
- 注册：`roll = RewTerm(func=mdp.roll_penalty, weight=-10.0, params={"deadband_rad": 0.15})`。

## 2. 权重标定（50 Hz，`step_dt=0.02`）

每步 = `weight × excess² × 0.02`：

| roll | excess² | 每步 | 持续 1 s |
|---:|---:|---:|---:|
| 0.15 | 0 | 0 | 0 |
| 0.30 | 0.0225 | −0.0045 | −0.225 |
| 0.45 | 0.0900 | −0.0180 | −0.900 |
| 0.60 | 0.2025 | −0.0405 | −2.025 |
| 0.80 | 0.4225 | −0.0845 | −4.225（随后 fall 终止） |

权重先取 **−10**；0.6 rad 时约为满额 tracking 的 2 倍/秒，但不至于在 0.15–0.25 rad
的正常姿态上持续扣分。

## 3. 为什么用世界系 roll

与 `robot_fall` 同源，避免“奖励用相对 roll、终止用世界 roll”两套口径。
15° 平滑坡只改变 pitch，正常通过时 roll≈0；减速带/波浪造成真实侧倾才罚。

## 4. 验证

- main：**741 passed, 4 skipped**（新增源码守卫）
- `imgo2_dvx`：**746 passed, 4 skipped**

## 5. 限制

- 未训练；−10 是首版，需按 15°/25–30 kg 的 FALL 与 tracking 是否被压得太狠再调；
- 若策略为了避 roll 明显降速/停滞，需下调权重或提高 deadband；
- 当前 v7 run 使用旧 fall −50 且无 roll，不能评价本次改动。
