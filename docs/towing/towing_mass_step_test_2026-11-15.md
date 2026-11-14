# 质量突变测试（2026-11-15）

> 目的：在 tow 段中途把小车质量从 10 kg 突变到 30 kg，观察上层 latent/动作是否响应，
> 并与固定 10/30 kg 对照。**当前只完成脚本和离线/参数校验，未在 GPU 上实跑仿真。**

## 1. 新增入口

### `play_towing_test.py`

新增参数：

```text
--mass-step-at-s 5.0
--mass-step-from 10
--mass-step-to 30
--latent-trace
```

- `--mass-step-at-s`：**tow 段开始后**的秒数（不是整回合时间），1.0 m/s 时 5 s≈5 m；
- 三个 mass-step 参数必须同时给；
- `--mass-step-from` 必须与所有 case 的初始 `--cart-masses` 一致（避免悄悄漏掉部分 env）；
- `--latent-trace`：逐记录步写 `latent_trace.csv`（16 维 latent + 当前质量 + phase），
  需要 `--upper-checkpoint`。

质量突变时同时更新 `base_link` 的：

- mass
- inertia
- CoM

使用 `assets/cart_mass.body_properties` 的解析式，不做整车等比缩放；不 reset 环境。

### `mass_step_sweep.py`

驱动脚本：

```bash
bash imgo2_rl/scripts/run_isaaclab.sh \
    imgo2_rl/scripts/towing/mass_step_sweep.py --dry-run
```

确认 6 条命令后再执行：

```bash
bash imgo2_rl/scripts/run_isaaclab.sh \
    imgo2_rl/scripts/towing/mass_step_sweep.py
```

默认矩阵（平地 0°、1.0 m/s、每连接 160 env）：

| 连接 | 列 | 配置 |
|---|---|---|
| compliant | 0–7 | 固定 10 kg |
| compliant | 0–7 | 固定 30 kg |
| compliant | 0–7 | **10 → 30 kg @ tow+5 s** |
| rigid | 8–15 | 固定 10 kg |
| rigid | 8–15 | 固定 30 kg |
| rigid | 8–15 | **10 → 30 kg @ tow+5 s** |

可选反向卸载：

```bash
... mass_step_sweep.py --include-reverse
```

会额外加 `30 → 10 kg` 两组。

## 2. 产物

每次 run 的目录：

```text
<output-root>/<connection>_<fixed10|fixed30|step10to30>/
  report.json / report.md
  latent_trace.csv       # 逐记录步：step/time/phase/cart_mass_kg/z0..z15
  experiment.json
```

`report.json` 和 `experiment.json` 的 `mass_step` 段记录：

```json
{
  "at_tow_elapsed_s": 5.0,
  "from_kg": 10.0,
  "to_kg": 30.0,
  "scope": "all envs; cases must all start at from_kg",
  "note": "cart base_link mass/inertia/com are updated together; no reset"
}
```

## 3. 建议分析

对每个连接，比较：

1. **固定 10 vs 固定 30**：同一时刻 latent 的差；
2. **step 10→30**：突变前/后短窗口 latent 的漂移；
3. **step 30→10**（可选）：卸载后是否回归；
4. 同时看 `upper_offset_vx`、动作幅值、vx、roll、绳张力、是否 FALL。

可用现有工具画 latent：

```bash
python3 imgo2_rl/scripts/tools/analyze_latent_clusters.py \
    --report <run>/report.json --phase tow --plots --connection-binary
```

但注意：`latent_trace.csv` 才是逐时刻响应；`--dump-latent` 仍只是 phase 均值。

## 4. 验证与限制

- 全量 `pytest imgo2_rl/tests -q`：**742 passed, 4 skipped, 1336 subtests**；
- `mass_step_sweep.py --dry-run` 通过，6 条命令已打印；
- 新增 CLI 校验测试：初始质量不匹配、缺参数、latent-trace 缺 upper 都会拒绝；
- **未在 GPU 实跑**：`set_masses/set_inertias/set_coms` 在仿真运行中的行为、瞬态冲击、
  `summarize_tow` 用最终质量的口径都需训练机验证；
- 质量突变是瞬时阶跃；如果机器人直接翻车，先试 10→20 kg 或加平滑过渡；
- 该测试只改评测场景，不改训练/策略权重；当前 v6 model2000 的 contract 不变。

## 5. 限制说明

- `summarize_tow` 的 `cart_mass_actual_kg` 在跑完后取最终质量；step case 的 pre-step
  绳索摘要可能被 final mass 影响，定量判读以 `latent_trace.csv` 与逐行 `cart_mass_kg` 为准；
- `latent_trace.csv` 是 `--record-every` 的采样（默认每 5 物理步），不是每个 5 ms 子步；
- 该实验只测“突变后的响应”，不训练自适应质量估计。
