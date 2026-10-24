> **2026-10-24 v6 变更（本文以下内容多为 v5 历史）**：用户要求去掉 1 维 vx 偏移头，动作改为 **12 维纯关节残差**；policy 帧 58→57、actor 79→78、critic 73→72、`towing_contract` v6。旧 v5 checkpoint/`policy.pt` 全部失效，必须从零重训后重新导出。

# model_1300 全列拖曳评测（2026-10-20）

> 用户 2026-10-19 要求「跑全列」，用训练 run
> `2026-10-19_17-44-30_reward_20261019_ppo2500` 的最后 checkpoint `model_1300.pt`
> 跑 `play_towing_test.py`。本机无 GPU，实际在训练机执行；本文记录完整结果。

## 1. 运行条件

- 场景：训练场景 60 列 × 20 行 = **1200 env**（全列：平地/5°/10°/减速带/波浪）
- 低层：PPO@2500；上层：`model_1300.pt`
- 速度 0.5/1.0 m/s；质量 10/20/30 kg；`--no-cart-fraction 0.125`（150 个无小车锚点）
- `--stop-progress 10 --tow-duration 24 --coast-duration 5`
- 结果：`state=completed`，`elapsed_s=5649`（约 94 min），`report.md/json` 完整
- 路径：`imgo2_rl/logs/towing/play_test/model1300_full/`

## 2. 总结果

- 可判 case **1050**（1200 − 150 无小车锚点）
- **任务通过 0/1050**
- 判定码：`JOINT_ERROR 1018`、`NA 150`、`LANE_DEVIATION 23`、`FALL 7`、`IMPACT_HIGH 2`

| 检查 | 通过 |
|---|---:|
| 小车速度跟随稳定性 | 275/350 |
| 横向/朝向保持 | 1020/1050 |
| 关节静差（JNT） | **0/1050** |
| 启停冲击 | 1048/1050 |
| 不翻车 | **1043/1050** |

失败原因（可叠加）：`startup_joint_error 1050`、`stop_joint_error 1043`、
`speed_unstable 75`、`lane_deviation 30`、`robot_fell 7`、`robot_rollover 7`、
`speed_dropout 2`、`start_stop_impact 2`、`speed_mae_high 1`。

## 3. 30 kg 不翻车：342/349，但波浪/减速带有 7 次真翻车

| 地形 | 30 kg case | 不翻车 | 横向失败 |
|---|---:|---:|---:|
| 平地 0° | 116 | **116** | 2 |
| 5° | 57 | **57** | 0 |
| 10° | 59 | **59** | 6 |
| 波浪 15.6° | 59 | **56** | 10 |
| 减速带 26.565° | 58 | **54** | 9 |

- **7 个 `FALL` 全部是 30 kg**：4 个在减速带（0.5 m/s，compliant/inextensible）、3 个在波浪（1.0 m/s，compliant/inextensible）。
- 这些 case 的 `roll ≈ 3.13–3.14 rad`、离面高度 **0.07–0.09 m**，是**真侧翻/倒扣**，不是阈值噪声。
- 平地、5°、10° 的 30 kg **全部不翻车**；20 kg 351/351、10 kg 350/350 也全部不翻车。

## 4. 小车速度稳定性：275/350，失败集中在 10°/波浪/减速带

10 kg（唯一判速度的档）：

| 地形 | 速度通过 |
|---|---:|
| 平地 0° | 117/117 |
| 5° | 56/56 |
| 10° | 35/61（57.4%） |
| 波浪 15.6° | 24/61（39.3%） |
| 减速带 26.565° | 43/55（78.2%） |

- 平地和 5° 全过；**10°、波浪、减速带**是速度不稳定/掉速的主要来源。
- 75 个速度失败中，`speed_unstable` 75、`speed_dropout` 2、`speed_mae_high 1`（原因可叠加）。

## 5. 横向与冲击

- 横向 1020/1050：23 个 `LANE_DEVIATION`，其中 30 kg 波浪 10、减速带 9、10° 6、平地 2；20 kg 平地/10°/减速带各 1；10 kg 0。
- 启停冲击 1048/1050，仅 2 个 `IMPACT_HIGH`（20 kg 10° 附近）。
- 横向失败与 30 kg 波浪/减速带翻车高度重合，优先查横向 PD/负载侧向力。

## 6. 结论

1. **“30 kg 不翻车”在平地和 10° 已经成立**，但**全列不成立**：波浪 3 次、减速带 4 次真翻车，roll≈π，必须修。
2. 任务通过 0/1050 的主因仍是 **JNT 0/1050**（旧 0.10/0.30 rad 阈值，训练未直接优化），不是这次训练退化。
3. 小车速度在 10°/波浪/减速带明显掉档，是下一个主要失效面。
4. 下次专项建议：
   - 30 kg × 波浪/减速带：先查横向失败与 roll 触发时刻，考虑加横向/roll 训练项或降低这两族的速度；
   - 10 kg × 10°/波浪：速度稳定性（std/p95）专项；
   - JNT：决定重标阈值还是把关节静差列为训练目标。

## 7. 复现

```bash
CKPT=logs/towing_rl_lab/towing_upper/2026-10-19_17-44-30_reward_20261019_ppo2500/model_1300.pt
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/towing/play_towing_test.py \
    --headless --num-envs 1200 \
    --velocities 0.5 1.0 --cart-masses 10 20 30 --no-cart-fraction 0.125 \
    --stop-progress 10 --tow-duration 24 --coast-duration 5 \
    --write-csv none --compact-log \
    --policy ppo --policy-checkpoint=logs/towing_rl_lab/policies/ppo_2500/policy.pt \
    --upper-checkpoint="$CKPT" \
    --output-dir=imgo2_rl/logs/towing/play_test/model1300_full
```

## 8. 部署目录归档与导出（2026-10-24）

### 8.1 `play.py --export` 与 `policy.pt`

用户要求把本次评测对应的 towing policy 放到部署目录并用 play 导出替换。核查后全仓只有
`imgo2_rl/scripts/rl_lab/towing/play.py`，原本没有 export 分支；本次新增：

- `--export` / `--export-dir`：在 import Isaac Sim 之前用纯 CPU torch 执行，不启动 AppLauncher；
- `upper_policy_runtime.TowingPolicyExport`：把 decoder/actor 两套 GRU hidden 作为模块 buffer，
  `@torch.jit.export reset()` 清零；`forward` 输入 `[1,58]` frame，输出 `[1,13]` clamp 后的
  归一化动作；
- `export_towing_policy()`：从 checkpoint 构网、`torch.jit.script`、落盘并立即重载/前向自检。

产物：

- `imgo2_deploy/policy/imgo2/towing/policy.pt`：1,941,632 B，
  sha256 `36313d2a132c3253602f6841dd2e7e4f71b0297d55f5c579e622636af5b0550d`；
- 同目录 `model_1300.pt`：10,245,431 B，
  sha256 `a54780813029b2764fc651593e2b20b87bef8e060b524eba67ce1195697f9e05`（训练态 checkpoint 留档）。

### 8.2 验证与限制

- `play.py --export` 在本机 CPU 跑通；`torch.jit.load` + `reset()` + `[1,13]` 前向通过；
- 与 checkpoint 的 actor/decoder 在同一批 5 帧上逐拍对照，max diff = **0**；
- `pytest imgo2_rl/tests/test_towing_upper_policy_runtime.py` **30 passed**；
- **仍未完成**：`config.yaml` 缺失、FSM 没有 towing 状态、C++ `Model::forward` 没有 reset 入口、
  libtorch 实载未验证、action_scale/偏移头接线未做。因此 TOW-05 仍是未完成；`policy.pt`
  是可加载的导出件，不是已接入部署的策略。
