# 拖曳能力与 Direct Stop 边界扫描（2026-09-21）

## 目的

为上层强化学习确定训练能力域。扫描同时回答两个不同问题：

1. 机器人在给定速度、质量和阻力下能否保持稳态拖曳；
2. 稳态可拖时，未经 shaping 的 Direct Stop 是否会低余量或追尾。

停车追尾不等于“拉不动”，脚本必须分别报告这两类边界。

## 扫描维度

主扫描默认：

- 速度：`0.2, 0.4, 0.6, 0.8, 1.0 m/s`；
- 小车质量：`5, 10, 15, 20, 25 kg`；
- 单轮黏性阻尼：`0.008, 0.032 N·m·s/rad`；
- 地面摩擦：固定 `0.8`；
- 绳模型：默认 `compliant`，两类绳分别运行。

共 50 个 case、5 次 Isaac Sim 启动。`tow_drag.py` 在每个速度内顺序跑完质量 × 轮阻，避免每个 case 重启仿真器。

质量和轮阻是主边界：质量控制惯性，轮阻控制稳态牵引需求和停车滑行。地面摩擦是机器人牵引／打滑的二级边界；先固定 0.8，随后在主扫描的边界附近用 `0.4, 0.8, 1.2` 复核，不默认扩大到 150 个 case。

## 命令

只检查将执行的命令：

```bash
python3 imgo2_rl/scripts/towing/scan_towing_boundary.py --dry-run
```

运行 compliant 主扫描：

```bash
python3 imgo2_rl/scripts/towing/scan_towing_boundary.py \
  --rope-model compliant --output-dir imgo2_rl/logs/towing/boundary/compliant_main
```

在边界附近复核摩擦，例如只扫 15/20/25 kg、0.6/0.8/1.0 m/s：

```bash
python3 imgo2_rl/scripts/towing/scan_towing_boundary.py \
  --velocities 0.6 0.8 1.0 --cart-masses 15 20 25 \
  --wheel-dampings 0.008 0.032 --ground-frictions 0.4 0.8 1.2 \
  --rope-model compliant --output-dir imgo2_rl/logs/towing/boundary/compliant_friction
```

不可伸长绳单独复核：

```bash
python3 imgo2_rl/scripts/towing/scan_towing_boundary.py \
  --rope-model inextensible --output-dir imgo2_rl/logs/towing/boundary/inextensible_main
```

重新汇总已有产物而不启动仿真：

```bash
python3 imgo2_rl/scripts/towing/scan_towing_boundary.py \
  --summarize imgo2_rl/logs/towing/boundary/compliant_main
```

## 分类口径

| 分类 | 含义 |
|---|---|
| `tow_infeasible` | 原有 summary 无效，或稳态跟速比 `<0.8`；这是“拉不稳／工况不可用” |
| `stop_collision` | 稳态可拖，但 Direct Stop 后小车追到机器人 |
| `stop_margin_low` | 未碰撞，但最终车头间隙 `≤0.10 m` |
| `direct_stop_safe` | 稳态可拖且 Direct Stop 后间隙高于阈值 |
| `unknown` | 记录不足，不能分类 |

`boundary.json` 会按绳模型、地面摩擦、轮阻和质量给出：已测最大稳态可拖速度、已测最大 Direct Stop 安全速度、首次出现停止风险的速度。这里的“最大”都只指离散测试网格，不能外推为连续极限。

## 产物

- 每个速度／摩擦组合保留原始 `tow_drag.py` 目录与 CSV；
- 根目录 `boundary.csv`：逐 case 表；
- 根目录 `boundary.json`：分类定义、实际运行网格、逐 case 结果和边界表。

## 验证与限制

- 离线单元测试覆盖默认速度范围、非法输入、分类互斥、边界提取与产物读取。
- `--dry-run` 已确认默认生成 5 次启动、50 个 case。
- 本轮没有启动 Isaac Sim，因此尚无新的质量／摩擦边界数值。
- 当前小车阻力以轮轴黏性阻尼为主，尚未实现独立的 breakaway/Coulomb 阻力；地面摩擦扫描不能替代真实 breakaway 参数扫描。
