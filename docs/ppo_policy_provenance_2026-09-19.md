# 键 1（PPO）权重来源复核（2026-09-19）

本轮起因：用户提出「`~/RL/sim2sim/Imgo2_deploy` 在 6 月 30 号附近有一个训练效果非常好的 PPO 版本，
已经部署了，要移植到我们这边的 ppo 文件夹」，但随后表示**记不清具体是哪一份**，先问「目前的 ppo 是哪一个」。

本记录只做**来源核对与字节核验**，不涉及训练、不替换策略。结论见 §2，用户决定见 §6。

## 1. 参考项目的目录语义

参考项目 `~/RL/sim2sim/Imgo2_deploy/policy/imgo2/` 下没有 `ppo/` 目录：

| 参考目录 | 算法 | 我们的对应目录 |
|---|---|---|
| `base_move/` | 普通 PPO（45 维 actor） | `imgo2_deploy/policy/imgo2/ppo/` |
| `himloco/` | HIM-Loco（45 维 × 6 帧） | `imgo2_deploy/policy/imgo2/himloco/` |
| `amp/` | AMP（45 维 actor + 判别器） | `imgo2_deploy/policy/imgo2/amp/` |

所以「参考项目 PPO 训练出来的部署件」= `policy/imgo2/base_move/` 里的权重。

## 2. 参考侧 `base_move/` 的文件清单（6 月 30 号前后）

三份文件**架构相同**（TorchScript 归档均为 31 个成员、`policy/code/__torch__/rsl_rl/models/mlp_model.py`、
45 维输入；文件大小同为 770326 B），互相之间只是权重不同：

| 文件 | 时间 | sha256 | md5 |
|---|---|---|---|
| `policy1.pt` | 2026-06-26 04:50 | `20afa4a9a79c30af…` | `614742af78bf502c…` |
| `policy_flat.pt` | 2026-06-30 12:45 | `53a57f909b61d2a7…` | `e083a385b00bc504…` |
| `policy.pt` | 2026-06-30 16:52 | `364f44bb33934e99…` | `e9e10008d4a59a8c…` |

关键区别：`base_move/config.yaml`（2026-06-30 00:58）里写的是

```yaml
imgo2/base_move:
  model_name: "policy.pt"
```

⇒ **参考项目实际加载/部署的是 16:52 那份 `policy.pt`**，而不是 12:45 的 `policy_flat.pt`
（后者是同一天中午单独存下来的一个版本）。`policy1.pt` 是 6/26 的早期版本，没有出现在 `model_name` 里。

## 3. 我们当前 ppo 是哪一份

| 项 | 值 |
|---|---|
| 权重文件 | `imgo2_deploy/policy/imgo2/ppo/policy.pt` |
| sha256 | `53a57f909b61d2a708d24d6c08a90e086409b6d4fce4fdd3e6f3e1849abce7cf` |
| 等于参考的 | **`base_move/policy_flat.pt`（2026-06-30 12:45）**，逐字节相同 |
| 入库提交 | `ee0a669`（2026-09-17「建立 PPO 四项评估标准并完成选型（PPO-01）」） |
| 配置 | `imgo2_deploy/policy/imgo2/ppo/config.yaml`，数值逐项抄自参考 `base_move/config.yaml` |

即：**我们 ppo 文件夹里的就是 6/30 那天的版本之一**，只是选了 12:45 的 `policy_flat.pt`，
不是参考侧 `model_name` 指向的 16:52 那份。配置数值对两份文件是一样的
（`rl_kp 25 / rl_kd 0.5`、`default_dof_pos 0/0.87/-1.82`、`action_scale 0.125/0.25`、`clip ±3`、
观测顺序 `ang_vel/gravity_vec/commands/dof_pos/dof_vel/actions`、`joint_mapping` 恒等），
所以**换文件是纯字节替换，不需要动配置**。

## 4. 两份 6/30 文件的实测差别

下表是 2026-09-17 在本机 Gazebo 无头链路上用 `imgo2_deploy/scripts/eval_gazebo_policy.py`
（自己注入 A→键 1→`axes[1]=vx`，从 `/odom` 真值 + `/joint_states` 统计）跑出的结果，`vx=0.5`，
判据是用户定的四项：位姿 / 速度跟随 / 抖动 / 周期性。原始数字见
[Gazebo 链路打通记录](gazebo_ros2_bringup_2026-09-17.md) 第 9 节与 README §5.4。

| 候选 | 实测速度 | z_mean | roll/pitch | yaw 漂 | 抖动 thigh/shank | 周期强度 | FL-FR |
|---|---|---|---|---|---|---|---|
| `policy_flat.pt`（我们现在用的） | 0.428 m/s | 0.288 | 2.5°/1.3° | −0.5°/s | 45 / 94 | 0.95 | +175° |
| `policy.pt`（参考侧 `model_name` 的那份） | 0.522 m/s | 0.251 | 5.0°/4.6° | +1.5°/s | 79 / 153 | 0.74 | +198° |

`vx=0` 静止时两份都抖动 0.0、yaw 漂 0.0。⇒ 分辨记忆里的「效果好」：偏**步态干净/稳** ⇒ `policy_flat.pt`（现在这份）；
偏**跟速度准/跑得利索** ⇒ `policy.pt`（16:52 那份，未移植）。

## 5. 依据与可复现方法

只用了标准库与 `sha256sum`，在没有 NVIDIA 驱动、没有 torch 的机器上也能复现（本机 `python3` 无 `torch` 模块）。

```bash
# ① 字节一致性
sha256sum imgo2_deploy/policy/imgo2/ppo/policy.pt \
  ~/RL/sim2sim/Imgo2_deploy/policy/imgo2/base_move/policy_flat.pt

# ② TorchScript 归档完整性（zipfile 是标准库；torch 的 .pt 就是 zip）
python3 - <<'EOF'
import zipfile, hashlib, os
for p in ['imgo2_deploy/policy/imgo2/ppo/policy.pt',
          os.path.expanduser('~/RL/sim2sim/Imgo2_deploy/policy/imgo2/base_move/policy_flat.pt'),
          os.path.expanduser('~/RL/sim2sim/Imgo2_deploy/policy/imgo2/base_move/policy.pt')]:
    with zipfile.ZipFile(p) as z:
        print(os.path.basename(p), z.testzip(), len(z.namelist()))
EOF

# ③ 给导出「验明正身」：按 storage 成员 md5 求交集（torch .pt 的每个 tensor 字节就是一个 zip 成员）
#    见本记录 §5.2，脚本思路同 README §8.3 的 2026-09-17 那条记录
```

**验证结果**：① 两份 sha256 完全相同；② 三份文件 `testzip()` 均为 `None`、成员数均为 31；
③ 本机 `~/RL/isaac/Imgo2_rl/logs` 下全部 177 个 checkpoint（含 `exported/`）与三份导出的 storage
**零命中**。

### 5.1 结论性推论（来自 ③）

三份 `base_move` 导出**不是**本机 `~/RL/isaac/Imgo2_rl` 那 5 个 run（2026-06-19 ~ 06-21）的产物。
本机 `~/RL/isaac/Imgo2_rl/README.md` 的 `play.py` 示例写的是
`/root/gpufree-data/Imgo2_rl/logs/...` ⇒ 训练在**训练服务器**上做，6/26、6/30 的导出是拷回本机的；
本机没有它们的 checkpoint，因此**无法**从 checkpoint 侧复核这三份的轮数、奖励或训练曲线。

### 5.2 一个容易混淆的同名文件

`~/RL/sim2sim/rl_sar/policy/imgo2/base_move/policy.pt`（2026-06-24 06:18，sha256 `454e9dba…`）
等于参考项目 **git HEAD** 里的那份（`git show HEAD:policy/imgo2/base_move/policy.pt`，md5 `01ca0a1c…`），
对应旧口径配置（`default_dof_pos 0/0.8/-1.5`、`clip ±100`、`kd 0.8`）。它**不是** 6/30 的新文件，
不要拿它当「6/30 版本」。

## 6. 用户决定与本轮改动

用户 2026-09-19 先选择**维持现状**，随后**确认记忆中的「6 月 30 号附近训练效果非常好的 PPO 版本」
就是现在这一份**（6/30 12:45 的 `policy_flat.pt`，即 `ppo/policy.pt` 的内容）。

⇒ 「把 6/30 的版本移植到我们 ppo 文件夹」这一诉求**已经满足**，无需再移植：该文件自 `ee0a669`
（2026-09-17）起就是键 1 的部署权重。本轮**没有改动任何策略文件、配置或 C++**；只在文档层记录核对结论
（新增本记录、README §5.4 加一句复核结论、§8.3 加一行维护记录、顶部核对日期改 2026-09-19）。

## 7. 未做 / 待确认

- **未做** libtorch / Gazebo 行为复测：文件在用、字节未变，行为与 `ee0a669` 时一致；本轮只做来源与完整性核验。
- **未移植**参考侧 `policy.pt`（6/30 16:52）。若日后更看重速度精度，可先按 README §5.4 的协议用
  `eval_gazebo_policy.py` 同协议复测，再决定是否替换（纯字节替换，配置无需改）。
- **未复现训练**：本机无 NVIDIA 驱动与训练环境，6/30 两次训练的实际轮数、奖励设置与收敛情况未知；
  三份导出的训练来源只有文件时间和参考项目的 `model_name` 两条线索。
- 参考项目里没有留下 6/30 的训练日志或评测记录（`log/` 只有 `build_*` 构建日志），
  所以「哪一份训练效果更好」在本机只能靠 §4 的部署侧四项指标判断。
