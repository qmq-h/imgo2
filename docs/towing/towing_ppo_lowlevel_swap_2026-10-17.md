# 冻结低层策略换成 PPO（`--policy ppo` / `--policy-checkpoint`，2026-10-17）

用户口径：「目前计划用 **ppo 替换底层的策略**，先用
`/root/Desktop/imgo2/logs/rsl_rl/base_move_rough/2026-10-16_21-26-48/model_4000.pt` 试试，
来跑**不带上层**测量的测试，设置 **1000 个环境**」。

结论：必要性测试台的冻结低层现在可选 `amp` / `ppo`，并新增 `--policy-checkpoint` 覆盖
**权重文件**（契约仍取 `--policy` 的名字）。两个底层的**接口逐项相同**，只有权重不同。

## 1. 为什么能直接换：接口逐项相同

| 契约项 | AMP（既有） | **PPO（本次）** | 依据 |
|---|---|---|---|
| 观测 | 45 = ang_vel 3 + gravity 3 + **commands 3** + dof_pos 12 + dof_vel 12 + last_action 12 | **相同** | `imgo2_deploy/policy/imgo2/{amp,ppo}/config.yaml` 的 `observations` 都是 `["ang_vel","gravity_vec","commands","dof_pos","dof_vel","actions"]` |
| 观测缩放 | 0.25 / 1.0 / 1.0 / 1.0 / 0.05 / 1.0 | **相同** | 同上（`ang_vel_scale`/`commands_scale`/`dof_pos_scale`/`dof_vel_scale`） |
| 动作 | 12，`scale = (0.125, 0.25, 0.25)×4`，`clip ±3` | **相同** | ppo `config.yaml` 的 `action_scale`/`clip_actions_*` |
| 默认姿态 | `(0, 0.87, −1.82)×4` | **相同** | ppo `config.yaml` 的 `default_dof_pos` |
| 关节顺序 | 逐腿 FL/FR/RL/RR（资产是广度优先，运行时用 `asset_permutation()` 重排） | **相同** | `base_move/rough_env_cfg.py` 的 `joint_names` 与 `_IMGO2_JOINT_NAMES` 逐项相等 |
| 控制周期 | `0.02 s`（dt 0.005 × decimation 4） | **相同** | 该 run 的 `params/env.yaml`：`decimation: 4` |
| 执行器 | `IMGO2_CFG`：stiffness 25 / damping 0.5 / effort 23.7 | **相同** | `params/env.yaml` 的 actuator 与 `assets/imgo2.py` 一致 |
| 观测归一化 | 无（导出 `normalizer=None`） | 无（该 run `empirical_normalization: false`） | `params/agent.yaml` |
| 结构 | TorchScript actor，45 → 12 | PPO `ActorCritic`（MLP 512/256/128，ELU，非 recurrent）⇒ 45 → 512 → 256 → 128 → 12 | 该 run 的 `model_state_dict`：`actor.0.weight (512, 45)` … `std (12,)`，`iter: 4000` |

命令包络也够：该 PPO run 训练时 `lin_vel_x ∈ (−1.5, 3.0)`、`lin_vel_y ±1.5`、
`ang_vel_z ±1.5`；拖曳任务只用 vx 0.5–1.5 m/s、PD 的 |vy| ≤ 0.4、|wz| ≤ 0.8。

## 2. 代码改动

| 文件 | 改动 |
|---|---|
| `tasks/.../towing/utils/policy_cfg.py` | 新增 `PPO_POLICY`（照 `imgo2_deploy/policy/imgo2/ppo/config.yaml`，与 AMP 同接口）、注册进 `POLICIES`、新增 `with_model_path(cfg, path)`（只换权重、契约不变，权重不存在直接 `FileNotFoundError`）；`_AMP_JOINT_NAMES` 改名为共用的 `_IMGO2_JOINT_NAMES` |
| `scripts/towing/play_towing_test.py` | 新增 `--policy-checkpoint PATH`（覆盖冻结低层权重，打印横幅：policy 名 + 实际权重 + 契约维数）；`[plan]` 里新增「冻结低层 = …」一行；`experiment.json.arguments` 自动带上这两个参数 |
| `tests/test_towing_policy_contract.py` | 新增 `PPOContractTests`：与 `ppo/config.yaml` 逐项一致、与 AMP 契约**逐字段相等**、`with_model_path` 只换权重、部署 `policy.pt` 是 45→12 的 TorchScript |
| `README.md` / 本文 | 命令、状态与限制 |

## 3. 怎么跑（训练机，需 GPU）

```bash
cd /root/Desktop/imgo2

# ① 把 training checkpoint 导出成 TorchScript（部署同款入口，--headless 也会写 exported/）
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/rsl_rl/play.py \
  --task Imgo2-basemove-rough-ppo \
  --checkpoint logs/rsl_rl/base_move_rough/2026-10-16_21-26-48/model_4000.pt --headless
# 产物：logs/rsl_rl/base_move_rough/2026-10-16_21-26-48/exported/policy.pt

# ② 不带上层的测量台（1000 环境），冻结低层用 PPO 的权重
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/towing/play_towing_test.py \
  --policy ppo \
  --policy-checkpoint logs/rsl_rl/base_move_rough/2026-10-16_21-26-48/exported/policy.pt \
  --num-envs 1000 --headless --write-csv failed
```

说明：

- **`--num-envs 1000` 是网格前缀**：当前网格是 60 列 × 20 行 = 1200 cell，1000 = 16 整行
  （960）+ 第 17 行的前 40 列 ⇒ 减速带 / 波浪块各只覆盖 16 行（仍各有 160 条 lane），
  分布不是严格均衡（0° 340、5°/10° 各 170、减速带/波浪各 160）。要整张网格就用 1200。
- **不要覆盖部署产物**：`imgo2_deploy/policy/imgo2/ppo/policy.pt` 是已验收的部署件，
  `--policy-checkpoint` 故意绕开它。
- 该 run 的 `exported/policy.pt` 是 2026-10-17 15:29 导出的，**没有记录来自哪一轮**
  （目录里 model 到 4100）。要确保是 4000 轮，就按 ① 重新导出一次。
- 不带上层 = 基线：12 维关节残差恒 0、vx 偏移恒 0，冻结策略拿到的就是任务指令本身。

## 4. 验证（本机离线）

- `pytest imgo2_rl/tests -q` ⇒ **627 passed, 4 skipped, 30 subtests passed**（含新增的
  `PPOContractTests` 6 项与 `test_towing_rl_lab_config_wiring.py` 3 项守卫）。
- `--dry-run --policy ppo --policy-checkpoint <导出件>` 在**无 torch** 的 `/usr/bin/python3`
  下退出 0，`[plan]` 打印冻结低层与权重路径。
- `check_towing_slope_grid.py` ⇒ 42 项 OK（与本改动无关，回归确认）。
- 本机 **CUDA 不可见**（`bash imgo2_rl/scripts/run_isaaclab.sh --check` → `cuda available: False`；
  `nvidia-smi` 报 NVML 权限不足），所以**一次仿真都没跑**。

## 5. 待确认（TOW-LOW-01）

状态：**已修（接口/契约离线验证通过），待训练机试跑**。缺什么才能完成：

1. `play.py` 导出 `model_4000.pt` 后，用 `--policy-checkpoint` 跑通 60 环境短冒烟：
   确认 `FrozenLowLevelPolicy` 的 45 维契约自检通过、机器人能站住并跟 0.5–1.5 m/s；
2. 1000 环境一轮完整基线（11 s/回合），比较 PPO 与 AMP 两套低层在同条件下的
   速度跟踪 / 跌倒 / 停车读数（`--policy amp` 与 `--policy ppo` 各一轮，参数其余相同）；
3. 若 PPO 低层在坡/减速带/波浪上明显更稳，再决定是否把训练侧的冻结低层一起换掉
   （训练侧 `upper_env_cfg` 目前仍固定加载 `--policy`/部署件，本轮**没动**训练路径的默认值）。
