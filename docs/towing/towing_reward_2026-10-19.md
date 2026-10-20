# 拖曳训练奖励随验收口径调整（2026-10-19）

> 用户原话：「clearance 和 min_clearance 置为 0；建议新加的奖励都不需要；fall 只看高度?
> 那波浪条件会不会冲突?；collision 全域降低到 -1.0；顺便帮我统计一下，之前用 ppo 测试的
> 拉车质量上限是多少。」
>
> 本文记录**训练侧奖励改动、fall/波浪核对、PPO 质量上限统计与待确认项**。
> 验收口径见 [towing_eval_criteria_2026-10-19.md](towing_eval_criteria_2026-10-19.md)。

## 1. 改了什么（只动现有项，不新增奖励）

| 奖励项 | 旧权重 | 新权重 | 说明 |
|---|---:|---:|---|
| `clearance`（软间隙障碍） | −10.0 | **0.0** | 新验收口径下停车接触/间隙不是失败判据；该项的「贴近就重罚」会驱动“停车后往前走几步” |
| `min_clearance`（出生几何硬间隙） | −5.0 | **0.0** | 同上；它是「不许比出生时更近」的主要梯度 |
| `collision`（车斗/轮与机器人接触） | −5.0（下坡 ×0.2 = −1.0） | **−1.0（全域）** | 只保留安全底线；不再按坡度缩放 |

**没有新增任何奖励项**。`tracking_velocity`、`action_rate`、`action_magnitude`、
`feet_slide`、`low_level_pos_error`、`towing_force_y`、`towing_force_rate`、
`stop_towing_force` 全部保持原值。`fall` 的**权重/终止保留**，但判据按用户随后要求扩展为
「高度 + pitch + roll」（见 §2）。

代码落点：[upper_env_cfg.py](../../imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/upper_env_cfg.py)
的 `UpperRewardsCfg`。`clearance` / `min_clearance` 的**函数、params、语义全部保留**，
只是权重 0；`RewardManager` 对 `weight == 0.0` 直接跳过，所以这两项**不会产生 TensorBoard
曲线**。要恢复只需把权重改回；历史推导仍在行内注释里。

`cart_collision_cost(env, downhill_scale=1.0)` 的 `downhill_scale` 形参与
`downhill_segment` 函数保留供诊断/复用，但 `UpperRewardsCfg.collision` **不再传**该参数
（默认 1.0 ⇒ 全域同权）。

### 1.1 量级

- `collision`：每步 `−1.0 × 0.05 = −0.05`，约等于 `tracking_velocity` 满额（+0.05/步）
  的 1 倍；顶住 1 s 约 −1，不再像旧 −5（每步 −0.25）那样主导早期学习。
- `clearance` / `min_clearance`：权重 0 ⇒ 每步 0；**不产生 TB 曲线**。

## 2. `fall` 的高度/pitch/roll 与波浪核对

### 2.1 当前事实（2026-10-19 第二轮已改）

- **训练侧**：`UpperRewardsCfg.fall` 与 `UpperTerminationsCfg.robot_fall` 都调用
  `mdp.robot_fall(minimum_height=0.18, pitch_limit_rad=0.8, roll_limit_rad=0.8)`。
  高度部分 = `local_z − surface_height`，`surface_height` 按列分派到
  `slope/rolling`、`speedbump_profile_height`、`wave_profile_height`（2026-10-16 混合网格）。
  **姿态部分** = `euler_xyz_from_quat` 的 `|pitch| > 0.8` 或 `|roll| > 0.8`（**瞬时**判据，
  不是评估侧的 20% 窗口）。
- **评估侧**：`stability_stats` 的
  `fell = min(surface_height) < 0.15 m` 或 `pitch_over_limit_frac > 20%` 或
  `roll_over_limit_frac > 20%`（roll 从 base 四元数解出，默认上限 0.8 rad）。

### 2.2 波浪冲突核对（实测旧 run）

| run | 地形 | `min_robot_surface_height_m` | 说明 |
|---|---|---:|---|
| `ppo_wave_v3` | 波浪列 50–59 | **0.2147 m**（p5 0.2267、中位 0.2331） | 100 case，`fell` 0/100；离 0.18 m 还有 3.5 cm |
| `ppo_bump_v3` | 减速带列 40–49 | **0.1872 m**（p5 0.1900、中位 0.2083） | 100 case，`fell` 0/100；离 0.18 m 约 7 mm |
| `model1000_flat` | 平地 0–19 | 0.1760 m（3/100 < 0.18，0/100 < 0.15） | 说明 0.18 比评估的 0.15 更严 |
| `model2000_flat` | 平地 0–19 | 0.1620 m（9/100 < 0.18，0/100 < 0.15） | 同上 |

**结论**：

1. **波浪不会把“高度 < 0.18 m”误判成跌倒**：高度是相对局部波浪面的，振幅 ±0.10 m 被
   `wave_profile_height` 抵消；实测最低 0.2147 m，留 3.5 cm 裕度。
2. **波浪也不会误触发 pitch/roll 判据**：波浪最大侧坡 15.6° = 0.272 rad，减速带侧坡
   26.565° = 0.464 rad，10° 坡 = 0.175 rad，全部低于 0.8 rad；正常过波/过带/爬坡不会触发。
   30° 坡（0.524 rad）也仍有 0.276 rad 余量，但当前任务没有 30° 地形。
3. **训练 fall 的 0.18 m 比评估的 0.15 m 更严**：最近两轮平地上有 3/100、9/100 个 case
   最低高度在 0.15–0.18 m 之间 —— 训练会在这些时刻终止，评估却不判 `fell`。`fall` 的
   `minimum_height` 当前**未对齐**（仍是 0.18）；若要训练/评估同口径，可改到 0.15。
4. **纯侧翻现在已被训练 fall 抓到**（`|roll| > 0.8`），纯俯仰也抓到（`|pitch| > 0.8`）。
   这与评估的 `fell` 方向一致；区别是训练用瞬时阈值，评估用 20% 样本占比。

## 3. PPO 测试的拉车质量上限（统计）

### 3.1 最高测过的质量

| run | 低层 | 质量档 kg | 速度 m/s | 环境/地形 | 结果 |
|---|---|---|---:|---|---|
| `mass_sweep_stage1` | PPO@2500 | **20 / 25 / 30 / 40 / 50 / 60** | 0.5 / 1.0 | 800 env，列 0–39（平地/5°/10°），10 m STOP | **最高 60 kg**；旧口径 304/800 OK |
| `big_terrain_ppo2500_20261018` | PPO@2500 | 5 / 10 / 15 / 20 | 0.5 / 1.0 | 1200 env，全网格 | 最高 20 kg |
| `model1000_flat` / `model2000_flat` | PPO@2500 | 10 / 20 / 30 | 0.5 / 1.0 | 100 env，平地 + 上层 checkpoint | 最高 30 kg |
| `ppo_flat_d2` / `ppo_g5_d2` / `ppo_g10_v3` / `ppo_bump_v3` / `ppo_wave_v3` | PPO@2500 | 5 / 10 / 15 / 20 / 25 | 0.5 / 1.0 | 100 env，按地形列 | 最高 25 kg |

**答案：PPO 测试过的最高拉车质量是 60 kg**（`mass_sweep_stage1`，PPO@2500，800 env，
20/25/30/40/50/60 kg，列 0–39）。注意该轮是 `--policy ppo` + 上层网络关闭（基线脚本指令），
且训练侧质量域当时是 **[5, 20] kg**，25–60 kg 全部是**外推**。

### 3.2 按“不翻车”看的安全上限

`mass_sweep_stage1` 逐质量的 `FALL`（旧 `verdict`，JNT 未计）：

| 质量 kg | case 数 | `FALL` | 分布 |
|---:|---:|---:|---|
| 20 | 133 | **0** | — |
| 25 | 134 | **0** | — |
| 30 | 134 | **24** | 全部 10° 坡（24/34 个 10° case） |
| 40 | 133 | **24** | 全部 10° 坡 |
| 50 | 133 | **29** | 全部 10° 坡 |
| 60 | 133 | **26** | 10° 25 个 + 5° 1 个 |

**在该轮网格上，“不翻车”的经验上限是 25 kg**；30 kg 起 10° 坡已经系统性 `FALL`
（30 kg 的 10° 档 24/34 翻）。这与用户当前“拉重车不翻车”的验收项直接相关：
如果 30 kg 是训练域上限，10° 坡是当前最明显的失效面。

⚠ 口径限制：

- 该轮是**旧判定码**（JNT 默认不计入）；按 2026-10-19 任务口径，所有拖车 case 的关节静差
  仍会判 `JOINT_ERROR`，所以“25 kg 安全”只表示**不翻车**，不表示任务通过。
- 该轮 25–60 kg 超出当时训练域 [5, 20]，不能当作“策略在域内能拉 60 kg”的证据。
- 30 kg 的 `FALL` 只在 10° 坡出现；平地/5° 的 30 kg 没有 `FALL`（但旧 `COL/SPD` 仍很多）。

### 3.3 30 kg 专项训练命令

`train.py` 新增 `--mass-range MIN MAX`（默认仍取 env cfg 的 5–30，范围必须满足
`5 ≤ MIN ≤ MAX ≤ 30`）。先跑 25–30 kg：

```bash
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/rl_lab/towing/train.py \
    --task=Imgo2-towing-upper-rl-lab --agent=rl_lab_cfg_entry_point \
    --headless --compact-log --num_envs 1200 --max_iterations 3000 --save_interval 100 \
    --mass-range 25 30 \
    --policy ppo \
    --policy_checkpoint=logs/towing_rl_lab/policies/ppo_2500/policy.pt \
    --run_name=reward_20261019_ppo2500_m25to30
```

需要严格固定 30 kg 时把 `--mass-range 25 30` 改成 `--mass-range 30 30`。默认 5–30 kg
训练仍可用（不传该参数），但 30 kg 只占连续质量域的一小段，重车专项建议显式收窄。

## 4. 验证

- `pytest imgo2_rl/tests -q` ⇒ **749 passed, 4 skipped, 1336 subtests**。
- 改动文件 `py_compile` 通过；`git diff --check` 干净。
- 相关测试已更新：
  - `tests/test_towing_reward_weights_2026_10_17.py`：`clearance` 0.0、`collision` −1.0
    （无 `downhill_scale`）；保留 `action_rate` / `cart_collision_cost` 函数测试；
  - `tests/test_towing_upper_rl_contract.py`：`min_clearance` 0.0、`collision` −1.0 全域；
  - `tests/test_towing_force_rate.py`：`min_clearance` 0.0。

## 5. 未验证 / 待训练机

1. **实跑**：本机无 GPU/Isaac Lab，权重改动一次仿真未跑。需要短跑 64 env × ~20 轮，
   看 `Episode_Reward/collision` 是否从旧 −0.25/步降到 −0.05/步、`clearance`/
   `min_clearance` 曲线消失（weight 0 不记录）后：
   - 停车段是否还出现“为避碰而继续前进”的行为；
   - 追尾/接触率是否上升（预期会，因不再重罚）；
   - `tracking_velocity` 是否成为主导项。
2. **`fall` 高度阈值**：训练的 0.18 m 与评估的 0.15 m 仍有差异（§2.2）；pitch/roll 已加，
   高度是否对齐到 0.15 待用户确认。
3. **mass sweep 的口径**：60 kg 是旧判定码/旧训练域下的外推；新口径下需要重跑才能给出
   “任务通过的质量上限”。若要回答“重车不翻车的上限”，至少应在新代码下复跑
   20/25/30/40 kg × 10° 列（或全网格）。
4. **没有新增奖励**是用户本轮明确要求；因此 JNT/横向/冲击仍只由**评估**约束，训练不会
   直接优化它们。若后续发现这三项不改善，再讨论加项（需契约/观测配套）。

## 6. 复现

```bash
# 离线测试
python3 -m pytest imgo2_rl/tests -q

# 看权重注册（源码）
grep -n "clearance = RewTerm\|min_clearance = RewTerm\|collision = RewTerm" \
    imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/upper_env_cfg.py

# 质量扫描报告
sed -n '1,80p' imgo2_rl/logs/towing/play_test/mass_sweep_stage1/report.md
```
