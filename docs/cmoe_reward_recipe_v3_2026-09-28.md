# CMoE 奖励配方 v3（2026-09-28）：对齐 PPO 的步态/姿态项 + 地形分区

用户 2026-09-28 决定（原话）："feet air time、slide，对齐 ppo；orientation 对齐但是不在障碍地形生效；相位核去掉。"
本记录写清**改了什么、依据是什么、验证到哪一步、还差什么**。

## 1. 四条改动（`CMoE_env_cfg.py`）

| 项 | 原值 | **新值** | 依据（PPO 对照） |
|---|---|---|---|
| `feet_air_time.weight` | 0.3 | **1.0** | PPO rough 用 1.0；我们只有 30% 的滞空/步幅压力 |
| `feet_slide.weight` | **0（被清零）** | **−0.05** | PPO rough 用 −0.05；这一项是"支撑脚不许打滑"，正对实测的**拖行**形态（level 6：FL duty 0.80 / RR 0.57、速度 0.127 vs 指令 0.633） |
| `flat_orientation_l2.weight` | −0.1 | **−5.0**（**障碍地形豁免**） | PPO rough 用 −5.0（相差 50×）；PPO 是**全地形**生效，这里比它宽松——过沟/上箱允许必要俯仰 |
| `feet_gait`（相位核） | 1.0 | **0（去掉）** | 相位核在这份高频步态上**本身饱和**（实测溢价 0.116/s = 跟踪项 5.0 的 2.3%）；而 **PPO 根本不用相位核**也能练出干净 trot ⇒ 留着只是多一个调不动的旋钮 |

**保留**：三项 1e-6 分类器探针（`gait_metric_trot/bound/pace`）与三项 `diag_*` —— 它们是逐列步态的唯一读数来源。

## 2. 新增：地形分区 + 掩码项

```python
EASY_TERRAIN_NAMES = ("flat", "hf_pyramid_slope", "hf_pyramid_slope_inv", "random_rough")   # 8/40 列
OBSTACLE_TERRAIN_NAMES = ("pyramid_stairs", "pyramid_stairs_inv", "boxes", "gap",
                          "hurdle", "mix", "narrow_stairs")                                  # 32/40 列
```
* 新增 `mdp.MaskedFlatOrientationL2`（与其它 `Masked*` 同款：`__init__` 缓存静态掩码、`__call__` 乘 `~mask`）；
* `flat_orientation_l2` 的 `free_terrain_names` = `OBSTACLE_TERRAIN_NAMES` ⇒ **只在易地形上生效**；
* 两张清单必须**互斥且并集 = 全部 `sub_terrains` 键**（11 类）——`tests/test_reward_terrain_partition.py`
  锁住这一点（历史教训：`forward_only_terrain_names` 漏项曾静默退回全向命令）。

## 3. 生效项数（`scripts/tools/check_reward_overrides.py`）

| 链 | 原 | 现 | 变化 |
|---|---|---|---|
| `cmoe` | 27 | **27** | `feet_gait` 出局、`feet_slide` 入列（净 0） |
| `cmoe-gaitfree` | 22 | **23** | ⚠️ `feet_gait` 在两条链上都是 0 ⇒ gaitfree 相对 cmoe 只再少**四项**手工 shaping |

⚠️ **连带效果（有意）**：`feet_slide −0.05` 与 `flat_orientation_l2 −5.0` **不在 gaitfree 的归零名单里**
⇒ 你现在训练的 `-gaitfree` 任务在**下一次新 run** 里也会吃到这两项（它们是"步态质量/姿态"，不是手工步态风格）。

## 4. `base_height_l2` 是不是按地形有偏移的？——**是，但机制是"每步实测局部地面"，不是手写偏移表**

* 配置：`rough_env_cfg.py:108-110` 设 `weight = −10.0`、`params["target_height"] = 0.30`、
  `asset_cfg.body_names = ["base"]`；传感器是 `SceneEntityCfg("height_scanner_base")`（base 正下方的射线）。
* 实现（`mdp/rewards.py::base_height_l2`）：
  `reward = (root_z − (target_height + mean_hits))² × clamp(−g_z, 0, 0.7)/0.7`，
  其中 `mean_hits` = **该环境自己**在基座下方那些**有效射线**的命中点世界 z 均值。
  ⇒ 目标高度 = **局部地面高度 + 0.30 m**，**逐环境、逐控制步**随地形更新；平地上 ≈0.30，台阶上 ≈0.30+h，
  斜坡上跟着坡面走（不是"世界平面 0.30"）。
* 三个必须知道的细节：
  1. **逐环境**判定（2026-09-24 修）：旧实现是整批判定，4096 个环境里只要有一个射线落空（例如它正悬在
     沟壑上方），所有环境都退回 `adjusted = root_z`（误差恒 0）⇒ −10 的高度罚被**整场关掉**、奖励还
     随 `num_envs` 变化。现在只有该环境自己全部射线落空才退回。
  2. **重力门**：`clamp(−g_z,0,0.7)/0.7` —— 机体越歪惩罚越小，翻倒时自动失效。
  3. `height_scanner_base` **必须留在基座正下方**（`CMoE_env_cfg.py:241` 的注释）：观测用的 `height_scanner`
     有 0.25 m 前移，若误用它会采到**前方**地面 ⇒ 目标高度系统性偏移。
* **没有**的能力：`target_height` 是**单一全局常量 0.30**，**不支持"每种地形不同的目标高度"**
  （例如"台阶上蹲低一点"）。要那种效果得给该项加逐地形目标（新功能）。

## 5. 与 PPO 的配平还差多少（要如实说）

| 口径 | PPO rough | CMoE v3 | 说明 |
|---|---|---|---|
| 步态项权重和（含 slide） | 15.05 | **15.05** | 已对齐 |
| 机身水平罚 | −5.0（全地形） | **−5.0**（易地形） | 已对齐（更宽松） |
| 速度跟踪 | `track_lin_vel_xy_exp 1.5`（机体系） | `track_world_vel_xy_exp **5.0**`（世界系） | **仍是 3.3×** |
| **shaping / task** | **≈ 7.1 : 1** | **≈ 2.7 : 1** | 只有把跟踪项从 5.0 降到 ~2.0，或把步态项整体 ×2，才追得上 PPO |
| 地形掩码 | 无 | 步态四项豁免 `boxes`/`gap`；水平罚豁免 7 类障碍 | 有意分区 |
| 每个专家拿到的梯度 | 单策略 100% | 门控权重缩放 ⇒ ~20%（实测集中后 E3/E4 各 ~30%） | MoE 固有摊薄 |

⇒ 这四条改动把"**步态质量**"补回来了（打滑、水平、滞空），但**配平**仍偏向任务项；若新 run 里步态还是
漂，下一个杠杆是 `track_world_vel_xy_exp 5.0 → 2.0~2.5`（或步态项整体 ×2）。

## 6. 验证（本轮实际跑过的）

* `python3 scripts/tools/check_reward_overrides.py`：`cmoe` 27 项 / `cmoe-gaitfree` 23 项，
  `flat_orientation_l2 = −5 ← func=MaskedFlatOrientationL2`、`feet_air_time = 1`、
  `feet_slide = −0.05`、`feet_gait` 进入"权重 0 ⇒ 移除"列表 ✅
* `python3 scripts/tools/check_terrain_columns.py`：**"全部掩码引用的地形名都有 ≥1 列 ✅"**
  （新掩码引用的 `OBSTACLE_TERRAIN_NAMES` 全部有效）
* `py_compile`（cfg + mdp）、`git diff --check` 干净；
* 全仓离线测试 **482 通过 / 8 跳过**（新增 `test_reward_terrain_partition.py` 9 项：清单互斥/覆盖全部地形/
  名字有效/掩码接线/相位核关闭且探针保留/air_time 与 slide 对齐 PPO；并更新了 7 处锁旧值的断言）
* **未验证**：没有起训练（改奖励必须新开 run，本机无 GPU）。需要训练机起一条新 run 才能看到
  `illegal_contact`／速度跟踪／逐列步态读数是否改善。

## 7. 下一步（训练机）

```bash
cd <repo>/imgo2_rl
# A：你现在这条路（先验供步态；本次改动只加了 feet_slide + 水平罚）
python3 scripts/rl_lab/cmoe/train.py --task=Imgo2-basemove-rough-cmoe-gaitfree ...
# B：完整配方（四项手工 shaping 也开）
python3 scripts/rl_lab/cmoe/train.py --task=Imgo2-basemove-rough-cmoe ...
```
验收（用 `docs/cmoe_gait_measurement_2026-09-28.md` 的链路）：
1. `flat`／`hf_pyramid_slope(_inv)`／`random_rough` 上 **FL-FR 相位 ≈ 0.5、FL-RR ≈ 0**（trot）；
2. 障碍地形不作要求（允许 bound/lockstep）；
3. `illegal_contact` 不上升、速度跟踪不退步；
4. `gait_airtime_mean` 不再往 0.28 s 漂（PPO 的步态是"快而浅"）。
