# v7：恢复 1 维 vx 偏移头（分支 `imgo2_dvx`，2026-11-12）

> 用户要求新增分支 `imgo2_dvx`，把上层 12 维动作改回 13 维（1 维 vx 偏移 + 12 维关节残差）。
> 本分支基于 `main` 的 50 Hz 配置；**未合并 main，未训练**。

## 1. 新契约

| 项 | v6（main） | v7（本分支） |
|---|---|---|
| 动作 | 12 维纯关节残差 | **13 = 1 vx + 12 关节** |
| policy 帧 | 57 | **58** |
| actor | 78 | **79** |
| critic | 72 | **73** |
| decoder frame | 57 | **58** |
| contract | v6 | **v7** |
| 控制频率 | 50 Hz | **50 Hz 不变** |

重建掩码仍为 **42 维**（58 − 13 − 3）。

## 2. 偏移语义（与旧 v5 一致）

```text
u          = cat(u_cmd(1), u_joint(12))
offset     = clamp(clamp(u_cmd, ±1) × 0.5, -0.2, +0.6)
active     = elapsed_s >= tow_start_s
loco_vx    = clamp(task_vx + active·offset, -1.0, +1.0)
delta      = clamp(u_joint, ±1) ⊙ action_scale
held       = frozen_joint_targets + delta
```

- 偏移加在**冻结策略推理之前**（进入底层观测）；
- 关节残差加在**冻结策略输出之后**；
- STOP 后偏移仍生效（可在包络内正/负）；
- 出生段不叠加偏移。

## 3. 改动文件

- `upper_logic.py`：恢复 `CMD_ACTION_DIM=1`、`split_action`、`command_offset`、
  `bounded_command_offset`、`loco_command_vx`；帧 58 / actor 79。
- `upper_mdp.py`：动作 13、偏移门控/两层限幅、`last_action` 13、`policy_frame` 58；
  `HierarchicalVelocityActionCfg` 恢复 `cmd_offset_scale/offset_min/offset_max`。
- `upper_env_cfg.py`：v7 文档；动作 cfg 默认值恢复。
- `upper_policy_runtime.py`：`FRAME_TERMS last_action 13`、`ACTION_DIM 13`、
  `CMD_ACTION_DIM 1`、`DEFAULT_CRITIC_OBS_DIM 73`、contract 7、导出 58→13。
- `towing_decoder.py` / `towing_on_policy_runner.py` / `towing_vec_env_wrapper.py` /
  `towing_algorithm_cfg.py` / `upper_ppo_cfg.py`：frame 58、contract 7、动作 13。
- `play_towing_test.py`：v7 常量、`compose_loco_vx_offset`、偏移日志/plan/契约校验。
- `play.py`：导出/日志 58→13。
- 测试：runtime/play/upper contract/CE Net 全部更新为 v7。

## 4. 验证

- 全量 `pytest imgo2_rl/tests -q`：**745 passed, 4 skipped, 1336 subtests passed**。
- 只做离线/源码契约验证；本机无 GPU，**未启动训练或仿真**。

## 5. 重要限制

- **所有 v6 及更早 checkpoint 失效**：`model_1500`、`model_3000`、`model_1300` 都不能加载。
- 必须从零训练一个 v7/58/13 的新上层；不能用 `--resume` 或改元数据绕过。
- 旧的 v5 13 维语义虽维度相同，但 contract 版本不同，本分支也拒绝。
- 下一步：在分支上跑 50 Hz/70 列/1400 env/5–30 kg 的 A/B，重点看 15°/25 kg FALL、
  速度、LANE/IMPACT/JNT 与偏移头分布；只有 B 明显优于 v6 才合并 main。
