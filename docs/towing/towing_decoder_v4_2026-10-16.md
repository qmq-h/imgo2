# decoder 去掉显式质量预测（契约 v3 → v4，2026-10-16）

用户口径：「**将 decoder 尾巴的质量取消预测**，我们遵循一个准则，只用 decoder 建模和四足机器人
直接相关的，**质量用 latent 假定**」。选定方案：**只删显式质量预测，latent 完全自由**（不额外挂
质量辅助目标；质量仍留在 critic 特权观测、物理与记录里）。

## 1. 变了什么

| | 契约 v3（2026-10-10 起） | 契约 **v4**（本次） |
|---|---|---|
| decoder 显式输出 | **6** = [vx, vy, **m**, Fx, Fy, Fz] | **5** = [vx, vy, Fx, Fy, Fz] |
| actor 输入 | 58 + 6 + 16 = **80** | 58 + 5 + 16 = **79** |
| critic 输入 | 73（含 `cart_privileged_parameters` 里的质量） | **73 不变** |
| policy 帧 / latent / 动作 | 58 / 16 / 13 = 1 + 12 | **不变** |
| decoder 监督项 | 速度 + 力 + 质量（质量按 `\|F_gt\| > 1 N` 门控加权） | **速度 + 力**（两项） |
| `towing_contract` | `{version: 3, frame_dim: 58, explicit_dim: 6, latent_dim: 16}` | `{version: 4, frame_dim: 58, explicit_dim: 5, latent_dim: 16}` |

**理由（用户准则）**：机器人自身速度与「作用在机器人上的拉力」是四足自己的量；**小车质量是负载
的属性**，不该由 decoder 显式回归。质量仍然通过两条路进系统：① critic 的特权观测
（`cart_privileged_parameters`，训练用）；② **latent**——actor 拿到 `[frame, estimate(5), latent(16)]`，
质量的隐式信息由 latent 承载（本方案不给 latent 加质量监督，是否真的编码质量由 `--dump-latent`
的离线分析检查）。

## 2. 改动的文件

| 文件 | 改动 |
|---|---|
| `scripts/rl_lab/rl_lab/modules/towing_decoder.py` | 删 `MASS_DIM` / `MASS_INDEX` / `mass_head`；`OUTPUT_DIM = 2 + 3 = 5`、`FORCE_SLICE = slice(2, 5)`；`loss()` 去掉质量项（返回 2 个分量）；删 `mass_supervision_weight`；`DynamicsDecoderTrainer` 去掉 `mass_coef` 与质量权重入参，并**把 `dones` / `hidden_state` 改成关键字专用**（旧的三位置参数调用会 `TypeError`，不会把权重静默当回合终止标志） |
| `scripts/rl_lab/rl_lab/config/towing_algorithm_cfg.py` | `TowingDecoderCfg` 删 `mass_coef`，docstring 改 5 维 |
| `scripts/rl_lab/rl_lab/wrapper/towing_vec_env_wrapper.py` | decoder 观测组字面量 7 → **5**；`get_decoder_supervision()` 只返回 targets（不再有质量权重列） |
| `scripts/rl_lab/rl_lab/runners/towing_on_policy_runner.py` | `TOWING_CONTRACT_VERSION` 3 → **4**；rollout 去掉 `weight_rollout`；日志分项 `(velocity, force)` 两项 |
| `tasks/.../towing/upper_logic.py` | `DecoderSpec.terms` 删 `("cart_mass", 1)`（dim 5）；**删 `mass_range` 字段**（工作域的权威位置只剩环境 cfg 与测量台） |
| `tasks/.../towing/upper_mdp.py` | `decoder_targets` 只拼 `[vx, vy, Fx, Fy, Fz]`；删 `decoder_mass_weight` |
| `tasks/.../towing/upper_env_cfg.py` | `DecoderCfg` 删 `mass_weight` 观测项；docstring 改 v4 / explicit 5 / actor 79 |
| `scripts/towing/upper_policy_runtime.py` | `EXPLICIT_DIM` 6 → **5**；`CHECKPOINT_CONTRACT_VERSION` 3 → **4**；报错文案改成「v3（explicit 6 / actor 80）已失效，v2 57/79 与 56/63 同样不兼容」 |
| `scripts/towing/play_towing_test.py` | 期望契约 `{version: 4, frame_dim: 58, explicit_dim: 5, latent_dim: 16}`；`DECODER_ESTIMATOR_ITEMS` 去掉 `mass_kg`（真值 3 项：vx/vy/张力），力的切片 `estimate[:, 2:5]` |
| `scripts/rl_lab/towing/play.py` | decoder 精度统计去掉质量 MAE / 监督占比两项 |
| `scripts/tools/analyze_latent.py`、`compare_play_runs.py` | `ESTIMATOR_ITEMS` 去掉 `mass_kg`（旧 report.json 里的质量字段仍在，只是不再纳入本口径；工作条件的 `cart_mass_kg` 标签保留） |
| **`tasks/.../towing/agents/upper_ppo_cfg.py`** | **2026-10-17 补修**：训练用 agent cfg 里的 `TowingDecoderCfg(..., mass_coef=1.0)` 漏删（见 §3.1） |
| **`scripts/rl_lab/rl_lab/modules/__init__.py`** | **2026-10-17 补修**：仍在 `from .towing_decoder import ... mass_supervision_weight`（该符号已删） |

## 3.1 训练机实跑补修（2026-10-17）

训练机第一次跑就报：

```
[FAILED] TypeError: TowingDecoderCfg.__init__() got an unexpected keyword argument 'mass_coef'
```

两处漏改，**都在离线测试覆盖不到的地方**：

1. `tasks/.../towing/agents/upper_ppo_cfg.py` 的类体里
   `decoder = TowingDecoderCfg(..., mass_coef=1.0)`。这个类体在**模块 import 时**求值，
   而任务注册（`gym.register(..., rsl_rl_cfg_entry_point=...)`）会 import 它 ⇒ 连
   **不训练**的测试台（`play_towing_test.py`，它要 import towing 任务拿场景）都会崩。
   离线测试没抓到，是因为该模块经 `isaaclab.utils.configclass`，无 Isaac Lab 的解释器
   import 不了。
2. `scripts/rl_lab/rl_lab/modules/__init__.py` 仍导出已删除的 `mass_supervision_weight`
   ⇒ `import rl_lab.modules` 会 ImportError（训练 runner 必经）。测试都是按**文件路径**加载
   `towing_decoder.py`，从不 import 这个包 `__init__`，所以也没抓到。

**修法**：删掉这两处（`mass_coef=` 与那条 export）。

**新增守卫** `tests/test_towing_rl_lab_config_wiring.py`（纯标准库 AST，不需要 isaaclab/torch）：

- 把 `rl_lab/config/*.py` 里每个 `*Cfg` 声明字段（含同包内基类继承）当白名单，扫
  **所有任务目录 + `scripts/rl_lab`** 里对它们的构造调用，任何未声明的 kwarg 即失败；
- 扫已删除符号（`mass_coef` / `mass_supervision_weight` / `decoder_mass_weight` /
  `weight_rollout`）的**代码级**引用（关键字参数、属性访问、import；文档字符串里的说明不算）；
- `rl_lab/modules/__init__.py` 从 `towing_decoder` 导入的名字必须在 `towing_decoder.py`
  顶层真实存在。

**反证**：把 `mass_coef=1.0` 临时写回 agent cfg，该守卫立刻 2 项失败；恢复后全绿。
（顺带：守卫的 AST 读取用 `utf-8-sig`，因为仓里个别既有脚本带 UTF-8 BOM，`ast.parse(str)`
不接受。）

## 3. 验证

- `pytest imgo2_rl/tests -q` ⇒ **627 passed, 4 skipped, 30 subtests passed**（2026-10-17 补修后；
  619 是补修前，含本轮改写的
  契约守卫：`test_decoder_targets_are_physical_units`、`test_decoder_head_widths_are_two_velocity_three_force`、
  `test_decoder_trainer_update_without_mass_weight`、`test_decoder_loss_reports_two_weighted_components`、
  `test_rl_lab_dimension_contract_matches_upper_logic`、runtime 的 v4 接受/旧契约拒收清单等）。
- 关键守卫逐条：
  - decoder 零帧输出 `(1, 5)`、`output_dim == 5`、**没有** `mass_head` 与 `mass_supervision_weight`；
  - `actor_dim == 79`（`UpperObservationSpec` 派生）且 `ActorCriticRecurrent.memory_a.rnn.input_size == 79`；
  - wrapper 的 `num_decoder_obs == 5`、`get_decoder_supervision()` 段落里不出现 `mass` 字样；
  - 旧契约一律在**加载前**被拒：v3（58/6/16）、v2（57/6/16）、`{version:4, explicit:6}` 半迁移、
    56/63 维、`latent 8`、字符串/None；
  - `strict=True` 仍能抓住伪造契约下的**旧 80 维 actor** 权重；
  - 工作域守卫改成「环境 cfg 的 `mass_range` ↔ 测量台 `TRAINING_MASS_RANGE_KG`」两处一致，
    并断言 `DecoderSpec` **不再有** `mass_range`。
- `compileall`、`git diff --check`、`git ls-files -i -c --exclude-standard` 通过。

## 4. 待修复／待确认（TOW-DEC-01）

状态：**已修（离线契约通过），待训练机验证**。缺什么才能完成：

1. **短训练**：确认 5 维 head 的梯度、`decoder velocity/force` 两项 loss 的量级与 KL 不塌缩；
   观察日志里的 `Mean decoder <name> loss` 现在只有 velocity / force 两行。
   另外确认**场景构建/任务 import 这一关**已过（2026-10-17 的两个漏改就是在这里暴露的）；
   训练机若再报别的 `TypeError: ... unexpected keyword argument`，先查该 cfg 的字段表。
2. **latent 是否真的携带质量**：跑一轮带 `--dump-latent` 的测试台，用
   `analyze_latent.py`/`analyze_latent_clusters.py` 检查「质量档在 latent 上的 η²/LOO」；
   若质量完全不可读，再决定是否给 latent 加辅助监督（本方案**故意不加**，保持 latent 自由）。
3. **旧 checkpoint 不可用**：v3 的 `policy.pt`（actor 80）与所有更早的导出件都必须重训后再导出；
   `upper_policy_runtime` 会在加载前抛 `TowingCheckpointContractError`，不会走到 shape mismatch。
4. 与地形改动（[混合地形记录](towing_mixed_terrain_2026-10-16.md)）叠加：本轮两项都应新建 run，
   不要 `--resume` 任何旧 run。
