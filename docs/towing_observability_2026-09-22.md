# 拖曳训练的观测指标设计与一个日志 bug（2026-09-22）

## 结论

冒烟那次 TensorBoard 只有 8 个标量、**没有任何 reward 分项与终止原因**，不是「没配日志」，
而是仓库自有 runner 读错了 Isaac Lab 的键。已修，并补了三个日志专用量。

## Bug：episode 统计写在 `extras["log"]`，runner 只读 `extras["episode"]`

Isaac Lab 的 `ManagerBasedRLEnv._reset_idx()` 把各 manager 的回合统计统一写进
`self.extras["log"]`（`source/isaaclab/isaaclab/envs/manager_based_rl_env.py:369`），
其中包含：

- `Episode_Reward/<term>`：`RewardManager.reset()` 对**每个** reward term 给出
  `回合和 / max_episode_length_s`（`reward_manager.py:115-121`）；
- `Episode_Termination/<term>`：`TerminationManager.reset()` 给出各终止项的触发比例
  （`termination_manager.py:140-144`）。

而仓库自有 `TowingOnPolicyRunner` 只读 `infos["episode"]`，该键在 Isaac Lab 里不存在，
于是这些量一个都没进 TensorBoard。**对照**：官方 rsl_rl runner 是**两个键都认**的
（`rsl_rl/runners/on_policy_runner.py:226-229`：先 `"episode"`，再 `elif "log"`）。

**修**：`towing_on_policy_runner.py` 改为同样的 `if "episode" ... elif "log" ...`。

**附带发现**：`RewardManager.compute()` 对 `weight == 0.0` 的项直接 `continue`
（`reward_manager.py:144-147`），既不调用函数也不更新 `_episode_sums` ⇒ **零权重 reward term
不会产生任何日志**。所以诊断项不能用 weight=0，本仓改用 `1.0e-6`。

## 修好后会自动出现的指标

以 `Episode/` 为前缀（runner 的 `_log` 写 `Episode/{key}`，名字有点重复但可用）：

**奖励分项**（10 项 = 原有 7 个 reward term + 新增 3 个诊断项）：

| 指标 | 读它来回答什么 |
|---|---|
| `Episode/Episode_Reward/tracking_velocity` | 上层速度跟踪是否在提高（唯一的正奖励项，权重 1.0）|
| `Episode/Episode_Reward/collision` | 撞车惩罚的发展（权重 −50，实际每步 −2.5）|
| `Episode/Episode_Reward/fall` | 跌倒频率（权重 −50）|
| `Episode/Episode_Reward/clearance` | 是否在靠近小车（softplus 障碍）|
| `Episode/Episode_Reward/stop_towing_force` | STOP 后绳力是否卸掉 |
| `Episode/Episode_Reward/extra_distance` | STOP 后是否继续前移 |
| `Episode/Episode_Reward/action_rate` | 动作抖动 |
| `Episode/Episode_Reward/obs_cart_present` | **无小车比例，期望≈0.875**（权重 1e-6，新增）|
| `Episode/Episode_Reward/obs_towing_force` | 绳力大小逐环境均值（新增）|
| `Episode/Episode_Reward/obs_towing_force_active` | 仅绳力>1 N 时的均值（新增）|

**终止原因**（3 项）：`Episode/Episode_Termination/{time_out,robot_fall,cart_collision}`。

## 为什么补 `obs_towing_force_active`

单看 `obs_towing_force` 的均值无法区分两种完全不同的故障：

- 全程没有拉力（绳一直是松的）→ 均值≈0；
- 有拉力但只有少数步有效（绳频繁绷紧又松开）→ 均值也被稀释到接近 0。

`obs_towing_force_active` 只统计绳力>1 N 的步，两者配合才能判断到底属于哪种。

## 运行时的读表顺序（建议）

1. `mean_episode_length`：向 200（= `episode_length_s / step_dt`）靠近说明回合能跑满、没被提前终止。
2. `Episode_Termination/{cart_collision,robot_fall}`：应趋近 0；若居高不下，先查 reward 量级而不是超参。
3. `Episode_Reward/tracking_velocity`：唯一的正奖励项，是「有没有在学」的主信号。
4. `obs_cart_present`：应≈0.875。显著偏离说明无小车采样或场景隔离有问题。
5. `obs_towing_force` + `obs_towing_force_active`：判断绳子是否真的在传力。
6. STOP 后三项（`stop_towing_force`、`clearance`、`extra_distance`）：用来抓「为卸载拉力而让小车
   逼近」的捷径。

## Bug 2：每个 Episode 指标在同一 step 被重复写 48 遍

修好键名之后，用户报告控制台「反复重复」。用 tfevents 定量核对，确认不是排版问题：

| run | Episode tag | 每 tag 写入次数 | 同一 step 重复条数 |
|---|---|---|---|
| `21-33-23` | 无 | — | 键名 bug，全部丢失 |
| `21-40-09` | 无 | — | 同上 |
| `21-43-27` | 13 | 48 | 611 |
| `21-49-15` | 13 | **960** | **12220** |

`960 = 20 轮 × 48`，其中 **48 正好等于 `num_steps_per_env`**。

**根因**：`episode_infos` 是在 rollout 的**每一步**只要有环境 reset 就追加一份
`infos["log"]`，一个 48 步的 rollout 会累积几十份同键字典。而旧实现对这个列表**逐份遍历**
写 TensorBoard、又**逐份遍历**拼控制台字符串 ⇒ 每个指标在每个 iteration 被写/打印约 48 次。

**修**：新增 `_aggregate_episode_infos()`，按键聚合成一份（标量取均值），写 TB 与打印都用
聚合结果；同时把 `_log` 拆成 `_aggregate_episode_infos`（聚合）、`_write_scalars`（TB）、
`_format_progress`（控制台）三个函数，`_log` 只做调度。`episode_infos` 现在全程只遍历一次
（原先两处各遍历一遍）。

**验证**：

- 端到端测试 `test_aggregation_writes_each_episode_scalar_once`：用真实 `SummaryWriter` +
  48 份同键字典，断言 `EventAccumulator` 读回每个 `Episode/*` **恰好 1 条**（不是 48 条）；
  已用负向测试确认该断言在旧实现下会失败（`KeyError`／条数 48）；
- `test_episode_infos_are_aggregated_before_logging`：静态断言聚合函数存在且为 `staticmethod`，
  负向测试确认改名或改签名即失败。

**遗留**：本次修复后尚未实跑；四次历史 run 全部跑在修复之前，因此没有「修复后」的 tfevents
可对照。`21-49-15` 那次的 12220 条重复写入与 `obs_cart_present` 全为 0，都是**旧代码产物**，
不代表环境行为。

## 控制台输出（按 `rl_lab` PPO runner 的样式对齐）

`Perf/collection_time` 与 `Perf/learning_time` 一直都在记录，但此前**只进 TensorBoard**，
不接 TensorBoard 时看不到进度与总时长。现已把 `_log` 的输出改为与
`rl_lab/runners/ppo_on_policy_runner.py::log()` 同一套内容与排版：

```
################################################################################
                           Learning iteration 5/40000

              Computation: 8262 steps/s (collection: 23.570s, learning 0.450s)
      Value function loss: 13.2894
           Surrogate loss: 0.7523
        Mean decoder loss: 0.1622
    Mean action noise std: 0.51
              Mean reward: -22.80
      Mean episode length: 200.00
Episode_Reward/tracking_velocity: 0.1200
Episode_Reward/obs_cart_present: 0.8750
Episode_Termination/time_out: 0.9000
--------------------------------------------------------------------------------
          Total timesteps: 393216
           Iteration time: 24.02s
            Avg iteration: 23.95s
               Total time: 47.59s (0.01h)
                      ETA: 958000.0s (266.11h)
```

要点：

- **每轮耗时**：`Iteration time`（采集+学习）、`Avg iteration`（本次运行均值）；
- **`Perf/total_fps`**：总吞吐 steps/s，跨环境数比较规模效率要看这个（例如 4096 env 与
  4 env 的每轮耗时差 4.3 倍，但吞吐差得多）；
- **ETA 的分母是 `iteration - start_iter + 1`**，只统计本次运行时间，resume 时不会被历史时间污染；
- **`start_iter` 必须作为 `_log` 形参传入**——它是 `learn()` 的局部变量。2026-09-22 实跑崩在
  `NameError: name 'start_iter' is not defined`，该错误只在训练跑到 `_log` 时才暴露；
  已加静态回归测试 `test_towing_runner_log_gets_start_iter`（断言形参存在且调用处实参个数匹配）。

## Bug 3：碰撞传感器 filter 用通配，导致碰撞判据静默失效

训练时反复出现：

```
[Error] [omni.physx.tensors.plugin] Filter pattern '/World/envs/env_*/Robot/*'
        did not match the correct number of entries (expected 256, found 4864)
```

**这不能当噪声忽略**：`cart_collision` 完全依赖这些传感器的 `force_matrix_w`。

**根因**：Isaac Lab 的 `ContactSensorCfg` 文档写明

> The reporting of filtered contacts only works when the sensor primitive `prim_path`
> corresponds to a **single primitive** in that environment.

而 5 个车体传感器用的都是 `filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/.*"]`，单个 filter 项
每环境展开成多个 prim。数字对得上：`4864 = 256 × 19`（机器人 17 个 link，`/Robot/*` 还匹配到
根 prim 等，共 19）。Isaac Lab 自带测试里 filter 用的全是 `{ENV_REGEX_NS}/Cube_2` 这类**单个 prim**。

**危害**：过滤后的接触上报不工作 ⇒ `force_matrix_w` 拿不到真实的「车斗／车轮 ↔ 机器人」接触力
⇒ `cart_collision` 恒假 ⇒ 碰撞 reward（权重 −50）与碰撞终止全部失效 ⇒ 策略可以随意撞车而
不受罚，TOW-03 里「排除为卸载拉力而让小车逼近的捷径」那条验收也失去意义。

**修**：新增 `_robot_link_names_from_urdf()` 从训练 URDF 读出 17 个 link 名，再由
`_robot_body_filters()` 逐个构造 filter 项（每项恰好解析出 1 个 prim），5 个传感器共用
`_ROBOT_BODY_FILTERS`。注意 `{ENV_REGEX_NS}` 展开成 `/World/envs/env_.*` 后被 Isaac Lab 换成
`env_*`，**环境命名空间里的通配是预期行为**，违规的只是 `Robot/` 后面那一段。

**顺带修掉一个既有路径 bug**：`upper_env_cfg.py` 里 `_USD_CACHE` 原写
`Path(__file__).resolve().parents[6]`，但本文件在 `.../tasks/manager_based/towing/` 下，
**仓库根是 `parents[7]`**（`assets/imgo2.py` 在浅 2 层的 `assets/` 下，那里才是 `parents[5]`）。
原写法让 USD 转换缓存落到 `<repo>/imgo2_rl/logs/usd/upper_towing`。已引入 `_REPO_ROOT` 统一修正。

**验证**：新增两条静态测试——
`test_collision_filters_resolve_one_prim_per_environment`（AST 剥掉字符串后断言可执行代码里
没有 `Robot/.*`／`Robot/*`；断言 5 个传感器都用显式列表；断言 `_REPO_ROOT` 为 `parents[7]`）与
`test_collision_filter_matches_urdf_link_count`（filter 项数须等于 URDF link 数）。
负向测试确认：把 filter 改回 `Robot/.*` 会让前者失败。同时**修正了一条断言旧设计的既有测试**
（`test_safety_and_history_producers_are_live` 原本断言 `Robot/.*` 出现 5 次，等于把错误锁死）。

**复跑结果（2026-09-22，256 环境 × 20 轮）**：`Episode_Reward/collision` 由**恒 0** 变为
**−0.168**，`Episode_Termination/cart_collision` 由**恒 0** 变为 **0.448**，
`Episode_Termination/time_out` 由 1.000 降到 0.552。这证明过滤后的接触上报确实恢复了工作。

**但 `Filter pattern` 报错没有消失，只是数字变了**：`expected 256, found 4864` →
**`expected 1, found 19`**。因此该报错**不是**来自上面修掉的 5 个车体 filter
（它们现在逐项列出、每项恰好 1 个 prim）。定位需要运行时数据，见
`imgo2_rl/scripts/tools/dump_towing_contacts.py`（打印场景内全部 ContactSensor 的
`prim_path`、展开后的 filter 列表、`body_names`、`filter_count`、`force_matrix_w.shape`）。

数字变化的另一种解释待验证：`22-02-55` 的 Episode 统计确认是**新代码**（重复写入 0，
与 `21-56-35` 一样），但两次的 `expected` 相差 256 倍，需用上述脚本确认是「每环境报一次」
还是「总计报一次」。**在拿到该输出前，不把这处报错判为无害。**

**未验证**：本机无 Isaac Sim，未实测 `force_matrix_w` 是否恢复真实数值、`found` 数是否变为 256。

## 性能：启动 135 s，每轮约 4.6 s 固定开销与显卡无关

用户报告「启动很慢」。用历次运行的 tfevents 定量拆开，是**两个不同的问题**：

### 启动：256 环境约 135 s

| envs | 配置落盘 → 第一条曲线 |
|---|---|
| 4 | 6 s |
| 256 | 134–141 s |
| 4096 | 143 s |

（`22-08-03` 那次只显示 8 s 是因为它只跑了 1 轮，量到的是曲线写入间隔，不是启动时间。）

### 每轮耗时：拟合出约 4.6 s 的固定开销

| run | envs | collect/轮 | learn/轮 | steps/s |
|---|---|---|---|---|
| `21-43-27` | 4 | 5.5 s | — | 约 35 |
| `21-49-15` | 256 | 6.51 s | 0.258 | 1816 |
| `22-02-55` | 256 | 6.74 s | 0.280 | 1749 |
| `21-40-09` | 4096 | 23.84 s | 0.45 | 约 8240 |

拟合 `每轮耗时 = 固定成本 + envs × 单环境成本`：

- 4 → 256 env：env 涨 64 倍、每轮只涨 1.18 倍 ⇒ 固定 ≈ **4.7 s/轮**，单环境 ≈ 8.1 µs
- 256 → 4096 env：env 涨 16 倍、每轮涨 3.5 倍 ⇒ 固定 ≈ **4.6 s/轮**，单环境 ≈ 4.7 µs

**两段独立拟合得到同一个固定成本（4.6–4.7 s/轮）**，说明它不随规模变化，与显卡算力无关。
吞吐量：256 env ≈ 1800 steps/s，4096 env ≈ 8240 steps/s（env ×16 只换来吞吐 ×4.7）。

### 已定位并修掉的主要来源：每物理步回读 PhysX

`ActionManager.apply_action()` 在**每个物理步**调用（`manager_based_rl_env.py:186`），
所以 `_apply_towing_physics` 每 5 ms 跑一次：4096 环境下每轮 `480 步 × 4096 env`。
而它原先每步都做：

```python
masses   = self._cart.root_physx_view.get_masses().to(self.device)     # 每步 GPU→CPU→GPU
inertias = self._cart.root_physx_view.get_inertias().to(self.device)   # 同上
robot=self._body_properties(...)   # 内部再 get_inertias()
```

每轮 `get_masses`／`get_inertias` 各被调用约 **196 万次**，每次都是一次设备往返（同步点）。

**修**（`upper_mdp.py`）：

- 新增 `_refresh_mass_cache()`：一次性读出机器人质量／对角惯量、小车质量／对角惯量、
  四轮惯量，缓存为张量；
- 新增 `_invalidate_mass_cache()`：`reset_towing_episode` 在 `set_masses`／`set_inertias`
  之后调用它。**这一步不可省**：质量每回合按 `scale` 随机化，缓存若不知道就会被复用成错值；
- `_apply_towing_physics` 改为使用缓存；`_body_properties` 增加 `inertia_diag`／`mass_total`
  参数（不传时才回退到回读）。

**注意哪些量不能缓存**：`inverse_inertia_world` 依赖**当前姿态**（`world_inverse_inertia`
用 `quat → 旋转矩阵` 旋转机体对角惯量），必须每步计算。缓存的只是**常量部分**。

**验证**：新增 `test_per_step_physics_does_not_readback_physx`（AST 断言
`_apply_towing_physics` 内不出现 `get_masses`／`get_inertias`；`_refresh_mass_cache` 负责回读；
reset 里失效调用必须位于 `set_masses` 之后）。两组负向测试确认会失败：把回读塞回热路径、
删掉 reset 里的失效调用。拖曳离线测试 216 通过、契约测试 25 通过。

**实测结论（用户反馈 2026-09-22）**：该改动**没有带来可观测的提升**。因此上面这条
「每轮 4.6 s 固定开销」的来源判断**未能被证实**——回读确实是真实存在的冗余（已消除，且比原来
更正确），但它**不是**每轮耗时的主因。

**对「慢」的重新认识**：用户的「慢」实际是**误判**——`print` 原先藏在
`if self.writer is not None` 里，条件不成立时一行进度都不输出（见下节），看起来像卡死。
真实数据是 256 环境约 6.5 s/轮，本身并不反常。**教训**：不要用拟合出的固定开销直接断定瓶颈，
先确认用户观察到的症状与量测口径一致。

剩余热路径成本（未再优化）：冻结低层策略推理（每轮约 49 万次）与每步的姿态四元数运算。
若日后确实需要提速，应先用 `torch.profiler` 或分段计时定位，而不是继续猜。

## 输出被条件吞掉：`print` 与 TensorBoard 写入共用一个判断

`learn()` 里原本是：

```python
if self.writer is not None:
    self._log(...)   # 里面既写 writer 也 print
```

`self.writer` 只在 `log_dir is not None` 时创建。于是**该条件不成立时控制台一行进度都不打**，
用户据此判断「运行很慢 / 卡住」。

**修**：控制台输出与 TensorBoard 写入解耦——`_log` 无条件调用，内部只在 writer 可用时
调 `_write_scalars`，而 `print` 一定执行并加 `flush=True`（重定向到文件时按块缓冲也会掩盖输出）。
另在 `learn()` 开头加一条启动行，说明环境数、总轮数、日志目录，并提示「第一条进度需等首次
rollout 完成」——256 环境从进程启动到第一条进度约 2.5 分钟，中间静默容易再次被误读为卡死。

**验证**：用桩对象实测 writer 为 `None` 时仍有完整进度输出（修复前该情形零输出），
writer 存在时 TensorBoard 写入次数不变。

## 限制与未验证

- 上述「修好后会出现哪些指标」的依据是**主干源码**（Isaac Lab 0.45.9 的 manager 实现 +
  官方 rsl_rl runner 的对照），**尚未在 Isaac Sim 里实跑确认**。冒烟那次一个 `Episode_*` 都没有，
  与「键名读错」的结论一致；但修复后未再跑。
- 诊断项用的 `1.0e-6` 权重虽可忽略，仍是**非零**改动，会微量进入总回报（每步 1e-6）。
- 未新增「参考指令速度」与「小车是否被拖动」两项。前者可从 actor 帧趋势间接看，后者可由
  `obs_towing_force` 推断；若后续仍不足，再考虑从 action term 里直接打点。
