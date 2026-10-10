# 拉力变化率惩罚 `towing_force_rate`（TOW-25，2026-10-10）

用户 2026-10-10 要求：

> 加一个拉力变化率的惩罚。但是考虑绳子瞬间卸载，就给一个掩码，在绳子卸载的时候拉力允许突变。

落地在
[`upper_mdp.py`](../imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/upper_mdp.py)
与
[`upper_env_cfg.py`](../imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/upper_env_cfg.py)，
测试在 [`test_towing_force_rate.py`](../imgo2_rl/tests/test_towing_force_rate.py)。

**本机没有 Isaac Lab**（`upper_mdp` 顶层 `import isaaclab`，而本机 `omni.kit` 缺失 ⇒
离线 import 不了）：以下全部结论只由**离线 `pytest` + AST + 纯 torch 逻辑复算**得出，
凡只有训练才能确认的一律标「**待训练机验证**」（见 §7）。README 问题表条目 = **TOW-25**。

---

## 1. 物理理由：为什么这个奖励必须"不对称"

这是本项存在的**全部依据**，与仓库里对两类连接的分工一致
（见 [`towing_rigid_link_2026-10-08.md`](towing_rigid_link_2026-10-08.md)、
`mdp/rope_model.py` 的模型定义）：

| | 绳（`CompliantRope` / `InextensibleRope`） | 刚性杆（`RigidLink`） |
|---|---|---|
| 约束 | **单边**：只能拉，不能推 | **双边**：拉/推都传递 |
| 张力符号 | 恒 `≥ 0`（松弛即 `T = 0`） | **有符号**（负值 = 推力，源码注释明写） |
| 停车机制 | **绳松弛 + 车斗自由滑行 + 机器人往前走两步避让** | 靠杆主动管理力（可推可拉） |
| 卸载 | **瞬间卸载（松弛）是正常且期望的** ⇒ 必须允许突变 | 目标是**缓慢卸载、避免冲击力过大** |

由此得到本项的方向语义（不是"对称地罚变化率"）：

1. **绷紧那一侧要罚**：`rate > 0` 且超过 `rate_limit_n_per_s` 的"快速加载"= 被猛拽 /
   绳被绷直的冲击，是真实的力尖峰来源；
2. **卸载到松弛那一下必须放行**：绳一松，力只能是 0——**不存在"用绳缓慢刹车"**，
   那个突变既不是策略能"缓"下来的，也正是期望行为；罚它等于惩罚正确答案；
3. **但"仍是负载态却断崖式掉力"要罚**：`mag_t` 还有明显值（`≥ slack_eps_n`）却在一拍内
   掉了大截，那是杆行/负载中途被抽掉力，对应真实冲击 ⇒ `down` 那一支负责它。

> 注意 ② 与 ③ 是同一个 `slack` 掩码的两面：掩码**只**按"当前力是否已掉到 ≈0"开合，
> 所以它天然只放行"卸载**到底**"的那一下，不会把"卸载到一半"也放掉。

---

## 2. 公式

```python
mag     = ‖towing_force_b‖                                  # N，与 obs_towing_force 同量纲
rate    = (mag_t − mag_{t−1}) / dt                          # N/s，dt = env.step_dt = 0.05 s
up      = relu(rate − rate_limit_n_per_s) / rate_limit_n_per_s
slack   = mag_t < slack_eps_n                               # 卸载到 ≈0：绳松弛 / 杆卸载
down    = relu(−rate − release_limit_n_per_s) / release_limit_n_per_s * (1 − slack)
penalty = clamp(up + down, 0, clip_max)                     # 无量纲
```

`penalty` 进 `RewardManager` 后每步代价 = `weight × penalty × step_dt`。

实现（`upper_mdp.towing_force_rate_penalty`，参数全部走形参、不写死）：

```python
term = _term(env)
mag = torch.linalg.vector_norm(term.towing_force_b, dim=1)
rate = (mag - term.prev_force_mag) / env.step_dt
# 刚 reset 的第一拍没有「上一拍」样本 ⇒ rate 精确为 0（不是拿 0 当上一拍去算）。
rate = torch.where(term.force_rate_has_prev, rate, torch.zeros_like(rate))
up = torch.relu(rate - rate_limit_n_per_s) / rate_limit_n_per_s
slack = mag < slack_eps_n
down = (torch.relu(-rate - release_limit_n_per_s) / release_limit_n_per_s
        * (~slack).to(rate.dtype))
return torch.clamp(up + down, 0.0, clip_max)
```

**为什么 `down` 要乘 `(1 − slack)` 而不是 `slack` 的补集用在 `up` 上**：`up` 是"加载侧"，
加载时 `mag_t` 必然已经离开 0（正在绷紧），掩码对它没有意义；掩码的作用域**只有卸载侧**。

### 2.1 与源码口径的核对（用户要求：发现冲突以源码为准）

| 项 | 结论 |
|---|---|
| `towing_force_b` 的符号 | 是**作用在机器人挂点上的力**（`_apply_towing_physics` 里 `force_on_robot` → `quat_apply_inverse` 到 base 系），`mag` 取模长 ⇒ 永远是 `≥ 0` |
| rigid 的负张力 | `RigidLink.rope_tension` **有符号**（`mdp/rope_model.py`：负值 = 推力），`force_on_robot = tension × direction` 整体反向；`mag` **无法区分拉/推**，但**过零必然经过 `mag ≈ 0 < slack_eps_n`** ⇒ 同一个掩码覆盖"杆过零"那一下（测试 `test_rigid_zero_crossing_is_allowed_in_both_directions` 两个方向都验了） |
| `slack_eps_n = 1.0` | 与既有 `towing_force_norm_active` 的判据 `towing_force_norm(env) > 1.0` **同一个 1 N 口径**（源码核对一致，没有另立阈值） |
| 无小车 env | **不额外乘 `cart_present`**：`_apply_towing_physics` 已把取绳力整体乘 `present` ⇒ 那些 env 的 `mag ≡ 0`、`rate ≡ 0` ⇒ 本项恒 0，与显式掩码等价（`reset()` 也清零） |

**没有发现与用户给定公式/默认值冲突之处**，只补了上面两处"以源码为准"的口径说明。

---

## 3. 参数与标定

| 参数 | 默认值 | 含义 / 依据 |
|---|---|---|
| `rate_limit_n_per_s` | **200.0** | 加载侧免罚上限（N/s）。**这是唯一需要调的旋钮** |
| `release_limit_n_per_s` | **600.0** | 卸载侧免罚上限（更宽松）；绳松弛本身允许突变，这一支只兜"负载态断崖" |
| `slack_eps_n` | **1.0** | 与 `towing_force_norm_active` 的 1 N 阈值同口径；严格小于（`mag_t == 1.0` 不算松弛） |
| `clip_max` | **3.0** | 无量纲上限，兜住极端峰值（起步绷直可达百牛级） |
| `weight`（`RewTerm`） | **−0.5** | 初始保守、容易调（见下） |

**标定链条**：

```
up = 1.0  当且仅当  rate = 2 × rate_limit_n_per_s = 400 N/s
每步代价 = weight × up × step_dt = −0.5 × 1.0 × 0.05 = −0.025
tracking_velocity 满额 = +1.0 × 0.05 = +0.05  ⇒  本项是一次猛拽 ≈ 半秒的跟速收益
```

即：**"一次明显猛拽"约等于半秒的跟速收益**——有动机去缓加载，但不会压倒跟踪项。
调参只需要改 `rate_limit_n_per_s`（平移"多快算猛拽"的判据）；只有极端工况才需要动
`clip_max`，`release_limit_n_per_s` 只有"杆行负载态卸载"这一路会用到。

**默认行为速查**（离线实跑，见 §5）：

| 情形 | 判据 | 结果 |
|---|---|---|
| 快速绷紧（0 → 20 N，rate = 400 N/s） | `up` | **1.0**（每步 −0.025） |
| 卸载到松弛（800 → 0.5 N） | `slack` 掩码 | **精确 0** |
| 杆过零（+400 → −0.2 N） | `slack` 掩码 | **精确 0** |
| 负载态断崖（100 → 60 N，rate = −800 N/s） | `down` | **1/3** |
| 限内（100 N/s 加载 / −400 N/s 卸载） | 两侧 | **精确 0** |
| 极端尖峰（100 → 800 N） | `clip_max` | **3.0**（封顶） |
| reset 后第一拍 | `force_rate_has_prev = False` | **精确 0** |

### 3.1 控制步级管线复现（离线，非仿真）

用与测试同源的"从源码抽函数体"手法搭了一个**控制步级**的管线复现（不是仓库脚本、不是仿真）：
每个控制步先按 `process_actions` 的顺序更新缓存（`has_prev ← tick_seen` → `prev ← mag` →
`tick_seen ← True`），再喂 10 个 5 ms 子步（含 `force_rate_max` 的 `max` 累积），最后调
奖励函数与诊断函数。人工给的 `‖F‖` 序列与读数如下（`weight = −0.5`）：

| 控制步 | `mag_t` (N) | 20 Hz `rate` (N/s) | `penalty` | 每步 `w·p·dt` | 200 Hz `max` (N/s) |
|---|---|---|---|---|---|
| t1 出生首拍 | 0.00 | **0（无上一拍）** | **0.0000** | 0 | 0 |
| t2 settle | 0.00 | 0.0 | 0.0000 | 0 | 0 |
| t3 起步绷直 0 → 20 N | 20.00 | 400.0 | **1.0000** | **−0.02500** | 600 |
| t4 稳态 20 N | 20.00 | 0.0 | 0.0000 | 0 | 0 |
| t5 绳松弛卸载 20 → 0.5 N | 0.50 | −390.0 | **0.0000**（`slack` 放行） | 0 | 1000 |
| t6 松弛保持 0.5 N | 0.50 | 0.0 | 0.0000 | 0 | 0 |
| t7 重新绷紧 0.5 → 20 N | 20.00 | 390.0 | 0.9500 | −0.02375 | 600 |
| t8 负载态小卸载 20 → 5 N | 5.00 | −300.0 | 0.0000（在 600 的免罚限内） | 0 | 400 |
| t9 极端尖峰 5 → 800 N | 800.00 | 15900.0 | **3.0000**（`clip_max`） | −0.075 | **30000** |

三处值得记的读数：

- **t1 = 0**：首拍门生效，出生那一拍精确不罚（管线语义与源码顺序一致）；
- **t5 = 0**：卸载到 0.5 N 的 rate 是 −390 N/s（20 Hz 只看首尾），`slack` 掩码把它放行；
- **t9**：20 Hz 看到 15900 N/s，**200 Hz 看到 30000 N/s** ⇒ 同一段过程被 20 Hz 打折约一半
  （本例里 `clip_max` 已封顶，所以惩罚值不受影响；但一般情形下这就是 §5 的盲区）。
- **t8 = 0** 是**预期**：20 → 5 N 只有 −300 N/s，在 `release_limit_n_per_s = 600` 的免罚限内；
  只有更陡的负载态卸载（如测试里的 100 → 60 N = −800 N/s）才会被 `down` 罚。

---

## 4. `mag_{t−1}` 的存放与"首拍精确为 0"

- 缓存在动作项上：`HierarchicalVelocityAction.prev_force_mag`（`mag_{t−1}`，
  单位 N）与 `force_rate_has_prev`（本拍是否有真实的"上一拍"样本）。
- **只在 `process_actions` 更新**（每个控制步一次，20 Hz，与 `_previous/_processed` 同一处）：

  ```python
  self.force_rate_has_prev.copy_(self._force_rate_tick_seen)
  self.prev_force_mag.copy_(torch.linalg.vector_norm(self.towing_force_b, dim=1))
  self._force_rate_tick_seen.fill_(True)
  ```

  **不在 200 Hz 物理子步里更新**：奖励与动作同节拍（20 Hz）才有唯一的"上一拍"。
- `reset()` 把 `prev_force_mag` / `force_rate_has_prev` / `_force_rate_tick_seen` 与
  诊断的 `force_rate_max` / `_phys_prev_force_mag` **全部清零/置 False**
  ⇒ **每回合第一拍的 rate 精确为 0**（`torch.where` 把无样本的那一拍置 0），
  与仓库其它项「spawn 精确为 0」的纪律一致：出生那一拍不会因"从 0 到出生力"被误罚。
  **这不是掩护真实猛拽的漏洞**：出生段是 settle（指令 0，`SETTLE_TIME_S = 1.0 s` ⇒
  20 个控制步），真正的起步绷直发生在第几十拍、早过了首拍门。
- `apply_actions`（200 Hz）只更新诊断自己的 `_phys_prev_force_mag`，
  **不碰** 20 Hz 的 `prev_force_mag`（有静态守卫钉住）。

---

## 5. 20 Hz vs 200 Hz 采样限制

控制步长 `step_dt = 0.05 s`（20 Hz），物理步长 `physics_dt = 0.005 s`（200 Hz）。

**盲区**：一次**在一个控制步内完成**的"猛拽 → 回弹"冲量，在 20 Hz 的 `mag_t` 序列上
可能与"缓慢加载到同一水平"**完全一样**（两个采样点都只看到首尾值）⇒ 本惩罚项对它
**没有梯度**。同理，杆在一个控制步内 `+F → −F` 的快速反向穿越，两个采样点的 `mag`
都是 `|F|`，20 Hz 差分看到 rate ≈ 0。

**诊断量 `obs_force_rate_max`**（`weight = 1.0e-6`，不塑造策略）量化这个盲区：
`_apply_towing_physics` 每个物理子步累加

```python
force_mag = torch.linalg.vector_norm(self.towing_force_b, dim=1)
self.force_rate_max.copy_(torch.maximum(
    self.force_rate_max,
    (force_mag - self._phys_prev_force_mag).abs() / self.cfg.physics_dt))
self._phys_prev_force_mag.copy_(force_mag)
```

奖励项 `obs_force_rate_max` **读后清零**（`clone()` 再 `zero_()`）⇒ 每个控制步一个
"该控制步内最大的 5 ms 跳变"。读法：`Episode_Reward/obs_force_rate_max ÷ 1e-6`（N/s）。

**怎么用**：若它**远大于** `rate_limit_n_per_s = 200 N/s`，说明真实冲量主要发生在子步
尺度上——**该提高控制频率或改物理侧限速，而不是继续加 `towing_force_rate` 的权重**
（加权只会让 20 Hz 能看见的那部分过重，看不见的那部分依然看不见）。
**待训练机验证**：这个比值目前**没有任何读数**（本机无 Isaac Lab）。

---

## 6. 与 `stop_towing_force` 的分工

两项都在管"力"，但管的是**完全不同的东西**，不冗余：

| | `stop_towing_force`（−1.0，既有） | `towing_force_rate`（−0.5，本项） |
|---|---|---|
| 量 | `‖F‖ / (‖F‖ + 10)`（**末端幅值**） | `Δ‖F‖ / dt`（**过程中的变化率**） |
| 相位 | 只在 `elapsed_s ≥ stop_time_s`（指令归零之后）生效 | 全程（settle/tow/STOP 都有定义） |
| 教什么 | "到点了就把拉力卸掉"（别一直拽着/把小车当锚） | "别猛拽、别在负载态断崖掉力"；绳松弛那一下放行 |
| 盲区 | 对"过程里的力尖峰"完全无感（只看当前幅值） | 对"子步内的冲量"无感（20 Hz 采样，见 §5） |

一句话：**`stop_towing_force` 管末端 `‖F‖ → 0`，`towing_force_rate` 管过程不突变。**
两者可以同时为 0（停车后缓慢卸力）也可以同时非 0（停车后仍猛拽一下），不是同一约束记两遍。

---

## 7. 未验证项（只有训练机能回答）

本机**没有 Isaac Lab**（`omni.kit` 缺失，`upper_mdp` 离线 import 不了），
`pytest imgo2_rl/tests -q` ⇒ **550 passed** 是**纯离线**结论。以下全部待训练机：

1. **权重与阈值的实跑标定**：−0.5 / 200 / 600 / 1.0 / 3.0 一个都没在仿真里标定过；
   `up = 1 ⇒ −0.025/步` 是**按 RewardManager 口径的算术**，不是实测奖励读数。
2. **reset 后第一拍是否真的精确为 0**：离线测的是函数语义（`force_rate_has_prev=False`）
   与源码顺序；`process_actions` 与 reward 在 `ManagerBasedRLEnv.step` 里的实际调用时序、
   以及 env 复位后第一拍的 reward 计算，都要在训练机读 `Episode_Reward/towing_force_rate`。
3. **20 Hz 采样盲区的严重程度**：只有 `Episode_Reward/obs_force_rate_max ÷ 1e-6` 的读数
   能回答（§5）。若它 ≫ 200 N/s，应先动控制频率/物理侧，而不是加权重。
4. **方向语义在真实轨迹上的表现**：牵引起步"绷直"应给明显负值、绳行松弛卸载那一下应为 0、
   杆行的"负载态断崖"应被 `down` 抓到——这些都是**预期**，没有一条有运行证据。
5. **与跟踪项的相对量级**：需要把 `Episode_Reward/towing_force_rate` 与
   `Episode_Reward/tracking_velocity` / `stop_towing_force` 并排看，确认不压倒跟踪项。

**建议的训练机步骤**：64 环境 × 20 轮冒烟 → ① 看第一拍是否为 0；② 看
`obs_force_rate_max` 与 200 N/s 的比值；③ 按相位看本项量级；④ 再决定
`rate_limit_n_per_s`（主旋钮）与权重。

**一处刻意没做的小事**：`towing_on_policy_runner.py` 的 `--compact-log` 分项 `picks`
**未加** `towing_force_rate`。原因：那一行把名字截断到 **9 个字符**
（`k.replace('Episode_Reward/', '')[:9]`），`towing_force_rate` 与既有的 `towing_force_y`
都会显示成 `towing_fo`（同一个标签、两个数，反而误导）。完整分项本来就在 TensorBoard 的
`Episode_Reward/<项名>` 里；若要进终端，必须同时把截断宽度调大（会改动所有行的格式），
留给训练机按需要决定。
