# 连续坡面剖面：平地 → 上坡 → 坡顶 → 下坡 → 平地（2026-10-09）

> 跑道总长 **15 m**（2026-10-09 用户收窄：原 20 m），剖面 11.25 m：
> 平地 2.25 / 上坡 3 / 坡顶 0.75 / 下坡 3 / 平地 2.25（坡度角 5° 或 10° 不变）。

## 需求与改动动机

用户澄清：不要「40 列 × 20 行、每格一个恒定坡度」的大坡面网格，而要**整合到一起的一条剖面** ——
平地上放一段上坡加一段下坡，**放在机器人的行进路线上**，让一个回合内依次经历
平地 → 上坡 → 坡顶 → 下坡 → 平地。

原设计的缺口正在这里：每格只有**一个**恒定坡度，机器人在整段路程里经历不到坡度切换，
也遇不到坡顶；而「上坡加载 → 过顶 → 下坡被小车推」恰恰是最能暴露上层控制器问题的一段。

参数按用户选定：**5° 与 10° 两档；上/下坡各 4 m、坡顶平段 1 m、两端平地各 3 m；仍 20 行 × 40 列
= 800 条 lane，每条同样剖面**。坡度量级由列决定（0 / 5 / 10），lane 之间只差这个量级。

**首版落地后用户进一步收窄**（原话：「收窄的意思是不要 20 m，而是 15 m，对应缩小上下坡+平地的
长度」）：**跑道总长从 20 m 收到 15 m**，剖面各段按同一比例 **0.75** 缩放 ——
即平地 2.25 m → 上坡 3 m → 坡顶 0.75 m → 下坡 3 m → 平地 2.25 m（剖面 11.25 m），
`BACK_M = 2.25`、`FORWARD_M = 12.75`、`ROW_SPACING_M = 17`；坡度角不变。副作用是好的：
**10 m 目标现在落在出口平地上**（出口平地从 9.0 m 起），整条剖面都在默认回合内走完。

## 剖面定义（`mdp/slope_geometry.py`）

lane 局部坐标 `x`（沿 +x 前进；lane 原点在剖面平地段表面上，`h(0) = 0`）：

| 段 | x 区间 | 坡度 | 高度 |
|---|---|---|---|
| 出生平地 | 0 – 2.25 m | 0 | 0 |
| 上坡 | 2.25 – 5.25 m | +grade | 线性升到 `3·tan(grade)` |
| 坡顶平段 | 5.25 – 6.0 m | 0 | `3·tan(grade)` |
| 下坡 | 6.0 – 9.0 m | −grade | 线性降回 0 |
| 出口平地 | 9.0 – 11.25 m | 0 | 0 |

- 剖面 11.25 m ≤ 前向余量 `FORWARD_M = 12.75 m`（模块 import 时断言）；跑道总长
  `BACK_M + FORWARD_M = 15 m`（模块 import 时也断言）；板厚 0.35 m、半宽 3 m 不变。
- 坡顶高度：5° 档 0.262 m、10° 档 0.529 m。
- 新增纯函数：`profile_rise` / `profile_height` / `profile_slope_degrees` / `profile_frame` /
  `profile_arc_length` / `profile_polyline`；常量 `FLAT_IN_M / UP_M / CREST_M / DOWN_M / EXIT_M /
  UP_START_M / CREST_START_M / DOWN_START_M / FLAT_OUT_START_M / PROFILE_LENGTH_M / MAX_GRADE_DEG`。
- 跑道间距 `ROW_SPACING_M = BACK_M + FORWARD_M + 2 = 17 m`（原 22 m）；40×20 的网格世界尺寸
  随之缩小。
- `connection_grid.slope_degrees` 现在返回**坡度量级**（0 / 5 / 10），不再按行列交替正负：
  上坡与下坡在**同一条 lane 内成对出现**，方向平衡是构造上成立的（旧设计靠正负交替避免
  「某个弹性档只遇到上坡」，现在不需要了）。可用 lane 数：0° 400 条、5° 与 10° 各 200 条。

## 出生与判据怎么跟着改

- **出生在平地段起点**（`x ≈ 0`）：姿态竖直，**删掉了出生旋转 `R_y(−θ)`**；小车仍按
  「两挂点三维距 = 目标值」在后方平地上摆放（切向/法向解退化为平地解，采样偏航仍参与挂点旋转）。
- **跌倒判据**从「相对某个恒定平面」改成「离**局部**剖面多高」：
  `z − origin_z − profile_height(grade, x)`。绝对 z 在新地形上完全不能用（走 10 m 时坡面已经
  抬升 0.7 m，机器人会被判跌倒）。向量化实现放在新模块
  [profile_torch.py](../imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/mdp/profile_torch.py)
  （奖励/终止每个控制步对 800 个环境都要算，必须批量），与标量版由测试逐点交叉核对。
- **前进量 / 出界**不变：出生在平地段 ⇒ 切向就是 `+x`（`terrain_tangent_w = (1,0,0)`、
  `terrain_normal_w = (0,0,1)`），`progress = lane 局部 x`，边界检查仍用局部 x/y 与边界余量。
- **timeout 用坡面弧长**：10 m 水平目标在最陡档（10°）上是
  `profile_arc_length(10, 10) = 10.0926 m`，于是
  `episode_length_s = 1 + 10.0926/0.4 + 2 = 28.23 s`（原 28.0 s）⇒ `max_episode_length = 564`
  步（原 560）。修改目标距离或最低速度时 timeout 自动跟随。

## 目标距离：跑道收窄后已不再是问题

首版（跑道 20 m、剖面 15 m）里 **8–12 m 是下坡段**、12–15 m 才是出口平地，所以默认 10 m 目标
会在下坡段中段结束、走不到出口平地。跑道收到 15 m 后剖面变成 11.25 m、**出口平地从 9.0 m 开始**，
于是默认 10 m 目标恰好落在**出口平地上**（往里 1 m）：一个回合内完整走完
平地 → 上坡 → 坡顶 → 下坡 → 平地。timeout 仍由弧长决定：
`profile_arc_length(10, 10) = 10.0926 m ⇒ 1 + 10.0926/0.4 + 2 = 28.23 s`（564 控制步），
与收窄前相同（坡段的弧长没变，只有平地段的分配变了）。

## 闭合实体（mesh）怎么建

每块 lane 是一条闭合实体：把 x–z 剖面的闭合多边形（上表面折线 + **平的**底面）沿 y 挤出。
平地 lane 的多边形退化成矩形（8 顶点 / 12 面），坡道 lane 是 16 顶点 / 28 面；
整块 mesh 9600 顶点 / 16000 面。构建时踩到并修掉三个坑（都由测试钉住）：

1. **底面必须平**：第一版写成「上表面下移 0.35 m」，于是整块板跟着坡走（等厚薄壳），
   体积只有正确值的一半左右（正确值 = 剖面多边形面积 × 板宽，10° 档约 43.4 m³）。
   用 `_signed_volume` 与鞋带面积同时抓出。
2. **端面扇形三角化不能从上表面最左点出发**：平地段两端与坡顶点在 z 上共线，会造出零面积
   三角形（5°/10° 档各 2 个，PhysX 吃退化三角形会报错）。改成从**底面角点**扇出。
3. **闭合边索引写错**：侧壁四边形在最后一条多边形边上要用 `nxt = 0`，不能写 `index + 1`
   （那会指到一个不存在的顶点，网格不闭合：边重数出现 1 与 3）。

现在每块 lane 都通过：每条无向边恰好 2 个面、有符号体积 = 剖面面积 × 板宽、无零面积三角形、
上表面面法向朝上。

## 改动的文件

| 文件 | 改动 |
|---|---|
| `mdp/slope_geometry.py` | 剖面常量与纯函数；`tile_mesh` 改成挤出的闭合剖面实体；`slope_frame` / `attachment_root_positions` 保留 |
| `mdp/connection_grid.py` | `slope_degrees` 改为坡度量级（0/5/10），去掉行列正负交替；docstring 同步 |
| `mdp/profile_torch.py`（新） | `profile_height_tensor`：向量化剖面高度，供奖励/终止用 |
| `upper_mdp.py` | 出生不再有坡面旋转；`slope_angle` → `hill_grade_deg`；`terrain_tangent_w/normal_w` 变成常量基；`robot_fall` 减局部剖面高度 |
| `upper_env_cfg.py` | timeout 用 `profile_arc_length(MAX_GRADE_DEG, goal_distance_m)`；注释同步 |
| `tests/test_towing_slope_geometry.py` | 4 项检查重写为剖面语义 + 新增 2 项 torch/标量交叉核对 + 新增 2 项剖面分段/弧长检查 |
| `tests/test_towing_upper_rl_contract.py` | 两条源码契约改成剖面语义（fall 用局部剖面、timeout 用弧长） |
| `scripts/towing/play_towing_test.py` + 其测试 | `terrain` 后端改成坡度量级 0/5/10；离面高度与相对俯仰按剖面算（`gravity` 后端数值逐位不变） |

## 验证（本机 `/opt/conda/envs/isaaclab`，Python 3.11.13）

- `imgo2_rl/tests` 全量 **434 项通过 0 失败**；其中 `test_towing_slope_geometry.py` 8 项
  （剖面分段/局部坡度、弧长、闭合实体拓扑+体积+顶面法向、出生在平地段与边界、慢速不超时、
  torch 与标量逐点一致），`test_towing_upper_rl_contract.py` 39 项。
- `python imgo2_rl/scripts/tools/check_towing_slope_grid.py` → 18 项通过（含坡面几何与网格契约）。
- `play_towing_test.py --dry-run --slope-backend terrain` 正常；传负档会明确报错
  （"剖面自带上下坡，没有纯下坡的 tile"）。
- 网格自检：9600 顶点 / 16000 面；三类 lane 的体积分别等于解析值（42.0 / 52.5 / 63.16 m³）。

## 未验证（缺什么才能完成）

1. **Isaac Sim 未跑**（本会话进程 `cuda available = False`）：mesh 导入与 PhysX 碰撞、
   出生首拍刚体误差、机器人真能走上/下坡、目标与 timeout 在运行时的行为都待训练机跑一次。
2. **跑道收窄后的新数值**（剖面 11.25 m、坡顶 0.529 m、行距 17 m）只在离线几何检查里验证过，
   实跑时还要确认网格导入、地形分块间距与出生几何仍成立。
3. **出口平地/坡顶的接触细节**：坡顶是 1 m 平段 + 两处折角，机器人腿跨过折角时是否卡边
   没有数据；如果卡边，可把折角倒成小圆角或加长坡顶平段。
4. 上层 VAE 训练、新地形下的策略效果、以及旧 checkpoint 兼容性（本就 version=2 不兼容）都不在本记录范围。
