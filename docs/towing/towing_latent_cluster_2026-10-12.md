# Towing latent 聚类检验：连接类型可分性与连接内的质量分布（2026-10-12）

> **2026-10-16 变更（混合网格：减速带 + 波浪）**：训练网格已变为 **60 列 × 20 行 = 1200 环境** —— 在原 40 列后追加了 10 列**减速带块**（列 40–49，一条 300/100/50 mm 等腰梯形凸条）与 10 列**波浪块**（列 50–59，3 周期正弦、振幅 10 cm，占与平坡相同的 2.25–9.0 m 窗口）。本文里的「40 列 × 20 行 = 800 / `GRID_SIZE` = 800 / 默认 `--num-envs 800`」等写法是变更前的事实；当前默认环境数是 **1200**（timeout 485 步，`STOP_DISTANCE_M = 9.5`）、`--terrain-profile` 只有 `hill|rolling`。详见 [混合地形记录](towing_mixed_terrain_2026-10-16.md)。

用户要求：**先检验聚类分布，先看能不能区分不同连接，再看不同连接下各个质量档的分布。**

## 结论（先读这段）

1. **能分开的是「绳 vs 杆」这条粗边界，不是三类连接。** 无监督 k-means 在站定阶段的 350 个受控样本上把 **210 个绳样本（compliant 140 + inextensible 70）压成一个簇、rigid 140 个分成两簇**，簇里**没有一个绳样本混进 rigid 簇**（绳 vs 杆：纯度 0.920、ARI 0.703）；把坡度和长度都按住（平地、绳杆同长度带）结论不变（纯度 0.918、ARI 0.695）。
2. **compliant（弹性绳，4 档 k/c）与 inextensible（低弹性绳）在 latent 里完全分不开**：任何一轮 k-means 都把它们放进同一个簇，站定阶段两者更是重叠在同一个点上（标准化空间 PC1 极差 **0.01**）。三类监督式留一分类的上限也只有 0.55–0.67（多数类 0.40–0.41）。
3. **牵引/滑行段（tow / coast）的全局聚类被质量与速度主导**，连接信息仍在但不是主导轴：tow 全量 700 例上 k-means k=3 对连接 **ARI 0.001**（≈随机），而把坡度固定到平地后升到 **0.327**；把质量与速度也固定（n=13）后绳 vs 杆 **ARI 1.000**。⇒ 连接在 latent 里**存在但方差远小于质量/速度**，不是「学不到」。
4. **质量档：站定阶段只在 rigid 连杆下可见，绳下完全不可见**——绳组 latent 半径 **0.020**（近似常数）、质量 η² 0.023/0.030 且置换 p=0.36/0.67、k=4 ARI ≈ 0、decoder 质量输出误差中位 **5.001 kg = 常数预测**；rigid 组半径 **3.468**、η² 0.737（p=0.002）、k=4 ARI 0.514、PC1 随质量单调（5 kg 0.15 → 20 kg 5.97）。**tow/coast 段三类连接都能分辨质量**，rigid 最强（tow：η² 0.545 vs 0.391/0.410，kNN 留一 0.993，k=4 ARI 0.460）。

物理读法与观测无关：站定时绳按出生比 0.8 松弛（`connection_grid.SLACK_RATIO`）⇒ 不传力 ⇒ 负载无可观测痕迹；rigid 是 `d ≡ L0` 的预载连杆 ⇒ 静止时也传力。牵引/滑行段三类都绷紧受力，于是都能被推断，耦合最强的 rigid 最清楚。

## 数据与口径

| 项 | 值 |
|---|---|
| 数据源 | [v3_800_policy_c008/report.json](imgo2_rl/logs/towing/play_test/v3_800_policy_c008/report.json)（`--dump-latent`，2026-10-11 20:01，800 case × station/tow/coast 各 16 维）——**本机唯一带 latent 的产物**（其余 report 的 `"latent"` 计数为 0） |
| 样本 | 800 = 700 有车 + 100 无车（`--no-cart-fraction 0.125`）；无车样本的 `cart_mass_kg` 只是占位标签，**默认排除**（`--include-nocart` 才纳入） |
| 工况网格 | 40 列 × 20 行；列 = 连接类型 × 坡度（前 20 列平地、10 列 5°、10 列 10°），行 = 长度（绳 0.5–1.5 m、杆 0.5–1.0 m，各自区间） |
| 平衡性 | 连接 compliant/inextensible/rigid = 320/160/320（含无车 40/20/40）；质量 5/10/15/20 kg 各 200；同一连接内质量四档均等；连接与坡度按比例平衡（各 50%/25%/25%） |
| 受控带 | **row 0–9 是绳与杆长度完全相同的带**（0.5–1.0 m），且每行都含三类连接 ⇒ 用 `--rows 0 9` 可以把「长度」这一混淆项按住；再叠加 `--filter grade_deg=0` 按住坡度 |
| latent 的输入 | 58 维 frame = `loco_command` 3 + `last_action` 13 + `base_ang_vel` 3 + `projected_gravity` 3 + `last_loco_action` 12 + `joint_pos` 12 + `joint_vel` 12（[upper_logic.py](imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/upper_logic.py) 的观测项与 [upper_mdp.py](imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/upper_mdp.py) 的 `policy_frame`）——**没有任何小车位置/速度、连接类型/长度、质量、力的输入** ⇒ 下面的可分性是「从机器人自身本体感觉与指令历史推断负载」，不是把标签喂进网络的泄漏 |

与 2026-10-11 那轮数字的差别只在样本口径：`analyze_latent.py` 用的是全部 800 例，本文默认排除无车 100 例。用 `--include-nocart` 可以**逐项复现**旧口径（见「验证」）。

## 方法

新增纯离线工具 [analyze_latent_clusters.py](imgo2_rl/scripts/tools/analyze_latent_clusters.py)（核心指标只用标准库，与仓库其它离线检查一致；只有 `--plots` 需要 matplotlib）：

1. 16 维 latent 按列标准化 → 手写幂迭代 PCA（复用 `analyze_latent.py` 的线性代数）；
2. **无监督**：k-means++ 多初始化（固定种子）→ 与标签比 **ARI / NMI / 纯度**（并与「多数类纯度」对照）；
3. **局部/非线性**：留一最近质心（含混淆矩阵）与留一 kNN(k=5)；
4. **显著性**：标签置换检验（默认 500 次）分别打乱 η² 与质心 LOO 的零分布，给出 p；ARI/NMI 本身就以 0 为随机水平；
5. **投影无关的组内尺度**：`latent 半径` = 组内逐维方差的平方根之和（≈0 ⇒ 该组 latent 近似常数，不受 PCA 二维投影误导）；
6. 受控子集：`--rows`、`--filter`（本例用 `grade_deg` / `cart_mass_kg` / `velocity_mps`）、`--connection-binary`（把两类绳并成 rope）。

判读规则：**ARI/NMI/纯度 ≈ 0/多数类 ⇒ 无监督分不出该标签；η² 与质心 LOO 的置换 p 才是「结构是否显著」**。轮廓系数在全量混合工况下会被大簇+小簇压低，只作参考。

## 1. 连接类型能不能分开

`latent 半径`/η² 是整组尺度，ARI 是 k-means 与标签的一致性（k=3 对三类、k=2 对绳 vs 杆）。

| 子集（阶段/筛选） | n | 连接 η² (p) | 绳vs杆 η² | 轮廓 | 质心 LOO | k=3 ARI (纯度) | 绳vs杆 ARI (纯度) |
|---|---|---|---|---|---|---|---|
| station 全量 | 700 | 0.664 (0.002) | 0.664 | 0.129 | 0.639 | 0.371 (0.753) | **0.704 (0.920)** |
| station row 0–9（同长度带） | 350 | 0.699 (0.002) | 0.699 | 0.143 | 0.551 | 0.522 (0.800) | **0.703 (0.920)** |
| station row 0–9 + 平地 | 170 | 0.705 (0.002) | 0.705 | 0.123 | 0.671 | 0.538 (0.824) | 0.695 (0.918) |
| tow 全量 | 700 | 0.147 (0.002) | 0.145 | 0.050 | 0.610 | **0.001 (0.417)** | 0.021 (0.600) |
| tow row 0–9 | 350 | 0.155 (0.002) | 0.151 | 0.063 | 0.629 | **0.001 (0.426)** | 0.035 (0.600) |
| tow row 0–9 + 平地 | 170 | 0.194 (0.002) | 0.193 | 0.063 | 0.541 | **0.327 (0.718)** | 0.061 (0.629) |
| tow 平地 + 10 kg + 1.0 m/s | 13 | 0.786 (0.002) | 0.765 | 0.285 | 0.769 | 0.781 (0.923) | **1.000 (1.000)** |
| coast 全量 | 700 | 0.161 (0.002) | 0.158 | 0.068 | 0.666 | 0.228 (0.640) | 0.139 (0.687) |
| coast row 0–9 + 平地 | 170 | 0.355 (0.002) | 0.354 | 0.056 | 0.600 | 0.312 (0.718) | **0.618 (0.894)** |

**簇构成（station row 0–9，k=3）**——把「区分连接」具体化成谁和谁混：

| 簇 | n | compliant | inextensible | rigid | 质量构成 |
|---|---|---|---|---|---|
| C0 | 210 | 140 | 70 | **0** | 四档均匀 |
| C1 | 47 | 0 | 0 | 47 | 5 kg 34 / 10 kg 13 |
| C2 | 93 | 0 | 0 | 93 | 10 kg 21 / 15 kg 36 / 20 kg 36 |

即：**k-means 实际找到的两个方向是「绳/杆」（完全干净）与「杆的载荷轻重」**；`compliant` 与 `inextensible` 从未被任何一轮聚类分开。留一最近质心的连接混淆矩阵（station row 0–9）同样显示绳内部互混：compliant 55 对 / 85 错（几乎全错判成 inextensible），inextensible 16 对 / 54 错，rigid 0 混入绳。

**受控后的坐标**（station row 0–9，标准化空间）：210 个绳样本 PC1 ∈ [−2.72, −2.69]（极差 0.01）、PC2 中位 −0.38，且**与质量档无关**（四档 PC1 中位都是 −2.70）；rigid 覆盖 PC1 ∈ [−0.87, 6.66]，PC1 中位随质量单调 0.15 → 4.11 → 5.50 → 5.97。

## 2. 每个连接内部的质量档分布

| 阶段/子集 | 连接 | n | latent 半径 | 质量 η² (p) | 质心 LOO（多数类） | kNN LOO | k=4 ARI | decoder 质量误差中位 |
|---|---|---|---|---|---|---|---|---|
| station | compliant | 280 | **0.022** | 0.005 (0.882) | 0.186 (0.250) | 0.200 | −0.009 | **5.001 kg** |
| station | inextensible | 140 | **0.024** | 0.004 (0.978) | 0.200 (0.250) | 0.207 | −0.012 | **5.002 kg** |
| station | rigid | 280 | 3.668 | **0.655 (0.002)** | 0.679 | 0.861 | 0.389 | 4.752 kg |
| station 平地 + row0–9 | compliant | 70 | 0.022 | 0.053 (0.305) | 0.329 | 0.157 | 0.003 | 5.001 kg |
| station 平地 + row0–9 | inextensible | 30 | 0.014 | 0.117 (0.321) | 0.333 | 0.200 | 0.030 | 5.000 kg |
| station 平地 + row0–9 | rigid | 70 | 3.387 | **0.765 (0.002)** | 0.757 | 0.857 | **0.629** | 4.609 kg |
| tow | compliant | 280 | 3.669 | 0.391 (0.002) | 0.736 | 0.893 | 0.235 | 2.799 kg |
| tow | inextensible | 140 | 3.557 | 0.410 (0.002) | 0.757 | 0.943 | 0.267 | 3.234 kg |
| tow | rigid | 280 | 3.787 | **0.545 (0.002)** | 0.893 | 0.993 | **0.460** | 3.023 kg |
| coast | compliant | 280 | 3.435 | 0.318 (0.002) | 0.654 | 0.818 | 0.283 | 3.179 kg |
| coast | inextensible | 140 | 3.786 | 0.261 (0.002) | 0.671 | 0.886 | 0.323 | 3.341 kg |
| coast | rigid | 280 | 3.821 | **0.456 (0.002)** | 0.811 | 0.946 | 0.433 | 2.483 kg |
| coast 平地 + row0–9 | 三类 | 70/30/70 | 2.6–3.8 | 0.69–0.74 (0.002) | 0.767–0.943 | 0.73–0.96 | 0.270–0.546 | 2.007–4.401 kg |
| **tow row0–9 + 速度固定 1.0 m/s**（去掉速度混淆） | compliant | 43 | 3.490 | **0.773 (0.002)** | 0.953 | 0.977 | **0.562** | 2.931 kg |
| 同上 | inextensible | 22 | 3.319 | **0.727 (0.002)** | 0.955 | 0.773 | 0.435 | 3.882 kg |
| 同上 | rigid | 55 | 3.618 | **0.811 (0.002)** | 0.964 | **1.000** | **0.685** | 2.791 kg |

**固定速度后质量可分性反而更高**（全局质量 η² 0.358 → **0.565**、kNN 0.940 → 0.975、k=4 ARI 0.217 → 0.394），说明「质量可读」不是质量与速度/长度混在一起的假象。质量轴的形状是**单调连续轴、不是四个团**：混淆集中在**低端 5 vs 10 kg**（tow 全量 compliant 70 例里 5 kg 有 12 例被判成 10 kg、10 kg 有 17 例被判成 5 kg），15/20 kg 之间只有零星互判；把坡度和长度按住后（平地 + row0–9）只余 15 vs 20 的 1–2 例。**注意这与 decoder 的显式质量输出不是一回事**：同一批 case 上 decoder 质量误差中位仍是 2.8–3.2 kg（5 kg 一档的间距），即 **latent 里有信息、读出端没读到那个精度**。

读法：
* **站定阶段**：绳组 latent 半径 0.020/0.024 ≈ 常数，四档质量、三种坡度、两种绳长都挤在同一点；质量 η² 的置换 p 是 0.36–0.98（不显著）、k=4 ARI ≈ 0、decoder 质量输出误差中位 5.001 kg（就是全局质量中位 ⇒ 输出常数）。**「站定时绳子不传力」这条物理判断在 latent 上和 decoder 输出上同时成立。**
* **rigid 在站定阶段反而是最强的质量读出口**：半径 3.468、η² 0.737、k=4 ARI 0.514、PC1 随质量单调，且 k=3 时直接按约 10 kg 把 rigid 切成轻/重两簇。平地带（去掉坡度后）更强（η² 0.765、ARI 0.629）⇒ 不是坡度效应。
* **tow / coast 段三类都能分辨质量，rigid 一致最强**（kNN 留一 0.99 / 0.95，k=4 ARI 0.46 / 0.43），绳类次之；质量估计误差中位也在同一排序（tow：rigid 3.02 / compliant 2.80 / inextensible 3.23 kg，coast：rigid 2.48 最小）。

## 3. 轮轴阻尼（轮摩擦）：行为层已经很清楚，latent 层缺第二个取值

用户追问「看看对轮摩擦的情况呢」。**结论：现在测不了 latent，但行为层能给出干净答案，而且缺的东西只差一个开关。**

### 3.1 为什么 latent 现在测不了（已核实）

* `--wheel-damping` 与 `--ground-friction` 都是**整轮一个标量**（[play_towing_test.py](imgo2_rl/scripts/towing/play_towing_test.py)：默认 0.008 / 0.8，即训练名义值），report 只在 `arguments` 里记全局值，**不逐 case 记录**（case 字段只有 `env_index/cell_index/column/row/connection/connection_length_m/grade_deg/velocity_mps/cart_mass_kg/cart_present`）⇒ 单轮报告内部**没有任何摩擦标签**可聚类。
* 本机 20 份 play_test report 里：**地面摩擦 μ 全部是 0.8**（训练侧随机 0.4–1.2，测试台上从未扫过）；**轮轴阻尼只有 0.008 与 0.032 两个值**。
* **带 latent 的只有 c=0.008 那一轮**。与它最对应的 c=0.032 轮 [`v3_800_policy`](imgo2_rl/logs/towing/play_test/v3_800_policy/report.json) 其余全同（同 checkpoint `model_2700.pt`、同 800 网格、同 4 档质量 × 3 档速度、同 μ=0.8、同代码 `4b2468ee`），只是 `dump_latent` 没开 ⇒ **「按阻尼分组的 latent 可分性」目前无数据**。

### 3.2 行为层：同一策略、同一网格，只把阻尼换一端（逐 cell 配对，700 有车样本）

| 轴 | n | 硬证据 c=0.032 | 硬证据 c=0.008 | delta | 滑移中位 m | 滑移 P95 m | 最小间隙中位 m | 甲板峰值力 P95 N |
|---|---|---|---|---|---|---|---|---|
| 全体 | 700 | 3.6% | **23.0%** | +19.4pt | 0.26 / 0.55 | 1.57 / **3.80** | 0.543 / 0.372 | 0.0 / **255.3** |
| compliant | 280 | 5.0% | 36.8% | +31.8pt | 0.47 / 1.11 | 1.74 / 3.83 | 0.618 / 0.188 | 0.9 / 444.0 |
| inextensible | 140 | 7.9% | 39.3% | +31.4pt | 0.36 / 0.88 | 1.82 / 4.17 | 0.606 / 0.147 | 31.5 / 369.3 |
| rigid | 280 | **0.0%** | 1.1% | **+1.1pt** | 0.13 / 0.16 | 0.56 / 2.80 | 0.463 / 0.460 | 0.0 / 0.0 |
| 速度 0.5 / 1.0 / **1.5** m/s | 230 / 240 / 230 | 1.3 / 0.4 / 9.1% | 9.1 / 9.6 / **50.9%** | +7.8 / +9.2 / **+41.7pt** | — | — | — | — |
| 质量 5 / 10 / 15 / **20** kg | 各 175 | 1.7 / 4.0 / 3.4 / 5.1% | 12.6 / 22.3 / 26.9 / **30.3%** | +10.9 / +18.3 / +23.4 / **+25.1pt** | — | — | — | — |
| 坡度 0° / 5° / 10° | 340 / 180 / 180 | 0.3 / 8.3 / 5.0% | 17.1 / 30.0 / 27.2% | +16.8 / +21.7 / +22.2pt | — | — | — | — |

* 口径与 [compare_play_runs.py](imgo2_rl/scripts/tools/compare_play_runs.py) 一致（硬证据 = `stop.deck_contact_peak_n > 0` 或 `stop.min_clearance_coast_m ≤ 0`）；**25/700 = 3.6% 与 161/700 = 23.0%**，在 800 格口径上就是该工具报的 **3.1% / 20.1%**（100 个无车格两轮都无碰撞，只起稀释作用）。
* 机制非常直白：阻尼减到 1/4 ⇒ **STOP 后小车滚得更远**（滑移中位 0.26 → 0.55 m，P95 1.57 → 3.80 m）⇒ 最小间隙被吃掉（0.543 → 0.372 m）⇒ 甲板被撞（峰值力 P95 从 0 涨到 255 N）。**代价几乎全部落在绳侧**：两类绳各 +31pt，rigid 只有 3 格（+1.1pt，因为它 `d≡L0` 两端被约束住，小车滚不出去）。
* 代理通道 `stop.contact` 在这条轴上完全读不出方向（rigid 甚至 52% → 44%，看着像"变好"）——再次印证判读只能用硬证据。
* **量级参照**：同一 800 网格上，**端到端上层策略在 c=0.008 只救回 68 格**（28.6% → 20.1%，[§8.9](docs/towing/towing_play_test_2700_2026-10-11.md)）；而**同一策略把阻尼从 0.032 换到 0.008 就多出 136 格碰撞**。两者基准不同、不能直接相减，但**阻尼的影响量级大于当前上层策略的收益量级**这一点是清楚的。
* 产物：[wheel_damping_behavior/](docs/towing/data/latent_cluster_2026-10-12/wheel_damping_behavior/)（逐轴表 `ab_by_axis.md`、两轮各自的 `play_test_stats` 输出与逐 case CSV、`compare_play_runs.py` 原始输出）。

### 3.3 补测 latent 需要什么（命令已 `--dry-run` 通过）

只比已归档的 c=0.032 那轮多一个 `--dump-latent`，其余参数照抄它的 `arguments`；checkpoint 本机在盘上（`logs/towing_rl_lab/towing_upper/2026-10-10_16-59-58_v3-5to20kg-0.5to1.5/model_2700.pt`），**需要 GPU shell，本轮没有执行**：

```bash
python imgo2_rl/scripts/towing/play_towing_test.py --headless --num-envs 800 \
    --velocities 0.5 1.0 1.5 --cart-masses 5 10 15 20 \
    --ground-friction 0.8 --wheel-damping 0.032 --no-cart-fraction 0.125 --dump-latent \
    --upper-checkpoint logs/towing_rl_lab/towing_upper/2026-10-10_16-59-58_v3-5to20kg-0.5to1.5/model_2700.pt \
    --output-dir imgo2_rl/logs/towing/play_test/v3_800_policy_c032_latent
```

拿到之后要做的分析：把两轮 latent 池化、按 `wheel_damping` 打标签，与质量/连接同样地做 η²、kNN 留一、k-means ARI 和置换检验（**需要给 `analyze_latent_clusters.py` 加多报告池化能力**，例如 `--report LABEL=PATH`；也可以先在两轮上分别算同一批指标再对比）。

两点必须写清的限制：① 阻尼是**整轮标量**，两个取值只能给出"两组能不能分开"的粗结论，要做 dose-response 曲线得跑多档（如 0.008/0.016/0.024/0.032）；② **地面摩擦 μ 从未变过**（全轮 0.8），要测它得另跑 `--ground-friction 0.4` 与 `1.2`（训练随机域两端）并加 `--dump-latent`——μ 与 c 都只影响"小车滚多远"，两轴一起做才能回答哪一个在 latent 里更主导。

## 图（`docs/towing/data/latent_cluster_2026-10-12/`）

本目录同时放了两套图：**本工具的 PCA 图**（线性投影、可复现、判读以 16 维指标为准）与**既有 t-SNE 工具的真实工况图**（`analyze_latent_tsne.py` 首次在真 latent 上运行，每个子目录含 `metadata.json`（源文件 SHA256、KL、trustworthiness）与逐点 CSV；为控制仓库体积只保留了 PNG，PDF 可用同命令重新生成）。

PCA（`metrics.json` + `points.csv` + 3 张 PNG）：

* [station/pca_labels.png](docs/towing/data/latent_cluster_2026-10-12/station/pca_labels.png)：站定阶段两种着色——左图绳（圆/三角叠成一点）与杆（方块铺满弧线）分居两处；右图同一坐标按质量着色，**绳那一团四色完全重叠，杆那一串按质量从蓝到红依次排开**；
* [station/pca_kmeans_connection.png](docs/towing/data/latent_cluster_2026-10-12/station/pca_kmeans_connection.png)：k-means 三簇与连接对照（绳一团 + 杆两团）；
* [station/pca_mass_by_connection.png](docs/towing/data/latent_cluster_2026-10-12/station/pca_mass_by_connection.png)：按连接分面板按质量着色（黑 × = 该档质心）；
* [tow/pca_mass_by_connection.png](docs/towing/data/latent_cluster_2026-10-12/tow/pca_mass_by_connection.png)：牵引段三面板，三类的质量质心都可分辨、rigid 最开；
* 其余受控子集：[station_rows0-9/](docs/towing/data/latent_cluster_2026-10-12/station_rows0-9/)、[tow_rows0-9_grade0/](docs/towing/data/latent_cluster_2026-10-12/tow_rows0-9_grade0/)、[coast/](docs/towing/data/latent_cluster_2026-10-12/coast/)、[tow_fixed_m10_v1_rows0-9/](docs/towing/data/latent_cluster_2026-10-12/tow_fixed_m10_v1_rows0-9/) 各有同款图。

t-SNE（perplexity 10/30/50、seed 0，均未按标签拟合）：

* [tsne_station/tsne_station_p30_s0.png](docs/towing/data/latent_cluster_2026-10-12/tsne_station/tsne_station_p30_s0.png)：**rigid（绿）单独成带且沿带按质量 5→20 kg 排序，两类绳在上方结构里完全互混**；坡度与速度面板在两团里都没有结构 —— 与本文「只有绳/杆 + rigid 质量可读」的结论一致；
* [tsne_tow/tsne_tow_p30_s0.png](docs/towing/data/latent_cluster_2026-10-12/tsne_tow/tsne_tow_p30_s0.png)：牵引段，连接/质量/速度都能看到方向，但四类着色互相叠压 —— 与「连接信息存在但方差远小于质量/速度」一致。
* 其余参数组合（p10/p50 的 PNG、各参数点的 CSV 与 KL）在同两个目录里，**不能只按最漂亮的一张下结论**。

## 复现

```bash
PY=/opt/conda/envs/isaaclab/bin/python          # Python 3.11.13；核心指标不需要 numpy
export MPLCONFIGDIR=/tmp/mplcfg                 # 默认 ~/.config/matplotlib 不可写时才需要
R=imgo2_rl/logs/towing/play_test/v3_800_policy_c008/report.json

# 站定阶段 + 同长度带（本文头条那张）
$PY imgo2_rl/scripts/tools/analyze_latent_clusters.py --report $R --phase station \
    --rows 0 9 --connection-binary --permutations 500 --plots \
    --output docs/towing/data/latent_cluster_2026-10-12/station_rows0-9

# 牵引段全量 / 平地带 / 平地+固定质量速度 / 平地+固定速度（质量对照）
$PY imgo2_rl/scripts/tools/analyze_latent_clusters.py --report $R --phase tow --output <dir>
$PY imgo2_rl/scripts/tools/analyze_latent_clusters.py --report $R --phase tow --rows 0 9 \
    --filter grade_deg=0 --output <dir>
$PY imgo2_rl/scripts/tools/analyze_latent_clusters.py --report $R --phase tow --rows 0 9 \
    --filter grade_deg=0 --filter cart_mass_kg=10 --filter velocity_mps=1.0 --output <dir>
$PY imgo2_rl/scripts/tools/analyze_latent_clusters.py --report $R --phase tow --rows 0 9 \
    --filter velocity_mps=1.0 --connection-binary --output <dir>   # 质量可读性对照

# 旧口径（含 100 个无车样本，复现 analyze_latent.py 的 800 例数字）
$PY imgo2_rl/scripts/tools/analyze_latent_clusters.py --report $R --phase tow --include-nocart \
    --permutations 500 --output <dir>
```

## 验证与限制

**已跑过的验证**

* 新工具 10 项单元测试：ARI/NMI 的已知值与退化分支（对齐 scikit-learn 口径）、三分离簇上 k-means 恢复标签（ARI>0.99）、轮廓系数、kNN/质心留一、置换检验的确定性与灵敏度、无车/row/滤波选择、坏向量计数、端到端写 `metrics.json`（`pytest imgo2_rl/tests/test_towing_latent_clusters.py -q` ⇒ 10 passed）。
* 全量离线测试：`pytest imgo2_rl/tests -q` ⇒ **601 passed**（在 `/opt/conda/envs/isaaclab/bin/python` 下，numpy/torch 可用）。
* **与既有工具交叉核对**：加 `--include-nocart`（800 例）后，本文工具给出的连接/质量/坡度/速度留一质心准确率 **0.573 / 0.715 / 0.367 / 0.829** 与最优单维 η² **0.263(z11) / 0.716(z9) / 0.056(z8) / 0.740(z5)**，与 [analyze_latent.py](imgo2_rl/scripts/tools/analyze_latent.py) 的输出**逐项一致**（也就是 README 维护记录里 2026-10-11 那行引用的数字）。
* **确定性**：同一 `--seed` 重跑，`metrics.json` 与 `points.csv` 逐字一致（已 `diff`）。
* 文中每个数字都来自 `docs/towing/data/latent_cluster_2026-10-12/<子集>/metrics.json`，图由同一次运行的 `--plots` 生成；t-SNE 图由 [t-SNE 工具](imgo2_rl/scripts/tools/analyze_latent_tsne.py)在同一份报告上跑出（这解决了该工具记录里「真实实验图待数据」的一半：数据已存在且已跑，**报告本身仍未归档**，见 [t-SNE 使用记录](docs/towing/towing_latent_tsne_2026-10-11.md)）。

**限制（不要把上面的结论读过头）**

* **只有一轮策略**（`v3_800_policy_c008`，低阻尼端 c=0.008）。没有第二个 checkpoint / 阻尼端可以复现，因此「连接可分」目前只在这一轮上成立。
* **机理未定**：绳与杆的差别同时包含「松弛距离 0.8L vs 全长 L」「是否传力」「弹性 vs 约束」，本文只能说明 latent 分得开，**不能区分是力语义还是出生几何**。
* k-means 的 ARI 依赖簇数 k 与初始化；本文固定 `--seed 0`、多初始化取最优，跨 k 的 ARI 不可直接比大小。
* 二维 PCA/t-SNE 只是可视化（本工具用 PCA，另有 [t-SNE 工具](docs/towing/towing_latent_tsne_2026-10-11.md)）；**结论以 16 维上的 η²/ARI/留一为准**。
* 最紧的受控子集只有 13 例（平地 + 10 kg + 1.0 m/s），只能当小样本旁证。
* 无车样本（100 例）的 `cart_mass_kg` 是占位值，默认排除；纳入后连接可分性明显变差（η² 0.664 → 0.087），说明无负载样本会让 latent 的负载结构变糊。
* 本轮只做离线复算，**没有启动仿真或训练**；不涉及 decoder 权重、训练配置或部署产物。

## 待办

见 README 的 **TOW-CLUS-01**：补第二轮带 latent 的报告（换 checkpoint 或换 c=0.032 阻尼端）复核绳/杆边界；若要定机理，需要在「出生间隙对齐但传力不同」的对照工况下重跑测试台（当前网格把两者绑在一起，离线无法拆分）。
