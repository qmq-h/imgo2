# v6 model2000 latent 分析（2026-11-15）

> 数据：`imgo2_rl/logs/towing/play_test/ppo53_v6_roll10_fall500_m2000_loaded_70col_20261114_latent/report.json`
> （`--dump-latent`；1400 有载 case，5–30 kg 六档，70 列/15° 最大；v6/12 维、
> 50 Hz、fall −500 + roll −10 训练）。latent 是每 case 的 **phase 均值**（16 维），
> 不是逐步轨迹；`grade_deg` 是车道最大坡度，不是当前局部坡段。
> 工具：`analyze_latent_clusters.py`（标准库 + matplotlib）与
> `analyze_latent_tsne.py`（numpy/sklearn/matplotlib）。

## 1. 结论

### station（静止）

- **绳 vs 杆可分**：kNN5 LOO **1.000**、质心 LOO 0.958、ARI **0.728**、纯度 0.927、
  轮廓 0.727；compliant / inextensible / rigid 三类的 kNN 0.761、ARI 0.504。
- **compliant 与 inextensible 仍混在一起**：三类 ARI 只有 0.504，混淆矩阵里两种绳互相错判。
- **质量只在 rigid 内明显**：rigid 质量 η² **0.611**、kNN 0.739、k=4 ARI 0.257；
  compliant / inextensible 的质量 η² 只有 0.008/0.019，基本不可分。
- 坡度在 station 不可分（η² 0.013，kNN 0.296 ≈ 多数类 0.286）。

⇒ **与 2026-10-12 的 station 结论一致**：能分“绳/杆”，分不出两种绳；质量只在 rigid 下可见。

### tow（牵引）

| 标签 | 多变量 η² | 质心 LOO | kNN5 LOO | 多数类 | 轮廓 |
|---|---:|---:|---:|---:|---:|
| 连接类型（3类） | 0.010 | 0.424 | 0.623 | 0.400 | −0.004 |
| 绳 vs 杆 | 0.009 | 0.557 | 0.835 | 0.600 | 0.015 |
| 质量档（6档） | 0.038 | 0.321 | 0.606 | 0.167 | −0.065 |
| 坡度量级（6档） | **0.237** | 0.476 | **0.806** | 0.286 | −0.009 |
| 速度档（2档） | 0.172 | **0.937** | **0.996** | 0.503 | 0.257 |

- 连接：高维 kNN 有 0.835，但 **ARI ≈ 0、轮廓 0.015**；t-SNE 上绳/杆完全叠在一起 ⇒
  **没有可用的低维连接簇**，只能算弱的高维痕迹。
- 质量：kNN 0.606 高于多数类 0.167，但 silhouette −0.065，二维投影里六档重叠；
  三个连接内部质量 η² 只有 0.038/0.047/0.041、k=4 ARI 0.007–0.010 ⇒ 质量在 tow 也弱。
- **地形/坡度最强**：η² 0.237、kNN 0.806；t-SNE 上 15°（红）与 0/5/10/wave/bump 形成清晰分带。
- **速度几乎完全可分**：kNN 0.996；t-SNE 中 0.5/1.0 m/s 分成两大块。

### coast（滑行）

- 连接：η² 0.016、kNN 0.542 ≈ 多数类 ⇒ 弱；
- 质量：η² 0.033、kNN 0.351 ⇒ 弱；
- 坡度：η² 0.120、kNN 0.671；
- 速度：kNN 0.895。

## 2. 图

- 连接（rope vs rigid，PCA）：
  `imgo2_rl/logs/towing/play_test/ppo53_v6_roll10_fall500_m2000_loaded_70col_20261114_latent/latent_plots/latent_connection_two_pca.png`
- 质量（PCA）：
  `.../latent_plots/latent_mass_pca.png`
- 地形（t-SNE）：
  `.../latent_plots/latent_terrain_tsne.png`
- 连接（t-SNE）：
  `.../latent_plots/latent_connection_two_tsne.png`
- 质量（t-SNE）：
  `.../latent_plots/latent_mass_tsne.png`
- 聚类工具原生图：`.../latent_clusters/pca_labels.png`、
  `pca_mass_by_connection.png`、`pca_kmeans_connection.png`

PCA：PC1 **37.7%**、PC2 **23.5%**；连接/质量在二维投影里都大面积重叠，地形/速度才是主分离轴。

## 3. 与之前结论是否一致

| 结论 | 旧（2026-10-12，v3/800） | 本次（v6/1400，fall/roll） | 是否一致 |
|---|---|---|---|
| station 绳 vs 杆可分 | ARI 0.704、kNN 高 | ARI 0.728、kNN 1.000 | **一致** |
| station 两种绳混在一起 | 是 | 是（ARI 0.504，互相错判） | **一致** |
| station 质量只在 rigid 可见 | rigid η² 0.655；绳≈常数 | rigid η² 0.611；绳 0.008/0.019 | **一致** |
| tow 连接弱 | ARI≈0 | ARI 0.010/0.001，t-SNE 混叠 | **一致** |
| tow 质量可分 | 旧报告：三类连接都可分，rigid 最强 | 本次质量 η² 0.038、kNN 0.606、ARI≈0 | **不一致/变弱** |
| tow 地形 | 旧数据只有 0/5/10 | 本次 0/5/10/15/wave/bump，地形最强 | 新增结论 |

解释：

- station 的绳/杆可分主要来自**静止时 rigid 预载、绳松弛**的物理差异；质量只在 rigid 下有痕迹，
  这与旧结论完全一致，是物理可解释的。
- tow 质量变弱可能来自本次 **v6/50Hz、fall −500 + roll −10、5–30 kg、15°/障碍网格**，
  以及 phase 均值把不同坡段/速度混在一起；不能直接说“模型学不到质量”，但**这版 latent
  在 tow 的质量信息明显弱于旧 v3**。
- tow 的强分离轴变成了 **速度 + 地形**，连接只剩弱高维痕迹；这点与旧“连接弱”一致，
  但质量结论不一致，需要按同工况（固定速度/地形）重跑才能判断是模型差异还是工况混淆。

## 4. 限制

- 只有 `tow` 的 phase 均值；看不出局部坡段和时序；
- `grade_deg` 是车道最大坡度，不是当前地形；
- 连接/质量标签与列/行绑定，可能是 cell 位置/地形在编码，不一定是物理参数；
- 单模型、单 checkpoint；未做多 seed；
- t-SNE 只用于可视化，不是聚类证据；可分性以 16 维 kNN/η²/ARI 为准。

## 5. 控制工况：平地 + 1.0 m/s，按连接拆质量

使用 `--filter grade_deg=0 --filter velocity_mps=1.0`，tow 阶段：

| 连接 | n | 质量 η² | kNN5 LOO | k=4 ARI |
|---|---:|---:|---:|---:|
| compliant | 85 | 0.194 | 0.424 | 0.072 |
| inextensible | 47 | 0.265 | 0.404 | 0.080 |
| rigid | 73 | 0.248 | 0.329 | 0.056 |

对比全部 tow 聚合的 mass η² 0.038：控制地形/速度并按连接拆开后，质量结构明显增强
（0.19–0.27），但仍没有低维质量簇（kNN 0.33–0.42、ARI≈0.07）。

图（已归档）：

- [平地 1.0，质量 PCA 按连接](data/latent_v6_fall500_2026-11-15/latent_mass_flat_v1_pca_by_connection.png)
- [平地 1.0，质量 t-SNE 按连接](data/latent_v6_fall500_2026-11-15/latent_mass_flat_v1_tsne_by_connection.png)

## 6. 与旧 v3 的质量图对比 / “head 差但 latent 好”的解释

旧 v3（2026-10-12，5/10/15/20 kg、800 env、20 Hz）的质量图：

- [旧 tow 全量](data/latent_cluster_2026-10-12/tow/pca_mass_by_connection.png)
- [旧 tow 平地 + row0–9](data/latent_cluster_2026-10-12/tow_rows0-9_grade0/pca_mass_by_connection.png)
- [旧 station](data/latent_cluster_2026-10-12/station/pca_mass_by_connection.png)

旧图里 5/10/15/20 的质量带很清楚；当前 v6 平地 1.0 按连接拆开后只有局部梯度。

但旧 v3 的显式 mass head 误差中位数约 **5.001 kg**（station 接近常数预测），tow 也约
2.8–3.2 kg，所以「head 差、latent 却能分」并不矛盾：

1. **latent 是表示，mass head 只是读出层**：head 因尺度/门控/优化不足塌成常数，不代表共享
   decoder GRU 没编码质量；质量损失仍会通过共享层塑造 latent。
2. **质量本身改变动力学**：牵引力、速度响应、绳张力/伸长都可被机器人本体感觉观测；即使没有
   head，latent 也能从动力学推断质量。
3. **旧工况更干净**：4 档质量、0/5/10°、20 Hz、800 env；当前 6 档、0–15° + 波浪/减速带、
   50 Hz、fall/roll，latent 方差被地形/速度占走。
4. **质量标签与列/行绑定**：旧图的质量带可能混入长度/几何/工况；旧数据用 row0–9/grade0
   控制了一部分，但不是完全独立。

因此旧图不能直接证明“显式 mass head 让 latent 更好”；要确认需做：

- 对旧 v3 latent 做线性/最近质心 mass probe，和 mass head 误差对比；
- 当前 v6 同域加 explicit mass head（v8：explicit 6，actor 79）与当前 v6 做 A/B；
- 同时看 head MAE、latent 质量 η²/kNN/ARI，以及控制平地 1.0 + 连接拆分后的图。

## 7. 归档数据

- 压缩 latent 全量：`docs/towing/data/latent_v6_fall500_2026-11-15/latent_points.csv.gz`
  - 4200 行 = 1400 case × station/tow/coast；
  - 每行 16 维 phase 均值 + connection/mass/grade/velocity/length/row/column；
  - 仅 `cart_present=true`。
- 元数据/源 report SHA256：同目录 `metadata.json`。
- 图：同目录 6 张（连接 PCA/t-SNE、质量 PCA/t-SNE、平地 1.0 质量按连接、地形 t-SNE）。
