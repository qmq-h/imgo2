# Towing latent t-SNE 使用记录（2026-10-11）

已完成：新增独立离线工具 `imgo2_rl/scripts/tools/analyze_latent_tsne.py`，保留原有标准库分析脚本。输入为带 `--dump-latent` 的 `report.json` 或短窗口 JSON，不加载 checkpoint、不启动 Isaac Sim、不修改训练网络。

## 使用

在仓库根目录，使用具有 numpy、scikit-learn、matplotlib 的 Python：

```bash
python -m pip install numpy scikit-learn matplotlib
python imgo2_rl/scripts/tools/analyze_latent_tsne.py --report imgo2_rl/logs/towing/play_test/<run>/report.json --phase tow --output imgo2_rl/logs/towing/play_test/<run>/tsne_tow
```

默认标准化完整16维输入，perplexity=10/30/50、seed=0/42，各参数组合独立拟合；不以标签选维或拟合。每组二维坐标在各面板完全相同，只改变连接、质量、车道最大坡度、速度着色。输出每组PNG/PDF、坐标CSV，以及metadata.json（源文件SHA256、版本、参数、过滤、标准化参数、缺失计数、KL与邻域trustworthiness）。不在二维坐标上做分类或计算物理类别识别准确率；trustworthiness只表示邻域保真。

受控子集示例：

```bash
python imgo2_rl/scripts/tools/analyze_latent_tsne.py --report <run>/report.json --phase tow --filter velocity_mps=1.0 --filter cart_mass_kg=10 --perplexities 10 30 --seeds 0 42 --output <run>/tsne_fixed_v_mass
```

支持按case里的其他字段过滤（例如连接长度、轮阻；须使用报告的真实键名）。默认排除显式 `cart_present=false` 的无车样本；旧报告缺此字段时无法凭空判定，需先确认元数据。样本数量不够的perplexity跳过并记录，全部不合法则报错；维数不为16或非有限数的向量排除并计数。有效输入少于5或完全相同则不绘图。`--include-nocart`仅用于明确的无负载对照。

## 局部地形输入

现有report使用station/tow/coast阶段均值，一个点代表一个case的阶段平均；`grade_deg`是车道最大坡度，不是当前位置地形。该模式只能作初步工况展示，不能据此声称局部坡段可识别。

`--samples windows.json`可读取一个列表或 `{"samples": [...]}`。每条记录格式如下：

```json
{"latent": ["此处必须替换为16个实际数值"], "case": {"connection": "rigid", "cart_mass_kg": 10, "grade_deg": 10, "velocity_mps": 1.0, "cart_present": true}, "local_slope_deg": -10, "segment": "down"}
```

使用实际记录的逐步或坡段短窗口z替换示意字符串。工具会另外按局部有符号坡度和坡段绘图；记录生产侧尚未新增，必须先补采样，不可将整段均值伪装成局部数据。窗口以独立case和路程/坡段对齐，避免相邻帧过多造成轨迹团块；对比不同工况应保持采样数量和规则一致。

## 读图与验证边界

t-SNE是非线性可视化，不产生簇标签。多参数图全部保存，避免只挑一张分离最漂亮的图；不同拟合图的轴与全局距离不可直接比较。物理类别辨识需要原16维空间的独立工况划分probe或真正的聚类指标支持。

方法参数参考：[scikit-learn TSNE 官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)。

待完成：归档最新v3包含latent的report.json，运行真实工况图；补局部坡段窗口记录再分析地形。当前本机可读的必要性/AMP envelope报告均不含latent，旧轨迹不能恢复这些向量。这轮不启动新仿真或训练，不提供虚构的实验散点。

验证：数据契约3项单元检查通过（报告布局/阶段、无车过滤、非法数值和16维校验、局部坡段标签）；60条显式标记合成样本的4组perplexity/seed输出PNG/PDF/CSV完整，固定seed重复CSV逐字一致；短窗口输入与固定质量筛选实跑通过。图片版式已检查。真实AMP报告因0条latent明确退出，未生成实验图。依赖只安装在本机工作区tmp/tsne_deps中，不改系统Python；绘图测试不代表towing辨识结论。
