# 小车拖曳 MuJoCo sim2sim 场景（2026-09-24）

## 现象与范围

`main` 原有 `imgo2_description/mjcf/scene.xml` 只有机器人和地面，不能回放 Isaac Lab 的拉小车工况。用户确定小车总质量为 **5–15 kg**。本轮先补 MuJoCo 物理场景，沿用现有机器人 FSM／AMP 底层控制入口；上层 recurrent 策略没有已验证 checkpoint，其导出和双 GRU 推理不在本轮假定为完成。

## 修法

- `mjcf/cart.xml` 按 `cart/cart.urdf` 建立 8.4 kg 车体、4 个各 0.4 kg 的轮子、0.5×0.32×0.1 m 车斗、0.08 m 轮半径与前挂点。轮轴黏性阻尼为 `0.016 N·m·s/rad`。车斗及轮子与机器人／地面的接触位均启用。
- 在机器人 `base` 加后挂点 `(-0.16,0,0)`，未更改已有惯量、关节、执行器和传感器顺序。两场景共用机器人和小车模型；`scene_tow_compliant.xml` 使用 MuJoCo 单边绳长约束的直接刚度／阻尼参数 `4000 N/m`、`100 N·s/m`，`scene_tow_inextensible.xml` 使用更硬的单边上限约束。两者绳长上限均为 0.8 m。默认两挂点相距 0.4 m，故初始松弛约 0.4 m。
- `rl_sim_mujoco imgo2 scene_tow_{compliant,inextensible} --cart-mass M` 接受 `5≤M≤15`。名义模型 10 kg；启动后将车体与四轮质量及主惯量同时乘 `M/10`，调用 `mj_setConst` 更新派生量。MuJoCo 文档确认此类运行时修改需调用 `mj_setConst`：[simulation API](https://mujoco.readthedocs.io/en/latest/programming/simulation.html)。
- 新增标准库工具 `imgo2_rl/scripts/tools/check_towing_mjcf.py`，检查 URDF 与 MJCF 的质量、惯量、轮轴位置／方向、挂点和场景绳长。MJCF 单边 tendon limit 用 `solreflimit`／`solimplimit`，见 [MuJoCo XML reference](https://mujoco.readthedocs.io/en/latest/XMLreference.html)。

## 验证

本机解释器：`C:\Users\qmq\AppData\Local\Programs\Python\Python311\python.exe`（Python 3.11）。

- `check_towing_mjcf.py`：通过。
- `python -m compileall -q imgo2_rl/scripts/tools/check_towing_mjcf.py`：通过。
- `git diff --check`：通过；`git ls-files -i -c --exclude-standard` 无已跟踪且被忽略的文件。
- 本机未安装 Python MuJoCo，部署 `library/mujoco` 与 libtorch 依赖也不在仓库内；**未验证 MJCF 编译加载、C++ 构建或拖曳运行**。静态检查不等于 sim2sim 通过。

## 待确认与完成条件

1. 在具备 MuJoCo／libtorch 的 Linux 环境构建 `bash imgo2_deploy/build.sh -mj`，分别加载两个场景；检查 12 个机器人 actuator 与原传感器索引不变，5／10／15 kg 的小车总质量分别正确。
2. 用已验证底层策略做牵引、置零和追尾回放，记录机器人／小车速度、绳长、张力、轮速、最小间隙及碰撞。MuJoCo 的 limit solver 与 Isaac Lab 的显式弹簧力／不可伸长冲量并非同一离散算法；应以实测瞬态调参，不能因参数同名宣称结果一致。
3. 上层训练在 TOW-03 验证后，按项目约定从 `play.py` 导出策略，再接 51 维帧、decoder 与 actor 双 GRU hidden state、20 Hz 上层／50 Hz 底层控制。Gazebo 小车和绳场景仍需另行接入。只有 MuJoCo／Gazebo 完整 episode 与训练侧关键指标对照后才能关闭 TOW-05。
