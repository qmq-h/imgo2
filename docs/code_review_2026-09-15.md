# 代码复审记录（2026-09-15）

范围：整仓库。方法：两个并行只读复核（`imgo2_rl/` 训练栈、`imgo2_deploy/` 与 `imgo2_description/` 部署栈）
加本人全仓扫描。**复核报告中的每一条结论都在源码里逐条核实后才采纳**，与源码不符的已剔除（见第 6 节）。
本机没有 Isaac Lab / torch / ROS / 编译器，因此没有任何一项经过实际运行，凡未运行的一律标注。

---

## 1. 阻断训练的两个缺陷（已修）

### 1.1 AMP 任务无法创建

- **现象**：导入 `base_move/amp_env_cfg.py` 抛 `AttributeError: module '...velocity.mdp' has no
  attribute 'reset_amp_reference_state'`。类体在导入期求值，因此 `Imgo2-basemove-flat-amp`
  与 `-play` 两个任务**根本无法实例化**，包括本文档此前推荐的那条训练命令。
- **原因**：`amp_env_cfg.py` 用 `mdp.reset_amp_reference_state`，而 `mdp/__init__.py:6` 的
  `from .amp_events import *` 是注释掉的；全仓 grep 只有三处命中——定义文件、那行注释、这处使用，
  **没有任何模块导入 `amp_events`**。
- **修法**：在 `amp_env_cfg.py` 内直接 `from ...mdp import amp_events as mdp_amp`，改用
  `mdp_amp.reset_amp_reference_state`。没有把它加回 `mdp/__init__.py`：`amp_events` 会拉入
  `rl_lab → torch → pybullet_utils`，那很可能正是它当初被注释的原因，放回 `__init__` 会让所有任务的
  注册都被拖累。
- **守卫**：`tests/test_amp_alignment.py::test_amp_reference_reset_term_is_imported_and_defined`
  （静态断言：函数存在、配置直接导入该模块、不再经由 `mdp` 引用）。
  负向测试：把 `func=` 改回 `mdp.` 后该测试失败；恢复后 sha256 不变。

### 1.2 `base_height_l2` 会让环境构造失败

- **现象**：环境构造时抛 `ValueError: ... 'base_height_l2:asset_cfg'. Not all regular expressions are matched!`
- **原因**：`RewardsCfg` 给的 `asset_cfg` 是 `SceneEntityCfg("robot", body_names="")`
  （`velocity_env_cfg.py:384`）。`rough_env_cfg.py:110` 与 `himloco_env_cfg.py:183` 都会把它改成具体
  body，AMP 配置没有。Isaac Lab 的行为（在本地 checkout 核对）：`scene_entity_cfg.py:227-254` 的
  `_resolve_body_names` 把非 `None` 的 `""` 转成 `[""]`，再经 `find_bodies → utils/string.py:178`
  的 `resolve_matching_names`，`re.fullmatch("", "base")` 匹配不到任何 body，于是在 `string.py:268` 抛错。
- **影响面**：奖励函数本身只读 `root_pos_w`（`rewards.py:660`），所以这是**解析期崩溃**，与奖励数值无关；
  离线测试只单独执行 `_keep_only_amp_task_rewards`，因此测不到。
- **修法**：在 `_keep_only_amp_task_rewards` 内设 `asset_cfg.body_names = [self.base_link_name]`。
- **守卫**：奖励契约测试扩展为传入 `asset_cfg` 并断言 `body_names == ["base"]`。
  负向测试：删掉该行后测试失败 3 项；恢复后 sha256 不变。

### 1.3 由此得出的一条推论（重要）

上述两处意味着**当前这棵树里的 AMP 任务此前不可能运行成功**。而用户反馈「AMP 已有步态但贴地爬行」，
说明那次训练用的是**另一个代码版本**（本地 `__pycache__/mdp/__init__` 的 cpython-314 变体编译自
「导入未注释」的源码，支持这一判断），或跑在服务器那份代码上。

**因此下一次短训练不是「重新验证」，而是这条代码路径的首次真正执行。** 相关前置检查见
[amp_alignment_review.md](amp_alignment_review.md) 第 7.1 节。

---

## 2. 工具与检查脚本的缺陷（已修）

| 项 | 问题 | 修法 / 验证 |
|---|---|---|
| `check_asset_paths.py` | 把**期望的相对路径硬编码在脚本里**，因此即使 `assets/imgo2.py` 声明的路径写错，检查也照样通过——恰好是目录改名最容易引入的错误类型 | 改为从源码读出 `_DEFAULT_URDF_PATH` / `_DEFAULT_MOTION_DIR` 的字面成分再解析；新增校验该 URDF 引用的 **17 个网格**是否都存在。负向测试：改错 `imgo2.py` 的路径后检查 FAIL（此前会通过），恢复后 sha256 不变 |
| `check_model_sync.py` | 只覆盖 `URDFS` 里写死的 4 份 URDF，**新增一份分歧副本会被完全忽略**——正是本仓库此前出问题的模式 | 新增第 6 项：用 `git ls-files *.urdf` 枚举全仓，未登记的一律 FAIL。负向测试：加入第 6 份副本后报错，清理后恢复通过 |
| `inertia_urdf.py` | 指向 `imgo2_model/imgo2_urdf/urdf/imgo2.urdf`，那是腿部件片段（无 `base`、无 `<robot>`，XML 都不能独立解析），pinocchio 必然在 `buildModelFromUrdf` 失败 | 改为指向完整训练模型；已核对目标存在、可解析、12 个 revolute 关节（nq = 19，与脚本 `q[7:19]` 一致）。**未用 pinocchio 实跑**（本机未安装） |
| `tests/test_amp_alignment.py` | 两处 `read_text("utf-8")` 之后 `ast.parse`，遇到带 UTF-8 BOM 的源文件抛 `invalid non-printable character U+FEFF` | 全仓 **27 个**上游文件带 BOM（Python 导入不受影响，文本回读再解析会中招），两处改为 `utf-8-sig` |
| `.gitignore` | 上一轮合并两个嵌套文件时漏掉了部署子树的 `*.json` 规则。**当时的等价性核对发现不了它**——那次只验证了「已存在的被忽略路径仍然被忽略」，而当时 deploy 下没有 `.json` 文件 | 已补回 `/imgo2_deploy/**/*.json`；探针确认该规则重新生效 |
| `README.md` / `amp_alignment_review.md` | 目录改名时把文档里**引用的历史路径** `/root/gpufree-data/Imgo2_rl/...` 一并改成了 `imgo2_rl`，属于改写史实（当时目录名确实是大写 I） | 已恢复原样并注明当时的目录名 |

---

## 3. 部署侧缺陷（已修，**未编译验证**）

### 3.1 ROS2 两版控制器的关节限位完全失效

- `ros2/src/robot_joint_controller.cpp:215-228` 与 `..._group.cpp:288-301` 里
  `std::clamp(x, lo, hi);` 是**丢弃返回值的语句**——`std::clamp` 不原地修改，因此位置/速度/力矩
  三个限位全部无效，共 6 处。
- 对比：ROS1 在 `ros/src/robot_joint_controller.cpp:11` 自己定义了**按引用修改**的
  `double clamp(double &value, ...)`，所以 ROS1 是生效的。这也解释了为什么这个缺陷此前没被发现。
- 修法：改为赋值形式；同时加 shared_ptr 空值与索引越界保护（`joints_urdf_` 由异步参数回调填充，
  回调未完成或失败时会为空指针），并在两个文件补 `#include <algorithm>`。

### 3.2 单关节版请求了错误的参数名

- `ros2/src/robot_joint_controller.cpp:45` 请求 `"robot_description_"`（多一个下划线），
  而 robot_state_publisher 发布的是 `"robot_description"`；group 版（`:74`）是正确的。
- 后果：URDF 从未解析成功，`joints_urdf_` 保持空指针，第一次调用限位函数即空指针解引用。
- 已修正为 `"robot_description"`。

### 3.3 其他两处（已修）

- `src/imgo2_deploy/scripts/actuator_net.py:15` 的 `BASE_PATH` 用 `"../"`，从 `scripts/` 只上溯一层
  落到 `src/imgo2_deploy/`，而调用处是 `join(BASE_PATH, "policy", ...)`，该目录下没有 `policy/`。
  已改为 `"../../../"`（= `imgo2_deploy/`），并核对 `policy/` 确实存在。
- `build.sh` 里 `cmake src/imgo2_deploy/`、`find src`、`rm -rf build/ …` 都相对当前工作目录，
  而脚本只算了 `SCRIPT_DIR` 没有 `cd`；从仓库根执行 `bash imgo2_deploy/build.sh` 会失败。
  已在 `source common.sh` 之前加 `cd "${SCRIPT_DIR}"`。**本机受限 shell 下 bash 无法创建信号管道，
  此行未做 `bash -n` 语法校验。**

**待办**：以上四处需在装有 ROS 2 的 Linux 上编译并跑通（DEPLOY-07）。

---

## 4. 已确认但尚未修的问题

### 4.1 构建与链路（见 README 问题表）

- **DEPLOY-04**：`imgo2_deploy/src/imgo2_deploy/library/thirdparty/joystick/` 目录存在但**为空且未跟踪**，
  而 `CMakeLists.txt` 在 `USE_MUJOCO` 下要求 `.../joystick/joystick.cc`、include 该目录，
  `rl_sim_mujoco.hpp:30` 还 `#include "joystick.hh"`；没有任何脚本会下载它 →
  `build.sh --mujoco` 在配置阶段就失败。
- **DEPLOY-05**：部署份 URDF 没有 `<transmission>`，也没有 `gazebo_ros_control` / `gz_ros2_control`
  的 `<gazebo><plugin>`，两个 world 里同样没有插件 → `rl_sim.cpp` 启动的 controller spawner
  没有 controller_manager / EffortJointInterface 可加载，`robot->getHandle()` 无法工作。
  可用的 transmission 与插件只在 `imgo2_description`，而它的插件库（`liblegged_hw_sim.so`）不在本仓库。
- **DEPLOY-06**：`rl_sim_mujoco.cpp` 用硬件侧的 `joint_mapping` 直接索引 MJCF，但现成的
  `imgo2_description/mjcf/imgo2.xml` 的关节/执行器顺序已是策略顺序 FL,FR,RL,RR，应当用恒等映射；
  一套 `joint_mapping` 无法同时服务两种索引空间。

### 4.2 需要决定或补充的项

| 项 | 现状 | 需要确定什么 |
|---|---|---|
| `imgo2_description/mjcf/` 复用（DEPLOY-02 线索） | `imgo2.xml` + `scene.xml` 自洽（10 个网格都存在；传感器顺序与 `rl_sim_mujoco.cpp:158-165` 完全对应） | 但它的基座惯量仍是 **6.53394** 那套、足端只有 sphere 碰撞、关节顺序为策略顺序。复用时必须改成规范值（质量 `5.53394020` + 惯量 `0.03866860/0.10411461/0.12554111`）并处理上面 DEPLOY-06 的映射问题 |
| `imgo2_description` 的安装规则 | `CMakeLists.txt:21-24` 只安装 `meshes urdf launch`，而 `launch/empty_world.launch:4,8` 需要 `$(find imgo2_description)/xacro/robot.xacro` | 决定是补装 `xacro/` 与 `mjcf/`，还是把 launch 改为不依赖 xacro（catkin devel 空间下因 `$(find)` 指向源码目录而恰好能用，install 空间下会坏） |
| `imgo2_description` 的外部依赖 | `launch/empty_world.launch` 依赖 `legged_common`、`legged_gazebo`；`gazebo.xacro` 需要 `liblegged_hw_sim.so` | 这些包不在本仓库，该 launch 无法独立运行；决定是否保留、或改为自包含 |
| himloco 配置与训练侧的差异（并入 DEPLOY-01） | `default_dof_pos`：部署 `±0.1 / 0.8~1.0 / −1.5` 对训练 `0 / 0.87 / −1.82`；PD：`rl_kp/kd 40/1` 对训练 actuator `25.0/0.5`；力矩上限 `33.5` **超过** URDF 与训练的 `23.7`；`commands_scale [2.0,2.0,0.25]` 而训练侧指令观测无显式缩放；`clip_actions ±100` 等于不裁剪，训练侧是 `(−3,3)` | 决定以哪一侧为准并统一。观测量顺序、各项缩放、`action_scale`、50 Hz 周期、6 帧历史是**一致**的，无需改 |
| `base.yaml` 的定位 | 它**不能**作为策略配置加载：`InitRL` 需要同目录的 `config.yaml`（不存在），也没有网络文件；`ReadYaml` 只记一条日志后返回空参数，FSM 捕获后落到 Passive。当前只有硬编码的 `imgo2/himloco` 可加载 | 决定它是保留为 RL 之前的模板（现状），还是补成完整策略配置 |
| `assets/amp_motions.py` | 全仓无人引用（只有 `check_asset_paths.py` 把它列为待检查文件），三个候选目录都不存在 → 恒返回空列表 | 决定删除或接入。`check_asset_paths.py` 看不到它的空结果 |
| `extension.toml:5` | `readme = "README.md"` 指向一个从未存在过的文件（与本次删除的子项目 README 无关） | 指向根 README 或删掉该字段 |
| Jetson 流程 | `scripts/download_inference_runtime.sh:74` 判断 `${IS_JETSON}`，但没有任何 shell 脚本设置它（只有 `CMakeLists.txt:105-107`）；`common.sh` 的 `detect_platform()` 从未被调用 → Jetson aarch64 会落进普通 Linux 分支并 `exit 1`，`install_pytorch_jetson.sh` 永远不被使用 | 决定是否修好该分支 |
| `fsm_imgo2.hpp` 的 GetUp/GetDown | `GetUp::Enter` 在映射仍是恒等时快照 `now_state`，`GetDown::Run` 再按置换后的映射写回 → 「起始姿态」会作用到错误的腿上；因该姿态左右对称所以暂未暴露 | 决定是否让快照与映射切换保持一致（属潜在缺陷，非当前故障） |
| 其他小项 | `observation_buffer.cpp:116-123` 与 `:141-148` 混用 term 索引与 time 索引（当前 6/6 恰好等价）；`rl_sim.cpp:191-204` 的 ROS1 spawner 失败不检测（ROS2 版会抛）；`rl_sim.cpp:369-370` 硬编码 Gazebo 模型索引 2；`robot_msgs/CMakeLists.txt` 安装不存在的 `include/robot_msgs/`；`.gitignore` 忽略 `*.onnx` 而 `convert_policy.py` 会导出 ONNX | 均为低影响，逐条决定是否处理 |

---

## 5. 已核实「没问题」的部分（不要重复排查）

- **43 维 AMP 观测**：12 + 12 + 3 + 3 + 12 + 1 = 43；项顺序（joint_pos、foot_pos_base、lin_vel、
  ang_vel、joint_vel、root_z）与 `motion_loader.py:337-349` 的拼接顺序一致。`observation_dim` 的
  `+1` 就是追加的 root_z，数值正确。`Normalizer(43)`、判别器输入 86、`ReplayBuffer(43)` 三者自洽。
- **映射方向**：`_apply_flat_mapping = data[:, mapping]` 表示「输出槽 i 取索引 mapping[i]」，
  即文件槽 → 仿真槽；`amp_events.py` 用逆映射写回仿真顺序，方向正确。恒等映射下两者都退化为恒等。
- **归一化与判别器**：`amp_ppo.py:228-240` 分类与梯度惩罚使用**同一套归一化张量**；`:258-260`
  的运行统计用 `raw_*` 更新。与 `amp_go2-main` 一致。
- **奖励与终止**：`_keep_only_amp_task_rewards` 只保留三个任务项；权重按 `1/step_dt` 补偿后每步为
  `1.0·r_vxy + 0.3·r_wz − 10·(z−0.30)²`，与 style 项（上限 1.4）同量级；`amp_task_reward_lerp=0.3`
  经 `train.py → runner:73` 确实到达判别器；基座触地终止在本配置下**非 None**（`illegal_contact=None`
  只写在 `Imgo2RoughEnvCfg`，AMP 继承的是公共基类）。
- **部署关节映射**：方向为 policy → SDK（`rl_sdk.cpp:119-128`），`InverseJointMapping` 是精确逆；
  逐元素核对 himloco 的 `[3,4,5,0,1,2,9,10,11,6,7,8]`：FL→3,4,5、FR→0,1,2、RL→9,10,11、RR→6,7,8，
  与 base.yaml 的 SDK 顺序 FR,FL,RR,RL 及 Unitree 电机顺序一致。`base.yaml` 的恒等映射只是 RL 之前
  的值，进入 locomotion 时被 himloco 覆盖。
- **部署份 URDF**：与训练份逐行多重集比较，只多 4 个 `<material>` 块，物理等价（与 MODEL-01 的说法一致）。
- **文档与代码一致性**：README 的 8 个任务 ID 与 8 个注册项完全对应；12 个脚本路径存在；
  `build.sh` 的 `--cmake`/`--mujoco` 有效；README 里其余 CLI 参数由 Isaac Lab 的 `AppLauncher` 提供。
- **全仓静态检查**：113 个 Python 文件按字节全部可编译；25 个 XML/URDF/xacro/launch 中仅 1 个解析失败
  （即已知的腿部件片段）；代码与配置中旧目录名残留 0 处；`git ls-files -i -c --exclude-standard` 为空。

---

## 6. 本次核实后剔除的复核误报

1. **「27 个 Python 文件有语法错误」** —— 是我自己的检查方式把 UTF-8 BOM 当普通字符读取所致。
   按字节编译 113/113 全过。
2. **「引用了已删除的 VERSION」** —— 是 `LIBTORCH_VERSION`、`mjVERSION_HEADER` 之类的子串误匹配。
   代码与配置中真正引用已删文件的只有 `.gitignore` 里一句说明性注释。
3. **8 处「未解析的路径引用」** —— 逐条核对为：目录表里的上下文相对路径、`tasks/.../` 形式的省略写法、
   以及维护记录里刻意提到的已删文件。
4. **`reorder_from_pybullet_to_isaac` 可疑** —— 它是有意为之的「同序重排」（对已按 FL,FR,RL,RR 录制的
   数据是恒等操作），不是漏改。
5. **`history_length: 5` 与 `observations_history: [0..5]` 不一致** —— 6 帧 = 历史 5 + 当前，一致。
6. **`imgo2_deploy/library/{inference_runtime,mujoco}` 缺失** —— 由下载脚本创建，路径与 CMake 期望一致。
7. **URDF 里的 `<mujoco>` 块** —— 被 urdfdom / IsaacLab 忽略，ROS 与 RL 路径都不读它。

---

## 7. 本轮的限制

- **未运行任何东西**：没有 Isaac Lab、没有训练或回放、没有 Gazebo/ROS、没有编译器、没有 pinocchio。
- 1.1 与 1.2 的修复由「静态守卫 + 阅读本地 Isaac Lab 源码」确认，**未在真实环境中构造过环境**；
  1.1 的修复还会在加载 AMP 配置时引入 `rl_lab → torch` 的导入，这一点在训练环境之外无法验证。
- 第 3 节的四处 C++ / shell 修复**只做了语义核对与括号平衡检查**，未编译、未 `bash -n`
  （本机受限 shell 下 bash 无法创建信号管道）。
- 复核报告里凡未经我核实的推断（例如 odom twist 的坐标系、判别器归一化混合分布的影响）都未写入本文档。
