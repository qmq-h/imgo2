# MJCF按Gazebo模型同步与验证（2026-10-23）

> 后续更新：逐物理步PD和反馈延迟/IMU噪声已实现并完成编译、组件测试与GUI短流程；本文件保留当时排查。当前设置与限制见[实现记录](mujoco_pd_feedback_2026-10-23.md)。

> 后续发现：用户报告生产GUI启动抽搐，已复现Passive采样PD时序敏感性；固定目标诊断通过不能代替启动稳定。当前控制待修，见[专项排查](mujoco_passive_timing_2026-10-23.md)。

## 当前结果

已按用户授权修改现有imgo2_description/mjcf/imgo2.xml，不新建第二套机器人模型。基准为ROS2 gazebo.launch.py默认加载的imgo2_real.gazebo.urdf与earth.world；未覆盖IMGO2_GAZEBO_URDF时适用。关节、惯性及碰撞静态门禁通过，MuJoCo3.2.7加载/运动学及限定固定站姿诊断通过。PPO步态、生产FSM时序和跨引擎接触等价仍未验证。

原模型差异见[修改前核对](mjcf_real_alignment_2026-10-23.md)，本次结果为[结构/惯性JSON](mjcf_gazebo_alignment_2026-10-23.json)、[运行诊断JSON](mjcf_gazebo_runtime_2026-10-23.json)。历史记录中的8body未同步结论已由本次修复取代。

## 修改内容及原因

| 内容 | 修改前 | 现在 |
|---|---|---|
| 4腿HIP惯性 | 每个.66982842kg、旧COM/张量 | 每个.66kg、逐具名link完整新COM/张量 |
| 4腿THIGH惯性 | 每个.96995719kg、旧COM/张量 | 每个1.044kg、逐具名link完整新COM/张量 |
| IMU惯性 | 只含无质量site | 增加固定base_imu body .01kg、对角惯量1e-6；传感器site位置不变 |
| 总质量 | 12.69955228kg | 12.96640984kg，与Gazebo含IMU一致 |
| joint damping/armature | 1/.1 | 0/0，与Gazebo URDF无额外被动joint dynamics的基线对应 |
| 物理步长 | .005s | .0005s，取earth.world的max_step_size，名义物理2000Hz |
| 滑动摩擦 | 非足1、足.4，接触可能被地面较大系数覆盖 | 非足.2、足.6，取Gazebo mu1=mu2；机器人geom priority1使其接触参数优先于地面 |

纯imgo2_real.urdf总重12.95640984kg，Gazebo版多.01kg是明确的IMU质量，不把它补进base或偷偷改base惯量。新增base_imu没有运动自由度/碰撞geom，关节与传感器顺序不变。当前3.2.7模型nq19/nv18/nu12，sensordata的quaternion/gyro读取位置保持36/40。

保持原轴方向、限位、16个机器人连接位姿与33个碰撞体的形状/局部位姿/尺寸。新增固定IMU后Gazebo共17连接。没有修改URDF/Xacro、Gazebo配置、real硬件/宏、策略、PD增益、参考FSM或控制线程代码。普通场景和两个拖曳场景共享该机器人，因此三个场景均受本次参数变更影响并做了诊断。

## 仍不等价的内容与边界

- MuJoCo cone/impratio、solref=.005/1与接触求解仍沿用现有基线。Gazebo接触kp/kd不能直接照搬为相同数值的MuJoCo solref；没有宣称完成接触刚度/耗散标定。
- robot priority1同时参与接触参数选择，保留各geom原condim/solref及旋转/滚动摩擦字段；不能把这些说成Gazebo ODE的完整映射。实际足地滑动系数.6已通过runtime contact读取确认。
- 保留当前机器人自碰撞mask策略；Gazebo部分link声明self_collide，存在差异，不在本轮不加区分地打开所有自碰撞。拖曳场景与小车的接触混合也受机器人priority影响，虽能加载/固定保持，牵引运动未验证。
- 物理步长变为.5ms，不代表C++的PD自动变为2000Hz：生产RobotControl仍名义200Hz、策略50Hz；Gazebo内部关节控制与MuJoCo物理/墙钟调度仍不同。没有改调度器或接入反馈延迟。
- motor仍是输出侧±23.7Nm的普通力矩执行器；没有复制训练DCMotor的完整速度-力矩包络/真实驱动滤波，不新增虚构电机参数。
- 初始高度/FSM、噪声、IMU数据交付、并发快照与队列缺口尚在，详见[部署复审](mujoco_sim2sim_review_2026-10-23.md)。物理对齐不是实机抖动根因确认。

## 可复现的同步与检查

新增同步器仅针对当前显式Imgo2布局，不是通用模型转换器。默认预览差异；显式--write才覆盖，重复执行无变化。惯性带非零旋转、非零IMU安装位姿、各向异性mu等不支持情形会拒绝，需要人工转换审核。

```bash
# 预览；完成本次同步后应显示already synchronized
/usr/bin/python3 imgo2_rl/scripts/tools/sync_mjcf_from_gazebo.py
# 以后源模型参数改变，明确需要同步时使用
/usr/bin/python3 imgo2_rl/scripts/tools/sync_mjcf_from_gazebo.py --write
# 静态对照、回归
/usr/bin/python3 imgo2_rl/scripts/tools/check_mjcf_current_robot.py \
  --urdf imgo2_description/urdf/imgo2_real.gazebo.urdf
/usr/bin/python3 imgo2_rl/tests/test_mjcf_current_robot.py
# 有本项目MuJoCo3.2.7开发库和c++编译器时，独立无界面固定目标诊断
/usr/bin/python3 imgo2_rl/scripts/tools/check_mjcf_runtime.py \
  --output /tmp/mjcf_runtime.json
```

同步器源/目标/world可用--source/--target/--world覆盖；改变源后仍需运行检查器核对几何/关节，不能把同步器当任意URDF全量转换器。runtime检查器不运行PPO/真实FSM，Python仅用标准库，临时编译小型C++诊断器链接实际MuJoCo并自动清理临时编译目录。

## 实际验证结果

- 静态对照passed=true、18个body惯性一致、17连接/12关节/33碰撞无差异；10/10回归通过，包括惯性/被动参数/摩擦优先级和同步幂等性。
- MuJoCo3.2.7在21姿态下与Gazebo URDF独立FK对照18body，最大位置误差1.11e-16m、旋转矩阵元素误差6.66e-16。
- 三场景分别做60/2和25/.5各10秒PD固定站姿，共6次保持；物理2000Hz，诊断PD按5ms更新，所有状态有限、warning0，均保有足地接触。普通场景末1秒base高度：60/2约.28779–.28782m，25/.5约.23856–.23861m；对应最大关节静差约.108/.372rad。这里无策略补偿，低增益受载下沉不等于PPO站姿或数值不稳定；也不能以宽松模型有效性标准代替关节跟踪性能验收。
- 第一版诊断曾因自设.3rad跟踪阈值在25/.5时报失败；观测到的是稳定受载静差而非崩溃/NaN。随后将角度静差作为输出指标，不再用该任意阈值判断模型合法；仍检查有限值/警告/支撑高度/接触。没有为通过检查而调模型质量或增益。
- 补做参考Passive参数0/8、初始零关节位的2秒独立诊断：无数值警告，但落地阶段峰值关节/自由度速度约80.9rad/s，不能写成启动流程已安全/平稳，实际GUI FSM启动待确认。该诊断没有执行生产FSM。
- towing静态契约和asset paths通过；check_model_sync仍因既有网格指纹漂移和4份未登记real/real_wrong URDF失败，本次未删除文件或擅改登记标准。
- py_compile/diff检查通过。应用C++未改，不需要为XML重新编译；以上没有运行PPO步态、硬件或长期后台任务。

## 使用

关闭原仿真进程后重新启动，使它完整加载新模型；不依赖现有GUI热重载路径更新应用中缓存的mjModel/mjData指针。

```bash
cd /home/tommy0929/Documents/imgo2/imgo2_deploy/ros2_ws
./cmake_build_mujoco/bin/rl_sim_mujoco imgo2 scene
```

先观察Passive落稳，终端0起身完成后1进入PPO，状态键无需回车。先零速度，再单次小速度；Enter为暂停。更改模型后旧的步态结果不再代表当前版本，下一步需记录当前模型hash及PPO实际运行结果。
