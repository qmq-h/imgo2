# MJCF与imgo2_real逐项对齐检查（2026-10-23）

> 更新：本文件保留修改前排查。现有MJCF已按用户授权同步Gazebo参数，8body差异已修；当前验证与未完成项以[同步记录](mjcf_gazebo_sync_2026-10-23.md)为准。

## 结论

基准为用户指定的imgo2_description/urdf/imgo2_real.urdf，目标为imgo2_description/mjcf/imgo2.xml（scene.xml引用）。当前不完全一致：4腿HIP/THIGH共8body的质量、质心、惯量未同步。结构与碰撞几何一致；不同姿态运动学也一致。没有改MJCF、URDF、Xacro、控制参数或策略；扩展了标准库检查脚本并补回归，避免只口头列出差异。

## 明确不同：8个刚体惯性数据

| body组 | 当前MJCF单个质量(kg) | imgo2_real单个质量(kg) | 需要核对/同步的内容 |
|---|---:|---:|---|
| FL/FR/RL/RR_HIP | .66982842 | .66 | 每腿独立质量、3维COM、6个惯量分量 |
| FL/FR/RL/RR_THIGH | .96995719 | 1.044 | 每腿独立质量、3维COM、6个惯量分量 |

总质量MJCF12.69955228kg，imgo2_real12.95640984kg，差.25685756kg，约1.982%。base质量5.53394020及惯性一致；4SHANK、4FOOT质量/COM/惯量一致。不要只把总重补到base，也不要只改mass不改COM和惯量。

例：FL_HIP的COM从旧[.05776150,.01631658,.00001927]变为基准[.05502,.01629,.00008]；FL_THIGH从[-.00429543,.04392351,-.03236426]变为[-.002105,.044398,-.032405]。后腿THIGH旧COM x为+.00429543，基准x为-.002105，不能按“左右/前后自然镜像”推测，应按每个具名link原值读取。HIP的前/后Ixx也非同一数值，必须逐腿拷贝完整张量。完整8body差异及基准值见[JSON](mjcf_real_alignment_2026-10-23.json)。基准中惯性rpy均0，MJCFfullinertia的顺序为Ixx/Iyy/Izz/Ixy/Ixz/Iyz，与URDF字段按名称对应；不能按URDF属性排列顺序直接复制。

本轮确认这些新值也存在于core.xacro，后续修复以用户确认的源保持一致，不另建模型副本。本轮未重新生成或覆盖任何模型。

## 已通过的结构、几何与运行时运动学核对

- 17个URDF link均有同名MJCF body；16个关节连接的parent/child、原点位置/旋转一致（含4个Ankle固定关节）。
- 12个活动关节的轴、角度范围、零位旋转中心一致；MJCF hinge pos为0，与body设在关节原点相容。顺序FL/FR/RL/RR不变。
- 33个碰撞体逐项类型、位置、旋转及尺寸一致。URDF box全尺寸转MJCF半尺寸；cylinder的length转半高，radius不变；sphere半径不变。没有把半尺寸误判成缩小模型。
- 12个motor对准同名joint，gear1、ctrlrange±23.7与URDF effort23.7对应。URDF velocity30.1不是普通MuJoCo motor自动执行的硬速度上限，不计作已实现速度保护。
- 使用工程实际MuJoCo3.2.7，编译临时C++检查器，对站姿加20组限位内确定性姿态运行mj_forward；独立标准库URDF FK计算17body位置和旋转矩阵。21x17项最大位置误差1.11e-16m，旋转矩阵元素误差6.66e-16。仅运动学，不是PPO闭环或接触动态通过。

## URDF一致不代表引擎动力学完全一致

1. MJCF每个活动关节仍有damping1/armature.1，这些是额外被动阻尼/转动惯量；imgo2_real没有对应显式joint dynamics。不是策略PD，不能为了“对齐”未经确认全设0或把它并入Kp/Kd。同步8body后仍需明确该基线。
2. 训练配置使用DCMotorCfg（23.7Nm、30.1rad/s、25/.5），MJCF是普通motor，限幅不等于完整速度-力矩包络或真实电机滤波/延迟模型；URDF本身不足以确定这些。
3. 摩擦、solref/solimp、积分器、接触求解和物理步长是MuJoCo专用设置，无法从纯URDF严格一一等同。当前机器人碰撞mask2/1不产生自碰撞，训练assets配置也明确enabled_self_collisions=False，不能为“更真实”自行打开。
4. 初始base=.5、关节初始0与训练初始z=.35/站姿不同；这是启动/FSM设定，不是腿长/FK错误，修改需结合启动流程。IMU site在base原点；当前PPO用姿态/gyro不直接用accel，不能从本次检查推导真实IMU滤波和延迟已经建模。

## 检查工具与验证

现有imgo2_rl/scripts/tools/check_mjcf_current_robot.py已扩展为惯性+结构/原点/轴/限位/motor/碰撞检查，仍仅Python标准库。显式不支持的惯性旋转/非xyz欧拉/fromto等会拒绝，避免静默比错；不是通用URDF转MJCF转换器。

```bash
/usr/bin/python3 imgo2_rl/scripts/tools/check_mjcf_current_robot.py \
  --json /tmp/mjcf_real_alignment.json
/usr/bin/python3 imgo2_rl/tests/test_mjcf_current_robot.py
```

本轮第一条退出1，准确报告8body惯性不同、结构issues空（预期未通过，不能写成模型验证通过）。第二条6/6通过：当前结构、轴破坏、原点破坏、碰撞尺寸破坏、力矩范围破坏、碰撞顺序不变性。py_compile/diff格式检查通过。JSON记录源模型SHA256、完整差异及运行时FK摘要。

待修复：将8body完整惯性数据按各具名link同步后重跑上述门禁与MuJoCo加载，明确被动动力学基线，再做PPO短程测试；本轮只检查，不擅自修改物理参数。已有模型总检查的网格登记/多份URDF未登记问题仍独立存在，不在本轮顺便删除或改名。
