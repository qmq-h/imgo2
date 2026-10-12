# AMP MPC 采集脚本与是否重采复核（2026-10-11）

## 结论

已找到项目内脚本 `imgo2_rl/scripts/record_legged_control_amp.py`；它是ROS1 rospy采集器，直接写61列AMP JSON，没有独立转换步骤。原来历史Windows采集脚本路径仅作为来源线索，不当作当前存在路径。用户本轮说明数据来自leggedcontrol MPC工程；究竟是MPC Gazebo还是硬件、实际工作区/参数仍待确认，不能继续仅凭历史文档称之为真机数据。

新策略不能笼统判为全部不如原24000：本轮25000相对微调目录24000的关节节律/偏航改善，高度和跟速更差。另核对原 `2026-09-24_00-57-38_xiaoji_real1/model_24000.pt` 与微调目录 `2026-10-11_00-00-01_xiaoji_real1_air_time_finetune/model_24000.pt` 的actor参数最大绝对差0.0098810643；此前真实对比使用后者，而非原文件。两者哈希和actor均不同；没有原文件的本轮同条件测试，不能精确排名原24000与所有新模型。训练奖励混合比例不同，回合奖励也不能直接排序。

## 采集链路与入口

- 脚本第41–42行订阅/odom与/joint_states；第97–105行查base到四足TF。
- 第107–115行按joint_names重排关节与速度；第118–136行写odom根姿态、xyzw四元数、twist。
- 第138–156行差分足端速度、拼61列；第173–185行直接写文件。
- 第190–227行CLI：--out、--rate默认50、--duration、--max-frames、--zero-root-xy、root_z_offset、话题与frame覆盖、可选cmd_vel发布。
- 默认腿名为LF/RF/LH/RH，关节HAA/HFE/KFE；当前Imgo2模型FL/FR/RL/RR及hip/thigh/shank。重采必须根据正在运行的MPC URDF/TF明确名称，不可直接用另一模型命名。默认输出路径是旧作者/home/ak1ra路径，重采必须明确--out。
- 使用零原点时优先--zero-root-xy保留真实根高；--zero-root将z也减掉，会破坏AMP高度含义。
- ROS1采集器不能直接放进ROS2 Humble环境运行；需原MPC工程ROS1依赖。无可用ROS1采集链路时未进行运行验证。

本机发现两份外部MPC候选（只读，未改动）：

1. `/home/tommy0929/Downloads/legged_ctrl/src/legged_control`，launch/load_controller.launch按robot_type取taskFile、referenceFile、gaitCommandFile；现有a1/aliengo/go1，无imgo2配置目录。
2. `/home/tommy0929/Downloads/legctrl_bigdog/src/legged_control`，另有bigdog配置。

legged_ctrl的legged_controllers/config/a1/gait.info第28–36行trot为LF_RH/RF_LH，switchingTimes0/.3/.6，规划周期.6 s；legctrl_bigdog的config/bigdog/gait.info第24–36行trot为0/.4/.8，周期.8 s，standing_trot仍.6 s。它们是本机候选内容，不证明用户录制时使用哪一个gait。实际选用文件应从MPC launch参数与当时步态命令确认。

legged_ctrl估计器LinearKalmanFilter.cpp:getOdomMsg明确将世界线速度逆旋到base，angularVelLocal写入twist；支持此发布器时采集器直接读取twist的body-frame假设。实际发布器另有FromTopicEstimate等路径，不能把注释当作所有来源均已验证。

## 源码可见风险（未证明旧数据已受影响）

1. **无时间同步**：取最新odom、最新JointState、四次Time(0)最新TF，没有同一采样时刻匹配，快步态/消息延迟会让q、dq与足端不一致。
2. **标称固定时间**：写FrameDuration=1/rate，无逐帧实际stamp。TF失败会跳过帧、odom重复或延迟没有新时间戳检查；采样抖动/漏采不能从现有文件恢复。足端差分使用odom stamp而TF来自latest，也有时间域不一致风险。但最后12列足端速度不进入当前43维判别器输入，不能直接据此归因AMP风格失败。
3. **运动起始不筛选**：计时/可选命令发布同时开始，无暖机与稳定窗口条件，短片段可能混入起步和停止。
4. **缺接触/命令元数据**：无法直接从数据求真实滞空或证明每片段执行的命令；应另存同步原始rosbag、接触力/接触标记、所用模型和gait及实际时间信息。元数据不要塞进训练loader扫描的运动文件目录。
5. **缺速度字段时填0**：JointState.velocity缺失会静默用0影响模仿；应采集前验证字段而不是把缺失当静止。
6. **手动结束保存**：当前write_dataset只由duration/max_frames路径调用，Ctrl+C异常处理仅pass，没有可靠的退出保存。重采至少明确duration或max_frames；修订退出保存尚未实施。

这些是当前副本源码的设计风险。历史录制可能用了其他版本/参数；不能据此推断已发生漏帧或解释所有高步频。

## 当前数据离线验证

实际训练默认目录来自assets/imgo2.py为 `imgo2_rl/datasets/imgo2_motion`，可用IMGO2_AMP_MOTION_DIR覆盖。根imgo2_dataset/datasets/imgo2_motion仍有同内容副本；两目录21文件逐字节相同。本轮没有删除或移动数据。

`inspect_amp_collection_data.py`标准库统计21段5097帧、FrameDuration均.02、MotionWeight均1；全部连续帧中完全相同的关节位置/足端位置对为0。这排除了最直接的整块重复帧现象，不等于证明时间同步或无漏采。

actual root speed平均：forward_0.3=.30494、forward_0.6=.47122、forward_0.9=.63902、forward_1.2=.83413 m/s；backward_0.3只有.02932 m/s，近站立。这说明文件名是命令标签，部分片段实际运动不匹配。

motion_loader按每条MotionWeight归一化抽轨迹，不按帧数比例。当前两个近静止片段（stance与backward_0.3）约占2/21=9.52%的轨迹抽样，不是站立片段短就必然低比例。站立数据不是天然错误，但应与训练站立需求匹配。

`audit_amp_dataset.py`对当前real URDF核对全部61列有限值、四元数和FK：有效数据21段；按正确腿序各片段FK坐标RMSE约0–0.002143 m，不支持严重腿序/关节符号错误。[字段统计](amp_collection_dataset_summary_2026-10-11.json)、[real FK统计](amp_collection_dataset_real_fk_2026-10-11.json)。参考步态约.6 s已接近期望，单纯重录相同MPC步态并不保证RL节律变慢。

复现：

```bash
/usr/bin/python3 imgo2_rl/scripts/tools/inspect_amp_collection_data.py
/usr/bin/python3 imgo2_rl/scripts/tools/audit_amp_dataset.py   --motion-dir imgo2_rl/datasets/imgo2_motion   --urdf imgo2_description/urdf/imgo2_real.urdf
```

## 建议与待确认

先确认实际MPC工作区、机器人模型、当前步态文件和数据来自仿真还是硬件，再修同步/时间记录并小批量录0.3/.5/.6 m/s稳定动作，每段至少覆盖多个完整周期（例如20–30 s）；保存同步接触和原始时间信息，按实际速度/周期验收。数据录制本轮没有启动，不发布cmd_vel。较高速与横移/转向覆盖按任务范围逐步补，不要求直接清空旧数据重采。

若小批数据在时间同步、接触时序或命令覆盖方面明显更好，再扩充新的独立数据集并做对照。换数据后判别器/归一化统计是否复用要检查输入尺度与分布；不能直接承诺旧AMP统计继续训练仍合适。

已完成：原24000与微调24000区别澄清、录制入口和本地MPC候选定位、源码风险复审、21段数据有限值/FK/重复帧/实际速度验证。待修复或确认：采集器同步与时间/接触元数据、退出保存、实际录制工作区/参数、原24000的同条件基准、新数据试采和训练收益；需要ROS1 MPC/原始rosbag及后续仿真或录制验证。未改采集器、未重新录制、未启动训练，不能把源码风险记为已修复。

## 原始 xiaoji_real1 24000 同条件补测（已完成）

为补齐前述基准，核对原运行现有正式导出exported/policy.pt与原checkpoint在128组seed1输入上的actor最大差0，临时使用该文件运行同一real Gazebo、同一ROS隔离环境、vx=.5、30墙钟秒/约20稳定仿真秒，未启动训练。结束后停止本轮Gazebo/rl_sim，恢复当前25000默认策略并核对SHA256 bf792f98faf17cdd...。

| 指标 | 原始24000（本次单测） | 25000（此前两测） |
|---|---:|---:|
| 关节周期 | .2825 s | .2700 s |
| 关节频率 | 3.5406 Hz | 3.7037 Hz |
| body vx | .50553 m/s | .58597–.58698 m/s |
| 跟速误差 | 1.106% | 17.19–17.40% |
| yaw漂移 | −.149°/s | −2.08至−2.22°/s |
| 平均高度 | .29128 m | .26297–.26301 m |
| roll/pitch全范围 | 3.765°/2.969° | 5.13–5.84°/4.22° |
| 大腿mean abs ddq | 55.72 rad/s² | 64.19–64.46 rad/s² |
| 小腿mean abs ddq | 73.53 rad/s² | 102.11–103.23 rad/s² |

原始24000单测稳定窗口无倒地、FL/FR相位180°、周期强度.828。相对这个实际原始策略，25000整体跟速/姿态更差且没有降频（周期略短），支持保留原始24000作为下一轮基准。并不能据此说所有未测checkpoint都较差；29000是另一来源，历史同速度body vx .491/周期.29但yaw−5.78，不能统一只看一个奖励排名。原始24000也没有达到.5–.6 s节律目标。

已修正此前笼统“25000有降频收益”的理解：那是对微调运行24000，不是对原始xiaoji_real1。用户问题涉及原始策略，因此以本次补测为准。JSON见 [实测](gazebo_amp_original24000_2026-10-11.json)、[频谱](amp_original24000_gait_spectrum_2026-10-11.json)。原始24000本次只测一个速度一个稳定窗口，未宣称多速度整体验收。

本轮原始基准缺口已补，实际MPC工作区/时钟、采集同步与重采收益仍待确认。采集脚本SHA256 9dde81cb0ae867dd5e660ddf43ebed69da988ca96697c37525973937f0eff7e3；所读a1 gait SHA256 cad4da4ac05edff45f2244feb7b436112f2ed7cf9d456a6650bb120c10e2708f，bigdog gait为2cf0a3369205b23bae9968f3504b3917c380d0bbf5528aa8dfe125351de6d3c5。

## 用户确认采集工程在另一系统（2026-10-11）

用户明确回复“没有在本系统上”。因此本机Downloads/legged_ctrl与legctrl_bigdog均不能作为实际采集源依据，已读gait周期只供参考。项目内record_legged_control_amp.py是可研究的副本，实际录制版本/启动参数需由采集机文件核对。需要用户提供采集时的脚本、实际加载的gait.info与录制/MPC启动命令，随后再核对模型URDF、估计器及原始rosbag。现有数据统计与原始24000本机sim2sim结论不受此确认影响；采集同步问题目前是副本源码风险，不认定实际数据已受影响。未操作远端系统或录制任务。
