# Gazebo AMP 裁剪修正与 Isaac 步态对照

日期：2026-10-13。关联AMP-20/21/22、DEPLOY-14。本文接续 [转换复核](gazebo_conversion_review_2026-10-13.md)，记录实际代码修正、验证和未修项。

## 1. 已修复的内容

当前Gazebo AMP配方启用`gazebo_clip_joint_targets: true`：

- `RL_Sim::Forward()`返回原始网络动作，下一步last_action保留raw，仍按观测clip_obs进行观察裁剪。
- `RunModel()`先使用原缩放/默认角转换，再用±3 rad裁剪目标；诊断力矩按实际发送目标重算。控制器真正应用的PD力矩仍由controller计算。
- 开关只由rl_sim读取，并要求active config_name为amp。`ReadYaml`合并键会留下上个策略的字段，故用活动名称同时约束，避免切到PPO/HIM后残留开关生效。
- 未修改共享SDK、MuJoCo、真机、模型、policy文件、训练奖励或推理线程周期；站立→AMP渐变仍保持0关闭。MuJoCo/硬件的裁剪约定不在本轮修复范围内。

改动：`imgo2_deploy/src/imgo2_deploy/src/rl_sim.cpp`、同包`include/policy_action_clip.hpp`/`rl_sim.hpp`与`imgo2_deploy/policy/imgo2/amp/config.yaml`。当前本机rl_sim已重新编译，install二进制是build产物软链，重启控制进程即可使用；无需重新导出策略。

## 2. 数值与运行验证

- C++回归覆盖scale/offset之后裁剪、raw保持、正负极值、无效边界/维数，以及AMP键残留时其它策略不启用；编译运行通过。rl_sim构建通过。
- 足底几何/FK、方向/惯量审计与周期判据共11项Python回归通过；新增工具语法、本地文档产物链接与git diff检查通过。
- 当前policy仍为motion_new_resume/model_25000官方导出，SHA256 `e40d81ec219c777a4aa88acffed935f12be526a2b8a52d7d56d38f3c9b7cc9d1`，文件未改。
- Gazebo四个方向各全新栈，real URDF、gui=false、软件GL，26墙钟秒，6秒起身，入AMP后跳过4秒，按仿真时间统计。所有本轮启动的进程已结束。
- 增加可选`IMGO2_POLICY_TRACE`：记录实际45维actor输入、12维raw输出及发送目标。四次共4182策略帧，使用同一TorchScript在CPU回放：actor差≤2.15e−6（不同torch/BLAS浮点路径），目标转换差0，逐帧last_action差0，均通过。真实raw超过±3的帧占比约：前进1.92%、后退0.096%、左转9.88%、右转1.62%；这是全策略窗口含切换的帧比例，不等于稳定窗口的关节边界比例。说明旧裁剪确会触发，但不能推断它解释全部运动问题。
- 仿真时间/墙钟比例约0.999；实际actor更新约48.2–48.4 Hz，名义50 Hz的墙钟线程还有计算开销。未发现仿真加速两倍引起步频翻倍；该整体速率不证明每帧反馈延迟已对齐。

[运行轨迹回放契约](gazebo_amp_target_clip_contract_2026-10-13.json)、[修正后的方向指标](gazebo_amp_motion_new_25000_target_clip_2026-10-13.json)、[足底几何高度](../amp/data/amp_motion_new_25000_foot_clearance_target_clip_2026-10-13.json)。原始telemetry/trace在本次run的`gazebo_audit_2026-10-13_target_clip/`，logs不入库。

## 3. 对照结果：修正没有解决贴地步态

同一actor在原Isaac再跑8环境900步（20 ms/步），跳过200步；新增原生关节时序，与Gazebo使用同一个大腿自相关函数。该次结果与前次原Isaac方向/接触结果一致，均无reset。两个引擎并非完全相同接触/执行器条件，不对数值差做单因归因。

足底高度为平地最低碰撞表面距离的95分位数；不是每步峰值均值，也不是FOOT link原点高度。

| 命令 | Isaac实际速度 | 修正后Gazebo实际速度 | Isaac足底高度p95 | Gazebo足底高度p95 |
|---|---|---|---|---|
| vx=+0.4 m/s | +0.251 m/s | +0.384 m/s | 约10–19 mm | 约12–27 mm |
| vx=−0.4 m/s | −0.196 m/s | −0.218 m/s | 前脚2.6/0.9 mm，后脚约0 | 约0.04–1.1 mm |
| yaw=+0.5 rad/s | +0.485 rad/s | +0.460 rad/s | 约0–6.2 mm | 约0–6.0 mm |
| yaw=−0.5 rad/s | −0.466 rad/s | −0.484 rad/s | 约0–5.0 mm | 约0.1–9.4 mm |

相较修正前：Gazebo前进.384→.384、后退−.220→−.218、左转.464→.460、右转−.479→−.484，没有明显的整体改善。单次前后测试不足以证明这些微小差值是修正的因果影响。

前进0.4：Gazebo大腿周期0.29–0.30 s，score约.60–.63；Isaac候选峰0.30–0.32 s，score只有.17–.27，节律较弱。两边都存在快、低的运动。参考数据的足端高度周期约0.6 s且抬高约7–8 cm，口径为参考几何轨迹。

后退/纯转向：腿间周期不同或无稳定峰，足端长期贴地。Gazebo后退大腿约.33 s的运动不能解释成.33 s离地迈步；右转个别关节的7.69 Hz候选更不能当整机步频。原Isaac后退后脚约98%接触、已完成腾空约20 ms，左转RL/右转RR也约98%接触。**Gazebo测量反映实际模拟行为，但不是“与训练完全相同的目标步态”；也不是参考动作已经复现。**

[双仿真同口径比较](../amp/data/amp_gazebo_isaac_direction_comparison_2026-10-13.json)、[原Isaac足端/接触结果](../amp/data/isaac_amp_motion_new_25000_feet_joint_2026-10-13.json)。完整抬高/接触口径沿用 [足端记录](../amp/amp_foot_lift_review_2026-10-13.md)。

## 4. 为什么前进看着好、后退/旋转差

已确认的事实：

1. 前进.4的Gazebo速度误差较小，但步高、周期及交替关系仍偏离参考；“速度跟得上”不等于步态已经合格。
2. 后退两边都欠跟踪、后脚长期接触；纯转向角速度其实能跟到目标附近，差的是足端离地与接触节律。因此不能用一个平均速度tracking值概括这些问题。
3. 保存的任务只有六项：速度tracking、身体高度、身体竖向速度/横滚俯仰角速度、关节限位。air_time、feet_height、feet_slide均None，没有直接要求抬脚/防滑/目标周期。速度tracking标准差.5：后退均值误差约.2 m/s代入理想核仍约exp(−(.2/.5)^2)=.85；该例不是实测分项均值，但说明欠跟踪仍可能获得较高分。
4. 保存AMP混合为.3任务+.7风格（AMP coef=2）；比例是公式系数，不代表实际奖励贡献恰好30%/70%。判别器观察局部状态对，没有显式接触计时或目标步态周期。
5. 21段数据都是600帧/.02 s，MotionWeight均1；前后直行各四档，左右纯转各两档。位置差分旋转到身体坐标后，与保存线速度RMSE约0–.011 m/s；后退vx约−.289/−.583/−.893/−.979，左右纯yaw约±.5/±1。没有发现后退速度符号错、线速度世界/身体坐标混淆或只采了前进。参考的后退/转向确有抬脚。

由这些事实支持的解释是：当前优化结果允许低抬脚/滑动也完成部分速度目标，方向间学习品质不同；原仿真已存在，不由旧裁剪单独造成。不能只用“AMP权重太大”或“某一方向数据坏”作为已证明根因；仍需有接触指标的奖励消融。

[新数据方向/坐标核对](../amp/data/amp_motion_new_direction_data_2026-10-13.json)。该核对使用文件内FrameDuration，不能验证采集机器原始ROS时间戳或所有metadata。

## 5. 下一步应该做什么

- 当前裁剪修正已生效；保持同一policy、PD与模型重启Gazebo/rl_sim查看。无须先重导出、换惯量或降低推理频率。
- 训练下一步针对AMP-22：设计摆动阶段离地高度（可先以足底4–6 cm为实验目标）、支撑脚相对地面滑动惩罚，以及目标周期.5–.6 s的接触节律约束。这里是待验证起点，不是已验证最佳参数；静止/纯转向需要正确门控。
- 仅提高普通奖励的混合系数不会自动增加抬脚：当前普通项没有这些约束，降低模仿可能让滑动更自由。先补充并验证约束，再对混合系数做单变量对照。
- 现有feet_slide先减去base速度，不能直接当正确的地面防滑奖励启用；现有feet_height的target指FOOT原点，须换算足底几何；旧air_time阈值.3不是目标周期.6。详见足端记录，避免仅加大旧滞空权重。
- 新建单变量训练实验，保留现在25000作为基线；比较前/后/左右转/组合指令下每脚高度、接触比例、滞空、速度误差和摔倒率。未授权或启动长期训练，也没有改训练奖励。

## 6. 仍然存在的部署差异

ROS2 controller在PD之前还调用PositionLimit，将收到的q目标裁剪到URDF各关节限位。Isaac JointAction只有配置的±3目标裁剪，物理关节限位在引擎里约束实际状态；两者不完全同义。此次4182帧×12通道中，前/后无额外裁剪，左转1个/右转2个RR髋目标越界，未禁用这一控制器约束。更广命令与瞬态仍待核对。

DCMotor速度相关力矩衰减、PD/反馈速率、差分速度、接触参数和自碰撞仍不同。切换抖动DEPLOY-14仍未修好。本轮只将**Gazebo AMP策略输出裁剪/last_action**标为已修复并验证，不把整个sim2sim接口或步态品质标为完成。

## 7. 复现与构建

本机当前构建目录已配置，实际使用：

```bash
cmake --build imgo2_deploy/build/imgo2_deploy --target rl_sim -j 2
c++ -std=c++17 -O2 imgo2_deploy/tests/test_policy_action_clip.cpp \
  -o /tmp/imgo2_policy_action_clip_test
/tmp/imgo2_policy_action_clip_test
```

原Isaac测量入口`eval_actor_directions.py --feet --linear_speed=0.4`增加了原始joint_positions；跨引擎离线比较使用`compare_gait_engines.py`；记录回放使用`imgo2_deploy/scripts/check_policy_trace.py`（在有torch/yaml的Isaac Python里运行）。脚本参数由`--help`查询，数据路径见上方JSON。

持久化的全新栈测试入口（同一当前部署policy，输出路径不选择checkpoint）：

```bash
python3 imgo2_deploy/scripts/run_gazebo_direction_audit.py \
  --output-root /tmp/imgo2_gazebo_direction_audit
```

本轮四次测试由同逻辑临时驱动实际执行；持久化驱动的CLI/语法已检查，未为验证它再重复启动全部栈。它使用ROS_DOMAIN_ID59/master11369，每次只结束自己启动的进程组；使用前确保该独立端口未被占用。GUI、长时间、多种子与真机不在本轮完成范围。
