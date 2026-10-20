# go_imu 站立 FSM（2026-10-14）

> 2026-10-15结构更新：全部具体状态已归入fsm/states，go_stand与部署共用唯一StandState；原行为与本文历史验证记录保留，当前结构见 [纠正记录](go_imu_state_split_2026-10-15.md#同日结构纠正记录删除双standstate与继承套层)。

> 本文记录当时实现与验证。当前配置类、顶层main和日志删除后的结构见 [五阶段整理记录](go_imu_cleanup_2026-10-14.md)；当时日志/宏/RunXXX描述不再代表当前实现。


## 已完成

参考fsm_imgo2.hpp的状态基类/生命周期组织方式，在go_imu新增fsm目录；不引入RL、宇树SDK或ROS。只实现Stand一个状态，内部Waiting→Rising→Holding三个阶段。原go_stand名称/外层薄入口保持，apps负责设备、日志、退出，FSM负责生成目标。原go、imu_sensor、部署参考FSM和rl_real_imgo2未修改。

| 文件 | 职责 |
|---|---|
| fsm/fsm.h | State基类（name/enter/run/exit/command）、单状态FSM宿主（start/update/stop）、非阻塞行输入解析StandInput |
| fsm/states/stand_state.h | StandSettings、唯一StandState、阶段/进度查询 |
| fsm/states/stand_state.cpp | 参数校验、五次曲线q/dq、等待和保持；站姿由调用方提供 |
| apps/stand_main.cpp | 键盘事件→FSM.update→GoMotorArray.setTargets，保留日志及故障退出 |
| tests/test_stand_fsm.cpp | 人工时钟与默认限位纯离线回归 |

## 行为与配置

start仍执行一次电机健康检查和原启动姿态归零；进入Stand后以25/0.5增益保持启动位置。键盘完整一行恰为0才触发（支持LF和CRLF）；单独0尚未回车、10或其他文本不触发。stdin按字节非阻塞读取，等待期间每轮继续发送保持目标、喂看门狗。

输入后以主机steady_clock实耗时间推进8秒五次曲线：s=10τ³−15τ⁴+6τ⁵，q=start+s×delta，dq=s_dot×delta。起末速度/加速度为0，峰值速度1.875×|delta|/T。四腿目标均为0/0.87/-1.82输出侧逻辑rad，原方向/offset在电机层处理。完成后dq=0、保持站姿。中途/完成后再次0不会重启。

轨迹时长与增益统一修改fsm/states/stand_state.h的StandSettings（duration_s=8、kp=25、kd=0.5、tau=0）；独立工具每腿站姿在stand_main.cpp的buildStandGoal。参数需重新编译，不新增生产CLI。测试目标保留GO_STAND_DURATION_S缩短时长的专用入口，生产不读取该变量。

运行：从仓库根构建，下面最后一行会实际连接电机，必须由用户在机械已固定且处于规定启动姿态时执行，不与monitor/probe共用串口。

```bash
cmake -S imgo2_deploy/go_imu -B /tmp/imgo2_stand_fsm -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/imgo2_stand_fsm -j4
/tmp/imgo2_stand_fsm/go_stand
# 输入0，再回车；Ctrl+C停止并尝试卸力。
```

日志目录、50Hz目标/反馈记录、退出统计、限幅预警和原线程看门狗保留。状态机本身不持有电机串口；FSM.stop仅退出状态，应用随后multi.stop才发送停机命令。它仍是电机工具，不启动IMU，没有新增策略或整机IMU故障联动。不要机器人已站起后重启程序再次按初始姿态归零。命令到达目标不代表机械反馈已经无误差到达。

## 验证与限制

临时Release构建成功，CTest 18/18通过。新增stand_fsm覆盖：完整行输入、无事件长期保持、触发点无跳变、中点位置和导数/峰值速度、重复事件不重计时、终点零速度及持续保持、重复stop、停止后拒绝update、非法时长；默认12关节整个8秒轨迹每10ms采样，全部通过已配置限幅而不裁剪。

既有stand_pty验证实际应用经FSM后的四总线线上命令、轨迹、日志和信号退出；其人工坐标测试与默认真实配置测试分开。其余17项回归包含硬件接口/串口/故障/监测等。未启动真实电机或IMU，没有实机站起结果。git pull受.git/FETCH_HEAD只读限制，保留已有工作树。

待确认：新FSM版本实机输入/站起/保持/退出、真实运动及限位裕量；过快配置仍按原行为打印峰值超限警告并由Safety裁剪，不自动延长时长。下一状态/策略切换尚未实现，未来必须显式设计完整命令交接和硬件故障门控。
