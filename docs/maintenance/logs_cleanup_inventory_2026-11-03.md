# logs评测目录清理核对（2026-11-03）

用户询问gazebo compare文件作用与能否删除，本轮仅清点、未删除或移动文件，未改训练/部署。

## 实际用途

imgo2_rl/logs/gazebo_compare_2026-10-30及b/c/d，是10月30日不同轮次Gazebo对比/重试原始产物，不是Gazebo运行依赖，也不是训练checkpoint目录；字母后缀表示不同测试批次，不能解读为全是逐字重复数据。其下latest/new/policy1/forward03/backward03等为被测策略和固定指令。

- gazebo.log、launch.log、rl_sim.log、eval.log、ros_logs：启动/控制器/FSM/通信诊断日志。
- stable.json、*_raw.json、CSV：位姿/速度/关节等原始时序，支撑重新画曲线、FK净空重建、重新计算步频或抖动。
- summary.json和每case普通JSON：已计算的速度、高度、周期、抖动摘要。
- *_transient.json：起身和策略切换全过程及电机命令/反馈，用于分析切换抖动，不是训练数据。
- gazebo_home/.gazebo/.sdformat等：测试隔离HOME产生的模拟器缓存/日志。
- .pt/.bak：测试时快照或临时替换的策略备份，不能一概当文本日志。

对训练和部署而言，这些评测文件删除后不会改变当前模型、奖励、策略权重或自动降低性能。但删除原始数据后，旧结论只能看docs里的记录，无法直接重算或重绘；docs/JSON中引用的原始路径也不再存在。运行评测脚本可重新生成新结果，不代表在未来代码/模型已经变化时能逐字复原旧轨迹。

## 大小与处理建议

本次du -sh：gazebo_compare_2026-10-30约87MiB、b约12MiB、c约16MiB、d约6.5MiB，合计约121MiB（显示为取整值）。这四个旧评测目录可以作为清理候选，前提是接受丢弃旧原始测试证据。没有授权删除，本轮没有执行。

另外ppo_compare_2026-11-03约444MiB、himloco_compare_2026-11-03约203MiB，是最近两次实际对比；已确认两目录不含.pt/.onnx训练权重。建议优先保留最新PPO原始时序用于后续续训对比，或者归档后清理。主要汇总/PNG在docs/deployment，原始时序未全部复制进去。

旧gazebo四目录所有14个.pt或.bak文件已做SHA256外部重复检查：每个均在这些目录之外找到字节一致副本。例：最早snapshot和多份original.bak匹配rsl_rl/base_move_rough/2026-10-29_00-07-14/exported/policy.pt；b批备份匹配2026-10-30_00-53-27/exported/policy.pt；c的new_policy匹配deploy/ppo/policy_latest_1.pt；d的new_policy匹配logs/mujoco_three_stage_2026-10-31/stage3_6900.pt等。此结论仅针对当前文件状态，不授权连同外部唯一副本一起删除。

本次源码搜索未发现训练/生产部署引用gazebo_compare目录作为运行输入。报告和汇总JSON存在历史数据路径引用，删除后须标记为“原始评测数据已清理”，不能仍声称这些路径可复算。

## 不要混删的内容

- logs/rsl_rl、amp_rsl_rl及日期命名训练运行目录：先看是否有model_*.pt、params/env.yaml/agent.yaml、events.out.tfevents；它们用于续训、导出、奖励/超参数溯源和训练曲线。
- 不可因为目录名包含compare或有.pt就批量通配删除；其它policy_*compare、mujoco_*目录中也可能有策略快照，需分别核对唯一性。
- deploy/ros2_ws/policy下当前策略和基准policy_1是运行输入，不属于本次日志清理候选。

已完成：目录类型/大小、旧策略备份外部重复、生产引用核对。待用户决定保留原始证据或接受删除；无实际删除、无运行验证需求。
