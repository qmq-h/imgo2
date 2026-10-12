# command.md 填写与命令核对（2026-10-11）

用户新建根目录command.md并要求填充五个标题；按用户指定位置保留该文件，覆盖AMP Height导出、Gazebo sim2sim、从头训练、续训和Isaac Sim GUI回放。完整命令仅在command.md展开，README提供入口。

已修复：说明缺少的每终端环境准备；run当前已移动到logs/amp_rsl_rl/base_move_amp_height/，旧上一级路径已失效，命令使用实际新路径。解释器采用本机已存在的env_isaaclab；跨机器路径标为本机示例需替换。train.py使用get_checkpoint_path按experiment/run名匹配，不能把绝对路径传作load_run；已有25000的run名与文件已实查。max_iterations是增量，续训1000从25000到26000；不会自动恢复历史奖励yaml。

训练示例明确覆盖任务比例0.7和feet_air_time.weight100，匹配用户近期实验；本地默认0.3/50未更改。此示例不是最优配方结论，当前低站高/速度超调仍见AMP-14。本轮不启动训练或仿真。

验证：/usr/bin/python3对11个Bash块分别执行bash -n，全部通过；调用当前IsaacLab-rsl5 get_checkpoint_path的实际函数AST成功定位model_25000.pt；checkpoint、专属导出、旧策略备份、解释器、real Gazebo模型、ROS安装setup与相对文档链接均存在。env_isaaclab Python的Hydra解析器确认agent.amp_task_reward_lerp、env.rewards.feet_air_time.weight与命令速度list覆盖语法正确。git diff --check通过。导出与Gazebo此前实测依据见gazebo_amp_25000_2026-10-11.md；本轮语法核对不等于重新运行。

待确认：GUI受历史GLX问题影响需用户实际查看；新训练/续训模板需在目标机器4环境10轮确认，1024环境只是例子不保证显存容量。沿用CHECK-01与AMP-14运行/效果待确认条目，不把长期训练或GUI标为通过。
