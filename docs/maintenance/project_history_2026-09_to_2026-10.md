# Imgo2 项目历史总结（2026 年 9–10 月）

本页汇总早期阶段性排查和验证。日期、配置、指标按原实验口径保留。运行入口和最新状态见[根 README](../../README.md)。

<a id="model-and-workspace"></a>

## 模型、仓库与工具

- 9 月中旬统一模型关节命名、轴、限位、质量和惯量，并将模型与网格集中到 `imgo2_description/`。旧 `imgo2_model/` 和训练、部署内的模型副本随后移除。历史 AMP 运动数据在 FL/FR/RL/RR 直接映射下，FK 最差坐标 RMSE 约 2.14 mm，交换 FR/RL 的最小误差约 225 mm。该结果对应当时的模型与录制数据。 [AMP 足端回放](../amp/gait_eval_amp_24500_2026-09-18.md)
- 模型拆为物理 Xacro 内核、Gazebo 接触/控制插件和 IMU 附件，生成 URDF 的消费者逐步收敛。10 月的 real 变体更改了部分连杆惯量和网格，仓库检查器仍需要补全模型族登记与网格指纹。11 月更新的机身几何及服务器模型路径另有实跑记录。 [模型同步](../deployment/model_sync_2026-11-22.md)
- 训练资源入口从机器绝对路径迁到仓库相对路径，支持环境变量覆盖。Python 环境、RSL-RL 与 checkpoint 版本必须匹配。策略由各算法的正式 `play.py` 导出，再核对输入输出和部署身份。 [训练前置检查](../training/ppo_training_preflight_2026-10-14.md) · [服务器导出命令](../xiaoji_cmd/策略导出与ROS2仿真.md)
- 单仓整合、分支迁移和文档主题分类均已完成过相应静态核对。历史 Git 分支审计及迁移报告是当时快照，复现时以当前分支、内容哈希和本地配置为准。 [架构与入口](project_architecture_2026-10-12.md)

<a id="amp"></a>

## AMP 数据、训练与仿真行走

- 9 月 AMP 对齐工作核对了参考动作、机械轴、观察维数和腿序。45 维 Actor 正式导出后可在 Gazebo 行走，早期竖直漂移或贴地步态不能简单归为权重未加载。保留的速度实验记录了 0.2–1.5 m/s 指令下的跟踪和对角相位。足端回放中步幅约为参考的 0.61 倍，步频约 3.1 倍，这些是当时的步态差距。 [AMP 实验与权重](../amp/amp_experiments_2026-09-17.md) · [足端复验](../amp/gait_eval_amp_24500_2026-09-18.md)
- 10 月训练复盘覆盖站定、足端滞空、参考速度分布、接触切换和重采动作数据。不同 checkpoint、脚高口径与仿真器之间的对照要保留源哈希和统一窗口。跟踪稳定与复现参考步幅是两个验收结果。 [新动作训练入口](../amp/amp_motion_new_training_2026-10-12.md) · [双引擎部署对照](../deployment/amp_engine_comparison_2026-10-14.md)

<a id="ppo-training"></a>

## PPO 配方与策略比较

- 10 月平地、粗糙地形和三阶段续训记录比较了速度指令范围、课程、脚高、滞空、镜像、横滚与动作变化奖励。策略排名会随前进、后退、转弯和地形条件变化。保留的正式导出和同条件对照应按 checkpoint/权重哈希比较。 较晚的训练对照（本机 `nov11_two_runs_review_2026-11-11.md`）
- 高度项需正确处理无效地面射线。支撑脚滑动用世界系足端速度检查。髋偏离、横滚角速度、倾斜角和摆动脚相对机身高度分别约束不同量。仅增大权重不能从旧结果推出实际止抖。 [奖励实现修订](../training/velocity_reward_fixes_2026-10-31.md) · [最新 Rough 配置核对](../training/pasted_rough_config_review_2026-11-22.md)
- Gazebo 曾生成 8 cm 和 15 cm 台阶场景，15 cm 场景为八阶、30 cm 踏面。模型加载与场景几何通过并不构成策略通过台阶的验收。18 cm 连续两阶上下行仍有独立任务与验证要求。 [Gazebo 启动](../../README.md#62-ros--gazebo) · [18 cm 任务路线](../training/ppo_18cm_stairs_route_2026-11-13.md)

<a id="simulation-and-deployment"></a>

## 仿真链路、时序与实机部署

- ROS 2 Gazebo 通过模型描述包、控制器与策略节点运行。路径解析和单行 robot_description 修复后，已验证模型生成、关节反馈和控制器链路。MuJoCo 由独立入口读取 MJCF。此前修正了基座四元数来自惯量主轴的坐标问题，并保留足端接触及静态模型核对。 [Gazebo 配置与复现](../deployment/gazebo_ros2_bringup_2026-09-17.md) · [MuJoCo 操作与接口](../xiaoji_cmd/mujoco_code_and_usage_2026-10-23.md)
- 真机入口逐步从串口/IMU/站起测试拆为硬件适配、策略 SDK、FSM 与诊断工具。硬件协议、方向、减速比和目标安全检查有离线覆盖。10 月的离线或短时验证只在各自条件成立，真实走动、安装零位及命令应用时间另需实机依据。 [真机流程](../deployment/go_imu_control_flow_2026-10-15.md) · [独立部署](../deployment/go_imu_sim2real_2026-10-15.md)
- 早期 5 ms 日志审查分辨了主循环、策略结果、串口反馈和电机内部执行时刻。相位差和事务耗时要各自标注。后续日志改为保存真实输入输出，横滚实验的证据和限制见[近期排查入口](../deployment/ppo_vibration_status_2026-11-22.md)。

<a id="towing"></a>

## 拖曳与研究工作

- 引入弹性绳、低弹性绳和被动刚体连杆，配合基座和小车状态、张力、接触、追尾及停车距离的记录。上层策略经历指令输出、冻结底层策略和关节位置残差等阶段。800 环境网格覆盖连接、长度、坡面和质量，并保存有车与无车基线。 [拖曳仿真验证](../towing/towing_simulation_validation.md) · [必要性测试数据](../towing/data/2026-10-09_necessity_800/overview.md)
- 控制和场景错误曾导致整批倒地，修复物理步进的渲染分支后通过重跑定位。latent 实验在站定段区分绳和杆较明显，弹性绳和低弹性绳难在该结果中分开。牵引、滑行的全量聚类主要被质量与速度驱动，ARI 约 0.001 属于当时混合样本的结果。轮阻尼 A/B 的行为效果和潜变量是否包含阻尼仍为不同问题。 [拖曳原始统计](../towing/data/2026-10-10_amp_envelope_800/overview.md) · [潜变量分析工具](../../imgo2_rl/scripts/tools/analyze_latent_clusters.py)
- 早期研究草稿提出上层 recurrent PPO、decoder 和 sim2sim 的路线。已有场景、离线数据和训练产物各有适用范围，上层策略最终部署能力仍按 README 问题表验收。

## 原文查找

本次移除的 Markdown 和替换前的入口文档保存在根目录的本地备份 `.git-backups/docs-consolidation-2026-11-22.tar.gz`。压缩包包含原路径及逐文件 SHA256。Git 跟踪的版本也可按历史提交查询。

<details>
<summary>原文索引（88 份）</summary>

- 新25000训练趋势、旧策略比较与零指令站立抖动 （`docs/amp/amp_25000_loss_and_standing_review_2026-10-13.md`）
- motion_new 4000轮策略正式导出与部署替换 （`docs/amp/amp_4000_deploy_2026-10-12.md`）
- AMP Height 滞空奖励启用记录（2026-10-10） （`docs/amp/amp_air_time_reward_2026-10-10.md`）
- AMP MPC 采集脚本与是否重采复核（2026-10-11） （`docs/amp/amp_collection_review_2026-10-11.md`）
- AMP 配置收缩记录（2026-09-22） （`docs/amp/amp_config_cleanup_2026-09-22.md`）
- 新 motion_new 25000：足端抬高、贴地与步态周期检查 （`docs/amp/amp_foot_lift_review_2026-10-13.md`）
- AMP低速全方向、静止平滑与抬脚微调方案 （`docs/amp/amp_low_speed_finetune_plan_2026-10-13.md`）
- AMP 新数据、速度范围与关闭滞空奖励复审 （`docs/amp/amp_settings_review_2026-10-12.md`）
- 当前25000策略的慢步态续训与奖励缺口 （`docs/amp/amp_slow_gait_resume_review_2026-10-13.md`）
- AMP「站着不动」对照报告（2026-09-17） （`docs/amp/amp_standstill_diagnosis_2026-09-17.md`）
- AMP 采样与时间步长检查（2026-10-10） （`docs/amp/amp_timing_audit_2026-10-10.md`）
- AMP Height 24000 轮训练复核（2026-10-10） （`docs/amp/amp_training_plateau_2026-10-10.md`）
- 新数据与AMP速度范围入口核对（2026-10-12） （`docs/amp/amp_velocity_range_2026-10-12.md`）
- 确定目标/反馈错位属于哪一段：测量与改进方案 （`docs/deployment/command_feedback_identification_plan_2026-10-21.md`）
- real / ros2_ws 目录迁移后的路径修复（2026-10-20） （`docs/deployment/deploy_layout_migration_2026-10-20.md`）
- PPO仿真与真机频率核对（2026-10-21） （`docs/deployment/deployment_frequency_audit_2026-10-21.md`）
- +0.5 m/s指令下PPO仿真与实机关节、力矩及IMU对照 （`docs/deployment/forward_sim_real_2026-10-22.md`）
- 第 29000 轮 AMP 的 Gazebo 无界面测量 （`docs/deployment/gazebo_amp_29000_2026-10-10.md`）
- 新 motion_new 25000：切换抖动、后退与转向排查 （`docs/deployment/gazebo_amp_motion_new_25000_2026-10-13.md`）
- Gazebo ROS2可配置关节/IMU反馈延迟 （`docs/deployment/gazebo_feedback_delay_2026-10-21.md`）
- Gazebo 无头步频测量（2026-10-09） （`docs/deployment/gazebo_gait_frequency_2026-10-09.md`）
- 新 5000 轮策略 vs policy_1：Gazebo/MuJoCo sim2sim 对比（2026-10-30 13:39） （`docs/deployment/gazebo_ppo_new_policy_2026-10-30.md`）
- 新 policy 791021 vs 原 policy / policy_1：sim2sim 复测（2026-10-30 17:14） （`docs/deployment/gazebo_ppo_policy_791021_2026-10-30.md`）
- 最新 policy ce77 vs 上一版 791021 / policy_1：sim2sim 检查（2026-10-30 20:52） （`docs/deployment/gazebo_ppo_policy_ce77_2026-10-30.md`）
- Gazebo 无界面对比：最新 policy.pt vs policy_1（2026-10-30） （`docs/deployment/gazebo_ppo_policy_compare_2026-10-30.md`）
- Gazebo 15cm八阶楼梯（2026-10-26） （`docs/deployment/gazebo_stairs_15cm_2026-10-26.md`）
- Gazebo楼梯降为8cm（2026-10-26） （`docs/deployment/gazebo_stairs_8cm_2026-10-26.md`）
- Gazebo 启动翻倒、再次落地与 Passive 姿态排查 （`docs/deployment/gazebo_startup_posture_2026-10-12.md`）
- Gazebo启动失败：外层description引用与重复服务 （`docs/deployment/gazebo_startup_repair_2026-10-13.md`）
- go_imu 五阶段整理与验证（2026-10-14） （`docs/deployment/go_imu_cleanup_2026-10-14.md`）
- go_imu入口精简与下一步测试（2026-10-15） （`docs/deployment/go_imu_entry_cleanup_2026-10-15.md`）
- 首次实机部署反馈（2026-10-16） （`docs/deployment/go_imu_first_hardware_2026-10-16.md`）
- go_imu 合并与联合监测（2026-10-10） （`docs/deployment/go_imu_integration_2026-10-10.md`）
- go_imu 默认限幅赋值修复（2026-10-14） （`docs/deployment/go_imu_limits_2026-10-14.md`）
- go_imu 通信与整机接口重构（2026-10-11） （`docs/deployment/go_imu_refactor_2026-10-11.md`）
- PPO重复切换与回站检查（2026-10-15） （`docs/deployment/go_imu_return_stand_review_2026-10-15.md`）
- go_imu 站立 FSM（2026-10-14） （`docs/deployment/go_imu_stand_fsm_2026-10-14.md`）
- go_imu 状态拆分、独立姿态与Passive卸力 （`docs/deployment/go_imu_state_split_2026-10-15.md`）
- Imgo2 多状态策略部署结构建议（2026-10-14） （`docs/deployment/imgo2_policy_structure_review_2026-10-14.md`）
- 200Hz IMU与gyro噪声对照实跑 （`docs/deployment/imu200_noise_validation_2026-10-21.md`）
- IMU安装位置与sim2real坐标核对（2026-10-14） （`docs/deployment/imu_frame_review_2026-10-14.md`）
- 新批量通信日志：四髋反馈与IMU振荡核对 （`docs/deployment/imu_hip_new_log_2026-10-21.md`）
- IMU 启动零位偏置 + 策略输入对应性核对（2026-10-23） （`docs/deployment/imu_startup_bias_2026-10-23.md`）
- 继续对照：仿真小延迟不抖后的关节动态与记录时序 （`docs/deployment/joint_feedback_dynamics_2026-10-21.md`）
- MJCF与imgo2_real逐项对齐检查（2026-10-23） （`docs/deployment/mjcf_real_alignment_2026-10-23.md`）
- 新policy.pt与真机policy_1：同条件对比及摆动足高建议 （`docs/deployment/mujoco_ce77_vs_policy1_2026-10-30.md`）
- MuJoCo 当前 AMP 对照的准备核查（2026-10-14） （`docs/deployment/mujoco_current_model_readiness_2026-10-14.md`）
- MuJoCo启动Passive抽搐排查（2026-10-23） （`docs/deployment/mujoco_passive_timing_2026-10-23.md`）
- PPO无界面Gazebo前后变速与IMU曲线（2026-10-16） （`docs/deployment/ppo_gazebo_ramp_2026-10-16.md`）
- 当前PPO转向与快速减速的双引擎实测（2026-10-26） （`docs/deployment/ppo_maneuvers_2026-10-26.md`）
- 键 1（PPO）权重来源复核（2026-09-19） （`docs/deployment/ppo_policy_provenance_2026-09-19.md`）
- PPO 上机前接口核对（2026-10-14） （`docs/deployment/ppo_real_readiness_2026-10-14.md`）
- 图中Gazebo PPO与338秒实机IMU对照 （`docs/deployment/ppo_sim_real_imu_comparison_2026-10-21.md`）
- 当前部署PPO来源与8cm楼梯双引擎确认（2026-10-26） （`docs/deployment/ppo_stairs_confirmation_2026-10-26.md`）
- 四足Sim2Real运动振荡：论文与排查路线 （`docs/deployment/sim2real_vibration_literature_plan_2026-10-21.md`）
- AMP sim2sim 适配记录（2026-09-17） （`docs/deployment/sim2sim_amp_2026-09-17.md`）
- MuJoCo 与 Gazebo 的步态/抖动差异（2026-09-19） （`docs/deployment/sim2sim_mujoco_vs_gazebo_2026-09-19.md`）
- 部署线程/观测时序复审：5 ms 主环到底加了多少延迟（2026-10-23） （`docs/deployment/thread_obs_timing_review_2026-10-23.md`）
- 训练侧 ↔ 部署侧还有哪些差异（不含仿真器物理）（2026-09-19） （`docs/deployment/train_vs_deploy_gap_2026-09-19.md`）
- 新上传批量总线代码与283秒日志复核 （`docs/deployment/uploaded_pipeline_log_review_2026-10-22.md`）
- 代码复审记录（2026-09-15） （`docs/maintenance/code_review_2026-09-15.md`）
- docs 散文件归类（2026-10-14） （`docs/maintenance/docs_reorganization_2026-10-14.md`）
- Git分支与远端提交核对 （`docs/maintenance/git_branch_audit_2026-10-21.md`）
- 分支推送与部署目录建议（2026-10-20） （`docs/maintenance/git_deploy_layout_review_2026-10-20.md`）
- 远端同步与冲突处理（2026-10-12） （`docs/maintenance/git_sync_2026-10-12.md`）
- 远端同步与变更复审（2026-10-09） （`docs/maintenance/git_sync_review_2026-10-09.md`）
- 2026-09-22 离线检查记录 （`docs/maintenance/offline_check_2026-09-22.md`）
- 部署策略 Git 历史核对（2026-10-14） （`docs/maintenance/policy_git_history_2026-10-14.md`）
- 按修改后的core生成real URDF （`docs/maintenance/urdf_real_regeneration_2026-10-13.md`）
- 评审：`paper_plan_imgo2.md`（未知绳索负载下的瞬态启停控制） （`docs/research/paper_plan_review_2026-09-18.md`）
- 拖曳能力与 Direct Stop 边界扫描（2026-09-21） （`docs/towing/towing_boundary_scan_2026-09-21.md`）
- 拖曳仿真实验充分性评估（2026-10-11） （`docs/towing/towing_experiment_sufficiency_review_2026-10-11.md`）
- 拖曳 Dynamics Decoder 与 recurrent PPO（2026-09-22） （`docs/towing/towing_gru_design_2026-09-22.md`）
- 冲击窗口独立统计（`play_towing_test.py`，2026-10-10） （`docs/towing/towing_impact_window_2026-10-10.md`）
- Towing latent 聚类检验：连接类型可分性与连接内的质量分布（2026-10-12） （`docs/towing/towing_latent_cluster_2026-10-12.md`）
- Towing latent t-SNE 使用记录（2026-10-11） （`docs/towing/towing_latent_tsne_2026-10-11.md`）
- 拖曳第三类连接：刚体球铰连杆 + 20×20 场景网格（训练侧，2026-10-08） （`docs/towing/towing_rigid_link_2026-10-08.md`）
- 平滑起伏地形（2026-10-12） （`docs/towing/towing_rolling_terrain_2026-10-12.md`）
- 拖曳上层 RL 冒烟实跑结果（2026-09-22） （`docs/towing/towing_smoke_run_2026-09-22.md`）
- 上层拖曳 RL 链路修复（2026-09-22） （`docs/towing/towing_upper_rl_fix_2026-09-22.md`）
- 拖曳工作域收紧：质量 5–20 kg、速度 0.5–1.5 m/s（2026-10-10） （`docs/towing/towing_work_domain_2026-10-10.md`）
- joint_pos_penalty复审与四髋角度软约束设计（2026-10-27） （`docs/training/hip_angle_constraint_design_2026-10-27.md`）
- PPO髋角、MPC落脚点与公开工程奖励/速度课程对照 （`docs/training/ppo_hip_mpc_public_comparison_2026-10-27.md`）
- PPO 后髋持续饱和：奖励、镜像符号与探索分布核对（2026-10-27） （`docs/training/ppo_hip_saturation_reward_audit_2026-10-27.md`）
- 当前PPO rough配置与feet_height_body原理复审 （`docs/training/ppo_rough_config_fhb_review_2026-10-30.md`）
- PPO rough奖励、模型与复杂地形训练复审（2026-10-26） （`docs/training/ppo_rough_review_2026-10-26.md`）
- PPO站立接管与低速训练方案（2026-10-27，建议尚未实施） （`docs/training/ppo_stand_entry_plan_2026-10-27.md`）
- 三个PPO训练目录的参数、效果与重训配方复核 （`docs/training/ppo_three_stage_reward_review_2026-10-31.md`）

</details>
