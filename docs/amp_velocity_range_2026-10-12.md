# 新数据与AMP速度范围入口核对（2026-10-12）

用户说明已重新采集到motion_new，要求定位速度限制修改位置；本轮仅核对入口，未修改训练范围或数据路径。

已确认：AMP Height训练速度范围在 `imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/base_move/amp_env_cfg.py` 第123–125行，Imgo2AmpMoveEnvCfg.__post_init__：lin_vel_x=(-1,1.5) m/s、lin_vel_y=(-1,1) m/s、ang_vel_z=(-1.57,1.57) rad/s。同类第118–119行关闭速度课程，不会自动扩大范围。play覆盖第265–267行固定vx=1、vy=0、yaw=0；训练范围与回放固定指令须区分。Fanziqi另有apply_rlamp_env_settings范围覆盖，不应误改那组来调整Height。

用户已修改assets/imgo2.py默认数据位置为 `_PROJECT_ROOT/datasets/imgo2_motion_new`（第16行），环境变量IMGO2_AMP_MOTION_DIR仍优先。当前本机配置目录不存在；全仓搜索找到实际新数据位于 `imgo2_dataset/datasets/imgo2_motion_new`。两条路径错位，需修正默认路径或用IMGO2_AMP_MOTION_DIR指向实际目录。不能称新数据加载已通过，也不擅自创建空目录。实际路径需要对应文件或环境变量确认。

已完成源码行号与目录存在检查；未启动训练。待确认：新数据加载路径修正、字段/时间/FK质量与实际速度覆盖、用户希望的范围。建议按新数据实际实现的速度设置范围，不仅按文件名中的命令值；具体数值本轮未作建议或修改。新数据训练须重启进程，并在独立运行核对保存yaml与加载文件数。

## 用户范围修改与RLAMP常量用途确认

用户已将Height基类训练范围设为vx±1 m/s、vy±.4 m/s、yaw±1 rad/s，当前行号136–138；RLAMP_COMMAND_RANGES第395行也已为同值。yaw是绕身体z轴角速度，1 rad/s≈57.2958度/秒，不是角度。

调用链核对：Height注册映射到Imgo2AmpMoveEnvCfg，其初始化不调用apply_rlamp_env_settings；Fanziqi配置继承该类后再调用apply_rlamp_env_settings，其中遍历RLAMP_COMMAND_RANGES覆盖范围。所以仅训练Imgo2-basemove-flat-amp-height时不需要改RLAMP常量；训练Fanziqi时需改那组最终覆盖值。同值无害。play仍单独固定指令。仅核对并记录用户修改，未替用户再改源码或运行训练，AMP-16新数据加载验证仍待完成。

## 任务名与实验名入口

任务名由base_move/__init__.py的gym.register(id=...)声明：Height第47行Imgo2-basemove-flat-amp-height、第57行play；Fanziqi第67行Imgo2-basemove-flat-amp-fanziqi、第77行play。kwargs中的env_cfg_entry_point与amp_rsl_rl_cfg把名字映射到环境和runner类。训练命令--task选择这个id；agent配置experiment_name=base_move_amp_height是日志实验目录名，二者不同。已核对注册表与CLI选择，本轮无代码修改，运行与新数据检查仍按AMP-16待确认。

## 同日复审更新

默认新数据路径错位现已修正，Height滞空项明确移除；资源/字段/时间/FK与6项配置测试通过，Isaac Lab短训练待验证。当前依据与具体修法统一见 [复审](amp_settings_review_2026-10-12.md)，前文缺失路径为历史发现。
