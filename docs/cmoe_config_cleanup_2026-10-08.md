# CMoE 配置清理（2026-10-08）

基线：远端 imgo2_CMoE 的 4a3459c。

## 已修复

- CMoE_env_cfg.py 从 1192 行缩至 623 行：删除历史调参长注释与注释掉的候选代码，缩短类说明。
- 删除 action_rate_l2、ang_vel_xy_l2 先归零后恢复的中间赋值。
- 删除最终关闭的 feet_gait 的函数替换、临时权重与参数覆盖；保留显式关闭和三种步态诊断。
- 删除关闭的 feet_stumble 的参数覆盖，以及与声明相同的中心线/航向地形参数赋值。
- 保留训练、先验训练、回放和复合道评测入口；保留观测、地形、课程、有效奖励及 None 守卫。
- 两条旧回归断言改为检查当前有效诊断和实际归零转换。

## 验证与限制

解释器：C:/Users/qmq/AppData/Local/Programs/Python/Python311/python.exe。

- test_check_reward_overrides.py：21 项通过。
- test_reward_terrain_partition.py：11 项通过。
- test_cmoe_posture_penalties.py：20 项通过、21 项跳过。
- 清理前后 AST 奖励审计：cmoe 28 项、cmoe-gaitfree 24 项，全部权重及有效项的函数覆盖一致。
- 配置源码编译通过；提交内容 git diff --cached --check 通过。
- test_gait_kernel_probe.py 曾运行：6 项通过、2 项跳过、3 项因缺 NumPy 报错；复合道测试模块因缺 NumPy 无法导入，不记为通过。

## 待修复／待确认

尚未构造 Isaac Lab 环境、短训或回放；需具备 NumPy/PyTorch、Isaac Lab/Isaac Sim 与 GPU 的训练环境验证。本轮不提供新的收敛结论。

原有维护记录及未跟踪排查文件保留。README 同步冲突双方记录均保留；本次只提交本轮清理记录，其余本地改动保持未提交。
