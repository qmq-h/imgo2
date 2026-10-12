# 2026-09-22 离线检查记录

## 范围

- 代码版本：`cf465a2`（`main` 与 `origin/main` 一致）。
- 工作目录：`C:\Users\qmq\Desktop\Imgo2`。
- 解释器：`C:\Users\qmq\AppData\Local\Python\bin\python.exe`，Python 3.14.6。
- 本轮只做无需 Isaac Sim、GPU、ROS 或硬件的检查，没有启动训练、仿真或后台任务。
- 检查前已有的本地删除 `research_exploration_plan.md` 保持不动。

## 已确认通过

从 `imgo2_rl/` 执行以下检查，退出码均为 0：

```powershell
python scripts/tools/check_asset_paths.py
python scripts/tools/check_model_sync.py
python scripts/tools/audit_amp_dataset.py
python scripts/tools/check_amp_joint_order.py
python -m compileall -q scripts source tests
python -m unittest discover -s tests -p 'test_towing*.py'
```

另从仓库根执行：

```powershell
git ls-files -i -c --exclude-standard
```

结果：

- 资源路径通过：训练 URDF 存在，17 个 mesh 引用无缺失，动作文件 21 份，小车 URDF 结构有效，无残留机器绝对路径。
- 模型同步通过：两份 Imgo2 URDF 的 12 个腿部关节、17 个核心 link、base 惯量、10 个网格及 FK 一致；另有 1 份小车 URDF 已登记。恒等映射最差 RMSE 0.00214 m，交换腿对照 0.22470 m。
- AMP 数据通过：21 份、5097 帧；恒等 FK 均值 0.00107 m、最大值 0.00214 m，跨腿交换均值 0.22491 m；髋外展对称性与位置／速度通道相关性检查通过。
- Python 语法编译通过。
- 拖曳测试共 194 项通过，12 项因缺少 PyTorch 或外部可选环境跳过；没有失败或错误。
- tracked-ignore 检查无输出，即没有已跟踪但被忽略的文件。

## 待修复／待确认

全量命令：

```powershell
python -m unittest discover -s tests -p 'test_*.py' -v
```

共收集 281 项，结果为 256 通过、24 跳过、1 个错误。唯一错误发生在收集 `test_gait_metrics.py` 时：当前解释器没有安装 `numpy`，因此模块导入失败；这不是测试断言失败，本轮也没有据此修改代码。

本机当前也没有可用的 PyTorch／Isaac Lab Python，所以 AMP 更新回归、质量 decoder 数值测试和 Isaac Lab 启动器检查继续跳过。完成 CHECK-01 需要：

1. 在带 NumPy 的解释器中重跑 `python -m unittest tests.test_gait_metrics -v`。
2. 在带 PyTorch 的训练环境中重跑全量测试，确认 AMP 更新和质量 decoder 测试不再跳过。
3. TOW-03 所列 physics adapter、真实间隙／碰撞 producer、自定义 runner 与 scripted policy 仍需训练机运行验证；本次离线检查不改变其状态。

## 已修复

本次检查没有确认新的代码缺陷，因此没有代码修复。维护缺口已补齐：README 的顶部核对日期、问题表 CHECK-01 和维护记录已同步更新，并明确区分已通过、跳过与未执行项。
