# PPO固定比例指令采样：追加实现与接入（2026-11-05）

## 状态与作用

> 2026-11-06补充：服务器已提供分组采样续训结果，本机完成正式导出及两引擎检查；MuJoCo慢步/净空改善，但Isaac后退与高度未通过，见[后续复核](../deployment/grouped_speed_review_2026-11-06.md)。下文11/5的“未训练”描述保留为实现当日边界。

已在`imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/mdp/commands.py`末尾追加`GroupedVelocityCommand`和`GroupedVelocityCommandCfg`，带中文注释。原文件逐字节前缀保持不变（追加前SHA256 `c36a86840b57bece7f21e04f5f128f89b7039084d32bac886aeec2db044fc609`）。现有任务仍引用旧Cfg，**本轮未改rough_env_cfg.py或激活任何任务**，AMP/HIM默认接线不变，未启动训练、未换策略。

已实现并通过短检查：10%静止、40%六方向低速、50%宽范围混合；低速范围独立于课程；父类standing mask同步；关闭heading防止纯yaw被覆盖；参数校验、部分/空批次及seed复现。初次负配置测试发现初始化失败时父类析构访问未初始化字段，已在新增类中预置该字段，重跑通过且最终验证哈希一致。

待确认：完整4096环境训练吞吐、分布改变的训练效果、低速步态/动态切换、rough/pit、Gazebo和硬件。短环境检查不等于训练效果已修复。分组采样不冻结旧策略、不锁定周期，也不保证慢步自动保留。

## 参数与采样算法

| 参数 | 默认值 | 作用 |
|---|---|---|
| standing_group_probability | .10 | 本次重采样全零指令概率 |
| low_speed_group_probability | .40 | 低速纯方向概率 |
| 剩余概率 | .50 | 宽范围三轴混合，读取现有ranges |
| low_lin_vel_x | (.25,.50) | 低速前后速度正幅值m/s |
| low_lin_vel_y | (.25,.40) | 低速横移速度正幅值m/s |
| low_ang_vel_z | (.20,.50) | 低速纯转向正幅值rad/s |

每次重采样独立抽组，低速组再均匀抽六方向（前/后/左/右/左转/右转），只激活一轴，其余两轴为0。因此每个低速方向约占总采样6.67%。比例是长期概率，不是每一时刻精确环境人数；环境不会永远固定在同一组。

父类先生成宽范围命令，新类覆盖静止/低速组并同步`is_standing_env`。`rel_standing_envs`必须为0，由新概率统一管理；`heading_command`必须False，否则明确报错。低速平移最小幅值必须>.2，以避开旧类硬死区。旧类代码及pit特殊处理原样保留：**pit上的实际指令仍会被强制为前进，不能声称那里的执行比例也等于分组比例**。

新类的`sampled_group`记录0静止/1低速/2混合，`sampled_low_direction`记录0/1前后、2/3左右、4/5左右转，非低速为−1；都是最近抽样快照，不是实际执行分类或时间平均。没有自动添加TensorBoard指标。

## rough_env_cfg.py如何接入

在文件顶部补充以下导入（存在时不重复）：

```python
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from imgo2_rl.tasks.manager_based.locomotion.velocity import mdp
```

在`Imgo2RoughEnvCfg.__post_init__()`**最后面**、已有Commands配置之后，加入以下块。此前的旧命令配置不必删除，但此块必须最后执行，避免之后又用旧值覆盖。最终范围沿用速度课程run的±1.2/±.8/±1.5，目标是保护低速样本同时保留宽范围；这些比例是候选默认值，尚未完成训练验收。

```python
# 新增分组采样：10%静止、40%低速六方向、50%宽范围混合。
self.commands.base_velocity = mdp.GroupedVelocityCommandCfg(
    asset_name="robot",
    resampling_time_range=(10.0, 10.0),
    debug_vis=False,
    heading_command=False,
    rel_heading_envs=0.0,
    rel_standing_envs=0.0,  # 静止比例由standing_group_probability统一控制
    standing_group_probability=0.10,
    low_speed_group_probability=0.40,
    low_lin_vel_x=(0.25, 0.50),
    low_lin_vel_y=(0.25, 0.40),
    low_ang_vel_z=(0.20, 0.50),
    ranges=mdp.GroupedVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.2, 1.2),
        lin_vel_y=(-0.8, 0.8),
        ang_vel_z=(-1.5, 1.5),
        heading=None,
    ),
)

# 显式重建课程：即使上面旧代码曾赋None，这里也能启用。
# 课程只修改ranges，即宽范围组；独立low_*范围不随课程扩大。
self.curriculum.command_levels_lin_vel = CurrTerm(
    func=mdp.command_levels_lin_vel,
    params={
        "reward_term_name": "track_lin_vel_xy_exp",
        "range_multiplier": (0.5, 1.0),
    },
)
self.curriculum.command_levels_ang_vel = CurrTerm(
    func=mdp.command_levels_ang_vel,
    params={
        "reward_term_name": "track_ang_vel_z_exp",
        "range_multiplier": (0.5, 1.0),
    },
)

# 先在平地验收低速与高速；奖励/PD/动作缩放保持原5300。
self.scene.terrain.terrain_type = "plane"
self.scene.terrain.terrain_generator = None
self.curriculum.terrain_levels = None
```

上述新Cfg的真实创建和课程装配已在本轮64环境测试中检查（实例化真实课程并核对初始/最终范围；定量计数仅调用重采样/后处理，不推进课程）。原课程的80%跟踪判据仍来自全部回合，不是只看宽范围组；静止/低速组可能让判据较容易通过。此实现保证保护组采样不会被范围扩张稀释，**不保证扩速质量门槛已经改成分组验收**。若不使用课程，将上面两个CurrTerm块换成`self.curriculum.command_levels_lin_vel = None`与`self.curriculum.command_levels_ang_vel = None`即可；分组仍生效。

## 验证记录与复现

解释器`/home/tommy0929/miniconda3/envs/env_isaaclab/bin/python`，Isaac Sim5.1、64环境平地，GPU实际运行10步；通过真实CommandManager创建新增类，并进行16,384次采样：

- 静止1659（10.13%）、低速6568（40.09%）、混合8157（49.79%）。
- 六方向计数1094/1092/1096/1123/1091/1072，非活动轴为0、正负号及幅值均通过。
- 父类_update_command在平地不覆盖指令、部分重采样不影响其他环境、空批次、固定seed复现通过。
- 将课程范围缩到平移死区内，低速组指令仍逐位不变，宽范围组正常使用新的范围。
- 全静止/全低速/全混合三种概率边界通过；非法概率、heading、父类静止比例、低速死区/倒置范围等6种配置拒绝通过。
- 最终源码哈希与验证记录一致；旧commands字节前缀核对通过，rough源文件未修改。

[机器记录与完整验证源程序](grouped_velocity_command_verification_2026-11-05.json)。验证源保存在JSON的`verification_source`字段，不仅留在临时目录。需要复现时可用标准库取出，随后由本机Isaac解释器运行：

```bash
# 在仓库根运行；解释器路径是本机示例，服务器替换为自己的训练环境。
python3 - <<'PY'
import json
from pathlib import Path
record = json.loads(Path("docs/training/grouped_velocity_command_verification_2026-11-05.json").read_text())
Path("/tmp/imgo2_check_grouped_commands.py").write_text(record["verification_source"])
PY
cd imgo2_rl
IMGO2_ISAACLAB_PYTHON=/home/tommy0929/miniconda3/envs/env_isaaclab/bin/python \
  bash scripts/run_isaaclab.sh /tmp/imgo2_check_grouped_commands.py
```

验证源中的ROOT也是本机绝对路径，换机器前应一并改成自己的仓库根。服务器首次接入后先4环境10轮冒烟，确认保存env.yaml的class_type为GroupedVelocityCommand、新概率及两项课程不为null，再进行训练；本轮未启动该训练。
