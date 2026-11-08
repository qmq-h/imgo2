# HIMLoco 与 PPO policy_1 复核（2026-11-02）

> 2026-11-03后续：已在沙箱外实际完成双引擎测试，旧环境阻塞不再作为当前状态。现有HIM在当前Isaac配置和Gazebo部署均未正常跟速；原训练配置仍缺。见[实测记录](himloco_ppo_comparison_2026-11-03.md)。

## 结论与边界

本轮没有获得 Isaac/Gazebo 运动对比数据，不能确认“同一份 imgo2 HIMLoco 权重在 Isaac 正常而 Gazebo 异常”。已确认部署键2加载 `himloco/himloco.pt`，键1配置仍加载 `ppo/policy.pt`，不是用户指定的 `policy_1.pt`。README §5.3/§5.4 与 `fsm_imgo2.hpp` 将 HIMLoco 文件标为 Go2 参考占位；本地未找到 `logs/himloco_rsl_rl` 训练目录，未完成原 checkpoint 数值匹配，不能把这个历史标签当成独立验证过的模型来源。

当前工作区原有多项未提交修改，本轮未 pull/merge，未替换模型或修改训练参数。仿真启动只读现有工作树，不代表某个干净提交的验证结果。

## 整体代码路线

相对路径均以仓库根为基准：

1. `imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/base_move/__init__.py` 注册 `Imgo2-basemove-rough-himloco` 与其 `-play` 任务。
2. 同目录 `himloco_env_cfg.py` 定义奖励、观测、动作及随机化；`assets/imgo2.py` 当前指向 `imgo2_description/urdf/imgo2_real.urdf`，名义 PD 为25/.5，默认角0/.87/-1.82。
3. `imgo2_rl/scripts/rl_lab/himloco/train.py` 创建环境并交给 `rl_lab/wrapper/himloco_vec_env_wrapper.py`。单帧45维，history_length=5表示六帧270维；最新帧在前。
4. `rl_lab/runners/him_on_policy_runner.py` 采样、记录终止前 critic 状态、计算 returns、更新；`rl_lab/algorithms/him_ppo.py` 执行 PPO 更新及 estimator 更新。
5. `rl_lab/modules/him_estimator.py` 把270维历史编码为3维速度估计与16维归一化隐变量；`him_actor_critic.py` 把当前45维与19维估计拼为64维，Actor 为64→512→256→128→12。训练还使用 critic 特权输入；部署不需要 critic。
6. `himloco/play.py` 加载 checkpoint，调用 `rl_lab/utils/export_policy.py` 导出组合 estimator+actor；部署 FSM 键2进入 HIMLoco，`src/rl_sim.cpp::Forward()` 确实有历史缓冲分支，并非始终给网络单帧45维。

普通 PPO 则直接把单帧45维送入 Actor。HIMLoco 本身也使用 PPO 优化，主要区别在历史编码器、估计器辅助训练及 Actor 的额外输入，不是另一种底层电机控制。

## 输入与控制契约

| 项目 | 当前 imgo2 HIMLoco 训练源码 | 当前 HIMLoco 部署配置 | PPO policy_1 使用的部署配置 |
|---|---|---|---|
| 单帧顺序 | commands/angular velocity/gravity/q/dq/action | 同左 | angular velocity/gravity/commands/q/dq/action |
| 历史 | 六帧，最新在前 | 六帧，最新在前 | 单帧 |
| 指令缩放 | 原值 | 2/2/.25 | 1/1/1 |
| 默认角 | 四腿0/.87/-1.82 | 髋±.1，前大腿.8/后1，小腿-1.5 | 四腿0/.87/-1.82 |
| PD | 25/.5 | 40/1 | 25/.5 |
| 力矩参数 | 23.7（DCMotor还有速度相关限制） | 33.5 | 23.5 |
| 关节顺序 | FL/FR/RL/RR | mapping交换左右；当前base列表已是FL/FR/RL/RR | 恒等 |
| 动作裁剪语义 | 缩放/偏置后的关节目标±3 | raw动作±100 | raw动作±3 |

部署的33.5只是策略配置，实际还受模型/控制器限制，不等于实测输出33.5Nm。训练动作clip与部署raw clip属于不同环节，不能仅看数值判断一致。配置差异说明当前键2不能视为“本仓训练策略已经对齐的部署”，并不意味着将所有参数改成 PPO 就能修复未知来源的权重。

## 已修复：评测脚本 HIMLoco 选择方向

现象：`imgo2_deploy/ros2_ws/scripts/eval_gazebo_policy.py --key 2` 将 axes[6] 设为+1。根因：控制器 `JoyCallback` 中 RB+DPadRight 对应 axes[6]<0；正方向对应 DPadLeft，FSM 不会因此进入键2的 HIMLoco。

修法：仅将该评测按键改为-1并更正文内注释，没有修改手柄真实绑定、策略权重或生产参数。

验证：`/usr/bin/python3 imgo2_rl/scripts/tools/check_gazebo_policy_selection.py -v` 两项测试通过（三策略分支子检查与起身阶段）；直接提取执行真实 `Eval.tick()`，核对实际 C++ 回调方向、RB键、其它十字键轴和速度命令。这个检查不导入 ROS，只证明发出的输入与源码绑定吻合；ROS/FSM闭环待运行。

## 本轮验证与阻塞

- 环境：`nvidia-smi` 无法连接 NVIDIA 驱动；`/home/tommy0929/miniconda3/envs/env_isaaclab/bin/python` 中 PyTorch 2.7.0+cu128、CUDA available=false。因此未运行 Isaac 动力学。
- Gazebo：实际执行无界面 launch，ROS/FastDDS 报 `getifaddrs: Operation not permitted`、`Error creating socket: Operation not permitted`，`/spawn_entity` 不可用。首次另有默认HOME下 `.gazebo` 只读错误；改临时HOME后重试以区分写目录问题与通信限制。未进入策略、没有可用轨迹，不能记为HIMLoco运动失败。
- CPU：两文件各1024个确定性随机输入，权重和输出均有限、输出1024×12。HIMLoco JIT 与其内部 estimator→速度/归一化latent→actor 重算最大差0；这只排除该组合推理路径的数值不一致，**没有**验证与原训练 checkpoint 相等，也没有证明步态正常。
- 哈希、网络形状与CPU结果见 [机器可读记录](himloco_ppo_review_2026-11-02.json)：HIMLoco `d9945f85…`，PPO policy_1 `53a57f90…`。启动日志的摘要随记录保留。

## 待修复／待确认

1. DEPLOY-01：缺与该 HIMLoco 权重对应的 imgo2 checkpoint、训练run保存的 env/agent 配置及正式导出来源。需要这些才能区分Go2占位、其它历史模型与当前训练结果；不能从网络形状确定训练机器人。
2. HIM-SIM-01：缺可用 NVIDIA 驱动/GPU 与允许本机 ROS 通信的执行环境。恢复后使用同一份权重，先平地0/.3/.5m/s与转向；记录实际加载哈希、FSM状态、指令、机身高度/速度、姿态、关节及历史帧时序，才可评价sim2sim。
3. 新增评测方向修复已离线验证，真实键2状态切换待上述ROS环境；PPO对照必须显式加载policy_1，不能误用当前policy.pt。没有必要在来源尚未确认时修改生产HIMLoco配置或启动重训。


## 2026-11-03 执行权限复核与用户现象补充

用户明确：机器人是在按键2进入HIMLoco后侧倒，属于待复现的运行现象；评测脚本十字键方向修复不能解释用户物理键盘按2后的侧倒。

重新用 `/usr/bin/python3` 独立创建AF_INET UDP/TCP socket并绑定127.0.0.1，两者均在创建/绑定尝试阶段返回 `PermissionError(1, 'Operation not permitted')`。这项检查不依赖ROS、Gazebo、模型或策略，确认本轮通信阻塞属于当前执行环境限制；关GUI不能解除。历史无界面成功记录不等于本会话拥有相同权限，也没有证据确定权限在哪次发生变化。当前权限配置不允许申请sandbox提升，未尝试通过其它通道绕过限制。

本轮没有新增代码修复：已修复项仍为上一轮的评测方向（离线通过）；待确认项仍为用户按2侧倒的根因与同权重双引擎对比。需允许ROS本机socket通信的执行环境才能继续Gazebo闭环；当前nvidia-smi失败只说明本执行环境不可用，不能据此认定用户正常终端的驱动损坏。
