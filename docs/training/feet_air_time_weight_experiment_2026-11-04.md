# feet_air_time权重1.8/0.1实验（2026-11-04，已准备、未运行）

## 目的与状态

用户决定先做一组降低airtime权重的实验，检查对机身、步频和净空的影响。只改变该项权重，不改阈值、reward函数、指令采样、模型或PD。使用已有5899作为共同起点，不重新随机初始化。

已完成：核对本机checkpoint路径、当前rough配置与实验参数、准备命令。未修改默认训练配置、未启动训练；服务器源码/依赖及实际生成env.yaml、完整效果待验证。本次有意开展实验，之前减少人工反复续训的长期目标保留。

## 两组定义

| 项目 | 对照 | 实验 |
|---|---:|---:|
| feet_air_time.weight | 1.8 | 0.1 |
| threshold | 0.5 | 0.5 |
| 起点 | 同一model_5899.pt | 同左 |
| 追加轮数 | 1000 | 1000 |
| seed/环境 | 42/4096 | 同左 |

两台服务器可并行，项目版本/未提交修改、模型和依赖须一致。0.1是与当前配置形成明显对照的候选，不宣称最佳；即使某公开Go2配方也用0.1，也不能直接证明适合Imgo2。先保留原始checkpoint，两个run分别命名。只将实验组与5899比，无法排除新增1000轮本身的影响，因此仍需同起点/同预算的1.8控制组。

## 命令（在各服务器imgo2_rl目录执行）

本机当前rough默认base_height=-16、lin_z=-2、vx±1.2/vy±.8，至少这些与5899配方不同，不能只改airtime就称为沿用5899。因此两组命令同时显式恢复共同设置。以下run目录名及checkpoint已在本机确认存在；服务器需已复制相同run。解释器沿用服务器现有run_isaaclab.sh配置。

### 服务器A：对照1.8

```bash
bash scripts/run_isaaclab.sh scripts/rl_lab/ppo/train.py \
  --task=Imgo2-basemove-rough-ppo --resume \
  --load_run=2026-11-04_00-37-16_low3900_speed_terrain_resume \
  --checkpoint=model_5899.pt --run_name=airtime18_control \
  --num_envs=4096 --max_iterations=1000 --seed=42 \
  --device=cuda:0 --headless \
  env.rewards.feet_air_time.weight=1.8 \
  env.rewards.feet_air_time.params.threshold=0.5 \
  env.rewards.base_height_l2.weight=-18.0 \
  env.rewards.lin_vel_z_l2.weight=-4.0 \
  env.rewards.ang_vel_xy_l2.weight=-0.05 \
  env.rewards.feet_height_body.weight=-4.0 \
  env.rewards.feet_height_body.params.target_height=-0.20 \
  'env.commands.base_velocity.ranges.lin_vel_x=[-1.0,1.0]' \
  'env.commands.base_velocity.ranges.lin_vel_y=[-0.6,0.6]' \
  'env.commands.base_velocity.ranges.ang_vel_z=[-1.0,1.0]' \
  env.scene.terrain.max_init_terrain_level=1 \
  env.scene.terrain.terrain_generator.curriculum=true
```

### 服务器B：实验0.1

```bash
bash scripts/run_isaaclab.sh scripts/rl_lab/ppo/train.py \
  --task=Imgo2-basemove-rough-ppo --resume \
  --load_run=2026-11-04_00-37-16_low3900_speed_terrain_resume \
  --checkpoint=model_5899.pt --run_name=airtime01_test \
  --num_envs=4096 --max_iterations=1000 --seed=42 \
  --device=cuda:0 --headless \
  env.rewards.feet_air_time.weight=0.1 \
  env.rewards.feet_air_time.params.threshold=0.5 \
  env.rewards.base_height_l2.weight=-18.0 \
  env.rewards.lin_vel_z_l2.weight=-4.0 \
  env.rewards.ang_vel_xy_l2.weight=-0.05 \
  env.rewards.feet_height_body.weight=-4.0 \
  env.rewards.feet_height_body.params.target_height=-0.20 \
  'env.commands.base_velocity.ranges.lin_vel_x=[-1.0,1.0]' \
  'env.commands.base_velocity.ranges.lin_vel_y=[-0.6,0.6]' \
  'env.commands.base_velocity.ranges.ang_vel_z=[-1.0,1.0]' \
  env.scene.terrain.max_init_terrain_level=1 \
  env.scene.terrain.terrain_generator.curriculum=true
```

两命令除run_name和airtime weight一致。父类默认terrain_levels必须保留（当前本机已确认）；若服务器仍有self.curriculum.terrain_levels=None，要先按5899恢复该项，否则只设generator.curriculum=true并不会恢复环境升降级。不要残留命令行terrain_levels=null覆盖。原采样/heading、其它奖励和模型等保持5899配方；命令不能替代核对服务器源码。

## 结果如何判定

训练生成的params/env.yaml和agent.yaml用于复核：除日志名、airtime weight，不应有其它手动差异。相同课程规则下实际地形级别会随两策略表现不同而不同，因此最终必须用相同外部地形/指令测试，不能只看训练平均值。

检查静止、前后±.3/±.5、横移±.3、左右yaw±.3及粗糙/台阶：实际速度误差、每足离地/支撑时间、完整周期、净空、机身高度、roll/pitch角运动与转向平移漂移。已有policy_1/5899做额外参考，不代替新增控制组。

主要假设：减小该项后，滞空时间可能变短/频率可能更高，也可能降低不必要腾空倾向和部分角运动；不能提前断言减重一定更稳、更高或更慢。只有净空/跟速/稳定性综合改善才算有益，减少抬脚或站着不走不是改善。

不能直接排名Train/mean_reward：权重下降会直接减小该项负贡献，即使策略行为不变总奖励也会上升。例如旧airtime分项约-.1927，保持完全相同运动时权重1.8→.1会约变-.0107，这不是足端行为已改善的证据。需看物理统计与原始时间，不只看日志是否接近0。

## 验证限制

这里只准备了可复制命令并核对静态参数，未运行Hydra环境构造或服务器短训，更没有完成效果验证。续训加载当前源配置而非自动恢复旧env.yaml。跨GPU/依赖的数值差异及单seed仍是限制；1000轮是本次预算，不保证充分收敛。实验后产物应按正式play.py导出再进行同条件评测，不提前覆盖生产策略。

静态命令验证：两个代码块均通过bash -n，解析后参数数量一致，仅run_name和airtime weight两处不同；这是语法/参数对照检查，不是运行验证。

## reset_base参数说明（2026-11-04）

已确认：父类randomize_reset_base调用reset_root_state_uniform，模式为reset，只在对应环境回合重置时随机初始化机身状态。当前rough配置的pose_range为x/y±1m、z增量0、roll/pitch±0.3rad、yaw±3.14rad；velocity_range为世界系xyz±0.2m/s、角速度roll/pitch±0.05rad/s、yaw增量0。源码将位置增量加到默认位置与环境原点，姿态增量与默认姿态组合，速度增量加到默认速度；z=0不代表机身高度为零。这不是速度指令采样，也不是运行中的持续推扰。

本轮只核对配置、父类事件与本机Isaac Lab函数实现，未改配置或启动仿真/训练。待确认：这些初态扰动对最终步态的影响尚未单独实测；airtime对照实验中应保持两组reset参数一致。

## 用户指定5800起点（2026-11-04）

已核对本机该run内文件实际名为model_5800.pt，存在；训练入口支持resume/load_run/checkpoint，max_iterations表示本次追加轮数。用户指定从5800开始，以下沿用降低airtime到0.1的实验意图，其他配置由执行机器当前源码决定；若需严格两组对照，还需套用上文的共同配置覆盖项。未启动训练，服务器文件存在性、实际有效配置及训练效果待验证。

```bash
cd /root/Desktop/imgo2/imgo2_rl
IMGO2_ISAACLAB_PYTHON="$(python -c 'import sys; print(sys.executable)')" \
  bash scripts/run_isaaclab.sh scripts/rl_lab/ppo/train.py \
  --task=Imgo2-basemove-rough-ppo --resume \
  --load_run=2026-11-04_00-37-16_low3900_speed_terrain_resume \
  --checkpoint=model_5800.pt --run_name=resume5800_airtime01 \
  --num_envs=4096 --max_iterations=1000 --seed=42 \
  --device=cuda:0 --headless \
  env.rewards.feet_air_time.weight=0.1
```

/root/Desktop/imgo2为用户服务器路径示例；本机对应/home/tommy0929/Documents/imgo2。执行前激活已安装项目的Isaac Lab环境，旧run需放在该项目logs/rsl_rl/base_move_rough下。新运行生成新的日志目录，不覆盖旧run。
