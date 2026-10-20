# AMP 第 24000 轮查看记录（2026-09-24）

## 已解决与验证

目标 run：`imgo2_rl/logs/amp_rsl_rl/base_move_amp_height/2026-09-24_00-57-38_xiaoji_real1`。
`model_24000.pt` 存在，SHA256：`c5c934842e8e2af3aef806b5c34abcd41d8cdf053ed2af410e4fd6ceb498a70e`。

现象：启动器默认 `/opt/conda/envs/isaaclab/bin/python` 不适用于本机。
处理：命令中通过既有 `IMGO2_ISAACLAB_PYTHON` 覆盖，不修改启动器。
本机解释器 `/home/tommy0929/miniconda3/envs/env_isaaclab/bin/python` 的模块发现检查确认 torch、tensorboard、isaaclab、imgo2_rl、rl_lab、onnx、pybullet_utils 可定位（不代表完整导入与运行通过）。`python -m tensorboard.main --version` 实跑返回 2.21.0；提示未安装 TensorFlow，可使用精简功能读取标量。
训练保存的 `params/amp_env_cfg.py` 与当前源码逐字一致。回放加载当前任务配置，不会自动恢复整个 params/env.yaml；该 YAML 的模型路径是原训练机路径。

## 查看命令（本机路径）

终端一：

```bash
conda activate env_isaaclab
cd /home/tommy0929/Documents/imgo2/imgo2_rl
python -m tensorboard.main --logdir logs/amp_rsl_rl/base_move_amp_height/2026-09-24_00-57-38_xiaoji_real1 --host 127.0.0.1 --port 6006
```

浏览器打开 http://127.0.0.1:6006 ，在 Scalars / Time Series 选 Step；先看原始值，再用 smoothing 观察趋势。TensorBoard 读取 event，不读取 .pt。`Train/*/time` 的 step 实际为秒，不能用于定位第 24000 轮。

终端二（本机桌面）：

```bash
conda activate env_isaaclab
cd /home/tommy0929/Documents/imgo2/imgo2_rl
export IMGO2_ISAACLAB_PYTHON=/home/tommy0929/miniconda3/envs/env_isaaclab/bin/python
bash scripts/run_isaaclab.sh scripts/rl_lab/amp/play.py \
  --task Imgo2-basemove-flat-amp-height-play \
  --num_envs 1 --real-time \
  --checkpoint /home/tommy0929/Documents/imgo2/imgo2_rl/logs/amp_rsl_rl/base_move_amp_height/2026-09-24_00-57-38_xiaoji_real1/model_24000.pt
```

不加 `--headless` 才显示 GUI。脚本自动启动 Isaac Sim，无需单独打开后导入 .pt。play 配置关闭参考初始化与所列随机化，并设 vx=1、vy=0、yaw rate=0。脚本还会在 run/exported 下写入或覆盖 policy.pt 和 policy.onnx。

## 离线日志结果与限制

使用 `/usr/bin/python3` 实跑仓库 `read_tfevents.py --run <上述目录> --steps 10000,20000,24000 --match mean_reward,mean_episode_length,mean_root_height,fraction_root_height,error_vel,Episode_Termination,disc_expert_pred,disc_policy_pred`。
34 个 tag，迭代轴最后 step=23999；请求 24000 时工具取不超过目标的最近点。

末点：高度 0.3062 m；高度低于 0.20 m 比例显示 0.0000（四位小数，非严格零证明）；illegal_contact=0.0062、time_out=0.9938；error_vel_xy=0.4507、error_vel_yaw=0.4453；mean_episode_length=1000；mean_reward=1819.1472。判别器 expert/policy 预测为 0.6539/-0.6532，不能将判别器 loss 或得分单独视为步态验收。

## 待修复／待确认

- 本轮没有代码修复；已解决的是查看命令与本机解释器路径说明。
- 未启动 TensorBoard 服务或 Isaac Sim GUI，未做 checkpoint 运行时加载、导出与步态验收。需要用户在本机桌面执行上述命令，观察是否跌倒、拖脚、抖动、侧偏及跟速；日志稳定性不等于动作自然。
- Git 工作区已有用户修改的 core.xacro、髋部 mesh、新 real URDF 与旧模型目录，本轮保持原样。需要训练时模型快照/哈希才能确认与当前回放资产完全一致，不能仅凭配置源码相同认定模型一致。
- `git pull --ff-only` 返回 Already up to date；仅更新根 README 与本记录。

## 补充：训练与回放模型引用核查

已确认：AMP 环境通过 `IMGO2_CFG` 使用 `assets/imgo2.py` 的 `URDF_PATH`，默认是仓库根 `imgo2_description/urdf/imgo2.urdf`。本机完整路径为 `/home/tommy0929/Documents/imgo2/imgo2_description/urdf/imgo2.urdf`，没有默认切换到 `imgo2_real.urdf`。`IMGO2_URDF_PATH` 可以覆盖它；本次检查进程没有设置该变量，用户另一个终端的环境未检查。

指定 run 的 `params/env.yaml` 实际记录 `/root/Desktop/imgo2/imgo2_description/urdf/imgo2.urdf`，由 `spawn_from_urdf` 导入。因此可以确认训练时引用的路径，不能据此确认当时文件内容与当前本地内容相同。修改 core.xacro 不会自动更新已生成 URDF；修改 URDF 引用的网格可能影响后续导入。

已解决：明确默认模型和历史训练路径；待确认：训练时模型/网格内容哈希、用户实际运行进程是否使用路径覆盖。本轮仅静态读取源码与保存配置，未启动仿真。

## 后续复核：默认路径已切换 real（同日，以本节为最新结论）

用户再次询问后重新读取源码，`assets/imgo2.py:20` 已改为 `imgo2_description/urdf/imgo2_real.urdf`，取代上节核查时的默认值。新启动的 AMP 训练/回放在没有 `IMGO2_URDF_PATH` 覆盖时会使用 real。已确认路径配置变更；未发现可读取的 train.py/play.py 运行进程，未进行仿真加载验证，不能断言已有进程已切换。旧 checkpoint 的历史训练模型不因此改变。待确认项仍是运行时加载与历史模型一致性。

## Isaac Sim 电机、摩擦与时间步配置位置

本节静态核查 `Imgo2-basemove-flat-amp-height`；其它算法需单独检查覆盖。

- `assets/imgo2.py` 的 `IMGO2_CFG.actuators["legs"]`：DCMotorCfg，stiffness=25、damping=0.5、effort_limit=23.7、saturation_effort=23.7、velocity_limit=30.1、friction=0。joint_drive 中的 0/0 是 URDF 导入驱动配置，不是策略所用 DCMotor PD 增益。rigid_props 的 linear_damping/angular_damping 是刚体阻尼，也不同于关节 Kd。
- `velocity_env_cfg.py:48`：地面 physics_material，静/动摩擦均 1.0，restitution=1.0，摩擦与恢复系数 combine_mode 均 multiply；第 720 行也将该材料配置赋给 sim.physics_material。
- `velocity_env_cfg.py:257`：机器人所有 body 材料在 startup 随机化，静摩擦 0.3–1.0、动摩擦 0.3–0.8、恢复系数 0–0.5，64 buckets。所以训练中不能把地面摩擦 1.0 当作唯一接触配置。
- `velocity_env_cfg.py:332`：reset 时对电机 Kp/Kd 分别按 uniform 0.5–2.0 倍缩放；名义 25/0.5 对应范围 12.5–50 与 0.25–1.0。
- `amp_env_cfg.py:231` 的 apply_amp_play_overrides 关闭材料、增益、质量等随机化；因此训练和 play 的参数分布不同。仅关闭随机化不意味着足部与地面的接触系数可直接由训练采样区间推断。
- `amp_env_cfg.py:128`：动作缩放 hip=0.125，其余=0.25，另有 clip 配置。
- `velocity_env_cfg.py:715`：decimation=4、sim.dt=0.005，物理步长 5 ms，策略周期 20 ms（50 Hz）。
- URDF 决定基础质量、惯量、关节轴/限位、碰撞几何，训练事件还可能随机化质量等。指定 run 的 params/env.yaml 保存训练时展开配置；本次核对其中名义 Kp/Kd/friction 与材料值相同。

已解决：定位参数层级与训练/play 覆盖关系；无代码修改。待确认：当前 real 模型运行时加载和实际接触材料/随机采样结果，需构造仿真并读取物理视图；本轮未启动仿真。
