# 修正模型与重新采集motion后的训练前核查

日期：2026-10-13。状态：离线模型/数据检查通过；4环境10轮GPU冒烟未运行，正式训练运行 gate 未完成。不要引用之前旧模型/旧数据的冒烟作为本次通过依据。

## 1. 当前实际输入与变更

任务 `Imgo2-basemove-flat-amp-height` 注册到AMPHeightRunnerCfg。模型是仓库根 `imgo2_description/urdf/imgo2_real.urdf`，默认动作目录是 `imgo2_rl/datasets/imgo2_motion_new`。相同文件夹名不代表内容相同，本次通过内容SHA记录数据身份，见 [模型/数据清单](data/amp_new_collection_manifest_2026-10-13.json)。

用户修正core左前/左后thigh惯量ixz；当前real两份从现有robot.xacro重新生成与磁盘文件逐字节一致。新采集的21段已覆盖当前RL目录，19份同名重采，原forward/backward_1.0删除并新增1.2；错误模型录制的数据保存在旁边wrong_model目录，不会被当前目录的glob加载。数据目录内只有21份txt，没有其它混入文件。

外层 `imgo2_dataset/datasets/imgo2_motion_new` 仍为旧副本：共有19个同名文件内容全部不同，其1.0片段与RL的1.2片段也不同。当前默认训练会加载RL新目录，外层副本不参与；服务器若指向外层目录或保留旧IMGO2_AMP_MOTION_DIR会读错数据。两个新增1.2文件尚未跟踪，需在服务器同步时明确包含，不能以当前远端clone已更新为假设。本轮未修改或复制用户数据。

## 2. 已完成的离线检查

解释器 `/usr/bin/python3`（系统Python3.10）；xacro/check_urdf为ROS2 Humble工具。

- 两份real URDF结构解析通过，17个本体link物理一致、仅Gazebo多IMU；所有惯量正定/三角不等式与正质量通过。见 [物理核对](data/amp_new_collection_model_physics_2026-10-13.json)。有效性检查不替代实机/CAD惯量辨识。
- 21段×600帧=12600帧，61列均有限，四元数最大范数误差4.44e−16；所有MotionWeight=1且无参考关节超限。
- 当前real URDF与21段足端坐标identity FK RMSE均0；交叉腿序约0.225m。独立hip左右反对称及关节q/dq相关核对通过，平均同腿相关0.958、交叉腿0.177，支持FL/FR/RL/RR恒等映射。见 [FK报告](data/amp_new_collection_fk_2026-10-13.json)。
- dt全部0.02s；物理200Hz、控制50Hz、rollout24步=.48s；专家相邻状态对与控制间隔一致。见 [当前目录时间报告](data/amp_new_collection_timing_2026-10-13.json)。已为check_amp_timing.py增加--motion-dir，原默认仍保留；本次明确指定新目录，避免检查旧目录却误称新数据通过。
- 实际均速与标签大体吻合：前进1.2约1.185m/s、后退1.2约−1.199；左右侧移0.4约±.399；纯yaw1约±1rad/s。存储body速度与根位置差分RMSE约0.0002–0.0121m/s。见 [方向/速度](data/amp_new_collection_directions_2026-10-13.json)。
- stance最大关节角范围0.00326rad、关节速度RMS0.00375rad/s，未见参考片段本身明显抖动。原始采集时间戳/接触力/是否真实卸载仍不能由这些离线数值证明。
- check_asset_paths通过：real模型、17mesh引用与新目录21文件存在。全仓check_model_sync仍因旧网格指纹与real/real_wrong四份未登记失败（MODEL-05），不是本次real物理核对失败。

## 3. 配置风险与实验选择

当前Height范围vx±1.2、vy±1、yaw±1.2；速度课程关闭。专家前后覆盖1.2，侧向仅0.3/0.4，yaw仅0.5/1。vy±1尤其明显超出专家侧移覆盖，建议第一轮先vy±0.4、yaw±1，vx可保留±1.2作为全范围基线。AMP不要求所有指令逐帧配对，因此超出覆盖不等同启动错误，但可能增加困难和风格/任务冲突。本轮未擅自修改用户范围。

若目标仍是先低速/减少抖动，当前范围不对应之前低速方案，而且只有原六项reward，静止抽样仍2%，0.2m/s平移死区仍存在。需另行实现低速分组指令和静止/平滑、净空/防滑约束，见 [具体方案](amp_low_speed_finetune_plan_2026-10-13.md)。不能因为重采了motion就宣称这些目标会自动得到。

当前训练P25/D0.5，部署24/0.6；该差异不阻止本次训练构建，但新策略sim2sim评价时要固定并记录两边实际增益。初次基线不同时修改增益、所有reward和判别器损失。

建议**从头建立新运行，不加--resume**：当前motion内容与模型动力学已更新，默认resume会继承旧判别器/归一化/优化器，不能仅凭61列与文件名相同认定完整恢复适合。actor迁移是独立实验，本轮未设计/验证。新运行记录model/data SHA。

## 4. 运行验证限制与下一步命令

本轮尝试两次申请权限运行4环境10轮CUDA短训练（本机env_isaaclab、seed42、从头、headless），均在启动进程前被自动权限审核超时拒绝。返回理由是审核未在deadline前完成，并非识别到具体危险行为；因此训练进程未启动，没有新run/checkpoint，没有运行或缓存刷新验证。按工具允许重试一次后停止额外申请，不能标为冒烟通过。

下面用于用户终端或之后获批准的短测；先确认实际保存配置与加载21个文件，再决定长期训练。force_usd_conversion字段在现有保存配置中已确认，用于明确重新转换模型；完整命令尚未实跑。

```bash
cd /home/tommy0929/Documents/imgo2/imgo2_rl
export IMGO2_ISAACLAB_PYTHON=/home/tommy0929/miniconda3/envs/env_isaaclab/bin/python
export PYTHONPATH="$PWD/source/imgo2_rl:$PWD/scripts/rl_lab${PYTHONPATH:+:$PYTHONPATH}"
unset IMGO2_URDF_PATH IMGO2_AMP_MOTION_DIR

bash scripts/run_isaaclab.sh scripts/rl_lab/amp/train.py \
  --task=Imgo2-basemove-flat-amp-height \
  --num_envs=4 --max_iterations=10 --seed=42 \
  --run_name=corrected_model_new_motion_smoke \
  --device=cuda:0 --headless \
  env.scene.robot.spawn.force_usd_conversion=true
```

验收：actor45、critic48、AMP43、动作12、专家21文件；4env10轮完整结束、无非有限loss或报错、保存params与model_10；检查模型实际转换路径/新惯量和范围yaml。10轮从头训练不能评估成熟步态，但能验证构建、奖励/判别器更新和存档。

## 5. 状态

2026-10-14补充：当前本地AMP Height仍启用base_height_l2，目标0.30m；每步系数−5，weight=−5/step_dt（当前−250）。平地sensor_cfg=None，读取base/root link世界z与目标的平方偏差，并有直立姿态门控。修改目标位置在amp_env_cfg.py:261，系数在244行。该项约束机身高度，不能代替足端净空奖励；本轮仅源码核对，未更改数值，服务器实际保存配置/日志与短训练仍待确认。

已修复/完成：本次模型/数据清单、路径、FK/腿序/速度/时间与超限检查；时间检查工具支持显式当前目录，避免旧目录误验证；文档依据已更新。

待确认：新模型/新采集AMP短测（权限审核超时，需本机/服务器实跑或明确授权后再尝试）、USD缓存/运行惯量、新策略收益、侧向范围选择、低速方案实现。外层旧副本与新增文件服务器同步待DATA-01；全模型登记/网格仍MODEL-05；增益/静止行为仍AMP-23/DEPLOY-14。
