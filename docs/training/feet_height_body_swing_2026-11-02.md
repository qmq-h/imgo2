# feet_height_body 仅摆动脚修改与回退

核对日期：2026-11-02。用户授权实际修改本机实现，要求旧函数整段注释、新同名函数紧邻下方。未修改或启用服务器代码、权重、目标、指令、模型、PD、部署策略，未启动训练。

## 现象与修改

旧函数在机身系计算足端高度误差，并以减去机身平移速度后的足端水平速度加权，没有支撑/摆动掩码。正常支撑足也可能被拉向摆动高度。新函数的原始正代价为：

`Σ_i [current_air_time_i>0] × (z_foot_body_i-target_height)² × tanh(tanh_mult × ||v_foot_world_xy_i||)`

沿用指令三维范数>0.1（纯转向包含在内）和直立姿态门控；RewardManager另乘weight和控制dt。应使用负weight。

- 足高仍为机身参考坐标，不是离地净空；没有新增地形射线或相位时钟。
- 支撑脚掩码为0，使用当前滞空计时而非历史接触力；接触传感器必须track_air_time=True。
- 新增sensor_cfg参数；父velocity_env_cfg.py已补contact_forces和.*_FOOT选择器。函数另带相同默认参数，已有自定义RewTerm缺参数时可由默认值解析。
- 传感器与机器人body_ids不能混用；按选择到的精确足端名称映射到传感器索引，缺足端时报错，支持两者排序不同。
- 改动位于公共mdp函数，任何实际启用该函数的任务都会采用新行为，不限于PPO。him_feet_height_body未改。

当前本机rough_env_cfg.py仍为weight=-0.0、target_height=-0.20，零权重会被禁用。此次仅按授权加入实现，没有擅自决定非零训练权重或启用它。将来开启时需确认最终params/env.yaml的feet_height_body不是null，且asset_cfg/sensor_cfg选中相同足端。target_height=-0.20并非抬脚20cm；需结合机身高度、足端原点/碰撞半径理解。

## 回退

rewards.py中搜索 `BEGIN LEGACY feet_height_body`，整段旧函数在新同名函数上方。回退时取消该块每行开头的注释，同时注释新函数，确保仅一个同名函数生效。若仍保留配置中的sensor_cfg参数，恢复旧签名时还需移除这一参数（父配置和自定义配置均需检查）；也可临时将恢复后的旧签名增加未使用的可选sensor_cfg以兼容调用。不应同时保留两个有效定义。

## 已验证

解释器：`/home/tommy0929/miniconda3/envs/env_isaaclab/bin/python`，CPU torch、实际函数AST与模拟scene，无Isaac导入。

- check_feet_height_body.py：10项通过，覆盖全支撑0、仅摆动贡献、传感器反序映射、支撑足高度不影响代价、机身平移速度不影响权重、纯转向、零指令、零世界足速、目标高度、缺传感器能力/足端时报错。
- 修改前原函数与注释还原逐字一致；check_velocity_reward_fixes.py --baseline-ref HEAD通过，三处历史块还原后整个rewards.py AST与当前HEAD一致。HEAD基准仅适用于本次未提交版本，未来提交后需指定修改前提交。
- 父配置语法检查通过（原文件含UTF-8 BOM，按utf-8-sig读取）。

机器结果：[CPU案例](feet_height_body_swing_2026-11-02_tensor.json)、[语法与回退](feet_height_body_swing_2026-11-02_structure.json)。工具在imgo2_rl/scripts/tools/，可在具备torch的Python中复跑。

## 已修与待验证

已修：实现排除支撑脚、去掉相对机身足速加权、显式足端名称对齐，旧代码完整保留。

待确认：当前实现是否改善净空/高度/转向，需有可用Isaac环境进行构造、短训和同条件策略评估。本机NVIDIA驱动不可用，未作这些运行验证。仅摆动惩罚可以被“不开始摆动”规避，且足速接近0时约束很弱；不能保证主动抬脚、对角配合或特定步频。机身高度/倾斜仍影响目标；摆动全程使用同一高度目标，未实现只作用于中段的相位约束。若进行训练试验，先固定其他项，从小负权重开始并检查分项贡献，而不原样恢复历史大权重。既有复核中的“未实施方案”段落为修改前历史，以本记录为准。


## 2026-11-02 启用后采样变慢：缓存修复（服务器计时待验证）

用户服务器日志：4096环境、24步（日志每轮约98304 transitions），collection=29.749s、learning=0.090s、3294 steps/s。4090显存7831/24564MiB、利用率4%、75W为单次快照，不足以单独证明瓶颈，但与主机侧等待一致。terrain_levels=3.22确认该次训练课程实际开启；此次足高修改和此前位置解释没有修改地形课程开关。没有相同配置的前后计时证据，不能断言地形对耗时无影响。

发现新增足高函数每步重复调用contact_sensor.body_names，该属性在本机IsaacLab实现中访问body_physx_view.prim_paths并构造名称列表；此前slice选择器还为求len重复调用，逐个选中索引又重复调用。这是本轮引入的性能缺陷和当前首要嫌疑，但尚不能把29.749s全部归因于它。此前CPU数值案例无法发现真实PhysX属性访问成本。

已改：首次调用时取得asset/sensor名称、验证并缓存匹配索引；稳态仅查询缓存，名称不再在每个控制步获取。缓存位于env，键含实体身份和所选body_ids；选择器变化会重建，普通episode reset不失效。奖励公式、门控、目标及权重不变。

新增回归检查：100次稳态调用零新增body_names读取；更改选择顺序仍正确映射。共12项CPU检查通过，原函数完整保留和回退AST检查仍通过。未实跑服务器Isaac计时，本机NVIDIA不可用。需把最新rewards.py同步到服务器，停止旧进程后重启同一命令（Python不会自动热更新），跳过最初初始化若干轮再比较collection/learning。无需重新安装editable项目。若仍慢，再同配置仅禁用feet_height_body短测并排查CPU配额、渲染、其他进程和地形/重置；不先改物理步长、环境数或奖励权重来掩盖问题。此次不应因性能排查而声称步态已修复。
