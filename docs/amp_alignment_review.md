# AMP 贴地爬行：对齐检查与参考项目对照

日期：2026-09-15。当前用户反馈：PPO 已完成训练和 sim 验证；AMP 有步态但贴地爬行，初始化姿态不是策略生成的。检查针对当前本地源码与数据，不等同于已复现训练结果。

## 1. 结论与修改范围

本地发现三组问题：

1. 已检查的本地数据几何支持 `FL, FR, RL, RR`。用户补充实际采集模型、脚本，并确认未覆盖默认关节/足端参数，进一步确认文件写入顺序为 LF/RF/LH/RH，等价于 FL/FR/RL/RR。原 `joint_mapping` 会额外交换 FR/RL，造成参考数据与判别器观察不一致，也影响参考重置。
2. AMP 归一化统计更新用错数据尺度；梯度惩罚与判别器分类输入也不在同一尺度。
3. 环境删掉高度奖励和非法接触终止，且任务奖励因 `dt` 和混合系数而远小于 AMP 奖励。

前两项已有代码修改，当前数据的恒等映射也已结合实际采集参数确认适用。第三项已配置一组可供新训练验证的高度约束、基座触地终止和奖励权重。它们共同具备造成异常训练的条件，但没有 checkpoint 与仿真运行证据，不能认定其中某项是贴地爬行的唯一根因。

## 2. 61 维数据如何进入 43 维判别器

切片采用 Python 的左闭右开表示法。

| 数据文件切片 | 含义 | AMP 观察切片 | 检查结论 |
|---|---|---|---|
| `0:3` | 世界系 root 位置 | 仅 z 进入 `42:43` | 本地 z 为 0.2787–0.3335 m，没有整体归零 |
| `3:7` | root 四元数，xyzw | 不直接进入判别器 | 重置代码转换为 Isaac Lab 的 wxyz；用于足端坐标、速度转换 |
| `7:19` | 绝对关节位置 | `0:12` | 判别器用绝对关节角，actor 用相对默认角度；两者用途不同，并非错位 |
| `19:31` | base 系足端位置 | `12:24` | 与训练 URDF 正运动学一致；足端位置需和关节使用同一腿顺序 |
| `31:34` | base 线速度 | `24:27` | 录制脚本直接使用 odometry twist，假设其为 base 系；未核验原始 ROS 发布者 |
| `34:37` | base 角速度 | `27:30` | 与仿真 `base_ang_vel` 的期望坐标系一致；原始 ROS 数据源仍待核实 |
| `37:49` | 关节速度 | `30:42` | 和关节位置采用相同排列与映射 |
| `49:61` | 足端局部速度 | 不进入判别器 | loader 最终截取到第 49 列；该字段未参与当前 AMP reward |

单帧 AMP 为 43 维，判别器拼接当前帧和下一帧后为 86 维。loader 和环境的字段拼接顺序一致。root 高度已进入 AMP，缺少的是独立、明确的高度任务约束，不是高度维度完全遗漏。

`amp_root_z()` 当前取世界坐标 z；平地任务地面为 z=0 时成立。若以后迁移到有高度偏移的地形，应同时调整参考高度、重置高度和在线高度定义。

## 3. joint_mapping：模型声明顺序与数据写入顺序

### 3.1 用户补充的真实采集链路

实际模型为 `imgo2_description/`，`xacro/robot.xacro` 包含 `urdf/imgo2_description.urdf`。该 URDF 的腿声明顺序确实是 **LF、LH、RF、RH**，即 FL、RL、FR、RR。模型中的 LH_HAA 位于 base 的 `(-0.1575, +0.067, 0.00025)`，对应左后腿，名称没有反置。

实际采集脚本为 `C:/Users/qmq/Desktop/RL/AMP/amp_go2-main/datasets/record_legged_control_amp.py`。它不按 URDF 文本顺序或收到的 JointState 数组顺序直接写入，而是执行：

```python
name_to_idx = {name: i for i, name in enumerate(msg.name)}
self.joint_index = [name_to_idx[name] for name in self.args.joint_names]
```

随后按 `self.joint_index` 写关节位置/速度；足端则独立按 `self.args.foot_frames` 的顺序查询 TF 并写入。

| 环节 | 顺序来源 | 当前源码值 |
|---|---|---|
| 采集 URDF 声明 | `imgo2_description.urdf` | LF、LH、RF、RH |
| 文件关节块 | 实际 `--joint-names` 参数 | 默认 LF、RF、LH、RH，每腿 HAA/HFE/KFE |
| 文件足端块 | 实际 `--foot-frames` 参数 | 默认 LF_FOOT、RF_FOOT、LH_FOOT、RH_FOOT |
| 训练端关节选择 | `joint_names` 与 `preserve_order=True` | FL、FR、RL、RR，每腿 hip/thigh/shank |
| 训练端足端选择 | `foot_body_names` 与 `preserve_order=True` | FL、FR、RL、RR |

因此：若实际采集使用默认值，文件已是 FL/FR/RL/RR，使用恒等映射；若两个参数都改成 LF/LH/RF/RH，原 FR/RL 交换映射有意义；若只改了其中一个参数，关节和足端就不能共用同一个映射。

用户已确认实际采集没有传入 `--joint-names` 或 `--foot-frames`，使用默认值。因此当前文件的关节块和足端块均为 LF/RF/LH/RH，恒等映射适用。此前仅展示足端坐标就概括整份数据顺序不够完整；现在由实际采集模型、按名称重排的脚本、用户确认的启动参数，以及此前本地数据的运动学结果共同支持这一结论。本轮仅补充源码追踪与用户确认，没有运行测试或再次改动映射。

### 3.2 原映射及此前离线证据

原映射是 `amp_env_cfg.py` 里的一个 4 元按腿排列：

```python
amp_leg_mapping = [0, 2, 1, 3]
```

它把腿块顺序换成 `[FL, RL, FR, RR]`，即交换 FR 与 RL；等价的 12 维平铺写法是 `[0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11]`。当时的观察辅助函数 `_apply_leg_mapping()` 用 `data[:, mapping, :]` 在腿轴（4 个元素）上索引，因此与这个 4 元列表配套；本轮改成 `_apply_flat_mapping()` 在 12 维平铺轴上索引，配合 12 元恒等映射。也就是说映射的粒度和取值形式同时变了，两者必须成对理解。

本地 loader 的 `reorder_from_pybullet_to_isaac()` 虽然带着「PyBullet → Isaac」的说明，实际拼接顺序一直是原样输出（本轮只是删掉了这个错误说明并把解包变量名改正），所以参考端始终是 `[FL, FR, RL, RR]`，而在线 AMP 观察被那个 4 元腿映射交换了 FR 与 RL。

用户提到的 reward 侧调整可以定位到这个变量：它作用于判别器观察的关节角、关节速度和足端位置，并以逆映射作用于参考重置。当前保留的两个速度跟踪 reward 只读取基座速度、指令和重力投影，没有再将腿顺序换回。给这些 reward 调权重也不能修正判别器输入的排列。

用训练 URDF 对全部 5097 帧做正运动学验证，逐坐标 RMSE 为：

| 假设的数据腿顺序 | 足端坐标 RMSE，按每份动作统计 |
|---|---|
| `FL, FR, RL, RR` | 约 0–0.002143 m |
| `FL, RL, FR, RR` | 约 0.225 m |

站立片段中，参考足端原点的世界 z 均值约 0.022008 m；URDF 推算约 0.022011 m。这里比较的是足端 link 原点，并非碰撞表面的触地点，不应强行把它改成零高度。

当前已将 `joint_mapping` 改为 `list(range(12))`，观测与参考重置共用这个约定。数据内容和 URDF 没有被改写。详细逐动作统计见 [amp_data_audit.json](amp_data_audit.json)，可通过 [audit_amp_dataset.py](../Imgo2_rl/scripts/tools/audit_amp_dataset.py) 重现。

### 3.3 追加的独立验证（不依赖采集参数回忆）

第 3.1 节的结论此前部分依赖用户对采集启动参数的确认。补充三项检查后结论不变，但依据更强：

1. **改用实际采集模型复算正运动学**：同一份 21 动作 / 5097 帧数据分别对训练 URDF 和实际采集模型 `imgo2_description/urdf/imgo2_description.urdf`（SHA256 `5e844fb8896960ef…`，足端链名 `LF_FOOT` 等）做 FK。按 `LF,RF,LH,RH` 配对时逐坐标 RMSE 均值 0.00107 m、最大 0.00214 m；按该 URDF 的声明顺序 `LF,LH,RF,RH` 配对时均值 0.22491 m。两个 URDF 给出完全相同的数值，说明它们在下肢运动学上等价。检查原脚本写死了训练端足端链名，本轮改为按采集模型链名运行，其余逻辑未改。
2. **与 URDF 无关的左右对称性**：准对称支撑时同侧对的髋外展角应互为反号。按 `FL,FR,RL,RR` 读，21 份动作的 `|hip_FL+hip_FR|` 与 `|hip_RL+hip_RR|` 均值分别为 0.0173 / 0.0208 rad；按 FR/RL 交换读则为 0.1204 / 0.1474 rad。左右腿标注因此被独立确认，这一步不使用 URDF，也不使用 FK。
3. **关节速度块排列**：对关节位置块做中心差分，与 `37:49` 的关节速度块逐通道求相关。同块配对 12 通道平均 r 为 0.95（thigh/shank 约 0.98，hip 因幅值仅约 0.5 rad/s 为 0.87–0.93）；FR/RL 交换后降到 0.76。位置块与速度块使用同一排列，块内 hip/thigh/shank 次序同时得到确认。

关节块与足端块来自两个独立 ROS 源（JointState 与 TF），二者能相互对上 1 mm 量级，不是同一个错误在自我印证。因此“文件内容是 `FL,FR,RL,RR`，等价于采集侧 `LF,RF,LH,RH`”可以由数据本身和采集模型证明，不必依赖对采集参数的回忆；`joint_mapping = list(range(12))` 对当前数据成立。

这三项检查可由 [check_amp_joint_order.py](../Imgo2_rl/scripts/tools/check_amp_joint_order.py) 重现，只用标准库，三项全部通过时返回 0。

一个结构性限制仍然存在：`amp_foot_pos_base` 复用的是同一个 12 维 `joint_mapping`，它只在关节块与足端块腿顺序一致时才能同时排对两者。若将来数据的两块顺序不同，需要把关节映射和足端映射拆开。

## 4. 与本地参考项目的对照

参考目录：`C:/Users/qmq/Desktop/RL/AMP/`。此次仅阅读参考项目。

| 项目 | Imgo2 修改前 | AMP_for_hardware-main：A1 | amp_go2-main：Go2 |
|---|---|---|---|
| 高度任务奖励 | 被过滤掉 | `base_height=0`，虽有 target=0.25 但不生效 | `base_height=-1`，target=0.38 |
| 接触终止 | `illegal_contact=None` | base、thigh、calf 触地终止 | base 触地终止 |
| 线/角速度权重 | `1.0 / 0.3`，之后乘 0.02 | `1.5 / dt`、`0.5 / dt`，抵消环境奖励的 dt | `4.0 / 2.0`，配合较小 style 系数和较大 task 混合系数 |
| AMP 系数 | `2.0` | `2.0` | `0.2` |
| task 混合系数 | `0.1` | `0.3` | `0.8` |
| 均值方差更新 | 已归一化数据，错误 | 同样存在该问题 | 使用未归一化数据 |
| 梯度惩罚输入 | 原始数据；分类用归一化数据 | 同样存在尺度不一致 | 使用与分类一致的归一化数据 |

不能仅依据 A1 关闭高度项，就认为所有 AMP 任务都不需要高度约束：该实现仍使用严格的身体触地终止和不同的任务奖励量级。也不能直接照搬 Go2 的 0.38 m 高度，它不符合当前 Imgo2 数据约 0.30 m 的高度。

主要参考文件：

- `AMP_for_hardware-main/legged_gym/envs/a1/a1_amp_config.py`
- `AMP_for_hardware-main/legged_gym/envs/base/legged_robot.py`
- `AMP_for_hardware-main/amp_rsl_rl/rsl_rl/algorithms/amp_ppo.py`
- `amp_go2-main/legged_gym/legged_gym/envs/go2/go2_amp_config.py`
- `amp_go2-main/legged_gym/legged_gym/envs/go2/go2_amp.py`
- `amp_go2-main/rsl_rl/rsl_rl/algorithms/ppo.py`
- 两个项目各自的 `rsl_rl/datasets/motion_loader.py`。

两个参考 loader 针对 PyBullet 数据执行的是 `[FR,FL,RR,RL] → [FL,FR,RL,RR]`，不能直接套在已经按 FL/FR/RL/RR 录制的 Imgo2 数据上。

## 5. 奖励量级与当前改动

Isaac Lab `RewardManager.compute(dt)` 实际计算 `term × weight × dt`。Imgo2 策略周期为 0.02 秒，而 AMP 判别器直接产生每步奖励，没有乘这个 dt。

修改前，在姿态正常且速度完全跟踪时：

```text
任务部分最大贡献 = 0.1 × 0.02 × (1.0 + 0.3) = 0.0026
AMP 部分最大贡献 = (1 - 0.1) × 2.0 = 1.8
```

这是理论上限的比较，并非训练日志的实测均值，但足以说明混合系数 0.1 不代表实际奖励中任务约占 10%。

现在环境每步的任务奖励为：

```text
r_task = 1.0 × r_vxy + 0.3 × r_wz - 10 × (z - 0.30)^2 × 姿态门控
r_total = 0.7 × r_style + 0.3 × r_task
```

高度项复用现有 `base_height_l2()`，其姿态门控与速度项类似；平地不读取高度扫描传感器。配置 weight 除以 `step_dt`，因此在当前周期下为 `50`、`15`、`-500`，环境乘 dt 后才得到上述每步系数。

速度完全跟踪时，任务部分最大贡献变为 0.39，AMP 部分最大贡献为 1.4。若身体保持水平，高度从 0.30 m 降到 0.10 m，高度项在混合奖励中减少 0.12。这个强度是待验证的起始值；不能单凭公式认定它足以消除贴地行为。

同时恢复基座触地终止，避免基座长期贴地仍持续累计奖励。没有添加低高度强制终止；日志中 0.20 m 只是监测阈值。

## 6. 其他已检查和仍待验证的环节

- **参考初始化**：使用数据四元数 xyzw→wxyz，再把 base 速度旋转到世界系；关节使用观察映射的逆映射。顺序已改为恒等。初始化正常不是策略训练成功的证据。
- **默认关节角与数据数值**：`IMGO2_CFG.init_state` 为 hip `0.0`、thigh `0.87`、shank `-1.82`，用关节名正则匹配，与腿排列无关。数据全帧均值 thigh `0.906`–`0.914`、shank `-1.805` 至 `-1.821`，与默认值相差不超过 `0.044` rad；hip 均值在 `-0.090` 至 `+0.106` 之间（各动作含转向，均值不为零），相差不超过 `0.106` rad。数值上没有量级错位。参考重置概率为 1.0，会覆盖默认姿态。
- **观察与参考的映射方向**：`_apply_flat_mapping(data, mapping)` 取 `data[:, mapping]`，语义是“输出槽 i 取仿真顺序的 `mapping[i]`”，即 `mapping` 表示文件槽到仿真槽；参考重置写回仿真顺序，必须用逆映射，`amp_events.py` 正是如此。恒等映射下两者都退化为恒等，非恒等映射时该约定也自洽。注意改动前的辅助函数是在腿轴（4 元）上索引的，替换数据集或恢复旧映射时要同时确认映射的元数（4 还是 12），否则会索引越界或静默排错。
- **足端观察**：世界足端位置先减 root 位置，再用 root 旋转的逆变换转回 base 系；离线数据与 URDF 对齐，不支持任意添加高度偏移来掩盖问题。
- **任务观察与动作**：actor/critic 的关节顺序由明确的 `joint_names` 控制，动作同样使用该列表；AMP 用绝对关节角，actor 用相对默认值是有意设计。
- **terminal AMP 状态**：环境在 reset 前读取 terminal AMP，runner 用它替换已经 reset 的下一状态，避免判别器把重置跳变当作真实运动。
- **速度 reward**：只约束速度并用重力投影约束朝向，不直接惩罚低高度。身体水平的爬行仍可能获得速度奖励。
- **时间插值**：loader 使用 `time / ((n-1)×dt) × n` 取帧，存在 n 与 n-1 的小时间尺度偏差；此次没有修改。它是另一个可单独修正的事项，当前证据不支持把它当作大幅失高的主因。
- **实际运行差异**：尚无当前 checkpoint、完整启动命令和日志，无法证明训练服务器使用了哪份配置，也未验证原始 odometry twist 的坐标约定。

## 7. 验证与下一步

已完成：

- 21 份动作、5097 帧的离线运动学与高度统计。
- 四项离线回归通过：旋转方向、URDF 零角度腿长、数据与 URDF 顺序、不同 dt 下高度项保留及每步奖励量级。本轮重新运行 `python -m unittest discover -s tests -p test_amp_alignment.py -v`，结果为 4 项通过、1 项按设计跳过。
- 六个修改或新增 Python 文件语法检查通过。
- 用实际采集模型 `imgo2_description.urdf` 复算 FK，与训练 URDF 结果一致（见 3.3）。
- 两项不依赖 URDF 的关节排列检查：髋外展左右对称性、关节位置块与速度块逐通道相关性（见 3.3）。
- 关节角数值与默认姿态的对照（见第 6 节）。

未完成：

- Torch CPU 更新回归已编写，但本机解释器缺少 torch/numpy，测试仍被跳过。本机可用的解释器是 `C:\Users\qmq\AppData\Local\Python\pythoncore-3.14-64\python.exe`（可由 `py -3` 启动）；PATH 上的 `python` 别名在本会话返回退出码 9009，不能用于运行检查。
- 未运行 Isaac Lab 环境，未重训、回放或验证高度改善。
- 关节排列与默认姿态的对齐已确认，但不构成贴地爬行原因的结论：训练服务器实际加载的 URDF、动作数据与配置仍未核对。

建议先在训练环境运行：

```bash
cd Imgo2_rl
python -m unittest discover -s tests -p test_amp_alignment.py -v
python scripts/rl_lab/amp/train.py --task=Imgo2-basemove-flat-amp --num_envs=256 --max_iterations=100 --seed=42 --headless
```

### 7.1 提交训练前的前置检查

以下检查都很便宜，能避免用一次完整训练去发现本可提前发现的问题。

1. **先把修复后的代码同步到训练机，不要按文件时间戳挑文件**。`Imgo2_rl` 下除少数文件外都带着 `2026-07-18 11:16` 这个批量拷贝留下的统一时间戳，真正属于本次修复的 `mdp/observations.py`、`mdp/amp_events.py` 也在其中；按修改时间挑选必然漏掉它们，而漏掉 `amp_events.py` 就等于参考重置失去逆映射。按内容确认，本次修复涉及 9 个文件（整树同步更省事）：

   | 文件（相对 `Imgo2_rl/`） | 修复内容 | 本轮验证版本 SHA256 前 12 位 |
   |---|---|---|
   | `source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/mdp/observations.py` | 映射辅助函数重写：由按腿轴（4 元）改为按 12 维平铺索引 | `5ec7fc70a237` |
   | `source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/mdp/amp_events.py` | `_invert_mapping`，参考重置逆映射 | `3b529139161a` |
   | `source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/base_move/amp_env_cfg.py` | 4 元 `amp_leg_mapping` 改为 12 元恒等 `joint_mapping`、高度项、基座触地终止、每步权重 | `cbc890318c5f` |
   | `source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/base_move/agents/amp_rsl_rl_cfg.py` | `amp_task_reward_lerp=0.3` | `ac00f1d0c76d` |
   | `scripts/rl_lab/rl_lab/algorithms/amp_ppo.py` | 归一化尺度（分类与梯度惩罚同尺度，统计用原始值） | `3a87fec7a01d` |
   | `scripts/rl_lab/rl_lab/runners/amp_on_policy_runner.py` | terminal AMP 状态、高度日志 | `2f818c58bf43` |
   | `scripts/rl_lab/rl_lab/datasets/motion_loader.py` | 仅注释与解包变量名：删掉「FR,FL,RR,RL → FL,FR,RL,RR」的错误说明，实际拼接顺序未变 | `6ed5328e5835` |
   | `scripts/rl_lab/amp/play_check.py` | 新增：回放并导出 `policy.pt` / `policy.onnx` | `e94407245943` |
   | `scripts/record_legged_control_amp.py` | 新增：采集脚本在工作区内的副本 | `9dde81cb0ae8` |
   | `scripts/tools/audit_amp_dataset.py` | 离线核对 | `7e0d7e634111` |
   | `scripts/tools/check_amp_joint_order.py` | 三项腿顺序检查（本轮新增） | `17d00857e2ea` |
   | `tests/test_amp_alignment.py` | 离线回归 | `7df0476aef8e` |

   这份清单来自 `Imgo2_rl` 仓库合并前的 `git diff` 与未跟踪文件列表，因此比凭内容整理的版本更可靠：本文此前只列了 9 个，漏掉了 `motion_loader.py`、`play_check.py` 和采集脚本副本。路径都从 `Imgo2_rl/` 起算，便于在服务器上逐条核对。哈希是这一轮验证过的版本，用于确认同步后内容是否真的到位（Linux 下 `sha256sum <文件> | cut -c1-12`），不是长期不变的约定；文件再改动后哈希随之更新。之所以用哈希而不是时间戳，是因为本工作区多数源码带同一批拷贝时间戳。

   本文其他位置提到的「六个修改或新增 Python 文件」是当初做语法检查的文件数，不等于需要同步的清单。
2. **补跑被跳过的 Torch 回归**。它是唯一针对归一化尺度不一致的测试，几秒钟即可完成，但需要 `torch`/`numpy`，因此只能在训练环境执行。注意本机通过的 4 项只覆盖运动学与奖励量级，不覆盖此项。
3. **确认服务器资源与数据副本**。`IMGO2_CFG` 的 URDF 路径和 `AMP_MOTION_FILES` 都硬编码为 `/root/gpufree-data/Imgo2_rl/...`（见 README 的 ENV-01）。若 glob 为空，AMPLoader 不会给出明确报错。此外恒等映射的结论是针对当前这份数据证明的，本地两份副本逐文件哈希一致、集合哈希为 `163d96671f601673a950b0da13d1c5fb`，服务器那份值得对照。
4. **第一次训练保持短**。本次涉及的改动文件（关节映射、归一化、奖励与终止配置）在本机只做过语法检查，从未被导入（本机没有 Isaac Lab），因此这个短训练同时是导入与配置构建的冒烟测试。

配置接线已在源码中核对，可作为参考：`amp_task_reward_lerp=0.3` 定义在 `agents/amp_rsl_rl_cfg.py`，runner 通过 `self.cfg = train_cfg.get("runner", train_cfg)` 读取，而 `train.py` 传入 `agent_cfg.to_dict()`，因此 0.3 生效（`AMPOnPolicyRunnerCfg` 的默认值仍是 0.1）。`base_height_l2(sensor_cfg=None)` 在平地直接使用目标高度，并有重力投影门控，不会访问不存在的扫描传感器。`illegal_contact = None` 只写在 `Imgo2RoughEnvCfg`，AMP 继承自公共基类，所以基座触地终止确实生效而不会在构建配置时抛错。

短训练确认数值、奖励项和终止条件正常后再扩大训练。应建立新运行，避免将旧判别器和旧归一化统计直接带入对照实验。

观察 `AMP/mean_root_height_m`、`AMP/fraction_root_height_below_0_20m`、`Train/weighted_task_reward_step`、`Train/mean_amp_disc_pred_step`、速度 reward 和 episode 长度。高度升高但 episode 极短不算成功，可能只是触地后频繁重置；需要结合回放判断。

为了区分效果来源，可在修正映射/归一化后，对比“有高度项”和“`env.rewards.base_height_l2.weight=0`”，其余随机种子、任务混合比例及训练预算相同。以上仅为实验安排，尚无效果结论。
