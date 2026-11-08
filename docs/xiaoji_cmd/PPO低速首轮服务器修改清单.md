# PPO低速首轮：服务器逐处修改清单

> **2026-11-02实际效果更新**：按该方案产生的最新1999策略已在部署MuJoCo复现非对角配合、后退贴地；本清单是配置修改历史，不能作为已成功步态配方。已有旧策略对照和修正路线见[最新复核](../training/ppo_low_speed_1999_review_2026-11-02.md)。服务器保存源码diff已含两奖励修复，缺的是目标步态/净空及实际运行统计，当前不建议原样盲续。

> **同日判断补充**：显式gait是可选路线，不是已证明必需项；两奖励修改与轮数影响尚未分离。当前先按[复核第8节](../training/ppo_low_speed_1999_review_2026-11-02.md#8-重新核对当前启停清单轮数和下一步当前建议)做服务器Isaac检查、固定配方追加500轮及必要的no-slide对照。本清单中的新奖励候选不自动实施。

> 后续实际状态（2026-10-31）：用户已授权修改本机奖励文件，两个旧函数已整段注释，新实现紧随其后；回退检查及14个生产函数CPU案例通过，见[实际修改记录](../training/velocity_reward_fixes_2026-10-31.md)。下方保留各次服务器方案/附件复审的当时状态；本机rough与采样器未修改，服务器同步和Isaac冒烟仍待确认。

日期：2026-10-31。基准是用户本轮粘贴的服务器`rough_env_cfg.py`，不是本机那份：服务器已是base=-20/target=.32、air=2.0、hip L1=-1、全关节偏差=-.1、mirror仅thigh/shank、两个足高项0。**本清单是待服务器应用的代码候选，未覆盖本机生产源码，未启动训练。** 仅本机CPU公式/采样检查不能代替Isaac场景构造与GPU冒烟。

以下服务器路径以已有仓库`/root/Desktop/imgo2`为例；若仓库实际在别处，只替换这个前缀。先激活服务器原来能训练的环境，不重新安装Isaac。

## 1. 本轮改哪些文件

```text
/root/Desktop/imgo2/imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/
├── base_move/rough_env_cfg.py   # 速度、站立比例、地形与课程
└── mdp/
    ├── commands.py             # 可选开关：纯方向采样
    └── rewards.py              # 核对并修正高度射线、脚滑
```

公共`velocity_env_cfg.py`不必改；具体rough任务覆盖公共默认值。服务器只提供了rough文件，所以不能断言服务器commands/rewards与本机完全相同：后两项应先查看同名函数，已经正确实现时不要重复替换。修改前自行保留这三个文件原件，勿覆盖其他算法或模型。

## 2. rough文件：地形

### 2.1 顶部增加import

紧挨现有`from isaaclab.utils import configclass`增加：

```python
import isaaclab.terrains as terrain_gen
```

### 2.2 替换原来的四类地形参数修改块

在`__post_init__()`中，找到从`sub_terrains["boxes"].grid_height_range`开始，到`sub_terrains["pyramid_stairs_inv"].step_height_range`结束的整块，替换为下面代码。缩进保持8个空格，与`self.scene.robot = ...`同级。

```python
        # 第一轮：80%平地、20%轻微粗糙；保留rough任务观测结构。
        terrain = self.scene.terrain.terrain_generator
        rough = terrain.sub_terrains["random_rough"]
        rough.proportion = 0.2
        rough.noise_range = (-0.01, 0.01)
        rough.noise_step = 0.005
        terrain.sub_terrains = {
            "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.8),
            "random_rough": rough,
        }
        self.scene.terrain.max_init_terrain_level = 0
```

此处是**替换**旧块，不是在旧块前删除stairs/boxes后再执行旧赋值，否则会KeyError。地形配置比例不等于所有训练时长恰好80/20。轻微粗糙noise_range是配置采样范围，插值与网格离散化仍以生成结果为准。

### 2.3 课程区域增加两行

保留已有两个速度课程为None，在其后加：

```python
        self.curriculum.terrain_levels = None
        self.scene.terrain.terrain_generator.curriculum = False
```

父类`__post_init__()`已执行过，所以这里显式同时关闭生成器课程。仅设max_init_terrain_level=0不能限制后续难度；本机random_rough生成函数也不使用difficulty，因此要直接缩小noise_range。

## 3. rough文件：指令

把末尾`Commands`区域的三条速度赋值替换为：

```python
        self.commands.base_velocity.ranges.lin_vel_x = (-0.6, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.35, 0.35)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.6, 0.6)
        self.commands.base_velocity.resampling_time_range = (5.0, 8.0)
        self.commands.base_velocity.rel_standing_envs = 0.15
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.heading = None
```

再完成第4节后，紧接增加下面三行；**没改第4节时先不要添加这些新字段**：

```python
        self.commands.base_velocity.directional_sampling = True
        self.commands.base_velocity.directional_min_lin_speed = 0.25
        self.commands.base_velocity.directional_min_ang_speed = 0.25
```

只有前8行时，是“低速混合指令+15%全零”的简化方案，不等于纯前/后/横移/转向分组。建议本轮完成分组，以覆盖已经暴露问题的后退、原地转向。

## 4. commands.py：增加纯方向采样开关

只修改`UniformThresholdVelocityCommand`与其对应Cfg，不替换其他命令类。`mdp/__init__.py`本机已有`from .commands import *`，不需新增注册。

### 4.1 在对应Cfg中增加三个字段

找到`class UniformThresholdVelocityCommandCfg`，在其已有`class_type`下面增加：

```python
    directional_sampling: bool = False
    directional_min_lin_speed: float = 0.25
    directional_min_ang_speed: float = 0.25
```

默认False，使没有主动开启的其他任务保持原采样逻辑。

### 4.2 替换该命令类的_resample_command方法

保留类的`__init__`与`_update_command`等其他方法，单独替换`_resample_command`：

```python
    def _resample_command(self, env_ids: Sequence[int]):
        # 父类只抽一次站立标记，后面沿用，避免重复叠加站立概率。
        super()._resample_command(env_ids)
        if not self.cfg.directional_sampling:
            self.vel_command_b[env_ids, :2] *= (
                torch.norm(self.vel_command_b[env_ids, :2], dim=1) > 0.2
            ).unsqueeze(1)
            return

        if self.cfg.heading_command:
            raise ValueError("directional_sampling requires heading_command=False")
        v_min = self.cfg.directional_min_lin_speed
        w_min = self.cfg.directional_min_ang_speed
        if v_min <= 0.2 or w_min <= 0.0:
            raise ValueError("Use lin min > 0.2 and angular min > 0")

        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        n = env_ids.numel()
        if n == 0:
            return
        # 六组在非站立样本中的相对概率；乘85%后为25/20/10/10/10/10%。
        probabilities = torch.tensor(
            [25.0, 20.0, 10.0, 10.0, 10.0, 10.0], device=self.device
        )
        group = torch.multinomial(probabilities, n, replacement=True)
        ranges = self.cfg.ranges
        bands = (
            (0, v_min, ranges.lin_vel_x[1]),
            (0, ranges.lin_vel_x[0], -v_min),
            (1, v_min, ranges.lin_vel_y[1]),
            (1, ranges.lin_vel_y[0], -v_min),
            (2, w_min, ranges.ang_vel_z[1]),
            (2, ranges.ang_vel_z[0], -w_min),
        )
        self.vel_command_b[env_ids] = 0.0
        self.is_heading_env[env_ids] = False
        for index, (axis, low, high) in enumerate(bands):
            if low > high:
                raise ValueError("Command ranges are smaller than directional minimum")
            ids = env_ids[group == index]
            self.vel_command_b[ids, axis] = (
                torch.rand(ids.numel(), device=self.device) * (high - low) + low
            )
        standing_ids = env_ids[self.is_standing_env[env_ids]]
        self.vel_command_b[standing_ids] = 0.0
```

预期全体新指令：15%站立、25%前进、20%后退、左右横移各10%、左右转各10%。若改rel_standing_envs，非站立六组仍按25:20:10:10:10:10分配，总体百分比会相应变化。这些是重采样概率，不是永久环境分组或精确时间占比。原_update_command里有pits专用逻辑，但本轮地形集合没有pits，不会触发；以后增加pits时须重审它对站立/方向指令的覆盖。

## 5. rewards.py：核对高度和脚滑函数

### 5.1 高度射线

若服务器`base_height_l2`仍使用全batch的`torch.isnan(ray_hits).any()`分支，可替换为下面函数：

```python
def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        heights = sensor.data.ray_hits_w[..., 2]
        valid = torch.isfinite(heights) & (heights.abs() < 1e6)
        counts = valid.sum(dim=1)
        sums = torch.where(valid, heights, torch.zeros_like(heights)).sum(dim=1)
        ground = sums / counts.clamp_min(1)
        # 首轮仅平地/轻粗糙：全失效环境用地形出生点z近似当地地面。
        # 不用“当前机身高度”当目标；健康环境不会被其他环境影响。
        ground = torch.where(counts > 0, ground, env.scene.env_origins[:, 2])
        adjusted_target = ground + target_height
        env._base_height_ray_valid_fraction = valid.float().mean()
        env._base_height_all_invalid_fraction = (counts == 0).float().mean()
    else:
        adjusted_target = target_height
    penalty = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target)
    upright = torch.clamp(
        -env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7
    ) / 0.7
    return penalty * upright
```

**回退适用范围**：环境出生点z只是在首轮平地/±1cm粗糙条件下的近似，不是坡地/楼梯当前位置的可靠高度。后续恢复复杂地形前，需要补当前位置有效地面缓存/估计及全失效处理。两个`env._...fraction`只是诊断张量，**不会自动出现在TensorBoard**；冒烟时可在调试中读取，若要日志曲线需另接logger。全失效不是可长期忽略的正常状态。

### 5.2 脚滑

若服务器`feet_slide`仍减去root_lin_vel再转到机身系，将整个函数替换为下面版本。静止地面上，支撑脚不滑时世界水平速度应接近0，而不是与机身速度相同。

```python
def feet_slide(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 沿用历史接触检测以减少阈值变动；短暂离地仍可能保留接触标记。
    contacts = (
        sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1).max(dim=1)[0] > 1.0
    )
    asset: RigidObject = env.scene[asset_cfg.name]
    velocity_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    penalty = torch.sum(velocity_xy.norm(dim=-1) * contacts, dim=1)
    upright = torch.clamp(
        -env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7
    ) / 0.7
    return penalty * upright
```

本候选适用于本轮静止平面/轻微粗糙；world xy是水平切向近似，后续斜坡应审查接触法向投影，移动地面还需扣除地面速度。沿用历史接触的滞后也需看真实统计。保留weight=-.05仅作为初始候选，修正语义后分项量级会变化。

## 6. 奖励保留什么、暂不增加什么

按本次服务器附件：base=-20/target=.32、air=2.0/threshold=.5、air variance=-8、hip L1=-1、全关节偏差=-.1、mirror=-1仅thigh/shank、stand_still=-.1，以及其余权重均先保留。**本轮不再同时把air从2.0改成1.8**，避免把采样/地形与air变化混在一起；第一个候选报告里的1.8留作以后独立对照。

`feet_height_body`和`feet_height`继续权重0，位于disable_zero_weight_rewards之前。不新增摆动净空函数，不增加相位观测；后续要做局部地面+期望摆动相位目标，单独实现/对照，不能只给旧FHB乘已离地mask就称已完整解决。PD/默认姿态/动作缩放/模型/观测顺序沿用现有值。

当前非法接触终止为None；本轮没有提供未经运行验证的终止阈值。满回合不能当作未摔倒，需要在短回放中直接确认姿态和接触；恢复base接触终止应另做reset/站立/台阶验证。

## 7. 服务器检查与训练命令

先确认处在原来能训练的Python环境。以下不重新安装Isaac，若已有editable安装也不必重装项目。

```bash
cd /root/Desktop/imgo2/imgo2_rl
export IMGO2_ISAACLAB_PYTHON="$(python -c 'import sys; print(sys.executable)')"
python -m py_compile \
  source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/base_move/rough_env_cfg.py \
  source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/mdp/commands.py \
  source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/mdp/rewards.py
```

语法检查通过后做4环境10轮冒烟（新运行，不带resume）：

```bash
bash scripts/run_isaaclab.sh scripts/rl_lab/ppo/train.py \
  --task=Imgo2-basemove-rough-ppo \
  --num_envs=4 --max_iterations=10 --seed=42 \
  --run_name=low_speed_stage1_smoke \
  --logger=tensorboard --device=cuda:0 --headless
```

冒烟需要核对实际params/env.yaml：base target=.32/weight=-20、air2、足高null、ranges .6/.35/.6、standing .15、heading false、directional_sampling true、地形只有flat/random_rough且.8/.2、地形课程null。训练能跑10轮不代表分组概率和运动质量通过；抽样统计应另外确认六方向仅单轴非零/全零约15%，以及射线异常率。改变重采样最大时间后，当前error_vel指标的归一化口径也变化，不能直接用新旧日志数值大小判断跟速优劣。

确认后开一批2000轮新训练（不是保证收敛的固定轮数）：

```bash
bash scripts/run_isaaclab.sh scripts/rl_lab/ppo/train.py \
  --task=Imgo2-basemove-rough-ppo \
  --num_envs=4096 --max_iterations=2000 --seed=42 \
  --run_name=low_speed_stage1 \
  --logger=tensorboard --device=cuda:0 --headless
```

现有runner的save_interval=100可保留。每500轮在相同指令下比较全零、前后.3/.5、左右横移.3、左右转.3：实际速度、漂移、机身高度、髋角/动作裁剪、gyro、步频/净空和跌倒。先形成可复现基线，再决定续训或加入新足高项，不只看总reward。

## 8. 已完成和待确认

已完成：按用户服务器附件核对已改奖励、明确逐处替换位置；9段Python片段独立语法检查通过。用本机env_isaaclab的CPU torch抽离函数核对：20万次采样站立15.018%、前进25.010%、后退19.962%、其余各约10%；单轴/范围/最小速度、子集重采样、空集合、关闭开关恢复原采样结果均通过。逐环境混合正常/部分坏/全坏射线例子返回平方误差[.0049,.0144,.0001]；世界足速0对应滑移0、单足.1m/s对应原始滑移.1。结果见[检查JSON](../training/ppo_server_recipe_validation_2026-10-31.json)。这里只提供候选，没有修改生产任务、命令类、奖励或策略。

待确认：服务器commands/rewards真实版本、Isaac类实例化/场景生成、GPU冒烟、分组/地形/射线运行统计、低速与后续rough/硬件效果；复杂地形高度回退、坡地切向滑移、倒地终止与新摆动足高仍需后续实现/验收。对应README PPO-ROUGH-01/PPO-GAIT-01。

## 9. 后续服务器附件复审：先保留混合指令，不修改源码

用户随后上传的附件SHA256为`fa6160d2a5c1e3d41f9a986f83af5e1b0e7047c1ca8b31af439d450c0147d1d7`，同日要求速度指令先保持当前方式，并要求先解释两处奖励修法，不修改代码。本节是最新决定：第4节纯方向分组仅作以后可选方案，现阶段不要求加入其新字段。

### 9.1 明确的语法问题与配置变化

- 附件第62行`terrain = self.scene.terrain.terrain_generator`前有16个空格，其他同级语句是8个。标准库AST检查实际报`IndentationError: unexpected indent`；仅在内存中将该行改为8空格后，整份文件语法通过。原附件、服务器文件和本机生产源码均未修改，服务器仍需修正这一行后再检查。
- 地形配置保留原键、各比例先0，再random_rough=.2、flat=.8；noise=[-.01,.01]/step=.005，课程关闭，静态声明与首轮意图一致，不代表场景已运行通过。
- base=-20/.32、两个足高0、髋约束/镜像保留。**最新air已改成1.5，不是此前附件的2.0**；这是额外奖励变动，不是语法错误。它减弱滞空项作用，不能保证更慢周期；用户可将其作为本轮新基线固定并记录，不反复同时更改多个量。
- 非法接触终止仍None，回合跑满不能证明无跌倒。

### 9.2 当前采样确实可能产生纯转向

条件是服务器仍沿用原`UniformThresholdVelocityCommand`：独立均匀抽vx/vy/wz，随后将平移范数≤.2的xy同时清零；heading已False，所以yaw不被朝向反馈覆盖。此时xy清零、wz非零自然产生纯转向，不需要人为分组才能出现。

按附件范围vx±.6、vy±.35、wz±.6和站立概率.15计算：圆形死区面积`pi*.2²`除以矩形面积`1.2*.7`约14.96%；乘非站立.85，得到全体新指令约**12.72%纯转向**。其中|wz|≥.2约**8.48%**，站立约15%。这是理想重采样概率，不是实测时间份额；服务器采样器版本未上传，不能保证其实现与本机一致。相比之下，纯前进/纯后退等单轴平移没有同样的角速度/横移置零机制，精确单轴指令不保证有固定份额。

因此此前“没有纯方向分组”不等于“完全没有纯转向”；先保留当前速度设置可作为简化基线，专项分组暂缓。

### 9.3 两个奖励的解释与修法边界

`base_height_l2`：在现有sensor分支中，用逐环境有效射线均值替代跨batch any/max异常判断；只让全失效环境使用明确的备用地面估计。首轮平地/轻粗糙可用scene.env_origins[:,2]作近似；以后坡地/楼梯应改当前位置可靠估计或带reset/地形切换处理的有效缓存，不能把全批目标改成当前机身高度。保留原平方误差、weight、dt和直立门控；记录有效射线/全失效比例以确认是否触发。第5.1节是这轮简单地形候选代码，不是已应用修复。

`feet_slide`：本轮静止地面下，使用触地足世界系水平速度范数，而不是足速减机身速度后再转机身系。沿用当前历史接触检测与直立门控，先保留weight=-.05并重新观察分项量级。正常支撑时机身向前、足相对地面静止应返回0；足滑.1m/s时每只有效接触足贡献.1原始量。坡地需法向投影，移动地面需扣地面速度，历史接触在刚离地时可能残留，后续接触时序仍需验证。第5.2节是候选代码，服务器函数内容未上传，不断言服务器仍有同样错误。

已完成：附件静态复审、单行缩进修正的内存语法检查、死区纯转向概率推导及修法说明。待确认：服务器缩进修复、两个函数实际版本及是否应用修正、Isaac冒烟/采样统计/效果；用户要求先不修改代码，本轮没有源码、配置、策略或服务器写入。
