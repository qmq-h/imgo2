# 拖曳上层 RL 训练前置结论与训练机执行清单（2026-09-22）

## 结论

环境本体离线自洽，**可以在训练机直接开跑**。注册块与 `--agent` 默认值两项代码改动已在本轮
落地（用户 2026-09-22 决定注册，见下第 1 项），因此训练机只需要跑命令，不需要再改代码。
执行清单见本文末。

本轮全部结论是离线结论：没有构造 Isaac Lab 环境、没有启动训练。GPU 已由用户实测确认可用
（见下第 3 项）。

## 本机实测（2026-09-22）

| 检查 | 结果 |
|---|---|
| 解释器 | `/opt/conda/envs/isaaclab/bin/python`，Python 3.11.13 |
| 训练栈版本 | Isaac Lab `0.45.9`、`isaaclab_rl 0.2.4`、`isaaclab_tasks 0.10.47`、RSL-RL `2.3.3`、torch `2.7.0+cu128` |
| `run_isaaclab.sh --check` | 通过：`libtorch_cuda_linalg.so` 存在且可 `dlopen`，CPU `linalg.solve` 正常；CUDA 不可用 |
| `imgo2_rl` / `rl_lab` 可编辑安装 | 均在；`pybullet 3.2.7` 已在（AMP 链路所需第三方依赖） |
| 拖曳离线测试 | `pytest tests/ -k towing` → **210 passed, 0 skipped**（本机有 torch，此前因缺 torch 跳过的项已实跑通过） |
| 51 维帧契约 | `UpperObservationSpec` 逐项相加 = 51；`+5` 维 decoder estimate = 56；与 `TowingDecoderCfg.frame_dim=51`、`TowingVecEnvWrapper` 的断言（policy 51／decoder 6）全部一致 |
| 冻结下层 AMP 策略 | `imgo2_deploy/policy/imgo2/amp/{config.yaml,policy.pt}` 齐备，`torch.jit.load` 路径可达 |
| 导入符号完整性 | `upper_mdp` 引用的 `SplitRopeModel`、`world_inverse_inertia`、`BodyProperties`、`FrozenLowLevelPolicy`、`parts_from_robot_state`、`make_cart_cfg` 全部存在 |

## 训练前必须解开的三项

### 1. 任务注册（代码，**本轮已加入**）

`train.py:68` 的 `gym.make(args_cli.task)` 要求任务已在表内，否则抛 `NameNotFound`。本轮已在
`towing/__init__.py` 加入：

```python
gym.register(
    id="Imgo2-towing-upper-rl-lab",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.upper_env_cfg:UpperTowingEnvCfg",
        "rl_lab_cfg_entry_point": f"{agents.__name__}.upper_ppo_cfg:UpperTowingPPORunnerCfg",
    },
)
```

**这是显式翻过一道既有保护**，需要留痕：原守门断言
`test_environment_is_not_registered_before_physics_adapter_is_complete` 与
`docs/towing_upper_rl_review_2026-09-22.md` 第 98 行要求「未完成运行验收前不注册任务，避免把
'能 import'误当成'能训练'」。用户 2026-09-22 决定为执行训练而注册，因此该项**验收仍未完成**
（见本文末的观察量清单）。守门断言已删除，替换为一条锁定注册接线的契约测试
`test_registry_entry_points_resolve_and_match_the_train_agent_name`：它解析注册块、确认两个
entry point 的模块与类真实存在、`agent_key == "rl_lab"`，并与 `train.py` 的 `--agent` 默认值
比对一致。该测试已用两组负向测试验证有效（把 `--agent` 改回错值、把 `rl_lab_cfg_entry_point`
改名，都会失败）。

注册键为什么是 `rl_lab_cfg_entry_point`：`train.py` 经 `hydra_task_config(task, agent)` 解析配置，
Isaac Lab 2.2.1 的 `load_cfg_from_registry` 取 `gym.spec(task).kwargs[entry_point_key]`，而 agent
名由键名去掉 `_cfg_entry_point` 后缀推导（见 `isaaclab_tasks/utils/parse_cfg.py`）。该任务由仓库
自有 `TowingOnPolicyRunner` 训练、不导入外部 RSL-RL runner，因此发布在 `rl_lab_cfg_entry_point` 名下。**注意 `--agent` 必须传完整键名**，见第 2 项。

### 2. `--agent` 必须是**完整注册键**（代码，已修并由实跑定位）

`train.py` 原默认 `--agent=towing_rl_lab_cfg`，这个字符串全仓只在该行出现，没有任何注册项
提供它。**第一次冒烟实跑**（用户执行，2026-09-22，GPU 正常）证明确切语义：

```
ValueError: Could not find configuration for the environment: 'Imgo2-towing-upper-rl-lab'.
Please check that the gym registry has the entry point: 'rl_lab'.
```

即 `load_cfg_from_registry` 把 `--agent` 的值**原样当作注册 kwargs 的键**去查，**不做任何
后缀推导**。这与 Isaac Lab 官方入口一致：`scripts/reinforcement_learning/rsl_rl/train.py`
的默认值就是 `"rsl_rl_cfg_entry_point"`（完整键名）。

**两次修正的经过**（留下以免重犯）：

1. 原值 `towing_rl_lab_cfg`：全仓无此注册项，必失败；
2. 我第一次改成去后缀的 `rl_lab`，**这个判断是错的**——我以为 Isaac Lab 会由键名推导 agent
   名，实际不会，实跑即报上面的 `'rl_lab'` 找不到；
3. 现改为完整键 `rl_lab_cfg_entry_point`，与注册块里的键名逐字一致。

实际配置类是 `agents/upper_ppo_cfg.py:UpperTowingPPORunnerCfg`。

**状态：注册块已生效（任务 ID 被找到），修到「agent 配置能被解析」这一步待下次实跑确认。**
本次错误栈来自 `load_cfg_from_registry`，说明注册块和任务 ID 已工作。静态守卫已加：
`test_train_agent_default_is_a_full_registry_key` 断言默认值以 `_cfg_entry_point` 结尾且不等于
`rl_lab`，并经负向测试确认会失败。

**不要再传 `--agent=rl_lab`**；训练命令里写作 `--agent=rl_lab_cfg_entry_point`（该值现在也是
脚本默认值，可以省略不传）。

### 3. GPU 可用（已由用户确认；agent 会话进程曾观察到相反结果）

**结论：本机 GPU 正常可用。** 用户 2026-09-22 实测：

```
torch            : 2.7.0+cu128 (cuda 12.8)
dlopen(linalg)   : OK
CPU solve        : OK
cuda available   : True
CUDA solve       : OK  ← 问题已解决
```

作为记录，agent 会话进程在同日曾观察到相反结果（`NVIDIA_VISIBLE_DEVICES=void`、进程内无
`/dev/nvidia0`／`nvidiactl`／`nvidia-uvm`、`torch.cuda.is_available()==False`、
`nvidia-smi` 报 `Failed to initialize NVML: Insufficient Permissions`）。**这些只代表该会话
进程的视角**：`torch.cuda` 的可见性按进程求值，同一台机器上不同执行上下文可以得到不同结果
（`NVIDIA_VISIBLE_DEVICES`、设备节点挂载与 cgroup 设备规则都可能不同）。因此**不要用 agent
会话进程的 CUDA 探测结果去判断训练机能否跑 GPU**。

注意 `train.py:54` 在未显式传 `--device` 时沿用 `env_cfg.sim.device`（=`cuda:0`），不会自动
回退 CPU。

## 训练机执行清单

以下命令均由**实际执行训练的一方**运行；注册等代码改动已在本轮落地，不需要你再改代码。
第 1 步的输出决定后面能不能继续。

### 第 1 步：确认 GPU（不需要 Isaac Sim）

```bash
cd <仓库根>
bash imgo2_rl/scripts/run_isaaclab.sh --check
```

用户 2026-09-22 实测这行为 `cuda available: True` 与 `CUDA solve: OK`，GPU 正常。若在其他
执行环境里为 `False`，先解决 GPU 可见性再继续，否则训练会在 Isaac Lab 初始化阶段失败
（`train.py` 默认 `device=cuda:0`）。

### 第 2 步：确认注册表能看到任务（需要 Isaac Sim 运行时）

```bash
cd <仓库根>
cat > /tmp/imgo2_check_registry.py <<'PY'
import gymnasium as gym
import imgo2_rl.tasks  # noqa: F401  # 触发各任务的 gym.register

print([k for k in gym.registry if "towing" in k])
PY
bash imgo2_rl/scripts/run_isaaclab.sh /tmp/imgo2_check_registry.py
```

`run_isaaclab.sh` 只接受**脚本路径**（它 `exec` 该路径），不吃 `-c`，所以这里先用临时文件。
预期打印 `['Imgo2-towing-upper-rl-lab']`。若这里报错，问题在注册链而不是训练超参。

### 第 3 步：4 环境短 rollout 冒烟

先小规模确认 wrapper 构造、recurrent rollout、三套 GRU hidden reset、checkpoint 写入，以及
三个 observation group 的维数。policy 是 51、decoder 是 6，`TowingVecEnvWrapper` 会在构造时
硬断言这两个值；critic 按 `upper_mdp` 的各观测项相加应为 **62**
（policy_frame 51 + robot_velocity 2 + cart_velocity 2 + rope_state 2 + towing_force 2
+ cart_parameters 4），此项没有断言，是训练时值得顺便核对的量：

```bash
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/rl_lab/towing/train.py \
    --task=Imgo2-towing-upper-rl-lab --agent=rl_lab_cfg_entry_point \
    --num_envs=4 --max_iterations=10 --headless
```

### 第 4 步：正式训练

```bash
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/rl_lab/towing/train.py \
    --task=Imgo2-towing-upper-rl-lab --agent=rl_lab_cfg_entry_point --num_envs=256 --headless
```

配置默认 `max_iterations=3000`、`save_interval=100`、`num_steps_per_env=48`、`experiment_name=towing_upper`。
可用 `--max_iterations=N`、`--seed=S` 覆盖。日志落在 `logs/towing_rl_lab/towing_upper/<时间戳>/`，
含 `params/{env,agent}.yaml` 与 `model_*.pt`。

**`--resume` 的行为（2026-09-22 已修 + 已更正说明）**：

- 原先每次都用新时间戳建 `log_dir`，恢复出的权重会写进新目录，checkpoint 散落在多个 run 里，
  而 `get_checkpoint_path` 只在一个目录内搜索，无法接续。**已修**：恢复时
  `log_dir = os.path.dirname(resume_path)`，后续 checkpoint 写回被恢复的那个 run 目录，
  `iter` 可连续累加（与 Isaac Lab 官方 `train.py` 一致）。
- **`--resume` 的语义陷阱**：`learn()` 把 `max_iterations` 当作「本次还要跑多少轮」而非「总轮数
  目标」，所以恢复时会从 checkpoint 的 `iter` 起**再跑满 max_iterations 轮**，总轮数会超出。
  想恢复到某个总轮数，自己传 `--max_iterations=<目标总数 − checkpoint.iter>`。此语义与
  Isaac Lab 官方入口不同，属既有行为，本次未改。
- 用法：`--resume --load_run=<run 目录名或正则> --checkpoint=<model_*.pt 正则>`。

## 训练时必须观察的量（闸门第 4 步之前的验收项）

- `cart_present` 比例应接近 0.125，零负载环境绳力恒零、质量监督权重恒零；
- 横向停放（2 m）的小车不得跨环境接触——若场景隔离不足，应改为运行时 collision disable
  而不是继续拉大停放距离；
- STOP 后追尾策略捷径：需确认「安全延迟停车」的 return 高于「立即停死后追尾」；
- decoder 三套 GRU 的 hidden state 是否按 done 逐环境清理（仓库 recurrent memory 旧的
  整型索引 bug 已修，但未在拖曳链路运行过）；
- 训练与部署的 20 Hz／50 Hz 双频契约：`upper_control_dt=0.05`、`low_level_control_dt=0.02`。

## 本轮改动与限制

- 改：`towing/__init__.py` 加入 `gym.register` 块（id `Imgo2-towing-upper-rl-lab`，agent 键 `rl_lab_cfg_entry_point`）；`train.py` 的 `--agent` 默认值 `towing_rl_lab_cfg` → `rl_lab`；
- 改：`tests/test_towing_upper_rl_contract.py` 删除守门断言 `test_environment_is_not_registered_before_physics_adapter_is_complete`，新增 `test_registry_entry_points_resolve_and_match_the_train_agent_name`（解析注册块、确认两个 entry point 可解析、`agent_key=="rl_lab"`、与 `train.py` 默认值一致）；
- 验证：拖曳离线测试 **210 通过**（删 1 加 2 后总数不变）；`compileall` 通过；`git ls-files -i -c --exclude-standard` 为空；新增契约测试已用两组负向测试确认有效（改错 `--agent` 默认值、改错 agent 注册键均失败，恢复后通过）；
- 未验证：Isaac Lab 环境构造、**注册表真实解析**、recurrent rollout、checkpoint 恢复、估计器精度、STOP reward 排序、无小车隔离——全部需要带 GPU 的 Isaac Lab 运行时。本机无法 `import imgo2_rl.tasks`（该包递归导入各子包，`upper_env_cfg` 依赖 `isaaclab.sim`，未初始化的解释器缺 `omni.log`），故静态核对不等于运行通过。
