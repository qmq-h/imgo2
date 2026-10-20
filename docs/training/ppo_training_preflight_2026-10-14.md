# PPO 训练前置核对（2026-10-14）

范围：base_move 的 flat/rough PPO；只检查源码、资源和本机安装定位，未启动 Isaac Sim 或训练，未改训练代码。

## 已核对／已修正

- 当前两项任务为 `Imgo2-basemove-flat-ppo`、`Imgo2-basemove-rough-ppo`；入口 `imgo2_rl/scripts/rl_lab/ppo/train.py`，默认 agent 键 `rsl_rl_cfg_entry_point`。这是官方 RSL-RL PPO，非 AMP 自定义 runner。
- `assets/imgo2.py` 自动推导 `imgo2_description/urdf/imgo2_real.urdf`；`IMGO2_URDF_PATH` 可覆盖。`/usr/bin/python3 imgo2_rl/scripts/tools/check_asset_paths.py` 退出 0，17 个网格引用全部存在；模型通常无需改路径，PPO 不消费 AMP 动作数据。
- 平地默认 4096 环境、5000 轮；rough 默认 2000 轮；24 步/轮、100 轮保存一次，网络 512/256/128，学习率 1e-3 自适应。物理 200 Hz、控制 50 Hz；动作 hip 0.125、其余 0.25，裁剪 ±3；目标高度 0.30 m。速度范围 vx ±1.0、vy ±0.8 m/s、wz ±1.5 rad/s。先保留这些默认值，按显存调整环境数。
- README 补入本次核对入口与待确认状态；没有完成代码修复。

## 待修复／待确认

1. 本机启动器默认 `/opt/conda/envs/isaaclab/bin/python` 不存在，需设置 `IMGO2_ISAACLAB_PYTHON`。现有 `/home/tommy0929/miniconda3/envs/env_isaaclab/bin/python` 的分发版本为 Isaac Lab 0.54.4、isaaclab-rl 0.5.2、RSL-RL 5.0.1、torch 2.7.0+cu128。
2. 同解释器的 `rl_lab` 定位到 `/home/tommy0929/Downloads/Imgo2_rl/scripts/rl_lab/rl_lab/__init__.py`；`imgo2_rl` 在仓库根检查时只是外层 namespace，未定位到应安装的内层扩展。训练前需用目标解释器重新 editable 安装当前两个包并核对定位；本轮未执行安装。
3. PPO runner 未声明 `obs_groups`，当前 Isaac Lab 基类默认 MISSING。RSL-RL 5 的 `resolve_obs_groups` 要求字典；旧 policy 迁移函数不补该字段。建议显式 `actor: [policy]`、`critic: [critic]`，先可用 Hydra 命令覆盖；兼容迁移和环境构造仍需短训练实跑。
4. rough 配置关闭 `illegal_contact`，flat 继承该设置；倒地不会因 base 接触立即结束。不是路径阻塞，但应按训练目标决定是否恢复，当前没有修改。

## 本机建议操作（未执行）

在仓库根：

```bash
conda activate env_isaaclab
export IMGO2_ISAACLAB_PYTHON="$CONDA_PREFIX/bin/python"
python -m pip install -e imgo2_rl/source/imgo2_rl
python -m pip install -e imgo2_rl/scripts/rl_lab
bash imgo2_rl/scripts/run_isaaclab.sh --check
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/rl_lab/ppo/train.py \
  --task Imgo2-basemove-flat-ppo --headless --num_envs 64 \
  --max_iterations 10 --run_name ppo_smoke \
  '+agent.obs_groups={actor:[policy],critic:[critic]}'
```

确认配置校验、模型生成、rollout、PPO 更新和 checkpoint 保存均完成后，再把环境数调到显存能承受的规模（默认 4096），轮数调回 5000，另建正式 run。无需 `--resume` 或初始 checkpoint。日志相对启动目录为 `logs/rsl_rl/base_move_flat/<时间戳>_<run_name>/`。导出用同算法 `play.py`，传实际 checkpoint。

验证限制：只有静态路径检查和包元数据读取；没有 GPU/CUDA 自检、配置运行时实例化或短训练，不能宣称 PPO 已可训练。代码依据以当前未提交工作树为准，未进行 pull，避免混入已有大量改动。
