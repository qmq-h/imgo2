# Imgo2 RL

Isaac Lab extension for Imgo2 reinforcement-learning tasks.

> 项目总览、训练与部署流程、已知问题和维护记录以根目录 [README.md](../README.md) 为准。本文只保留本子项目的安装与命令速查。

## Installation

```bash
python -m pip install -e source/imgo2_rl
python -m pip install -e scripts/rl_lab
```

旧文档中的 `script/himloco_rsl_rl` 路径已失效，当前算法包位于 `scripts/rl_lab`。

## Check registered tasks

```bash
python scripts/tools/list_envs.py
python scripts/tools/zero_agent.py --task=Imgo2-basemove-rough-ppo --num_envs=100
```

## Train

```bash
python scripts/rsl_rl/train.py --task=Imgo2-basemove-rough-ppo --headless
python scripts/rsl_rl/train.py --task=Imgo2-basemove-flat-ppo --headless
python scripts/rl_lab/himloco/train.py --task=Imgo2-basemove-rough-himloco --headless
python scripts/rl_lab/amp/train.py --task=Imgo2-basemove-flat-amp --headless
```

## Play

```bash
python scripts/rsl_rl/play.py \
    --task=Imgo2-basemove-rough-ppo \
    --num_envs=100 \
    --checkpoint=<run>/model_4999.pt

python scripts/rl_lab/himloco/play.py \
    --task=Imgo2-basemove-rough-himloco-play \
    --num_envs=100 \
    --checkpoint=<run>/model_1999.pt
```

`<run>` 指这一次训练自己的日志目录。旧文档中写死的 `/root/gpufree-data/...` checkpoint 属于个人训练机，不要直接沿用；本机未核验任何 checkpoint 是否存在。

## Offline checks

只用标准库，不需要 Isaac Lab：

```bash
python scripts/tools/audit_amp_dataset.py --output logs/amp_data_audit.json
python scripts/tools/check_amp_joint_order.py
python -m unittest discover -s tests -p test_amp_alignment.py -v
```

`check_amp_joint_order.py` 默认读取工作区内的 `imgo2_description/urdf/imgo2_description.urdf`。AMP 对齐结论及其适用范围见根 README 的「AMP 对齐排查与下一轮训练」。
