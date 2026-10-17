# main 同步与 decoder 设计初查（2026-10-17）

> 本记录保存拉取与初查时的事实；用户随后明确 CE Net 设计并授权修改，最新实现和验证状态见 [CE Net 记录](../towing/towing_ce_net_2026-10-17.md)。

## 已完成／已修复

- Windows PowerShell 下执行 `git fetch origin main` 和 `git pull --ff-only origin main`（使用命令级 openssl / 7890 代理覆盖）；返回 Already up to date。
- HEAD 与 origin/main 同为 `4d66f2c`，`git rev-list --left-right --count HEAD...origin/main` 为 0/0。没有需要修复的同步冲突，未使用 stash 或改全局配置。
- 保留既有 README、聚类记录、训练入口、rough 配置与未跟踪文件改动。

## decoder 当前结构（源码核对）

依据 `imgo2_rl/scripts/rl_lab/rl_lab/modules/towing_decoder.py` 和 `config/towing_algorithm_cfg.py`：58 维帧经 encoder + GRU 形成 16 维变分 latent，再解码为 `[vx, vy, mass, Fx, Fy, Fz]`。Actor 拼接原始帧、6 维估计与 16 维 latent（共 80 维），后两者 detach，PPO 不反传到估计网络。监督物理量损失与 KL 训练这部分网络，并非下一帧本体观测重建。

## 待修复／待确认

用户反馈 decoder 与预期不符；需要明确期望输入、输出、训练目标以及 actor 接收哪些特征，才能确定偏差根因和修法。本轮只核对源码，没有改网络、运行测试或仿真，也不能据此判定设计正确。
