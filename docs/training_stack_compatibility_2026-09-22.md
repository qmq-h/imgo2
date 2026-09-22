# 训练栈版本与配置兼容性（2026-09-22）

## 版本基线

用户确认当前训练机使用：

- Isaac Lab 2.2.1；
- RSL-RL 2.3.3。

本机没有这套训练环境，本轮没有构造 Isaac Lab 环境或启动训练。版本信息按用户提供记录，接口结论依据 Isaac Lab v2.2.1 与 RSL-RL v2.3.3 对应源码核对。

## 发现与修复

上层拖曳配置曾使用更新版 Isaac Lab 的 `RslRlRNNModelCfg`、独立 `actor`／`critic` 字段和 `obs_normalization`。Isaac Lab 2.2.1 的 runner 配置没有这些类型和字段，只接受统一的 `policy = RslRlPpoActorCriticRecurrentCfg(...)`，因此原配置会在训练机导入阶段失败。

短暂改为 2.2.1 接口后，最终采用与现有 AMP 相同的处理：towing 训练完全使用仓库 `rl_lab` 的 `ActorCriticRecurrent`、`PPO`、`RolloutStorage` 与 `TowingOnPolicyRunner`，上层配置不再导入 `isaaclab_rl.rsl_rl`。因此 RSL-RL 2.3.3 的 runner/config API 变化不会影响这条训练链路；actor／critic 仍由仓库实现的 `memory_a`／`memory_c` 维护两套独立 GRU hidden state。

仓库 `TowingOnPolicyRunner` 单独持有 `EmpiricalNormalizer`，只在 critic 路径更新和应用；actor 的 51 维显式缩放输入与 decoder estimate 不经过该 normalizer。normalizer state 已与 PPO、decoder 及两个 optimizer 一起纳入 checkpoint。

接线复核同时修正了仓库 recurrent memory 的旧 reset 问题：环境返回的 done 是整型 0/1 张量，原实现直接拿它索引 hidden state，只会反复清环境 0／1；现在先转换为布尔 mask，再按完成环境清理 actor／critic GRU。AMP 当前使用 MLP，因而此前没有触发该问题。

普通 base-move PPO 配置中同样移除了 2.2.1 不存在的 `RslRlMLPModelCfg` import 和 actor／critic 分项 normalization 参数；其 `empirical_normalization=False` 行为保持不变。

## 验证与限制

- 离线契约检查确认 towing 配置不再导入外部 RSL-RL 配置或 runner；
- Python 语法编译通过；完整拖曳离线测试 191 项通过、12 项因可选依赖跳过；
- 仍需在训练机执行配置 import、4 环境短 rollout 和 recurrent minibatch；
- towing runner 已实现但尚未在 Isaac Lab 训练机运行，因此 critic-only normalization 和 recurrent rollout 仍是“已接线，待运行验证”。
