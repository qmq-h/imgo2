"""上层拖曳策略的推理运行时（纯 torch + 标准库，**不 import Isaac Lab**）。

用途：给必要性测试台 `play_towing_test.py` 的 `--upper-checkpoint` 开关提供「策略 vs 基线」
对照所需的那一半——从训练侧 checkpoint 里恢复 actor + dynamics decoder，按**训练侧同一
帧定义**逐拍组装 57 维 policy 帧，输出 12 维归一化关节位置残差。

契约（全部由源码核对，逐条在下面注明出处）：

1. **帧（57 维）** = `cat(loco_command(3), processed_actions(12), base_ang_vel_b(3)·0.25,
   projected_gravity(3), last_loco_action(12), joint_pos_rel(12), joint_vel(12)·0.05)`。
   出处：`upper_mdp.policy_frame`；顺序/维数与 `upper_logic.UpperObservationSpec.terms` 一致
   （本模块 `FRAME_TERMS` 只是同一张布局表，离线测试逐项交叉核对）。
2. **decoder**：`TowingDynamicsDecoder(frame_dim=57, feature_dim=128, hidden_dim=128,
   num_layers=1, latent_dim=16, force_scale=10.0)`，用 `forward_with_latent(frame, hidden)`
   且**必须 `.eval()`**（`sample=None` 会取 `self.training=False` ⇒ latent 取均值，与训练
   rollout 一致）⇒ 返回 `(estimate(6), latent(16), hidden)`。
3. **actor**：`ActorCriticRecurrent`（hidden `[256,128,64]`、GRU 256×1、elu），输入 =
   `cat(frame, estimate, latent)` = **79** 维（`towing_decoder.augment_actor_observation`）。
   确定性推理用 `act_inference()`（取均值，与 `rl_lab/towing/play.py` 同口径）；GRU 隐状态
   由模块内部的 `Memory` 持有，`reset()` 清空。
4. **残差**：`delta = clamp(action, ±1) · action_scale`，`action_scale` 取冻结策略契约
   （`policy_cfg.action_scale`：hip 0.125、thigh/shank 0.25）。出处：`upper_mdp.process_actions`
   （`_processed = actions.clamp(-1,1)`、`delta_joint_pos = _processed · residual_scale`）。
   `processed` 就是下一拍帧里的 `last_action`（`upper_mdp.upper_last_action`）⇒ 本类显式持有
   `_last_action`（首拍为零，与训练一致：`reset()` 把 `_processed` 清零）。
5. **checkpoint**：`torch.load(path, map_location="cpu", weights_only=False)` ⇒ 键
   `model_state_dict`、`decoder_state_dict`、`towing_contract`
   （`{'version': 2, 'frame_dim': 57, 'explicit_dim': 6, 'latent_dim': 16}`）、`iter`。
   加载前**硬校验** `towing_contract` 四个字段：旧 56/63 维 checkpoint 必须被拒绝，而不是
   等到 `load_state_dict(strict=True)` 才报一句难读的 shape mismatch。
6. 网络超参从注册的 agent cfg 建（`agents/upper_ppo_cfg.py::UpperTowingPPORunnerCfg`，
   它镜像训练超参）；`UpperNetworkSpec.from_agent_cfg()` 接受任意带
   `.policy` / `.decoder` 属性的对象（`configclass`、`SimpleNamespace`、dict 都行），
   所以离线测试不需要 import isaaclab。

**注意**：本模块只负责「帧 → 残差」。上层的**控制节拍**由调用方负责：训练侧
`upper_control_dt = 0.05 s`（20 Hz），两次上层更新之间残差保持不变
（`HierarchicalVelocityAction.apply_actions` 每次冻结策略刷新都重算 `joint_targets + delta`）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib.util
import sys
import types
from typing import Any, Callable, Mapping, Optional, Sequence

import torch


#: policy 帧的布局表：`(项名, 维数, 缩放)`。名字与顺序必须与 `upper_mdp.policy_frame`
#: 的拼接顺序、`upper_logic.UpperObservationSpec.terms` 逐项一致（离线测试双向核对）。
FRAME_TERMS: tuple[tuple[str, int, float], ...] = (
    ("loco_command", 3, 1.0),       # 送冻结策略的脚本速度指令（含 PD 的 vy/wz）
    ("last_action", 12, 1.0),       # 上一拍上层动作 **clamp 到 ±1**（未缩放）
    ("base_ang_vel", 3, 0.25),      # 机体系角速度
    ("projected_gravity", 3, 1.0),  # 机体系重力方向（单位向量）
    ("last_loco_action", 12, 1.0),  # 冻结策略**自己**的动作
    ("joint_pos", 12, 1.0),         # joint_pos − default_joint_pos（策略关节顺序）
    ("joint_vel", 12, 0.05),        # 策略关节顺序
)
FRAME_DIM = sum(dim for _, dim, _ in FRAME_TERMS)

#: decoder 显式输出（`towing_decoder.OUTPUT_DIM`：速度 2 + 质量 1 + 力 3）与 latent。
EXPLICIT_DIM = 6
LATENT_DIM = 16
ACTION_DIM = 12
#: 训练侧上层动作裁剪（`upper_mdp.process_actions` 里写死 `actions.clamp(-1.0, 1.0)`）。
ACTION_CLIP = 1.0
CHECKPOINT_CONTRACT_VERSION = 2

#: critic 观测维数（actor 里附带 critic 网络，`load_state_dict(strict=True)` 需要它）。
#: 出处：`upper_env_cfg.UpperTowingEnvCfg.observations.critic` =
#: `policy_frame(57) + robot_velocity(2) + cart_velocity(2) + rope_state(4) +
#: towing_force(3) + cart_privileged_parameters(4)` = 72。
#: 离线测试解析 `upper_env_cfg.py` 的 `CriticCfg` 项逐项核对这个常量。
DEFAULT_CRITIC_OBS_DIM = 72


class TowingCheckpointContractError(ValueError):
    """checkpoint 的 `towing_contract` 与运行时契约不符（含旧 56/63 维 checkpoint）。"""


@dataclass(frozen=True)
class NetworkModules:
    """按需加载的三个 rl_lab 网络入口（测试可注入按文件路径加载的同类对象）。"""

    actor_critic: type
    decoder: type
    augment_actor_observation: Callable[..., torch.Tensor]
    reset_gru_hidden: Callable[..., Any]


def _ensure_rl_lab_on_path() -> None:
    """把 `imgo2_rl/scripts/rl_lab` 加进 `sys.path`（训练脚本用同一约定）。"""
    candidate = Path(__file__).resolve().parents[1] / "rl_lab"
    if candidate.is_dir():
        text = str(candidate)
        if text not in sys.path:
            sys.path.insert(0, text)


def _load_network_modules_by_path() -> NetworkModules:
    """离线兜底：按文件路径加载 actor/decoder，不拉进 `rl_lab.utils`（那里 import isaaclab）。"""
    modules_dir = Path(__file__).resolve().parents[1] / "rl_lab" / "rl_lab" / "modules"
    if not modules_dir.is_dir():
        raise ImportError(f"找不到 rl_lab.modules 目录：{modules_dir}")
    names = ("rl_lab", "rl_lab.modules", "rl_lab.utils",
             "rl_lab.modules.actor_critic", "rl_lab.modules.actor_critic_recurrent",
             "rl_lab.modules.towing_decoder")
    saved = {name: sys.modules.get(name) for name in names}

    def load(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载模块 {name}：{path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    try:
        for name in ("rl_lab", "rl_lab.modules", "rl_lab.utils"):
            package = types.ModuleType(name)
            package.__path__ = []
            sys.modules[name] = package
        # `actor_critic_recurrent` 顶层 `from rl_lab.utils import unpad_trajectories`；
        # 那是 PPO 批量更新路径用的，推理不用，给个恒等桩即可（与仓库既有离线测试同一做法）。
        sys.modules["rl_lab.utils"].unpad_trajectories = lambda out, masks: out
        load("rl_lab.modules.actor_critic", modules_dir / "actor_critic.py")
        recurrent = load("rl_lab.modules.actor_critic_recurrent",
                         modules_dir / "actor_critic_recurrent.py")
        decoder_module = load("rl_lab.modules.towing_decoder",
                              modules_dir / "towing_decoder.py")
        return NetworkModules(
            actor_critic=recurrent.ActorCriticRecurrent,
            decoder=decoder_module.TowingDynamicsDecoder,
            augment_actor_observation=decoder_module.augment_actor_observation,
            reset_gru_hidden=decoder_module.reset_gru_hidden,
        )
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def load_network_modules() -> NetworkModules:
    """加载 actor/decoder（惰性，模块顶部不 import rl_lab；离线进程走按路径兜底）。"""
    _ensure_rl_lab_on_path()
    try:
        from rl_lab.modules.actor_critic_recurrent import ActorCriticRecurrent
        from rl_lab.modules.towing_decoder import (
            TowingDynamicsDecoder, augment_actor_observation, reset_gru_hidden)
    except ImportError:
        return _load_network_modules_by_path()
    return NetworkModules(actor_critic=ActorCriticRecurrent,
                          decoder=TowingDynamicsDecoder,
                          augment_actor_observation=augment_actor_observation,
                          reset_gru_hidden=reset_gru_hidden)


def _field(source: Any, name: str, default: Any = None) -> Any:
    """从 `configclass` / 对象 / dict 里取字段（三种都支持，便于离线测试注入）。"""
    if isinstance(source, Mapping):
        if name in source:
            return source[name]
    elif hasattr(source, name):
        return getattr(source, name)
    if default is None:
        raise ValueError(f"配置里缺少字段 {name!r}：{source!r}")
    return default


@dataclass(frozen=True)
class UpperNetworkSpec:
    """构建 actor/decoder 所需的最小超参集合（默认值即注册 cfg 的镜像）。"""

    frame_dim: int = FRAME_DIM
    explicit_dim: int = EXPLICIT_DIM
    latent_dim: int = LATENT_DIM
    num_actions: int = ACTION_DIM
    num_critic_obs: int = DEFAULT_CRITIC_OBS_DIM
    actor_hidden_dims: tuple[int, ...] = (256, 128, 64)
    critic_hidden_dims: tuple[int, ...] = (256, 128, 64)
    activation: str = "elu"
    rnn_type: str = "gru"
    rnn_hidden_size: int = 256
    rnn_num_layers: int = 1
    init_noise_std: float = 0.5
    decoder_feature_dim: int = 128
    decoder_hidden_dim: int = 128
    decoder_num_layers: int = 1
    decoder_force_scale: float = 10.0

    @property
    def actor_obs_dim(self) -> int:
        return self.frame_dim + self.explicit_dim + self.latent_dim

    @classmethod
    def from_agent_cfg(cls, cfg: Any, *, num_critic_obs: Optional[int] = None) -> "UpperNetworkSpec":
        """从注册的 agent cfg（`UpperTowingPPORunnerCfg` 或其同构对象）取网络超参。"""
        policy = _field(cfg, "policy")
        decoder = _field(cfg, "decoder")
        if num_critic_obs is None:
            num_critic_obs = _field(policy, "num_critic_obs", DEFAULT_CRITIC_OBS_DIM)
        return cls(
            frame_dim=int(_field(decoder, "frame_dim")),
            explicit_dim=int(_field(decoder, "output_dim", EXPLICIT_DIM)),
            latent_dim=int(_field(decoder, "latent_dim")),
            num_actions=int(_field(cfg, "num_actions", ACTION_DIM)),
            num_critic_obs=int(num_critic_obs),
            actor_hidden_dims=tuple(int(value) for value in _field(policy, "actor_hidden_dims")),
            critic_hidden_dims=tuple(int(value) for value in _field(policy, "critic_hidden_dims")),
            activation=str(_field(policy, "activation")),
            rnn_type=str(_field(policy, "rnn_type", "gru")),
            rnn_hidden_size=int(_field(policy, "rnn_hidden_size", 256)),
            rnn_num_layers=int(_field(policy, "rnn_num_layers", 1)),
            init_noise_std=float(_field(policy, "init_noise_std", 0.5)),
            decoder_feature_dim=int(_field(decoder, "feature_dim")),
            decoder_hidden_dim=int(_field(decoder, "hidden_dim")),
            decoder_num_layers=int(_field(decoder, "num_layers")),
            decoder_force_scale=float(_field(decoder, "force_scale", 10.0)),
        )

    def validate(self) -> None:
        if self.frame_dim != FRAME_DIM:
            raise ValueError(
                f"decoder.frame_dim={self.frame_dim} 与本运行时的帧布局 {FRAME_DIM} 不一致"
                f"（`upper_logic.UpperObservationSpec` 已变？先同步 FRAME_TERMS）")
        if self.explicit_dim != EXPLICIT_DIM or self.latent_dim != LATENT_DIM:
            raise ValueError(
                f"decoder explicit/latent 维数 {self.explicit_dim}/{self.latent_dim} != "
                f"{EXPLICIT_DIM}/{LATENT_DIM}")
        if self.num_actions <= 0 or self.num_critic_obs <= 0:
            raise ValueError("num_actions / num_critic_obs 必须为正")


def default_network_spec(*, num_critic_obs: Optional[int] = None) -> UpperNetworkSpec:
    """从注册的 agent cfg 建 spec；注册文件读不到时退回 `UpperNetworkSpec` 的默认值。"""
    _ensure_rl_lab_on_path()
    try:
        from imgo2_rl.tasks.manager_based.towing.agents.upper_ppo_cfg import (
            UpperTowingPPORunnerCfg)
    except ImportError as exc:  # 离线进程没有 isaaclab：用镜像默认值，加载时仍会 strict 校验
        print(f"[upper] 读不到注册的 agent cfg（{type(exc).__name__}: {exc}）；"
              f"使用 UpperNetworkSpec 默认值（= 注册 cfg 的镜像）")
        return UpperNetworkSpec(num_critic_obs=num_critic_obs or DEFAULT_CRITIC_OBS_DIM)
    return UpperNetworkSpec.from_agent_cfg(UpperTowingPPORunnerCfg(),
                                           num_critic_obs=num_critic_obs)


def expected_towing_contract(spec: UpperNetworkSpec) -> dict:
    return {"version": CHECKPOINT_CONTRACT_VERSION, "frame_dim": spec.frame_dim,
            "explicit_dim": spec.explicit_dim, "latent_dim": spec.latent_dim}


def validate_towing_contract(contract: Any, spec: UpperNetworkSpec) -> dict:
    """硬校验 checkpoint 的 `towing_contract`（旧 56/63 维 checkpoint 在这里就被拒）。"""
    expected = expected_towing_contract(spec)
    if not isinstance(contract, Mapping):
        raise TowingCheckpointContractError(
            f"checkpoint 里没有 towing_contract（得到 {type(contract).__name__}）："
            f"这不是上层拖曳 VAE/残差契约的 checkpoint，不能用于策略对照")
    found = dict(contract)
    if found != expected:
        raise TowingCheckpointContractError(
            f"checkpoint 的 towing_contract {found} 与当前契约 {expected} 不一致："
            f"旧 run 的 56/63 维（frame 51/56、actor 56/63）checkpoint 与残差方案不兼容，"
            f"必须用新契约重训；不要用改元数据的方式绕过")
    return expected


def load_towing_checkpoint(path) -> dict:
    """读 checkpoint（CPU）并确认三个必需键存在。"""
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"找不到上层 checkpoint：{checkpoint_path}")
    state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise TowingCheckpointContractError(
            f"checkpoint 顶层不是 dict（得到 {type(state).__name__}）：{checkpoint_path}")
    missing = [key for key in ("towing_contract", "model_state_dict", "decoder_state_dict")
               if key not in state]
    if missing:
        raise TowingCheckpointContractError(
            f"checkpoint 缺少键 {missing}：{checkpoint_path}（需要 towing runner 的联合 checkpoint）")
    return state


def _validate_action_scale(action_scale: Sequence[float], *, num_actions: int) -> torch.Tensor:
    """残差尺度校验，规则与 `upper_logic.UpperActionSpec.validate` 相同。"""
    values = [float(value) for value in action_scale]
    if len(values) != num_actions:
        raise ValueError(f"action_scale 有 {len(values)} 项，动作维数是 {num_actions}")
    for value in values:
        if not (value == value and value not in (float("inf"), float("-inf"))):  # NaN/Inf
            raise ValueError(f"action_scale 必须全部为有限数，收到 {value!r}")
        if value <= 0.0:
            raise ValueError(f"action_scale 必须全为正（否则该关节的残差通道被静默关闭）：{value!r}")
    return torch.tensor(values, dtype=torch.float32)


class UpperPolicyRuntime:
    """从 checkpoint 恢复的「帧 → 12 维关节位置残差」运行时。

    典型用法（每 `upper_control_dt` 调一次；两次之间保持 `delta` 不变）::

        runtime = UpperPolicyRuntime(ckpt, num_envs=N, action_scale=policy_cfg.action_scale,
                                     device="cuda:0", deterministic=True)
        delta, processed = runtime.act(
            loco_command=cmd, base_ang_vel=ang_vel_b, projected_gravity=gravity_b,
            last_loco_action=loco_action, joint_pos_rel=joint_pos_rel, joint_vel=joint_vel)
        held = frozen_joint_targets + delta      # 下发 held，并用它当 JNT 指标参考
    """

    def __init__(self, checkpoint, *, num_envs: int, action_scale: Sequence[float],
                 device: str = "cpu", deterministic: bool = True,
                 spec: Optional[UpperNetworkSpec] = None,
                 agent_cfg: Any = None, num_critic_obs: Optional[int] = None,
                 network_modules: Optional[NetworkModules] = None) -> None:
        if int(num_envs) <= 0:
            raise ValueError(f"num_envs 必须是正整数，收到 {num_envs!r}")
        self.device = torch.device(device)
        self.deterministic = bool(deterministic)
        self.num_envs = int(num_envs)
        modules = network_modules or load_network_modules()
        if spec is None:
            spec = (UpperNetworkSpec.from_agent_cfg(agent_cfg, num_critic_obs=num_critic_obs)
                    if agent_cfg is not None else default_network_spec(num_critic_obs=num_critic_obs))
        spec.validate()
        self.spec = spec
        self._augment = modules.augment_actor_observation
        self._reset_gru_hidden = modules.reset_gru_hidden

        self.decoder = modules.decoder(
            frame_dim=spec.frame_dim, feature_dim=spec.decoder_feature_dim,
            hidden_dim=spec.decoder_hidden_dim, num_layers=spec.decoder_num_layers,
            latent_dim=spec.latent_dim, force_scale=spec.decoder_force_scale)
        if int(self.decoder.output_dim) != spec.explicit_dim:
            raise ValueError(f"decoder.output_dim={self.decoder.output_dim} != spec.explicit_dim "
                             f"{spec.explicit_dim}")
        if int(self.decoder.latent_dim) != spec.latent_dim:
            raise ValueError(f"decoder.latent_dim={self.decoder.latent_dim} != "
                             f"spec.latent_dim {spec.latent_dim}")
        self.actor = modules.actor_critic(
            num_actor_obs=spec.actor_obs_dim, num_critic_obs=spec.num_critic_obs,
            num_actions=spec.num_actions, actor_hidden_dims=list(spec.actor_hidden_dims),
            critic_hidden_dims=list(spec.critic_hidden_dims), activation=spec.activation,
            rnn_type=spec.rnn_type, rnn_hidden_size=spec.rnn_hidden_size,
            rnn_num_layers=spec.rnn_num_layers, init_noise_std=spec.init_noise_std)

        state = load_towing_checkpoint(checkpoint)
        self.contract = validate_towing_contract(state["towing_contract"], spec)
        self.actor.load_state_dict(state["model_state_dict"], strict=True)
        self.decoder.load_state_dict(state["decoder_state_dict"], strict=True)
        self.actor.to(self.device).eval()
        self.decoder.to(self.device).eval()
        self.checkpoint_path = str(Path(checkpoint).expanduser())
        self.iteration = int(state.get("iter") or 0)

        self.action_scale = _validate_action_scale(
            action_scale, num_actions=spec.num_actions).to(self.device)
        # 上一拍上层动作（clamp 到 ±1，**未缩放**）：首拍为零，与训练 `reset()` 一致。
        self._last_action = torch.zeros(self.num_envs, spec.num_actions,
                                        dtype=torch.float32, device=self.device)
        self.decoder_hidden = None
        self.reset()
        self._verify_forward()

    # ------------------------------------------------------------------ 契约自检
    def _verify_forward(self) -> None:
        """用一批零帧跑一次前向，确认导出/加载后的维数与有限性（与冻结策略同一做法）。"""
        zeros = torch.zeros(1, FRAME_DIM, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            estimate, latent, _ = self.decoder.forward_with_latent(zeros, None, sample=False)
            action = self.actor.act_inference(self._augment(zeros, estimate, latent))
        if tuple(estimate.shape) != (1, self.spec.explicit_dim):
            raise ValueError(f"decoder 零帧输出的 estimate 形状 {tuple(estimate.shape)} 不符")
        if tuple(latent.shape) != (1, self.spec.latent_dim):
            raise ValueError(f"decoder 零帧输出的 latent 形状 {tuple(latent.shape)} 不符")
        if tuple(action.shape) != (1, self.spec.num_actions):
            raise ValueError(f"actor 零帧输出的动作形状 {tuple(action.shape)} 不符")
        if not bool(torch.isfinite(action).all()):
            raise ValueError("actor 对零帧产生了非有限输出（权重/上下文的 dtype 有问题？）")
        # 探针会推进 actor 的 GRU 隐状态，清掉，让第一次 `act()` 从零状态起步。
        self.reset()

    # ------------------------------------------------------------------ 复位
    def reset(self, env_ids=None) -> None:
        """清零残差与 GRU 隐状态。

        本测试台每个 env 只跑一个 episode（不 reset），正常路径就是 `reset()`；`env_ids`
        保留给「部分环境重开」的复用场景（与 `Memory.reset(dones)` 同语义）。
        """
        if env_ids is None:
            self._last_action.zero_()
            self.decoder_hidden = None
            self.actor.memory_a.hidden_states = None
            self.actor.memory_c.hidden_states = None
            return
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).reshape(-1)
        self._last_action[ids] = 0.0
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        dones[ids] = True
        self.actor.reset(dones)
        self.decoder_hidden = self._reset_gru_hidden(self.decoder_hidden, dones)

    # ------------------------------------------------------------------ 帧组装
    def build_frame(self, *, loco_command, base_ang_vel, projected_gravity,
                    last_loco_action, joint_pos_rel, joint_vel) -> torch.Tensor:
        """按 `FRAME_TERMS` 组装 57 维 policy 帧（`last_action` 取本类持有的上一拍动作）。"""
        provided = {
            "loco_command": loco_command,
            "base_ang_vel": base_ang_vel,
            "projected_gravity": projected_gravity,
            "last_loco_action": last_loco_action,
            "joint_pos": joint_pos_rel,
            "joint_vel": joint_vel,
        }
        parts = []
        for name, dim, scale in FRAME_TERMS:
            value = self._last_action if name == "last_action" else provided[name]
            tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device)
            if tensor.ndim != 2 or tensor.shape[0] != self.num_envs or tensor.shape[1] != dim:
                raise ValueError(
                    f"帧项 {name!r} 形状 {tuple(tensor.shape)} 不符：期望 "
                    f"({self.num_envs}, {dim})")
            parts.append(tensor if scale == 1.0 else tensor * scale)
        frame = torch.cat(parts, dim=-1)
        if frame.shape[-1] != FRAME_DIM:
            raise ValueError(f"组装出的帧 {tuple(frame.shape)} != {FRAME_DIM} 维")
        return frame

    # ------------------------------------------------------------------ 推理
    def act(self, *, loco_command, base_ang_vel, projected_gravity,
            last_loco_action, joint_pos_rel, joint_vel):
        """一拍：返回 `(delta, processed)`；`processed` 进入下一拍的 `last_action` 位。"""
        frame = self.build_frame(
            loco_command=loco_command, base_ang_vel=base_ang_vel,
            projected_gravity=projected_gravity, last_loco_action=last_loco_action,
            joint_pos_rel=joint_pos_rel, joint_vel=joint_vel)
        # 用 `no_grad` 而不是 `inference_mode`：GRU 隐状态会**跨调用持有**，而 inference tensor
        # 在 InferenceMode 之外不能原地改（`Memory.reset(dones)` 正是原地清零）⇒ 部分 env
        # 复位会在 `reset(env_ids=...)` 处报 "Inplace update to inference tensor"。
        with torch.no_grad():
            estimate, latent, self.decoder_hidden = self.decoder.forward_with_latent(
                frame, self.decoder_hidden, sample=False)
            observation = self._augment(frame, estimate, latent)
            if self.deterministic:
                action = self.actor.act_inference(observation)
            else:
                action = self.actor.act(observation)
        processed = torch.clamp(action, -ACTION_CLIP, ACTION_CLIP)
        delta = processed * self.action_scale
        self._last_action.copy_(processed.detach())
        return delta, processed
