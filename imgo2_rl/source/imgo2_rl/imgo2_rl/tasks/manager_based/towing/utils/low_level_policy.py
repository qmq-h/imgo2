"""Frozen low-level policy adapter: velocity command -> joint position targets.

计划 P4：「先完全不改 locomotion」——把已有底层当冻结策略使用。本模块把 policy_cfg
声明的契约应用到 torch 张量上，负责

1. 按契约顺序组装观测（顺序/缩放只写在 `policy_cfg.py`，这里只walk布局）；
2. 调 TorchScript 策略（`play.py` 导出的 `policy.pt`，normalizer=None）；
3. 裁剪动作并换算成关节位置目标 `default_dof_pos + action_scale * clip(action)`。

语义按部署侧（已在 Gazebo 用三条契约验证）对齐：

- `last_action` 是**裁剪后**的网络输出（部署 `obs.actions = Forward()`，而 `Forward()`
  末尾就 clamp，`ComputeOutput()` 用的也是这份值）；
- 观测在拼接后整体 clip 到 ±`clip_obs`，再喂网络；
- 关节状态按**策略顺序**传入（AMP 的策略顺序是逐腿，而 Isaac Lab 的资产顺序是广度优先，
  必须先用 `LowLevelPolicyCfg.asset_permutation()` 重排）。

重导入（torch / Isaac Lab）只发生在本模块，`policy_cfg.py` 仍是纯标准库，便于离线校验。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch

from .policy_cfg import LowLevelPolicyCfg


class LowLevelOutput(NamedTuple):
    """一步的输出：观测、裁剪后动作、以及换算好的关节位置目标。"""

    observation: torch.Tensor
    action: torch.Tensor
    joint_targets: torch.Tensor


@dataclass
class _Parts:
    base_ang_vel: torch.Tensor
    projected_gravity: torch.Tensor
    velocity_command: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor


class FrozenLowLevelPolicy:
    """加载 `play.py` 导出的 TorchScript 策略，按契约喂观测、出动作。"""

    def __init__(self, cfg: LowLevelPolicyCfg, *, device: str = "cuda:0"):
        cfg.validate()
        self.cfg = cfg
        self.device = torch.device(device)
        path = cfg.model_path
        if not path.is_file():
            raise FileNotFoundError(
                f"找不到 {cfg.name} 的导出策略：{path}（用对应算法的 play.py --headless 导出）")
        self._model = torch.jit.load(str(path), map_location=self.device)
        self._model.eval()
        self._default_dof_pos = torch.tensor(cfg.default_dof_pos, dtype=torch.float32,
                                             device=self.device)
        self._action_scale = torch.tensor(cfg.action_scale, dtype=torch.float32,
                                         device=self.device)
        self._clip_lower = torch.tensor(cfg.clip_actions_lower, dtype=torch.float32,
                                        device=self.device)
        self._clip_upper = torch.tensor(cfg.clip_actions_upper, dtype=torch.float32,
                                        device=self.device)
        self._last_action = torch.zeros(cfg.num_joints, dtype=torch.float32, device=self.device)
        self._verify_contract()

    # -------------------------------------------------------------- 契约自检
    def _verify_contract(self):
        """用一批零输入做一次前向，确认导出件的输入/输出维数与契约一致。

        这是部署侧「三条契约」的第一条在训练侧的等价物：维数不对时应当**在这里**
        报错，而不是在拖曳跑了一半时出现形状不匹配。
        """
        probe = torch.zeros(1, self.cfg.num_observations, dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            out = self._model(probe)
        if out.shape != (1, self.cfg.num_joints):
            raise ValueError(
                f"{self.cfg.name} 导出件的输出形状 {tuple(out.shape)} 与契约的 "
                f"(1, {self.cfg.num_joints}) 不一致")
        if not torch.isfinite(out).all():
            raise ValueError(f"{self.cfg.name} 导出件对零输入产生了非有限输出")

    # -------------------------------------------------------------- reset 契约
    def reset(self, env_ids=None) -> int:
        """回到契约规定的初始状态，返回建议的「站定」步数。

        `last_action` 归零与部署一致（部署在 `RL::InitRL()` 里把 actions 清零）。
        `reset_settle_s` 期间不应施加速度指令：参考部署的 FSM 也是先 GetUp/站定再交给策略。
        """
        if env_ids is None or self._last_action.ndim == 1:
            self._last_action.zero_()
        else:
            self._last_action[env_ids] = 0
        return int(round(self.cfg.reset_settle_s / self.cfg.control_dt))

    # -------------------------------------------------------------- 观测组装
    def build_observation(self, parts: _Parts) -> torch.Tensor:
        cfg = self.cfg
        batch = parts.base_ang_vel.shape[0]
        derived = {
            "base_ang_vel": parts.base_ang_vel,
            "projected_gravity": parts.projected_gravity,
            "velocity_commands": parts.velocity_command.expand(batch, 3),
            "joint_pos_rel": parts.joint_pos - self._default_dof_pos,
            "joint_vel_rel": parts.joint_vel,
            "last_action": self._last_action.expand(batch, cfg.num_joints),
        }
        chunks = []
        for term, start, dim in cfg.observation_layout():
            if term not in derived:
                raise KeyError(f"契约声明了观测项 '{term}'，但适配器不认识它（契约与实现漂移）")
            value = derived[term]
            if value.shape[-1] != dim:
                raise ValueError(f"观测项 '{term}' 维数 {value.shape[-1]} != 契约的 {dim}")
            scale = torch.tensor(cfg.observation_scales[start:start + dim],
                                 dtype=torch.float32, device=self.device)
            chunks.append(value * scale)
        observation = torch.cat(chunks, dim=-1)
        return observation.clamp(-cfg.clip_obs, cfg.clip_obs)

    def step(self, parts: _Parts) -> LowLevelOutput:
        """一步推理；`parts` 必须已经按策略顺序排好关节量。"""
        observation = self.build_observation(parts)
        with torch.inference_mode():
            raw = self._model(observation)
        action = torch.maximum(torch.minimum(raw, self._clip_upper), self._clip_lower)
        self._last_action = action.detach().reshape(-1) if action.shape[0] == 1 else action.detach()
        targets = self._default_dof_pos + self._action_scale * action
        return LowLevelOutput(observation=observation, action=action, joint_targets=targets)


def parts_from_robot_state(*, base_ang_vel, projected_gravity, velocity_command,
                           joint_pos, joint_vel) -> _Parts:
    """构造观测部件；关节量需已按策略顺序（用 asset_permutation() 重排后再传入）。

    独立成函数是为了让「哪些量、什么顺序、什么坐标系」在调用处一眼可见：
    `base_ang_vel` 是**体系**角速度，`projected_gravity` 是体系重力方向（单位向量）。
    """
    return _Parts(base_ang_vel=base_ang_vel, projected_gravity=projected_gravity,
                  velocity_command=velocity_command, joint_pos=joint_pos, joint_vel=joint_vel)
