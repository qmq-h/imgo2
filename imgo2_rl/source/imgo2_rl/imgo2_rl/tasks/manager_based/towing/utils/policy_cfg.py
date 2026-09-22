"""Frozen low-level policy contracts for the towing experiments (P4+).

计划 P4 要求「先完全不改 locomotion」：把已有底层当**冻结**策略使用，只喂速度指令、
不更新参数。AMP 与 PPO 两个底层的观测顺序/缩放、动作缩放、控制周期与 reset 契约
**不默认相同**（架构记录 §5），因此每个策略一份显式 cfg，不共享隐式默认值。

本模块**只用标准库**：它声明契约并提供派生量（观测布局、缩放向量、动作→关节目标），
供离线校验与运行端共用。真正的张量组装与 torch 推理在 `low_level_policy.py`。

契约的权威来源是部署侧配置 `imgo2_deploy/policy/imgo2/<name>/config.yaml` ——
那份配置已经过 Gazebo 三条契约验证（导出件/checkpoint actor、libtorch 加载、接口维数），
`tests/test_towing_policy_contract.py` 会逐项与本文件交叉核对，防止两处漂移。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def _is_sequence_of_numbers(values) -> bool:
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)


def repo_root() -> Path:
    """按 `imgo2_description/` 标记目录向上查找仓库根。

    不用 `parents[n]`：本文件在 `tasks/manager_based/towing/utils/` 下，
    与 `assets/` 下的同级模块深度不同，写死下标很容易算错（且移动文件即失效）。
    """
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "imgo2_description").is_dir():
            return candidate
    raise RuntimeError("找不到仓库根：向上没有含 imgo2_description/ 的目录")


@dataclass(frozen=True)
class LowLevelPolicyCfg:
    """一个冻结底层的完整接口契约。

    `observation_terms` / `observation_dims` / `observation_scales` 三者按位置对齐；
    缩放按**观测项**给出（部署 yaml 的语义），展开后用 `observation_scales` 表示。
    """

    name: str
    model_name: str
    policy_dir: Path
    num_observations: int
    observation_terms: tuple[str, ...]
    observation_dims: tuple[int, ...]
    observation_scales: tuple[float, ...]
    clip_obs: float
    observation_history: tuple[int, ...]
    num_joints: int
    joint_names: tuple[str, ...]
    default_dof_pos: tuple[float, ...]
    action_scale: tuple[float, ...]
    clip_actions_lower: tuple[float, ...]
    clip_actions_upper: tuple[float, ...]
    joint_mapping: tuple[int, ...]
    control_dt: float
    reset_settle_s: float

    # ------------------------------------------------------------------ 派生量
    @property
    def model_path(self) -> Path:
        return self.policy_dir / self.model_name

    @property
    def total_observation_dim(self) -> int:
        return sum(self.observation_dims)

    def observation_layout(self) -> tuple[tuple[str, int, int], ...]:
        """返回 (观测项, 起始下标, 维数)，顺序即拼接顺序。"""
        layout, start = [], 0
        for term, dim in zip(self.observation_terms, self.observation_dims):
            layout.append((term, start, dim))
            start += dim
        return tuple(layout)

    def joint_targets(self, action) -> tuple:
        """关节位置目标 = default_dof_pos + action_scale * action（与部署一致）。

        传入的 action 必须是**已裁剪**的网络输出：部署侧 `obs.actions = Forward()`
        而 `Forward()` 末尾就是 clamp，`ComputeOutput()` 用的也是这份值。
        """
        if len(action) != self.num_joints:
            raise ValueError(f"Expected {self.num_joints} actions, got {len(action)}")
        return tuple(default + scale * value
                     for default, scale, value in zip(self.default_dof_pos, self.action_scale, action))

    def asset_permutation(self, asset_joint_names) -> tuple[int, ...]:
        """给出「策略第 i 个关节 ↔ **资产数组**第 perm[i] 号」的置换（供 Isaac Lab 用）。

        为什么需要它：Isaac Lab / PhysX 的 DOF 顺序是按运动学树**广度优先**排的，本模型
        实测为 `FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, ...`（全部 hip → 全部 thigh →
        全部 shank），**不是** URDF 的声明顺序（URDF 里是逐腿 FL hip/thigh/shank, FR …）。
        而 AMP 任务把策略的观测与动作 `joint_names` **显式**设成逐腿顺序
        （`amp_env_cfg.py` 的 `self.actions.joint_pos.joint_names = self.joint_names` 以及
        `observations.policy.joint_{pos,vel}` 两处），所以策略顺序 = 逐腿顺序。
        两者不重排就会把观测拼错、把动作打到错误的关节上。
        """
        asset = list(asset_joint_names)
        if len(set(asset)) != len(asset):
            raise ValueError("资产关节名有重复，无法建立唯一映射")
        index = {name: position for position, name in enumerate(asset)}
        missing = [name for name in self.joint_names if name not in index]
        unknown = [name for name in asset if name not in set(self.joint_names)]
        if missing or unknown:
            raise ValueError(
                f"模型关节与策略契约的关节集合不一致：契约缺 {missing}，模型多 {unknown}")
        return tuple(index[name] for name in self.joint_names)

    # ------------------------------------------------------------------ 校验
    def validate(self) -> None:
        """契约自洽性检查；任何一条不成立都直接报错，不静默降级。"""
        problems = []
        if len(self.observation_terms) != len(self.observation_dims):
            problems.append("observation_terms 与 observation_dims 长度不一致")
        if self.total_observation_dim != self.num_observations:
            problems.append(f"观测维数合计 {self.total_observation_dim} != num_observations {self.num_observations}")
        if len(self.observation_scales) != self.num_observations:
            problems.append(f"observation_scales 长度 {len(self.observation_scales)} != {self.num_observations}")
        for label, values, expected in (
            ("joint_names", self.joint_names, self.num_joints),
            ("default_dof_pos", self.default_dof_pos, self.num_joints),
            ("action_scale", self.action_scale, self.num_joints),
            ("clip_actions_lower", self.clip_actions_lower, self.num_joints),
            ("clip_actions_upper", self.clip_actions_upper, self.num_joints),
            ("joint_mapping", self.joint_mapping, self.num_joints),
            ("observation_scales", self.observation_scales, self.num_observations),
        ):
            if len(values) != expected:
                problems.append(f"{label} 长度 {len(values)} != {expected}")
            elif label != "joint_names" and not _is_sequence_of_numbers(values):
                problems.append(f"{label} 含非数值项")
        if not all(isinstance(name, str) and name for name in self.joint_names):
            problems.append("joint_names 必须是非空字符串")
        if len(set(self.joint_names)) != len(self.joint_names):
            problems.append("joint_names 有重复")
        if sorted(self.joint_mapping) != list(range(self.num_joints)):
            problems.append("joint_mapping 必须是 0..num_joints-1 的置换")
        if not self.clip_obs > 0:
            problems.append("clip_obs 必须为正")
        if not self.control_dt > 0:
            problems.append("control_dt 必须为正")
        if self.reset_settle_s < 0:
            problems.append("reset_settle_s 不能为负")
        if any(lo >= hi for lo, hi in zip(self.clip_actions_lower, self.clip_actions_upper)):
            problems.append("clip_actions_lower 必须逐项小于 upper")
        if any(scale <= 0 for scale in self.action_scale):
            problems.append("action_scale 必须全为正")
        if any(scale <= 0 for scale in self.observation_scales):
            problems.append("observation_scales 必须全为正")
        if any(len(v) != 3 for v in self.observation_history):
            problems.append("observation_history 目前只支持 (term_index, dim, lag) 三元组")
        if problems:
            raise ValueError(f"LowLevelPolicyCfg '{self.name}' 契约不成立: " + "; ".join(problems))


# ---------------------------------------------------------------------- AMP
# 部署契约：imgo2_deploy/policy/imgo2/amp/config.yaml（45 维 actor）。
# 观测顺序取自训练侧 velocity_env_cfg.py 的 PolicyCfg，去掉 base_lin_vel（AMP-05）。
_AMP_JOINT_NAMES = (
    "FL_hip_joint", "FL_thigh_joint", "FL_shank_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_shank_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_shank_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_shank_joint",
)

AMP_POLICY = LowLevelPolicyCfg(
    name="amp",
    model_name="policy.pt",
    policy_dir=repo_root() / "imgo2_deploy" / "policy" / "imgo2" / "amp",
    num_observations=45,
    observation_terms=("base_ang_vel", "projected_gravity", "velocity_commands",
                       "joint_pos_rel", "joint_vel_rel", "last_action"),
    observation_dims=(3, 3, 3, 12, 12, 12),
    # ang_vel 0.25 / gravity 1.0 / commands 1.0 / dof_pos 1.0 / dof_vel 0.05 / last_action 1.0
    observation_scales=((0.25,) * 3 + (1.0,) * 3 + (1.0,) * 3
                        + (1.0,) * 12 + (0.05,) * 12 + (1.0,) * 12),
    clip_obs=100.0,
    observation_history=(),
    num_joints=12,
    joint_names=_AMP_JOINT_NAMES,
    default_dof_pos=(0.0, 0.87, -1.82) * 4,
    action_scale=(0.125, 0.25, 0.25) * 4,
    clip_actions_lower=(-3.0,) * 12,
    clip_actions_upper=(3.0,) * 12,
    joint_mapping=tuple(range(12)),
    # Policy inference period. Deployment uses a 5 ms physics/control step with decimation=4,
    # and the training environment has the same 0.005 * 4 = 0.02 s step_dt.
    control_dt=0.02,
    # reset 契约：策略在置位后需要先站定再接管。具体时长在 P4 仿真里确认，
    # 这里给一个显式的初值而不是隐式默认，避免「reset 后立刻按速度指令」。
    reset_settle_s=1.0,
)

POLICIES = {policy.name: policy for policy in (AMP_POLICY,)}


def get_policy(name: str) -> LowLevelPolicyCfg:
    try:
        policy = POLICIES[name]
    except KeyError:
        raise KeyError(f"Unknown frozen policy '{name}'; available: {sorted(POLICIES)}") from None
    policy.validate()
    return policy
