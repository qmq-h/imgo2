"""`mix` 受控测试场景的**命令项**：恒定前进指令 + 横向 PD + 航向 PD。

2026-10-04（用户）：任务 ``Imgo2-basemove-rough-cmoe-mix-test`` 的规格是「场景只有 `mix`、
速度只给前进、横向用 PD 外环、heading 保持 0」，且**不是新策略任务**（checkpoint 必须能直接加载）。

设计要点（逐条对应规格，详见 `docs/cmoe_mix_test_scene_2026-10-04.md`）：

1. **继承 Isaac Lab 的 `UniformVelocityCommand`**（复用它的 ``vel_command_b`` / ``heading_target`` /
   ``is_heading_env`` / ``is_standing_env`` / ``metrics`` 缓冲与 ``_update_metrics``），
   只重写 ``_resample_command`` / ``_update_command``。
2. ``vx_cmd``：**只在 resample 时采样一次**（本任务 cfg 把 ``ranges.lin_vel_x`` 设为 ``(1.0, 1.0)``
   且 ``resampling_time_range = (1.0e9, 1.0e9)``）⇒ 整回合恒为 **1.0 m/s**；``_update_command``
   不碰第 0 列。
3. **横向 PD**：``vy_cmd = clip(kp_y·(y_des − y_local) + kd_y·(0 − vy_local), ±vy_max)``；
   位置用**世界系**横向偏移（相对本环境出生原点）、速度用**世界系**横向速度（见 ``mix_test_pd``）。
4. **航向 PD**：``wz_cmd = clip(kp_h·(yaw_des − yaw) + kd_h·(0 − wz), ±wz_max)``；
   ``yaw`` 取 ``root_quat_w`` 的偏航角（Isaac Lab 的 ``heading_w`` 就是它），``wz`` 取**机体系**偏航角速度。
   ⚠️ **选择**：cfg 里 ``heading_command = True``（规格 ② 要求，且它是 `ranges.heading` 的合法性前提），
   但**本命令项不使用内置的 heading P 控制器**（``heading_control_stiffness`` 那条）—— 内置控制器只有
   P 项、且被 ``ranges.ang_vel_z`` 夹住（本任务该范围为 ``(0, 0)``，会把 wz 恒夹成 0）。
   本项每步直接用上面的 PD 写 ``vel_command_b[:, 2]``，等价于「关掉内置控制器、由本 PD 直接写 wz」。
5. ``vy_cmd``/``wz_cmd`` 写进 ``vel_command_b``（奖励与观测实际读的就是它）；另维护一个**世界系镜像**
   ``vel_command_w`` 供回放/审计读。⚠️ 基类与本仓均**没有** ``vel_command_w``，这是本项新增的纯附加缓冲，
   **不进入观测、不进入动作空间、不影响 checkpoint 兼容性**。
6. **纯函数化**：PD 数学在 ``mdp/mix_test_pd.py``（只依赖 torch），可离线单测。
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.utils import configclass

import imgo2_rl.tasks.manager_based.locomotion.velocity.mdp as mdp

from .mix_test_pd import (
    DEFAULT_KD_H,
    DEFAULT_KD_Y,
    DEFAULT_KP_H,
    DEFAULT_KP_Y,
    DEFAULT_VY_MAX,
    DEFAULT_WZ_MAX,
    DEFAULT_Y_DES,
    DEFAULT_YAW_DES,
    lateral_heading_pd,
    rotate_base_to_world,
)


class MixTestVelocityCommand(mdp.UniformVelocityCommand):
    """恒定前进 + 横向 PD + 航向 PD 的速度指令项（指令层外环，不改动作空间）。"""

    cfg: "MixTestVelocityCommandCfg"
    """The configuration of the command generator."""

    def __init__(self, cfg: "MixTestVelocityCommandCfg", env):
        super().__init__(cfg, env)
        # 世界系镜像（纯附加：基类没有这个缓冲）。见模块 docstring 第 5 点。
        self.vel_command_w = torch.zeros(self.num_envs, 3, device=self.device)

    def __str__(self) -> str:
        msg = "MixTestVelocityCommand（恒定前进 + 横向/航向 PD）:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tLateral PD: kp_y={self.cfg.kp_y}, kd_y={self.cfg.kd_y}, vy_max={self.cfg.vy_max}\n"
        msg += f"\tHeading PD: kp_h={self.cfg.kp_h}, kd_h={self.cfg.kd_h}, wz_max={self.cfg.wz_max}"
        return msg

    def _resample_command(self, env_ids: Sequence[int]):
        """采样**一次**前进速度（`lin_vel_x`，本任务恒 1.0 m/s）；横向/航向先置 0，随后由 PD 每步写。"""
        super()._resample_command(env_ids)
        self.vel_command_b[env_ids, 1] = 0.0
        self.vel_command_b[env_ids, 2] = 0.0

    def _update_command(self):
        """每个控制步都重算 PD（PD 是时变的），并写回 ``vel_command_b`` / ``vel_command_w``。"""
        data = self.robot.data
        # -- y_local：世界系横向位置相对**本环境出生原点**的偏移（＝离赛道中心线的横向距离）
        y_local = data.root_pos_w[:, 1] - self._env.scene.env_origins[:, 1]
        # -- vy_local：世界系横向线速度（与 y_local 同坐标系；也与 track_world_vel_xy_exp 一致）
        vy_local = data.root_lin_vel_w[:, 1]
        # -- yaw：root_quat_w 的偏航角（Isaac Lab 的 heading_w ＝ 把机体系 +x 投到世界系后 atan2）
        yaw = data.heading_w
        # -- wz_local：机体系偏航角速度（命令写的就是机体系角速度）
        wz_local = data.root_ang_vel_b[:, 2]

        vy_cmd, wz_cmd = lateral_heading_pd(
            y_local,
            vy_local,
            yaw,
            wz_local,
            kp_y=self.cfg.kp_y,
            kd_y=self.cfg.kd_y,
            vy_max=self.cfg.vy_max,
            y_des=self.cfg.y_des,
            kp_h=self.cfg.kp_h,
            kd_h=self.cfg.kd_h,
            wz_max=self.cfg.wz_max,
            yaw_des=self.cfg.yaw_des,
        )
        self.vel_command_b[:, 1] = vy_cmd
        self.vel_command_b[:, 2] = wz_cmd

        # 保持基类「站立环境指令归零」的语义（本任务 rel_standing_envs=0.0 ⇒ 恒不触发）。
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        if len(standing_env_ids) > 0:
            self.vel_command_b[standing_env_ids, :] = 0.0

        vx_w, vy_w = rotate_base_to_world(yaw, self.vel_command_b[:, 0], self.vel_command_b[:, 1])
        self.vel_command_w[:, 0] = vx_w
        self.vel_command_w[:, 1] = vy_w
        self.vel_command_w[:, 2] = self.vel_command_b[:, 2]


@configclass
class MixTestVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    """:class:`MixTestVelocityCommand` 的配置（PD 增益默认值＝用户给定值，可经 cfg 覆盖）。"""

    class_type: type = MixTestVelocityCommand

    kp_y: float = DEFAULT_KP_Y
    kd_y: float = DEFAULT_KD_Y
    vy_max: float = DEFAULT_VY_MAX
    y_des: float = DEFAULT_Y_DES
    kp_h: float = DEFAULT_KP_H
    kd_h: float = DEFAULT_KD_H
    wz_max: float = DEFAULT_WZ_MAX
    yaw_des: float = DEFAULT_YAW_DES
