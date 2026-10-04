"""`mix` 受控测试场景的**横向/航向 PD 外环**——纯数学，只依赖 ``torch``，可离线单测。

2026-10-04（用户）：新增「继承 play 任务」的测试场景任务
``Imgo2-basemove-rough-cmoe-mix-test``：场景里**只有 `mix` 一种地形**、**速度指令只给前进**
（默认恒定 **1.0 m/s**）、**横向用 PD 外环控制**、**heading 保持 0**。目的：把「横向漂移」与
「航向漂移」从评测里剔除，只考察策略通过复合障碍（`mix`）的能力。

本文件只放**纯函数**（不 import isaaclab、不碰 env），因此能在没有 Isaac Lab / GPU 的机器上
直接跑单测；命令项类（`MixTestVelocityCommand`）在同目录的 ``mix_test_command.py`` 里，
它把这里的公式接到 ``root_pos_w`` / ``root_lin_vel_w`` / ``heading_w`` / ``root_ang_vel_b`` 上。

公式（两条都是**指令层**外环，输出直接写进 ``vel_command_b[:, 1]`` 与 ``vel_command_b[:, 2]``，
**不改动作空间、不改观测契约**）::

    vy_cmd = clip(kp_y · (y_des − y) + kd_y · (0 − vy), −vy_max, +vy_max)
    wz_cmd = clip(kp_h · (yaw_des − yaw) + kd_h · (0 − wz), −wz_max, +wz_max)

默认增益（用户给定，可经 cfg 字段覆盖）::

    kp_y = 1.0   kd_y = 0.3   vy_max = 0.6   （m/s）
    kp_h = 1.5   kd_h = 0.3   wz_max = 1.0   （rad/s）
    y_des = 0.0  yaw_des = 0.0

⚠️ 约定（与文档一致，改动请同步 `docs/cmoe_mix_test_scene_2026-10-04.md`）：

* ``y``   = **世界系**横向位置相对**本环境出生原点**的偏移（``root_pos_w[:, 1] − env_origins[:, 1]``）；
* ``vy``  = **世界系**横向线速度（``root_lin_vel_w[:, 1]``）——与 ``y`` 同坐标系，也与评测奖励
  ``track_world_vel_xy_exp``（世界系）一致；
* ``yaw`` = 基座偏航角（``heading_w``，由 ``root_quat_w`` 把机体系 +x 投到世界系后 ``atan2`` 得到）；
* ``wz``  = **机体系**偏航角速度（``root_ang_vel_b[:, 2]``）——命令写的是机体系角速度，
  且既有的 ``error_vel_yaw`` 度量用的也是 ``root_ang_vel_b[:, 2]``。
"""

from __future__ import annotations

import torch

# ------------------------------------------------------------------ 默认增益（用户给定）
DEFAULT_KP_Y = 1.0
DEFAULT_KD_Y = 0.3
DEFAULT_VY_MAX = 0.6
DEFAULT_KP_H = 1.5
DEFAULT_KD_H = 0.3
DEFAULT_WZ_MAX = 1.0
DEFAULT_Y_DES = 0.0
DEFAULT_YAW_DES = 0.0


def lateral_heading_pd(
    y: torch.Tensor,
    vy: torch.Tensor,
    yaw: torch.Tensor,
    wz: torch.Tensor,
    kp_y: float = DEFAULT_KP_Y,
    kd_y: float = DEFAULT_KD_Y,
    vy_max: float = DEFAULT_VY_MAX,
    y_des: float = DEFAULT_Y_DES,
    kp_h: float = DEFAULT_KP_H,
    kd_h: float = DEFAULT_KD_H,
    wz_max: float = DEFAULT_WZ_MAX,
    yaw_des: float = DEFAULT_YAW_DES,
) -> tuple[torch.Tensor, torch.Tensor]:
    """横向 + 航向 PD，返回 ``(vy_cmd, wz_cmd)``（与输入同形状的 tensor，已按上限夹住）。

    Args:
        y: 世界系横向位置误差的**被测值**（相对出生原点的 y 偏移，正=偏左/偏 +y）。
        vy: 世界系横向线速度（阻尼项用 −kd_y·vy）。
        yaw: 基座偏航角（rad，``heading_w``）。
        wz: 机体系偏航角速度（rad/s，阻尼项用 −kd_h·wz）。
        kp_y/kd_y/vy_max: 横向 PD 的比例、微分增益与输出上限。
        y_des: 横向位置目标（默认 0.0 ⇒ 贴着赛道中心线）。
        kp_h/kd_h/wz_max: 航向 PD 的比例、微分增益与输出上限。
        yaw_des: 航向目标（默认 0.0 ⇒ 始终朝世界 +x）。

    Returns:
        ``(vy_cmd, wz_cmd)``：两个已 ``clip`` 的 tensor；``y=0, vy=0, yaw=0, wz=0`` 时两者都是 0。
    """
    vy_cmd = torch.clip(kp_y * (y_des - y) + kd_y * (0.0 - vy), -vy_max, vy_max)
    wz_cmd = torch.clip(kp_h * (yaw_des - yaw) + kd_h * (0.0 - wz), -wz_max, wz_max)
    return vy_cmd, wz_cmd


def rotate_base_to_world(yaw: torch.Tensor, vx_b: torch.Tensor, vy_b: torch.Tensor):
    """把机体系水平速度指令旋转到世界系（仅供 ``vel_command_w`` 镜像与离线审计用）。

    纯数学，不参与奖励与观测：本任务的奖励（``track_world_vel_xy_exp`` 等）读的仍是
    ``command``（＝``vel_command_b``）；``vel_command_w`` 只是给回放/审计一个「世界系目标」的读数。
    """
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    vx_w = cos_yaw * vx_b - sin_yaw * vy_b
    vy_w = sin_yaw * vx_b + cos_yaw * vy_b
    return vx_w, vy_w
