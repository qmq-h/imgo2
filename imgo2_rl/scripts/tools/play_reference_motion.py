"""在 Isaac Lab 里播放参考动作轨迹（AMP 数据集），用于与策略回放做视觉与量化对比。

目的：训练日志只有情节级标量，无法判断策略的「高频步态」是「忠实复现参考」还是
「失控抖动」。把参考轨迹原样放出来，就能直接对照步频、关节幅值、抬脚高度。

用法（需要 GPU + 显示；无显示时加 --headless 只看打印的逐帧状态）：

    cd <工作区根>/imgo2_rl
    # 列出可用动作
    python scripts/tools/play_reference_motion.py --list
    # 播放指定动作（默认第一个行走动作）
    python scripts/tools/play_reference_motion.py --motion imgo2_forward_0.9
    # 加速播放（每渲染帧推进 5 个仿真帧）
    python scripts/tools/play_reference_motion.py --motion imgo2_forward_0.9 --speed 5

实现要点：
  * 坐标变换直接复用 `mdp.reset_amp_reference_state` 的口径（四元数 xyzw→wxyz、
    速度用 root_quat 旋进世界系、xy 加上 env_origins），避免两套写法漂移。
  * 每帧把参考状态**写回仿真**（root pose + 关节位置/速度），机器人被参考轨迹带着走，
    物理不参与——这样看到的就是纯粹的参考动作。
  * `--speed` 控制渲染推进（真实 0.02 s/帧，60 fps 渲染下 1 帧渲染 = 3 个仿真帧时约等速）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play an AMP reference motion inside Isaac Lab.")
parser.add_argument("--motion", type=str, default=None, help="动作名（不含 .txt）或 .txt 路径；默认第一个行走动作。")
parser.add_argument("--list", action="store_true", help="只列出可用动作后退出。")
parser.add_argument("--speed", type=int, default=3, help="每渲染帧推进的仿真帧数（默认 3 ≈ 实时）。")
parser.add_argument("--loops", type=int, default=0, help="循环次数，0 = 无限。")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import imgo2_rl.tasks  # noqa: F401, E402
from imgo2_rl.assets.imgo2 import AMP_MOTION_FILES  # noqa: E402
from isaaclab.utils import math as math_utils  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

TASK = "Imgo2-basemove-flat-amp-height-play"


def _available_motions() -> list[Path]:
    return sorted(Path(p) for p in AMP_MOTION_FILES)


def _pick_motion(name: str | None) -> Path:
    files = _available_motions()
    if not files:
        raise SystemExit(f"AMP_MOTION_FILES 为空，检查 assets/imgo2.py 的路径解析；当前 {AMP_MOTION_FILES!r}")
    if name is None:
        walking = [p for p in files if "stance" not in p.name]
        return (walking or files)[0]
    cand = Path(name)
    if cand.exists():
        return cand
    for p in files:
        if p.stem == name:
            return p
    raise SystemExit(f"未找到动作 {name}；可用：{', '.join(p.stem for p in files)}")


def main() -> int:
    files = _available_motions()
    if args_cli.list:
        print(f"可用动作 {len(files)} 个：")
        for p in files:
            print(f"  {p.stem}")
        return 0

    motion = _pick_motion(args_cli.motion)
    print(f"[ref] 播放 {motion.name}")

    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=1)
    env_cfg.scene.num_envs = 1
    # 播放参考轨迹时不希望被随机化干扰
    for name in ("randomize_apply_external_force_torque", "randomize_rigid_body_mass_base",
                 "randomize_rigid_body_mass_others", "randomize_rigid_body_material"):
        if getattr(env_cfg.events, name, None) is not None:
            setattr(env_cfg.events, name, None)
    env = gym.make(TASK, cfg=env_cfg).unwrapped
    env.reset()

    # 用 loader 直接读动作：轨迹选择按文件名匹配
    from rl_lab.datasets.motion_loader import AMPLoader

    loader = AMPLoader(
        env.device,
        time_between_frames=env.step_dt,
        preload_transitions=False,
        num_preload_transitions=0,
        motion_files=[str(motion)],
    )
    traj = 0
    duration = float(loader.trajectory_lens[traj])
    n_frames = int(loader.trajectories_full[traj].shape[0])
    print(f"[ref] 时长 {duration:.3f} s，{n_frames} 帧，dt={env.step_dt} s")

    robot = env.scene["robot"]
    # 必须用 AMP 环境显式声明的 12 个关节（不能用 ".*_joint"：模型共有 16 个关节，
    # 含四个固定的 *_Ankle_joint，宽度会与数据的 12 维不符）
    joint_names = list(env_cfg.joint_names)
    joint_ids, resolved = robot.find_joints(joint_names, preserve_order=True)
    assert len(joint_ids) == 12, f"期望 12 个 AMP 关节，实际 {len(joint_ids)}: {resolved}"
    env_ids = torch.arange(1, device=env.device)

    # 观测（供 --headless 时打印校核）
    def ref_state(t: float):
        frame = loader.get_full_frame_at_time(traj, t).to(env.device).unsqueeze(0)
        joint_pos = AMPLoader.get_joint_pose_batch(frame)
        joint_vel = AMPLoader.get_joint_vel_batch(frame)
        root_pos = AMPLoader.get_root_pos_batch(frame)
        root_quat = math_utils.convert_quat(AMPLoader.get_root_rot_batch(frame), to="wxyz")
        lin = math_utils.quat_apply(root_quat, AMPLoader.get_linear_vel_batch(frame))
        ang = math_utils.quat_apply(root_quat, AMPLoader.get_angular_vel_batch(frame))
        return joint_pos, joint_vel, root_pos, root_quat, lin, ang

    t = 0.0
    loops = 0
    printed = 0
    while simulation_app.is_running():
        for _ in range(max(1, args_cli.speed)):
            jp, jv, rp, rq, lv, av = ref_state(t)
            rp = rp.clone()
            rp[:, :2] += env.scene.env_origins[env_ids, :2]
            robot.write_root_pose_to_sim(torch.cat((rp, rq), dim=-1), env_ids=env_ids)
            robot.write_root_velocity_to_sim(torch.cat((lv, av), dim=-1), env_ids=env_ids)
            robot.write_joint_state_to_sim(jp, jv, joint_ids=joint_ids, env_ids=env_ids)
            # 这些 write_*_to_sim 直接写物理状态，无需再调 write_data_to_sim；
            # 推进一步让渲染器看到本帧，再刷新场景数据缓存。
            env.sim.step(render=True)
            env.scene.update(env.step_dt)
            t += env.step_dt
            if t >= duration:
                t = 0.0
                loops += 1
                if args_cli.loops and loops >= args_cli.loops:
                    env.close()
                    return 0

            if args_cli.headless and printed < 5:
                print(f"  t={t:5.2f}s  root_z={float(rp[0, 2]):.3f} m  "
                      f"joint_pos[:3]={[round(float(v), 3) for v in jp[0, :3]]}  "
                      f"joint_vel[:3]={[round(float(v), 2) for v in jv[0, :3]]}")
                printed += 1

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
