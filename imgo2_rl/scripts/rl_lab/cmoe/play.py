# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Play, record and export a checkpoint trained with the local CMoE stack."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Play a checkpoint trained with CMoE.")
parser.add_argument("--video", action="store_true", default=False, help="Record a playback video.")
parser.add_argument("--video_length", type=int, default=200, help="Recorded video length in steps.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--agent", type=str, default="cmoe_rsl_rl_cfg")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument(
    "--scan187",
    action="store_true",
    default=False,
    help="Replay a checkpoint trained with the legacy 187-D height scan (17x11 @ 1.6x1.0 m).",
)
parser.add_argument(
    "--terrain_level",
    type=int,
    default=None,
    help="把回放环境的**课程等级**钉死在 N（N ∈ 0..num_rows−1：训练/play 的 num_rows=20 ⇒ 0..19；"
         "评测场景 `...-mix-test` 的 num_rows=1 ⇒ **只有 0 合法**）并冻结课程升降级，"
         "用于检查指定难度下的地形；等级本身**不会**被夹到 9。"
         "默认 None＝沿用任务配置（`Imgo2CMoERoughPlayEnvCfg` 是 `max_init_terrain_level=5`，"
         "即随机落在 0–5 级；`...-mix-test` 的难度由 `terrain_generator.difficulty_range=(0.70, 0.70)` "
         "精确固定为 d=0.70，**不要**再传本参数）。",
)
parser.add_argument(
    "--prior",
    type=str,
    default=None,
    help="**[step-0 教师回放]** 不加载 CMoE checkpoint，改把一份 **45 维先验策略**（AMP 训练 checkpoint"
         " 或 play.py 导出的 TorchScript）当固定策略：只喂 CMoE 观测的前 45 维（当前帧本体感知），"
         "动作裁 ±`--prior_clip`。用于在花 GPU 时间做「初始化＋锚」之前，先看这份先验在 8 类地形列上"
         "能走多远。⚠️ 此时**不导出** policy.pt（先验不是 CMoE 网络）。"
         "见 docs/cmoe_trot_warmstart_2026-09-25.md。",
)
parser.add_argument(
    "--prior_clip",
    type=float,
    default=3.0,
    help="--prior 的动作裁剪上界（部署侧与训练侧动作项都是 ±3）。<=0 表示不裁。",
)
parser.add_argument(
    "--prior_report_every",
    type=int,
    default=0,
    help="每 N 步打印一次「每个地形列上的终止次数」（0＝关闭）。判断先验在哪一列活不下来用。",
)
parser.add_argument(
    "--steps",
    type=int,
    default=0,
    help="跑够 N 步就退出并打印汇总（0＝不限制，靠关窗口/--video_length 结束）。",
)
parser.add_argument(
    "--force_expert",
    type=int,
    default=None,
    metavar="K",
    help="**[诊断] 把门控强制 one-hot 到第 K 个专家**：混合输出严格等于该专家的输出 ⇒ 可以逐个专家回放，"
         "判断「5 个专家到底有没有功能分工」（配合 `--dump_gait` + `scripts/tools/expert_report.py`）。"
         "不设＝正常混合。见 docs/moe_references_2026-09-28.md。",
)
parser.add_argument(
    "--dump_gait",
    type=str,
    default=None,
    help="**[步态量测]** 把回放时每步/每环境的**四足触地序列**等 dump 成 npz，随后用"
         " `scripts/tools/gait_report.py --npz <该文件>` 离线算「每个地形上的占空比/周期/相位」"
         "并判定 trot/bound/pace。训练曲线里的三个核系数在 duty≈0.5 时会趋同（盲区），"
         "相位才是可分的量。见 docs/cmoe_trot_warmstart_2026-09-25.md §24。",
)
parser.add_argument(
    "--dump_steps",
    type=int,
    default=1000,
    help="--dump_gait 的采样步数（默认 1000；只有 --steps 也为 0 时才用它决定何时退出）。",
)
cli_args.add_cmoe_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import time

import gymnasium as gym
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import imgo2_rl.tasks  # noqa: F401
from rl_lab.config import CMoEOnPolicyRunnerCfg
from rl_lab.runners import CMoEOnPolicyRunner
from rl_lab.utils import export_cmoe_policy_as_jit, export_cmoe_policy_as_onnx
from rl_lab.utils.pretrained_prior import load_prior, prior_actions, teacher_actor
from rl_lab.utils.gait_dump import (
    GaitDumper,
    canonical_foot_indices,
    expand_terrain_names,
    format_gait_summary,
)
from rl_lab.wrapper import CMoEVecEnvWrapper


CONTACT_FORCE_THRESHOLD = 1.0   # 与 mdp/feet_air_time 的触地判据同量级（牛）


def build_gait_recorder(env, path: str, max_steps: int, gate_source=None):
    """构造回放的步态 recorder：返回 `(dumper, record, save)` 或 `(None, None, None)`。

    Isaac 侧的取数（接触传感器 → 契约足序、terrain_types/levels、指令、base 状态）都在这里；
    累加与 npz 落盘在 `rl_lab.utils.gait_dump.GaitDumper`（纯 numpy，可离线单测）。
    **任何异常都只警告并停用**：量测不该把回放弄挂。
    """
    try:
        from isaaclab.managers import SceneEntityCfg

        scene = env.unwrapped.scene
        sensor = scene["contact_forces"]
        spec = SceneEntityCfg("contact_forces", body_names=".*_FOOT")
        spec.resolve(scene)
        sensor_ids = list(spec.body_ids)
        sensor_names = [sensor.body_names[i] for i in sensor_ids]
        order, foot_names = canonical_foot_indices(sensor_names)
        feet_ids = [sensor_ids[i] for i in order]

        terrain = scene.terrain
        generator = env.unwrapped.cfg.scene.terrain.terrain_generator
        # ⚠️ 必须按比例铺成 num_cols 个逐列名：`sub_terrains` 只有 ~11 个键，而地形有 40 列
        # （2026-09-28 真实 dump 里列 11 之后全被显示成"列N"就是这个 bug）。
        terrain_names = expand_terrain_names(
            list(generator.sub_terrains.keys()),
            [generator.sub_terrains[key].proportion for key in generator.sub_terrains],
            generator.num_cols,
        )
        robot = scene["robot"]
        command_name = "base_velocity"
        dumper = GaitDumper(path, dt=float(env.unwrapped.step_dt), foot_names=foot_names,
                            terrain_names=terrain_names, max_steps=max_steps)
        print(f"[GAIT] 录制步态序列：足序 {foot_names}（契约 FL,FR,RL,RR）；"
              f"地形列 {len(terrain_names)} 个 {terrain_names}")
    except Exception as exc:  # noqa: BLE001 - 量测失败不影响回放
        print(f"[WARN] 无法启用步态 dump（{exc}）；本次回放只回放不量测")
        return None, None, None

    state = {"failed": False}

    def record() -> None:
        if state["failed"]:
            return
        try:
            with torch.inference_mode():
                forces = sensor.data.net_forces_w_history                  # [N, H, B, 3]
                magnitude = torch.linalg.vector_norm(forces, dim=-1).amax(dim=1)   # [N, B]
                contact = (magnitude[:, feet_ids] > CONTACT_FORCE_THRESHOLD).cpu().numpy()
                dumper.record(
                    contact,
                    base_height=robot.data.root_pos_w[:, 2],
                    base_lin_vel=robot.data.root_lin_vel_b,
                    cmd=env.unwrapped.command_manager.get_command(command_name),
                    terrain_type=terrain.terrain_types,
                    terrain_level=terrain.terrain_levels,
                    gate=None if gate_source is None else gate_source(),
                )
        except Exception as exc:  # noqa: BLE001 - 单步失败不打断回放
            print(f"[WARN] 步态 dump 的单步记录失败（{exc}）；后续不再记录")
            state["failed"] = True

    def save() -> None:
        try:
            summary = dumper.save()
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] 步态 dump 落盘失败（{exc}）")
            return
        print(format_gait_summary(
            summary,
            report_hint=(f"python3 scripts/tools/gait_report.py --npz {summary['path']}"),
        ))

    return dumper, record, save


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: CMoEOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_cmoe_cfg(agent_cfg, args_cli)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.scan187:
        # 2026-09-24 之前用 187 维高度扫描训出的 checkpoint，必须用旧几何才能 load_state_dict
        # （契约 637/235 vs 现在的 527/125）。这里在 env cfg 构造之后覆盖，所以不会触发
        # CMoE_env_cfg 里「必须是 77 条射线」的断言；只影响回放，不影响训练。
        env_cfg.scene.height_scanner.pattern_cfg.size = (1.6, 1.0)
        env_cfg.scene.height_scanner.offset.pos = (0.0, 0.0, 20.0)
        print("[INFO] --scan187: 使用旧几何 17x11=187 维高度扫描（仅用于回放旧 checkpoint）")
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.device = env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "cmoe", agent_cfg.experiment_name))
    if args_cli.prior:
        # step-0 教师回放：没有 CMoE checkpoint，也不导出策略；日志/视频落在 prior_play/。
        resume_path = None
        log_dir = os.path.join(log_root_path, "prior_play")
        os.makedirs(log_dir, exist_ok=True)
        print(f"[INFO] --prior 教师回放：日志/视频根目录 {log_dir}")
    else:
        if args_cli.checkpoint:
            resume_path = retrieve_file_path(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(
                log_root_path,
                agent_cfg.load_run,
                agent_cfg.load_checkpoint,
            )
        log_dir = os.path.dirname(resume_path)
        print(f"[INFO] Loading CMoE checkpoint: {resume_path}")
    env_cfg.log_dir = log_dir

    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )
    if args_cli.terrain_level is not None:
        # 2026-09-24：play 默认只从 0–5 级起步（`max_init_terrain_level=5`），单环境又几乎不会晋级，
        # 于是**永远看不到配置区间的上端**（台阶 15 cm / boxes 30 cm 要 level≈9）。这个开关把等级钉死。
        # ⚠️ 本机无 GPU、此路径**未经运行验证**；因此全程 fail-soft：失败只打印告警，回放照常进行。
        try:
            terrain = env.unwrapped.scene.terrain
            origins = getattr(terrain, "terrain_origins", None)
            if origins is None or getattr(terrain, "terrain_levels", None) is None:
                print("[WARN] --terrain_level: 该任务不是课程地形（terrain_type != generator），忽略")
            else:
                level = int(args_cli.terrain_level)
                terrain.terrain_levels[:] = level
                terrain.env_origins[:] = origins[terrain.terrain_levels, terrain.terrain_types]
                # 冻结课程：否则回合结束仍会按判据升降级（一局之内就飘走）
                try:
                    manager = env.unwrapped.curriculum_manager
                    for index, name in enumerate(manager.active_terms):
                        if name == "terrain_levels":
                            manager._term_cfgs[index].func = lambda *args, **kwargs: {}
                    print("[INFO] --terrain_level: 已冻结课程（terrain_levels → no-op）")
                except Exception as exc:  # noqa: BLE001 - 调试开关，失败不影响回放
                    print(f"[WARN] --terrain_level: 课程未冻结（{exc}），等级可能在一局内变化")
                env.unwrapped.reset()  # 让机器人按新的 env_origins 重新出生
                print(f"[INFO] --terrain_level={level}: 已把全部环境的地形等级钉死在 {level}")
        except Exception as exc:  # noqa: BLE001 - 调试开关，失败不影响回放
            print(f"[WARN] --terrain_level 失败（忽略，继续按默认等级回放）：{exc}")
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = CMoEVecEnvWrapper(env, history_steps=agent_cfg.history_steps)
    print(
        "[INFO] CMoE observations: "
        f"policy={env.num_one_step_obs}, history_steps={env.history_steps}, "
        f"terrain={env.num_terrain_obs}, actor_total={env.num_obs}, "
        f"critic={env.num_privileged_obs}, actions={env.num_actions}"
    )

    if args_cli.prior:
        # step-0 教师回放：先验只吃当前帧本体感知（45 维），不碰 CMoE 网络、不导出 policy.pt。
        prior = load_prior(args_cli.prior)
        if prior.actor_obs_dim != env.num_one_step_obs:
            raise ValueError(
                f"先验输入维 {prior.actor_obs_dim} != CMoE 当前帧本体感知 {env.num_one_step_obs}；"
                "契约不符，先跑 scripts/tools/check_cmoe_expert_init.py 核对"
            )
        policy = teacher_actor(prior).to(env.device).eval()
        print(
            f"[INFO] --prior: {prior.summary()}；只喂观测前 {env.num_one_step_obs} 维（当前帧本体感知），"
            f"动作裁 ±{args_cli.prior_clip}"
        )
    else:
        runner = CMoEOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(resume_path, load_optimizer=False)
        policy = runner.get_inference_policy(device=env.device)

        export_dir = os.path.join(log_dir, "exported")
        export_cmoe_policy_as_jit(runner.alg.actor_critic, export_dir, "policy.pt")
        export_cmoe_policy_as_onnx(runner.alg.actor_critic, export_dir, "policy.onnx")

    if args_cli.force_expert is not None:
        actor_critic = getattr(runner.alg, "actor_critic", None) if runner is not None else None
        if actor_critic is None or not hasattr(actor_critic, "experts"):
            print("[WARN] --force_expert 需要 CMoE actor_critic（有 experts 属性）；本次忽略")
        else:
            num_experts = len(actor_critic.experts)
            if not 0 <= args_cli.force_expert < num_experts:
                print(f"[WARN] --force_expert={args_cli.force_expert} 超出 [0,{num_experts})；本次忽略")
            else:
                actor_critic.forced_expert = int(args_cli.force_expert)
                print(f"[INFO] --force_expert={args_cli.force_expert}：门控被强制 one-hot 到该专家"
                      "（混合输出＝该专家单独的输出）")

    observations = env.get_observations()
    dt = env.unwrapped.step_dt

    # 逐列终止统计：`--prior`（step-0）默认打开，普通回放默认关闭、行为与以前一致。
    stats_enabled = bool(args_cli.prior) or args_cli.prior_report_every > 0
    num_cols = 10
    try:
        num_cols = int(env.unwrapped.cfg.scene.terrain.terrain_generator.num_cols)
    except Exception as exc:  # noqa: BLE001 - 只用于统计，取不到就按 10 列
        print(f"[WARN] 取不到地形列数（{exc}），按 {num_cols} 列统计")
    done_counts = torch.zeros(num_cols, dtype=torch.long, device=env.device)

    def report(step: int) -> None:
        if not stats_enabled:
            return
        counts = " ".join(f"列{i}={int(v)}" for i, v in enumerate(done_counts.tolist()))
        print(f"[REPORT] step={step} 各列终止次数: {counts} | 合计 {int(done_counts.sum())}")

    timestep = 0

    # 步态量测（2026-09-28）：`--dump_gait <npz>` 时逐控制步记录足端接触等，退出前落盘，
    # 之后用 scripts/tools/gait_report.py 离线算「每地形占空比/周期/相位」并判 trot/bound/pace。
    gait_dumper = gait_record = gait_save = None
    if args_cli.dump_gait:
        if args_cli.dump_gait.lower().endswith(".npz") is False:
            args_cli.dump_gait = args_cli.dump_gait + ".npz"
        gait_dumper, gait_record, gait_save = build_gait_recorder(
            env, args_cli.dump_gait, args_cli.dump_steps,
            # 门控权重是 softmax 后的概率（CMoEActorCritic.gating_network 末层 softmax）⇒ 可直接交叉表
            gate_source=lambda: runner.alg.actor_critic.gate_weights,
        )
        # 只给了 --dump_gait 而没给 --steps 时，顺手让它跑够 dump_steps 就退出并落盘
        if gait_dumper is not None and not args_cli.steps and not args_cli.video:
            args_cli.steps = int(args_cli.dump_steps)
            print(f"[GAIT] 未指定 --steps ⇒ 采满 --dump_steps={args_cli.dump_steps} 步后退出并落盘")

    gait_saved = False

    def finish_gait() -> None:
        nonlocal gait_saved
        if gait_save is not None and not gait_saved:
            gait_saved = True
            gait_save()

    print("[INFO] Starting CMoE playback...")
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            if args_cli.prior:
                actions = prior_actions(policy, observations, env.num_one_step_obs, args_cli.prior_clip)
            else:
                actions = policy(observations)
            observations, _, _, dones, _, _, _ = env.step(actions)
        timestep += 1
        if gait_record is not None and not gait_saved:
            gait_record()
        if stats_enabled:
            try:
                if bool(dones.any()):
                    column_ids = env.unwrapped.scene.terrain.terrain_types[dones.bool()].to(torch.long)
                    done_counts += torch.bincount(column_ids, minlength=num_cols)[:num_cols]
            except Exception as exc:  # noqa: BLE001 - 统计失败不影响回放
                if timestep == 1:
                    print(f"[WARN] 逐列终止统计不可用（{exc}），只回放不统计")
        if args_cli.prior_report_every > 0 and timestep % args_cli.prior_report_every == 0:
            report(timestep)
        if args_cli.steps and timestep >= args_cli.steps:
            report(timestep)
            print(f"[INFO] --steps={args_cli.steps} 已跑满，退出回放")
            finish_gait()
            break
        if args_cli.video and timestep >= args_cli.video_length:
            report(timestep)
            finish_gait()
            break
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    finish_gait()      # 关窗口/异常退出等其余路径也要落盘
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
