#!/usr/bin/env python3
"""Run paired natural-support target replay and own-feedback policy trials."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "imgo2_deploy/logs/new_motion_20261121_source"
DOCS = ROOT / "docs/deployment"
KINDS = ("hip", "thigh", "shank")
JOINTS = [f"{leg}_{kind}_joint" for leg in ("FL", "FR", "RL", "RR") for kind in KINDS]


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def axes(stem):
    return [f"{stem}_{i:02}" for i in range(12)]


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def read_walk_inputs():
    windows = [r for r in read_csv(DOCS / "new_motion_2026-11-21_walk.csv")
               if r["session"].endswith("_ground")]
    if len(windows) != 4:
        raise ValueError("Expected four previously selected ground windows")
    windows.sort(key=lambda r: (r["session"], r["direction"]))
    inputs = []
    for w in windows:
        path = SOURCE / w["session"] / "robot.csv"
        start, end = float(w["start_s"]), float(w["end_s"])
        rows = read_csv(path)
        used = [r for r in rows if start - 0.06 <= float(r["t_mono_ms"]) / 1000 <= end + 0.06]
        times = [float(r["t_mono_ms"]) / 1000 - start for r in used]
        if not times[0] <= 0 < 5.0 <= times[-1] or not all(0 < b-a < .05 for a,b in zip(times,times[1:])):
            raise ValueError(f"Incomplete command trace: {w}")
        for row in used:
            if (int(row["mode"]) != 3 or int(row["valid_mask"]) != 4095 or int(row["err_mask"]) != 0
                or any(float(row[k]) != 25 for k in axes("kp"))
                or any(float(row[k]) != .8 for k in axes("kd"))
                or any(float(row[k]) != 0 for k in axes("dq_des") + axes("tau_ff"))):
                raise ValueError(f"Unexpected gain, invalid feedback or mode: {w}")
        inputs.append((w, path, used, times))
    return inputs


def run(args):
    import bisect
    import numpy as np
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import DCMotor
    from isaaclab.assets import Articulation
    from isaaclab.utils.math import quat_apply

    sys.path.insert(0, str(ROOT / "imgo2_rl/source/imgo2_rl"))
    from imgo2_rl.assets import imgo2, imgo2_actuators
    from imgo2_rl.assets.imgo2 import IMGO2_CFG, URDF_PATH, make_imgo2_resistance_cfg
    from analyze_standup_replay import ground_estimate

    dt = args.physics_dt
    obs_stride, log_stride = round(.02 / dt), round(.005 / dt)
    if not (1 <= log_stride and math.isclose(obs_stride * dt, .02) and math.isclose(log_stride * dt, .005)):
        raise ValueError("Physics dt must divide both 5ms and 20ms")

    identity = json.loads((DOCS / "new_motion_2026-11-21_policy_all_rows.json").read_text())
    policy_file = ROOT / identity["policy"]
    if fingerprint(policy_file) != identity["sha256"]:
        raise ValueError("Expected the policy numerically matched to the ground logs")
    policy = torch.jit.load(str(policy_file), map_location=args.device).eval()
    model = Path(URDF_PATH)
    inputs = read_walk_inputs()
    meta_paths = sorted({p.with_name("meta.json") for _,p,_,_ in inputs})
    metadata = [json.loads(p.read_text()) for p in meta_paths]
    policy_keys = ("policy_to_hardware", "default_pose", "action_scale", "lower",
                   "upper", "commands_scale", "obs_scales", "limits")
    policy_cfg = {key:metadata[0]["config"][key] for key in policy_keys}
    if (any({key:m["config"][key] for key in policy_keys} != policy_cfg for m in metadata)
        or any(m["config"]["timing"]["policy_ms"] != 20 for m in metadata)
        or policy_cfg["policy_to_hardware"] != list(range(12))):
        raise ValueError("Unexpected mismatch in the recorded deployment contract")
    args.output.mkdir(parents=True, exist_ok=False)
    cases = []
    initial = []
    replay_schedules = []
    nsteps = round(8.0 / dt)
    for w, path, rows, times in inputs:
        idx = bisect.bisect_right(times, 0.0) - 1
        row = {k: float(v) for k, v in rows[idx].items()}
        height = ground_estimate(model, row)["minimum_nonpenetrating_base_height_m"] + .001
        cases.append({"mode": "replay", "direction": w["direction"], "window": w,
                      "source": str(path.relative_to(ROOT)), "sha256": fingerprint(path),
                      "initial_base_height_m": height, "actual_initial_log_time_s": row["t_mono_ms"] / 1000})
        initial.append((height, [row["quat_"+k] for k in "wxyz"],
                        [row[k] for k in axes("q")], [row[k] for k in axes("dq")],
                        [row["gyro_"+k] for k in "xyz"]))
        schedule = np.searchsorted(times, np.minimum(5.0, np.arange(nsteps)*dt), side="right")-1
        targets = np.asarray([[float(r[k]) for k in axes("q_des")] for r in rows], dtype=np.float32)
        replay_schedules.append(targets[schedule])
    nominal = policy_cfg["default_pose"]
    nominal_row = dict(zip(axes("q"), nominal), quat_w=1., quat_x=0., quat_y=0., quat_z=0.)
    h0 = ground_estimate(model, nominal_row)["minimum_nonpenetrating_base_height_m"] + .001
    for direction in ("forward", "reverse"):
        cases.append({"mode": "closedloop", "direction": direction, "initial_base_height_m": h0,
                      "vx_cmd_m_s": .5 if direction == "forward" else -.5})
        initial.append((h0, [1., 0., 0., 0.], nominal, [0.] * 12, [0.] * 3))

    # Each robot has its own strip on a single flat plane. Variant pairs share geometry and initial state.
    material = sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0,
                                              restitution=0.0, friction_combine_mode="multiply",
                                              restitution_combine_mode="multiply")
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device=args.device, physics_material=material))
    ground = sim_utils.GroundPlaneCfg(physics_material=material)
    ground.func("/World/ground", ground)
    variants = [("baseline", IMGO2_CFG.copy()), ("candidate", make_imgo2_resistance_cfg())]
    if type(variants[0][1].actuators["legs"]) is not type(IMGO2_CFG.actuators["legs"]):
        raise RuntimeError("Candidate configuration replaced the default actuator")
    robots = []
    origins = []
    for g, (name, cfg) in enumerate(variants):
        group_origins = []
        for j in range(len(cases)):
            cell = g*len(cases)+j
            offset = (12.*(cell//4), 8.*(cell%4), 0.)
            group_origins.append(offset)
            sim_utils.create_prim(f"/World/{name}/env_{j}", "Xform", translation=offset)
        origins.append(group_origins)
        cfg.prim_path = f"/World/{name}/env_.*/Robot"
        cfg.spawn.usd_dir = str(args.output.resolve() / "usd_cache")
        cfg.spawn.usd_file_name = "walking_robot.usd"
        cfg.actuators["legs"].stiffness = 25.
        cfg.actuators["legs"].damping = .8
        cfg.actuators["legs"].friction = 0.
        cfg.actuators["legs"].dynamic_friction = 0.
        cfg.actuators["legs"].viscous_friction = 0.
        robots.append(Articulation(cfg))
    sim.reset()
    if type(robots[0].actuators["legs"]) is not DCMotor:
        raise RuntimeError("Walking baseline must use the library DCMotor")
    dev = sim.device
    replay = torch.as_tensor(np.stack(replay_schedules, axis=1), device=dev)
    default = torch.tensor(nominal, device=dev)
    scale = torch.tensor(policy_cfg["action_scale"], device=dev)
    action_min = torch.tensor(policy_cfg["lower"], device=dev)
    action_max = torch.tensor(policy_cfg["upper"], device=dev)
    q_min = torch.tensor(policy_cfg["limits"]["q_min"], device=dev)
    q_max = torch.tensor(policy_cfg["limits"]["q_max"], device=dev)
    obs_scale = policy_cfg["obs_scales"]
    cmd = torch.tensor([[.5,0,0],[-.5,0,0]], device=dev)
    scaled_cmd = cmd * torch.tensor(policy_cfg["commands_scale"], device=dev)
    order = []
    buffers = []
    last_actions = []
    targets = []
    zeros = []
    for group, robot in enumerate(robots):
        if robot.is_fixed_base or robot.num_instances != len(cases):
            raise RuntimeError("Expected six independently initialized floating-base robots per variant")
        idx = [robot.joint_names.index(j) for j in JOINTS]
        if len(idx) != len(robot.joint_names):
            raise ValueError("Unexpected extra articulation joints")
        order.append((idx, [idx.index(i) for i in range(12)]))
        quat = torch.tensor([x[1] for x in initial], device=dev)
        quat = quat / quat.norm(dim=1, keepdim=True)
        pos = torch.tensor(origins[group], device=dev)
        pos[:, 2] = torch.tensor([x[0] for x in initial], device=dev)
        velocity = torch.zeros((len(cases), 6), device=dev)
        velocity[:, 3:] = quat_apply(quat, torch.tensor([x[4] for x in initial], device=dev))
        robot.write_root_pose_to_sim(torch.cat((pos, quat), dim=1))
        robot.write_root_velocity_to_sim(velocity)
        robot.write_joint_state_to_sim(
            torch.tensor([x[2] for x in initial], device=dev)[:, order[-1][1]],
            torch.tensor([x[3] for x in initial], device=dev)[:, order[-1][1]])
        robot.reset()
        buffers.append([])
        last_actions.append(torch.zeros((2, 12), device=dev))
        targets.append(default.expand(len(cases),12).clone())
        zeros.append(torch.zeros_like(targets[-1]))

    settle = round(2.0/dt)
    with torch.inference_mode():
        for step in range(nsteps):
            for i, robot in enumerate(robots):
                idx, inv = order[i]
                d = robot.data
                target = targets[i]
                target[:4] = replay[step]
                if step >= settle and step % obs_stride == 0:
                    obs = torch.cat((d.root_ang_vel_b[4:] * obs_scale["ang_vel"],
                                     d.projected_gravity_b[4:], scaled_cmd,
                                     (d.joint_pos[4:, idx]-default) * obs_scale["dof_pos"],
                                     d.joint_vel[4:, idx] * obs_scale["dof_vel"],
                                     last_actions[i]), dim=1).clamp(-obs_scale["clip_obs"],obs_scale["clip_obs"])
                    action = policy(obs).clamp(action_min,action_max)
                    last_actions[i] = action
                    target[4:] = (default + scale * action).clamp(q_min,q_max)
                robot.set_joint_position_target(target[:, inv])
                robot.set_joint_velocity_target(zeros[i])
                robot.set_joint_effort_target(zeros[i])
                robot.write_data_to_sim()
                if step % log_stride == 0:
                    actuator = robot.actuators["legs"]
                    motor = getattr(actuator, "motor_effort", actuator.applied_effort)
                    resist = getattr(actuator, "resistance_effort", zeros[i])
                    state = (d.root_pos_w[:, 2:3], d.root_quat_w, d.root_lin_vel_b,
                             d.root_ang_vel_b, d.joint_pos[:, idx], d.joint_vel[:, idx], target,
                             actuator.computed_effort[:, idx], motor[:, idx], resist[:, idx],
                             actuator.applied_effort[:, idx])
                    buffers[i].append(torch.cat(state, dim=1).detach().cpu().numpy())
            sim.step(render=False)
            for robot in robots:
                robot.update(dt)
            if step % max(1, round(1/dt)) == 0:
                print(f"Walk simulation {step*dt:.1f}/8.0s", flush=True)

    columns = ["base_z", *("quat_"+k for k in "wxyz"), *("base_v"+k for k in "xyz"), *("gyro_"+k for k in "xyz")]
    for stem in ("q","dq","q_des","tau_pd","tau_motor","tau_resistance","tau_joint"):
        columns.extend(axes(stem))
    output_cases = []
    for (variant, _), values in zip(variants,buffers):
        array = np.asarray(values)
        if not np.isfinite(array).all():
            raise RuntimeError(f"Nonfinite simulation output in {variant}")
        applied = array[:, :, -12:]
        resistance = array[:, :, -24:-12]
        motor = array[:, :, -36:-24]
        if not np.array_equal(applied, motor-resistance) or np.any(resistance[:, :, 3:]):
            raise RuntimeError(f"Torque telemetry or joint mapping mismatch in {variant}")
        for j, case in enumerate(cases):
            filename = f"{variant}_{j:02}.csv"
            series = array[:,j]
            if case["mode"] == "replay":
                series = series[:1000]
            with (args.output / filename).open("w",newline="") as out:
                writer = csv.writer(out)
                writer.writerow(["test_s",*columns])
                writer.writerows((k*.005, *map(float,row)) for k,row in enumerate(series))
            _, quat, q, dq, gyro = initial[j]
            output_cases.append(dict(case, variant=variant, case_id=j, filename=filename,
                                     initial_quat_wxyz=quat, initial_q_rad=q, initial_dq_rad_s=dq,
                                     initial_body_gyro_rad_s=gyro, initial_world_base_velocity_m_s=[0, 0, 0]))
    meta = dict(completed=True, physics_dt_s=dt, policy_period_s=.02, log_period_s=.005,
                policy_sha256=identity["sha256"], robot_urdf_sha256=fingerprint(model), pd=[25,.8],
                policy_inference_count_per_case=(nsteps-settle)//obs_stride,
                code_sha256={str(Path(p).relative_to(ROOT)): fingerprint(Path(p))
                             for p in (__file__,imgo2.__file__,imgo2_actuators.__file__)},
                actuator_parameters=[v.actuators["legs"].to_dict() for _,v in variants],
                policy_contract_sha256={str(p.relative_to(ROOT)):fingerprint(p) for p in meta_paths},
                ground_material=[1.,1.,0.], cases=output_cases,
                source_windows_sha256=fingerprint(DOCS/"new_motion_2026-11-21_walk.csv"),
                replay_timing="host robot.csv timestamp with zero-order hold, no fitted phase shift",
                initialization="recorded q/dq and quaternion/body gyro in replay; zero unmeasured base linear velocity; FK minimum height plus 1mm; nominal standing in closed loop",
                closed_loop="2s nominal position hold, then policy at +/-0.5m/s; evaluate seconds 3 to 8",
                tau_alignment="torque evaluated on same pre-physics-step q/dq in each CSV row")
    (args.output/"metadata.json").write_text(json.dumps(meta,indent=2)+"\n")


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physics-dt", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app
    try:
        run(args)
    finally:
        app.close(wait_for_replicator=False, skip_cleanup=True)


if __name__ == "__main__":
    main()
