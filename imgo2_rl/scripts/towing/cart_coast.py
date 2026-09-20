"""P1 drop and P2 coast experiments for the passive cart, without RL.

Run with an Isaac Lab Python environment. --help needs only standard Python.
Each case runs for a bounded duration. Defaults scan two speeds and four
bearing damping coefficients; no resistance coefficient is pre-calibrated.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import uuid

RL_ROOT = Path(__file__).resolve().parents[2]
REPO = RL_ROOT.parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("drop", "coast"), default="coast")
    parser.add_argument("--velocities", type=float, nargs="+", default=[0.5, 1.0], help="Initial forward velocities, m/s")
    parser.add_argument("--damping", type=float, nargs="+", default=[0.0, 0.008, 0.016, 0.032], help="Per-wheel b, N m s/rad")
    parser.add_argument("--duration", type=float, default=10.0, help="Measured simulation seconds per case")
    parser.add_argument("--settle-time", type=float, default=1.0, help="Coast warmup before applying initial velocity")
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--stop-speed", type=float, default=0.02)
    parser.add_argument("--stop-hold", type=float, default=0.5)
    parser.add_argument("--drop-height", type=float, default=0.03, help="Initial wheel clearance above ground, m")
    parser.add_argument("--output-dir", type=Path, help="New experiment directory; never overwrite an existing run")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    for key in ("duration", "settle_time", "dt", "stop_speed", "stop_hold", "drop_height"):
        value = getattr(args, key)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{key.replace('_', '-')} must be finite and positive")
    if args.dt > 0.01 or args.dt >= args.stop_hold or args.duration < args.stop_hold + args.dt:
        parser.error("Require dt <= 0.01, dt < stop-hold and duration >= stop-hold + dt")
    if not all(math.isfinite(v) and 0 < v <= 2.0 for v in args.velocities):
        parser.error("Initial velocities must be in (0, 2] m/s")
    if not all(math.isfinite(b) and 0 <= b <= 0.1 for b in args.damping):
        parser.error("Per-wheel damping must be in [0, 0.1] N m s/rad for this initial calibration")
    return args


def git_info():
    def run(*args):
        result = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=10)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"commit": run("rev-parse", "HEAD"), "working_tree": run("status", "--porcelain")}


def main(args):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output_dir or RL_ROOT / "logs/towing/cart_coast" / f"{stamp}_{uuid.uuid4().hex[:8]}").expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)

    def json_file(path, data):
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n", encoding="utf-8")

    manifest = {"state": "starting", "arguments": vars(args), "git": git_info(), "python": platform.python_version()}
    json_file(output / "experiment.json", manifest)
    print(f"[INFO] Output: {output}", flush=True)
    application = None
    try:
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(headless=args.headless, device=args.device)
        application = launcher.app

        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene
        from imgo2_rl.assets.cart import make_cart_cfg, resolve_cart_path
        from imgo2_rl.tasks.manager_based.towing.cart_scene_cfg import CartSceneCfg
        from imgo2_rl.tasks.manager_based.towing.mdp.resistance import viscous_resistance
        from imgo2_rl.tasks.manager_based.towing.utils.recording import CartRecorder, LEGS, write_json
        sys.path.insert(0, str(RL_ROOT / "scripts/tools"))
        from summarize_cart_coast import summarize_run
        from cart_coast_metrics import summarize_sweep

        cart_cfg, model = make_cart_cfg(output / "usd", drop_height=args.drop_height)
        path = resolve_cart_path()
        sim_cfg = sim_utils.SimulationCfg(
            dt=args.dt, device=args.device,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8, dynamic_friction=0.8, restitution=0.0,
                friction_combine_mode="average", restitution_combine_mode="min"))
        sim = sim_utils.SimulationContext(sim_cfg)
        sim.set_camera_view((2.0, 2.0, 1.5), (0.0, 0.0, 0.15))
        scene_cfg = CartSceneCfg(num_envs=1, env_spacing=3.0, cart=cart_cfg)
        scene = InteractiveScene(scene_cfg)
        sim.reset()
        cart, contacts = scene["cart"], scene["wheel_contacts"]
        joint_ids, joint_names = cart.find_joints(list(model["joint_names"]), preserve_order=True)
        contact_ids, contact_names = contacts.find_bodies(list(model["wheel_names"]), preserve_order=True)
        if len(joint_ids) != 4 or cart.num_joints != 4 or len(contact_ids) != 4:
            raise RuntimeError("Imported cart must have four joints and four named wheel contacts")
        zero_effort = torch.zeros_like(cart.data.joint_pos)
        radius = model["wheel_radius_m"]
        if args.mode == "coast" and max(args.velocities) / radius > min(model["velocity_limits_radps"].values()):
            raise ValueError("Requested rolling wheel speed exceeds the URDF velocity limit")
        dt = sim.get_physics_dt()
        if not math.isclose(dt, args.dt, rel_tol=1e-6):
            raise RuntimeError("Simulator dt differs from requested dt")
        versions = {"python": platform.python_version(), "torch": torch.__version__}
        for package in ("isaaclab", "isaacsim"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = "unknown"
        runtime = {
            "body_names": cart.body_names, "joint_names": joint_names,
            "contact_names": contact_names, "joint_indices": joint_ids,
            "masses_kg": cart.root_physx_view.get_masses().cpu().tolist(),
            "inertias_kgm2": cart.root_physx_view.get_inertias().cpu().tolist(),
            "joint_stiffness": cart.root_physx_view.get_dof_stiffnesses().cpu().tolist(),
            "joint_damping": cart.root_physx_view.get_dof_dampings().cpu().tolist(),
        }
        runtime_mass = sum(runtime["masses_kg"][0])
        if not math.isclose(runtime_mass, model["total_mass_kg"], rel_tol=1e-4):
            raise RuntimeError(f"Imported mass {runtime_mass} differs from URDF {model['total_mass_kg']}")
        if any(abs(v) > 1e-8 for key in ("joint_stiffness", "joint_damping") for row in runtime[key] for v in row):
            raise RuntimeError("Imported wheels have nonzero servo gains/damping")
        common = {"model": model, "model_path": str(path), "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "runtime": runtime, "versions": versions, "git": manifest["git"], "dt_s": dt,
                  "stop_speed_mps": args.stop_speed, "stop_hold_s": args.stop_hold,
                  "settle_time_s": args.settle_time, "drop_height_m": args.drop_height,
                  "requested_duration_s": args.duration, "device": args.device,
                  "material": {"static_friction": 0.8, "dynamic_friction": 0.8, "restitution": 0.0},
                  "solver_iterations": {"position": 8, "velocity": 4},
                  "torque_sample_convention": "applied during interval ending at sample; zero at t=0"}
        write_json(output / "runtime.json", common)

        def sample(t, effort):
            state = cart.data.root_state_w[0].detach().cpu().tolist()
            omega = cart.data.joint_vel[0, joint_ids].detach().cpu().tolist()
            forces = contacts.data.net_forces_w[0, contact_ids, 2].detach().cpu().tolist()
            tau = effort[0, joint_ids].detach().cpu().tolist()
            x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz = state
            roll = math.atan2(2*(qw*qx+qy*qz), 1-2*(qx*qx+qy*qy))
            pitch = math.asin(max(-1, min(1, 2*(qw*qy-qz*qx))))
            yaw = math.atan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))
            row = dict(time_s=t, x_m=x, y_m=y, z_m=z, vx_mps=vx, vy_mps=vy, vz_mps=vz,
                       roll_rad=roll, pitch_rad=pitch, yaw_rad=yaw, wx_radps=wx, wy_radps=wy, wz_radps=wz)
            for leg, w, f, torque in zip(LEGS, omega, forces, tau):
                row.update({f"{leg}_omega_radps": w, f"{leg}_tau_nm": torque, f"{leg}_normal_n": f})
            if not all(math.isfinite(v) for v in row.values()):
                raise RuntimeError("Non-finite cart state")
            return row

        def advance(b):
            if not application.is_running():
                raise RuntimeError("Simulation application closed before case completed")
            effort = zero_effort.clone()
            effort[:, joint_ids] = viscous_resistance(cart.data.joint_vel[:, joint_ids], b)
            if not torch.isfinite(effort).all() or effort.abs().max().item() > min(model["effort_limits_nm"].values()):
                raise RuntimeError("Unsafe/non-finite resistance torque")
            cart.set_joint_effort_target(effort)
            scene.write_data_to_sim()
            sim.step()
            scene.update(dt)
            return effort

        cases = [(0.0, 0.0)] if args.mode == "drop" else [(v, b) for v in args.velocities for b in args.damping]
        results = []
        for case_number, (velocity, damping) in enumerate(cases):
            case_dir = output / f"case_{case_number:02d}_v{velocity:g}_b{damping:g}"
            config = {**common, "mode": args.mode, "initial_velocity_mps": velocity, "damping_nms_per_rad": damping}
            with CartRecorder(case_dir, config) as recorder:
                write_json(case_dir / "status.json", {"state": "running"})
                try:
                    root = cart.data.default_root_state.clone()
                    root[:, :3] += scene.env_origins
                    cart.write_root_pose_to_sim(root[:, :7])
                    cart.write_root_velocity_to_sim(root[:, 7:])
                    cart.write_joint_state_to_sim(cart.data.default_joint_pos.clone(), cart.data.default_joint_vel.clone())
                    cart.set_joint_effort_target(zero_effort)
                    scene.reset()
                    scene.update(dt)
                    if args.mode == "coast":
                        for _ in range(math.ceil(args.settle_time / dt)):
                            advance(0.0)
                        settled = sample(0, zero_effort)
                        if (math.sqrt(sum(settled[k]**2 for k in ("vx_mps", "vy_mps", "vz_mps"))) > 0.05
                            or max(abs(settled[k]) for k in ("wx_radps", "wy_radps", "wz_radps")) > 0.1
                            or abs(settled["z_m"] - model["resting_height_m"]) > 0.025
                            or max(abs(settled[k]) for k in ("roll_rad", "pitch_rad", "yaw_rad")) > 0.1
                            or any(settled[f"{leg}_normal_n"] <= 0.1 for leg in LEGS)):
                            raise RuntimeError("Cart did not settle on all four wheels; increase settle time or inspect drop")
                        root_velocity = torch.zeros_like(cart.data.root_state_w[:, 7:])
                        root_velocity[:, 0] = velocity
                        cart.write_root_velocity_to_sim(root_velocity)
                        wheel_velocity = torch.zeros_like(cart.data.joint_vel)
                        wheel_velocity[:, joint_ids] = velocity / radius
                        cart.write_joint_state_to_sim(cart.data.joint_pos.clone(), wheel_velocity)
                    recorder.append(sample(0, zero_effort))
                    for step in range(1, math.ceil(args.duration / dt) + 1):
                        applied = advance(damping)
                        row = sample(step * dt, applied)
                        recorder.append(row)
                        if abs(row["roll_rad"]) > 0.6 or abs(row["pitch_rad"]) > 0.6 or row["z_m"] < 0:
                            raise RuntimeError("Cart tipped or penetrated ground")
                    write_json(case_dir / "status.json", {"state": "completed"})
                except (Exception, KeyboardInterrupt) as exc:
                    write_json(case_dir / "status.json", {"state": "failed", "error": f"{type(exc).__name__}: {exc}"})
                    raise
            result = summarize_run(case_dir)
            results.append({"directory": case_dir.name, "velocity_mps": velocity, "damping": damping, **result})
            write_json(output / "sweep.json", results)
            if args.mode == "coast":
                write_json(output / "calibration.json", summarize_sweep(results))
            print(f"[CASE] v={velocity:g} b={damping:g}: valid={result['valid']} stopped={result['stopped']} D={result['stop_distance_m']}", flush=True)
        manifest["state"] = "completed"
        manifest["valid"] = all(r["valid"] for r in results)
        json_file(output / "experiment.json", manifest)
        return 0 if manifest["valid"] else 2
    except (Exception, KeyboardInterrupt) as exc:
        manifest.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        json_file(output / "experiment.json", manifest)
        raise
    finally:
        if application is not None:
            application.close()


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
