#!/usr/bin/env python3
"""Short deterministic Isaac direction audit of an exported actor (no training/export)."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
REPO = PROJECT.parent
sys.path.insert(0, str(PROJECT / "source/imgo2_rl"))
sys.path.insert(0, str(PROJECT / "scripts/rl_lab"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Imgo2-basemove-flat-amp-height-play")
parser.add_argument("--policy_path", type=Path, default=REPO / "imgo2_deploy/policy/imgo2/amp/policy.pt")
parser.add_argument("--steps", type=int, default=900)
parser.add_argument("--warmup", type=int, default=200)
parser.add_argument("--json", type=Path)
parser.add_argument("--feet", action="store_true", help="Record actual foot contact events and geometric sole clearance.")
parser.add_argument("--linear_speed", type=float, default=0.5)
parser.add_argument("--foot_raw_json", type=Path)
parser.add_argument("--cases", help="Comma-separated subset, e.g. stand; uses one environment per selected command.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not 0 <= args.warmup < args.steps:
    parser.error("Require 0 <= warmup < steps")
launcher = AppLauncher(args)
app = launcher.app

import gymnasium as gym
import torch
import imgo2_rl.tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from rl_lab.wrapper import AmpVecEnvWrapper

names = ["stand", "forward", "backward", "turn_fast", "backward_turn", "backward_fast"]
if args.feet:
    names = ["stand", "forward", "backward", "turn_left", "turn_right", "turn_fast", "backward_turn", "backward_fast"]
if args.cases:
    selected = args.cases.split(',')
    if len(set(selected)) != len(selected) or set(selected)-set(names):
        parser.error("Unknown or duplicate direction case")
    names = selected
cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
cfg.scene.num_envs = len(names)
cfg.sim.device = args.device
cfg.seed = 42
cfg.scene.robot.spawn.force_usd_conversion = True
disabled = []
for name, value in vars(cfg.events).items():
    if value is not None and getattr(value, "mode", None) in ("startup", "interval"):
        setattr(cfg.events, name, None)
        disabled.append(name)

wrapped = None
try:
    env = gym.make(args.task, cfg=cfg)
    wrapped = AmpVecEnvWrapper(env)
    robot = env.unwrapped.scene["robot"]
    # Pin bytes and record identity: another terminal may replace deploy/policy.pt.
    policy_bytes = args.policy_path.read_bytes()
    policy_sha256 = hashlib.sha256(policy_bytes).hexdigest()
    policy = torch.jit.load(io.BytesIO(policy_bytes), map_location=args.device).eval()
    vx = args.linear_speed
    command_by_name = {"stand":[0,0,0], "forward":[vx,0,0], "backward":[-vx,0,0],
                       "turn_left":[0,0,.5], "turn_right":[0,0,-.5], "turn_fast":[0,0,1],
                       "backward_turn":[-vx,0,.5], "backward_fast":[-1,0,0]}
    command_rows = [command_by_name[name] for name in names]
    commands = torch.tensor(command_rows, device=args.device)
    metrics = [[] for _ in names]
    resets = [0] * len(names)
    if args.feet:
        import xml.etree.ElementTree as ET
        import summarize_foot_clearance as sole
        foot_names = [leg + "_FOOT" for leg in sole.LEGS]
        body_ids, found_names = robot.find_bodies(foot_names, preserve_order=True)
        sensor = env.unwrapped.scene["contact_forces"]
        sensor_ids, sensor_names = sensor.find_bodies(foot_names, preserve_order=True)
        if list(found_names) != foot_names or list(sensor_names) != foot_names:
            raise ValueError("Foot body/sensor order does not match FL/FR/RL/RR")
        model = ET.parse(cfg.scene.robot.spawn.asset_path).getroot()
        shapes = [sole.foot_shapes(model, name) for name in foot_names]
        ordered_joint_names = [leg+'_'+joint+'_joint' for leg in sole.LEGS for joint in ('hip','thigh','shank')]
        joint_ids, found_joints = robot.find_joints(ordered_joint_names, preserve_order=True)
        if list(found_joints) != ordered_joint_names:
            raise ValueError("Joint audit order does not match FL/FR/RL/RR")
        records = [{"times": [], "clearance": [], "contact": [],
                    "joint_names": ordered_joint_names, "joint_positions": [], "raw_actions": [], "joint_targets": [],
                    "landings": [[] for _ in foot_names], "air_times": [[] for _ in foot_names],
                    "raw_landing_count": [0]*4} for _ in names]
    with torch.inference_mode():
        for step in range(args.steps):
            env.unwrapped.command_manager.get_command("base_velocity").copy_(commands)
            obs = env.unwrapped.observation_manager.compute()["policy"]
            actions = policy(obs)
            result = wrapped.step(actions)
            for i, done in enumerate(result[4].cpu().tolist()):
                resets[i] += bool(done)
            if step >= args.warmup:
                values = torch.stack([robot.data.root_lin_vel_b[:, 0], robot.data.root_lin_vel_b[:, 1],
                                      robot.data.root_ang_vel_b[:, 2],
                                      robot.data.root_pos_w[:, 2] - env.unwrapped.scene.env_origins[:, 2]], dim=1).cpu().tolist()
                for i, row in enumerate(values):
                    metrics[i].append(row)
                if args.feet:
                    positions = robot.data.body_pos_w[:,body_ids,:].cpu().tolist()
                    quats = robot.data.body_quat_w[:,body_ids,:].cpu().tolist()
                    ground = env.unwrapped.scene.env_origins[:,2].cpu().tolist()
                    contacts = (sensor.data.net_forces_w[:,sensor_ids,:].norm(dim=-1)>1.0).cpu().tolist()
                    first = sensor.compute_first_contact(wrapped.dt)[:,sensor_ids].cpu().tolist()
                    last_air = sensor.data.last_air_time[:,sensor_ids].cpu().tolist()
                    ordered_positions = robot.data.joint_pos[:,joint_ids].cpu().tolist()
                    raw_actions = actions.cpu().tolist()
                    joint_targets = env.unwrapped.action_manager.get_term("joint_pos").processed_actions.cpu().tolist()
                    for i in range(len(names)):
                        heights=[]
                        for foot in range(4):
                            quat=quats[i][foot]
                            rot=sole.fk.quaternion_rotation(quat[1:]+quat[:1])
                            heights.append(sole.collision_bottom(positions[i][foot],rot,shapes[foot])-ground[i])
                            if first[i][foot]:
                                records[i]["raw_landing_count"][foot]+=1
                                if last_air[i][foot]>=0.02:
                                    records[i]["landings"][foot].append(step*wrapped.dt)
                                    records[i]["air_times"][foot].append(last_air[i][foot])
                        records[i]["times"].append(step*wrapped.dt)
                        records[i]["clearance"].append(heights)
                        records[i]["contact"].append(contacts[i])
                        records[i]["joint_positions"].append(ordered_positions[i])
                        records[i]["raw_actions"].append(raw_actions[i])
                        records[i]["joint_targets"].append(joint_targets[i])
            if step in (0, args.warmup, args.steps - 1):
                print("Native Isaac evaluation step", step, flush=True)
    output = {"task": args.task, "policy": str(args.policy_path.resolve()), "policy_sha256": policy_sha256, "control_dt": wrapped.dt,
              "steps": args.steps, "warmup_steps": args.warmup,
              "domain_randomization_disabled": disabled, "cases": [],
              "limits": "One environment per command; current local URDF and nominal startup physics. Not an exact engine A/B. No training/export."}
    for i, name in enumerate(names):
        seq = metrics[i]
        output["cases"].append({"case": name, "command": commands[i].cpu().tolist(),
                                "mean_vx": sum(r[0] for r in seq) / len(seq),
                                "mean_vy": sum(r[1] for r in seq) / len(seq),
                                "mean_wz": sum(r[2] for r in seq) / len(seq),
                                "mean_z": sum(r[3] for r in seq) / len(seq),
                                "min_z": min(r[3] for r in seq), "episode_resets": resets[i]})
        if args.feet:
            detail={}
            for foot,foot_name in enumerate(foot_names):
                record=records[i]
                air=record["air_times"][foot]
                events=record["landings"][foot]
                intervals=[b-a for a,b in zip(events,events[1:])]
                interval=sole.quantile(intervals,.5) if intervals else None
                detail[foot_name]={**sole.height_summary(record["times"],[h[foot] for h in record["clearance"]]),
                                  "contact_fraction":sum(c[foot] for c in record["contact"])/len(record["contact"]),
                                  "raw_landing_count":record["raw_landing_count"][foot],
                                  "qualified_landings_air_at_least_20ms":len(events),
                                  "completed_air_time_mean_s":sum(air)/len(air) if air else None,
                                  "landing_median_period_s":interval,
                                  "landing_frequency_hz":1/interval if interval else None}
            output["cases"][-1]["per_foot"]=detail
    if args.feet:
        output["foot_limits"]="Geometry uses FOOT collision primitives relative to a flat plane; contacts are actual force>1N. Landing periods exclude air-time<20ms chatter; one environment per command."
        if args.foot_raw_json:
            args.foot_raw_json.write_text(json.dumps({name:r for name,r in zip(names,records)})+'\n')
    if args.json:
        args.json.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2), flush=True)
finally:
    if wrapped is not None:
        wrapped.close()
    app.close()
