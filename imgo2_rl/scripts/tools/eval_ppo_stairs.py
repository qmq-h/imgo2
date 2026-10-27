#!/usr/bin/env python3
"""Headless deployed JIT policy evaluation on flat ground or the actual Gazebo stair boxes.

Loads an existing policy; never trains or exports. Optional alignment overrides affect
only this diagnostic environment, not the training/deployment configuration files.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

PROJECT = Path(__file__).resolve().parents[2]
REPO = PROJECT.parent
sys.path.insert(0, str(PROJECT / "source/imgo2_rl"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--terrain", choices=["plane", "stairs"], required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--world", type=Path, default=REPO / "imgo2_deploy/ros2_ws/src/imgo2_deploy/worlds/stairs_8cm.world")
parser.add_argument("--policy", type=Path, default=REPO / "imgo2_deploy/ros2_ws/policy/imgo2/ppo/policy.pt")
parser.add_argument("--speed", type=float, default=.4)
parser.add_argument("--profile", choices=["constant", "maneuvers"], default="constant")
parser.add_argument("--maneuver-low-speed", type=float, default=.1)
parser.add_argument("--seconds", type=float, default=30)
parser.add_argument("--stand-seconds", type=float, default=3)
parser.add_argument("--native-materials", action="store_true", help="Keep configured terrain material rather than diagnostic .6 friction/zero restitution.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.profile == "maneuvers":
    sys.path.insert(0, str(REPO / "imgo2_deploy/ros2_ws/scripts"))
    from ppo_motion_profile import command_at, TOTAL_SECONDS
    if args.terrain != "plane":
        parser.error("maneuvers requires --terrain plane")
    args.seconds = TOTAL_SECONDS
args.output.mkdir(parents=True, exist_ok=True)
launcher = AppLauncher(args)
app = launcher.app

import gymnasium as gym
import torch
import isaaclab.sim as sim
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
import imgo2_rl.tasks  # noqa: F401


def boxes_from_world():
    root = ET.parse(args.world).getroot()
    model = root.find("world/model[@name='stairs_8cm']")
    if model is None:
        raise ValueError("Expected stairs_8cm model in the Gazebo world")
    boxes = []
    for link in model.findall("link"):
        pose = list(map(float, link.findtext("pose").split()))
        if any(abs(v)>1e-9 for v in pose[3:]):
            raise ValueError("This evaluator supports only unrotated stair boxes")
        size = tuple(map(float, link.findtext("collision/geometry/box/size").split()))
        boxes.append({"name":link.attrib["name"], "pos":pose[:3], "size":size})
    return boxes


env = None
try:
    cfg = load_cfg_from_registry("Imgo2-basemove-flat-ppo", "env_cfg_entry_point")
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    cfg.seed = 42
    cfg.episode_length_s = max(45, args.seconds+5)
    cfg.scene.robot.spawn.force_usd_conversion = True
    cfg.observations.policy.enable_corruption = False
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.rel_standing_envs = 0
    cfg.commands.base_velocity.resampling_time_range = (1000.,1000.)
    disabled = []
    for name, value in vars(cfg.events).items():
        if value is not None and getattr(value,"mode",None) in ("startup","reset","interval"):
            setattr(cfg.events,name,None)
            disabled.append(name)
    # This evaluation consumes no rewards/critic, and tests explicit failures itself.
    cfg.terminations.terrain_out_of_bounds = None
    if not args.native_materials:
        cfg.scene.terrain.physics_material.static_friction = .6
        cfg.scene.terrain.physics_material.dynamic_friction = .6
        cfg.scene.terrain.physics_material.restitution = 0
        cfg.sim.physics_material = cfg.scene.terrain.physics_material
    boxes = boxes_from_world()
    if args.terrain == "stairs":
        for box in boxes:
            setattr(cfg.scene, "audit_"+box["name"], AssetBaseCfg(
                prim_path="/World/audit_stairs/"+box["name"],
                spawn=sim.CuboidCfg(size=box["size"], collision_props=sim.CollisionPropertiesCfg(),
                    physics_material=sim.RigidBodyMaterialCfg(static_friction=1.,dynamic_friction=1.,restitution=0.)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(box["pos"]))))
    env = gym.make("Imgo2-basemove-flat-ppo", cfg=cfg)
    env.reset()
    core = env.unwrapped
    robot = core.scene["robot"]
    if not args.native_materials:
        materials = robot.root_physx_view.get_material_properties()
        materials[...,0:2]=.6
        materials[...,2]=0
        robot.root_physx_view.set_material_properties(materials,torch.tensor([0],dtype=torch.int64))
    ordered = [f"{l}_{j}_joint" for l in ("FL","FR","RL","RR") for j in ("hip","thigh","shank")]
    joint_ids, found = robot.find_joints(ordered,preserve_order=True)
    assert list(found)==ordered
    foot_ids, _ = robot.find_bodies([l+"_FOOT" for l in ("FL","FR","RL","RR")],preserve_order=True)
    policy_bytes=args.policy.read_bytes()
    policy=torch.jit.load(io.BytesIO(policy_bytes),map_location=args.device).eval()
    assert tuple(policy(torch.zeros((1,45),device=args.device)).shape)==(1,12)
    rows=[];done_count=0;failure=None;max_x=-math.inf;max_z=-math.inf
    with torch.inference_mode():
        for step in range(round(args.seconds/core.step_dt)):
            t=step*core.step_dt
            vx=args.speed if t>=args.stand_seconds else 0.
            wz=0.; stage="constant"
            if args.profile == "maneuvers":
                vx,wz,stage=command_at(t,args.maneuver_low_speed)
            cmd=torch.tensor([[vx,0,wz]],device=args.device)
            core.command_manager.get_command("base_velocity")[:]=cmd
            obs=core.observation_manager.compute()["policy"]
            if tuple(obs.shape)!=(1,45):raise ValueError(f"Unexpected policy observation {obs.shape}")
            action=policy(obs)
            # Explicit deployment-order reference: catches unintended scale/order changes.
            reference=torch.cat([robot.data.root_ang_vel_b*.25,robot.data.projected_gravity_b,
                cmd,
                robot.data.joint_pos[:,joint_ids]-robot.data.default_joint_pos[:,joint_ids],
                (robot.data.joint_vel[:,joint_ids]-robot.data.default_joint_vel[:,joint_ids])*.05,
                core.action_manager.prev_action],dim=1)
            # Native last_action reads the current previously applied action.
            reference[:,-12:]=core.action_manager.action
            diff=(obs-reference).abs().max().item()
            if diff>1e-5:raise ValueError(f"Observation contract differs from deployment: {diff}")
            _,_,terminated,truncated,_=env.step(action)
            done_count+=int((terminated|truncated).item())
            p=robot.data.root_pos_w[0].cpu().tolist();v=robot.data.root_lin_vel_b[0].cpu().tolist()
            e=[x.item() for x in euler_xyz_from_quat(robot.data.root_quat_w)]
            e=[math.atan2(math.sin(x),math.cos(x)) for x in e]
            f=robot.data.body_pos_w[0,foot_ids,:].cpu().tolist()
            q=robot.data.joint_pos[0,joint_ids].cpu().tolist()
            max_x=max(max_x,p[0]);max_z=max(max_z,p[2])
            terrain_z=0.
            if args.terrain=="stairs":
                for box in boxes:
                    if abs(p[0]-box['pos'][0])<=box['size'][0]/2 and abs(p[1]-box['pos'][1])<=box['size'][1]/2:
                        terrain_z=max(terrain_z,box['pos'][2]+box['size'][2]/2)
            gyro=robot.data.root_ang_vel_b[0].cpu().tolist()
            rows.append([t,vx,*p,*v,*e,terrain_z,*(x for foot in f for x in foot),*q,diff,wz,stage,*gyro])
            if step%250==0:print(f"{args.terrain}: t={t:.1f} x={p[0]:.3f} y={p[1]:.3f} z={p[2]:.3f}",flush=True)
            if done_count:failure="environment reset";break
            if t>1 and max(abs(e[0]),abs(e[1]))>math.radians(60):failure="tilt_over_60deg";break
            if t>1 and p[2]<.12:failure="base_below_12cm";break
            if not torch.isfinite(obs).all() or not torch.isfinite(action).all():failure="nonfinite";break
    header=["t_s","command_vx","x","y","z","vx_body","vy_body","vz_body","roll_rad","pitch_rad","yaw_rad","ground_z_at_base"]
    header += [f"{leg}_foot_{axis}" for leg in ("FL","FR","RL","RR") for axis in "xyz"]
    header += ordered+["observation_max_diff","command_wz","stage","gyro_x_rads","gyro_y_rads","gyro_z_rads"]
    with (args.output/"trace.csv").open("w") as f:
        writer=csv.writer(f);writer.writerow(header);writer.writerows(rows)
    final=rows[-1];moving=[r for r in rows if r[0]>=args.stand_seconds]
    success=(args.terrain=="stairs" and final[2]>=4.1 and abs(final[3])<.8 and final[4]>.84 and failure is None)
    result={"terrain":args.terrain,"task_for_robot_and_observations":"Imgo2-basemove-flat-ppo",
        "policy":str(args.policy),"policy_sha256":hashlib.sha256(policy_bytes).hexdigest(),
        "robot_urdf":cfg.scene.robot.spawn.asset_path,"robot_sha256":hashlib.sha256(Path(cfg.scene.robot.spawn.asset_path).read_bytes()).hexdigest(),
        "gazebo_world":str(args.world),"world_sha256":hashlib.sha256(args.world.read_bytes()).hexdigest(),
        "stairs_geometry":boxes if args.terrain=="stairs" else [],"control_dt":core.step_dt,"physics_dt":cfg.sim.dt,
        "profile":args.profile,"maneuver_low_speed":args.maneuver_low_speed,
        "speed_command":args.speed if args.profile=="constant" else None,
        "stand_seconds":args.stand_seconds if args.profile=="constant" else 6.,"simulated_seconds":final[0]+core.step_dt,
        "obs_noise":False,"disabled_events":disabled,"native_materials":args.native_materials,
        "actuator_gains":{k:{"kp":a.stiffness[0].cpu().tolist(),"kd":a.damping[0].cpu().tolist()} for k,a in robot.actuators.items()},
        "final_position":final[2:5],"max_x":max_x,"max_base_z":max_z,
        "max_pitch_abs_deg":math.degrees(max(abs(r[header.index("pitch_rad")]) for r in rows)),
        "mean_moving_vx":sum(r[5] for r in moving)/len(moving),"resets":done_count,"failure":failure,
        "stairs_completed":success,"observation_contract_max_diff":max(r[header.index("observation_max_diff")] for r in rows),
        "limitations":"Single seed/case, current real URDF, clean observations/disabled randomization; default aligned materials. This is not original training reconstruction or exact PhysX/ODE equivalence."}
    (args.output/"summary.json").write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['terrain','final_position','max_x','max_base_z','failure','stairs_completed']},indent=2),flush=True)
finally:
    if env is not None:env.close()
    app.close()
