#!/usr/bin/env python3
"""Evaluate an existing JIT policy in the current Isaac task, without a runner/export.

Original training configuration is not reconstructed from JIT weights. Results
must be labelled as current-source tests when the source checkpoint is absent.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT/'source/imgo2_rl'), str(PROJECT/'scripts/rl_lab')]
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--task', required=True)
p.add_argument('--policy', type=Path, required=True)
p.add_argument('--history', type=int, default=1, help='Total frames, newest first')
p.add_argument('--steps', type=int, default=1000)
p.add_argument('--warmup', type=int, default=200)
p.add_argument('--json', type=Path, required=True)
p.add_argument('--terrain', choices=('plane', 'rough'), default='plane')
p.add_argument('--terrain-difficulty', type=float, default=.5)
p.add_argument('--fixed-reset', action='store_true', help='Use default root pose/velocity and nominal actuator gains')
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
if not 0 <= a.warmup < a.steps or a.history < 1:
    p.error('Invalid warmup/steps/history')
if not 0. <= a.terrain_difficulty <= 1.:
    p.error('terrain-difficulty must be in [0, 1]')
app = AppLauncher(a).app
import gymnasium as gym
import torch
import imgo2_rl.tasks
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
import play_utils

env = None
try:
    cfg = load_cfg_from_registry(a.task, 'env_cfg_entry_point')
    cfg.seed = 42
    cases = [('stand',[0.,0.,0.]), ('forward03',[.3,0.,0.]),
             ('backward03',[-.3,0.,0.]), ('turn_left03',[0.,0.,.3])]
    terrain_names = []
    if a.terrain == 'rough':
        generator = cfg.scene.terrain.terrain_generator
        if generator is None:
            raise ValueError('rough evaluation requires a terrain generator')
        terrain_names = list(generator.sub_terrains)
        cases = [(name+'_'+direction, command) for name in terrain_names
                 for direction,command in [('forward03',[.3,0.,0.]),('backward03',[-.3,0.,0.])]]
        generator.seed = 42
        generator.curriculum = True  # One deterministic terrain type per column, no adaptive promotions.
        generator.num_rows = 1
        generator.num_cols = len(terrain_names)
        generator.size = (12., 12.)
        generator.difficulty_range = (a.terrain_difficulty, a.terrain_difficulty)
        for terrain in generator.sub_terrains.values():
            terrain.proportion = 1. / len(terrain_names)
        cfg.scene.terrain.max_init_terrain_level = 0
    cfg.scene.num_envs = len(cases)
    cfg.sim.device = a.device
    if a.terrain == 'plane':
        cfg.scene.terrain.terrain_type = 'plane'
        cfg.scene.terrain.terrain_generator = None
    else:
        cfg.scene.terrain.terrain_type = 'generator'
    if a.terrain == 'rough' or a.fixed_reset:
        cfg.events.randomize_reset_base.params['pose_range'] = {
            'x': (0.,0.), 'y': (0.,0.), 'z': (0.,0.),
            'roll': (0.,0.), 'pitch': (0.,0.), 'yaw': (0.,0.)}
        cfg.events.randomize_reset_base.params['velocity_range'] = {}
    if a.fixed_reset:
        cfg.events.randomize_actuator_gains = None
    for key in vars(cfg.curriculum):
        if not key.startswith('_'):
            setattr(cfg.curriculum, key, None)
    for key, term in vars(cfg.events).items():
        if term is not None and getattr(term, 'mode', '') in ('startup','interval'):
            setattr(cfg.events, key, None)
    cfg.observations.policy.enable_corruption = False
    play_utils.disable_command_debug_visualizers(cfg)
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.rel_standing_envs = 0.
    # Do not respawn a new random command during the fixed-command window.
    cfg.commands.base_velocity.resampling_time_range = (10000.,10000.)
    cfg.episode_length_s = max(cfg.episode_length_s, a.steps*cfg.sim.dt*cfg.decimation+1)
    env = gym.make(a.task, cfg=cfg)
    u = env.unwrapped
    if a.terrain == 'rough':
        actual_types = u.scene.terrain.terrain_types.cpu().tolist()
        if actual_types != [i//2 for i in range(len(cases))]:
            raise ValueError(f'Unexpected terrain assignment: {actual_types}')
    env.reset()
    robot = u.scene['robot']
    feet = [leg+'_FOOT' for leg in ('FL','FR','RL','RR')]
    fids, _ = robot.find_bodies(feet, preserve_order=True)
    sensor = u.scene['contact_forces']
    sids, _ = sensor.find_bodies(feet, preserve_order=True)
    blob = a.policy.read_bytes()
    policy = torch.jit.load(io.BytesIO(blob), map_location=a.device).eval()
    cmd = torch.tensor([row[1] for row in cases], device=a.device)
    dt = cfg.sim.dt*cfg.decimation
    history = torch.zeros(len(cases), 45*a.history, device=a.device)
    records = [[] for _ in cases]
    resets = [0]*len(cases)
    with torch.inference_mode():
        for step in range(a.steps):
            u.command_manager.get_command('base_velocity').copy_(cmd)
            obs = u.observation_manager.compute()['policy']
            if obs.shape[1] != 45:
                raise ValueError(f'Expected one 45-dim frame, got {obs.shape}')
            history = torch.cat((obs, history[:, :45*(a.history-1)]), dim=1)
            action = policy(history)
            result = env.step(action)
            terminated, timeout = result[-3], result[-2]
            done = terminated | timeout
            for i, flag in enumerate(done.cpu().tolist()):
                resets[i] += int(flag)
            history[done] = 0.
            if step >= a.warmup:
                root = torch.cat((robot.data.root_lin_vel_b, robot.data.root_ang_vel_b,
                    robot.data.root_pos_w[:,2:3]-u.scene.env_origins[:,2:3],
                    robot.data.projected_gravity_b), dim=1).cpu().tolist()
                contact = (sensor.data.net_forces_w[:,sids,:].norm(dim=-1)>1).cpu().tolist()
                air = sensor.data.last_air_time[:,sids].cpu().tolist()
                foot_z = (robot.data.body_pos_w[:,fids,2]-u.scene.env_origins[:,2:3]).cpu().tolist()
                acts = action.cpu().tolist()
                q = robot.data.joint_pos.cpu().tolist()
                for i in range(len(cases)):
                    records[i].append(dict(t=step*dt,root=root[i],contact=contact[i],last_air=air[i],foot_origin_z=foot_z[i],action=acts[i],q=q[i]))
            if step % 200 == 0:
                print('Evaluation step',step,flush=True)
    output = dict(task=a.task, policy=str(a.policy.resolve()),sha256=hashlib.sha256(blob).hexdigest(),
        dt=dt,history_frames=a.history,body_names=list(robot.body_names),joint_names=list(robot.joint_names),
        default_joint_pos=robot.data.default_joint_pos[0].cpu().tolist(),
        stiffness=robot.data.joint_stiffness[0].cpu().tolist(),damping=robot.data.joint_damping[0].cpu().tolist(),
        fixed_reset=a.fixed_reset,
        actuator_gains={name:dict(stiffness=actuator.stiffness.cpu().tolist(), damping=actuator.damping.cpu().tolist())
                        for name,actuator in robot.actuators.items()},
        terrain=a.terrain,terrain_names=terrain_names,terrain_difficulty=a.terrain_difficulty if a.terrain=='rough' else None,
        limits='Current source config, noise/startup/interval randomization disabled. Not original training reproduction; foot z/base z are relative to patch origin, not local ground clearance.',cases=[])
    for i,(name,command) in enumerate(cases):
        rows=records[i]
        mean=lambda k: sum(r['root'][k] for r in rows)/len(rows)
        output['cases'].append(dict(case=name,command=command,resets=resets[i],mean_vx=mean(0),mean_vy=mean(1),mean_wz=mean(5),mean_z=mean(6),gyro_xy_rms=(sum(r['root'][3]**2+r['root'][4]**2 for r in rows)/len(rows))**.5,
            upright_fraction=sum(-r['root'][9]>.7 for r in rows)/len(rows),rows=rows))
    a.json.write_text(json.dumps(output))
    print('Saved',a.json,flush=True)
finally:
    if env is not None:
        env.close()
    app.close()
