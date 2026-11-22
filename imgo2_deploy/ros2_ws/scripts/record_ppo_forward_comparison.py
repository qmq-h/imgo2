#!/usr/bin/env python3
"""Bounded headless +0.5 m/s PPO recording; never accesses real hardware.
Raw ROS command/state/IMU streams keep independent host and simulation times.
"""
import argparse
import bisect
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]

def load_profile(path):
    """Zero-order hold of captured high-level commands, relative to first input."""
    with Path(path).open() as f:
        rows=list(csv.DictReader(f))
    times=[float(r['input_captured_ms'])/1000 for r in rows]
    if not times or any(b<=a for a,b in zip(times,times[1:])):
        raise ValueError('Command profile must have increasing input timestamps')
    commands=[tuple(float(r[k]) for k in ('cmd_vx','cmd_vy','cmd_wz')) for r in rows]
    if not all(math.isfinite(v) and abs(v)<=1.0 for c in commands for v in c):
        raise ValueError('Invalid command profile')
    return [t-times[0] for t in times],commands,times[0]

def collect(out, hold_seconds, vx=.5, expected_kd=.5, profile=None, stand_only=False, expected_kp=25.):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Imu, Joy
    from robot_msgs.msg import RobotCommand, RobotState
    from nav_msgs.msg import Odometry
    from rosgraph_msgs.msg import Clock
    handles = []
    def writer(name, fields):
        f = (out/name).open('w'); handles.append(f)
        w = csv.writer(f); w.writerow(fields); return w
    joint_fields = ['host_s','sim_s','t_s'] + [f'{k}_{i:02}' for k in ('q','dq','tau') for i in range(12)]
    tw = writer('clock.csv',['host_s','sim_s'])
    sw = writer('state.csv', joint_fields)
    cw = writer('command.csv',['host_s','sim_s','t_s']+[f'{k}_{i:02}' for k in ('q','dq','tau','kp','kd') for i in range(12)])
    iw = writer('imu.csv',['host_s','sim_s','t_s','quat_w','quat_x','quat_y','quat_z','gyro_x','gyro_y','gyro_z','accel_x','accel_y','accel_z'])
    ow = writer('odom.csv',['host_s','sim_s','t_s','x','y','z','yaw',
                           'quat_w','quat_x','quat_y','quat_z',
                           'vx','vy','vz','wx','wy','wz'])
    vw = writer('velocity.csv',['host_s','sim_s','t_s','vx','vy','wz','stage'])
    rclpy.init(); node = Node('ppo_forward_comparison_recorder')
    clock = [None]; origin = [None]; fallen = [False]; counts = [0,0,0,0]
    ppo_start = [None]; stand_request = [None]
    profile_data=load_profile(profile) if profile else None
    def stamp():
        return [time.monotonic(),clock[0],clock[0]-origin[0]-6] if origin[0] is not None else None
    def on_clock(m):
        exact=m.clock.sec+m.clock.nanosec*1e-9
        clock[0] = max(clock[0] or 0,exact) if profile_data else exact
        if origin[0] is None: origin[0] = clock[0]
        tw.writerow([time.monotonic(),clock[0]])
    def state(m):
        t=stamp()
        if t and len(m.motor_state)==12:
            sw.writerow(t+[getattr(j,k if k!='tau' else 'tau_est') for k in ('q','dq','tau') for j in m.motor_state]); counts[0]+=1
    def command(m):
        t=stamp()
        if t and len(m.motor_command)==12:
            cw.writerow(t+[getattr(j,k) for k in ('q','dq','tau','kp','kd') for j in m.motor_command]); counts[1]+=1
            # Anchor the ten seconds to actual PPO PD publication, not model-load time.
            if t[2]>=0 and ppo_start[0] is None and all(
                    abs(j.kp-expected_kp)<1e-3 and abs(j.kd-expected_kd)<1e-3 for j in m.motor_command):
                ppo_start[0]=t[1]
    def imu(m):
        t=stamp()
        if not t:return
        exact=m.header.stamp.sec+m.header.stamp.nanosec*1e-9
        if profile_data:clock[0]=max(clock[0] or 0,exact)
        t[1]=exact;t[2]=exact-origin[0]-6
        q=m.orientation;g=m.angular_velocity;a=m.linear_acceleration
        iw.writerow(t+[q.w,q.x,q.y,q.z,g.x,g.y,g.z,a.x,a.y,a.z]);counts[2]+=1
        roll=math.atan2(2*(q.w*q.x+q.y*q.z),1-2*(q.x*q.x+q.y*q.y))
        pitch=math.asin(max(-1,min(1,2*(q.w*q.y-q.z*q.x))))
        if t[2]>1 and max(abs(roll),abs(pitch))>math.pi/3:fallen[0]=True
    def odom(m):
        t=stamp()
        if not t:return
        exact=m.header.stamp.sec+m.header.stamp.nanosec*1e-9
        t[1]=exact;t[2]=exact-origin[0]-6
        p=m.pose.pose.position;q=m.pose.pose.orientation
        v=m.twist.twist.linear;g=m.twist.twist.angular
        ow.writerow(t+[p.x,p.y,p.z,math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z)),
                        q.w,q.x,q.y,q.z,v.x,v.y,v.z,g.x,g.y,g.z]);counts[3]+=1
    node.create_subscription(Clock,'/clock',on_clock,qos_profile_sensor_data)
    node.create_subscription(RobotState,'/robot_joint_controller/state',state,qos_profile_sensor_data)
    node.create_subscription(RobotCommand,'/robot_joint_controller/command',command,10)
    node.create_subscription(Imu,'/imu',imu,qos_profile_sensor_data)
    node.create_subscription(Odometry,'/odom',odom,qos_profile_sensor_data)
    pub=node.create_publisher(Joy,'/joy',10)
    deadline=time.monotonic()+300;last=-1;complete=False
    try:
        while rclpy.ok() and time.monotonic()<deadline:
            rclpy.spin_once(node,timeout_sec=.002)
            if fallen[0]:raise RuntimeError('tilt >60 degrees')
            t=stamp()
            if not t:continue
            if t[1]-last>=.02:
                last=t[1];j=Joy();j.buttons=[0]*11;j.axes=[0.0]*8
                if t[2]<0:
                    j.buttons[0]=1;stage='getup'
                elif stand_only:
                    j.buttons[0]=1;stage='stand_hold'
                else:
                    if ppo_start[0] is None or t[1]<ppo_start[0]+hold_seconds:
                        j.buttons[5]=1;j.axes[7]=1.0;j.axes[1]=vx
                        stage=f'ppo_request_{vx:+.2f}' if ppo_start[0] is None else f'ppo_{vx:+.2f}'
                        if profile_data:
                            pt,pc,_=profile_data
                            elapsed=max(0,t[1]-ppo_start[0]) if ppo_start[0] is not None else 0
                            idx=max(0,bisect.bisect_right(pt,elapsed)-1)
                            j.axes[1],j.axes[0],j.axes[3]=pc[idx]
                            stage='ppo_command_history'
                    else:
                        j.buttons[0]=1;stage='return_to_stand'
                        if stand_request[0] is None:stand_request[0]=t[1]
                pub.publish(j);vw.writerow(t+[j.axes[1],j.axes[0],j.axes[3],stage])
            if stand_only and t[2]>=hold_seconds:complete=True;break
            if not stand_only and t[2]>20 and ppo_start[0] is None:raise RuntimeError('PPO PD command not observed')
            if stand_request[0] is not None and t[1]>=stand_request[0]+5:complete=True;break
        if not complete or min(counts)<500:raise RuntimeError(f'incomplete recording: {counts}')
    finally:
        for f in handles:f.close()
        node.destroy_node();rclpy.shutdown()
        (out/'collection.json').write_text(json.dumps({'complete':complete,'counts':counts,
            'clock_origin_sim_s':origin[0], 'hold_seconds':hold_seconds,
            'ppo_pd_start_sim_s':ppo_start[0], 'stand_request_sim_s':stand_request[0],
            'command_vx':vx, 'expected_ppo_kd':expected_kd, 'expected_ppo_kp':expected_kp,
            'stand_only':stand_only,
            'profile_source':str(profile) if profile else None,
            'profile_source_origin_s':profile_data[2] if profile_data else None,
            'profile_interpolation':'zero-order hold; aligned to observed PPO PD; clock advanced by stamped IMU in profile mode' if profile_data else None,
            'schedule':(f'6s getup/stand; continue GetUp standing hold {hold_seconds}s; no PPO request' if stand_only else f'6s getup/stand; '+('captured velocity command history' if profile_data else f'direct PPO vx={vx}')+f'; {hold_seconds}s after PPO PD observed; switch to GetUp/stand and record5s'),
            'relative_time_definition':'t_s = sim_s - clock_origin_sim_s - 6; actual hold start/end saved separately'},indent=2)+'\n')

def run(out, hold_seconds, vx=.5, expected_kd=.5, profile=None, rl_sim=None, gui=False, stand_only=False, expected_kp=25.):
    base=os.environ.copy()
    for k in ('PYTHONHOME','PYTHONPATH','LD_LIBRARY_PATH'):base.pop(k,None)
    raw=subprocess.check_output(['bash','--noprofile','--norc','-c','source /opt/ros/humble/setup.bash\nsource imgo2_deploy/ros2_ws/install/setup.bash\nenv -0'],cwd=ROOT,env=base)
    env=dict(v.decode().split('=',1) for v in raw.split(b'\0') if b'=' in v)
    env.update(ROS_DOMAIN_ID='67',ROS_LOCALHOST_ONLY='1',GAZEBO_MASTER_URI='http://127.0.0.1:11377',LIBGL_ALWAYS_SOFTWARE='1',ROS_LOG_DIR=str(out/'ros_logs'),IMGO2_GAZEBO_URDF=str(ROOT/'imgo2_description/urdf/imgo2_real.gazebo.urdf'))
    env['IMGO2_POLICY_TRACE']=str(out/'policy_raw.csv')
    env['IMGO2_FEEDBACK_DELAY_TRACE']=str(out/'feedback_age.csv')
    paths=['imgo2_deploy/ros2_ws/policy/imgo2/ppo/policy.pt','imgo2_deploy/ros2_ws/policy/imgo2/ppo/config.yaml','imgo2_deploy/ros2_ws/policy/imgo2/base.yaml','imgo2_description/urdf/imgo2_real.gazebo.urdf','imgo2_deploy/ros2_ws/install/lib/imgo2_deploy/rl_sim']
    def fingerprint(path):
        path=Path(path)
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    # Isolated runners do not require a production policy file. Preserve absence
    # as part of the source guard; never create or restore a missing policy.
    if not rl_sim and not (ROOT/paths[0]).is_file():
        raise FileNotFoundError(ROOT/paths[0])
    hashes={p:fingerprint(ROOT/p) for p in paths}
    if rl_sim:
        hashes[str(rl_sim)]=hashlib.sha256(Path(rl_sim).read_bytes()).hexdigest()
    if profile:
        hashes[str(profile)]=hashlib.sha256(Path(profile).read_bytes()).hexdigest()
    meta={'files':hashes,'motor_extra_delay_ms':0,'imu_extra_delay_ms':0,'gui':gui,
          'ros_domain_id':env['ROS_DOMAIN_ID'],'gazebo_master_uri':env['GAZEBO_MASTER_URI'],
          'hold_seconds':hold_seconds,'command_vx':vx,'expected_ppo_kd':expected_kd, 'expected_ppo_kp':expected_kp,
          'stand_only':stand_only,
          'profile_source':str(profile) if profile else None,
          'rl_sim_binary':str(rl_sim) if rl_sim else paths[-1],
          'policy_trace':'paired actual actor input/raw output; state snapshot from same policy call'};procs=[];files=[]
    def launch(name,cmd):
        f=(out/(name+'.log')).open('w');files.append(f)
        p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);procs.append(p);return p
    try:
        launch('gazebo',['ros2','launch','imgo2_deploy','gazebo.launch.py',f'gui:={str(gui).lower()}']);time.sleep(5)
        launch('rl_sim',[str(rl_sim or ROOT/paths[-1]),'--ros-args','-p','use_sim_time:=true',
                        '-p','gazebo_motor_feedback_delay_ms:=0.0','-p','gazebo_imu_feedback_delay_ms:=0.0'])
        deadline=time.monotonic()+70
        while 'loop_keyboard' not in (out/'rl_sim.log').read_text(errors='replace'):
            if any(p.poll() is not None for p in procs) or time.monotonic()>deadline:raise RuntimeError('startup failed; see logs')
            time.sleep(.5)
        child=launch('recorder',['/usr/bin/python3',str(Path(__file__).resolve()),'--collect','--output',str(out),
                                 '--hold-seconds',str(hold_seconds),'--vx',str(vx),'--expected-kd',str(expected_kd),'--expected-kp',str(expected_kp)]+(['--profile',str(profile)] if profile else [])+(['--stand-only'] if stand_only else []))
        deadline=time.monotonic()+310
        while child.poll() is None:
            if any(p.poll() is not None for p in procs[:-1]) or time.monotonic()>deadline:raise RuntimeError('collection interrupted')
            time.sleep(.5)
        if child.returncode:raise RuntimeError('collector failed')
        if not stand_only and 'RL Controller [ppo]' not in (out/'rl_sim.log').read_text(errors='replace'):raise RuntimeError('PPO entry not confirmed')
        meta['completed']=True
    finally:
        for p in reversed(procs):
            if p.poll() is None:os.killpg(p.pid,signal.SIGINT)
        for p in reversed(procs):
            try:p.wait(timeout=8)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=5)
        for f in files:f.close()
        meta['sources_unchanged']=all(fingerprint(ROOT/p)==v for p,v in hashes.items())
        (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--collect',action='store_true')
    p.add_argument('--hold-seconds',type=float,default=10.)
    p.add_argument('--vx',type=float,default=.5,help='Constant forward/backward command, m/s')
    p.add_argument('--expected-kd',type=float,default=.5,help='Expected published PPO damping; does not change gains')
    p.add_argument('--expected-kp',type=float,default=25.,help='Expected published PPO stiffness; does not change gains')
    p.add_argument('--profile',type=Path,help='Paired real policy.csv; replay velocity commands using own Gazebo feedback')
    p.add_argument('--rl-sim',type=Path,help='Diagnostic executable using isolated saved policy/config')
    p.add_argument('--gui',action='store_true')
    p.add_argument('--stand-only',action='store_true',help='Remain in GetUp standing hold; do not enter PPO')
    a=p.parse_args()
    if not 5<=a.hold_seconds<=60:p.error('--hold-seconds must be between 5 and 60')
    if not -.5<=a.vx<=.5 or not .1<=a.expected_kd<=3 or not 10<=a.expected_kp<=40:p.error('Invalid vx or expected damping')
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
    if a.profile:
        a.profile=a.profile.resolve();load_profile(a.profile)
    if a.stand_only and a.profile:p.error('--stand-only cannot be combined with --profile')
    if a.rl_sim:a.rl_sim=a.rl_sim.resolve()
    collect(a.output,a.hold_seconds,a.vx,a.expected_kd,a.profile,a.stand_only,a.expected_kp) if a.collect else run(a.output,a.hold_seconds,a.vx,a.expected_kd,a.profile,a.rl_sim,a.gui,a.stand_only,a.expected_kp)
