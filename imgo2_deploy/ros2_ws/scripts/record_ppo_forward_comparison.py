#!/usr/bin/env python3
"""Bounded headless +0.5 m/s PPO recording; never accesses real hardware.
Raw ROS command/state/IMU streams keep independent host and simulation times.
"""
import argparse
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

def collect(out):
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
    ow = writer('odom.csv',['host_s','sim_s','t_s','x','y','z','yaw'])
    vw = writer('velocity.csv',['host_s','sim_s','t_s','vx','vy','wz'])
    rclpy.init(); node = Node('ppo_forward_comparison_recorder')
    clock = [None]; origin = [None]; fallen = [False]; counts = [0,0,0,0]
    def stamp():
        return [time.monotonic(),clock[0],clock[0]-origin[0]-6] if origin[0] is not None else None
    def on_clock(m):
        clock[0] = m.clock.sec+m.clock.nanosec*1e-9
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
    def imu(m):
        t=stamp()
        if not t:return
        exact=m.header.stamp.sec+m.header.stamp.nanosec*1e-9
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
        ow.writerow(t+[p.x,p.y,p.z,math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))]);counts[3]+=1
    node.create_subscription(Clock,'/clock',on_clock,qos_profile_sensor_data)
    node.create_subscription(RobotState,'/robot_joint_controller/state',state,qos_profile_sensor_data)
    node.create_subscription(RobotCommand,'/robot_joint_controller/command',command,10)
    node.create_subscription(Imu,'/imu',imu,qos_profile_sensor_data)
    node.create_subscription(Odometry,'/odom',odom,qos_profile_sensor_data)
    pub=node.create_publisher(Joy,'/joy',10)
    deadline=time.monotonic()+120;last=-1;complete=False
    try:
        while rclpy.ok() and time.monotonic()<deadline:
            rclpy.spin_once(node,timeout_sec=.002)
            if fallen[0]:raise RuntimeError('tilt >60 degrees')
            t=stamp()
            if not t:continue
            if t[1]-last>=.02:
                last=t[1];j=Joy();j.buttons=[0]*11;j.axes=[0.0]*8
                if t[2]<0:j.buttons[0]=1
                else:
                    j.buttons[5]=1;j.axes[7]=1.0
                    j.axes[1]=min(.5,max(0,t[2]-2)*.1) # five seconds acceleration, then constant
                pub.publish(j);vw.writerow(t+[j.axes[1],0,0])
            if t[2]>=17:complete=True;break
        if not complete or min(counts)<500:raise RuntimeError(f'incomplete recording: {counts}')
    finally:
        for f in handles:f.close()
        node.destroy_node();rclpy.shutdown()
        (out/'collection.json').write_text(json.dumps({'complete':complete,'counts':counts,'schedule':'6s getup; 2s PPO zero; 5s ramp to .5; 10s hold'},indent=2)+'\n')

def run(out):
    base=os.environ.copy()
    for k in ('PYTHONHOME','PYTHONPATH','LD_LIBRARY_PATH'):base.pop(k,None)
    raw=subprocess.check_output(['bash','--noprofile','--norc','-c','source /opt/ros/humble/setup.bash\nsource imgo2_deploy/ros2_ws/install/setup.bash\nenv -0'],cwd=ROOT,env=base)
    env=dict(v.decode().split('=',1) for v in raw.split(b'\0') if b'=' in v)
    env.update(ROS_DOMAIN_ID='67',ROS_LOCALHOST_ONLY='1',GAZEBO_MASTER_URI='http://127.0.0.1:11377',LIBGL_ALWAYS_SOFTWARE='1',ROS_LOG_DIR=str(out/'ros_logs'),IMGO2_GAZEBO_URDF=str(ROOT/'imgo2_description/urdf/imgo2_real.gazebo.urdf'))
    for k in ('IMGO2_POLICY_TRACE','IMGO2_FEEDBACK_DELAY_TRACE'):env.pop(k,None)
    paths=['imgo2_deploy/ros2_ws/policy/imgo2/ppo/policy.pt','imgo2_deploy/ros2_ws/policy/imgo2/ppo/config.yaml','imgo2_deploy/ros2_ws/policy/imgo2/base.yaml','imgo2_description/urdf/imgo2_real.gazebo.urdf','imgo2_deploy/ros2_ws/install/lib/imgo2_deploy/rl_sim']
    hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
    meta={'files':hashes,'motor_extra_delay_ms':0,'imu_extra_delay_ms':0,'gui':False};procs=[];files=[]
    def launch(name,cmd):
        f=(out/(name+'.log')).open('w');files.append(f)
        p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);procs.append(p);return p
    try:
        launch('gazebo',['ros2','launch','imgo2_deploy','gazebo.launch.py','gui:=false']);time.sleep(5)
        launch('rl_sim',[str(ROOT/paths[-1]),'--ros-args','-p','gazebo_motor_feedback_delay_ms:=0.0','-p','gazebo_imu_feedback_delay_ms:=0.0'])
        deadline=time.monotonic()+70
        while 'loop_keyboard' not in (out/'rl_sim.log').read_text(errors='replace'):
            if any(p.poll() is not None for p in procs) or time.monotonic()>deadline:raise RuntimeError('startup failed; see logs')
            time.sleep(.5)
        child=launch('recorder',['/usr/bin/python3',str(Path(__file__).resolve()),'--collect','--output',str(out)])
        deadline=time.monotonic()+130
        while child.poll() is None:
            if any(p.poll() is not None for p in procs[:-1]) or time.monotonic()>deadline:raise RuntimeError('collection interrupted')
            time.sleep(.5)
        if child.returncode:raise RuntimeError('collector failed')
        if 'RL Controller [ppo]' not in (out/'rl_sim.log').read_text(errors='replace'):raise RuntimeError('PPO entry not confirmed')
        meta['completed']=True
    finally:
        for p in reversed(procs):
            if p.poll() is None:os.killpg(p.pid,signal.SIGINT)
        for p in reversed(procs):
            try:p.wait(timeout=8)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=5)
        for f in files:f.close()
        meta['sources_unchanged']=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==v for p,v in hashes.items())
        (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--collect',action='store_true');a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
    collect(a.output) if a.collect else run(a.output)
