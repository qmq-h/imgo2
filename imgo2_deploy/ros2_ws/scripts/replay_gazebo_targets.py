#!/usr/bin/env python3
"""Bounded SIMULATION ONLY recorded-target replay; uses original ROS2 PD controller.
Builds a temporary diagnostic plugin and URDF, without editing production files.
"""
import argparse
import bisect
import csv
import hashlib
import json
import math
import os
import re
from pathlib import Path
import signal
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
JOINTS = [f'{leg}_{kind}_joint' for leg in ('FL','FR','RL','RR') for kind in ('hip','thigh','shank')]

def read(path):
    with path.open() as f:
        return [{k:float(v) for k,v in row.items() if v not in ('','ppo','stand')} for row in csv.DictReader(f)]

def asof(rows,key,t):
    return rows[max(0,bisect.bisect_right([r[key] for r in rows],t)-1)]

def prepare(source,out,kind,start,duration,height):
    files = [source/'robot.csv'] if kind=='real' else [source/x for x in ('command.csv','state.csv','odom.csv')]
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    if kind=='real':
        rows=read(files[0]); times=[r['t_mono_ms']*.001 for r in rows]
        for r,t in zip(rows,times):r['source_s']=t
        init=asof(rows,'source_s',start)
        # Unknown real translation/height/velocity; preserve logged body orientation.
        initial=[0,0,height]+[init['quat_'+s] for s in ('w','x','y','z')]+[0]*6
        q=[init[f'q_{i:02}'] for i in range(12)];dq=[init[f'dq_{i:02}'] for i in range(12)]
        keys=('q_des','dq_des','tau_ff','kp','kd');timekey='source_s'
    else:
        rows,state,odom=map(read,files)
        # RobotCommand has no source stamp: estimate sim time by odom header vs receipt.
        hosts=[r['host_s'] for r in odom]
        def estimate(h):
            n=min(len(hosts)-1,max(1,bisect.bisect_right(hosts,h)))
            a,b=odom[n-1],odom[n];w=(h-a['host_s'])/(b['host_s']-a['host_s'])
            return a['sim_s']+w*(b['sim_s']-a['sim_s'])
        for r in rows:r['source_s']=estimate(r['host_s'])
        for r in state:r['source_s']=estimate(r['host_s'])
        init=asof(state,'source_s',start);o=asof(odom,'sim_s',start)
        initial=[o[k] for k in ('x','y','z','quat_w','quat_x','quat_y','quat_z','vx','vy','vz','wx','wy','wz')]
        q=[init[f'q_{i:02}'] for i in range(12)];dq=[init[f'dq_{i:02}'] for i in range(12)]
        keys=('q','dq','tau','kp','kd');timekey='source_s'
    initial+=q+dq
    first=asof(rows,timekey,start); chosen=[(0,first)]+[(r[timekey]-start,r) for r in rows if start<r[timekey]<=start+duration]
    if chosen[-1][0]<duration-.05:raise ValueError('source does not cover duration')
    # Keep every logged outer-loop command, even repeated target vectors (ZOH).
    with (out/'targets.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['t_s']+[f'{k}_{i:02}' for k in ('q','dq','tau','kp','kd') for i in range(12)])
        for t,r in chosen:w.writerow([t]+[r[f'{k}_{i:02}'] for k in keys for i in range(12)])
    (out/'initial.csv').write_text(','.join(map(str,initial))+'\n')
    return {'source':str(source),'kind':kind,'source_start_s':start,'duration_s':duration,'source_hashes':hashes,
            'source_rows':len(chosen),'initial':initial,'timing':('real host loop times; unknown wire arrival' if kind=='real' else 'command host receipt mapped to odom header sim times; no command source stamp'),
            'initial_limit':('real base height/translation/linear velocity absent; height assumed; link velocities/contact solver state not captured' if kind=='real' else 'snapshot uses nearest past base/joint samples; link velocities and contact solver state not captured')}

def run(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    meta=prepare(args.source.resolve(),out,args.kind,args.start,args.duration,args.height)
    if args.targets_override:
        override=args.targets_override.resolve()
        replacement=read(override)
        expected=['t_s']+[f'{k}_{i:02}' for k in ('q','dq','tau','kp','kd') for i in range(12)]
        if not replacement or list(replacement[0])!=expected:raise ValueError('invalid override columns')
        tt=[r['t_s'] for r in replacement]
        if tt[0]!=0 or any(b<=a for a,b in zip(tt,tt[1:])) or abs(tt[-1]-args.duration)>.05:raise ValueError('invalid override timing')
        if not all(math.isfinite(v) for r in replacement for v in r.values()):raise ValueError('nonfinite override')
        shutil.copyfile(out/'targets.csv',out/'original_targets.csv')
        shutil.copyfile(override,out/'targets.csv')
        meta['original_source_rows']=meta['source_rows'];meta['source_rows']=len(replacement)
        meta['targets_override']={'path':str(override),'sha256':hashlib.sha256(override.read_bytes()).hexdigest(),'note':'precomputed logged-obs actor outputs; no phase correction'}
    base=os.environ.copy()
    for k in ('PYTHONHOME','PYTHONPATH','LD_LIBRARY_PATH'):base.pop(k,None)
    raw=subprocess.check_output(['bash','--noprofile','--norc','-c','source /opt/ros/humble/setup.bash\nsource imgo2_deploy/ros2_ws/install/setup.bash\nenv -0'],cwd=ROOT,env=base)
    env=dict(v.decode().split('=',1) for v in raw.split(b'\0') if b'=' in v)
    build=out/'plugin_build';src=out/'plugin_src';src.mkdir()
    code=Path(__file__).with_name('gazebo_target_replay.cpp')
    (src/'CMakeLists.txt').write_text('''cmake_minimum_required(VERSION 3.16)
project(imgo2_target_replay)
set(CMAKE_CXX_STANDARD 17)
find_package(gazebo REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(gazebo_ros REQUIRED)
find_package(robot_msgs REQUIRED)
add_library(imgo2_target_replay SHARED "'''+str(code)+'''")
target_include_directories(imgo2_target_replay PRIVATE ${GAZEBO_INCLUDE_DIRS})
target_link_libraries(imgo2_target_replay ${GAZEBO_LIBRARIES})
ament_target_dependencies(imgo2_target_replay gazebo_ros robot_msgs)
''')
    library=getattr(args,'plugin_library',None)
    if library:
        library=Path(library).resolve()
        if not library.is_file():raise ValueError('missing diagnostic plugin library')
        donor=library.parent.parent/'metadata.json'
        if not donor.is_file() or json.loads(donor.read_text()).get('plugin_sha256')!=hashlib.sha256(code.read_bytes()).hexdigest():
            raise ValueError('reused plugin source fingerprint not verified')
        meta['reused_plugin_library']=str(library)
    else:
        with (out/'build.log').open('w') as f:
            subprocess.run(['cmake','-S',str(src),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
            subprocess.run(['cmake','--build',str(build),'-j2'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        library=build/'libimgo2_target_replay.so'
    meta['plugin_library_sha256']=hashlib.sha256(library.read_bytes()).hexdigest()
    urdf=ROOT/'imgo2_description/urdf/imgo2_real.gazebo.urdf'
    protected=[urdf,ROOT/'imgo2_description/config/robot_control_ros2.yaml',ROOT/'imgo2_deploy/ros2_ws/policy/imgo2/ppo/config.yaml',ROOT/'imgo2_deploy/real/config/robot.yaml']
    protected=[p for p in protected if p.exists()]
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    tree=ET.parse(urdf);g=ET.SubElement(tree.getroot(),'gazebo')
    p=ET.SubElement(g,'plugin',name='diagnostic_target_replay',filename=str(library))
    ET.SubElement(p,'directory').text=str(out)
    if getattr(args,'fixed_base',False):
        ET.SubElement(p,'fixed_base').text='true'
    meta['fixed_base']=getattr(args,'fixed_base',False)
    resistance=getattr(args,'resistance_parameters',None)
    if resistance:
        resistance=Path(resistance).resolve()
        resistance_library=Path(args.resistance_library).resolve()
        parameters=json.loads(resistance.read_text())
        if not parameters or not resistance_library.is_file():raise ValueError('Missing resistance parameters/library')
        with (out/'resistance_parameters.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['joint','fs_nm','b_nms','va_rad_s'])
            for joint,par in parameters.items():
                name=joint if joint.endswith('_joint') else joint+'_joint'
                if name not in JOINTS:raise ValueError('Unknown resistance joint')
                if par.get('motor_lag_s',0)!=0:raise ValueError('Gazebo resistance plugin has no motor lag state')
                fs=par.get('fs_nm',0.);b=par.get('b_nms',0.);va=par.get('va_rad_s',.02)
                if not (0<=fs<=2 and 0<=b<=10 and 0<va<=1):raise ValueError('Out of diagnostic resistance bounds')
                w.writerow([name,fs,b,va])
        pg=ET.SubElement(tree.getroot(),'gazebo')
        rp=ET.SubElement(pg,'plugin',name='diagnostic_rack_resistance',filename=str(resistance_library))
        ET.SubElement(rp,'parameters').text=str(out/'resistance_parameters.csv')
        ET.SubElement(rp,'log').text=str(out/'resistance_trace.csv')
        meta['resistance_model']={'equation':'Fs*tanh(dq/Va)+B*dq, additive opposing joint effort','parameters':parameters,
            'parameters_sha256':hashlib.sha256(resistance.read_bytes()).hexdigest(),
            'library_sha256':hashlib.sha256(resistance_library.read_bytes()).hexdigest()}
    tree.write(out/'replay.urdf',encoding='unicode')
    (out/'controller.yaml').write_text('robot_joint_controller:\n  ros__parameters:\n    joints: ['+', '.join(JOINTS)+']\n')
    env.update(ROS_DOMAIN_ID='78',ROS_LOCALHOST_ONLY='1',GAZEBO_MASTER_URI='http://127.0.0.1:11398',LIBGL_ALWAYS_SOFTWARE='1',ROS_LOG_DIR=str(out/'ros_logs'),IMGO2_GAZEBO_URDF=str(out/'replay.urdf'))
    processes=[];handles=[];recorder=None
    def launch(name,cmd):
        f=(out/(name+'.log')).open('w');handles.append(f)
        p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);processes.append(p);return p
    def wait_file(name,limit):
        deadline=time.monotonic()+limit
        while not (out/name).exists():
            if any(p.poll() is not None for p in processes) or time.monotonic()>deadline:raise RuntimeError('wait '+name+' failed; see logs')
            time.sleep(.1)
    try:
        launch('gazebo',['ros2','launch','imgo2_deploy','gazebo.launch.py','gui:='+str(args.gui).lower()])
        wait_file('plugin_ready',65)
        sp=launch('controller',['ros2','run','controller_manager','spawner','robot_joint_controller','-p',str(out/'controller.yaml'),'-c','/controller_manager','--controller-manager-timeout','60'])
        if sp.wait(timeout=75):raise RuntimeError('controller failed')
        processes.remove(sp)
        time.sleep(args.gui_warmup if args.gui else 1)
        if args.gui:
            camera=subprocess.run(['gz','camera','-c','gzclient_camera','-f','imgo2_gazebo'],env=env,capture_output=True,text=True,timeout=5)
            (out/'camera.log').write_text(camera.stdout+camera.stderr)
            meta['camera_follow_returncode']=camera.returncode
        if args.video_encoder:
            windows=subprocess.check_output(['xwininfo','-root','-tree'],env=env,text=True)
            match=re.search(r'(0x[0-9a-f]+) "Gazebo"',windows)
            if not match:raise RuntimeError('Gazebo GUI window not found for recording')
            window_id=match.group(1)
            f=(out/'video.log').open('w');handles.append(f)
            recorder=subprocess.Popen([str(args.video_encoder.resolve()),'-y','-f','x11grab','-draw_mouse','0',
                '-window_id',window_id,'-framerate','30','-i',env['DISPLAY'],
                '-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-c:v','libx264','-preset','ultrafast',
                '-crf','23','-pix_fmt','yuv420p',str(out/'replay_gui.mp4')],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            time.sleep(1)
            if recorder.poll() is not None:raise RuntimeError('GUI recording failed; see video.log')
            meta['gui_window_id']=window_id
        meta['gui_enabled']=args.gui
        meta['release_wall_time']=time.time()
        (out/'release').touch()
        wait_file('done',90)
        meta['completed']=True
        if args.hold_after:
            subprocess.run(['ros2','service','call','/pause_physics','std_srvs/srv/Empty','{}'],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=5,check=True)
            time.sleep(args.hold_after)
    finally:
        if recorder and recorder.poll() is None:
            os.killpg(recorder.pid,signal.SIGINT)
            try:recorder.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(recorder.pid,signal.SIGKILL);recorder.wait()
        for p in reversed(processes):
            if p.poll() is None:os.killpg(p.pid,signal.SIGINT)
        for p in reversed(processes):
            try:p.wait(timeout=8)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=5)
        for f in handles:f.close()
        meta['production_files_unchanged']=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in before.items())
        meta['protected_hashes']=before
        meta['plugin_sha256']=hashlib.sha256(code.read_bytes()).hexdigest()
        meta['controller']='existing robot_joint_controller; no rl_sim/policy runner; original PD/limits'
        (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--kind',choices=('sim','real'),required=True)
    p.add_argument('--start',type=float,required=True);p.add_argument('--duration',type=float,default=8)
    p.add_argument('--height',type=float,default=.2941)
    p.add_argument('--targets-override',type=Path,help='Diagnostic prepared target CSV; preserves initial state from source')
    p.add_argument('--gui',action='store_true',help='Show Gazebo on the local display')
    p.add_argument('--fixed-base',action='store_true',help='Diagnostic world-to-base fixed joint; legs remain dynamic')
    p.add_argument('--plugin-library',type=Path,help='Reuse a diagnostic library whose adjacent metadata verifies current source hash')
    p.add_argument('--gui-warmup',type=float,default=8,help='Seconds for GUI to initialize before motion')
    p.add_argument('--hold-after',type=float,default=0,help='Pause and retain GUI for this many seconds after replay')
    p.add_argument('--video-encoder',type=Path,help='Local ffmpeg executable; record the Gazebo window (requires --gui)')
    a=p.parse_args()
    if not 1<=a.duration<=20:p.error('duration must be 1..20s')
    if not 0<=a.gui_warmup<=30 or not 0<=a.hold_after<=60:p.error('GUI waits must be bounded (warmup <=30s, hold <=60s)')
    if a.video_encoder and not a.gui:p.error('--video-encoder requires --gui')
    run(a)
