#!/usr/bin/env python3
"""Bounded simulation-only defense media capture; production assets/policies are read only.

Run with the Isaac Python for plot/render preparation, and system Python in sourced
ROS for Gazebo collection. No real robot devices or training are accessed.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
WS = ROOT / 'imgo2_deploy/ros2_ws'
OUT = ROOT / 'imgo2_deploy/logs/defense_media_2026-11-22'
DELIVER = OUT / 'deliverables'
def encoder():
    configured=os.environ.get('IMGO2_MEDIA_FFMPEG') or shutil.which('ffmpeg')
    if configured:return configured
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def terrain_boxes(kind):
    boxes=[]
    def box(name,x,y,z,sx,sy,sz,color): boxes.append(dict(name=name,pos=(x,y,z),size=(sx,sy,sz),color=color))
    if kind=='stairs':
        h=float(os.environ.get('IMGO2_MEDIA_STAIR_HEIGHT','0.05'));depth=.30;start=1.2
        if not 0<h<=.08:raise ValueError('stair height must be in (0,.08] metres')
        for i in range(3):
            z=h*(i+1);box(f'step_up_{i}',start+(i+.5)*depth,0,z/2,depth,2.4,z,(.62,.69,.76,1))
        box('top_platform',2.55,0,3*h/2,.9,2.4,3*h,(.62,.69,.76,1))
        for i in range(3):
            z=h*(3-i);box(f'step_down_{i}',3.0+(i+.5)*depth,0,z/2,depth,2.4,z,(.62,.69,.76,1))
    if kind=='rough':
        rng=random.Random(42)
        for i in range(16):
            for j in range(8):
                z=rng.uniform(.005,.020);box(f'rough_{i}_{j}',1.2+(i+.5)*.25,(j-3.5)*.30,z/2,.25,.30,z,(.58,.66,.58,1))
    if kind=='slope':
        rise=math.tan(math.radians(5))*1.8
        for i in range(72):
            z=rise*((i+.5)/36 if i<36 else (72-i-.5)/36)
            box(f'ramp_{i}',1.2+(i+.5)*.05,0,z/2,.05,2.4,z,(.65,.64,.55,1))
    return boxes

def prepare():
    DELIVER.mkdir(parents=True,exist_ok=True)
    inputs=OUT/'inputs';inputs.mkdir(exist_ok=True)
    cfg=WS/'policy/imgo2/ppo/config.yaml';base=WS/'policy/imgo2/base.yaml'
    import yaml
    c=yaml.safe_load(cfg.read_text());c['imgo2/ppo']['rl_kd']=[.8]*12
    b=yaml.safe_load(base.read_text());b['imgo2']['mujoco_motor_feedback_delay_ms']=0.;b['imgo2']['mujoco_imu_feedback_delay_ms']=0.
    (inputs/'policy_config.yaml').write_text(yaml.safe_dump(c,sort_keys=False))
    (inputs/'base_config.yaml').write_text(yaml.safe_dump(b,sort_keys=False))
    for iteration in (5000,6000):
        source=ROOT/'imgo2_deploy/logs/ppo_nov22_review'/('nov22_5000.pt' if iteration==5000 else 'intermediate/nov22_6000.pt')
        saved=inputs/str(iteration);saved.mkdir(exist_ok=True)
        shutil.copyfile(source,saved/'policy_used.pt')
        for name in ('policy_config.yaml','base_config.yaml'):shutil.copyfile(inputs/name,saved/name)
    body=ET.parse(ROOT/'imgo2_description/mjcf/imgo2.xml')
    for mesh in body.findall('.//mesh'):
        mesh.set('file',str((ROOT/'imgo2_description/mjcf'/mesh.get('file')).resolve()))
    body.write(inputs/'robot.xml')
    for kind in ('flat','stairs','rough','slope'):
        scene=ET.parse(ROOT/'imgo2_description/mjcf/scene.xml');root=scene.getroot();root.find('include').set('file',str(inputs/'robot.xml'))
        visual=root.find('visual');ET.SubElement(visual,'quality',shadowsize='4096',offsamples='4')
        asset=root.find('asset')
        for tex in asset.findall('texture'):
            if tex.get('name')=='groundplane':tex.set('rgb1','.25 .29 .33');tex.set('rgb2','.21 .25 .29');tex.set('markrgb','.40 .44 .48')
        for material in asset.findall('material'):material.set('reflectance','0');material.set('texrepeat','1.5 1.5')
        world=root.find('worldbody')
        for box in terrain_boxes(kind):
            ET.SubElement(world,'geom',name=box['name'],type='box',pos=' '.join(map(str,box['pos'])),size=' '.join(str(v/2) for v in box['size']),rgba=' '.join(map(str,box['color'])),contype='1',conaffinity='2',friction='1 .005 .0001')
        scene.write(inputs/f'{kind}.xml')
    production=[cfg,base,ROOT/'imgo2_description/mjcf/imgo2.xml',ROOT/'imgo2_description/urdf/imgo2_real.gazebo.urdf',WS/'policy/imgo2/ppo/policy.pt']
    manifest={'production_before':{str(p):digest(p) for p in production},'policies':{str(i):digest(inputs/str(i)/'policy_used.pt') for i in (5000,6000)},'scope':'native simulator media, bounded deterministic scenarios, no training/hardware','terrain':{'stairs':'3 up/3 down, default 5 cm rise, 30 cm tread, .9 m platform; optional IMGO2_MEDIA_STAIR_HEIGHT override','rough':'seed42 5-20mm tiled surface; not identical to Isaac height field','slope':'5deg up/down, staircase approximation 5cm cells'},'pd':[25,.8],'additional_feedback_delay_ms':0}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))

def simulate_mujoco():
    inputs=OUT/'inputs';bench=OUT/'benchmark'
    if not bench.exists():
        # Reuse the repository benchmark build; do not replace installed binaries.
        subprocess.run(['/usr/bin/python3',str(ROOT/'imgo2_rl/scripts/tools/eval_mujoco_policies.py'),'--output',str(OUT/'build_benchmark'),'--policies',str(inputs/'5000/policy_used.pt'),'--cases','forward03','--seconds','8','--policy-config',str(inputs/'policy_config.yaml'),'--base-config',str(inputs/'base_config.yaml')],check=True)
        shutil.copy2(OUT/'build_benchmark/benchmark',bench)
    bench.chmod(0o755)
    results=[]
    cases=[('flat',5000,.5,0.,0.),('stairs',6000,.3,0.,0.),('rough',6000,.3,0.,0.),('slope',6000,.3,0.,0.)]
    for kind,iteration,vx,vy,wz in cases:
        folder=OUT/('mujoco_'+kind);folder.mkdir(exist_ok=True)
        cmd=[str(bench),str(inputs/f'{kind}.xml'),str(inputs/str(iteration)/'policy_used.pt'),str(inputs/'base_config.yaml'),str(inputs/'policy_config.yaml'),str(folder/'states.csv'),str(vx),str(vy),str(wz),'24']
        run=subprocess.run(cmd,text=True,capture_output=True,timeout=120)
        (folder/'run.log').write_text(run.stdout+run.stderr);run.check_returncode()
        result=json.loads(run.stdout.strip().splitlines()[-1]);result.update(kind=kind,iteration=iteration,command=[vx,vy,wz])
        with (folder/'states.csv').open() as f:rows=list(csv.DictReader(f))
        valid=[r for r in rows if float(r['t_s'])>=0]
        result['x_start']=float(valid[0]['x_m']);result['x_end']=float(valid[-1]['x_m']);result['end_time_s']=float(valid[-1]['t_s'])
        result['roll_abs_max_deg']=max(abs(float(r['roll_rad'])) for r in valid)*180/math.pi
        result['pitch_abs_max_deg']=max(abs(float(r['pitch_rad'])) for r in valid)*180/math.pi
        result['passed_terrain']=kind=='flat' or result['x_end']>(3.9 if kind=='stairs' else (5.2 if kind=='rough' else 4.8))
        (folder/'metrics.json').write_text(json.dumps(result,indent=2));results.append(result);print(result,flush=True)
    (OUT/'mujoco_results.json').write_text(json.dumps(results,indent=2))

def render_mujoco(kind,start=3.,end=20.):
    folder=OUT/('mujoco_'+kind);data=json.loads((folder/'metrics.json').read_text());end=min(end,data['end_time_s'])
    cmd=[str(OUT/'render_mujoco'),str(OUT/'inputs'/f'{kind}.xml'),str(folder/'states.csv'),'1920','1080','30',str(start),str(end),'125','1.65']
    with (folder/'render.log').open('w') as log:
        native=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=log)
        video=DELIVER/f'mujoco_{kind}_1080p.mp4'
        encode=subprocess.run([encoder(),'-y','-f','rawvideo','-pixel_format','rgb24','-video_size','1920x1080','-framerate','30','-i','pipe:0','-an','-c:v','libx264','-preset','medium','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=native.stdout,stdout=log,stderr=log,timeout=400)
        native.stdout.close();code=native.wait(timeout=20)
        if code or encode.returncode:raise RuntimeError('native render/encode failed')
    from PIL import Image
    for angle,name in ((125,'three_quarter'),(90,'side')) if kind=='flat' else ((125,kind),):
        frame_time=5.0 if kind=='flat' else min(end,8.)
        raw=subprocess.run([str(OUT/'render_mujoco'),str(OUT/'inputs'/f'{kind}.xml'),str(folder/'states.csv'),'3840','2160','30',str(frame_time),str(frame_time),str(angle),'1.4'],capture_output=True,timeout=60,check=True).stdout
        Image.frombytes('RGB',(3840,2160),raw).save(DELIVER/f'mujoco_{name}_4k.png',dpi=(300,300))
    (folder/'render_metadata.json').write_text(json.dumps({'native_renderer':'MuJoCo','source_csv_sha256':digest(folder/'states.csv'),'start_s':start,'end_s':end,'fps':30,'video_resolution':[1920,1080],'screenshot_resolution':[3840,2160],'note':'native rendering of measured closed-loop simulator poses, linear interpolation at video frame times; playback 1x'},indent=2))

def gazebo_world(kind,folder,*,stairs_camera=False):
    world_tree=ET.parse(WS/'src/imgo2_deploy/worlds/earth.world');world=world_tree.getroot().find('world')
    for box in terrain_boxes(kind):
        model=ET.SubElement(world,'model',name=box['name']);ET.SubElement(model,'static').text='true'
        ET.SubElement(model,'pose').text=' '.join(map(str,box['pos']))+' 0 0 0';link=ET.SubElement(model,'link',name='body')
        for tag in ('collision','visual'):
            item=ET.SubElement(link,tag,name=tag);geometry=ET.SubElement(item,'geometry');shape=ET.SubElement(geometry,'box');ET.SubElement(shape,'size').text=' '.join(map(str,box['size']))
            if tag=='visual':
                mat=ET.SubElement(item,'material');ET.SubElement(mat,'ambient').text=' '.join(map(str,box['color']));ET.SubElement(mat,'diffuse').text=' '.join(map(str,box['color']))
            else:
                friction=ET.SubElement(ET.SubElement(item,'surface'),'friction');ode=ET.SubElement(friction,'ode');ET.SubElement(ode,'mu').text='1';ET.SubElement(ode,'mu2').text='1'
    follow=ET.SubElement(world,'plugin',name='media_follow',filename=str(OUT/'libmedia_follow.so'))
    if stairs_camera:
        ET.SubElement(follow,'offset').text='-.5 -1.8 .75 0 .40 1.14'
    model=ET.SubElement(world,'model',name='defense_camera');ET.SubElement(model,'static').text='true';ET.SubElement(model,'pose').text='-1.45 -1.65 1.1 0 .32 .85';link=ET.SubElement(model,'link',name='camera')
    for name,width,height,rate in [('video',1920,1080,30),('still',3840,2160,.2)]:
        frames=folder/name;frames.mkdir(exist_ok=True)
        sensor=ET.SubElement(link,'sensor',name=name,type='camera');ET.SubElement(sensor,'always_on').text='true';ET.SubElement(sensor,'update_rate').text=str(rate)
        camera=ET.SubElement(sensor,'camera');ET.SubElement(camera,'horizontal_fov').text='0.9' if stairs_camera else '0.8';image=ET.SubElement(camera,'image');ET.SubElement(image,'width').text=str(width);ET.SubElement(image,'height').text=str(height);ET.SubElement(image,'format').text='R8G8B8'
        clip=ET.SubElement(camera,'clip');ET.SubElement(clip,'near').text='.05';ET.SubElement(clip,'far').text='100'
        plugin=ET.SubElement(sensor,'plugin',name='media_frames_'+name,filename=str(OUT/'libmedia_frames.so'));ET.SubElement(plugin,'output').text=str(frames)
    path=folder/'capture.world';world_tree.write(path);return path

def collect_gazebo(kind, *, label=None, iteration=None, vx=None, seconds=24, stairs_camera=False):
    folder=OUT/('gazebo_'+(label or kind));folder.mkdir(exist_ok=False)
    world=gazebo_world(kind,folder,stairs_camera=stairs_camera)
    base=os.environ.copy()
    for key in ('PYTHONHOME','PYTHONPATH','LD_LIBRARY_PATH'):base.pop(key,None)
    raw=subprocess.check_output(['bash','--noprofile','--norc','-c','source /opt/ros/humble/setup.bash\nsource imgo2_deploy/ros2_ws/install/setup.bash\nenv -0'],cwd=ROOT,env=base)
    env=dict(item.decode().split('=',1) for item in raw.split(b'\0') if b'=' in item)
    env.update(ROS_DOMAIN_ID='88',ROS_LOCALHOST_ONLY='1',GAZEBO_MASTER_URI='http://127.0.0.1:11428',ROS_LOG_DIR=str(folder/'ros_logs'),IMGO2_GAZEBO_URDF=str(ROOT/'imgo2_description/urdf/imgo2_real.gazebo.urdf'),IMGO2_POLICY_TRACE=str(folder/'policy_raw.csv'),GAZEBO_MODEL_DATABASE_URI='')
    env.pop('LIBGL_ALWAYS_SOFTWARE',None)
    if iteration is None:iteration=5000 if kind=='flat' else 6000
    if vx is None:vx=.5 if kind=='flat' else .3
    tree=OUT/'gazebo_runner/policy/imgo2/ppo/policy.pt';shutil.copyfile(OUT/f'inputs/{iteration}/policy_used.pt',tree)
    processes=[];files=[]
    def launch(name,cmd):
        handle=(folder/(name+'.log')).open('w');files.append(handle)
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True);processes.append(proc);return proc
    try:
        launch('gazebo',['ros2','launch','imgo2_deploy','gazebo.launch.py','gui:=false','wname:='+str(world.with_suffix(''))]);time.sleep(5)
        launch('controller',[str(OUT/'gazebo_runner/rl_sim_saved'),'--ros-args','-p','use_sim_time:=true','-p','gazebo_motor_feedback_delay_ms:=0.0','-p','gazebo_imu_feedback_delay_ms:=0.0'])
        deadline=time.monotonic()+90
        while 'loop_keyboard' not in (folder/'controller.log').read_text(errors='replace'):
            if any(p.poll() is not None for p in processes) or time.monotonic()>deadline:raise RuntimeError('Gazebo startup failed; inspect '+str(folder))
            time.sleep(.5)
        child=launch('collector',['/usr/bin/python3',str(WS/'scripts/record_ppo_forward_comparison.py'),'--collect','--output',str(folder),'--hold-seconds',str(seconds),'--vx',str(vx),'--expected-kd','.8'])
        code=child.wait(timeout=310)
        if code:raise RuntimeError('Gazebo motion collection failed')
        (folder/'capture_metadata.json').write_text(json.dumps({'kind':kind,'iteration':iteration,'policy_sha256':digest(tree),'urdf_sha256':digest(ROOT/'imgo2_description/urdf/imgo2_real.gazebo.urdf'),'native_renderer':'Gazebo CameraSensor','scope':'own-feedback closed-loop PPO; translation-follow camera only','additional_feedback_delay_ms':0,'pd':[25,.8]},indent=2))
    finally:
        for proc in reversed(processes):
            if proc.poll() is None:
                try:os.killpg(proc.pid,signal.SIGINT)
                except ProcessLookupError:pass
        for proc in reversed(processes):
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=5)
        for handle in files:handle.close()

def encode_gazebo(kind):
    import bisect
    from PIL import Image
    folder=OUT/('gazebo_'+kind);meta=json.loads((folder/'collection.json').read_text());origin=meta['ppo_pd_start_sim_s'];start=origin+2;end=origin+22
    with (folder/'video/frames.csv').open() as f:rows=list(csv.DictReader(f))
    times=[float(r['sim_s']) for r in rows];chosen=[]
    for index in range(round((end-start)*30)):
        t=start+index/30;pos=min(len(times)-1,bisect.bisect_left(times,t));path=folder/'video'/f'frame_{int(rows[pos]["frame"]):06d}.jpg'
        if not path.exists():path=path.with_suffix('.png')
        chosen.append(path)
    listing=folder/'video_frames.txt'
    listing.write_text(''.join("file '"+str(p)+"'\nduration 0.033333333333\n" for p in chosen)+"file '"+str(chosen[-1])+"'\n")
    with (folder/'encode.log').open('w') as log:
        subprocess.run([encoder(),'-y','-f','concat','-safe','0','-i',str(listing),'-an','-vf','fps=30','-c:v','libx264','-crf','17','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',str(DELIVER/f'gazebo_{kind}_1080p.mp4')],stdout=log,stderr=log,check=True,timeout=180)
    with (folder/'still/frames.csv').open() as f:stills=list(csv.DictReader(f))
    row=min(stills,key=lambda r:abs(float(r['sim_s'])-(origin+6)))
    image=Image.open(folder/'still'/f'frame_{int(row["frame"]):06d}.png');image.save(DELIVER/f'gazebo_{kind}_4k.png',dpi=(300,300))
    (folder/'encode_metadata.json').write_text(json.dumps({'start_sim_s':start,'end_sim_s':end,'fps':30,'frames':len(chosen),'unique_native_frames':len(set(chosen)),'native_time_range':[times[0],times[-1]],'render':'native Gazebo camera, resampled to simulation time; no speedup'},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','simulate-mujoco','render-mujoco','gazebo','encode-gazebo']);parser.add_argument('--kind',default='flat',choices=['flat','stairs','rough','slope']);parser.add_argument('--start',type=float,default=3);parser.add_argument('--end',type=float,default=20)
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    if args.action=='simulate-mujoco':simulate_mujoco()
    if args.action=='render-mujoco':render_mujoco(args.kind,args.start,args.end)
    if args.action=='gazebo':collect_gazebo(args.kind)
    if args.action=='encode-gazebo':encode_gazebo(args.kind)
