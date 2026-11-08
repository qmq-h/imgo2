#!/usr/bin/env python3
"""Short Gazebo PPO policy comparison. Source ROS and workspace first.

Uses DDS domain 93/Gazebo port 11367. Temporarily selects PPO policy_1 and
restores config in finally; do not edit that config concurrently. No training.
"""
from pathlib import Path
import os, subprocess, signal, time, json, hashlib, argparse
ROOT=Path(__file__).resolve().parents[3]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--policies',nargs='+',default=['policy.pt','policy_1.pt'],help='PPO filenames or absolute exported-policy snapshots')
parser.add_argument('--cases',default='stand,forward03,backward03,turn_left03')
args=parser.parse_args()
OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=True)
WS=ROOT/'imgo2_deploy/ros2_ws'
cfg=WS/'policy/imgo2/ppo/config.yaml'
original=cfg.read_bytes();replacement=original
assert b'model_name: "policy.pt"' in original
for name in args.policies:
 assert '\n' not in name and '"' not in name and (cfg.parent/name).is_file()
# Separate DDS/Gazebo ports from user's normal terminals.
os.environ.update(ROS_DOMAIN_ID='93',ROS_LOCALHOST_ONLY='1',GAZEBO_MASTER_URI='http://127.0.0.1:11367',LIBGL_ALWAYS_SOFTWARE='1')
def stop(p):
 if p is None: return
 try: os.killpg(p.pid,signal.SIGINT)
 except ProcessLookupError: return
 try: p.wait(timeout=8)
 except subprocess.TimeoutExpired:
  try: os.killpg(p.pid,signal.SIGTERM)
  except ProcessLookupError: pass
  try:p.wait(timeout=4)
  except subprocess.TimeoutExpired:
   try:os.killpg(p.pid,signal.SIGKILL)
   except ProcessLookupError:pass

def start(cmd,path):
 f=path.open('w');p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,cwd=WS);f.close();return p

def wait_for(p,path,marker,seconds):
 end=time.monotonic()+seconds
 while time.monotonic()<end:
  if marker in path.read_bytes():return
  if p.poll() is not None:raise RuntimeError(f'{path}: process exited {p.returncode}')
  time.sleep(.5)
 raise RuntimeError(f'Timed out waiting for {marker!r} in {path}')

CASES={'stand':(0.,0.,0.),'forward03':(.3,0.,0.),'backward03':(-.3,0.,0.),'turn_left03':(0.,0.,.3),'turn_right03':(0.,0.,-.3),'left03':(0.,.3,0.),'right03':(0.,-.3,0.)}
case_names=args.cases.split(',')
assert all(c in CASES for c in case_names)
cases=[(c,*CASES[c]) for c in case_names]
results=[]
try:
 for filename in args.policies:
  label=Path(filename).stem;key=1
  # Absolute evaluation snapshots stay outside deploy; SDK resolves this
  # relative model_name from the existing PPO configuration directory.
  model_name=os.path.relpath(filename,cfg.parent) if Path(filename).is_absolute() else filename
  replacement=original.replace(b'model_name: "policy.pt"',('model_name: "'+model_name+'"').encode())
  cfg.write_bytes(replacement)
  for name,vx,vy,wz in cases:
   stem='gazebo_'+label+'_'+name
   launch=rl=evaluation=None
   print('START',stem,flush=True)
   try:
    lp=OUT/(stem+'_launch.log');rp=OUT/(stem+'_rl.log')
    launch=start(['ros2','launch','imgo2_deploy','gazebo.launch.py','gui:=false','wname:=earth'],lp)
    wait_for(launch,lp,b'Loaded gazebo_ros2_control',55)
    rl=start(['ros2','run','imgo2_deploy','rl_sim'],rp)
    wait_for(rl,rp,b'RL_Sim start',35)
    time.sleep(1)
    evaluation=start(['/usr/bin/python3',str(WS/'scripts/eval_gazebo_policy.py'),'--key',str(key),'--vx',str(vx),'--vy',str(vy),'--wz',str(wz),'--duration','24','--sim-time','--json',str(OUT/(stem+'.json')),'--raw-json',str(OUT/(stem+'_raw.json')),'--transient-json',str(OUT/(stem+'_transient.json'))],OUT/(stem+'_eval.log'))
    code=evaluation.wait(timeout=50)
    if code:raise RuntimeError(f'evaluator exited {code}')
    results.append(dict(case=stem,status='completed'))
    print('DONE',stem,flush=True)
   except Exception as e:
    results.append(dict(case=stem,status='failed',error=str(e)));print('FAILED',stem,str(e),flush=True)
   finally:
    stop(evaluation);stop(rl);stop(launch)
   time.sleep(1)
finally:
 if cfg.read_bytes()==replacement:cfg.write_bytes(original)
 elif cfg.read_bytes()!=original:print('WARNING: concurrent PPO config edit, not overwriting',flush=True)
 (OUT/'gazebo_batch_status.json').write_text(json.dumps(dict(results=results,ppo_config_restored=cfg.read_bytes()==original),indent=2))
