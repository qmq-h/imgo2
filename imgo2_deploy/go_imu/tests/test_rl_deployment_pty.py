"""Full deployment against 4 PTY motor buses + PTY IMU; no real device nodes."""
import fcntl
import math
import os
import pathlib
import pty
import select
import signal
import struct
import subprocess
import sys
import tempfile
import termios
import threading
import time
from pty_helpers import reply, crc16, control_is_zero, target_q, START_Q
def imu_packet():
    payload=bytes([0x41,16])+struct.pack('<iiii',1000000,0,0,0)+bytes([0x20,12])+struct.pack('<iii',0,0,0)
    body=struct.pack('<HB',1,len(payload))+payload
    a=b=0
    for x in body:a=(a+x)&255;b=(b+a)&255
    return b'YS'+body+bytes([a,b])

exe, policy_root=sys.argv[1:]

def scenario(mode):
    pairs=[pty.openpty() for _ in range(5)]
    keyboard_master,keyboard_slave=pty.openpty()
    before=termios.tcgetattr(keyboard_slave)
    stop=threading.Event();pause_imu=threading.Event();freeze_motion=threading.Event();frames=[[] for _ in range(4)];errors=[]
    buffer=b'';proc=None;threads=[]
    def motor_server():
        pending=[b'' for _ in range(4)];positions=[dict(START_Q) for _ in range(4)]
        while not stop.is_set():
            ready=select.select([pair[0] for pair in pairs[:4]],[],[],.01)[0]
            for master in ready:
                bus=[p[0] for p in pairs[:4]].index(master)
                pending[bus]+=os.read(master,65536)
                while len(pending[bus])>=17:
                    frame,pending[bus]=pending[bus][:17],pending[bus][17:]
                    if crc16(frame[:15])!=int.from_bytes(frame[15:],'little'):errors.append('bad CRC')
                    frames[bus].append((time.monotonic(),frame));mid=frame[2]&15
                    if not control_is_zero(frame) and not freeze_motion.is_set():positions[bus][mid]=target_q(frame)
                    data=bytearray(reply(mid))
                    pos=round(positions[bus][mid]*6.33/(2*math.pi)*32768)
                    data[7:11]=struct.pack('<i',pos);data[5:7]=b'\0\0'
                    data[14:16]=struct.pack('<H',crc16(data[:14]))
                    os.write(master,data)
    def imu_server():
        while not stop.is_set():
            if not pause_imu.is_set():os.write(pairs[4][0],imu_packet())
            stop.wait(.01)
    def wait_for(token,timeout=4):
        nonlocal buffer
        deadline=time.monotonic()+timeout
        while token.encode() not in buffer and time.monotonic()<deadline:
            if select.select([proc.stdout],[],[],.01)[0]:
                chunk=os.read(proc.stdout.fileno(),65536)
                if not chunk:break
                buffer+=chunk
            if proc.poll() is not None:break
        assert token.encode() in buffer,(mode,token,buffer.decode(errors='replace'),proc.poll())
    def child_setup():
        os.setsid();fcntl.ioctl(keyboard_slave,termios.TIOCSCTTY,0)
    try:
        for fn in [motor_server,imu_server]:
            t=threading.Thread(target=fn);t.start();threads.append(t)
        with tempfile.TemporaryDirectory(prefix='rl_pty_') as tmp:
            config=pathlib.Path(tmp)/'deployment.yaml'
            config.write_text('deployment:\n  stand_s: 0.2\n  crouch_s: 0.2\n  blend_s: 0.05\n  settle_s: 0.05\n  mapping_confirmed: '+('false' if mode=='locked' else 'true')+'\n')
            env=os.environ.copy()
            env['GO_RL_TEST_DEVICES']=','.join(os.ttyname(p[1]) for p in pairs[:4])
            env['GO_RL_TEST_IMU']=os.ttyname(pairs[4][1])
            if mode=='slow':env['GO_RL_TEST_INFERENCE_DELAY_MS']='200'
            proc=subprocess.Popen([exe,'--policy-dir',policy_root,'--config',str(config)],env=env,
                                  cwd=tmp,stdin=keyboard_slave,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                  preexec_fn=child_setup)
            wait_for('初始保持')
            os.write(keyboard_master,b'0');time.sleep(.05)
            if select.select([proc.stdout],[],[],0)[0]:buffer+=os.read(proc.stdout.fileno(),65536)
            assert '开始站起'.encode() not in buffer
            os.write(keyboard_master,b'\n');wait_for('状态: StandHold')
            time.sleep(.08)
            os.write(keyboard_master,b'1\n')
            if mode=='locked':
                wait_for('PPO锁定')
                os.write(keyboard_master,b'2\n');wait_for('AMP未开放')
                os.write(keyboard_master,b'9\n');wait_for('状态: Passive')
            elif mode=='slow':wait_for('Fault:')
            else:
                wait_for('状态: PPO');time.sleep(.12)
                os.write(keyboard_master,b'w');wait_for('cmd vx=0.10') # no Enter
                os.write(keyboard_master,b' ');wait_for('cmd vx=0.00')
                if mode=='fault':pause_imu.set();wait_for('Fault:')
                elif mode=='arrival':
                    freeze_motion.set();os.write(keyboard_master,b'9\n');wait_for('crouch arrival timeout')
                else:
                    os.write(keyboard_master,b'9\n');wait_for('状态: Passive')
            if mode in ['fault','slow','arrival']:
                fault_time=time.monotonic();time.sleep(.18)
                os.write(keyboard_master,b'0\n');time.sleep(.03)
                for bus in frames:
                    zero=[f for t,f in bus if t>fault_time+.06]
                    assert zero and all(control_is_zero(f) for f in zero),'fault did not unload all buses'
            else:
                # Fixed 60/2 and policy 25/.5 are encoded at the output-side gains.
                fixed=int(60/(6.33**2)*1280); policy=int(25/(6.33**2)*1280)
                assert all(any(int.from_bytes(f[11:13],'little')==fixed for _,f in bus) for bus in frames)
                if mode=='normal':assert all(any(int.from_bytes(f[11:13],'little')==policy for _,f in bus) for bus in frames)
            if mode in ['locked','normal']:
                passive_time=time.monotonic();time.sleep(.12)
                for bus in frames:
                    unloaded=[f for t,f in bus if t>passive_time+.04]
                    # q remains a valid reference; kp/kd/tau/dq must all be zero.
                    assert unloaded and all(f[3:7]==b'\0'*4 and f[11:15]==b'\0'*4 for f in unloaded),'Passive retained force commands'
            os.write(keyboard_master,b'\x03')
            rest,_=proc.communicate(timeout=4);buffer+=rest
            assert proc.returncode==(2 if mode in ['fault','slow','arrival'] else 0),(mode,proc.returncode,buffer)
            assert termios.tcgetattr(keyboard_slave)==before,'terminal not restored'
            assert sorted(os.listdir(tmp))==['deployment.yaml'],'unexpected logs'
            assert not errors,errors
            print('RL full PTY scenario passed:',mode)
    finally:
        if proc and proc.poll() is None:proc.kill();proc.communicate()
        stop.set()
        for t in threads:t.join(timeout=2)
        for pair in pairs:
            for fd in pair:os.close(fd)
        os.close(keyboard_master);os.close(keyboard_slave)
for mode in ['locked','normal','fault','slow','arrival']:scenario(mode)
