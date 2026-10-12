"""Integration with simulated devices only; no configured hardware paths are opened."""
import os
import pty
import select
import struct
import subprocess
import sys
import threading
import time
import unittest
from pty_helpers import reply, crc16, control_is_zero
EXE=sys.argv.pop(1)

def imu_packet(fields=True):
    kinds=[(0x41,[1000000,0,0,0])]
    if fields:kinds.append((0x20,[0,0,0]))
    payload=b''.join(bytes([kind,len(values)*4])+struct.pack('<'+'i'*len(values),*values)
                     for kind,values in kinds)
    body=struct.pack('<HB',1,len(payload))+payload
    a=b=0
    for value in body:a=(a+value)&255;b=(b+a)&255
    return b'YS'+body+bytes([a,b])

class HardwareTest(unittest.TestCase):
    def run_case(self,mode):
        pairs=[pty.openpty() for _ in range(3)]
        stop=threading.Event();armed=threading.Event();restore=threading.Event()
        commands=[[],[]];errors=[];fault_seen=[None]
        threads=[]
        def motor_loop(index):
            master=pairs[index][0];pending=b'';injected=False
            while not stop.is_set():
                try:
                    if not select.select([master],[],[],.01)[0]:continue
                    pending+=os.read(master,4096)
                    while len(pending)>=17:
                        frame,pending=pending[:17],pending[17:]
                        if frame[:2]!=b'\xfe\xee' or crc16(frame[:15])!=int.from_bytes(frame[15:],'little'):
                            errors.append('invalid command')
                        commands[index].append((time.monotonic(),frame))
                        motor_id=frame[2]&15
                        dropping=index==0 and motor_id==0 and (
                            mode=='missing_motor' or armed.is_set() and not restore.is_set() and
                            mode in ['motor_failure','diagnostic'])
                        if dropping:continue
                        data=reply(motor_id)
                        if mode=='diagnostic' and index==0 and not injected:
                            bad=bytearray(data);bad[-1]^=255
                            data=bytes(bad)+reply((motor_id+1)%3)+data
                            injected=True
                        os.write(master,data)
                except OSError as error:
                    if not stop.is_set():errors.append(str(error))
                    break
        def imu_loop():
            while not stop.is_set():
                if not (mode=='imu_timeout' and armed.is_set() and not restore.is_set()):
                    try:os.write(pairs[2][0],imu_packet(mode!='missing_fields'))
                    except OSError:return
                stop.wait(.01)
        try:
            for index in range(2):
                thread=threading.Thread(target=motor_loop,args=(index,));thread.start();threads.append(thread)
            thread=threading.Thread(target=imu_loop);thread.start();threads.append(thread)
            imu_path='/tmp/no-imgo2-imu-device' if mode=='missing_imu' else os.ttyname(pairs[2][1])
            proc=subprocess.Popen([EXE,mode,os.ttyname(pairs[0][1]),os.ttyname(pairs[1][1]),imu_path],
                                  stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            output=[];pending='';deadline=time.monotonic()+5
            while proc.poll() is None and time.monotonic()<deadline:
                if not select.select([proc.stdout],[],[],.02)[0]:continue
                pending+=os.read(proc.stdout.fileno(),4096).decode()
                while '\n' in pending:
                    line,pending=pending.split('\n',1);output.append(line+'\n')
                    if line.startswith('ARMED'):
                        armed.set()
                        if mode=='imu_disconnect':
                            os.close(pairs[2][0]);os.close(pairs[2][1]);pairs[2]=(-1,-1)
                    if line.startswith('FAULT'):
                        fault_seen[0]=time.monotonic();restore.set()
            if proc.poll() is None:proc.kill()
            rest,stderr=proc.communicate(timeout=2);output.append(pending+rest)
            self.assertEqual(proc.returncode,0,''.join(output)+stderr)
            self.assertIn('PASS',''.join(output))
            self.assertFalse(errors,errors)
            if mode=='diagnostic':
                self.assertTrue(all(control_is_zero(frame) for bus in commands for _,frame in bus))
            elif mode not in ['missing_imu','missing_fields','missing_motor','motor_stale']:
                self.assertTrue(any(not control_is_zero(frame) for bus in commands for _,frame in bus))
            if fault_seen[0] is not None:
                # Exclude an already in-flight transaction; inspect the autonomous-stop interval,
                # before the deliberate stop/start recovery at 250 ms.
                for bus in commands:
                    selected=[frame for t,frame in bus if fault_seen[0]+.06<t<fault_seen[0]+.21]
                    if mode!='motor_stale':self.assertTrue(selected,'bus did not continue zero probing')
                    self.assertTrue(all(control_is_zero(frame) for frame in selected))
        finally:
            stop.set()
            for thread in threads:thread.join(timeout=1)
            for pair in pairs:
                for fd in pair:
                    if fd>=0:os.close(fd)
    def test_normal(self):self.run_case('normal')
    def test_imu_timeout(self):self.run_case('imu_timeout')
    def test_imu_disconnect(self):self.run_case('imu_disconnect')
    def test_invalid_command(self):self.run_case('invalid_command')
    def test_motor_failure(self):self.run_case('motor_failure')
    def test_motor_stale(self):self.run_case('motor_stale')
    def test_missing_imu(self):self.run_case('missing_imu')
    def test_missing_fields(self):self.run_case('missing_fields')
    def test_missing_motor(self):self.run_case('missing_motor')
    def test_parallel_diagnostics(self):self.run_case('diagnostic')
if __name__=='__main__':unittest.main()
