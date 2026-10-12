"""Finite scan and multi-port self-check use fake PTYs only."""
import os
import pty
import select
import subprocess
import sys
import threading
from pty_helpers import reply, control_is_zero, crc16
exe=sys.argv[1]
pairs=[pty.openpty(),pty.openpty()]
stop=threading.Event();frames=[[],[]];errors=[];threads=[]
def serve(index):
    pending=b'';master=pairs[index][0]
    while not stop.is_set():
        if not select.select([master],[],[],.01)[0]:continue
        pending+=os.read(master,4096)
        while len(pending)>=17:
            frame,pending=pending[:17],pending[17:]
            frames[index].append(frame)
            if not control_is_zero(frame) or crc16(frame[:15])!=int.from_bytes(frame[15:],'little'):
                errors.append('invalid or nonzero probe command')
            motor_id=frame[2]&15
            if motor_id<3:os.write(master,reply(motor_id))
try:
    for index in range(2):
        thread=threading.Thread(target=serve,args=(index,));thread.start();threads.append(thread)
    result=subprocess.run([exe,'-t','5','-n','1','-i','0,1,2']+[os.ttyname(p[1]) for p in pairs],
                          stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=5)
    assert result.returncode==0,result.stderr
    assert '并发发送自检' in result.stdout,result.stdout
    assert result.stdout.count('应答正常')>=2,result.stdout
    assert all(len(values)>=16 for values in frames),[len(v) for v in frames]
    assert not errors,errors
    assert 'ID0 应答' in result.stdout and 'ID2 应答' in result.stdout,result.stdout
    help_result=subprocess.run([exe,'--help'],stdout=subprocess.PIPE,text=True,timeout=2)
    assert help_result.returncode==0
    print('Probe finite scan, multi-port send/receive and zero control passed')
finally:
    stop.set()
    for thread in threads:thread.join(timeout=1)
    for pair in pairs:
        for fd in pair:os.close(fd)
