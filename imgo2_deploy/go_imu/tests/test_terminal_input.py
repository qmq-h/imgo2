import os
import fcntl
import pty
import select
import signal
import subprocess
import sys
import termios
import time
exe=sys.argv[1]
for mode in ['normal','signal','throw']:
    master, slave=pty.openpty()
    before=termios.tcgetattr(slave)
    def setup():
        os.setsid()
        fcntl.ioctl(slave, termios.TIOCSCTTY, 0)
    proc=subprocess.Popen([exe]+(['throw'] if mode=='throw' else []), stdin=slave,stdout=subprocess.PIPE,stderr=subprocess.PIPE,preexec_fn=setup)
    buffer=b''
    def wait_for(token, timeout=2):
        global buffer
        deadline=time.monotonic()+timeout
        while token not in buffer and time.monotonic()<deadline:
            if select.select([proc.stdout],[],[],.02)[0]:
                chunk=os.read(proc.stdout.fileno(),4096)
                if not chunk:break
                buffer+=chunk
        assert token in buffer,(token,buffer)
    try:
        wait_for(b'READY')
        if mode!='throw':
            raw=termios.tcgetattr(slave)
            assert not raw[3]&termios.ICANON and not raw[3]&termios.ECHO
            assert raw[3]&termios.ISIG
            os.write(master,b'w');wait_for(b'KEY 119')  # no Enter
            os.write(master,b'0');time.sleep(.05)
            if select.select([proc.stdout],[],[],0)[0]:buffer+=os.read(proc.stdout.fileno(),4096)
            assert b'STATE' not in buffer,buffer
            os.write(master,b'\n');wait_for(b'STATE 0')
            os.write(master,b'10\n');time.sleep(.03)
            os.write(master,b'9\n');wait_for(b'STATE 9')
            if mode=='signal':os.write(master,b'\x03')
            else:proc.send_signal(signal.SIGTERM)
        proc.communicate(timeout=3)
        assert proc.returncode==(1 if mode=='throw' else 0),proc.returncode
        assert termios.tcgetattr(slave)==before,'terminal not restored'
    finally:
        if proc.poll() is None:proc.kill();proc.communicate()
        os.close(master);os.close(slave)
print('Immediate keys, Enter states, SIGINT/SIGTERM/exception terminal restoration passed')

# A pipe EOF is distinct from an idle keyboard and must end the input lifecycle.
proc=subprocess.Popen([exe],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
proc.stdin.close()
proc.wait(timeout=3)
assert proc.returncode==0
print('Closed stdin detected without spinning')
