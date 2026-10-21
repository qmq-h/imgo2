"""No hardware: motor/IMU PTYs plus terminal output verify combined monitoring."""
import errno
import fcntl
import os
import pty
import re
import select
import signal
import struct
import subprocess
import sys
import termios
import threading
import time
import unittest
from pty_helpers import reply, control_is_zero, crc16

EXE = sys.argv.pop(1)

def packet(euler=True, tid=1):
    fields = [(0x10, [47000, 202000, 9809000]),
              (0x20, [0, -114592, 0]), (0x41, [1000000, 0, 0, 0])]
    if euler:
        fields.append((0x40, [-307000, 1182000, -57670000]))
    payload = b''.join(bytes([kind, len(values)*4]) + struct.pack('<'+'i'*len(values), *values)
                       for kind, values in fields)
    body = struct.pack('<HB', tid, len(payload)) + payload
    a = b = 0
    for value in body:
        a = (a + value) & 255
        b = (b + a) & 255
    return b'YS' + body + bytes([a,b])

class Rig:
    def __init__(self, terminal=False, size=(30, 250), missing=False):
        self.motor, self.motor_slave = pty.openpty()
        self.imu, self.imu_slave = pty.openpty()
        self.done = threading.Event()
        self.commands = []
        self.errors = []
        self.output = b''
        self.terminal = terminal
        env = dict(os.environ, GO_IMU_TEST_DEVICE='/tmp/no-such-imgo2-imu' if missing else os.ttyname(self.imu_slave))
        if terminal:
            self.output_master, self.output_slave = pty.openpty()
            fcntl.ioctl(self.output_slave, termios.TIOCSWINSZ, struct.pack('HHHH', *size, 0, 0))
            target = self.output_slave
        else:
            target = subprocess.PIPE
        self.proc = subprocess.Popen([EXE, '-p', '50', os.ttyname(self.motor_slave)],
                                     env=env, stdout=target, stderr=subprocess.PIPE)
        self.thread = threading.Thread(target=self.motor_loop)
        self.thread.start()
        self.collector = threading.Thread(target=self.collect)
        self.collector.start()

    def motor_loop(self):
        pending = b''
        try:
            while not self.done.is_set():
                if not select.select([self.motor], [], [], .02)[0]: continue
                pending += os.read(self.motor, 1024)
                while len(pending) >= 17:
                    frame, pending = pending[:17], pending[17:]
                    if frame[:2] != b'\xfe\xee' or crc16(frame[:15]) != int.from_bytes(frame[15:], 'little'):
                        self.errors.append('invalid command')
                    if not control_is_zero(frame): self.errors.append('nonzero control')
                    self.commands.append(frame)
                    os.write(self.motor, reply(frame[2] & 15))
        except OSError as error:
            self.errors.append(str(error))

    def collect(self):
        fd = self.output_master if self.terminal else self.proc.stdout.fileno()
        while not self.done.is_set():
            if select.select([fd], [], [], .02)[0]:
                try: data = os.read(fd, 65536)
                except OSError: break
                if not data: break
                self.output += data

    def send(self, duration=.35, euler=True):
        until = time.monotonic()+duration
        tid = 0
        while time.monotonic()<until:
            # Multiple parsed frames per read must count toward RX rate.
            data = packet(euler, tid)+packet(euler, tid+1)
            os.write(self.imu, data[:3])
            time.sleep(.001)
            os.write(self.imu, data[3:])
            tid=(tid+2)%65536
            time.sleep(.019)

    def close(self):
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
        self.proc.wait(timeout=3)
        time.sleep(.05)
        self.done.set()
        self.thread.join(timeout=1)
        self.collector.join(timeout=1)
        for fd in [self.motor,self.motor_slave,self.imu,self.imu_slave]+(
                    [self.output_master,self.output_slave] if self.terminal else []):
            if fd >= 0: os.close(fd)
        if self.proc.stdout: self.proc.stdout.close()
        self.proc.stderr.close()

class CombinedTest(unittest.TestCase):
    def rig(self, **kw):
        rig=Rig(**kw)
        self.addCleanup(rig.close)
        return rig

    def test_data_rate_stale_missing_and_disconnect(self):
        rig=self.rig()
        rig.send(1.3)
        time.sleep(.1)
        text=rig.output.decode()
        self.assertIn('RECEIVING',text)
        self.assertRegex(text,r'RPY \(deg\)\s+1\.182\s+-0\.307\s+-57\.670')
        rates=[float(x) for x in re.findall(r'RX: ([\d.]+) Hz',text)]
        self.assertTrue(any(70 < rate < 130 for rate in rates),rates)
        self.assertNotIn('\x1b',text)
        rig.send(.25,False)
        time.sleep(.1)
        self.assertRegex(rig.output.decode(),r'RPY \(deg\)\s+--\s+--\s+--')
        time.sleep(.65)
        self.assertIn('STALE',rig.output.decode())
        before=len(rig.commands)
        os.close(rig.imu);rig.imu=-1
        os.close(rig.imu_slave);rig.imu_slave=-1
        time.sleep(.3)
        self.assertIn('IMU error:',rig.output.decode())
        self.assertGreater(len(rig.commands),before)
        self.assertFalse(rig.errors,rig.errors)

    def test_missing_device_does_not_stop_motors(self):
        rig=self.rig(missing=True)
        time.sleep(.3)
        self.assertIn('IMU error:',rig.output.decode())
        self.assertGreater(len(rig.commands),3)
        self.assertIsNone(rig.proc.poll())

    def test_terminal_full_compact_small_and_restore(self):
        rig=self.rig(terminal=True)
        rig.send(.2)
        self.assertIn(b'\x1b[?1049h',rig.output)
        self.assertIn(b'qOut=',rig.output)
        self.assertIn(b'RPY (deg)',rig.output)
        fcntl.ioctl(rig.output_slave,termios.TIOCSWINSZ,struct.pack('HHHH',24,80,0,0))
        time.sleep(.15)
        self.assertIn(b'Compact:',rig.output)
        fcntl.ioctl(rig.output_slave,termios.TIOCSWINSZ,struct.pack('HHHH',8,30,0,0))
        start=len(rig.output)
        time.sleep(.2)
        self.assertIn(b'Terminal too small',rig.output[start:])
        # Each painted line fits, and each frame has at most rows-1 newlines.
        frames=rig.output[start:].split(b'\x1b[H')[1:]
        for frame in frames:
            self.assertLessEqual(frame.count(b'\n'),7)
            for line in frame.split(b'\r\n'):
                clean=re.sub(rb'\x1b\[[0-9;?]*[A-Za-z]',b'',line)
                self.assertLessEqual(len(clean),29)
        rig.proc.send_signal(signal.SIGINT)
        rig.proc.wait(timeout=3)
        time.sleep(.1)
        self.assertIn(b'\x1b[?25h\x1b[?1049l',rig.output)
        self.assertEqual(rig.proc.returncode,0)

if __name__ == '__main__': unittest.main()
