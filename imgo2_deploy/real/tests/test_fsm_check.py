"""Run the real FSM/runner through a keyboard PTY, without opening hardware."""
import fcntl
import math
import os
from pathlib import Path
import pty
import re
import select
import signal
import struct
import subprocess
import sys
import tempfile
import termios
import time

exe, policy, config = sys.argv[1:]
ansi = re.compile(rb'\x1b\[[0-9;?]*[A-Za-z]')


def policy_gain(field):
    """Read the first value of rl_kp/rl_kd from policy/ppo/config.yaml."""
    text = (Path(policy) / 'ppo' / 'config.yaml').read_text()
    match = re.search(rf'{field}:\s*\[([^]]+)\]', text)
    assert match, f'{field} missing from policy config'
    return float(match.group(1).split(',')[0])


KP = policy_gain('rl_kp')
KD = policy_gain('rl_kd')


class Session:
    def __init__(self, directory, columns=80):
        self.master, self.slave = pty.openpty()
        fcntl.ioctl(self.slave, termios.TIOCSWINSZ, struct.pack('HHHH', 24, columns, 0, 0))
        self.saved = termios.tcgetattr(self.slave)
        self.process = subprocess.Popen([exe, '--policy-dir', policy, '--config', config],
                                        stdin=self.slave, stdout=self.slave, stderr=self.slave,
                                        cwd=directory)
        self.data = b''

    def pump(self, duration):
        end = time.monotonic() + duration
        while time.monotonic() < end:
            if select.select([self.master], [], [], .02)[0]:
                self.data += os.read(self.master, 65536)

    def text(self):
        return ansi.sub(b'', self.data).replace(b'\r', b'').decode(errors='replace')

    def wait(self, predicate, timeout=4):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self.pump(.03)
            if predicate(self.text()):
                return
            assert self.process.poll() is None, self.text()
        raise AssertionError(self.text()[-4000:])

    def send(self, text):
        os.write(self.master, text.encode())

    def rows(self, state):
        frames = re.findall(r'State: ' + re.escape(state) +
                            r' \| OFFLINE\nTarget\(rad\)[^\n]*\n'
                            r'(FL[^\n]*\nFR[^\n]*\nBL[^\n]*\nBR[^\n]*\n)', self.text())
        return [[float(value) for line in frame.strip().splitlines()
                 for value in line.split()[1:]] for frame in frames]

    def stop(self):
        self.process.send_signal(signal.SIGINT)
        self.process.wait(timeout=3)
        self.pump(.1)
        assert self.process.returncode == 0, self.text()
        assert b'\x1b[?25l' in self.data and b'\x1b[?25h\x1b[?1049l' in self.data
        assert termios.tcgetattr(self.slave) == self.saved

    def close(self):
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait()
        os.close(self.master)
        os.close(self.slave)


with tempfile.TemporaryDirectory(prefix='fsm_check_') as tmp:
    session = Session(tmp)
    try:
        session.wait(lambda t: 'State: InitialHold |' in t)
        session.send('1\n01\n7\n')
        session.pump(.15)
        assert 'State: PPO |' not in session.text() and 'State: Stand |' not in session.text()
        session.send('0')
        session.pump(.15)
        assert 'State: Stand |' not in session.text(), 'state key acted before Enter'
        session.send('\n')
        session.wait(lambda t: 'State: StandHold |' in t, 10)
        standing = session.rows('Stand')
        assert len(standing) > 20
        assert abs(standing[0][2] - standing[-1][2]) > .8, standing
        goal = session.rows('StandHold')[-1]
        assert max(abs(a-b) for a,b in zip(goal, [0,.87,-1.82]*4)) < 1e-5
        session.pump(.65)
        session.send('2\n')
        session.wait(lambda t: 'AMP未开放' in t)
        session.send('1\n')
        session.wait(lambda t: 'State: PPO |' in t)
        session.pump(.8)
        outputs = session.rows('PPO')
        assert len(outputs) > 4 and all(math.isfinite(x) for row in outputs for x in row)
        assert max(abs(a-b) for a,b in zip(outputs[-1], goal)) > .001, outputs
        assert re.search(r'seq=[1-9][0-9]*', session.text())
        session.send('w')
        session.wait(lambda t: 'cmd: 0.10 0.00 0.00' in t)
        offset = len(session.text())
        session.send(' ')
        session.wait(lambda t: 'cmd: 0.00 0.00 0.00' in t[offset:])
        # Return from PPO takes 2s, while early 1 must explain rejection and never queue.
        offset = len(session.text())
        session.send('0\n')
        session.wait(lambda t: 'State: Stand |' in t[offset:])
        session.send('1\n')
        session.wait(lambda t: '尚未就绪' in t[offset:])
        session.wait(lambda t: 'State: StandHold |' in t[offset:], 3)
        session.pump(.65)
        assert 'State: PPO |' not in session.text()[offset:], 'early 1 was queued'
        offset = len(session.text())
        session.send('1\n')
        session.wait(lambda t: 'State: PPO |' in t[offset:])
        session.pump(.8)
        assert f'PD={KP:.2f}/{KD:.2f}' in session.text()[offset:]
        session.send('9\n')
        session.wait(lambda t: 'State: Passive / Unloaded |' in t, 11)
        crouching = session.rows('Passive / Crouching')
        assert len(crouching) > 20 and abs(crouching[-1][2]-crouching[0][2]) > .8
        assert 'PD=0.00/0.00' in session.text()
        session.send('0\n')
        offset = len(session.text())
        session.wait(lambda t: 'State: Stand |' in t[offset:])
        assert 'State: Fault |' not in session.text(), session.text()
        assert session.data.count(b'\x1b[H') > 100
        session.stop()
    finally:
        session.close()

    narrow = Session(tmp, 24)
    try:
        narrow.wait(lambda t: 'Terminal too small' in t)
        narrow.stop()
    finally:
        narrow.close()

    # Pipes remain plain text; EOF also takes the normal cleanup path.
    redirected = subprocess.Popen([exe, '--policy-dir', policy, '--config', config,
                                   '--duration', '.4'], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=tmp)
    redirected.wait(timeout=5)
    output, error = redirected.communicate()
    assert redirected.returncode == 0, error
    assert b'\x1b' not in output and b'State: InitialHold' in output
    closed = subprocess.run([exe, '--policy-dir', policy, '--config', config], input=b'',
                            capture_output=True, timeout=5, cwd=tmp)
    assert closed.returncode == 0, closed.stderr
    assert list(Path(tmp).iterdir()) == [], 'offline test created files'

print('FSM check passed: Enter transitions, repeated 0-1-0-1 with 2s PPO return, early-key rejection, trajectory/PPO targets, immediate velocity, '
      'AMP placeholder, unload/restand, fixed/narrow display, SIGINT/EOF restoration and redirection')
