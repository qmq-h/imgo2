"""Linux-only multi-bus PTY test for the 12-motor hold demo; no physical device is opened."""
import errno
import os
import pty
import re
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pty_helpers import (MOTOR_IDS, START_Q, GEAR_RATIO, KP_WIRE, KD_WIRE,  # noqa: E402
                         crc16, reply, control_is_zero, target_q, target_dq)

EXE = sys.argv.pop(1)
BUS_COUNT = 4
# 采集轮数：既够做帧级检查，也让“上层 1 kHz / 总线 200 Hz”的双频率比例估计得准一些。
ROUNDS_TO_COLLECT = 20


class DemoTest(unittest.TestCase):
    def scenario(self, kind='normal', fault_bus=0, terminate=signal.SIGTERM):
        masters, slaves = [], []
        for _ in range(BUS_COUNT):
            master, slave = pty.openpty()
            masters.append(master)
            slaves.append(slave)
        env = os.environ.copy()
        env['GO_DEMO_TEST_DEVICES'] = ','.join(os.ttyname(s) for s in slaves)
        # 设成标准坐标系（offset 0、direction +1），让"线上角度 == 逻辑角度"
        # （见 tests/pty_devices.h）。不受用户的标定与镜像方向影响。
        env['GO_TEST_IDENTITY_FRAME'] = '1'

        work_dir = tempfile.mkdtemp(prefix='go_no_log_')
        frames = [[] for _ in range(BUS_COUNT)]
        bufs = [bytearray() for _ in range(BUS_COUNT)]
        silent = [False] * BUS_COUNT
        signalled = False

        with tempfile.TemporaryFile() as output:
            proc = subprocess.Popen([EXE], env=env, cwd=work_dir, stdout=output, stderr=subprocess.PIPE)
            try:
                deadline = time.monotonic() + 20.0
                while time.monotonic() < deadline:
                    ready, _, _ = select.select(masters, [], [], 0.01)
                    for master in ready:
                        bus = masters.index(master)
                        try:
                            bufs[bus].extend(os.read(master, 65536))
                        except OSError as exc:
                            if exc.errno == errno.EIO:
                                continue
                            raise
                        while len(bufs[bus]) >= 17:
                            frame = bytes(bufs[bus][:17])
                            del bufs[bus][:17]
                            self.assertEqual(frame[:2], b'\xfe\xee')
                            self.assertEqual(crc16(frame[:15]),
                                             int.from_bytes(frame[15:], 'little'))
                            self.assertIn(frame[2] & 0x0f, MOTOR_IDS)
                            frames[bus].append(frame)
                            # 故障场景：该总线收完启动探测的 3 帧后就不再应答
                            # （必须用 > 3，否则会连探测的最后一帧都不回，变成 start() 失败）
                            if kind == 'fault' and bus == fault_bus and len(frames[bus]) > 3:
                                silent[bus] = True
                            if not silent[bus]:
                                os.write(master, reply(frame[2] & 0x0f))
                    if (not signalled
                            and all(len(f) >= 3 + 3 * ROUNDS_TO_COLLECT for f in frames)):
                        proc.send_signal(terminate)
                        signalled = True
                    if proc.poll() is not None and not ready:
                        break

                self.assertIsNotNone(proc.poll(), 'demo did not stop within deadline')
                _, err = proc.communicate(timeout=5)
                output.seek(0)
                out = output.read()

                if kind == 'normal':
                    self.assertEqual(proc.returncode, 0, (out[-400:], err))
                    self.assertEqual(err, b'')
                    self.assertLessEqual(len(out.splitlines()), 2, out)
                    self.assertEqual(os.listdir(work_dir), [], '运行不应创建日志文件')
                    for bus in range(BUS_COUNT):
                        own = frames[bus]
                        # 启动探测：前 3 帧是全零控制量，ID 依次为 0、1、2
                        self.assertGreaterEqual(len(own), 3 + 3 * ROUNDS_TO_COLLECT)
                        self.assertEqual([f[2] & 0x0f for f in own[:3]], list(MOTOR_IDS))
                        self.assertTrue(all(control_is_zero(f) for f in own[:3]))
                        # 之后每 3 帧一轮保持：ID 依次 0、1、2，目标=各自启动位置、速度 0
                        motion = own[3:]
                        for start in range(0, len(motion) - 2, 3):
                            group = motion[start:start + 3]
                            if any(control_is_zero(f) for f in group):
                                break
                            self.assertEqual([f[2] & 0x0f for f in group], list(MOTOR_IDS))
                            for mid, frame in zip(MOTOR_IDS, group):
                                self.assertAlmostEqual(target_q(frame), START_Q[mid], delta=0.002)
                                self.assertAlmostEqual(target_dq(frame), 0.0, delta=0.001)
                                self.assertEqual(int.from_bytes(frame[11:13], 'little'), KP_WIRE)
                                self.assertEqual(int.from_bytes(frame[13:15], 'little'), KD_WIRE)

                else:
                    self.assertEqual(proc.returncode, 2, (out[-400:], err))
                    self.assertIn(b'control failed', err)
                    # 探测 3 帧全零；第 4 帧是触发失败的那条运动目标；之后只剩零指令（卸力）
                    own = frames[fault_bus]
                    self.assertTrue(all(control_is_zero(f) for f in own[:3]))
                    self.assertGreater(len(own), 4)
                    self.assertTrue(all(control_is_zero(f) for f in own[4:]))
            finally:
                shutil.rmtree(work_dir, ignore_errors=True)
                if proc.poll() is None:
                    proc.kill()
                proc.communicate()
                for master in masters:
                    os.close(master)
                for slave in slaves:
                    os.close(slave)

    def test_hold_all_buses(self):
        self.scenario()

    def test_sigint(self):
        self.scenario(terminate=signal.SIGINT)

    def test_bus_fault_stops_control(self):
        self.scenario('fault', fault_bus=2)

    def test_missing_test_devices(self):
        env = os.environ.copy()
        env.pop('GO_DEMO_TEST_DEVICES', None)
        proc = subprocess.run([EXE], env=env, capture_output=True, timeout=5)
        self.assertEqual(proc.returncode, 1)
        self.assertIn(b'missing test devices', proc.stderr)


if __name__ == '__main__':
    unittest.main()
