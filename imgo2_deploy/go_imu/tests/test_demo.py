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
                         crc16, reply, control_is_zero, target_q, target_dq, read_log)

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
        # 每次运行都该新建一个日志文件；指到临时目录，别污染 build/
        log_dir = tempfile.mkdtemp(prefix='go_demo_log_')
        env['GO_DEMO_LOG_DIR'] = log_dir
        # 设成标准坐标系（offset 0、direction +1），让"线上角度 == 逻辑角度"
        # （见 tests/pty_devices.h）。不受用户的标定与镜像方向影响。
        env['GO_TEST_IDENTITY_FRAME'] = '1'

        frames = [[] for _ in range(BUS_COUNT)]
        bufs = [bytearray() for _ in range(BUS_COUNT)]
        silent = [False] * BUS_COUNT
        signalled = False

        with tempfile.TemporaryFile() as output:
            proc = subprocess.Popen([EXE], env=env, stdout=output, stderr=subprocess.PIPE)
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
                    # 上下层双频率：上层每 1 ms 确认一次目标（1 kHz），每条总线每 5 ms
                    # 跑一轮（200 Hz）。若上层是阻塞驱动总线的（旧 control() 那种），
                    # 这两个数会几乎相等。demo 退出时把这两个计数打在 stdout 上。
                    summary = re.search(rb'upper ticks=(\d+) bus rounds=(\d+)', out)
                    self.assertIsNotNone(summary, out[-400:])
                    ticks, rounds = int(summary.group(1)), int(summary.group(2))
                    self.assertGreaterEqual(rounds, ROUNDS_TO_COLLECT, out[-400:])
                    # 下界才是关键：阻塞驱动总线时这个比例会退化成约 1，所以 >2 就能区分。
                    # 上界只是兜"明显算错"，给足余量（机器忙时伪终端应答变慢，比例会变大）。
                    ratio = ticks / rounds
                    self.assertGreater(ratio, 2.0,
                                       f'上层没有明显快于总线：ticks={ticks} rounds={rounds}')
                    self.assertLess(ratio, 20.0,
                                    f'双频率比例异常：ticks={ticks} rounds={rounds}')
                    hz = re.search(rb'upper_hz=(\d+) bus_hz=(\d+)', out)
                    self.assertIsNotNone(hz, out[-400:])
                    upper_hz, bus_hz = int(hz.group(1)), int(hz.group(2))
                    # 同上：下界证明"上层远快于总线"，上界/下界都留足余量
                    self.assertGreater(upper_hz, 200, out[-400:])
                    self.assertGreater(bus_hz, 80, out[-400:])
                    self.assertLess(bus_hz, 400, out[-400:])
                    # 循环质量统计量。必须真的采到：如果读得太晚（比如 stop() 之后
                    # 才读，那时 Loop 已经销毁）会全是 0 —— 这个坑踩过一次。
                    stats = re.search(
                        rb'interval_min_us=(\d+) interval_max_us=(\d+) '
                        rb'body_max_us=(\d+) gap_max_us=(\d+)', out)
                    self.assertIsNotNone(stats, out[-400:])
                    i_min, i_max, body_max, gap_max = (int(x) for x in stats.groups())
                    self.assertGreater(i_min, 0, out[-400:])
                    self.assertGreater(body_max, 0, out[-400:])
                    self.assertGreaterEqual(gap_max, 0, out[-400:])
                    self.assertLessEqual(i_min, i_max, out[-400:])
                    # 5 ms 节拍：间隔贴着 5000us，而且不该出现几十毫秒的卡顿
                    self.assertGreater(i_min, 4000, out[-400:])
                    self.assertLess(i_max, 200000, out[-400:])
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
                    self.checkLog(log_dir)
                else:
                    self.assertEqual(proc.returncode, 2, (out[-400:], err))
                    self.assertIn(b'control failed', err)
                    # 探测 3 帧全零；第 4 帧是触发失败的那条运动目标；之后只剩零指令（卸力）
                    own = frames[fault_bus]
                    self.assertTrue(all(control_is_zero(f) for f in own[:3]))
                    self.assertGreater(len(own), 4)
                    self.assertTrue(all(control_is_zero(f) for f in own[4:]))
            finally:
                shutil.rmtree(log_dir, ignore_errors=True)
                if proc.poll() is None:
                    proc.kill()
                proc.communicate()
                for master in masters:
                    os.close(master)
                for slave in slaves:
                    os.close(slave)

    def checkLog(self, log_dir):
        """每次运行一个日志文件：表头说明列含义，每行 t_s + 12 个目标 + 12 个反馈。"""
        comments, header, rows = read_log(log_dir)
        self.assertTrue(any(line.startswith('# 关节对应关系:') for line in comments), comments)
        expect = (['t_s'] + ['q_des%d' % i for i in range(12)] + ['q%d' % i for i in range(12)])
        self.assertEqual(header, expect)
        self.assertGreater(len(rows), 2, rows[:3])
        for row in rows:
            self.assertEqual(len(row), 25, row)
        times = [float(row[0]) for row in rows]
        self.assertEqual(times, sorted(times), times[:5])

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
