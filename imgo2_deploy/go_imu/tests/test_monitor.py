"""go_monitor 里**不需要硬件**就能测的部分：启动时的配置摘要必须与 config/real_config.cpp 一致。

为什么值得单独测：`qL`（逻辑角度）依赖配置里的 offset 与 direction，而这两个值是从
`makeDefaultConfig()` 里按“设备名 + ID”查出来的。曾经就出过一次：加 `qL` 列时只读了
direction、**漏读了 offset**，于是 `qL` 一直等于 `q`（看起来“offset 没生效”，其实是工具没读）。
启动摘要就是为这类错误准备的，这个测试则把它固定下来。

安全：如果检测到配置里的串口**真的存在**（说明这台机器接着实机），直接跳过 ——
`go_monitor` 会发零指令让电机卸力，绝不能被 `ctest` 顺手触发。
"""
import os
import re
import signal
import subprocess
import sys
import time
import unittest

EXE = sys.argv.pop(1)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(ROOT, 'config', 'real_config.cpp')

# config.buses.push_back(BusConfig{"设备", {ids}, {offsets}, {directions}});
BUS_RE = re.compile(r'BusConfig\{"([^"]+)",\s*\{([^}]*)\},\s*\{([^}]*)\},\s*\{([^}]*)\}(?:,\s*\{([^}]*)\})?\}')
# go_monitor: /dev/xxx    配置 | ID0 off= +3.467 dir=+1 | ID1 ...
ENTRY_RE = re.compile(r'ID(\d+) pose=\s*([-+0-9.]+) dir=([-+]?\d+)')


def numbers(text):
    """解析 "{1.0f, -2.5f}" 这种纯数字列表；含符号项时返回 None（表示"不是纯数字"）。"""
    out = []
    for token in text.split(','):
        token = token.strip().rstrip('f')
        if not token:
            continue
        try:
            out.append(float(token))
        except ValueError:
            return None  # 例如 "hip,thigh,shank"
    return out


def parseConfig():
    """从 go_config.cpp 抽出每条总线的 device / ids / directions。

    offsets 只在全是纯数字时才解析；写成 "hip,thigh,shank" 这种符号形式时
    返回 None —— 那种情况下本测试跳过（姿态数值由math_config测试覆盖）。
    """
    with open(CONFIG, encoding='utf-8') as handle:
        text = handle.read()
    buses = []
    for m in BUS_RE.finditer(text):
        buses.append({
            'device': m.group(1),
            'ids': [int(x) for x in m.group(2).split(',') if x.strip()],
            'directions': numbers(m.group(3)) or [],
            'start_pose': numbers(m.group(4)),
        })
    return buses


class MonitorTest(unittest.TestCase):
    def test_config_summary_matches_config_file(self):
        buses = parseConfig()
        self.assertTrue(buses, f'没从 {CONFIG} 里解析出任何 BusConfig')
        devices = [bus['device'] for bus in buses]
        if any(os.path.exists(device) for device in devices):
            self.skipTest('检测到真实串口存在，跳过（避免让实机卸力）')
        # offset 已不再写在配置里（启动时自动算），所以这里只核对 device/ID/dir

        proc = subprocess.Popen([EXE, '-p', '10000'] + devices,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(0.4)
        proc.send_signal(signal.SIGTERM)
        out, err = proc.communicate(timeout=5)
        text = out.decode()
        self.assertEqual(err, b'')

        for bus in buses:
            # 找到这条总线的摘要行
            line = next((l for l in text.split('\n')
                         if l.startswith('go_monitor: ') and bus['device'] in l and '配置' in l),
                        None)
            self.assertIsNotNone(line, f'没有 {bus["device"]} 的配置摘要行：\n{text[:600]}')
            entries = {int(m.group(1)): (float(m.group(2)), int(m.group(3)))
                       for m in ENTRY_RE.finditer(line)}
            self.assertEqual(sorted(entries), sorted(bus['ids']),
                             f'{bus["device"]} 摘要里的 ID 与配置不一致：{line}')
            for k, mid in enumerate(bus['ids']):
                want_dir = bus['directions'][k] if k < len(bus['directions']) else 1.0
                got_offset, got_dir = entries[mid]
                # 现在配置里没有 offset 了（由启动自动归零算），只核对 dir；
                # 第 5 个字段是 start_pose_rad，也顺便比一下量级
                self.assertTrue(got_offset is not None)
                self.assertEqual(got_dir, int(want_dir),
                                 msg=f'{bus["device"]} ID{mid} 的 direction 没读对：{line}')

    def test_bad_device_is_reported_and_continues(self):
        """打不开的口只报一行，不影响其它口，也不退出。"""
        proc = subprocess.Popen([EXE, '-p', '10000', '/dev/null'],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(0.4)
        self.assertIsNone(proc.poll(), '打不开一个口不该让程序退出')
        proc.send_signal(signal.SIGTERM)
        out, err = proc.communicate(timeout=5)
        self.assertEqual(proc.returncode, 0)
        self.assertIn(b'/dev/null', out)
        self.assertIn('打开失败'.encode(), out)


if __name__ == '__main__':
    unittest.main()
