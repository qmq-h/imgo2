"""`read_tfevents.py` 的离线回归测试（纯标准库，不需要 tensorboard / torch / GPU）。

这里**自己合成** event 文件（按 TFRecord 帧 + `tensorflow.Event` 的 protobuf 手写字节），
而不是依赖仓库里已有的训练日志：日志会被清理或换机器，合成数据才能稳定复现解析口径。
覆盖三件容易写错的事：① 一帧里多个 tag、同一 tag 跨多帧的作用域；② 正在写一半的尾帧要被丢弃；
③ "取不超过目标迭代点的最后一个值"的取值语义。

Run: python3 -m unittest discover -s tests -p test_read_tfevents.py
"""

import os
import struct
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "tools"))

import read_tfevents as rt  # noqa: E402


def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def _key(field: int, wire: int) -> bytes:
    return _varint((field << 3) | wire)


def _event(step: int, scalars: dict[str, float], wall_time: float = 0.0) -> bytes:
    """构造一个 `Event` protobuf（只带 wall_time / step / Summary.simple_value）。"""
    summary = b""
    for tag, value in scalars.items():
        encoded_tag = tag.encode()
        value_msg = (_key(1, 2) + _varint(len(encoded_tag)) + encoded_tag
                     + _key(2, 5) + struct.pack("<f", value))
        summary += _key(1, 2) + _varint(len(value_msg)) + value_msg
    body = (_key(1, 1) + struct.pack("<d", wall_time)
            + _key(2, 0) + _varint(step)
            + _key(5, 2) + _varint(len(summary)) + summary)
    return body


def _frame(payload: bytes) -> bytes:
    # 帧 = uint64 长度 | uint32 masked_crc | 载荷 | uint32 masked_crc（CRC 只被跳过，不校验）
    return struct.pack("<Q", len(payload)) + b"\x00\x00\x00\x00" + payload + b"\x00\x00\x00\x00"


def _write(path, records, tail: bytes = b"") -> None:
    with open(path, "wb") as handle:
        for record in records:
            handle.write(_frame(record))
        handle.write(tail)


class ReadTfeventsTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_round_trip_and_step_scoping(self):
        path = os.path.join(self.tmp.name, "events.out.tfevents.1")
        _write(path, [
            _event(0, {"A": 1.0, "B": -2.5}),
            _event(1, {"A": 3.0}),
            _event(2, {"A": 5.0, "B": 0.25}),
        ])
        series = rt.read_scalars(path)
        self.assertEqual(sorted(series), ["A", "B"])
        self.assertEqual(series["A"], [(0, 1.0), (1, 3.0), (2, 5.0)])
        # B 只在 0、2 帧出现 ⇒ 不被补零，也不串到别的 step
        self.assertEqual(series["B"], [(0, -2.5), (2, 0.25)])

    def test_truncated_tail_is_dropped(self):
        """训练进行中文件尾部是"写了一半"的帧，必须安全丢弃而不是抛异常。"""
        path = os.path.join(self.tmp.name, "events.out.tfevents.2")
        good = _frame(_event(0, {"A": 1.0})) + _frame(_event(1, {"A": 2.0}))
        partial = struct.pack("<Q", 500) + b"\x00\x00\x00\x00" + b"\x10\x20"   # 声明 500 字节却只有 2 字节
        with open(path, "wb") as handle:
            handle.write(good + partial)
        series = rt.read_scalars(path)
        self.assertEqual(series["A"], [(0, 1.0), (1, 2.0)])

    def test_value_at_uses_last_value_not_after_target(self):
        points = [(0, 1.0), (500, 2.0), (1500, 3.0)]
        self.assertEqual(rt.value_at(points, 0), 1.0)
        self.assertEqual(rt.value_at(points, 499), 1.0)
        self.assertEqual(rt.value_at(points, 500), 2.0)
        self.assertEqual(rt.value_at(points, 1499), 2.0)
        self.assertEqual(rt.value_at(points, 99999), 3.0)
        self.assertTrue(rt.value_at(points, -1) != rt.value_at(points, -1))   # nan（目标早于首个点）

    def test_resolve_event_file_picks_newest_and_accepts_paths(self):
        root = os.path.join(self.tmp.name, "logs", "amp_rsl_rl")
        old_run = os.path.join(root, "exp", "2026-01-01_00-00-00")
        new_run = os.path.join(root, "exp", "2026-01-02_00-00-00")
        for run, value in ((old_run, 1.0), (new_run, 2.0)):
            os.makedirs(run)
            _write(os.path.join(run, "events.out.tfevents.%s" % os.path.basename(run)),
                   [_event(0, {"A": value})])
        os.utime(os.path.join(old_run, os.listdir(old_run)[0]), (1_000_000, 1_000_000))
        os.utime(os.path.join(new_run, os.listdir(new_run)[0]), (2_000_000, 2_000_000))

        picked = rt.resolve_event_file(None, root)
        self.assertEqual(picked, os.path.join(new_run, os.listdir(new_run)[0]))
        # 直接给文件、给 run 目录（或实验目录）都要能用
        self.assertEqual(rt.resolve_event_file(picked, root), picked)
        self.assertEqual(rt.resolve_event_file(root, root), picked)
        with self.assertRaises(SystemExit):
            rt.resolve_event_file(os.path.join(self.tmp.name, "nope"), root)

    def test_clock_axis_tags_do_not_inflate_the_iteration(self):
        """`*/time` 系列用墙钟秒当 x 轴（rsl_rl 的 tot_time），不能被当成迭代轮号。

        2026-09-18 我自己的第一版工具就对全部 tag 取 max step，于是把秒数（6903）当成轮号报出去，
        而真实进度是 5595 —— 两者相差一个「每轮秒数」。这里用合成数据锁住这个陷阱。
        """
        path = os.path.join(self.tmp.name, "events.out.tfevents.4")
        # 真实文件里每个 tag 各写一条记录、各自的 step 不同：迭代轴记轮号，*/time 记墙钟秒
        _write(path, [_event(10, {"Train/mean_reward": 1.0}),
                      _event(80, {"Train/mean_reward": 2.0}),
                      _event(60, {"Train/mean_reward/time": 12.0}),
                      _event(100, {"Train/mean_reward/time": 100.0})])
        series = rt.read_scalars(path)
        self.assertEqual(series["Train/mean_reward"][-1][0], 80)
        self.assertEqual(series["Train/mean_reward/time"][-1][0], 100)   # 秒
        step, tag = rt.iteration_step(series)
        self.assertEqual((step, tag), (80, "Train/mean_reward"))
        import contextlib
        import io
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self.assertEqual(rt.main([path]), 0)
        out = buffer.getvalue()
        self.assertIn("iterations=80", out)
        self.assertIn("墙钟秒", out)                                       # 明确提示时钟轴 tag

    def test_main_cli_prints_table_and_tags(self):
        path = os.path.join(self.tmp.name, "events.out.tfevents.3")
        _write(path, [_event(0, {"AMP/mean_root_height_m": 0.25, "Custom/metric": 9.0,
                                 "Train/mean_reward": 1.0}),
                      _event(1000, {"AMP/mean_root_height_m": 0.30, "Custom/metric": 5.0,
                                    "Train/mean_reward": 2.0})])
        import contextlib
        import io

        def run(args):
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                self.assertEqual(rt.main(args), 0)
            return buffer.getvalue()

        out = run([path, "--steps", "0,1000"])
        self.assertIn("iterations=1000", out)
        self.assertIn("AMP/mean_root_height_m", out)
        self.assertIn("0.3000", out)
        self.assertNotIn("Custom/metric", out)            # 默认 match 里没有它
        self.assertIn("Custom/metric", run([path, "--tags"]))
        self.assertIn("Custom/metric", run([path, "--match", "Custom"]))
        self.assertIn("9.0000", run([path, "--all", "--steps", "0,1000"]))


if __name__ == "__main__":
    unittest.main()
