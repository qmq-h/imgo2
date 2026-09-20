"""Tow summary/verdict tests; standard Python, no simulator.

判据要能在没有仿真器的机器上用**真实轨迹**复算，所以判读逻辑在
`scripts/tools/summarize_tow.py`（纯标准库）。这里用合成轨迹覆盖各条判据，
重点是那条容易搞反的语义：**步态纹波大但水平稳定，应当判通过**——
计划要的是「T(t) 进入相对稳定区间」，不是「方差小」。
"""

import importlib.util
import math
from pathlib import Path
import sys
import unittest

RL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RL / "scripts/tools"))
from summarize_tow import summarize_tow  # noqa: E402


def _row(t, cmd, vR, vL, tension, *, dist=1.001, pitch=0.05, z=0.284):
    return {"time_s": t, "user_cmd_mps": cmd, "ref_cmd_mps": cmd,
            "robot_vx_mps": vR, "load_vx_mps": vL,
            "rope_tension_n": tension, "rope_distance_m": dist,
            "robot_x_m": 0.0, "load_x_m": -1.0 + 0.5 * t,
            "robot_z_m": z, "load_z_m": 0.15,
            "body_pitch_rad": pitch, "body_pitch_rate_radps": 0.0}


def _run(*, command=0.5, dt=0.005, settle_s=1.0, command_s=5.0, takeup_s=0.1,
         tension=lambda t: 5.0, robot_vx=lambda t: 0.5, load_vx=lambda t: 0.5,
         pitch=0.05, settle_load_vx=0.0, z=0.284):
    """合成一条拖动轨迹：站定阶段 + 指令阶段（前 takeup_s 内张力为 0）。"""
    rows, step, t = [], 0, 0.0
    for _ in range(int(round(settle_s / dt))):
        step += 1; t = step * dt
        rows.append(_row(t, 0.0, 0.0, settle_load_vx, 0.0, z=z + 0.05))
    for _ in range(int(round(command_s / dt))):
        step += 1; t = step * dt
        el = t - settle_s
        rows.append(_row(t, command, robot_vx(el), load_vx(el),
                         0.0 if el < takeup_s else tension(el), pitch=pitch, z=z))
    return rows


class VerdictTests(unittest.TestCase):
    def test_successful_tow_is_valid(self):
        summary = summarize_tow(_run(), user_command=0.5)
        self.assertEqual(summary["failures"], [])
        self.assertTrue(summary["valid"])
        self.assertAlmostEqual(summary["steady_robot_vx_mps"], 0.5, places=6)
        self.assertAlmostEqual(summary["steady_load_vx_mps"], 0.5, places=6)
        self.assertAlmostEqual(summary["steady_tension_n"], 5.0, places=6)
        self.assertAlmostEqual(summary["steady_tracking_ratio"], 1.0, places=6)

    def test_gait_ripple_alone_is_not_instability(self):
        """纹波大、水平稳 ⇒ 通过（这条最容易被判据写反）。"""
        summary = summarize_tow(_run(tension=lambda t: 5.0 + 4.0 * math.sin(2 * math.pi * 7.5 * t)),
                                user_command=0.5)
        self.assertGreater(summary["tension_ripple_ratio"], 0.5)      # 纹波很大
        self.assertLess(summary["tension_drift_ratio"], 0.1)          # 但水平没漂
        self.assertEqual(summary["failures"], [])
        self.assertTrue(summary["valid"])

    def test_drifting_tension_fails(self):
        summary = summarize_tow(_run(tension=lambda t: 1.0 + 4.0 * t), user_command=0.5)
        self.assertIn("tension_not_steady", summary["failures"])
        self.assertFalse(summary["valid"])

    def test_stationary_pair_does_not_pass_on_speed_gap_alone(self):
        """两者都静止时 speed_gap 恒为 0：必须靠跟速比把它拦下来（实测踩过）。"""
        summary = summarize_tow(_run(tension=lambda t: 0.0, robot_vx=lambda t: 0.0,
                                     load_vx=lambda t: 0.0), user_command=0.5)
        self.assertLess(abs(summary["steady_speed_gap_mps"]), 0.01)
        self.assertIn("robot_not_tracking_command", summary["failures"])
        self.assertIn("rope_never_taut_during_tow", summary["failures"])
        self.assertFalse(summary["valid"])

    def test_slack_after_takeup_fails(self):
        # 收掉松弛后又反复回到松弛：一半时间 T=0
        summary = summarize_tow(_run(tension=lambda t: 0.0 if int(t / 0.005) % 2 else 5.0),
                                user_command=0.5)
        self.assertIn("rope_not_continuously_taut", summary["failures"])
        self.assertFalse(summary["valid"])

    def test_takeup_slack_alone_is_expected(self):
        """起步收松弛期间的 T=0 是设计预期，不算失败。"""
        summary = summarize_tow(_run(takeup_s=0.5), user_command=0.5)
        self.assertAlmostEqual(summary["time_to_taut_s"], 1.5, places=6)
        self.assertEqual(summary["tension_slack_fraction_after_takeup"], 0.0)
        self.assertEqual(summary["failures"], [])

    def test_load_speed_mismatch_fails(self):
        summary = summarize_tow(_run(load_vx=lambda t: 0.2), user_command=0.5)
        self.assertIn("robot_and_load_speeds_differ", summary["failures"])

    def test_excessive_pitch_fails(self):
        summary = summarize_tow(_run(pitch=0.8), user_command=0.5)
        self.assertIn("body_pitch_excessive", summary["failures"])

    def test_settle_drift_is_reported(self):
        """站定阶段小车的自漂要作为一等数字报出来（实测 0.239 m）。"""
        summary = summarize_tow(_run(settle_load_vx=0.3), user_command=0.5)
        self.assertGreater(abs(summary["settle_load_drift_m"]), 0.2)
        self.assertAlmostEqual(summary["settle_max_abs_load_vx_mps"], 0.3, places=6)

    def test_rejects_empty_or_bad_input(self):
        with self.assertRaises(ValueError):
            summarize_tow([], user_command=0.5)
        with self.assertRaises(ValueError):
            summarize_tow(_run(), user_command=0.0)
        with self.assertRaises(ValueError):
            summarize_tow(_run(), user_command=float("nan"))
        with self.assertRaises(ValueError):
            summarize_tow([_row(0.005, 0.0, 0.0, 0.0, 0.0)], user_command=0.5)  # 没有指令阶段


if __name__ == "__main__":
    unittest.main()
