"""Tow summary/verdict tests; standard Python, no simulator.

判据要能在没有仿真器的机器上用**真实轨迹**复算，所以判读逻辑在
`scripts/tools/summarize_tow.py`（纯标准库）。这里用合成轨迹覆盖各条判据，
重点是三条容易搞反的语义：

- **步态纹波大但水平稳定，应当判通过**——计划要的是「T(t) 进入相对稳定区间」，
  不是「方差小」；
- **站定阶段绳不得被拉直**——实测踩过：机器人按 0.35 m 出生会向前窜 0.65 m/s，
  把绳拉直（73–86 N）并把小车甩出 0.24 m，看上去就像「小车自己有初速度」；
- **滑行段（阶跃停止）**要能报出小车滑行距离、最小间距（追尾风险）与张力归零耗时。
"""

import math
from pathlib import Path
import sys
import unittest

RL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RL / "scripts/tools"))
from summarize_tow import summarize_tow  # noqa: E402


def _row(t, phase, cmd, vR, vL, tension, *, dist=1.001, pitch=0.05, z=0.284,
         robot_x=0.0, load_x=0.0):
    return {"phase": phase, "time_s": t, "user_cmd_mps": cmd, "ref_cmd_mps": cmd,
            "robot_vx_mps": vR, "load_vx_mps": vL,
            "rope_tension_n": tension, "rope_distance_m": dist,
            "robot_x_m": robot_x, "load_x_m": load_x,
            "robot_z_m": z, "load_z_m": 0.15,
            "body_pitch_rad": pitch, "body_pitch_rate_radps": 0.0}


def _run(*, command=0.5, dt=0.005, settle_s=1.0, tow_s=5.0, coast_s=0.0, takeup_s=0.1,
         tension=lambda t: 5.0, settle_peak_tension=0.0, robot_vx=lambda t: 0.5,
         load_vx=lambda t: 0.5, pitch=0.05, settle_load_vx=0.0, with_phase=True,
         settle_robot_vx=0.0,
         coast_robot_vx=lambda t: 0.0, coast_load_vx=lambda t: 0.5,
         coast_tension=lambda t: 0.0, coast_gap=lambda t: 0.5):
    """合成一条拖动轨迹：station（站定）→ tow（拖曳）→ 可选 coast（指令归零滑行）。"""
    rows, step, t = [], 0, 0.0
    position = {"robot": 0.0, "load": -1.0}

    def add(phase, cmd, vr, vl, ten, elapsed, **kw):
        nonlocal step, t
        step += 1
        t = step * dt
        position["robot"] += vr * dt        # 位置由速度积分，便于核对位移类指标
        position["load"] += vl * dt
        row = _row(t, phase, cmd, vr, vl, ten, robot_x=position["robot"],
                   load_x=position["load"], **kw)
        if not with_phase:
            row.pop("phase")            # 模拟没有 phase 列的旧记录
        rows.append(row)

    for _ in range(int(round(settle_s / dt))):
        # 站定阶段绳本应松弛；settle_peak_tension 用来模拟「被出生窜动拉直」
        add("station", 0.0, settle_robot_vx, settle_load_vx, settle_peak_tension, t, z=0.30)
    for _ in range(int(round(tow_s / dt))):
        el = t - settle_s + dt
        add("tow", command, robot_vx(el), load_vx(el), 0.0 if el < takeup_s else tension(el),
            el, pitch=pitch)
    for _ in range(int(round(coast_s / dt))):
        el = t - settle_s - tow_s + dt
        add("coast", 0.0, coast_robot_vx(el), coast_load_vx(el), coast_tension(el), el,
            dist=coast_gap(el), pitch=pitch)
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
        self.assertEqual(summary["samples"], {"station": 200, "tow": 1000, "coast": 0})

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

    def test_takeup_distance_and_time_are_reported(self):
        """「走了多远才发力」要报出来，便于核对设计值（≈ --slack）。"""
        summary = summarize_tow(_run(takeup_s=1.0, robot_vx=lambda t: 0.5), user_command=0.5)
        self.assertAlmostEqual(summary["takeup_time_s"], 1.0, delta=0.01)
        self.assertAlmostEqual(summary["takeup_robot_travel_m"], 0.5, delta=0.01)

    def test_takeup_is_none_when_rope_never_taut(self):
        summary = summarize_tow(_run(tension=lambda t: 0.0), user_command=0.5)
        self.assertIsNone(summary["takeup_time_s"])
        self.assertIsNone(summary["takeup_robot_travel_m"])

    def test_load_speed_mismatch_fails(self):
        summary = summarize_tow(_run(load_vx=lambda t: 0.2), user_command=0.5)
        self.assertIn("robot_and_load_speeds_differ", summary["failures"])

    def test_excessive_pitch_fails(self):
        summary = summarize_tow(_run(pitch=0.8), user_command=0.5)
        self.assertIn("body_pitch_excessive", summary["failures"])

    def test_settle_drift_is_reported(self):
        summary = summarize_tow(_run(settle_load_vx=0.3), user_command=0.5)
        self.assertGreater(abs(summary["settle_load_drift_m"]), 0.2)
        self.assertAlmostEqual(summary["settle_max_abs_load_vx_mps"], 0.3, places=6)


class StationPhaseTests(unittest.TestCase):
    """站定阶段绳不得被拉直——「小车有初速度」就是从这里来的。"""

    def test_rope_taut_during_settle_fails(self):
        # 实测：机器人按 0.35 m 出生，落地前窜把绳拉直，站定段张力峰值 73–86 N
        summary = summarize_tow(_run(settle_peak_tension=73.0), user_command=0.5)
        self.assertEqual(summary["settle_max_tension_n"], 73.0)
        self.assertIn("rope_taut_during_settle", summary["failures"])
        self.assertFalse(summary["valid"])

    def test_slack_rope_during_settle_is_fine(self):
        summary = summarize_tow(_run(settle_peak_tension=0.0), user_command=0.5)
        self.assertEqual(summary["settle_max_tension_n"], 0.0)
        self.assertNotIn("rope_taut_during_settle", summary["failures"])

    def test_robot_lurching_during_settle_fails(self):
        """出生高度不当会让机器人自己窜出去（实测 0.30 m 出生窜了 0.994 m，把绳拉直后翻倒）。

        0.35 m 出生两次实跑都是 0.044 m，所以阈值 0.15 m 能把两者干净分开。"""
        ok = summarize_tow(_run(settle_robot_vx=0.044), user_command=0.5)
        self.assertNotIn("robot_lurches_during_settle", ok["failures"])
        bad = summarize_tow(_run(settle_robot_vx=1.0), user_command=0.5)
        self.assertGreater(bad["settle_robot_travel_m"], 0.15)
        self.assertIn("robot_lurches_during_settle", bad["failures"])
        self.assertFalse(bad["valid"])

    def test_settle_robot_travel_is_reported(self):
        summary = summarize_tow(_run(settle_robot_vx=0.044), user_command=0.5)
        self.assertAlmostEqual(summary["settle_robot_travel_m"], 0.044, delta=0.005)

    def test_small_settle_tension_below_threshold_is_tolerated(self):
        summary = summarize_tow(_run(settle_peak_tension=0.5), user_command=0.5)
        self.assertNotIn("rope_taut_during_settle", summary["failures"])


class CoastPhaseTests(unittest.TestCase):
    """阶跃停止（计划 P6）：机器人停下、小车继续滑行。"""

    def _coast(self, **kw):
        params = dict(tow_s=5.0, coast_s=5.0,
                      coast_load_vx=lambda t: 0.5 * math.exp(-t / 1.08),
                      coast_tension=lambda t: max(0.0, 5.0 - 20.0 * t),
                      coast_gap=lambda t: 1.0 - 0.5 * 1.08 * (1 - math.exp(-t / 1.08)))
        params.update(kw)
        return summarize_tow(_run(**params), user_command=0.5)

    def test_robot_stops_and_cart_coasts(self):
        summary = self._coast()
        self.assertEqual(summary["samples"]["coast"], 1000)
        self.assertTrue(summary["valid"], summary["failures"])
        # 小车在滑行段继续前进（被拖到机器人停下后靠惯性滑）
        self.assertGreater(summary["coast_load_travel_m"], 0.3)
        # 间距被吃掉（小车追近机器人）
        self.assertLess(summary["coast_min_gap_m"], summary["coast_gap_at_stop_m"])
        # 张力归零有明确耗时，且之后没有重新绷紧
        self.assertIsNotNone(summary["coast_time_to_slack_s"])
        self.assertAlmostEqual(summary["coast_time_to_slack_s"], 0.25, delta=0.01)
        self.assertEqual(summary["coast_retension_peak_n"], 0.0)

    def test_robot_not_stopping_fails(self):
        summary = self._coast(coast_robot_vx=lambda t: 0.4)
        self.assertGreater(summary["coast_final_robot_vx_mps"], 0.1)
        self.assertIn("robot_did_not_stop", summary["failures"])
        self.assertFalse(summary["valid"])

    def test_load_reaching_robot_fails(self):
        summary = self._coast(coast_gap=lambda t: max(0.0, 1.0 - 0.3 * t))
        self.assertEqual(summary["coast_min_gap_m"], 0.0)
        self.assertIn("load_reached_robot", summary["failures"])
        self.assertFalse(summary["valid"])

    def test_retension_after_slack_is_reported_not_failed(self):
        """追尾后绳重新绷紧是计划要观察的现象，报告而不判失败。"""
        summary = self._coast(coast_gap=lambda t: 0.4 if t < 1.0 else 1.2,
                              coast_tension=lambda t: 0.0 if t < 1.0 else 30.0)
        self.assertEqual(summary["coast_retension_peak_n"], 30.0)
        self.assertNotIn("load_reached_robot", summary["failures"])
        self.assertTrue(summary["valid"], summary["failures"])

    def test_no_coast_phase_leaves_coast_fields_absent(self):
        summary = summarize_tow(_run(coast_s=0.0), user_command=0.5)
        self.assertEqual(summary["samples"]["coast"], 0)
        self.assertNotIn("coast_min_gap_m", summary)


class BackwardCompatibilityTests(unittest.TestCase):
    def test_rows_without_phase_column_are_segmented_by_command(self):
        """旧记录（无 phase 列）也要能复算——阶段按指令推断。"""
        rows = _run(tow_s=2.0, coast_s=1.0, with_phase=False)
        self.assertNotIn("phase", rows[0])
        summary = summarize_tow(rows, user_command=0.5)
        self.assertEqual(summary["samples"], {"station": 200, "tow": 400, "coast": 200})
        self.assertTrue(summary["valid"], summary["failures"])

    def test_rejects_empty_or_bad_input(self):
        with self.assertRaises(ValueError):
            summarize_tow([], user_command=0.5)
        with self.assertRaises(ValueError):
            summarize_tow(_run(), user_command=0.0)
        with self.assertRaises(ValueError):
            summarize_tow(_run(), user_command=float("nan"))
        with self.assertRaises(ValueError):
            summarize_tow([_row(0.005, "station", 0.0, 0.0, 0.0, 0.0)], user_command=0.5)


if __name__ == "__main__":
    unittest.main()
