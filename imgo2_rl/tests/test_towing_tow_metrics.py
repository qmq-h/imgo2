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

import importlib.util
import math
from pathlib import Path
import sys
import unittest

RL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RL / "scripts/tools"))
from summarize_tow import summarize_tow  # noqa: E402


def _module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


recording = _module_at("towing_recording_metrics_test",
                       RL / "source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/utils/recording.py")

JOINT_NAMES = ["FL_hip_joint", "FL_thigh_joint", "FL_shank_joint",
               "FR_hip_joint", "FR_thigh_joint", "FR_shank_joint",
               "RL_hip_joint", "RL_thigh_joint", "RL_shank_joint",
               "RR_hip_joint", "RR_thigh_joint", "RR_shank_joint"]


def add_witness_columns(rows, *, joints=(0.0, 0.87, -1.82), robot_z=0.2735, deck_fx=0.0,
                        load_z=0.15, deck_fx_station=0.0):
    """给合成轨迹补上「接触力 + 姿态 + 关节角」三组见证列（模拟新记录的字段）。

    `deck_fx` 只作用在拖曳之后（真实的追尾发生在拖曳开始之后）；`deck_fx_station`
    模拟「生成时就贴上」，那是另一条判据（`load_touching_at_settle`）。
    """
    for row in rows:
        row.update(recording.joint_position_fields(list(joints) * 4))
        row["robot_z_m"] = robot_z
        row["load_z_m"] = load_z
        row.update({"robot_quat_x": 0.0, "robot_quat_y": 0.0, "robot_quat_z": 0.0,
                    "robot_quat_w": 1.0, "load_quat_x": 0.0, "load_quat_y": 0.0,
                    "load_quat_z": 0.0, "load_quat_w": 1.0,
                    "cart_deck_fx_n": deck_fx_station if row["phase"] == "station" else deck_fx,
                    "cart_wheel_fx_n": 0.0})
        # 绳索模型的统一日志列（两套模型同一套字段）：伸长与冲量都由张力推得
        tension = row["rope_tension_n"]
        row.update({"rope_extension_m": tension / 4000.0, "rope_length_rate_mps": 0.0,
                    "rope_taut": 1.0 if tension > 0 else 0.0,
                    "rope_impulse_ns": tension * 0.005})
    return rows


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

    def test_load_moving_during_settle_fails(self):
        """拖曳开始前小车必须静止在设计位置——不再用代码摆正，改为检查。"""
        ok = summarize_tow(_run(settle_load_vx=0.0), user_command=0.5)
        self.assertNotIn("load_moved_during_settle", ok["failures"])
        # 复现实测：重置把小车挪了 0.081 m 并注入速度
        bad = summarize_tow(_run(settle_load_vx=0.08), user_command=0.5)
        self.assertGreater(abs(bad["settle_load_drift_m"]), 0.02)
        self.assertIn("load_moved_during_settle", bad["failures"])
        self.assertFalse(bad["valid"])

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

    def test_load_reaching_robot_is_reported_and_not_a_failure(self):
        """撞上是这套设置允许出现的正常结果（用户 2026-09-21），所以只报告、不判失败。

        判据也不能用挂点间距：这里让撞击真的发生（滑行段负载单步速度跃变），
        再看它是否被正确识别成 `reached_robot`，并且 `valid` 仍然为真。
        """
        def load_vx(t):
            return 0.5 if t < 1.0 else 0.5 * math.exp(-(t - 1.0) / 0.1)
        summary = self._coast(coast_gap=lambda t: max(0.0, 1.0 - 0.3 * t),
                              coast_load_vx=load_vx)
        self.assertEqual(summary["coast_min_gap_m"], 0.0)
        self.assertTrue(summary["reached_robot"])
        self.assertIn("load_velocity_jump", summary["reached_robot_source"])
        self.assertGreater(summary["max_load_dv_mps"], 0.015)
        self.assertNotIn("load_reached_robot", summary["failures"])
        self.assertTrue(summary["valid"], summary["failures"])

    def test_old_record_without_witness_channels_is_marked_unreliable(self):
        """旧记录既无接触力也无关节角 ⇒ 只能退回挂点间距，来源名必须点明这一点。"""
        summary = self._coast(coast_gap=lambda t: max(0.0, 1.0 - 0.3 * t))
        self.assertIsNone(summary["min_clearance_m"])
        self.assertIsNone(summary["deck_contact_peak_n"])
        self.assertTrue(summary["reached_robot"])                      # 挂点口径兜底
        self.assertEqual(summary["reached_robot_source"], "attachment_gap_fallback")
        self.assertTrue(summary["valid"], summary["failures"])

    def test_no_witness_and_no_catch_up_says_clear(self):
        summary = self._coast()
        self.assertFalse(summary["reached_robot"])
        self.assertEqual(summary["reached_robot_source"], "attachment_gap_fallback_clear")

    def test_deck_contact_force_is_a_witness(self):
        """车斗永远不碰地面，所以它的接触力非零只可能是机器人压上来。"""
        rows = add_witness_columns(_run(coast_s=1.0), deck_fx=12.0)
        summary = summarize_tow(rows, user_command=0.5, joint_names=JOINT_NAMES)
        self.assertAlmostEqual(summary["deck_contact_peak_n"], 12.0, places=6)
        self.assertIn("deck_contact_force", summary["reached_robot_source"])
        self.assertTrue(summary["reached_robot"])
        self.assertTrue(summary["valid"], summary["failures"])

    def test_deck_contact_at_settle_is_a_failure(self):
        """生成时就贴上机器人（station 段车斗受力）必须单独判失败。

        依据：挂点间距 0.40 m 听着安全，实测车头间隙只有 0.0955 m（§5.19），
        所以「贴上了」这件事不能等到「追尾」那条判据里去发现。
        """
        rows = add_witness_columns(_run(coast_s=0.5), deck_fx_station=9.0)
        summary = summarize_tow(rows, user_command=0.5, joint_names=JOINT_NAMES)
        self.assertAlmostEqual(summary["deck_contact_peak_station_n"], 9.0, places=6)
        self.assertIn("load_touching_at_settle", summary["failures"])
        self.assertFalse(summary["valid"])

    def test_rope_impulse_and_extension_stats(self):
        """规格要求的离线统计：J = ∫T dt、（峰值/稳态）伸长、张力变化率。"""
        rows = add_witness_columns(_run(tow_s=1.0, takeup_s=0.01, tension=lambda t: 10.0))
        summary = summarize_tow(rows, user_command=0.5, joint_names=JOINT_NAMES)
        # 稳态 10 N、1 s、dt=5 ms ⇒ ∫T dt ≈ 10 × 1.0 = 10 N·s（station/coast 段张力为 0）
        self.assertAlmostEqual(summary["rope_impulse_tow_ns"], 10.0, delta=0.2)
        self.assertAlmostEqual(summary["rope_impulse_total_ns"], 10.0, delta=0.2)
        self.assertAlmostEqual(summary["rope_impulse_engagement_ns"], 2.0, delta=0.2)  # 0.2 s 窗口
        # 稳态窗口取「tow 段最后 1 s」，本用例 tow 段正好 1 s ⇒ 含首个张力为 0 的样本
        self.assertAlmostEqual(summary["rope_steady_extension_mm"], 10.0 / 4000.0 * 1000.0, delta=0.05)
        self.assertAlmostEqual(summary["rope_peak_extension_mm"], 10.0 / 4000.0 * 1000.0, delta=0.05)
        self.assertAlmostEqual(summary["rope_taut_fraction"], 0.5, delta=0.05)  # 1 s tow / 2 s 总长
        # 张力变化率峰值 = 绷直那一步的跳变 10 N / 5 ms = 2000 N/s（这正是要看的变化率）
        self.assertAlmostEqual(summary["max_tension_rate_n_per_s"], 2000.0, delta=50.0)

    def test_stop_transient_metrics_measure_the_reaction_window(self):
        """停止瞬态：`clearance_at_stop_m` / `time_to_contact_after_stop_s` / `load_vx_at_contact_mps`。

        这三个量是训练目标（收到停止指令后机器人往前几步防追尾）需要的：**时间窗**就是
        「往前几步」必须在多久之内把间隙重新拉开；`load_vx_at_contact_mps` 是撞击速度的实测值。
        """
        rows = add_witness_columns(_run(coast_s=1.0))
        # 让负载在 coast 段持续靠近：位置从 x_R−x_L=1.0084（间隙 +0.36）线性收到 0.6084（间隙 −0.04）
        for index, row in enumerate(rows):
            if row["phase"] != "coast":
                continue
            fraction = index / max(1, len(rows) - 1)
            row["load_x_m"], row["robot_x_m"] = 0.0, 1.0084 - 0.40 * fraction
        summary = summarize_tow(rows, user_command=0.5, joint_names=JOINT_NAMES)
        self.assertGreater(summary["clearance_at_stop_m"], 0.0)         # 指令归零时还没接触
        self.assertIsNotNone(summary["time_to_contact_after_stop_s"])
        self.assertGreater(summary["time_to_contact_after_stop_s"], 0.0)
        self.assertIsNotNone(summary["load_vx_at_contact_mps"])

    def test_stop_transient_is_none_when_it_never_touches(self):
        rows = add_witness_columns(_run(coast_s=0.5))
        for row in rows:                     # 保持大间距：不会接触
            row["load_x_m"], row["robot_x_m"] = 0.0, 1.5
        summary = summarize_tow(rows, user_command=0.5, joint_names=JOINT_NAMES)
        self.assertIsNone(summary["time_to_contact_after_stop_s"])
        self.assertIsNone(summary["load_vx_at_contact_mps"])
        self.assertIsNotNone(summary["clearance_at_stop_m"])            # 但起始间隙仍要报

    def test_rope_stats_are_absent_on_old_records(self):
        """旧 run 没有绳索模型列 ⇒ 这些统计必须是 None，而不是被当成 0。"""
        rows = _run(coast_s=0.5)
        summary = summarize_tow(rows, user_command=0.5)
        self.assertIsNone(summary["rope_impulse_total_ns"])
        self.assertIsNone(summary["rope_peak_extension_mm"])

    def test_station_clearance_is_reported_separately(self):
        """station 段的车头间隙单独报（它是初始条件的安全余量），不计入 reached_robot。"""
        rows = add_witness_columns(_run(coast_s=0.5))
        for row in rows:                    # 全程 x_R − x_L = 0.6084 ⇒ 间隙 −0.04 m
            row["load_x_m"], row["robot_x_m"] = 0.0, 0.6084
        summary = summarize_tow(rows, user_command=0.5, joint_names=JOINT_NAMES)
        self.assertLess(summary["min_clearance_station_m"], 0.0)
        self.assertLess(summary["min_clearance_m"], 0.0)

    def test_geometric_clearance_witness_fires_when_the_pose_reaches(self):
        """几何间隙 ≤ 0 ⇒ 接触。判据用的是 FK 算出的真实间隙，不是挂点间距。

        默认站姿、直立姿态下 间隙 = (x_R − x_L) − 0.3984 − 0.25，所以把两体摆到
        x_R − x_L = 0.6084（即间隙 −0.04 m）应当判为已经接触。
        """
        rows = add_witness_columns(_run(coast_s=1.0))
        for row in rows:                     # 间隙由**位置**算出，与 rope_distance_m 无关
            row["load_x_m"], row["robot_x_m"] = 0.0, 0.6084
        summary = summarize_tow(rows, user_command=0.5, joint_names=JOINT_NAMES)
        self.assertLess(summary["min_clearance_m"], 0.0)
        self.assertIn("geometric_clearance", summary["reached_robot_source"])
        self.assertTrue(summary["reached_robot"])
        self.assertTrue(summary["valid"], summary["failures"])

    def test_geometric_clearance_says_clear_when_it_does_not_reach(self):
        """x_R − x_L = 1.0084 ⇒ 间隙 +0.36 m，未接触；来源必须是 witnesses_clear。"""
        rows = add_witness_columns(_run(coast_s=1.0))
        for row in rows:
            row["load_x_m"], row["robot_x_m"] = 0.0, 1.0084
        summary = summarize_tow(rows, user_command=0.5, joint_names=JOINT_NAMES)
        self.assertGreater(summary["min_clearance_m"], 0.3)
        self.assertFalse(summary["reached_robot"])
        self.assertEqual(summary["reached_robot_source"], "witnesses_clear")

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


class ElasticityTests(unittest.TestCase):
    """绳的弹性诊断（§5.18）：伸长、绷直过冲、固有频率/阻尼比与步长上限。

    关键点是**这些量只用配置常量 + 记录里的张力**，所以已有 run 不重跑也能复算；
    这里用合成轨迹 + 一份最小 config 把算式钉住。
    """

    CONFIG = {
        "rope": {"rest_length_m": 0.8, "stiffness_n_per_m": 4000.0,
                 "damping_ns_per_m": 100.0, "initial_slack_m": 0.4},
        "cart_model": {"wheel_radius_m": 0.08, "inertias_kgm2": {"wheel_fl": [0.00069, 0.00128, 0.00069]},
                       "total_mass_kg": 10.0},
        "cart_mass_actual_kg": 10.0,
        "robot_mass_kg": 12.6996,
    }

    def _summary(self, config):
        # tow 段：张力先冲到 70 N（绷直过冲）再落到稳态 10 N
        def tension(t):
            return 70.0 if t < 0.05 else 10.0
        rows = _run(tow_s=1.0, takeup_s=0.01, tension=tension)
        return summarize_tow(rows, user_command=0.5, config=config)

    def test_stretch_and_overshoot(self):
        summary = self._summary(self.CONFIG)
        self.assertAlmostEqual(summary["steady_tension_n"], 10.0, places=6)
        self.assertAlmostEqual(summary["takeup_peak_tension_n"], 70.0, places=6)
        self.assertAlmostEqual(summary["tension_overshoot_ratio"], 7.0, places=6)
        # 伸长 δ = T/k：稳态 10/4000 = 2.5 mm、峰值 70/4000 = 17.5 mm
        self.assertAlmostEqual(summary["rope_stretch_steady_mm"], 2.5, places=6)
        self.assertAlmostEqual(summary["rope_stretch_peak_mm"], 17.5, places=6)

    def test_reduced_mass_uses_total_robot_mass_not_the_base_link(self):
        """折合质量必须用**整机**质量：base 单链只有 5.5339 kg，整机 12.6996 kg。

        用错会让 ω、ζ、步长上限全偏（手算时踩过，差约 1.6 倍）。
        """
        summary = self._summary(self.CONFIG)
        mass_eff = 10.0 + 4 * 0.00128 / 0.08 ** 2          # 10.8 kg
        expected = 1.0 / (1.0 / mass_eff + 1.0 / 12.6996)  # 5.8365 kg
        self.assertAlmostEqual(summary["rope_reduced_mass_kg"], expected, places=6)
        self.assertNotAlmostEqual(summary["rope_reduced_mass_kg"],
                                  1.0 / (1.0 / mass_eff + 1.0 / 5.5339), places=2)
        omega = math.sqrt(4000.0 / expected)
        self.assertAlmostEqual(summary["rope_natural_freq_hz"], omega / (2 * math.pi), places=6)
        zeta = 100.0 / (2 * math.sqrt(4000.0 * expected))
        self.assertAlmostEqual(summary["rope_damping_ratio"], zeta, places=6)
        # 显式弹簧步长上限 dt < 2/(ω(ζ+√(ζ²+1)))
        self.assertAlmostEqual(summary["spring_dt_limit_ms"],
                               2.0 / (omega * (zeta + math.sqrt(zeta ** 2 + 1))) * 1000.0, places=6)
        self.assertGreater(summary["spring_dt_limit_ms"], 5.0)   # 当前 dt=5 ms 有余量

    def test_stiffer_rope_shrinks_the_dt_limit(self):
        """k 提到真实绳量级时步长上限会掉到 5 ms 以下——这是不能随便加刚度的硬约束。"""
        summary = self._summary(dict(self.CONFIG, rope=dict(self.CONFIG["rope"],
                                                           stiffness_n_per_m=1.6e6)))
        self.assertLess(summary["spring_dt_limit_ms"], 5.0)

    def test_missing_config_still_reports_the_measurable_part(self):
        summary = self._summary(None)
        self.assertEqual(summary["takeup_peak_tension_n"], 70.0)
        self.assertAlmostEqual(summary["tension_overshoot_ratio"], 7.0, places=6)
        self.assertIsNone(summary["rope_stretch_peak_mm"])
        self.assertIn("config", summary["elasticity_note"])

    def test_incomplete_config_names_what_is_missing(self):
        config = {"rope": {"stiffness_n_per_m": 4000.0}, "cart_mass_actual_kg": 10.0}
        summary = self._summary(config)
        self.assertIsNone(summary["rope_natural_freq_hz"])
        self.assertIn("cart_model.wheel_radius_m", summary["elasticity_note"])


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
