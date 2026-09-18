"""`gait_metrics.py` 的离线回归（纯 numpy，不需要 GPU / Isaac Lab / torch）。

Run: python -m unittest discover -s tests -p test_gait_metrics.py -v
（在 `imgo2_rl/` 目录下执行；本机可用 `python3`。）

锁定的都是「算错了会得出错误结论」的地方：
  * `summarize_contact` 的逐环境聚合顺序 —— 写成 `c[:, e].mean()` 会得到 k/E 的离散值，
    看起来像「有些环境的足整段不落地」；
  * 相位估计 —— 只取第一个落地事件/只取 0 号环境会被单次抖动带偏；
  * 游程切分必须覆盖全时长（sum(air)+sum(stance) == T*dt）；
  * 姿态欧拉角与「恒定倾斜」判定。
"""

import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/tools"))
from gait_metrics import (  # noqa: E402
    body_attitude_summary,
    build_report,
    circular_phase,
    landing_events,
    peak_to_peak_per_env,
    per_env_mean,
    run_lengths,
    summarize_contact,
)


def _contact_from_air_fraction(fractions, T):
    """按给定的空中占比造 [E,T] 的 0/1 接触序列（空中在前、支撑在后，便于手算）。"""
    rows = []
    for frac in fractions:
        n_air = int(round(frac * T))
        rows.append([0] * n_air + [1] * (T - n_air))
    return np.array(rows, dtype=np.int8)


class ContactAggregationTests(unittest.TestCase):
    def test_per_env_air_fraction_matches_time_axis_not_env_axis(self):
        """逐环境空中占比必须是对**时间轴**求均值，均值要与全样本自洽。"""
        T = 40
        fractions = [1.0, 0.0, 0.5]          # 一个整段空中、一个整段支撑、一个各半
        contact = _contact_from_air_fraction(fractions, T)[:, :, None]   # [E,T,F=1]
        per_foot, _ = summarize_contact(contact, 0.02, ["FL_FOOT"])
        m = per_foot["FL_FOOT"]
        self.assertEqual(len(m["contact_air_fraction_per_env"]), 3)
        for got, want in zip(m["contact_air_fraction_per_env"], fractions):
            self.assertAlmostEqual(got, want, places=12)
        self.assertAlmostEqual(m["duty_factor_per_env"][0], 0.0, places=12)
        self.assertAlmostEqual(m["duty_factor_per_env"][1], 1.0, places=12)
        # 聚合顺序自洽：逐环境均值 == 全样本时间加权值
        self.assertAlmostEqual(np.mean(m["contact_air_fraction_per_env"]), m["air_fraction"], places=12)
        self.assertAlmostEqual(m["air_fraction"] + m["duty_factor"], 1.0, places=12)
        # 旧写法 c[:, 0].mean() 得到的是「t=0 时刻各环境的均值」= 1/3，与 0.5 不同
        wrong = float(1.0 - contact[:, 0, 0].mean())
        self.assertNotAlmostEqual(wrong, m["air_fraction"], places=3)

    def test_landing_count_and_frequency(self):
        T = 100
        c = np.zeros((1, T, 1), dtype=np.int8)
        c[0, ::10, 0] = 1                     # 每 10 步落地一次 → 每 100 步 9 个上升沿
        per_foot, _ = summarize_contact(c, 0.02, ["FL_FOOT"])
        m = per_foot["FL_FOOT"]
        self.assertEqual(m["landings_per_env"], [9])
        self.assertAlmostEqual(m["stride_freq_hz"], 9 / (T * 0.02), places=12)
        self.assertAlmostEqual(m["step_period_s"], T * 0.02 / 9, places=12)

    def test_run_lengths_cover_full_duration(self):
        seq = np.array([0, 0, 1, 1, 1, 0, 1, 0, 0], dtype=np.int8)
        air, stance = run_lengths(seq, 0.5)
        self.assertAlmostEqual(sum(air) + sum(stance), len(seq) * 0.5, places=12)
        # 游程：air 2 步 / stance 3 步 / air 1 步 / stance 1 步 / air 2 步
        self.assertEqual(sorted(air), [0.5, 1.0, 1.0])
        self.assertEqual(sorted(stance), [0.5, 1.5])

    def test_landing_events_ignores_initial_contact(self):
        # t=0 就在接触不算「落地」；只有 0→1 才算
        self.assertEqual(list(landing_events([1, 1, 0, 1, 1, 0, 0, 1])), [3, 7])

    def test_gait_period_from_reference_foot(self):
        T = 130
        c = np.zeros((2, T, 1), dtype=np.int8)
        for e in range(2):
            c[e, ::13, 0] = 1                 # 周期 13 步 = 0.26 s
        per_foot, extra = summarize_contact(c, 0.02, ["FL_FOOT"])
        self.assertEqual(extra["gait_period_n_envs"], 2)
        self.assertAlmostEqual(extra["gait_period_s"], 0.26, places=12)


class PhaseTests(unittest.TestCase):
    def test_half_period_shift_is_recovered(self):
        ref = [10.0, 25.0, 40.0]              # 周期 15
        phase, concentration = circular_phase([17.5, 32.5], ref, 15.0)
        self.assertAlmostEqual(phase, 0.5, places=12)
        self.assertAlmostEqual(concentration, 1.0, places=12)

    def test_in_phase_is_zero(self):
        phase, concentration = circular_phase([25.0, 40.0], [10.0, 25.0, 40.0], 15.0)
        self.assertAlmostEqual(phase, 0.0, places=12)
        self.assertAlmostEqual(concentration, 1.0, places=12)

    def test_single_event_does_not_decide_the_phase(self):
        """参考足 3 个周期、目标足带一个抖动事件时，相位不应被那一个事件带偏。"""
        ref = [10.0, 25.0, 40.0]
        clean, _ = circular_phase([17.5, 32.5, 47.5], ref, 15.0)
        dirty, _ = circular_phase([17.5, 32.5, 47.5, 41.0], ref, 15.0)   # 多余的脏事件
        self.assertAlmostEqual(clean, 0.5, places=6)
        self.assertLess(abs(dirty - clean), 0.08)

    def test_missing_samples_return_none(self):
        self.assertEqual(circular_phase([], [1.0, 5.0], 4.0), (None, None))
        self.assertEqual(circular_phase([1.0], [1.0], 4.0), (None, None))
        self.assertEqual(circular_phase([1.0], [1.0, 5.0], 0.0), (None, None))


class StrideAndAttitudeTests(unittest.TestCase):
    def test_peak_to_peak_per_env(self):
        v = np.array([[0.0, 1.0], [0.5, 1.5], [2.0, 3.0]])   # [T,E]
        mean_v, max_v, per_env = peak_to_peak_per_env(v)
        self.assertAlmostEqual(per_env[0], 2.0, places=12)
        self.assertAlmostEqual(per_env[1], 2.0, places=12)
        self.assertAlmostEqual(mean_v, 2.0, places=12)
        self.assertAlmostEqual(max_v, 2.0, places=12)

    def test_per_env_mean(self):
        v = np.array([[0.0, 2.0], [1.0, 4.0]])                # [T,E]
        self.assertEqual(per_env_mean(v), [0.5, 3.0])

    def test_constant_pitch_and_roll(self):
        pitch_deg, roll_deg = 3.0, -1.0
        # R = Ry(pitch) @ Rx(roll) 的 wxyz 形式
        hp, hr = math.radians(pitch_deg) / 2, math.radians(roll_deg) / 2
        q = np.array([[math.cos(hp) * math.cos(hr),
                       math.cos(hp) * math.sin(hr),
                       math.sin(hp) * math.cos(hr),
                       -math.sin(hp) * math.sin(hr)]])
        quat = np.repeat(q[None, :, :], 50, axis=0)           # [T=50, E=1, 4]
        s = body_attitude_summary(quat, lean_threshold_deg=1.0)
        self.assertAlmostEqual(s["body_pitch_deg_mean"], pitch_deg, places=6)
        self.assertAlmostEqual(s["body_roll_deg_mean"], roll_deg, places=6)
        self.assertAlmostEqual(s["body_pitch_deg_rms"], 0.0, places=6)
        self.assertTrue(s["body_pitch_is_constant_lean"])

    def test_small_lean_is_not_flagged(self):
        """各环境一致但幅度极小的倾斜不该被判成「恒定倾斜」。"""
        q = np.array([1.0, 0.0, math.sin(math.radians(0.2) / 2), 0.0])
        quat = np.repeat(q[None, None, :], 20, axis=0)        # [T=20, E=1, 4]
        s = body_attitude_summary(quat, lean_threshold_deg=1.0)
        self.assertFalse(s["body_pitch_is_constant_lean"])

    def test_bad_shapes_raise(self):
        with self.assertRaises(ValueError):
            summarize_contact(np.zeros((2, 5)), 0.02, ["FL_FOOT"])
        with self.assertRaises(ValueError):
            body_attitude_summary(np.zeros((5, 2, 3)))

    def test_yaw_drift_rate_distinguishes_drift_from_oscillation(self):
        T = 101
        # 单调漂移 10°/s：0.02 s 步长、100 步 = 2 s ⇒ 首末差 20°
        rate = math.radians(10.0) * 0.02 * np.arange(T)
        q_drift = np.zeros((T, 1, 4))
        q_drift[:, 0, 0], q_drift[:, 0, 3] = np.cos(rate / 2), np.sin(rate / 2)
        s = body_attitude_summary(q_drift, dt=0.02)
        self.assertAlmostEqual(s["body_yaw_drift_rate_deg_s"], 10.0, places=6)
        self.assertAlmostEqual(s["body_yaw_drift_deg"], 20.0, places=6)
        # 来回摆（净位移为 0）峰峰值很大但漂移率≈0
        swing = math.radians(30.0) * np.sin(np.linspace(0, 4 * np.pi, T))
        q_swing = np.zeros((T, 1, 4))
        q_swing[:, 0, 0], q_swing[:, 0, 3] = np.cos(swing / 2), np.sin(swing / 2)
        s2 = body_attitude_summary(q_swing, dt=0.02)
        self.assertGreater(s2["body_yaw_drift_deg"], 50.0)
        self.assertLess(abs(s2["body_yaw_drift_rate_deg_s"]), 1.0)


class BuildReportTests(unittest.TestCase):
    """端到端跑一遍报告汇总（回放采样→统计），用合成 trot 信号。"""

    # 合成参数：周期 20 步 = 0.4 s，支撑 12 步 = 占空比 0.6，对角步态
    T, E, PERIOD, DUTY, SHIFT = 300, 3, 20, 12, [0, 10, 10, 0]
    FEET = ["FL_FOOT", "FR_FOOT", "RL_FOOT", "RR_FOOT"]

    def _inputs(self, warmup=20):
        contact = np.zeros((self.E, self.T, 4), dtype=np.int8)
        foot_pos = np.zeros((self.T, self.E, 4, 3))
        for e in range(self.E):
            for f, shift in enumerate(self.SHIFT):
                phase = (np.arange(self.T) - shift) % self.PERIOD
                contact[e, phase < self.DUTY, f] = 1
                foot_pos[:, e, f, 0] = 0.3 * phase / self.PERIOD          # 摆动/支撑全程推进
                foot_pos[:, e, f, 2] = 0.1 * (phase >= self.DUTY)         # 摆动期抬起
        quat = np.zeros((self.T, self.E, 4))
        hp = math.radians(3.0) / 2
        quat[:, :, 0], quat[:, :, 2] = math.cos(hp), math.sin(hp)         # 恒定 pitch +3°
        joint_pos = np.zeros((self.T, self.E, 12))
        for i in range(12):
            joint_pos[:, :, i] = (i + 1) * 0.01 * np.sin(2 * np.pi * np.arange(self.T)[:, None] / 25)
        ep_len = np.tile(np.arange(self.T)[:, None], (1, self.E))
        done = np.zeros((self.T, self.E), dtype=np.int8)
        return dict(contact=contact, base_z=np.full((self.T, self.E), 0.31),
                    vel_err_xy=np.full((self.T, self.E), 0.1),
                    vel_err_yaw=np.full((self.T, self.E), 0.05), quat=quat,
                    foot_pos_b=foot_pos, joint_pos=joint_pos, ep_len=ep_len, done=done,
                    dt=0.02, foot_names=self.FEET, checkpoint="synthetic.pt",
                    iteration=1234, warmup=warmup)

    def test_report_keys_and_trot_phase(self):
        r = build_report(**self._inputs())
        # warmup 被切掉
        self.assertEqual(r["steps"], self.T - 20)
        self.assertEqual(r["steps_recorded"], self.T)
        self.assertEqual(r["warmup_steps"], 20)
        self.assertEqual(r["num_envs"], self.E)
        self.assertEqual(r["iter"], 1234)
        # 周期/步频/占空比（步频按窗口内的落地计数，窗口左边界会丢掉一次上升沿）
        self.assertAlmostEqual(r["gait_period_s"], self.PERIOD * 0.02, places=9)
        self.assertAlmostEqual(r["per_foot"]["FL_FOOT"]["stride_freq_hz"],
                               1.0 / (self.PERIOD * 0.02), delta=0.2)
        self.assertAlmostEqual(r["per_foot"]["FL_FOOT"]["duty_factor"],
                               self.DUTY / self.PERIOD, places=6)
        # 对角步态：FR/RL 与 FL 差 180°，RR 与 FL 同相；集中度接近 1
        self.assertAlmostEqual(r["per_foot"]["FR_FOOT"]["phase_deg_vs_FL"], 180.0, delta=5.0)
        self.assertAlmostEqual(r["per_foot"]["RL_FOOT"]["phase_deg_vs_FL"], 180.0, delta=5.0)
        self.assertLess(abs(r["per_foot"]["RR_FOOT"]["phase_deg_vs_FL"]), 5.0)
        for name in self.FEET[1:]:
            self.assertGreater(r["per_foot"][name]["phase_concentration"], 0.99)
        # 行程/抬脚（base 系峰峰）
        self.assertAlmostEqual(r["per_foot"]["FL_FOOT"]["foot_stride_x_m"], 0.3 - 0.3 / self.PERIOD, places=6)
        self.assertAlmostEqual(r["per_foot"]["FL_FOOT"]["foot_lift_z_m"], 0.1, places=6)
        # 姿态与关节键
        self.assertAlmostEqual(r["body_pitch_deg_mean"], 3.0, places=6)
        self.assertTrue(r["body_pitch_is_constant_lean"])
        self.assertEqual(sorted(r["joint_peak_to_peak_rad"], key=lambda k: int(k[3:])),
                         [f"leg{i}" for i in range(12)])
        self.assertAlmostEqual(r["joint_peak_to_peak_rad"]["leg11"], 2 * 0.12, places=3)
        # 没有重置：末步缓冲值应等于记录的步数
        self.assertEqual(r["resets_total"], 0)
        self.assertEqual(r["episode_length_final_steps_per_env"], [self.T - 1] * self.E)

    def test_resets_are_reported(self):
        kw = self._inputs()
        kw["done"][100, 1] = 1
        kw["ep_len"][100:, 1] = np.arange(self.T - 100)   # 该环境在第 100 步被重置
        r = build_report(**kw)
        self.assertEqual(r["resets_total"], 1)
        self.assertEqual(r["envs_with_reset"], 1)
        self.assertLess(r["episode_length_final_steps_per_env"][1],
                        r["episode_length_final_steps_per_env"][0])

    def test_env_stuck_in_air_is_reported_consistently(self):
        kw = self._inputs()
        kw["contact"][0, :, 0] = 0                        # 0 号环境的 FL 整段不落地
        r = build_report(**kw)
        m = r["per_foot"]["FL_FOOT"]
        self.assertAlmostEqual(m["contact_air_fraction_per_env"][0], 1.0, places=12)
        self.assertAlmostEqual(np.mean(m["contact_air_fraction_per_env"]), m["air_fraction"], places=12)

    def test_dump_timeseries_length_matches_steps(self):
        r = build_report(dump_timeseries=True, **self._inputs())
        self.assertEqual(len(r["timeseries"]["t"]), r["steps"])
        self.assertEqual(len(r["timeseries"]["base_z"]), r["steps"])
        self.assertEqual(len(r["timeseries"]["contact"][0]), r["steps"])


if __name__ == "__main__":
    unittest.main()
