"""`towing_task_criteria.py` 的离线测试（纯标准库，不需要 torch / Isaac Sim）。

覆盖 2026-10-19 收束口径的三条检查、严重度、质量分档、汇总与旧报告复判。
真实 `play_towing_test.py` 的集成（`metrics["task"]` 落盘、report.md/json 字段）由
`test_towing_play_test.py` 的 `TaskCriteriaIntegrationTests` 覆盖。
"""

from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

RL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RL / "scripts/towing"))


def _module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


task = _module_at("towing_task_criteria_under_test",
                  RL / "scripts/towing/towing_task_criteria.py")


def metrics(*, mass=10.0, velocity=1.0, cart_present=True, speed=None, impact=None,
            stability=None, invalid=False, lane=None, startup=None, stop=None):
    return {
        "case": {"cart_mass_kg": mass, "velocity_mps": velocity,
                 "grade_deg": 0.0, "connection": "compliant", "env_index": 0},
        "cart_present": cart_present,
        "speed": speed if speed is not None else {
            "available": True, "mae_mps": 0.02, "steady_mae_mps": 0.02,
            "steady_speed_std_mps": 0.01, "steady_abs_err_p95_mps": 0.03,
        },
        # 2026-10-19 第二轮恢复为失败项：横向保持 / 关节静差默认给「通过」的读数，
        # 免得每个用例都被 JNT 掩盖；专门测这两项时再覆盖。
        "lane": lane if lane is not None else {
            "available": True, "mode": "pd", "y_max_abs_m": 0.01,
            "heading_max_abs_rad": 0.01, "y_rms_m": 0.005,
            "heading_rms_rad": 0.005},
        "startup": startup if startup is not None else {
            "joint_rms_rad": 0.02, "joint_max_rad": 0.05},
        "stop": stop if stop is not None else {
            "joint_rms_rad": 0.02, "joint_max_rad": 0.05},
        "impact": impact if impact is not None else {
            "available": True,
            "startup": {"available": True, "joint_rms_rad": 0.2,
                        "rms_over_steady": 1.0, "rms_delta_rad": 0.0,
                        "torque_saturated_frac": 0.0},
            "takeup": {"available": True, "joint_rms_rad": 0.2,
                       "rms_over_steady": 1.0, "rms_delta_rad": 0.0,
                       "torque_saturated_frac": 0.0},
            "stop": {"available": True, "joint_rms_rad": 0.2,
                     "rms_over_steady": 1.0, "rms_delta_rad": 0.0,
                     "torque_saturated_frac": 0.0},
            "stop_contact": {"available": False},
            "steady": {"available": True, "joint_rms_rad": 0.2,
                       "rms_over_steady": None, "rms_delta_rad": None,
                       "torque_saturated_frac": 0.0},
        },
        "stability": stability if stability is not None else {
            "available": True, "fell": False, "invalid": invalid,
            "max_abs_roll_rad": 0.05, "min_robot_surface_height_m": 0.27,
            "max_abs_pitch_rel_rad": 0.05,
        },
    }


class BandTests(unittest.TestCase):
    def test_light_and_heavy_split_at_15kg(self):
        self.assertEqual(task.mass_band(10.0), "light")
        self.assertEqual(task.mass_band(15.0), "light")
        self.assertEqual(task.mass_band(15.0001), "heavy")
        self.assertEqual(task.mass_band(30.0), "heavy")

    def test_custom_split(self):
        self.assertEqual(task.mass_band(20.0, light_max_kg=25.0), "light")
        self.assertEqual(task.mass_band(26.0, light_max_kg=25.0), "heavy")

    def test_missing_mass_is_none(self):
        self.assertIsNone(task.mass_band(None))
        self.assertIsNone(task.mass_band(float("nan")))

    def test_labels(self):
        self.assertEqual(task.band_label("light"), "小车")
        self.assertEqual(task.band_label("heavy"), "重车")


class SpeedCheckTests(unittest.TestCase):
    def test_good_light_case_passes_all_three_checks(self):
        result = task.evaluate_task(metrics(mass=10.0))
        self.assertTrue(result["applicable"])
        self.assertTrue(result["ok"])
        self.assertEqual(result["code"], "PASS")
        self.assertEqual(result["band"], "light")
        check = result["checks"]["speed_tracking"]
        self.assertTrue(check["applicable"])
        self.assertTrue(check["ok"])

    def test_mae_above_20pct_fails(self):
        result = task.evaluate_task(metrics(speed={
            "available": True, "steady_mae_mps": 0.25,
            "steady_speed_std_mps": 0.01, "steady_abs_err_p95_mps": 0.03}))
        self.assertEqual(result["code"], "SPEED_UNSTABLE")
        self.assertIn("speed_mae_high", result["reasons"])

    def test_std_above_10pct_fails(self):
        result = task.evaluate_task(metrics(speed={
            "available": True, "steady_mae_mps": 0.02,
            "steady_speed_std_mps": 0.15, "steady_abs_err_p95_mps": 0.03}))
        self.assertEqual(result["code"], "SPEED_UNSTABLE")
        self.assertIn("speed_unstable", result["reasons"])

    def test_p95_above_40pct_fails(self):
        result = task.evaluate_task(metrics(speed={
            "available": True, "steady_mae_mps": 0.02,
            "steady_speed_std_mps": 0.01, "steady_abs_err_p95_mps": 0.45}))
        self.assertEqual(result["code"], "SPEED_UNSTABLE")
        self.assertIn("speed_dropout", result["reasons"])

    def test_heavy_cart_does_not_judge_speed(self):
        result = task.evaluate_task(metrics(mass=30.0, speed={
            "available": True, "steady_mae_mps": 9.0,
            "steady_speed_std_mps": 9.0, "steady_abs_err_p95_mps": 9.0}))
        check = result["checks"]["speed_tracking"]
        self.assertFalse(check["applicable"])
        self.assertIsNone(check["ok"])
        self.assertTrue(result["ok"])
        self.assertEqual(result["code"], "PASS")

    def test_old_report_without_steady_fields_falls_back(self):
        result = task.evaluate_task(metrics(speed={
            "available": True, "mae_mps": 0.05, "p95_abs_err_mps": 0.10}))
        check = result["checks"]["speed_tracking"]
        self.assertTrue(check["applicable"])
        self.assertTrue(check["ok"])
        self.assertIsNone(check["values"]["steady_speed_std_ratio"])
        self.assertIsNone(check["values"]["steady_abs_err_p95_ratio"])
        self.assertEqual(check["values"]["window"], "full_tow_fallback")

    def test_old_full_tow_p95_is_not_used_for_stability(self):
        """旧报告只有全程 p95；不能把它当稳态稳定性（会把起步瞬态算进去）。"""
        result = task.evaluate_task(metrics(speed={
            "available": True, "steady_mae_mps": 0.02,
            "p95_abs_err_mps": 0.9}))
        check = result["checks"]["speed_tracking"]
        self.assertTrue(check["ok"])
        self.assertIsNone(check["values"]["steady_abs_err_p95_ratio"])


class ImpactCheckTests(unittest.TestCase):
    def _impact(self, **overrides):
        block = {"available": True, "joint_rms_rad": 0.2, "rms_over_steady": 1.0,
                 "rms_delta_rad": 0.0, "torque_saturated_frac": 0.0}
        block.update(overrides)
        return {"available": True, "startup": dict(block), "takeup": {"available": False},
                "stop": {"available": False}, "stop_contact": {"available": False},
                "steady": {"available": True, "joint_rms_rad": 0.2}}

    def test_normal_startup_passes(self):
        result = task.evaluate_task(metrics(impact=self._impact()))
        self.assertTrue(result["checks"]["start_stop_impact"]["ok"])

    def test_rms_ratio_above_limit_fails(self):
        result = task.evaluate_task(metrics(impact=self._impact(rms_over_steady=1.6)))
        self.assertEqual(result["code"], "IMPACT_HIGH")
        self.assertIn("start_stop_impact", result["reasons"])

    def test_rms_delta_above_limit_fails(self):
        result = task.evaluate_task(metrics(impact=self._impact(rms_delta_rad=0.11)))
        self.assertEqual(result["code"], "IMPACT_HIGH")

    def test_torque_saturation_above_limit_fails(self):
        result = task.evaluate_task(metrics(impact=self._impact(torque_saturated_frac=0.06)))
        self.assertEqual(result["code"], "IMPACT_HIGH")

    def test_contact_itself_is_not_a_failure(self):
        impact = self._impact()
        impact["stop_contact"] = {"available": True, "joint_rms_rad": 0.2,
                                  "rms_over_steady": 0.8, "rms_delta_rad": -0.04,
                                  "torque_saturated_frac": 0.0, "contact": True}
        result = task.evaluate_task(metrics(impact=impact))
        self.assertEqual(result["code"], "PASS")
        self.assertTrue(result["checks"]["start_stop_impact"]["ok"])

    def test_contact_impact_can_fail(self):
        impact = self._impact()
        impact["stop_contact"] = {"available": True, "joint_rms_rad": 0.4,
                                  "rms_over_steady": 2.0, "rms_delta_rad": 0.2,
                                  "torque_saturated_frac": 0.0, "contact": True}
        result = task.evaluate_task(metrics(impact=impact))
        self.assertEqual(result["code"], "IMPACT_HIGH")
        self.assertIn("stop_contact",
                      result["checks"]["start_stop_impact"]["values"]["failed_windows"])

    def test_no_impact_windows_is_not_applicable_not_failure(self):
        result = task.evaluate_task(metrics(impact={"available": False}))
        check = result["checks"]["start_stop_impact"]
        self.assertFalse(check["applicable"])
        self.assertIsNone(check["ok"])
        self.assertEqual(result["code"], "PASS")


class LaneAndJointTests(unittest.TestCase):
    """2026-10-19 第二轮：横向保持与关节静差恢复为失败项。"""

    def test_lane_offset_above_limit_fails(self):
        result = task.evaluate_task(metrics(lane={
            "available": True, "mode": "pd", "y_max_abs_m": 0.31,
            "heading_max_abs_rad": 0.01}))
        self.assertEqual(result["code"], "LANE_DEVIATION")
        self.assertIn("lane_deviation", result["reasons"])
        self.assertTrue(result["checks"]["lane_keeping"]["applicable"])

    def test_heading_above_limit_fails(self):
        result = task.evaluate_task(metrics(lane={
            "available": True, "mode": "pd", "y_max_abs_m": 0.01,
            "heading_max_abs_rad": math.radians(11.0)}))
        self.assertEqual(result["code"], "LANE_DEVIATION")

    def test_lane_keeping_off_is_not_applicable(self):
        result = task.evaluate_task(metrics(lane={
            "available": True, "mode": "off", "y_max_abs_m": 9.0,
            "heading_max_abs_rad": 9.0}))
        self.assertFalse(result["checks"]["lane_keeping"]["applicable"])
        self.assertIsNone(result["checks"]["lane_keeping"]["ok"])
        self.assertEqual(result["code"], "PASS")

    def test_startup_joint_rms_above_limit_fails(self):
        result = task.evaluate_task(metrics(startup={
            "joint_rms_rad": 0.11, "joint_max_rad": 0.20}))
        self.assertEqual(result["code"], "JOINT_ERROR")
        self.assertIn("startup_joint_error", result["reasons"])

    def test_stop_joint_peak_above_limit_fails(self):
        result = task.evaluate_task(metrics(stop={
            "joint_rms_rad": 0.02, "joint_max_rad": 0.31}))
        self.assertEqual(result["code"], "JOINT_ERROR")
        self.assertIn("stop_joint_error", result["reasons"])

    def test_both_joint_windows_can_fail(self):
        result = task.evaluate_task(metrics(
            startup={"joint_rms_rad": 0.2, "joint_max_rad": 0.5},
            stop={"joint_rms_rad": 0.2, "joint_max_rad": 0.5}))
        self.assertEqual(result["code"], "JOINT_ERROR")
        self.assertIn("startup_joint_error", result["reasons"])
        self.assertIn("stop_joint_error", result["reasons"])

    def test_lane_beats_joint_in_code(self):
        result = task.evaluate_task(metrics(
            lane={"available": True, "mode": "pd", "y_max_abs_m": 0.5,
                  "heading_max_abs_rad": 0.0},
            startup={"joint_rms_rad": 0.2, "joint_max_rad": 0.5}))
        self.assertEqual(result["code"], "LANE_DEVIATION")
        self.assertIn("startup_joint_error", result["reasons"])

    def test_joint_beats_speed(self):
        result = task.evaluate_task(metrics(
            speed={"available": True, "steady_mae_mps": 0.5,
                   "steady_speed_std_mps": 0.01, "steady_abs_err_p95_mps": 0.03},
            startup={"joint_rms_rad": 0.2, "joint_max_rad": 0.5}))
        self.assertEqual(result["code"], "JOINT_ERROR")

    def test_missing_lane_and_joint_are_not_failures(self):
        payload = metrics()
        payload.pop("lane")
        payload.pop("startup")
        payload.pop("stop")
        result = task.evaluate_task(payload)
        self.assertFalse(result["checks"]["lane_keeping"]["applicable"])
        self.assertFalse(result["checks"]["joint_static_error"]["applicable"])
        self.assertEqual(result["code"], "PASS")

    def test_legacy_thresholds_are_used(self):
        payload = metrics(startup={"joint_rms_rad": 0.15, "joint_max_rad": 0.2})
        self.assertEqual(task.evaluate_task(payload)["code"], "JOINT_ERROR")
        self.assertEqual(task.evaluate_task(payload, legacy_thresholds={
            "joint_rms_limit_rad": 0.20})["code"], "PASS")


class FallCheckTests(unittest.TestCase):
    def test_fell_fails_even_for_light_cart(self):
        result = task.evaluate_task(metrics(mass=10.0, stability={
            "available": True, "fell": True, "invalid": False,
            "max_abs_roll_rad": 0.05}))
        self.assertEqual(result["code"], "FALL")
        self.assertIn("robot_fell", result["reasons"])

    def test_rollover_fails_even_without_fell(self):
        result = task.evaluate_task(metrics(mass=30.0, stability={
            "available": True, "fell": False, "invalid": False,
            "max_abs_roll_rad": 0.95}))
        self.assertEqual(result["code"], "FALL")
        self.assertIn("robot_rollover", result["reasons"])

    def test_roll_under_limit_passes(self):
        result = task.evaluate_task(metrics(mass=30.0, stability={
            "available": True, "fell": False, "invalid": False,
            "max_abs_roll_rad": 0.79}))
        self.assertEqual(result["code"], "PASS")

    def test_missing_roll_does_not_fail(self):
        result = task.evaluate_task(metrics(mass=30.0, stability={
            "available": True, "fell": False, "invalid": False,
            "max_abs_roll_rad": None}))
        self.assertEqual(result["code"], "PASS")


class SeverityTests(unittest.TestCase):
    def test_invalid_wins(self):
        result = task.evaluate_task(metrics(stability={
            "available": True, "fell": True, "invalid": True, "max_abs_roll_rad": 2.0}))
        self.assertEqual(result["code"], "INVALID")
        self.assertIn("invalid_record", result["reasons"])

    def test_fall_beats_impact_and_speed(self):
        impact = ImpactCheckTests()._impact(rms_over_steady=3.0)
        result = task.evaluate_task(metrics(speed={
            "available": True, "steady_mae_mps": 5.0, "steady_speed_std_mps": 5.0,
            "steady_abs_err_p95_mps": 5.0}, impact=impact, stability={
                "available": True, "fell": True, "invalid": False, "max_abs_roll_rad": 0.0}))
        self.assertEqual(result["code"], "FALL")

    def test_impact_beats_speed(self):
        impact = ImpactCheckTests()._impact(rms_delta_rad=1.0)
        result = task.evaluate_task(metrics(speed={
            "available": True, "steady_mae_mps": 5.0, "steady_speed_std_mps": 5.0,
            "steady_abs_err_p95_mps": 5.0}, impact=impact))
        self.assertEqual(result["code"], "IMPACT_HIGH")

    def test_no_cart_is_not_applicable(self):
        result = task.evaluate_task(metrics(cart_present=False))
        self.assertFalse(result["applicable"])
        self.assertEqual(result["code"], "NA")
        self.assertIsNone(result["ok"])
        self.assertEqual(result["reasons"], [])


class ThresholdTests(unittest.TestCase):
    def test_legacy_speed_mae_limit_is_merged(self):
        merged = task.resolve_thresholds({"speed_mae_ratio_limit": 0.3})
        self.assertEqual(merged["speed_mae_ratio_limit"], 0.3)
        self.assertEqual(merged["speed_std_ratio_limit"],
                         task.DEFAULT_TASK_THRESHOLDS["speed_std_ratio_limit"])

    def test_task_thresholds_override_only_known_keys(self):
        merged = task.resolve_thresholds({}, {"speed_std_ratio_limit": 0.2,
                                              "unknown_key": 99})
        self.assertEqual(merged["speed_std_ratio_limit"], 0.2)
        self.assertNotIn("unknown_key", merged)

    def test_task_dict_can_override_speed_mae_limit(self):
        merged = task.resolve_thresholds({"speed_mae_ratio_limit": 0.3},
                                         {"speed_mae_ratio_limit": 0.15})
        self.assertEqual(merged["speed_mae_ratio_limit"], 0.15)

    def test_inputs_are_not_mutated(self):
        legacy = {"speed_mae_ratio_limit": 0.25}
        task_thresholds = {"roll_limit_rad": 0.5}
        result = task.evaluate_task(metrics(), legacy_thresholds=legacy,
                                    task_thresholds=task_thresholds)
        self.assertEqual(legacy, {"speed_mae_ratio_limit": 0.25})
        self.assertEqual(task_thresholds, {"roll_limit_rad": 0.5})
        self.assertEqual(result["thresholds"]["roll_limit_rad"], 0.5)

    def test_custom_mae_limit_changes_verdict(self):
        payload = metrics(speed={"available": True, "steady_mae_mps": 0.15,
                                 "steady_speed_std_mps": 0.01,
                                 "steady_abs_err_p95_mps": 0.03})
        self.assertEqual(task.evaluate_task(payload, legacy_thresholds={
            "speed_mae_ratio_limit": 0.20})["code"], "PASS")
        self.assertEqual(task.evaluate_task(payload, legacy_thresholds={
            "speed_mae_ratio_limit": 0.10})["code"], "SPEED_UNSTABLE")


def summary_of(*payloads):
    out = []
    for payload in payloads:
        if "task" not in payload:
            payload["task"] = task.evaluate_task(payload)
        out.append({"metrics": payload, "case": payload["case"]})
    return out


class SummaryTests(unittest.TestCase):
    def test_counts_by_band_and_check(self):
        good_light = task.evaluate_task(metrics(mass=10.0))
        bad_light = task.evaluate_task(metrics(mass=10.0, speed={
            "available": True, "steady_mae_mps": 0.5, "steady_speed_std_mps": 0.01,
            "steady_abs_err_p95_mps": 0.03}))
        good_heavy = task.evaluate_task(metrics(mass=30.0))
        fell_heavy = task.evaluate_task(metrics(mass=30.0, stability={
            "available": True, "fell": True, "invalid": False, "max_abs_roll_rad": 0.0}))
        summaries = [
            {"metrics": {"case": {"grade_deg": 0.0}, "task": good_light}},
            {"metrics": {"case": {"grade_deg": 0.0}, "task": bad_light}},
            {"metrics": {"case": {"grade_deg": 5.0}, "task": good_heavy}},
            {"metrics": {"case": {"grade_deg": 5.0}, "task": fell_heavy}},
        ]
        result = task.summarize_task(summaries)
        self.assertEqual(result["with_task"], 4)
        self.assertEqual(result["applicable"], 4)
        self.assertEqual(result["pass"], 2)
        self.assertEqual(result["bands"]["light"]["cases"], 2)
        self.assertEqual(result["bands"]["light"]["checks"]["speed_tracking"]["applicable"], 2)
        self.assertEqual(result["bands"]["light"]["checks"]["speed_tracking"]["pass"], 1)
        self.assertEqual(result["bands"]["heavy"]["cases"], 2)
        self.assertEqual(result["bands"]["heavy"]["checks"]["no_fall"]["fail"], 1)
        self.assertEqual(result["bands"]["heavy"]["checks"]["no_fall"]["pass"], 1)
        self.assertEqual(result["codes"]["PASS"], 2)
        self.assertEqual(result["reasons"]["speed_mae_high"], 1)
        self.assertEqual(result["reasons"]["robot_fell"], 1)
        self.assertEqual(result["grades"]["grade0"]["pass"], 1)
        self.assertEqual(result["grades"]["grade5"]["pass"], 1)

    def test_records_without_task_are_not_counted(self):
        result = task.summarize_task([
            {"metrics": {"case": {"grade_deg": 0.0}}},
            {"metrics": {"case": {"grade_deg": 0.0},
                         "task": task.evaluate_task(metrics(mass=10.0))}},
        ])
        self.assertEqual(result["records"], 2)
        self.assertEqual(result["with_task"], 1)
        self.assertEqual(result["applicable"], 1)
        self.assertEqual(result["pass"], 1)

    def test_conclusion_lines_mention_three_axes_and_legacy_exclusion(self):
        result = task.summarize_task(summary_of(
            metrics(mass=10.0), metrics(mass=30.0)))
        text = "\n".join(task.task_conclusion_lines(result))
        self.assertIn("小车档", text)
        self.assertIn("重车档", text)
        self.assertIn("速度跟随稳定", text)
        self.assertIn("启停冲击", text)
        self.assertIn("不翻车", text)
        self.assertIn("`LAT`", text)
        self.assertIn("`JNT`", text)
        self.assertIn("`COL`", text)
        self.assertIn("仍是失败项", text)
        self.assertIn("不要求停车后刻意往前走", text)

    def test_summary_table_has_five_checks_and_bands(self):
        result = task.summarize_task(summary_of(
            metrics(mass=10.0), metrics(mass=30.0)))
        table = "\n".join(task.format_task_summary_table(result))
        self.assertIn("| 档 | 速度跟随稳定性 | 横向/朝向保持 | 关节静差 | 启停冲击 | "
                      "不翻车 | 任务通过 |", table)
        self.assertIn("小车（≤ 15 kg）", table)
        self.assertIn("重车（> 15 kg）", table)
        self.assertIn("—（不判）", table)      # 重车不判速度
        self.assertIn("2/2", table)            # 两条 case 都通过

    def test_empty_summary_table_is_empty(self):
        self.assertEqual(task.format_task_summary_table(task.summarize_task([])), [])

    def test_old_report_conclusion_is_explicitly_unavailable(self):
        text = "\n".join(task.task_conclusion_lines(task.summarize_task([])))
        self.assertIn("没有 `metrics.task` 字段", text)
        self.assertIn("towing_task_criteria.py", text)


class MatrixTests(unittest.TestCase):
    def test_matrix_uses_short_codes_and_counts(self):
        payloads = [
            {"metrics": {"case": {"grade_deg": 0.0, "velocity_mps": 1.0,
                                  "connection": "compliant", "cart_mass_kg": 10.0},
                         "task": task.evaluate_task(metrics(mass=10.0))}},
            {"metrics": {"case": {"grade_deg": 0.0, "velocity_mps": 1.0,
                                  "connection": "compliant", "cart_mass_kg": 10.0},
                         "task": task.evaluate_task(metrics(mass=10.0, speed={
                             "available": True, "steady_mae_mps": 0.5,
                             "steady_speed_std_mps": 0.01,
                             "steady_abs_err_p95_mps": 0.03}))}},
            {"metrics": {"case": {"grade_deg": 5.0, "velocity_mps": 1.0,
                                  "connection": "rigid", "cart_mass_kg": 30.0},
                         "task": task.evaluate_task(metrics(mass=30.0, stability={
                             "available": True, "fell": True, "invalid": False,
                             "max_abs_roll_rad": 0.0}))}},
        ]
        matrix = task.format_task_matrix(payloads, grades=[0.0, 5.0])
        self.assertIn("坡度量级 平地", matrix)
        self.assertIn("坡度量级 5° 坡", matrix)
        self.assertIn("SPD×2", matrix)       # 最严重的一条 + 该格 env 数
        self.assertIn("FALL×1", matrix)

    def test_empty_grades_return_empty(self):
        self.assertEqual(task.format_task_matrix([], grades=[0.0]), "")


class ReplayTests(unittest.TestCase):
    def test_replay_fills_task_for_old_report(self):
        old = {"thresholds": {"speed_mae_ratio_limit": 0.2},
               "cases": [{"metrics": metrics(mass=10.0)}]}
        summary = task._evaluate_report(old)
        self.assertTrue(old["cases"][0]["metrics"]["task"]["version"])
        self.assertEqual(summary["with_task"], 1)
        self.assertEqual(summary["pass"], 1)

    def test_replay_does_not_overwrite_existing_task(self):
        payload = metrics(mass=10.0, speed={
            "available": True, "steady_mae_mps": 5.0, "steady_speed_std_mps": 5.0,
            "steady_abs_err_p95_mps": 5.0})
        payload["task"] = task.evaluate_task(payload)
        self.assertEqual(payload["task"]["code"], "SPEED_UNSTABLE")
        summary = task._evaluate_report({"cases": [{"metrics": payload}]})
        self.assertEqual(summary["codes"]["SPEED_UNSTABLE"], 1)


if __name__ == "__main__":
    unittest.main()
