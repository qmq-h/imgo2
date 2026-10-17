"""Offline tests for the per-segment / per-connection towing report tool.

用**合成 run**（experiment.json + summaries/*.json + env*/tow.csv）把聚合口径钉住：
地形族与段的切分、绳/刚体归类、先按 case 再跨 case 的分位数、步态周期估计、启动/停车事件、
以及「没有逐行 CSV 时必须明确报错」。
"""
import csv
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

RL = Path(__file__).resolve().parents[1]
TOOL = RL / "scripts/tools/towing_segment_report.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("towing_segment_report_under_test", TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules["towing_segment_report_under_test"] = module
    spec.loader.exec_module(module)
    return module


TOOL_MODULE = load_tool()

ROW_FIELDS = ("phase", "time_s", "user_cmd_mps", "robot_vx_b_mps", "robot_vx_mps",
              "joint_err_rms_rad", "rope_tension_n", "rope_taut", "rope_impulse_ns",
              "robot_progress_m", "robot_jp_01", "cart_deck_fx_n")


def write_case(run, *, env, column, row, connection, speed, mass,
               summary_metrics=None):
    name = (f"env{env:04d}_c{column:02d}r{row:02d}_g5_v{speed:g}"
            f"_{connection}_m{mass:g}kg")
    directory = run / name
    directory.mkdir(parents=True)
    dt = 0.025
    rows = []
    # station 0–1 s、tow 1–10 s、coast 10–11 s；拖曳段 1 m/s ⇒ 每行前进 0.025 m
    step = 0
    progress = 0.0
    for _ in range(40):                       # station
        rows.append(dict(phase="station", time_s=(step + 1) * dt, user_cmd_mps=0.0,
                         robot_vx_b_mps=0.0, robot_vx_mps=0.0, joint_err_rms_rad=0.02,
                         rope_tension_n=0.0, rope_taut=0.0, rope_impulse_ns=0.0,
                         robot_progress_m=progress, robot_jp_01=0.8,
                         cart_deck_fx_n=0.0))
        step += 1
    for _ in range(360):                      # tow：1 → 10 m
        rows.append(dict(phase="tow", time_s=(step + 1) * dt, user_cmd_mps=speed,
                         robot_vx_b_mps=speed, robot_vx_mps=speed,
                         joint_err_rms_rad=0.05, rope_tension_n=8.0, rope_taut=1.0,
                         rope_impulse_ns=0.2, robot_progress_m=progress,
                         robot_jp_01=0.8 + 0.2 * math.sin(2 * math.pi * 2.5 * (step * dt)),
                         cart_deck_fx_n=0.0))
        progress += speed * dt
        step += 1
    for index in range(40):                   # coast：20 行内从 0.4 衰减到 0（真的停住）
        coast_vx = max(0.0, 0.4 * (1.0 - index / 20.0))
        rows.append(dict(phase="coast", time_s=(step + 1) * dt, user_cmd_mps=0.0,
                         robot_vx_b_mps=0.0, robot_vx_mps=coast_vx, joint_err_rms_rad=0.04,
                         rope_tension_n=2.0, rope_taut=0.0, rope_impulse_ns=0.0,
                         robot_progress_m=progress, robot_jp_01=0.8, cart_deck_fx_n=12.0))
        step += 1
    with (directory / "tow.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ROW_FIELDS)
        writer.writeheader()
        for entry in rows:
            writer.writerow({key: entry[key] for key in ROW_FIELDS})
    summaries = run / "summaries"
    summaries.mkdir(exist_ok=True)
    metrics = summary_metrics or {}
    payload = {"case": {"env_index": env, "cell_index": env, "column": column, "row": row,
                        "connection": connection, "connection_length_m": 0.6,
                        "grade_deg": 5.0, "velocity_mps": speed, "cart_mass_kg": mass,
                        "cart_present": True},
               "metrics": {"startup": metrics.get("startup", {}),
                           "stop": metrics.get("stop", {}),
                           "verdict": {"code": "OK", "reasons": []}},
               "verdict": {"code": "OK", "reasons": []}}
    (summaries / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
    return directory


def write_run(root, *, cart_present=True):
    run = root / "20261017T000000Z_test"
    run.mkdir(parents=True)
    (run / "experiment.json").write_text(json.dumps({
        "arguments": {"terrain_profile": "hill", "dt": 0.005, "record_every": 5,
                      "velocities": [1.0], "num_envs": 3},
        "cases": []}), encoding="utf-8")
    metrics = {"startup": {"joint_rms_rad": 0.08, "joint_max_rad": 0.2, "torque_saturated_frac": 0.0,
                           "time_to_90pct_s": 0.2, "overshoot_ratio": 0.05},
               "stop": {"contact": False, "min_clearance_coast_m": 0.4,
                        "clearance_at_stop_m": 0.5, "deck_contact_peak_n": 0.0,
                        "time_to_contact_after_stop_s": None,
                        "cart_coast_distance_m": 0.1, "joint_max_rad": 0.25,
                        "joint_rms_rad": 0.05, "settle_time_s": 0.5}}
    # 波浪列 50（3 周期，上下坡交替）、减速带列 40、坡列 20；连接按 绳/绳/刚体
    write_case(run, env=0, column=50, row=0, connection="compliant", speed=1.0, mass=5.0,
               summary_metrics=metrics)
    write_case(run, env=1, column=40, row=0, connection="inextensible", speed=1.0, mass=5.0,
               summary_metrics=metrics)
    rigid = {"startup": {"joint_rms_rad": 0.15, "joint_max_rad": 0.4,
                         "torque_saturated_frac": 0.1, "time_to_90pct_s": 0.3,
                         "overshoot_ratio": 0.2},
             "stop": {"contact": True, "min_clearance_coast_m": 0.05,
                      "clearance_at_stop_m": 0.2, "deck_contact_peak_n": 30.0,
                      "time_to_contact_after_stop_s": 0.3,
                      "cart_coast_distance_m": 0.3, "joint_max_rad": 0.35,
                      "joint_rms_rad": 0.09, "settle_time_s": 0.6}}
    write_case(run, env=2, column=20, row=0, connection="rigid", speed=1.0, mass=5.0,
               summary_metrics=rigid)
    return run


class HelperTests(unittest.TestCase):
    def test_family_and_connection_mapping(self):
        self.assertEqual(TOOL_MODULE.family_of(0), "平地")
        self.assertEqual(TOOL_MODULE.family_of(20), "5°坡")
        self.assertEqual(TOOL_MODULE.family_of(30), "10°坡")
        self.assertEqual(TOOL_MODULE.family_of(40), "减速带")
        self.assertEqual(TOOL_MODULE.family_of(59), "波浪")
        self.assertEqual(TOOL_MODULE.connection_class("compliant"), "绳")
        self.assertEqual(TOOL_MODULE.connection_class("inextensible"), "绳")
        self.assertEqual(TOOL_MODULE.connection_class("rigid"), "刚体")
        with self.assertRaises(SystemExit):
            TOOL_MODULE.connection_class("elastic")

    def test_gait_period_on_a_pure_sine_and_on_noise(self):
        dt = 0.025
        samples = [math.sin(2 * math.pi * 2.5 * index * dt) for index in range(200)]
        period, strength = TOOL_MODULE.gait_period(samples, dt)
        self.assertAlmostEqual(period, 0.4, places=2)
        self.assertGreater(strength, 0.85)
        # 常数序列没有周期
        self.assertEqual(TOOL_MODULE.gait_period([0.3] * 200, dt), (None, None))
        # 太短不估
        self.assertEqual(TOOL_MODULE.gait_period(samples[:20], dt), (None, None))

    def test_quantile_matches_linear_interpolation(self):
        self.assertAlmostEqual(TOOL_MODULE.quantile([0.0, 1.0], 0.5), 0.5)
        self.assertAlmostEqual(TOOL_MODULE.quantile([0.0, 10.0, 20.0], 0.10), 2.0)
        self.assertIsNone(TOOL_MODULE.quantile([], 0.5))


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run = write_run(Path(self.tmp.name))
        self.per_case, self.skipped = TOOL_MODULE.collect([self.run])
        self.lines = TOOL_MODULE.build_report(self.per_case, profile="hill")
        self.text = "\n".join(self.lines)

    def tearDown(self):
        self.tmp.cleanup()

    def test_cases_and_classes(self):
        self.assertEqual(len(self.per_case), 3)
        classes = sorted(record["case"]["connection_class"] for record in self.per_case)
        self.assertEqual(classes, ["刚体", "绳", "绳"])
        families = sorted(record["case"]["family"] for record in self.per_case)
        self.assertEqual(families, ["5°坡", "减速带", "波浪"])
        self.assertEqual(self.skipped, [])

    def test_segments_cover_the_lane_and_split_up_down(self):
        wave = next(record for record in self.per_case
                    if record["case"]["family"] == "波浪")
        self.assertIn("平地(出生)", wave["segments"])
        self.assertIn("上坡", wave["segments"])
        self.assertIn("下坡", wave["segments"])
        up = wave["segments"]["上坡"]
        down = wave["segments"]["下坡"]
        self.assertGreater(up["samples"], 0)
        self.assertGreater(down["samples"], 0)
        self.assertAlmostEqual(up["abs_err"], 0.0, places=9)     # 合成数据 vx == cmd
        self.assertAlmostEqual(down["ratio"], 1.0, places=9)
        # 3 周期正弦：上/下坡样本量应相近
        self.assertLess(abs(up["samples"] - down["samples"]), 0.5 * max(up["samples"], down["samples"]))
        bump = next(record for record in self.per_case
                    if record["case"]["family"] == "减速带")
        self.assertIn("带顶", bump["segments"])
        self.assertIn("平地(带前)", bump["segments"])

    def test_gait_period_recovered_in_every_oriented_segment(self):
        for record in self.per_case:
            period = record["segments"]["下坡"]["gait_period"]
            if period is not None:
                # 行距 0.025 s ⇒ 周期分辨率就是一个 lag，允许 1 个 lag 的量化误差
                self.assertLessEqual(abs(period - 0.4), 0.03, msg=f"period={period}")

    def test_report_has_all_requested_blocks(self):
        for title in ("1. 平地：速度跟踪", "1b. 平地 × 速度档",
                      "2. 障碍段：速度跟踪（上/下坡分开）",
                      "2b. 障碍段 × 速度（绳）", "2b. 障碍段 × 速度（刚体）",
                      "3. 下坡：小车影响与步态",
                      "3b. 下坡 × 速度（绳）", "3b. 下坡 × 速度（刚体）",
                      "4. 启动：关节超限与冲击", "5. 停车：是否撞到机器人与冲击",
                      "6. 判定码分布"):
            self.assertIn(title, self.text)
        # 表里必须同时出现绳与刚体的行
        self.assertIn("| 绳 |", self.text)
        self.assertIn("| 刚体 |", self.text)
        self.assertIn("波浪 · 上坡", self.text)
        self.assertIn("5°坡 · 下坡", self.text)
        self.assertIn("减速带 · 带顶", self.text)

    def test_startup_and_stop_aggregation_uses_summary_metrics(self):
        rope = [record for record in self.per_case
                if record["case"]["connection_class"] == "绳"]
        rigid = [record for record in self.per_case
                 if record["case"]["connection_class"] == "刚体"]
        self.assertEqual(sum(1 for record in rope if record["events"]["startup_over_max_limit"]), 0)
        self.assertEqual(sum(1 for record in rigid if record["events"]["startup_over_max_limit"]), 1)
        self.assertEqual(sum(1 for record in rope if record["events"]["startup_over_rms_limit"]), 0)
        self.assertEqual(sum(1 for record in rigid if record["events"]["stop_contact"]), 1)
        self.assertEqual(sum(1 for record in rope if record["events"]["stop_contact"]), 0)
        # 起拖窗口（tow 前 0.2 s）内的张力/冲量峰值
        self.assertAlmostEqual(rope[0]["events"]["takeup_tension"], 8.0, places=6)
        self.assertAlmostEqual(rope[0]["events"]["takeup_impulse"], 0.2, places=6)
        # 停车窗口里机器人自身还在动的速度（合成数据首行 0.4 m/s，20 行内衰减到 0）
        self.assertAlmostEqual(rope[0]["events"]["stop_robot_move"], 0.4, places=6)
        # 「真正停住」：coast 末 |vx| = 0
        self.assertTrue(rope[0]["events"]["stop_rest"])
        self.assertAlmostEqual(rope[0]["events"]["stop_final_vx"], 0.0, places=9)
        self.assertIn("停车段走完", self.text)
        self.assertAlmostEqual(rope[0]["events"]["stop_deck_force_window"], 12.0, places=6)

    def test_coast_not_finished_is_flagged(self):
        """coast 结束时机器人还在动 ⇒ 报表要显式告警（用户要求「完整 cmd=0 到真正停止」）。"""
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp))
            case = run / "env0000_c50r00_g5_v1_compliant_m5kg"
            rows = list(csv.DictReader((case / "tow.csv").open(encoding="utf-8")))
            for row in rows:
                if row["phase"] == "coast":
                    row["robot_vx_mps"] = "0.3"      # 一直没停
            with (case / "tow.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=ROW_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            per_case, _ = TOOL_MODULE.collect([run])
            text = "\n".join(TOOL_MODULE.build_report(per_case, profile="hill"))
            self.assertIn("在 coast 结束时仍在动", text)
            record = next(item for item in per_case if item["case"]["family"] == "波浪")
            self.assertFalse(record["events"]["stop_rest"])
            self.assertAlmostEqual(record["events"]["stop_final_vx"], 0.3, places=6)

    def test_missing_csv_fails_with_a_clear_message(self):
        empty = Path(self.tmp.name) / "20261017T999999Z_empty"
        empty.mkdir()
        (empty / "experiment.json").write_text(json.dumps(
            {"arguments": {"terrain_profile": "hill", "dt": 0.005, "record_every": 5}}),
            encoding="utf-8")
        with self.assertRaises(SystemExit) as caught:
            TOOL_MODULE.load_run(empty)
        self.assertIn("--write-csv all", str(caught.exception))

    def test_main_writes_markdown_and_json(self):
        out = Path(self.tmp.name) / "segment_report.md"
        code = TOOL_MODULE.main([str(self.run), "--out", str(out)])
        self.assertEqual(code, 0)
        self.assertTrue(out.is_file())
        self.assertTrue(out.read_text(encoding="utf-8").startswith("# 拖曳分段报表"))
        payload = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
        self.assertEqual(payload["profile"], "hill")
        self.assertEqual(len(payload["cases"]), 3)
        self.assertEqual([skip[:2] for skip in payload["skipped"]], [])


class NoCartTests(unittest.TestCase):
    def test_nocart_cases_are_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp))
            name = run / "env0000_c50r00_g5_v1_compliant_m5kg"
            renamed = run / (name.name + "_nocart")
            name.rename(renamed)
            summary = run / "summaries" / f"{name.name}.json"
            summary.rename(run / "summaries" / f"{renamed.name}.json")
            per_case, skipped = TOOL_MODULE.collect([run])
            self.assertEqual(len(per_case), 3)
            self.assertEqual(skipped, [])


if __name__ == "__main__":
    unittest.main()
