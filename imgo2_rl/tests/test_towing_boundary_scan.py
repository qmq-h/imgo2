"""Offline tests for the boundary-scan orchestrator."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


RL = Path(__file__).resolve().parents[1]
SCRIPT = RL / "scripts/towing/scan_towing_boundary.py"
spec = importlib.util.spec_from_file_location("scan_towing_boundary_test", SCRIPT)
scan = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = scan
spec.loader.exec_module(scan)


class BoundaryScanTests(unittest.TestCase):
    def test_default_grid_is_point_two_through_one(self):
        args = scan.parse_args([])
        self.assertEqual(args.velocities, (0.2, 0.4, 0.6, 0.8, 1.0))
        self.assertEqual(args.ground_frictions, (0.8,))
        self.assertEqual(len(args.velocities) * len(args.cart_masses) *
                         len(args.wheel_dampings), 50)

    def test_velocity_validation_rejects_out_of_range_and_duplicates(self):
        for argv in (["--velocities", "0.1"], ["--velocities", "1.1"],
                     ["--velocities", "0.4", "0.4"]):
            with self.assertRaises(SystemExit):
                scan.parse_args(argv)

    def test_classification_keeps_pullability_separate_from_stop_collision(self):
        base = {"valid": True, "steady_tracking_ratio": 0.95,
                "reached_robot": False, "final_clearance_m": 0.2}
        self.assertEqual(scan.classify(base, tracking_min=0.8, clearance_margin=0.1),
                         "direct_stop_safe")
        self.assertEqual(scan.classify({**base, "final_clearance_m": 0.05},
                                       tracking_min=0.8, clearance_margin=0.1),
                         "stop_margin_low")
        self.assertEqual(scan.classify({**base, "reached_robot": True},
                                       tracking_min=0.8, clearance_margin=0.1),
                         "stop_collision")
        self.assertEqual(scan.classify({**base, "steady_tracking_ratio": 0.7},
                                       tracking_min=0.8, clearance_margin=0.1),
                         "tow_infeasible")

    def test_boundary_table_reports_tested_velocity_limits(self):
        common = {"rope_model": "compliant", "ground_friction": 0.8,
                  "wheel_damping": 0.032,
                  "cart_mass_kg": 10.0}
        rows = [{**common, "velocity_mps": 0.2, "classification": "direct_stop_safe"},
                {**common, "velocity_mps": 0.4, "classification": "stop_margin_low"},
                {**common, "velocity_mps": 0.6, "classification": "stop_collision"},
                {**common, "velocity_mps": 0.8, "classification": "tow_infeasible"}]
        boundary = scan.boundary_table(rows)[0]
        self.assertEqual(boundary["max_tested_tow_feasible_velocity_mps"], 0.6)
        self.assertEqual(boundary["max_direct_stop_safe_velocity_mps"], 0.2)
        self.assertEqual(boundary["first_tested_stop_risk_velocity_mps"], 0.4)

    def test_load_rows_reads_velocity_from_case_config(self):
        with tempfile.TemporaryDirectory() as temp:
            case = Path(temp) / "v0p20" / "case_00"
            case.mkdir(parents=True)
            (case / "config.json").write_text(
                json.dumps({"user_command_mps": 0.2, "ground_friction": 0.8}), "utf-8")
            summary = {"valid": True, "steady_tracking_ratio": 0.9,
                       "reached_robot": False, "final_clearance_m": 0.2,
                       "min_clearance_m": 0.15, "cart_mass_kg": 5.0,
                       "wheel_damping": 0.032, "rope_model": "compliant", "failures": []}
            (case / "summary.json").write_text(json.dumps(summary), "utf-8")
            rows = scan.load_rows(Path(temp), tracking_min=0.8, clearance_margin=0.1)
        self.assertEqual(rows[0]["velocity_mps"], 0.2)
        self.assertEqual(rows[0]["classification"], "direct_stop_safe")


if __name__ == "__main__":
    unittest.main()
