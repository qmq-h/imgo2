"""Cart model and coast data regression tests; standard Python, no simulator."""

import csv
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import xml.etree.ElementTree as ET

RL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RL / "scripts/tools"))
sys.path.insert(0, str(RL / "source/imgo2_rl/imgo2_rl/assets"))
from cart_model import read_cart_model
from cart_coast_metrics import summarize, summarize_sweep, svg_plot
from summarize_cart_coast import summarize_run

URDF = RL.parent / "imgo2_description/cart/cart.urdf"


def module_at(name, relative):
    spec = importlib.util.spec_from_file_location(name, RL / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


resistance = module_at("cart_resistance_test", "source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/mdp/resistance.py")
recording = module_at("cart_recording_test", "source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/utils/recording.py")


def sample(t, v, x):
    row = dict.fromkeys(recording.FIELDS, 0.0)
    row.update(time_s=t, vx_mps=v, x_m=x, z_m=0.15)
    for leg in recording.LEGS:
        row[f"{leg}_omega_radps"] = v / 0.08
        row[f"{leg}_normal_n"] = 24.525
    return row


class CartModelTests(unittest.TestCase):
    def changed_model(self, change):
        root = ET.parse(URDF).getroot()
        change(root)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cart.urdf"
            ET.ElementTree(root).write(path, encoding="utf-8")
            return read_cart_model(path)

    def test_nominal_mass_inertia_and_ground_clearance(self):
        model = read_cart_model(URDF)
        self.assertAlmostEqual(model["total_mass_kg"], 10.0)
        self.assertAlmostEqual(model["resting_height_m"], 0.15)
        self.assertEqual(model["attachment_position_m"], (0.25, 0.0, 0.0))
        self.assertEqual(len(model["masses_kg"]), 5)

    def test_mass_edit_requires_consistent_inertia(self):
        with self.assertRaisesRegex(ValueError, "inertia mismatch"):
            self.changed_model(lambda r: r.find("link/inertial/mass").set("value", "20"))

    def test_negative_mass_rejected(self):
        with self.assertRaisesRegex(ValueError, "mass"):
            self.changed_model(lambda r: r.find("link/inertial/mass").set("value", "-1"))

    def test_reversed_wheel_axis_rejected(self):
        with self.assertRaisesRegex(ValueError, "axis"):
            self.changed_model(lambda r: r.find("joint/axis").set("xyz", "0 -1 0"))

    def test_hidden_resistance_rejected(self):
        with self.assertRaisesRegex(ValueError, "Hidden"):
            self.changed_model(lambda r: r.find("joint/dynamics").set("damping", "0.1"))

    def test_finite_angle_joint_rejected(self):
        with self.assertRaisesRegex(ValueError, "continuously"):
            self.changed_model(lambda r: r.find("joint").set("type", "revolute"))

    def test_visual_collision_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "dimensions"):
            self.changed_model(lambda r: r.find("link[@name='wheel_fl']/collision/geometry/cylinder").set("radius", "0.09"))

    def test_duplicate_link_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            self.changed_model(lambda r: ET.SubElement(r, "link", name="base_link"))


class CoastMetricsTests(unittest.TestCase):
    def summarize(self, rows, **kwargs):
        return summarize(rows, radius=0.08, resting_height=0.15, **kwargs)

    def test_exponential_known_stop_time_and_distance(self):
        rows = [sample(i*0.01, math.exp(-i*0.01), 1-math.exp(-i*0.01)) for i in range(501)]
        result = self.summarize(rows)
        self.assertTrue(result["valid"])
        self.assertTrue(result["stopped"])
        self.assertAlmostEqual(result["stop_time_s"], math.ceil(-math.log(0.02)/0.01)*0.01)
        self.assertAlmostEqual(result["stop_distance_m"], 1-math.exp(-result["stop_time_s"]))

    def test_timeout_does_not_become_stop(self):
        result = self.summarize([sample(i*0.1, 1, i*0.1) for i in range(21)])
        self.assertTrue(result["valid"])
        self.assertFalse(result["stopped"])
        self.assertIsNone(result["stop_distance_m"])
        self.assertAlmostEqual(result["observed_distance_m"], 2)

    def test_temporary_pause_then_motion_is_not_final_stop(self):
        rows = [sample(i*0.1, 0 if i < 10 else 0.5, 0) for i in range(21)]
        self.assertFalse(self.summarize(rows)["stopped"])

    def test_short_final_pause_fails_hold(self):
        rows = [sample(i*0.1, 0.5 if i < 17 else 0, 0) for i in range(21)]
        self.assertFalse(self.summarize(rows)["stopped"])

    def test_reverse_motion_uses_speed_norm(self):
        result = self.summarize([sample(i*0.1, -1, -i*0.1) for i in range(21)])
        self.assertFalse(result["stopped"])

    def test_nonfinite_and_duplicate_time_rejected(self):
        for rows in ([sample(0, 1, 0), sample(1, float("nan"), 1)], [sample(0, 1, 0), sample(0, 1, 1)]):
            with self.assertRaises(ValueError):
                self.summarize(rows)

    def test_missing_contact_and_tipping_invalidate(self):
        rows = [sample(0, 0, 0), sample(1, 0, 0)]
        rows[-1].update(fl_normal_n=0, roll_rad=0.7)
        result = self.summarize(rows)
        self.assertFalse(result["valid"])
        self.assertIn("missing_final_wheel_contact", result["failures"])

    def test_torque_sign_and_sample_timing(self):
        rows = [sample(0, 1, 0), sample(1, 0, 0.5)]
        rows[1]["fl_tau_nm"] = 1
        self.assertIn("resistance_injects_energy", self.summarize(rows)["failures"])
        rows[1]["fl_tau_nm"] = -1
        self.assertEqual(self.summarize(rows)["positive_resistance_work_j"], 0)

    def test_resistance_is_dissipative_in_both_directions(self):
        for omega in (-10, -0.001, 0, 0.001, 10):
            self.assertLessEqual(omega * resistance.viscous_resistance(omega, 0.016), 0)
        for damping in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                resistance.viscous_resistance(1, damping)

    def test_csv_roundtrip_plot_and_partial_run_status(self):
        config = {"model": read_cart_model(URDF), "mode": "coast", "stop_speed_mps": 0.02, "stop_hold_s": 0.5}
        with tempfile.TemporaryDirectory() as parent:
            directory = Path(parent) / "run"
            with recording.CartRecorder(directory, config) as recorder:
                for i in range(101):
                    recorder.append(sample(i*0.1, math.exp(-i*0.1), 1-math.exp(-i*0.1)))
                with self.assertRaises(ValueError):
                    recorder.append(sample(10, 0, 1))
            result = summarize_run(directory)
            self.assertFalse(result["valid"])
            self.assertIn("run_not_completed", result["failures"])
            recording.write_json(directory / "status.json", {"state": "completed"})
            self.assertTrue(summarize_run(directory)["valid"])
            self.assertEqual(ET.parse(directory / "coast.svg").getroot().tag, "{http://www.w3.org/2000/svg}svg")
            with self.assertRaises(FileExistsError):
                recording.CartRecorder(directory, config)

    def test_completed_but_truncated_csv_is_invalid(self):
        config = {"model": read_cart_model(URDF), "mode": "coast", "stop_speed_mps": 0.02,
                  "stop_hold_s": 0.5, "dt_s": 0.1, "requested_duration_s": 2.0}
        with tempfile.TemporaryDirectory() as parent:
            directory = Path(parent) / "run"
            with recording.CartRecorder(directory, config) as recorder:
                for i in range(11):
                    recorder.append(sample(i*0.1, 0, 0))
            recording.write_json(directory / "status.json", {"state": "completed"})
            result = summarize_run(directory)
            self.assertFalse(result["valid"])
            self.assertIn("incomplete_or_irregular_samples", result["failures"])

    def test_sweep_candidates_require_valid_stopped_one_mps_run(self):
        def case(v, b, distance, valid=True, stopped=True):
            return dict(velocity_mps=v, damping=b, stop_distance_m=distance, valid=valid, stopped=stopped)
        result = summarize_sweep([
            case(1, 0, None, stopped=False), case(1, 0.008, 2.1), case(1, 0.016, 1.0),
            case(1, 0.032, 0.6, valid=False), case(0.5, 0.064, 1.0)])
        self.assertEqual(result["candidate_damping_nms_per_rad"], [0.016])
        self.assertTrue(result["groups"][1]["distance_decreases_with_damping"])
        nonmonotone = summarize_sweep([case(1, 0.008, 0.7), case(1, 0.016, 1.0)])
        self.assertFalse(nonmonotone["groups"][0]["distance_decreases_with_damping"])


class CartAdapterAndFingerprintTests(unittest.TestCase):
    def test_efforts_reach_backend_after_base_writes_zero(self):
        # Execute the actual class with a recording backend, not Isaac Sim.
        # This verifies ordering/data forwarding only, not physical response.
        source = ast.parse((RL / "source/imgo2_rl/imgo2_rl/assets/cart.py").read_text(encoding="utf-8"))
        definition = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "PassiveCartArticulation")
        calls = []

        class Base:
            def write_data_to_sim(self):
                calls.append("base_zero")

        torch_stub = SimpleNamespace(arange=lambda count, **kwargs: list(range(count)), long="long")
        namespace = {"Articulation": Base, "torch": torch_stub}
        exec(compile(ast.Module(body=[definition], type_ignores=[]), "cart.py", "exec"), namespace)
        cart = namespace["PassiveCartArticulation"]()
        target = [[-0.1, 0.2, 0.0, -0.3], [-0.2, 0.1, 0.0, -0.4]]
        cart.cfg = SimpleNamespace(actuators={})
        cart.data = SimpleNamespace(joint_effort_target=target)
        cart.num_instances, cart.device = 2, "cpu"
        setter = Mock(side_effect=lambda efforts, indices: calls.append((efforts, indices)))
        cart.root_physx_view = SimpleNamespace(set_dof_actuation_forces=setter)
        cart.write_data_to_sim()
        self.assertEqual(calls, ["base_zero", (target, [0, 1])])
        cart.cfg.actuators = {"motor": object()}
        with self.assertRaises(ValueError):
            cart.write_data_to_sim()
        self.assertEqual(setter.call_count, 1)

    def test_mesh_fingerprint_uses_cross_platform_filename_order(self):
        source = ast.parse((RL / "scripts/tools/check_model_sync.py").read_text(encoding="utf-8"))
        definition = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "mesh_fingerprint")
        namespace = {"hashlib": hashlib, "Path": Path}
        exec(compile(ast.Module(body=[definition], type_ignores=[]), "check_model_sync.py", "exec"), namespace)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "base.STL").write_bytes(b"base")
            (root / "Wheel.STL").write_bytes(b"wheel")
            expected = '\n'.join(f'{name}:{hashlib.sha256(data).hexdigest()}'
                                 for name, data in [('Wheel.STL', b'wheel'), ('base.STL', b'base')])
            self.assertEqual(namespace["mesh_fingerprint"](root), (hashlib.sha256(expected.encode()).hexdigest()[:12], 2))
            (root / "Wheel.STL").write_bytes(b"corrupt")
            self.assertNotEqual(namespace["mesh_fingerprint"](root)[0], hashlib.sha256(expected.encode()).hexdigest()[:12])


if __name__ == "__main__":
    unittest.main()
