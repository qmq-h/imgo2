"""CPU checks of the real AMP loader's seconds-to-frame interpolation."""
import ast
import glob
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import numpy as np
    import torch
    from pybullet_utils import transformations
except ImportError:
    np = torch = transformations = None

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "scripts/rl_lab/rl_lab"


def module_at(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def loader_class():
    # Avoid importing Isaac Sim through rl_lab.envs, while executing the
    # actual loader and actual quaternion helpers on real CPU tensors.
    pose = module_at(PACKAGE / "datasets/pose3d.py")
    utils = module_at(PACKAGE / "utils/utils.py")
    motion_tree = ast.parse((PACKAGE / "datasets/motion_util.py").read_text())
    standardize = next(n for n in motion_tree.body if isinstance(n, ast.FunctionDef)
                       and n.name == "standardize_quaternion")
    motion_ns = {"np": np, "pose3d": pose}
    exec(compile(ast.Module(body=[standardize], type_ignores=[]), "motion_util.py", "exec"), motion_ns)
    tree = ast.parse((PACKAGE / "datasets/motion_loader.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AMPLoader")
    ns = {"torch": torch, "np": np, "glob": glob, "json": json,
          "transformations": transformations, "utils": utils, "pose3d": pose,
          "motion_util": SimpleNamespace(standardize_quaternion=motion_ns[standardize.name])}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "motion_loader.py", "exec"), ns)
    return ns[cls.name]


@unittest.skipIf(torch is None, "CPU torch/numpy/pybullet required")
class MotionTimingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Loader = loader_class()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        frames = []
        for i in range(5):
            t = i * 0.02
            frames.append([t, 0, 0.3, 0, 0, 0, 1] + [t] * 12 + [0] * 42)
        path = Path(self.directory.name) / "ramp.txt"
        path.write_text(json.dumps({"FrameDuration": 0.02, "MotionWeight": 1, "Frames": frames}))
        self.path = str(path)
        self.loader = self.Loader("cpu", 0.02, motion_files=[self.path])

    def test_scalar_and_batch_interpolate_seconds_and_endpoints(self):
        times = np.array([-0.01, 0, 0.02, 0.03, 0.06, 0.08, 0.10])
        ids = np.zeros(len(times), dtype=int)
        expected = np.clip(times, 0, 0.08)
        amp = self.loader.get_frame_at_time_batch(ids, times)
        full = self.loader.get_full_frame_at_time_batch(ids, times)
        np.testing.assert_allclose(amp[:, 0].numpy(), expected, atol=1e-7)
        np.testing.assert_allclose(full[:, 0].numpy(), expected, atol=1e-7)
        for index, t in enumerate(times):
            torch.testing.assert_close(self.loader.get_frame_at_time(0, t), amp[index])
            torch.testing.assert_close(self.loader.get_full_frame_at_time(0, t), full[index])

    def test_expert_transition_has_the_requested_elapsed_time(self):
        loader = self.Loader("cpu", 0.02, preload_transitions=True,
                             num_preload_transitions=1000, motion_files=[self.path])
        delta = loader.preloaded_s_next[:, 0] - loader.preloaded_s[:, 0]
        torch.testing.assert_close(delta, torch.full_like(delta, 0.02))

    def test_start_time_is_uniform_without_a_clamped_zero_spike(self):
        with patch.object(np.random, "uniform", return_value=0.25):
            self.assertAlmostEqual(self.loader.traj_time_sample(0), 0.015)
        with patch.object(np.random, "uniform", return_value=np.array([0.25, 0.75])):
            times = self.loader.traj_time_sample_batch(np.array([0, 0]))
        np.testing.assert_allclose(times, [0.015, 0.045])

    def test_all_real_motions_return_first_second_and_last_frames(self):
        files = sorted((ROOT / "datasets/imgo2_motion").glob("*.txt"))
        self.assertEqual(len(files), 21)
        loader = self.Loader("cpu", 0.02, motion_files=[str(p) for p in files])
        for index, frames in enumerate(loader.trajectories_full):
            dt = loader.trajectory_frame_durations[index]
            for frame_index in (0, 1, len(frames) - 1):
                with self.subTest(motion=files[index].name, frame=frame_index):
                    result = loader.get_full_frame_at_time_batch(
                        np.array([index]), np.array([frame_index * dt]))[0]
                    torch.testing.assert_close(result, frames[frame_index], atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
