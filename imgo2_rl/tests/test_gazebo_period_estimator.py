"""Check actual Gazebo estimator without importing ROS 2."""
import ast
import math
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'imgo2_deploy/ros2_ws/scripts/eval_gazebo_policy.py'
TREE = ast.parse(SOURCE.read_text())
FUNC = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == 'autocorr_peak')
NS = {'math': math}
exec(compile(ast.Module(body=[FUNC], type_ignores=[]), str(SOURCE), 'exec'), NS)


class PeriodEstimatorTests(unittest.TestCase):
    def test_known_periods_with_floating_timestamp_dt(self):
        for period in (0.16, 0.27, 0.5, 0.6):
            with self.subTest(period=period):
                signal = [math.sin(2 * math.pi * i * 0.01 / period) for i in range(2000)]
                estimate, score, _ = NS['autocorr_peak'](signal, 0.010000000000001563)
                self.assertAlmostEqual(estimate, period, delta=0.011)
                self.assertGreater(score, 0.8)

    def test_boundary_maximum_is_not_a_cycle(self):
        signal = [math.exp(-i / 20) for i in range(2000)]
        mean = sum(signal) / len(signal)
        estimate, _, _ = NS['autocorr_peak']([v - mean for v in signal], 0.010000000000001563)
        self.assertTrue(math.isnan(estimate))

    def test_constant_has_no_period(self):
        self.assertTrue(math.isnan(NS['autocorr_peak']([0.0] * 2000, 0.01)[0]))


if __name__ == '__main__':
    unittest.main()
