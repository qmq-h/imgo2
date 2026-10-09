"""Run the reproducible stdlib-only terrain/episode checks without Isaac Lab."""

from pathlib import Path
import sys
import unittest

TESTS = Path(__file__).resolve().parents[2] / 'tests'
sys.path.insert(0, str(TESTS))
suite = unittest.TestSuite()
loader = unittest.TestLoader()
for module in ('test_towing_connection_grid', 'test_towing_slope_geometry'):
    suite.addTests(loader.loadTestsFromName(module))
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
