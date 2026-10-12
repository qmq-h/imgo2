"""Data contract checks; no Isaac Sim or plotting libraries required."""
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / "scripts/tools/analyze_latent_tsne.py"
SPEC = importlib.util.spec_from_file_location("latent_tsne", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LatentInputs(unittest.TestCase):
    def test_phase_and_case_layouts_and_nocart_mask(self):
        entries = [
            {"case": {"connection": "rigid"}, "metrics": {"latent": {"tow": [1]*16}}},
            {"metrics": {"case": {"cart_present": False}, "latent": {"tow": [2]*16}}},
            {"metrics": {"case": {"cart_mass_kg": 10}, "latent": {"coast": [3]*16}}},
        ]
        vectors, records, skipped = MODULE.collect({"cases": entries}, "tow")
        self.assertEqual(vectors, [[1.0]*16])
        self.assertEqual(records[0]["labels"]["connection"], "rigid")
        self.assertEqual(skipped, {"nocart": 1, "missing_latent": 1, "invalid_latent": 0})
        self.assertEqual(len(MODULE.collect({"cases": entries}, "tow", include_nocart=True)[0]), 2)

    def test_rejects_nonfinite_and_wrong_dimensions(self):
        entries = [{"latent": vector} for vector in ([0]*15, [float("nan")]*16, [float("inf")]*16, ["bad"]*16)]
        vectors, _, skipped = MODULE.collect(entries, "tow", sample_mode=True)
        self.assertEqual(vectors, [])
        self.assertEqual(skipped["invalid_latent"], 4)

    def test_window_local_labels_preserved(self):
        entry = {"latent": [0]*16, "case": {"grade_deg": 10},
                 "local_slope_deg": -10, "segment": "down"}
        _, records, _ = MODULE.collect({"samples": [entry]}, "tow", sample_mode=True)
        self.assertEqual(records[0]["labels"]["grade_deg"], 10)
        self.assertEqual(records[0]["labels"]["local_slope_deg"], -10)
        self.assertEqual(records[0]["labels"]["segment"], "down")


if __name__ == "__main__":
    unittest.main()
