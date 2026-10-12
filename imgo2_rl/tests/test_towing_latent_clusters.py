"""聚类检验工具的数据契约与指标正确性检查；只用标准库，不需要 numpy/Isaac Sim。"""
import importlib.util
import json
import tempfile
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / "scripts/tools/analyze_latent_clusters.py"
SPEC = importlib.util.spec_from_file_location("latent_clusters", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def blobs():
    """三个互相分开的三维簇（确定性抖动，不用 numpy）。"""
    centres = ([0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0])
    points, labels = [], []
    for label, centre in enumerate(centres):
        for step in range(6):
            points.append([value + (step % 3) * 0.01 for value in centre])
            labels.append(label)
    return points, labels


class Metrics(unittest.TestCase):
    def test_adjusted_rand_index_known_values(self):
        self.assertAlmostEqual(MODULE.adjusted_rand_index([0, 0, 1, 1], [0, 0, 1, 1]), 1.0)
        self.assertAlmostEqual(MODULE.adjusted_rand_index([0, 0, 1, 1], [0, 1, 0, 1]), -0.5)
        # 退化情形与 scikit-learn 一致（两个单一簇记 1.0，一侧单一簇记 0.0）
        self.assertEqual(MODULE.adjusted_rand_index([0, 0, 0], [1, 1, 1]), 1.0)
        self.assertEqual(MODULE.adjusted_rand_index([0, 0, 0], [0, 1, 2]), 0.0)

    def test_normalized_mutual_information_bounds(self):
        self.assertAlmostEqual(MODULE.normalized_mutual_information([0, 1, 2], [0, 1, 2]), 1.0)
        self.assertEqual(MODULE.normalized_mutual_information([0, 0, 0], [1, 1, 1]), 1.0)
        self.assertEqual(MODULE.normalized_mutual_information([0, 0, 0], [0, 1, 2]), 0.0)
        self.assertAlmostEqual(MODULE.normalized_mutual_information([0, 0, 1, 1], [0, 1, 0, 1]), 0.0)

    def test_kmeans_recovers_separated_blobs(self):
        points, labels = blobs()
        assign, _, inertia = MODULE.kmeans(points, 3, seed=0)
        self.assertGreater(MODULE.adjusted_rand_index(labels, assign), 0.99)
        self.assertLess(inertia, 1.0)

    def test_silhouette_high_for_separated_clusters(self):
        points, labels = blobs()
        self.assertGreater(MODULE.silhouette(points, labels), 0.8)
        self.assertLess(MODULE.silhouette(points, [i % 2 for i in range(len(points))]), 0.3)

    def test_knn_and_centroid_leave_one_out(self):
        points, labels = blobs()
        accuracy, majority = MODULE.knn_loo(points, labels, k=5)
        centroid, _, _ = MODULE.nearest_centroid_loo(points, labels)
        self.assertEqual(accuracy, 1.0)
        self.assertEqual(centroid, 1.0)
        self.assertAlmostEqual(majority, 1 / 3)

    def test_permutation_pvalue_is_deterministic_and_sensitive(self):
        points, labels = blobs()
        real = MODULE.permutation_pvalue(points, labels, MODULE.multivariate_eta2,
                                        permutations=50, seed=7)
        again = MODULE.permutation_pvalue(points, labels, MODULE.multivariate_eta2,
                                         permutations=50, seed=7)
        self.assertEqual(real, again)
        self.assertLess(real[2], 0.05)
        shuffled_labels = [0, 1, 2] * 6
        noise = MODULE.permutation_pvalue(points, shuffled_labels, MODULE.multivariate_eta2,
                                          permutations=50, seed=7)
        self.assertGreater(noise[2], 0.5)


class InputSelection(unittest.TestCase):
    def _payload(self):
        cases = []
        for index in range(12):
            case = {"connection": ("rigid" if index % 3 == 0 else "compliant"),
                    "cart_mass_kg": 5.0 if index % 2 else 10.0,
                    "cart_present": index % 4 != 3, "row": index // 2, "column": index}
            cases.append({"case": case, "metrics": {"case": case, "latent": {"tow": [float(index)] * 16},
                                                    "estimator": {"abs_err": {"tow": {"mass_kg": index * 0.5}}}}})
        return {"cases": cases}

    def test_nocart_rows_and_filters(self):
        vectors, cases, errors, skipped = MODULE.load_cases(
            self._payload(), "tow", include_nocart=False, filters={}, rows=None)
        self.assertEqual(len(vectors), 9)
        self.assertEqual(skipped["nocart"], 3)
        self.assertTrue(all(case["cart_present"] for case in cases))
        self.assertEqual(errors[0], 0.0)
        subset, _, _, skipped_rows = MODULE.load_cases(
            self._payload(), "tow", include_nocart=False, filters={"connection": "rigid"}, rows=(3, 4))
        self.assertEqual(len(subset), 2)  # row 3/4 的 rigid 是 index 6 与 9（index 3 是无车样本）
        self.assertGreater(skipped_rows["filtered"], 0)
        everything, _, _, _ = MODULE.load_cases(
            self._payload(), "tow", include_nocart=True, filters={}, rows=None)
        self.assertEqual(len(everything), 12)

    def test_rejects_bad_vectors(self):
        payload = {"cases": [{"case": {"cart_present": True}, "metrics": {"latent": {"tow": vector}}}
                             for vector in ([0.0] * 15, [float("nan")] * 16, ["x"] * 16)]}
        vectors, _, _, skipped = MODULE.load_cases(payload, "tow", include_nocart=False,
                                                   filters={}, rows=None)
        self.assertEqual(vectors, [])
        self.assertEqual(skipped["invalid_latent"], 3)


class EndToEnd(unittest.TestCase):
    def _report(self, count=24):
        centres = {"compliant": 0.0, "inextensible": 5.0, "rigid": 10.0}
        cases = []
        for index in range(count):
            connection = ("compliant", "inextensible", "rigid")[index % 3]
            mass = (5.0, 10.0, 15.0, 20.0)[index % 4]
            vector = [centres[connection] + mass * 0.01] + [0.0] * 15
            case = {"connection": connection, "cart_mass_kg": mass, "grade_deg": 0.0,
                    "velocity_mps": 1.0, "cart_present": True, "row": index % 20,
                    "column": index % 40}
            cases.append({"case": case, "metrics": {"case": case, "latent": {"tow": vector}}})
        return {"cases": cases}

    def test_writes_metrics_and_flags_binary_grouping(self):
        with tempfile.TemporaryDirectory() as folder:
            report = Path(folder) / "report.json"
            report.write_text(json.dumps(self._report()), encoding="utf-8")
            out = Path(folder) / "out"
            code = MODULE.main(["--report", str(report), "--phase", "tow", "--output", str(out),
                                "--connection-binary", "--permutations", "20"])
            self.assertEqual(code, 0)
            metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["selected"], 24)
            self.assertIn("connection_binary", metrics["labels"])
            self.assertIn("connection", metrics["clustering"][0]["target"])
            self.assertGreater(metrics["labels"]["connection"]["centroid_loo"], 0.9)
            self.assertTrue((out / "points.csv").is_file())

    def test_requires_enough_samples(self):
        with tempfile.TemporaryDirectory() as folder:
            report = Path(folder) / "report.json"
            report.write_text(json.dumps(self._report(count=6)), encoding="utf-8")
            code = MODULE.main(["--report", str(report), "--phase", "tow",
                                "--output", str(Path(folder) / "out")])
            self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
