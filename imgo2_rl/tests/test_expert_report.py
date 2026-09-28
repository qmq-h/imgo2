"""`expert_report.py` 的离线测试（纯标准库 + numpy，不需要 GPU / Isaac Sim / torch）。

自造 npz（合成门控权重 + 地形列）来钉四件事：① 按地形的平均权重与"主责专家"；
② `argmax 占比` 才是分工证据（构造"普遍参与但不主责"的反例）；③ 熵与均匀基线的口径；
④ 边界（缺 `gate` ⇒ 退出码 1 且不抛异常、cmd 过滤、样本不足标记、单一地形组）。

Run: python3 -m unittest discover -s tests -p test_expert_report.py
"""

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "scripts" / "tools" / "expert_report.py"
_spec = importlib.util.spec_from_file_location("expert_report", TOOL)
report_mod = importlib.util.module_from_spec(_spec)
sys.modules["expert_report"] = report_mod
_spec.loader.exec_module(report_mod)

STEPS = 400
NAMES = ("flat", "boxes", "gap")


def _gate_for(owners, steps=STEPS, tail=0.05):
    """合成门控：每个环境由 `owners[env]` 指定主责专家（该专家 1-tail，其余均分 tail）。"""
    envs = len(owners)
    k = max(owners) + 1
    gate = np.full((steps, envs, k), tail / max(k - 1, 1), dtype=np.float64)
    for env, owner in enumerate(owners):
        gate[:, env, :] = tail / max(k - 1, 1)
        gate[:, env, owner] = 1.0 - tail
    return gate


def _save(path, gate, terrain_type, names=NAMES, cmd=1.0, level=None, contact=None):
    payload = {
        "gate": gate.astype(np.float32),
        "terrain_type": np.asarray(terrain_type, dtype=np.int32),
        "terrain_names": np.asarray(names, dtype="U"),
    }
    if cmd is not None:
        payload["cmd"] = np.tile(np.array([cmd, 0.0, 0.0], dtype=np.float32),
                                 (gate.shape[0], gate.shape[1], 1))
    if level is not None:
        payload["terrain_level"] = np.asarray(level, dtype=np.int32)
    if contact is not None:
        payload["contact"] = contact
    np.savez_compressed(path, **payload)
    return path


class TestExpertReport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_owner_is_the_dominant_expert_per_terrain(self):
        """每个地形列由一个专家主责 ⇒ 该列 mean_weight 最大者与 argmax 占比都应指向它。"""
        gate = _gate_for([0, 0, 3, 3, 4, 4])          # K=5
        path = _save(self.dir / "a.npz", gate, [0, 0, 1, 1, 2, 2])
        report = report_mod.build_report(report_mod.prepare(report_mod.load_npz(path)))
        owners = {group["terrain"]: (group["owner"], group["argmax_share"][group["owner"]])
                  for group in report["groups"]}
        self.assertEqual(owners["flat"][0], 0)
        self.assertEqual(owners["boxes"][0], 3)
        self.assertEqual(owners["gap"][0], 4)
        for terrain, (_owner, share) in owners.items():
            self.assertGreater(share, 0.95, f"{terrain} 的 argmax 占比应接近 1")
        self.assertEqual(report["num_experts"], 5)

    def test_argmax_share_separates_participation_from_ownership(self):
        """反例：两个专家权重都高、但只有一个是 argmax ⇒ 平均权重不区分"参与"和"主责"。"""
        gate = np.zeros((100, 2, 3), dtype=np.float64)
        gate[:, 0, :] = [0.45, 0.45, 0.10]           # 0 与 1 并列高，0 略胜
        gate[:, 1, :] = [0.40, 0.40, 0.20]
        path = _save(self.dir / "b.npz", gate, [0, 0])
        prepared = report_mod.prepare(report_mod.load_npz(path))
        group = report_mod.build_report(prepared)["groups"][0]
        # 两个专家的**平均权重相同**（都"普遍参与"）⇒ 平均权重区分不出主责
        self.assertAlmostEqual(group["mean_weight"][0], group["mean_weight"][1], places=3)
        self.assertAlmostEqual(group["mean_weight"][0], 0.425, places=3, msg="(0.45+0.40)/2")
        self.assertGreater(group["argmax_share"][0], 0.9, "argmax 占比才说明谁在负责这一列")
        self.assertLess(group["argmax_share"][1], 0.1)

    def test_entropy_matches_uniform_and_collapsed_limits(self):
        """均匀门控 ⇒ 熵 = lnK；塌缩到单专家 ⇒ 熵 ≈ 0。"""
        uniform = np.full((50, 2, 4), 0.25, dtype=np.float64)
        collapsed = np.zeros((50, 2, 4), dtype=np.float64)
        collapsed[:, :, 2] = 1.0
        report_u = report_mod.build_report(report_mod.prepare(
            report_mod.load_npz(_save(self.dir / "u.npz", uniform, [0, 0]))))
        report_c = report_mod.build_report(report_mod.prepare(
            report_mod.load_npz(_save(self.dir / "c.npz", collapsed, [0, 0]))))
        self.assertAlmostEqual(report_u["global_entropy"], float(np.log(4)), places=5)
        self.assertAlmostEqual(report_c["global_entropy"], 0.0, places=5)
        self.assertEqual(report_c["groups"][0]["owner"], 2)

    def test_cmd_filter_drops_idle_steps(self):
        """`--vx-min` 把"没在走"的步滤掉：静止时不应把门控分布算进去。"""
        gate = _gate_for([0, 1])
        gate[:200, :, :] = 0.5                        # 前半段：两专家均分（静止）
        gate[200:, :, :] = 0.0
        gate[200:, 0, 0] = 1.0                        # 后半段：静止段是"E0 独占"
        gate[200:, 1, 1] = 1.0
        cmd = np.zeros((STEPS, 2, 3), dtype=np.float32)
        cmd[200:, :, 0] = 1.0                         # 只有后半段在走
        path = _save(self.dir / "d.npz", gate, [0, 0], cmd=None)
        with np.load(path) as data:
            payload = {key: data[key] for key in data.files}
        payload["cmd"] = cmd
        np.savez_compressed(path, **payload)
        group = report_mod.build_report(report_mod.prepare(report_mod.load_npz(path)))["groups"][0]
        # `effective_steps` 是**环境×步**（400 = 200 步 × 2 环境）；过滤后 env0=E0、env1=E1
        self.assertEqual(group["effective_steps"], 400)
        self.assertAlmostEqual(group["mean_weight"][0], 0.5, places=5,
                               msg="静止段被滤掉后一半环境走 E0、一半走 E1")

    def test_by_name_groups_columns_of_the_same_terrain(self):
        """`--by-name`：同一地形的多列合并成一行（真实地形每类占连续若干列）。"""
        gate = _gate_for([0, 0, 3, 3])                  # 4 个环境：前两个 E0、后两个 E3
        path = _save(self.dir / "h.npz", gate, [0, 1, 4, 5],
                     names=("flat", "flat", "col2", "col3", "gap", "gap"))
        # 不给 --by-name：按列分组 ⇒ 4 组
        plain = report_mod.build_report(report_mod.prepare(report_mod.load_npz(path)))
        self.assertEqual(len(plain["groups"]), 4)
        # --by-name：flat/gap 各一组 ⇒ 2 组，且主责分别是 E0 / E3
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = report_mod.main(["--npz", str(path), "--by-name"])
        self.assertEqual(code, 0)
        rows = {line.split()[0]: line.split() for line in buffer.getvalue().splitlines()
                if line.strip() and not line.startswith(("#", "-")) and line.split()[0] in ("flat", "gap")}
        self.assertEqual(set(rows), {"flat", "gap"})
        self.assertEqual(rows["flat"][-2], "E0")
        self.assertEqual(rows["gap"][-2], "E3")

    def test_names_override_fixes_a_wrong_dump(self):
        """`--names` 覆盖错位的列名（旧 dump 的 11 键 vs 40 列问题）。"""
        gate = _gate_for([0, 4])
        path = _save(self.dir / "i.npz", gate, [0, 1], names=("flat", "gap"))
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            report_mod.main(["--npz", str(path), "--names", "gap,gap", "--by-name"])
        self.assertIn("gap", buffer.getvalue())
        self.assertNotIn("flat", buffer.getvalue())

    def test_missing_gate_is_a_clean_error(self):
        path = self.dir / "e.npz"
        np.savez_compressed(path, contact=np.zeros((10, 1, 4), dtype=bool),
                            terrain_names=np.asarray(NAMES, dtype="U"))
        buffer = io.StringIO()
        with contextlib.redirect_stderr(buffer):
            code = report_mod.main(["--npz", str(path)])
        self.assertEqual(code, 1)
        self.assertIn("gate", buffer.getvalue())
        with self.assertRaises(KeyError):
            report_mod.load_npz(path)

    def test_cli_prints_table_and_json_is_valid(self):
        gate = _gate_for([0, 0, 4])
        path = _save(self.dir / "f.npz", gate, [0, 0, 2], level=[6, 6, 6])
        out_json = self.dir / "out.json"
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = report_mod.main(["--npz", str(path), "--json", str(out_json)])
        self.assertEqual(code, 0)
        table = buffer.getvalue()
        self.assertIn("flat", table)
        self.assertIn("gap", table)
        report = json.loads(out_json.read_text(encoding="utf-8"))
        self.assertEqual(report["groups"][0]["terrain"], "flat")
        self.assertAlmostEqual(report["groups"][0]["level"], 6.0, places=6)
        self.assertNotIn("NaN", out_json.read_text(encoding="utf-8"))

    def test_small_sample_is_marked_not_fatal(self):
        gate = _gate_for([0])
        path = _save(self.dir / "g.npz", gate, [0])
        group = report_mod.build_report(report_mod.prepare(report_mod.load_npz(path)),
                                        min_steps=10_000)["groups"][0]
        self.assertEqual(group["status"], "insufficient")
        self.assertAlmostEqual(group["owner"], 0)


if __name__ == "__main__":
    unittest.main()
