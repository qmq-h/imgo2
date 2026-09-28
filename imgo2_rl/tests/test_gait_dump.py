"""`gait_dump.py` + `cmoe/play.py` 的步态 dump 契约测试（纯离线，不需要 GPU / Isaac Sim）。

链路：**合成接触序列 → `GaitDumper` 写出 npz → `gait_report.py` 读回并判定**。
这条链路是 2026-09-28 加"步态量测"时的关键风险点：Isaac 侧取数（play.py 的胶水）无法在无 GPU
机器上跑，但**契约**（键名/形状/足序/列名）可以在这里钉死 —— 一旦 play.py 写歪了键或足序，
本文件的端到端断言就会红。

Run: python3 -m unittest discover -s tests -p test_gait_dump.py
"""

import contextlib
import io
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO / "scripts" / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

# `rl_lab.utils` 的包 `__init__` 会拉进 isaaclab（本机没有 omni.log）⇒ 按**文件路径**加载本模块
# （它只依赖 numpy，可以直接 exec；与 tests/test_cmoe_expert_init.py 的私有命名空间做法同一目的）。
_spec_dump = importlib.util.spec_from_file_location(
    "gait_dump_under_test", REPO / "scripts" / "rl_lab" / "rl_lab" / "utils" / "gait_dump.py"
)
gait_dump = importlib.util.module_from_spec(_spec_dump)
sys.modules["gait_dump_under_test"] = gait_dump
_spec_dump.loader.exec_module(gait_dump)

FOOT_ORDER = gait_dump.FOOT_ORDER
GaitDumper = gait_dump.GaitDumper
canonical_foot_indices = gait_dump.canonical_foot_indices
format_gait_summary = gait_dump.format_gait_summary

# `gait_report.py` 按文件路径加载（它自己会去 sys.path 找 gait_kernel_probe）
_spec = importlib.util.spec_from_file_location("gait_report", TOOLS_DIR / "gait_report.py")
report_mod = importlib.util.module_from_spec(_spec)
sys.modules["gait_report"] = report_mod
_spec.loader.exec_module(report_mod)

DT = 0.02
PERIOD = 0.6      # 30 步整周期（相位分辨率 1/30）
DUTY = 0.5
STEPS = 900


def synthetic_contact(gait: str, steps: int = STEPS, envs: int = 1) -> np.ndarray:
    """按 `gait_offsets()` 的相位模型合成 `contact[steps, envs, 4]`（足序 FL, FR, RL, RR）。"""
    offsets = report_mod.gait_offsets(gait)
    frames = np.zeros((steps, envs, len(FOOT_ORDER)), dtype=bool)
    for env in range(envs):
        for step in range(steps):
            time = step * DT
            for foot_id, foot in enumerate(FOOT_ORDER):
                frames[step, env, foot_id] = ((time / PERIOD + offsets[foot]) % 1.0) < DUTY
    return frames


class TestCanonicalFootOrder(unittest.TestCase):
    def test_maps_isaac_body_names_to_contract_order(self):
        """传感器给的是打乱的全 body 列表 ⇒ 必须按 FL,FR,RL,RR 抽出来（不是按出现顺序）。"""
        names = ["base", "RR_FOOT", "FR_THIGH", "FL_FOOT", "RL_SHANK", "FR_FOOT", "RL_FOOT"]
        indices, picked = canonical_foot_indices(names)
        self.assertEqual(picked, ["FL_FOOT", "FR_FOOT", "RL_FOOT", "RR_FOOT"])
        self.assertEqual([names[i] for i in indices], picked)

    def test_accepts_short_and_lowercase_names(self):
        for names in (["fl", "fr", "rl", "rr"], ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]):
            _indices, picked = canonical_foot_indices(names)
            self.assertEqual([name.lower()[:2] for name in picked], ["fl", "fr", "rl", "rr"])

    def test_missing_foot_raises_so_caller_can_fail_soft(self):
        with self.assertRaises(ValueError):
            canonical_foot_indices(["FL_FOOT", "FR_FOOT", "RL_FOOT"])


class TestExpandTerrainNames(unittest.TestCase):
    """列名铺开：`sub_terrains` 是"每类占连续若干列"，**不能**直接把 keys[i] 当第 i 列。

    2026-09-28 真实 dump 踩过：11 个键 vs 40 列 ⇒ 列 11 之后全显示成「列N」。
    规则逐字取自 Isaac Lab `terrain_generator.py:233-241`。
    """

    def test_equal_proportions_split_evenly(self):
        keys = [f"t{i}" for i in range(10)]
        names = gait_dump.expand_terrain_names(keys, [0.1] * 10, 40)
        self.assertEqual(len(names), 40)
        self.assertEqual([names.count(k) for k in keys], [4] * 10)
        self.assertEqual(names[:4], ["t0"] * 4)
        self.assertEqual(names[36:], ["t9"] * 4)

    def test_real_cmoe_proportions_reproduce_observed_blocks(self):
        """CMoE 真实比例（gap 0.30、三个 0.05）⇒ 与真实 dump 观察到的块长一致。"""
        keys = ["pyramid_stairs", "pyramid_stairs_inv", "boxes", "random_rough",
                "hf_pyramid_slope", "hf_pyramid_slope_inv", "gap", "hurdle", "mix",
                "narrow_stairs", "flat"]
        props = [0.10, 0.10, 0.10, 0.05, 0.05, 0.05, 0.30, 0.10, 0.10, 0.10, 0.10]
        names = gait_dump.expand_terrain_names(keys, props, 40)
        counts = {key: names.count(key) for key in keys}
        self.assertEqual(len(names), 40)
        self.assertEqual(counts["gap"], 11, "gap 占 0.30 应拿到最多列")
        self.assertEqual(counts["pyramid_stairs"], 4)
        self.assertEqual(counts["boxes"], 4)
        self.assertEqual(counts["random_rough"], 2)
        self.assertEqual(counts["hf_pyramid_slope"], 1)
        self.assertEqual(names[16:27], ["gap"] * 11, "gap 是连续块（16–26）")
        self.assertEqual(names[37:], ["flat"] * 3)

    def test_single_terrain_covers_every_column(self):
        self.assertEqual(gait_dump.expand_terrain_names(["flat"], [1.0], 5), ["flat"] * 5)

    def test_bad_proportions_raise(self):
        with self.assertRaises(ValueError):
            gait_dump.expand_terrain_names(["a", "b"], [0.5], 4)
        with self.assertRaises(ValueError):
            gait_dump.expand_terrain_names(["a"], [0.0], 4)


class TestGaitDumperSchema(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "gait.npz"

    def tearDown(self):
        self.tmp.cleanup()

    def test_dump_round_trip_and_terrain_mode(self):
        """写出后键名/形状/类型符合契约；`terrain_level` 取窗口内众数（reset 会改等级）。"""
        dumper = GaitDumper(self.path, dt=DT, terrain_names=("flat", "boxes", "gap"))
        contact = synthetic_contact("trot", steps=60, envs=2)
        for step in range(60):
            level = 3 if step < 40 else 5          # 模拟窗口内晋级
            dumper.record(
                contact[step],
                base_height=np.full(2, 0.4, dtype=np.float32),
                cmd=np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (2, 1)),
                base_lin_vel=np.zeros((2, 3), dtype=np.float32),
                terrain_type=np.array([0, 2], dtype=np.int64),
                terrain_level=np.array([level, level], dtype=np.int64),
            )
        summary = dumper.save()
        self.assertEqual(summary["steps"], 60)
        self.assertEqual(summary["num_envs"], 2)
        data = np.load(self.path)
        self.assertEqual(data["contact"].shape, (60, 2, 4))
        self.assertEqual(data["contact"].dtype, np.bool_)
        self.assertEqual(data["cmd"].shape, (60, 2, 3))
        self.assertEqual(data["terrain_type"].tolist(), [0, 2])
        self.assertEqual(data["terrain_level"].tolist(), [3, 3], "众数：40 步 3 级 vs 20 步 5 级")
        self.assertEqual(list(data["terrain_names"]), ["flat", "boxes", "gap"])
        self.assertEqual(data["foot_names"].tolist(), list(FOOT_ORDER))
        self.assertAlmostEqual(float(data["dt"]), DT, places=6)

    def test_save_is_idempotent_and_requires_steps(self):
        dumper = GaitDumper(self.path, dt=DT)
        with self.assertRaises(RuntimeError):
            dumper.save()                                    # 一步都没记 ⇒ 不写空文件
        dumper.record(np.zeros((2, 4), dtype=bool))
        first = dumper.save()
        second = dumper.save()
        self.assertEqual(first, second, "重复 save 应返回同一 summary")
        self.assertTrue(self.path.exists())

    def test_missing_optional_fields_still_writes_contact_only(self):
        """只给 contact（连地形都没给）也必须能落盘 —— 报告侧 fail-soft 退化。"""
        dumper = GaitDumper(self.path, dt=DT)
        for step in range(10):
            dumper.record(np.ones((3, 4), dtype=bool))
        summary = dumper.save()
        self.assertEqual(summary["keys"], ["contact", "dt", "terrain_names", "foot_names"])
        data = np.load(self.path)
        self.assertNotIn("cmd", data.files)
        report = report_mod.build_report(report_mod.prepare_arrays(dict(data), lambda *_: None))
        self.assertEqual(len(report["groups"]), 1, "没有 terrain_type 时退化成单组而不是报错")

    def test_shape_mismatch_is_dropped_not_fatal(self):
        dumper = GaitDumper(self.path, dt=DT)
        dumper.record(np.zeros((2, 4), dtype=bool))
        dumper.record(np.zeros((3, 4), dtype=bool))           # 环境数变了 ⇒ 丢这一步
        dumper.record(np.zeros((2, 4), dtype=bool))
        summary = dumper.save()
        self.assertEqual(summary["steps"], 2)
        self.assertEqual(summary["shape_warnings"], 1)

    def test_wrong_foot_count_raises(self):
        dumper = GaitDumper(self.path, dt=DT)
        with self.assertRaises(ValueError):
            dumper.record(np.zeros((2, 3), dtype=bool))

    def test_summary_line_mentions_path_and_report_hint(self):
        dumper = GaitDumper(self.path, dt=DT, terrain_names=("flat",))
        dumper.record(np.zeros((1, 4), dtype=bool))
        text = format_gait_summary(dumper.save(), report_hint="python3 scripts/tools/gait_report.py --npz x")
        self.assertIn("已写出", text)
        self.assertIn("gait_report.py", text)


class TestDumpToReportEndToEnd(unittest.TestCase):
    """**最关键的一条**：dump 出来的文件必须能被报告工具按地形正确判定步态。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "mixed.npz"

    def tearDown(self):
        self.tmp.cleanup()

    def _dump(self, per_env_gait, terrain_type, levels, names, envs=6):
        """每个环境跑一种步态；同一地形列里放两个环境（检验跨环境圆周平均）。"""
        dumper = GaitDumper(self.path, dt=DT, terrain_names=names)
        contacts = [synthetic_contact(gait, steps=STEPS, envs=len(terrain_type))
                    for gait in per_env_gait]  # 一次算一个步态，稍后按环境拼
        # 重新按环境生成：直接用每环境的步态名
        full = np.zeros((STEPS, envs, 4), dtype=bool)
        for env, gait in enumerate(per_env_gait):
            full[:, env, :] = synthetic_contact(gait, steps=STEPS, envs=1)[:, 0, :]
        del contacts
        for step in range(STEPS):
            dumper.record(
                full[step],
                cmd=np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (envs, 1)),
                terrain_type=np.asarray(terrain_type, dtype=np.int32),
                terrain_level=np.asarray(levels, dtype=np.int32),
            )
        return dumper.save()

    def test_per_terrain_verdicts_match_the_synthetic_gaits(self):
        # 三列：flat 上两个环境都 trot；gap 上两个环境都 bound；boxes 上两个环境都 pace
        per_env = ["trot", "trot", "bound", "bound", "pace", "pace"]
        terrain_type = [0, 0, 2, 2, 1, 1]
        names = ("flat", "boxes", "gap")
        summary = self._dump(per_env, terrain_type, [3] * 6, names)
        self.assertEqual(summary["keys"][0], "contact")

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = report_mod.main(["--npz", str(self.path), "--min-steps", "100"])
        table = buffer.getvalue()
        self.assertEqual(code, 0)

        # 每个地形一组，且判定与合成步态一致（含 名称→列索引 的映射方向）
        by_name = {}
        for line in table.splitlines():
            cells = line.split()
            if len(cells) >= 3 and cells[0] in names:
                by_name[cells[0]] = cells
        self.assertEqual(set(by_name), set(names), f"应出现三个地形组，实际：\n{table}")
        self.assertEqual(by_name["flat"][-2], "trot", table)
        self.assertEqual(by_name["gap"][-2], "bound", table)
        self.assertEqual(by_name["boxes"][-2], "pace", table)

    def test_pinned_terrain_level_is_reported(self):
        """--by-level：等级被钉死时表里应显示该等级（play.py --terrain_level 的用法）。"""
        self._dump(["trot"] * 2, [0, 0], [7, 7], ("flat",), envs=2)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            report_mod.main(["--npz", str(self.path), "--min-steps", "100", "--by-level"])
        self.assertIn("7", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
