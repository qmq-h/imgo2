"""`gait_report.py`（按地形分组的离线步态报告）的回归测试。

全部离线：自己用 ``numpy.savez_compressed`` 往 ``tempfile`` 目录写合成 npz，
不依赖仓库里的训练产物、不需要 GPU / Isaac Lab / torch。

锁定的是**口径**，不是实现细节：

* 合成 trot/bound/pace/lockstep 必须被判成它们自己，且 6 对相位与
  ``gait_report.reference_phase_deg()`` 的对应行**逐对相等**——这条不变量同时钉住了
  「相位定义」「参考模式表」「残差用圆周距离」三件事；
* 已知周期的方波 period 误差 < 15%（并锁定「四足之和给半周期」这个反例，防止有人把
  `--period-source` 默认值改成 sum）；
* 边界不抛异常：样本不足 / 全腾空 / 缺字段（老 npz）/ 缺 `contact` / 文件不存在；
* `--vx-min` 真的在过滤低指令步。
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "imgo2_rl" / "scripts" / "tools" / "gait_report.py"

# 工具用 `sys.path.insert` 引 `gait_kernel_probe`（FEET / gait_offsets），
# 所以先把 tools 目录放到路径上，再直接按文件路径加载本工具（与 test_gait_kernel_probe.py 同风格）。
TOOLS_DIR = TOOL_PATH.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

_spec = importlib.util.spec_from_file_location("gait_report", TOOL_PATH)
report_mod = importlib.util.module_from_spec(_spec)
sys.modules["gait_report"] = report_mod
_spec.loader.exec_module(report_mod)

DT = 0.02
PERIOD = 0.6      # 30 步整周期：合成数据的参考周期
DUTY = 0.5
STEPS = 800


# ---------------------------------------------------------------------------
# 合成数据
# ---------------------------------------------------------------------------

def synth_contact(gait: str, steps: int = STEPS, envs: int = 2, period: float = PERIOD,
                  duty: float = DUTY, dt: float = DT) -> np.ndarray:
    """按 `gait_offsets()` 的相位模型合成 ``contact[T, N, 4]``（足序 FL, FR, RL, RR）。"""
    offsets = report_mod.gait_offsets(gait)
    frames = np.zeros((steps, envs, 4), dtype=bool)
    for env in range(envs):
        for step in range(steps):
            time = step * dt
            for foot_id, foot in enumerate(report_mod.FEET):
                frames[step, env, foot_id] = ((time / period + offsets[foot]) % 1.0) < duty
    return frames


def save_npz(path: Path, contact: np.ndarray, terrain_type=None, terrain_level=None,
             terrain_names=("flat",), cmd_speed: float = 1.0, dt: float = DT,
             with_optional: bool = True) -> Path:
    """按 play.py 的 schema 写一个合成 npz；``with_optional=False`` 时只写 `contact`。"""
    steps, envs, feet = contact.shape
    payload = {"contact": contact}
    if with_optional:
        payload.update({
            "base_height": np.full((steps, envs), 0.31, dtype=np.float32),
            "cmd": np.tile(np.asarray([cmd_speed, 0.0, 0.0], dtype=np.float32), (steps, envs, 1)),
            "base_lin_vel": np.tile(np.asarray([cmd_speed, 0.0, 0.0], dtype=np.float32),
                                    (steps, envs, 1)),
            "terrain_type": np.asarray(terrain_type if terrain_type is not None else [0] * envs,
                                       dtype=np.int32),
            "terrain_level": np.asarray(terrain_level if terrain_level is not None else [0] * envs,
                                        dtype=np.int32),
            "terrain_names": np.asarray(list(terrain_names)),
            "dt": np.float32(dt),
            "foot_names": np.asarray([f"{foot}_FOOT" for foot in report_mod.FEET]),
        })
    np.savez_compressed(path, **payload)
    return path


def square_wave(period_steps: int, steps: int = 900, duty: float = 0.5) -> np.ndarray:
    """已知周期的方波（0/1），period_steps 精确整步。"""
    return np.asarray([1.0 if (step % period_steps) < period_steps * duty else 0.0
                       for step in range(steps)])


def analyze(arrays: dict, **kwargs) -> dict:
    """`prepare_arrays` + `build_report` 的便捷封装。"""
    return report_mod.build_report(arrays, **kwargs)


class GaitReportCase(unittest.TestCase):
    """公共夹具：临时目录 + 从 npz 到「单组报告」的短路径。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def load(self, path: Path, warn=None) -> dict:
        return report_mod.prepare_arrays(report_mod.load_npz(path), warn or (lambda _msg: None))

    def first_group(self, contact: np.ndarray, **kwargs) -> dict:
        """contact → 单组报告（默认全部可选字段齐全、地形为单列 flat）。"""
        path = save_npz(self.tmp / "single.npz", contact)
        report = analyze(self.load(path), **kwargs)
        self.assertEqual(len(report["groups"]), 1, report["groups"])
        return report["groups"][0]

    def phases_of(self, group: dict) -> dict:
        """组的 6 对相位 → ``{"FL-FR": 0.5, ...}``。"""
        self.assertIsNotNone(group["metrics"], f"组没有算出指标：{group['notes']}")
        return group["metrics"]["phase"]

    def assert_verdict(self, group: dict, expected: str) -> None:
        self.assertIsNotNone(group["metrics"], group["notes"])
        self.assertEqual(group["metrics"]["verdict"], expected,
                         f"{group['metrics']['verdict_reason']}；残差 {group['metrics']['residuals']}")

    def assert_phases_match_reference(self, group: dict, gait: str) -> None:
        """实测的 6 对相位必须等于参考表的对应行（容差 0.05 周期 = 1.5 步）。"""
        measured = self.phases_of(group)
        reference = report_mod.reference_phase_deg()[gait]
        for pair in report_mod.PAIRS:
            key = f"{report_mod.FEET[pair[0]]}-{report_mod.FEET[pair[1]]}"
            self.assertLess(report_mod.circular_distance(measured[key], reference[pair]), 0.05,
                            f"{gait} 的 {key} 相位 {measured[key]:.3f} != 参考 {reference[pair]:.3f}")


# ---------------------------------------------------------------------------
# 参考模式表自身
# ---------------------------------------------------------------------------

class TestReferencePhases(GaitReportCase):
    """参考表是判定基准，先把它钉住：数值必须就是 task 里写的那组关系。"""

    def test_reference_table_is_the_canonical_pairing(self):
        reference = report_mod.reference_phase_deg()
        for gait in report_mod.REFERENCE_GAITS:
            self.assertIn(gait, reference)

        def get(gait, a, b):
            return reference[gait][(report_mod.FEET.index(a), report_mod.FEET.index(b))]

        # trot：对角对同相、其余反相
        for a, b in (("FL", "RR"), ("FR", "RL")):
            self.assertLess(report_mod.circular_distance(get("trot", a, b), 0.0), 1.0e-9)
        for a, b in (("FL", "FR"), ("FL", "RL"), ("FR", "RR"), ("RL", "RR")):
            self.assertLess(report_mod.circular_distance(get("trot", a, b), 0.5), 1.0e-9)
        # bound：前沿对与后沿对同相
        for a, b in (("FL", "FR"), ("RL", "RR")):
            self.assertLess(report_mod.circular_distance(get("bound", a, b), 0.0), 1.0e-9)
        # pace：同侧对同相
        for a, b in (("FL", "RL"), ("FR", "RR")):
            self.assertLess(report_mod.circular_distance(get("pace", a, b), 0.0), 1.0e-9)
        # lockstep：四足全同相
        for pair in report_mod.PAIRS:
            self.assertLess(report_mod.circular_distance(reference["lockstep"][pair], 0.0), 1.0e-9)

    def test_circular_distance_wraps(self):
        self.assertAlmostEqual(report_mod.circular_distance(0.95, 0.05), 0.1, places=9)
        self.assertAlmostEqual(report_mod.circular_distance(0.0, 0.5), 0.5, places=9)
        self.assertAlmostEqual(report_mod.circular_distance(0.5, 0.0), 0.5, places=9)
        self.assertAlmostEqual(report_mod.circular_distance(0.4, 0.6), 0.2, places=9)


# ---------------------------------------------------------------------------
# 判定：四种合成步态
# ---------------------------------------------------------------------------

class TestSyntheticGaits(GaitReportCase):
    def test_synthetic_trot(self):
        """trot：判定 trot，对角对相位 ≈ 0.0、FL-FR ≈ 0.5，且与参考表逐对相等。"""
        group = self.first_group(synth_contact("trot", envs=2))
        self.assert_verdict(group, "trot")
        self.assert_phases_match_reference(group, "trot")
        phases = self.phases_of(group)
        self.assertAlmostEqual(phases["FL-FR"], 0.5, delta=0.05)
        self.assertAlmostEqual(phases["FL-RL"], 0.5, delta=0.05)
        self.assertAlmostEqual(phases["FL-RR"], 0.0, delta=0.05)
        self.assertAlmostEqual(phases["FR-RL"], 0.0, delta=0.05)
        # 残差必须真的分得开：trot 自己 ≈ 0，其它模式远大于它
        residuals = group["metrics"]["residuals"]
        self.assertLess(residuals["trot"], 0.05)
        for other in ("bound", "pace", "lockstep"):
            self.assertGreater(residuals[other], 0.2)

    def test_synthetic_bound(self):
        group = self.first_group(synth_contact("bound", envs=2))
        self.assert_verdict(group, "bound")
        self.assert_phases_match_reference(group, "bound")
        phases = self.phases_of(group)
        self.assertAlmostEqual(phases["FL-FR"], 0.0, delta=0.05)     # 前沿对同相
        self.assertAlmostEqual(phases["RL-RR"], 0.0, delta=0.05)     # 后沿对同相
        self.assertAlmostEqual(phases["FL-RL"], 0.5, delta=0.05)     # 前后反相

    def test_synthetic_pace(self):
        group = self.first_group(synth_contact("pace", envs=2))
        self.assert_verdict(group, "pace")
        self.assert_phases_match_reference(group, "pace")
        phases = self.phases_of(group)
        self.assertAlmostEqual(phases["FL-RL"], 0.0, delta=0.05)     # 同侧对同相
        self.assertAlmostEqual(phases["FR-RR"], 0.0, delta=0.05)
        self.assertAlmostEqual(phases["FL-FR"], 0.5, delta=0.05)     # 异侧前沿反相

    def test_lockstep_is_lockstep_or_unclear_but_never_trot(self):
        """四足完全同相：判定要么 lockstep、要么 `无明确模式`（相位差全 0，任何配对都不同相）；
        但绝不能判成 trot/bound/pace —— 它们的参考表里至少有一对是 0.5。"""
        group = self.first_group(synth_contact("lockstep", envs=2))
        self.assert_verdict(group, "lockstep")
        self.assertIn(group["metrics"]["verdict"], ("lockstep", "无明确模式"))
        phases = self.phases_of(group)
        for key, value in phases.items():
            self.assertLess(report_mod.circular_distance(value, 0.0), 0.05, f"{key} 应为同相")

    def test_cross_env_phase_averaging_is_circular(self):
        """跨环境做的是**圆周**平均：两环境相位差 0.05 时均值应靠近 0.0，而不是被拉到 0.5。"""
        base = synth_contact("trot", envs=1)
        shifted = np.roll(base, 1, axis=0)           # 整体平移 1 步（≈0.033 周期）
        contact = np.concatenate([base, shifted], axis=1)
        group = self.first_group(contact)
        self.assert_verdict(group, "trot")
        self.assertAlmostEqual(self.phases_of(group)["FL-FR"], 0.5, delta=0.05)
        for value in group["metrics"]["phase_concentration"].values():
            self.assertGreater(value, 0.9, "平移 1 步后集中度仍应接近 1")


# ---------------------------------------------------------------------------
# 周期估计
# ---------------------------------------------------------------------------

class TestPeriod(GaitReportCase):
    def test_known_square_wave_within_15_percent(self):
        for period_steps in (10, 17, 30, 50):
            with self.subTest(period_steps=period_steps):
                expected = period_steps * DT
                estimated = report_mod.period_from_autocorr(square_wave(period_steps), DT)
                self.assertTrue(np.isfinite(estimated), "应找到显著峰")
                self.assertLess(abs(estimated - expected) / expected, 0.15,
                                f"period 估计 {estimated} 偏离 {expected} 超过 15%")

    def test_trot_contact_period_through_the_full_pipeline(self):
        group = self.first_group(synth_contact("trot", envs=2, period=PERIOD))
        self.assertTrue(group["metrics"]["period_valid"])
        self.assertLess(abs(group["metrics"]["period_s"] - PERIOD) / PERIOD, 0.15)

    def test_four_foot_sum_breaks_the_period(self):
        """反例锁定「四足之和」这个口径：trot 的**和**一个周期内有两个脉冲，自相关峰落在 T/2
        （duty 偏离 0.5 时读成 0.3 s）；duty=0.5 时和恒为 2，自相关完全平坦 ⇒ 干脆没有显著峰。
        两种情况下它都给不出真实周期，所以默认是 `--period-source foot0`。"""
        for duty in (0.3, 0.5):
            with self.subTest(duty=duty):
                contact = synth_contact("trot", envs=1, period=PERIOD, duty=duty)
                foot0 = report_mod.estimate_period(contact, DT, source="foot0")[0]
                summed = report_mod.estimate_period(contact, DT, source="sum")[0]
                self.assertLess(abs(foot0 - PERIOD) / PERIOD, 0.15, "单足必须给出真实周期")
                broken = (not np.isfinite(summed)) or abs(summed - PERIOD) / PERIOD > 0.3
                self.assertTrue(broken, f"四足之和不应给出真实周期，却给了 {summed}")

    def test_no_peak_gives_nan(self):
        """整段不触地 / 纯随机都没有显著峰 ⇒ NaN（不是 0，也不是异常）。"""
        self.assertFalse(np.isfinite(report_mod.period_from_autocorr(np.zeros(400), DT)))
        rng = np.random.default_rng(0)
        self.assertFalse(np.isfinite(report_mod.period_from_autocorr(rng.random(400) > 0.5, DT)))
        self.assertFalse(np.isfinite(report_mod.period_from_autocorr(np.ones(400), DT)))


# ---------------------------------------------------------------------------
# 边界：样本不足 / 全腾空 / 缺字段
# ---------------------------------------------------------------------------

class TestEdges(GaitReportCase):
    def test_insufficient_steps_is_labelled_and_skipped(self):
        group = self.first_group(synth_contact("trot", steps=60, envs=1), min_steps=200)
        self.assertEqual(group["status"], "insufficient")
        self.assertIsNone(group["metrics"])
        self.assertEqual(group["steps_valid"], 60)
        self.assertTrue(any("样本不足" in note for note in group["notes"]))
        # 打印路径也不能抛
        self.assertIn("样本不足", report_mod.format_report(
            analyze(self.load(save_npz(self.tmp / "few.npz", synth_contact("trot", steps=60, envs=1))),
                    min_steps=200)))

    def test_all_feet_airborne_is_safe(self):
        group = self.first_group(np.zeros((400, 2, 4), dtype=bool), min_steps=100)
        self.assertEqual(group["status"], "no_contact")
        self.assertIsNone(group["metrics"])
        self.assertTrue(any("整段未触地" in note for note in group["notes"]))
        text = report_mod.format_report(
            analyze(self.load(save_npz(self.tmp / "air.npz", np.zeros((400, 2, 4), dtype=bool))),
                    min_steps=100))
        self.assertIn("全腾空", text)

    def test_one_foot_never_lands_gives_nan_phase(self):
        """只有 FL 落地：FL 自己的周期还能算，但涉及其它足的相位必须是 NaN + 标注。"""
        contact = synth_contact("trot", envs=1)
        contact[:, :, 1:] = False
        group = self.first_group(contact, min_steps=100)
        self.assertIsNotNone(group["metrics"])
        self.assertFalse(group["metrics"]["duty"][1] == group["metrics"]["duty"][1])   # FR duty = NaN
        self.assertFalse(np.isfinite(group["metrics"]["phase"]["FL-FR"]))
        self.assertFalse(np.isfinite(group["metrics"]["phase"]["FL-RL"]))
        self.assertEqual(group["metrics"]["verdict"], "无法判定")
        self.assertTrue(any("整段不触地" in note for note in group["notes"]))

    def test_legacy_npz_without_optional_keys(self):
        """老 npz 只有 `contact`：不抛异常，退化成单组 + dt 默认值 + 不过滤，指标仍算得出。"""
        path = save_npz(self.tmp / "legacy.npz", synth_contact("trot", envs=1), with_optional=False)
        group = self.first_group(synth_contact("trot", envs=1))  # 对照：完整 schema
        arrays = self.load(path)
        self.assertEqual(arrays["dt"], report_mod.DEFAULT_DT)
        self.assertIsNone(arrays["cmd"])
        report = analyze(arrays, min_steps=100)
        self.assertEqual(len(report["groups"]), 1)
        legacy = report["groups"][0]
        self.assertEqual(legacy["steps_valid"], legacy["steps_total"], "缺 cmd 时应全部计入")
        self.assert_verdict(legacy, "trot")
        self.assertAlmostEqual(legacy["metrics"]["period_s"], group["metrics"]["period_s"], delta=0.02)
        self.assertGreaterEqual(len(arrays["notes"]), 2, "缺字段必须留下告警")

    def test_missing_contact_raises_schema_error(self):
        path = self.tmp / "no_contact.npz"
        np.savez_compressed(path, dt=np.float32(DT))
        with self.assertRaises(report_mod.SchemaError):
            report_mod.prepare_arrays(report_mod.load_npz(path), lambda _msg: None)

    def test_mismatched_shapes_do_not_crash(self):
        """cmd/terrain_type 长度写错时退化为「不过滤 / 单组」，不抛异常。"""
        path = save_npz(self.tmp / "bad.npz", synth_contact("trot", envs=2),
                        terrain_type=[0, 0, 0], terrain_level=[0, 0, 0])
        arrays = self.load(path)
        self.assertTrue((arrays["terrain_index"] == -1).all(), "长度不匹配 ⇒ 单组")
        group = analyze(arrays, min_steps=100)["groups"][0]
        self.assertEqual(group["status"], "ok")
        self.assert_verdict(group, "trot")

    def test_empty_contact_raises_schema_error(self):
        path = save_npz(self.tmp / "empty.npz", np.zeros((0, 1, 4), dtype=bool))
        with self.assertRaises(report_mod.SchemaError):
            report_mod.prepare_arrays(report_mod.load_npz(path), lambda _msg: None)


# ---------------------------------------------------------------------------
# 分组与过滤
# ---------------------------------------------------------------------------

class TestGroupingAndFiltering(GaitReportCase):
    def build_two_terrains(self) -> Path:
        """4 个环境：列 0（flat）2 个走 trot、列 1（boxes）2 个走 bound。"""
        trot = synth_contact("trot", envs=2, steps=STEPS)
        bound = synth_contact("bound", envs=2, steps=STEPS)
        contact = np.concatenate([trot, bound], axis=1)
        return save_npz(self.tmp / "two.npz", contact,
                        terrain_type=[0, 0, 1, 1], terrain_level=[0, 0, 1, 1],
                        terrain_names=("flat", "boxes"))

    def test_groups_by_terrain_column(self):
        report = analyze(self.load(self.build_two_terrains()), min_steps=100)
        self.assertEqual([group["terrain"] for group in report["groups"]], ["flat", "boxes"])
        self.assertEqual([group["envs"] for group in report["groups"]], [2, 2])
        self.assert_verdict(report["groups"][0], "trot")
        self.assert_verdict(report["groups"][1], "bound")

    def test_by_level_splits_groups(self):
        report = analyze(self.load(self.build_two_terrains()), min_steps=100, by_level=True)
        keys = [(group["terrain"], group["level"]) for group in report["groups"]]
        self.assertEqual(keys, [("flat", 0), ("boxes", 1)])

    def test_level_shows_mixed_without_by_level(self):
        trot = synth_contact("trot", envs=2, steps=STEPS)
        contact = np.concatenate([trot, trot], axis=1)
        path = save_npz(self.tmp / "mixed.npz", contact, terrain_type=[0, 0, 0, 0],
                        terrain_level=[0, 0, 3, 3], terrain_names=("flat",))
        report = analyze(self.load(path), min_steps=100)
        self.assertEqual(len(report["groups"]), 1)
        self.assertEqual(report["groups"][0]["level"], -1, "等级混杂时应标 -1（打印成『混合』）")
        self.assertIn("混合", report_mod.format_report(report))

    def test_vx_min_filters_low_command_steps(self):
        """前半段指令 0.1 m/s（不算在走）、后半段 1.0 m/s：有效步必须只有后半段。"""
        contact = synth_contact("trot", envs=1, steps=STEPS)
        steps, envs, _feet = contact.shape
        cmd = np.tile(np.asarray([1.0, 0.0, 0.0], dtype=np.float32), (steps, envs, 1))
        cmd[: STEPS // 2, 0, 0] = 0.1
        path = self.tmp / "vx.npz"
        np.savez_compressed(path, contact=contact, cmd=cmd,
                            terrain_type=np.asarray([0], dtype=np.int32),
                            terrain_level=np.asarray([0], dtype=np.int32),
                            terrain_names=np.asarray(["flat"]), dt=np.float32(DT),
                            base_height=np.full((steps, envs), 0.31, dtype=np.float32),
                            base_lin_vel=np.zeros((steps, envs, 3), dtype=np.float32),
                            foot_names=np.asarray([f"{foot}_FOOT" for foot in report_mod.FEET]))
        group = analyze(self.load(path), min_steps=100)["groups"][0]
        self.assertEqual(group["steps_valid"], STEPS // 2)
        self.assertEqual(group["steps_total"], STEPS)
        self.assertAlmostEqual(group["metrics"]["cmd_speed_mean_mps"], 1.0, delta=0.05)
        self.assert_verdict(group, "trot")

    def test_vx_min_kills_the_group_when_nothing_moves(self):
        """指令全程为 0 ⇒ 有效步 0 ⇒ 标样本不足，不抛异常（也不拿静止数据硬算步态）。"""
        contact = synth_contact("trot", envs=1)
        path = save_npz(self.tmp / "still.npz", contact, cmd_speed=0.0)
        group = analyze(self.load(path), min_steps=100)["groups"][0]
        self.assertEqual(group["steps_valid"], 0)
        self.assertEqual(group["status"], "insufficient")
        self.assertIsNone(group["metrics"])


# ---------------------------------------------------------------------------
# 输出契约
# ---------------------------------------------------------------------------

class TestOutputs(GaitReportCase):
    def test_side_metrics_are_reported(self):
        group = self.first_group(synth_contact("trot", envs=2))
        metrics = group["metrics"]
        self.assertAlmostEqual(metrics["base_height_mean_m"], 0.31, delta=0.01)
        self.assertAlmostEqual(metrics["base_speed_mean_mps"], 1.0, delta=0.01)
        self.assertAlmostEqual(metrics["cmd_speed_mean_mps"], 1.0, delta=0.01)
        self.assertEqual(len(metrics["duty"]), 4)
        for value in metrics["duty"]:
            self.assertAlmostEqual(value, 0.5, delta=0.05)

    def test_text_report_contains_the_table_columns(self):
        path = save_npz(self.tmp / "print.npz", synth_contact("trot", envs=2),
                        terrain_names=("flat",))
        text = report_mod.format_report(analyze(self.load(path), min_steps=100))
        for column in ("地形", "等级", "有效步", "period", "dutyFL", "dutyFR", "dutyRL", "dutyRR",
                       "FL-FR相", "FL-RL相", "判定", "残差"):
            self.assertIn(column, text)
        self.assertIn("trot", text)

    def test_json_output_is_valid_json_with_nulls(self):
        path = save_npz(self.tmp / "json.npz", synth_contact("trot", envs=2))
        report = analyze(self.load(path), min_steps=100)
        out = self.tmp / "out" / "report.json"
        report_mod.write_json(report, out)
        raw = out.read_text(encoding="utf-8")
        self.assertNotIn("NaN", raw, "非法 JSON：NaN 必须写成 null")
        loaded = json.loads(raw)
        self.assertEqual(loaded["metadata"]["num_envs"], 2)
        self.assertEqual(loaded["groups"][0]["metrics"]["verdict"], "trot")

    def test_main_cli_end_to_end(self):
        path = save_npz(self.tmp / "cli.npz", synth_contact("trot", envs=2))
        out = self.tmp / "cli.json"
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = report_mod.main(["--npz", str(path), "--min-steps", "100",
                                    "--vx-min", "0.3", "--json", str(out)])
        self.assertEqual(code, 0)
        self.assertIn("trot", buffer.getvalue())
        self.assertTrue(out.is_file())

    def test_main_reports_missing_file_without_raising(self):
        buffer = io.StringIO()
        with contextlib.redirect_stderr(buffer):
            code = report_mod.main(["--npz", str(self.tmp / "does_not_exist.npz")])
        self.assertEqual(code, 1)
        self.assertIn("读 npz 失败", buffer.getvalue())



class TestPeriodFallbackAndByName(unittest.TestCase):
    """2026-09-28 真实 dump 暴露的两件事：① 高 duty（>0.85）时第一只足的自相关没有显著峰 ⇒
    `period_source="foot0"` 给 NaN，应**自动回退**到 any；② 每类地形占连续若干列 ⇒ 需要 `--by-name`。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _npz(self, name, contact, terrain_type, terrain_names, cmd=1.0):
        path = self.dir / name
        payload = {
            "contact": contact.astype(bool),
            "dt": np.float32(DT),
            "terrain_type": np.asarray(terrain_type, dtype=np.int32),
            "terrain_names": np.asarray(terrain_names, dtype="U"),
        }
        if cmd is not None:
            payload["cmd"] = np.tile(np.array([cmd, 0.0, 0.0], dtype=np.float32),
                                     (contact.shape[0], contact.shape[1], 1))
        np.savez_compressed(path, **payload)
        return path

    def test_foot0_failure_falls_back_to_any_foot(self):
        steps, envs = 600, 1
        # FL（foot0）整段触地 ⇒ 自相关无峰；RR 是干净的周期方波 ⇒ any 能估出周期
        contact = np.zeros((steps, envs, 4), dtype=bool)
        contact[:, :, 0] = True
        for step in range(steps):
            contact[step, :, 3] = ((step * DT) / 0.6 % 1.0) < 0.5
        path = self._npz("fb.npz", contact, [0], ("flat",))
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = report_mod.main(["--npz", str(path), "--min-steps", "100"])
        self.assertEqual(code, 0)
        self.assertIn("回退", buffer.getvalue(), "foot0 无峰时应自动回退到 any 并在表里标明来源")

    def test_by_name_merges_columns_of_one_terrain(self):
        names = ("flat", "flat", "gap", "gap")
        contact = np.zeros((200, 4, 4), dtype=bool)
        for env, column in enumerate([0, 1, 2, 3]):
            for step in range(200):
                for foot in range(4):
                    contact[step, env, foot] = ((step * DT) / 0.6 + (0.5 if foot % 2 else 0.0)) % 1.0 < 0.5
        path = self._npz("bn.npz", contact, [0, 1, 2, 3], names)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            report_mod.main(["--npz", str(path), "--min-steps", "100", "--by-name"])
        rows = [line.split()[0] for line in buffer.getvalue().splitlines()
                if line.strip() and not line.startswith(("#", "-")) and line.split()[0] in ("flat", "gap")]
        self.assertEqual(sorted(rows), ["flat", "gap"], "同名的列应合并（每类一行）")

    def test_by_name_actually_aggregates_multi_column_types(self):
        """回归测试（2026-09-28 修的 bug）：`--by-name` 必须**真的聚合**，不只是换个标签。

        原实现的 `groups` 算在 `terrain_index` 重映射**之前** ⇒ 分组仍是原始列索引（40 行），
        而标签已经换成"去重后的 11 个名字" ⇒ 整体错位。实测后果：真实 dump 里那一行 "flat"
        其实是 **column 10**（一个 boxes 列），据此判"平地=trot"会得到错误结论。
        """
        steps, dt = 400, 0.02
        names = ["flat", "flat", "boxes", "boxes", "boxes", "gap"]
        contact = np.zeros((steps, len(names), 4), dtype=bool)
        for env in range(len(names)):
            for foot in range(4):
                offset = 0.5 if foot in (0, 3) else 0.0
                for step in range(steps):
                    contact[step, env, foot] = ((step * dt) / 0.6 + offset) % 1.0 < 0.5
        path = self._npz("agg.npz", contact, list(range(len(names))), names)

        def groups(extra):
            out = self.dir / f"r{len(extra)}.json"
            with contextlib.redirect_stdout(io.StringIO()):
                report_mod.main(["--npz", str(path), "--min-steps", "50", "--json", str(out)] + extra)
            import json as _json
            return _json.loads(out.read_text(encoding="utf-8"))["groups"]

        per_column = groups([])
        self.assertEqual(len(per_column), 6, "不加 --by-name 时逐列一行")
        by_name = groups(["--by-name"])
        self.assertEqual(len(by_name), 3, "加了 --by-name 应当只剩 3 类")
        got = {g["terrain"]: g["envs"] for g in by_name}
        self.assertEqual(got, {"flat": 2, "boxes": 3, "gap": 1},
                         "同名列的环境数必须合并（这正是原 bug 丢掉的信息）")

if __name__ == "__main__":
    unittest.main()
