"""Offline regression tests for the gait phase-kernel probe.

锁定两件事：① 相位核参数在 `CMoE_env_cfg` 的四处**必须一致**（三处 `gait_metric_*` ＋ `feet_gait`
的 post_init 覆盖），且当前是变陡后的 `std=0.2 / max_err=0.5`；② 反演结论本身——**四足锁相时三态
读数相等**，所以"三态接近相等"不能读成"哪个略高就是哪种步态"。纯标准库，无需 torch / Isaac Lab。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "imgo2_rl" / "scripts" / "tools" / "gait_kernel_probe.py"

_spec = importlib.util.spec_from_file_location("gait_kernel_probe", TOOL_PATH)
probe = importlib.util.module_from_spec(_spec)
sys.modules["gait_kernel_probe"] = probe
_spec.loader.exec_module(probe)

# run H（2026-09-24_22-08-42_cmoe_H_sharpkernel）1849 轮的全地形均值
RUN_H_OBSERVED = [0.768, 0.782, 0.765]


class TestKernelConfig(unittest.TestCase):
    def test_params_are_sharpened_and_shared_across_four_places(self):
        std, cap, declarations = probe.read_kernel_params(probe.CMOE_CFG)
        self.assertAlmostEqual(std, 0.2, places=9)
        self.assertAlmostEqual(cap, 0.5, places=9)
        self.assertEqual(len(declarations), 3)

    def test_config_drift_is_rejected(self):
        """负向对照：只改一处 `std` ⇒ 四处不一致必须报错（否则分类器读数不再代表奖励）。"""
        original = probe.CMOE_CFG.read_text(encoding="utf-8")
        drifted = original.replace('"std": 0.2,', '"std": 0.3,', 1)
        self.assertNotEqual(original, drifted, "配置里没有找到可替换的 \"std\": 0.2,")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "CMoE_env_cfg.py"
            path.write_text(drifted, encoding="utf-8")
            with self.assertRaises(ValueError):
                probe.read_kernel_params(path)

    def test_declarations_are_the_three_canonical_pairings(self):
        _, _, declarations = probe.read_kernel_params(probe.CMOE_CFG)
        pairs = {label: pair for label, pair in declarations}
        self.assertEqual(pairs["trot"], (("FL", "RR"), ("FR", "RL")))
        self.assertEqual(pairs["bound"], (("FL", "FR"), ("RL", "RR")))
        self.assertEqual(pairs["pace"], (("FL", "RL"), ("FR", "RR")))


class TestKernelInversion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.std, cls.cap, cls.declarations = probe.read_kernel_params(probe.CMOE_CFG)

    def _row(self, gait: str, period: float = 0.34, duty: float = 0.5):
        return probe.simulate(self.std, self.cap, self.declarations,
                              probe.gait_offsets(gait), period, duty, dt=0.02)

    def test_lockstep_makes_all_three_declarations_equal(self):
        row = self._row("lockstep")
        self.assertLess(max(row.values()) - min(row.values()), 1.0e-9)

    def test_clean_trot_is_discriminated_at_reference_period(self):
        row = self._row("trot", period=0.93)
        self.assertAlmostEqual(row["trot"], 1.0, places=6)
        self.assertGreater(row["trot"] - max(row["bound"], row["pace"]), 0.3)

    def test_run_h_pattern_is_nearest_to_lockstep(self):
        """run H 的三态读数按**形状**归一化后最接近锁相，而不是 trot/bound/pace。"""
        target = probe.shape(RUN_H_OBSERVED)

        def distance(row):
            return sum((a - b) ** 2 for a, b in zip(probe.shape(list(row.values())), target)) ** 0.5

        rows = {name: self._row(name) for name in ("trot", "bound", "pace", "lockstep")}
        best = min(rows, key=lambda name: distance(rows[name]))
        self.assertEqual(best, "lockstep")
        # 「最接近」必须显著，不能只是数值上小一点点
        self.assertLess(distance(rows["lockstep"]), 0.1 * distance(rows["trot"]))


def _synthetic_contacts(offsets: dict[str, float], period: float, duty: float, dt: float = 0.02,
                        steps: int = 640):
    """按相位模型合成 `contact` (steps, 1, 4)，列序 FL, FR, RL, RR（与 npz 一致）。"""
    import numpy as np

    frames = np.zeros((steps, 1, 4), dtype=np.int8)
    for index in range(steps):
        time = index * dt
        for foot_id, foot in enumerate(probe.FEET):
            frames[index, 0, foot_id] = 1 if ((time / period + offsets[foot]) % 1.0) < duty else 0
    return frames


class TestContactReplay(unittest.TestCase):
    """`replay_contacts`：用真实接触序列重算三态读数（2026-09-25 为了判定"三态相等"而加）。"""

    @classmethod
    def setUpClass(cls):
        cls.std, cls.max_err, cls.declarations = probe.read_kernel_params(probe.CMOE_CFG)
        cls.cap = cls.max_err ** 2

    def _replay(self, offsets, period=0.34, duty=0.5):
        return probe.replay_contacts(_synthetic_contacts(offsets, period, duty), 0.02,
                                     self.std, self.cap, self.declarations, min_ep_len=0)

    def test_trot_replay_is_trot_led(self):
        row = self._replay(probe.gait_offsets("trot"))
        self.assertGreater(row["trot"] - row["bound"], 0.05, f"trot 应领先：{row}")
        self.assertIn("trot 领先", probe.verdict(row))

    def test_lockstep_replay_has_equal_declarations(self):
        row = self._replay(probe.gait_offsets("lockstep"))
        spread = max(row[label] for label in probe.PAIR_LABELS) - min(row[label] for label in probe.PAIR_LABELS)
        self.assertLess(spread, 1.0e-12)
        self.assertLess(row["mismatch"], 1.0e-12, "锁相时六对时间差应为 0")
        self.assertIn("三态相等", probe.verdict(row))

    def test_last_air_time_uses_touchdown_edge(self):
        """`last_air_time` 是**刚结束**的那段腾空（0.5 duty × 0.34 s ⇒ ≈0.17 s），不是每步清零的当前值。"""
        row = self._replay(probe.gait_offsets("trot"), period=0.34, duty=0.5)
        self.assertAlmostEqual(row["last_air_time"], 0.17, delta=0.03)

    def test_std_is_what_buys_discrimination(self):
        """**分辨率由 `std` 决定，`max_err` 只管地板**（2026-09-25 在真实接触序列上量的）。

        * 只把 `max_err` 0.5→0.1、`std` 保持 0.2 ⇒ 溢价**纹丝不动**（地板反而升到 0.905）；
        * `std` 0.2→0.01 ⇒ 溢价 0.116→**0.526**（这就是"相位项为什么一直没梯度"的答案）；
        * 不要 `std` 的 hinge 核（只有 `max_err=0.15`）同样给出 >0.4 的溢价 ⇒ `std` **不是必须保留**的。
        """
        import pathlib
        npz = pathlib.Path(probe.ROOT.parent) / "imgo2_rl/logs/gait_24500_vx1.0.npz"
        if not npz.is_file():
            self.skipTest(f"缺少先验步态记录 {npz}")
        import numpy as np

        data = np.load(npz, allow_pickle=True)

        def premium(std, max_err, kernel="gauss"):
            row = probe.replay_contacts(data["contact"], 0.02, std, max_err ** 2, self.declarations,
                                        ep_len=data["ep_len"], done=data["done"], min_ep_len=50, kernel=kernel)
            return row["trot"] - row["bound"]

        self.assertLess(premium(0.2, 0.5), 0.2)
        self.assertAlmostEqual(premium(0.2, 0.1), premium(0.2, 0.5), delta=0.02)
        self.assertGreater(premium(0.01, 0.1), 0.4)
        self.assertGreater(premium(0.0, 0.15, kernel="hinge"), 0.4)

    def test_real_prior_reference(self):
        """AMP 先验自己的读数（有 npz 才跑；logs/ 被 gitignore，换机器自动 skip）。"""
        import pathlib
        npz = pathlib.Path(probe.ROOT.parent) / "imgo2_rl/logs/gait_24500_vx1.0.npz"
        if not npz.is_file():
            self.skipTest(f"缺少先验步态记录 {npz}")
        import numpy as np

        data = np.load(npz, allow_pickle=True)
        row = probe.replay_contacts(data["contact"], 0.02, self.std, self.cap, self.declarations,
                                    ep_len=data["ep_len"], done=data["done"], min_ep_len=50)
        # 先验自己的签名：trot 明确领先，但**不是**理想化的 trot 1.000 / bound 0.28
        self.assertGreater(row["trot"] - row["bound"], 0.05)
        for label in probe.PAIR_LABELS:
            self.assertGreater(row[label], 0.7, f"该高频步态下三态都被挤在高位：{row}")
        self.assertLess(row["mismatch"], 0.2, "先验的六对时间差是 0.1 s 级 ⇒ max_err=0.5 根本够不着地板")


if __name__ == "__main__":
    unittest.main()
