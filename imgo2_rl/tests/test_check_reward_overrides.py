"""`check_reward_overrides.py` 自身的回归测试（只用标准库，不需要 Isaac Lab）。

守两件事：
1. 它读出的 **CMoE 最终生效奖励集** 与 2026-09-24 定稿一致（16 项、`feet_gait` 不在其中、
   三个 masked 类挂在正确的项上）。这套配方改过多次且被"晚赋值覆盖"坑过两次，值得钉住。
2. 它**真的会报警**：把某个原本非零的项在后面赋成 0 时，必须给出 "被清零" 提示
   （自检用一次性临时文件，不动仓库里的配置）。
"""

from __future__ import annotations

import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "scripts" / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import check_reward_overrides as chk  # noqa: E402


class TestCmoeEffectiveRewards(unittest.TestCase):
    """CMoE rough 的最终生效奖励集（2026-09-24 定稿）。"""

    @classmethod
    def setUpClass(cls):
        cls.weights, cls.funcs, cls.notes = chk.run_chain(chk.CHAINS["cmoe"][0][1])
        cls.effective = {t: v for t, v in cls.weights.items() if isinstance(v, (int, float)) and v != 0}

    def test_effective_term_count(self):
        # 2026-09-24：16 → 18（parkour 式"全球速度"约束）→ 19（加回 feet_air_time_variance −8.0）
        #            → 20（开 feet_gait，掩码版）
        self.assertEqual(len(self.effective), 20, f"生效项数变了：{sorted(self.effective)}")

    def test_world_vel_replaces_body_vel(self):
        self.assertEqual(self.effective["track_world_vel_xy_exp"], 5.0)
        self.assertNotIn("track_lin_vel_xy_exp", self.effective,
                         "机体系速度跟踪应已被世界系版本取代")

    def test_parkour_soft_terms_present(self):
        self.assertEqual(self.effective["lin_pos_y"], -0.4)
        self.assertEqual(self.effective["yaw_abs"], -0.2)

    def test_gaitshaping_values(self):
        # 三项照搬 PPO 的固定步态 shaping，都挂 masked 类。
        self.assertEqual(self.effective["joint_mirror"], -1.0)
        self.assertEqual(self.effective["feet_height_body"], -5.0)
        self.assertEqual(self.effective["feet_air_time"], 1.0)
        self.assertIn("MaskedJointMirror", self.funcs["joint_mirror"])
        self.assertIn("MaskedFeetHeightBody", self.funcs["feet_height_body"])
        self.assertIn("MaskedFeetAirTime", self.funcs["feet_air_time"])
        # PPO 那套里量级最大的步态项，2026-09-24 晚加回（掩码版）
        self.assertEqual(self.effective["feet_air_time_variance"], -8.0)
        self.assertIn("MaskedFeetAirTimeVariance", self.funcs["feet_air_time_variance"])

    def test_feet_gait_enabled_with_masked_class(self):
        """2026-09-24 晚用户："那就开 feet gait，同样加掩码"。"""
        self.assertEqual(self.effective["feet_gait"], 1.0)
        self.assertIn("TrotWithoutGapReward", self.funcs["feet_gait"])

    def test_not_restored_terms(self):
        # `feet_slide` 仍未恢复（用户未要求）；若日后恢复，请同步更新本测试与 docs。
        self.assertNotIn("feet_slide", self.effective)

    def test_no_masked_func_is_dead(self):
        for term in self.funcs:
            self.assertIn(term, self.effective,
                          f"{term} 的 func 被换成了自定义类但权重为 0 ⇒ 死代码")


class TestZeroingWarning(unittest.TestCase):
    """把非零项在后面清零时必须报警（`joint_mirror` 失效的模式）。"""

    def _write(self, tmpdir: str, name: str, body: str) -> Path:
        path = Path(tmpdir) / name
        path.write_text(textwrap.dedent(body), encoding="utf-8")
        return path

    def test_detects_silent_zeroing(self):
        base = """
            class RewardsCfg:
                alpha = RewTerm(func=mdp.alpha, weight=0.0)
                beta = RewTerm(func=mdp.beta, weight=0.0)
        """
        derived = """
            class DerivedCfg:
                def __post_init__(self):
                    self.rewards.alpha.weight = -1.0
                    self.rewards.beta.weight = -1.0
                    # 晚赋值把 alpha 抹掉 —— 必须被抓出来
                    self.rewards.alpha.weight = 0.0
        """
        with tempfile.TemporaryDirectory() as tmp:
            p_base = self._write(tmp, "base_cfg.py", base)
            p_derived = self._write(tmp, "derived_cfg.py", derived)
            weights, _funcs, notes = chk.run_chain(
                [(p_base, "RewardsCfg"), (p_derived, "DerivedCfg")]
            )
        self.assertEqual(weights["alpha"], 0.0)
        self.assertEqual(weights["beta"], -1.0)
        self.assertTrue(any("alpha" in n and "清零" in n for n in notes),
                        f"没有报出 alpha 被清零：{notes}")
        self.assertFalse(any("beta" in n and "清零" in n for n in notes),
                         f"不该报 beta：{notes}")


if __name__ == "__main__":
    unittest.main()
