"""`check_reward_overrides.py` 自身的回归测试（只用标准库，不需要 Isaac Lab）。

守两件事：
1. 它读出的 **CMoE 最终生效奖励集** 与 2026-09-24 定稿一致（16 项、`feet_gait` 不在其中、
   三个 masked 类挂在正确的项上）。这套配方改过多次且被"晚赋值覆盖"坑过两次，值得钉住。
2. 它**真的会报警**：把某个原本非零的项在后面赋成 0 时，必须给出 "被清零" 提示
   （自检用一次性临时文件，不动仓库里的配置）。
"""

from __future__ import annotations

import ast
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "scripts" / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import check_reward_overrides as chk  # noqa: E402


def _number(node: ast.AST):
    """字面量数字；`math.sqrt(0.5)` 这类表达式返回 None（表示"这处不参与比较"）。"""
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _gait_kernel_params(steps, term_names):
    """按 `CHAINS` 顺序回放 cfg 源码，读出各 term 最终的 `std`/`max_err`。

    看两类赋值：① 类体 `name = RewTerm(..., params={...})` 的字面量；
    ② `__post_init__` 里 `self.rewards.<term>.params["std"] = ...` 的覆盖（后写的生效）。
    """
    params: dict[str, dict[str, object]] = {}
    for path, class_name in steps:
        tree = ast.parse(Path(path).read_text(encoding="utf-8-sig"))
        cls = next((n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == class_name), None)
        if cls is None:
            continue
        for node in ast.walk(cls):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in term_names and isinstance(node.value, ast.Call):
                for kw in node.value.keywords:
                    if kw.arg == "params" and isinstance(kw.value, ast.Dict):
                        for key, value in zip(kw.value.keys, kw.value.values):
                            number = _number(value)
                            if isinstance(key, ast.Constant) and key.value in ("std", "max_err") and number is not None:
                                params.setdefault(target.id, {})[key.value] = number
            elif isinstance(target, ast.Subscript):
                owner = target.value
                if (isinstance(owner, ast.Attribute) and owner.attr == "params"
                        and isinstance(owner.value, ast.Attribute) and owner.value.attr in term_names):
                    try:
                        key = ast.literal_eval(target.slice)
                    except Exception:
                        continue
                    number = _number(node.value)
                    if key in ("std", "max_err") and number is not None:
                        params.setdefault(owner.value.attr, {})[key] = number
    return params


class TestCmoeEffectiveRewards(unittest.TestCase):
    """CMoE rough 的最终生效奖励集（2026-09-24 定稿）。"""

    @classmethod
    def setUpClass(cls):
        cls.weights, cls.funcs, cls.notes = chk.run_chain(chk.CHAINS["cmoe"][0][1])
        cls.effective = {t: v for t, v in cls.weights.items() if isinstance(v, (int, float)) and v != 0}

    def test_effective_term_count(self):
        # 2026-09-24：16 → 18（parkour 式"全球速度"约束）→ 19（加回 feet_air_time_variance −8.0）
        #            → 20（开 feet_gait，掩码版）→ 23（＋3 个"步态度量"项，权重 1e-6，只为记录）
        #            → 25（＋掩码版 lin_vel_z_l2 −2.0、＋diag_bounce 度量）
        #            → 27（＋diag_air_time、diag_pair_mismatch 两个接触时序诊断，权重 1e-6）
        self.assertEqual(len(self.effective), 27, f"生效项数变了：{sorted(self.effective)}")

    def test_vertical_velocity_penalty_restored(self):
        """2026-09-24 晚（用户："都还是蹦蹦跳跳的走的"）：竖直速度罚从"清零"改为"掩码恢复"。"""
        self.assertEqual(self.effective["lin_vel_z_l2"], -2.0)
        self.assertIn("MaskedLinVelZ", self.funcs["lin_vel_z_l2"])
        # feet_air_time 降权（削弱"奖励腾空"的分量）
        self.assertEqual(self.effective["feet_air_time"], 0.3)

    def test_bounce_metric_is_diagnostic_only(self):
        self.assertIn("diag_bounce", self.effective)
        self.assertLessEqual(abs(float(self.effective["diag_bounce"])), 1e-5)

    def test_gait_metric_terms_are_diagnostics_only(self):
        """三个步态度量项（trot/bound/pace 成对方式）权重必须极小，纯粹用来记录。"""
        for name in ("gait_metric_trot", "gait_metric_bound", "gait_metric_pace"):
            self.assertIn(name, self.effective)
            self.assertLessEqual(abs(float(self.effective[name])), 1e-5,
                                 f"{name} 应是度量项（权重 ≤1e-5），不能真的有奖励量级")

    def test_contact_timing_diagnostics_are_diagnostic_only(self):
        """2026-09-24 夜新增的两个接触时序诊断项（`ā` 与六对时间差尺度）也只是记录。"""
        for name in ("diag_air_time", "diag_pair_mismatch"):
            self.assertIn(name, self.effective)
            self.assertLessEqual(abs(float(self.effective[name])), 1e-5,
                                 f"{name} 应是诊断项（权重 ≤1e-5）")

    def test_phase_kernel_is_sharpened_and_classifier_matches_reward(self):
        """2026-09-24 夜（用户同意）：相位核变陡，且**分类器三项必须与 `feet_gait` 同参数**。

        旧值 `std=√0.5=0.7071`、`max_err=0.2` ⇒ 单核地板 `exp(−2·0.2²/0.7071)=0.893`：**配对差半个
        周期（≈0.45 s）也拿 89 分**，6 核乘积只跨 [0.508, 1]。run G 实测 `trot−bound` 仅 −0.010
        且第 50→436 轮完全平坦（期间速度核 0.20→0.62）⇒ 边际奖励只有 0.01/s，PPO 不会理它。
        新值地板 `exp(−2·0.5²/0.2)=0.082`，参考步态（T 0.93 s、duty 0.5）trot 1.000 / bound 0.302
        ⇒ 溢价 **0.264 → 0.698/s**。分类器与奖励不同参数时读数就不再代表奖励 ⇒ 一起钉住。
        """
        steps = chk.CHAINS["cmoe"][0][1]
        names = ("feet_gait", "gait_metric_trot", "gait_metric_bound", "gait_metric_pace")
        params = _gait_kernel_params(steps, names)
        for name in names:
            self.assertEqual(params.get(name, {}).get("std"), 0.2,
                             f"{name} 的 std 没跟相位核同步（分类器会与奖励脱钩）")
            self.assertEqual(params.get(name, {}).get("max_err"), 0.5,
                             f"{name} 的 max_err 没跟相位核同步（分类器会与奖励脱钩）")

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
        # 2026-09-24 晚：1.0 → 0.3（该式对"滞空更久"的梯度恒为 +1 ⇒ 降权以削弱"奖励腾空/弹跳"）
        self.assertEqual(self.effective["feet_air_time"], 0.3)
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
