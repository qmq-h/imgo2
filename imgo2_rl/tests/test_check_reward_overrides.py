"""`check_reward_overrides.py` 自身的回归测试（只用标准库，不需要 Isaac Lab）。

守两件事：
1. 它读出的 **CMoE 最终生效奖励集** 与当前定稿一致（`cmoe` **28 项** / `cmoe-gaitfree` **24 项**、
   各 masked 类挂在正确的项上；逐项数值见本文件断言）。这套配方改过多次且被"晚赋值覆盖"坑过
   两次，值得钉住。
   **2026-10-01**：只还原「奖励/权重」四项到 4000 轮存档口径（`lin_vel_z_l2 −2`、
   `undesired_contacts −0.5`、`contact_forces −0.02`、`base_height_flat_l2 0.0 禁用`），
   生效项数 29 → **28**、25 → **24**。
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
        # 2026-09-30：27 → **29**（＋`base_height_flat_l2` −35 平地去重力门高度罚、
        #            ＋`diag_base_height` 1e-6 有符号高度误差诊断）。
        # 2026-10-01：29 → **28**（只还原「奖励/权重」四项到 4000 轮存档口径：
        #            `base_height_flat_l2` −35 → **0.0（禁用）** ⇒ 移出生效表；
        #            `lin_vel_z_l2` −4 → −2、`undesired_contacts` −5 → −0.5、
        #            `contact_forces` −0.1 → −0.02 三项仍非零 ⇒ 项数不变）。
        self.assertEqual(len(self.effective), 28, f"生效项数变了：{sorted(self.effective)}")

    def test_vertical_velocity_penalty_restored(self):
        """2026-09-24 晚（用户："都还是蹦蹦跳跳的走的"）：竖直速度罚从"清零"改为"掩码恢复"。

        2026-09-30：曾 −2.0 → −4.0（治"抬脚过高/弹跳"）。
        **2026-10-01 还原**：−4.0 → **−2.0**（与 4000 存档 L461 逐字一致）。
        掩码**仍是** `("boxes", "gap")`（跃起必须免费）。
        """
        self.assertEqual(self.effective["lin_vel_z_l2"], -2.0)
        self.assertIn("MaskedLinVelZ", self.funcs["lin_vel_z_l2"])
        # 2026-09-28：feet_air_time 从 0.3 改回 **1.0**（对齐 PPO）；弹跳改由竖直速度罚 +
        # −5.0 的机身水平罚管。
        self.assertEqual(self.effective["feet_air_time"], 1.0)

    def test_flat_only_strict_height_term_is_disabled(self):
        """2026-10-01 还原：flat-only、去重力门的 `base_height_flat_l2` 由 −35 改回 **0.0（禁用）**。

        4000 轮那份存档（`.../2026-09-29_20-16-27_cmoe_v5_5_hard/params/CMoE_env_cfg.py`）里
        **没有**本项；term 定义与 `MaskedBaseHeightL2Strict` 类**保留**，作为"备用/可再启用"
        （把 `__post_init__` 里那一行改回 −35 即可）。`disable_zero_weight_rewards()` 会把它移出
        生效表 ⇒ 不得出现在 `effective` 里。
        """
        self.assertEqual(self.weights["base_height_flat_l2"], 0.0)
        self.assertNotIn("base_height_flat_l2", self.effective)
        # func 仍是自定义类（作为备用保留）；它现在是"有意保留的死代码"（见下面的白名单）
        self.assertIn("MaskedBaseHeightL2Strict", self.funcs["base_height_flat_l2"])

    def test_signed_height_diagnostic_is_diagnostic_only(self):
        """2026-09-30：逐列 `gait_base_height_<地形>`（有符号高度误差）只是记录（1e-6），**保留**。"""
        self.assertEqual(self.effective["diag_base_height"], 1e-6)
        self.assertLessEqual(abs(float(self.effective["diag_base_height"])), 1e-5)

    def test_contact_penalties_restored(self):
        """2026-10-01 还原：`undesired_contacts` −5.0 → **−0.5**、
        `contact_forces` −0.1 → **−0.02**（= 4000 时的父类默认值 `rough_env_cfg` 的 `-2e-2`）。"""
        self.assertEqual(self.effective["undesired_contacts"], -0.5)
        self.assertEqual(self.effective["contact_forces"], -0.02)

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
        # 2026-09-28（用户："feet_air_time 对齐 PPO"）：0.3 → **1.0**（PPO rough 原值）。
        # 当年降到 0.3 是为了削弱"奖励腾空/弹跳"；现在改为**对齐 PPO 的配平**（shaping/task ≈7:1），
        # 弹跳问题交给 `lin_vel_z_l2 −2`（掩码）与 50× 强的 `flat_orientation_l2` 管。
        self.assertEqual(self.effective["feet_air_time"], 1.0)
        self.assertIn("MaskedJointMirror", self.funcs["joint_mirror"])
        self.assertIn("MaskedFeetHeightBody", self.funcs["feet_height_body"])
        self.assertIn("MaskedFeetAirTime", self.funcs["feet_air_time"])
        # PPO 那套里量级最大的步态项，2026-09-24 晚加回（掩码版）
        self.assertEqual(self.effective["feet_air_time_variance"], -8.0)
        self.assertIn("MaskedFeetAirTimeVariance", self.funcs["feet_air_time_variance"])

    def test_phase_kernel_removed_by_design(self):
        """2026-09-28（用户："相位核去掉"）：`feet_gait` 归零、从生效表移除。

        依据：相位核在这份高频步态上**本身饱和**（实测溢价 0.116/s = 跟踪项的 2.3%），而 PPO
        不用相位核也能练出干净 trot ⇒ 保留它只是多一个调不动的旋钮。三项分类器探针（1e-6）保留。
        """
        self.assertNotIn("feet_gait", self.effective)
        for probe in ("gait_metric_trot", "gait_metric_bound", "gait_metric_pace"):
            self.assertIn(probe, self.effective)

    def test_feet_slide_restored_to_align_ppo(self):
        """2026-09-28（用户："feet_slide 对齐 PPO"）：−0.05（PPO rough 原值）。

        这一项是"支撑脚不许打滑"，正对着实测的拖行形态（level 6：FL duty 0.80 / RR 0.57）。
        """
        self.assertEqual(self.effective["feet_slide"], -0.05)

    def test_no_masked_func_is_dead(self):
        """带自定义 func 的项必须权重非零 —— 例外见 `INTENTIONALLY_DEAD_MASKED`。"""
        intentionally_dead = {
            "feet_gait",             # 2026-09-28 用户决定去掉相位核（保留探针读数）
            "base_height_flat_l2",   # 2026-10-01 还原为 4000 口径 ⇒ 权重 0（term/类保留，改回 −35 即可再启用）
        }
        for term in self.funcs:
            if term in intentionally_dead:
                continue
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


class TestGaitFreeEffectiveRewards(unittest.TestCase):
    """`cmoe-gaitfree`（2026-09-25：步态交给 45 维先验）的生效集。

    与 `cmoe` 的关系是**严格子集**：只少那五项手工步态 shaping，其余一项不动 ——
    这条不变量比逐个断言更抗漂移。
    """

    @classmethod
    def setUpClass(cls):
        cls.weights, cls.funcs, cls.notes = chk.run_chain(chk.CHAINS["cmoe-gaitfree"][0][1])
        cls.effective = {t: v for t, v in cls.weights.items() if isinstance(v, (int, float)) and v != 0}
        cls.shaping_weights, _f, _n = chk.run_chain(chk.CHAINS["cmoe"][0][1])
        cls.shaping_effective = {t: v for t, v in cls.shaping_weights.items()
                                 if isinstance(v, (int, float)) and v != 0}

    def test_five_gait_shaping_terms_are_zero(self):
        for term in ("joint_mirror", "feet_air_time", "feet_height_body",
                     "feet_air_time_variance", "feet_gait"):
            self.assertNotIn(term, self.effective, f"{term} 应已归零（步态交给先验）")
            self.assertEqual(self.weights.get(term), 0.0, f"{term} 的最终权重应为 0.0")

    def test_is_a_strict_subset_of_the_shaping_recipe(self):
        removed = set(self.shaping_effective) - set(self.effective)
        # 2026-09-28 起 `feet_gait` 在**两条链上都是 0**（用户："相位核去掉"）⇒ gaitfree 相对 cmoe
        # 只再少**四项**（手工步态 shaping），生效项数 29 → 25。
        # 2026-10-01 还原后：两条链都少 `base_height_flat_l2`（权重 0 禁用）⇒ **28 → 24**。
        self.assertEqual(removed, {"joint_mirror", "feet_air_time", "feet_height_body",
                                   "feet_air_time_variance"},
                         f"gaitfree 相对 cmoe 只应少这四项，实际少了 {sorted(removed)}")
        self.assertEqual(len(self.effective), len(self.shaping_effective) - 4)
        self.assertEqual(len(self.effective), 24, f"生效项数变了：{sorted(self.effective)}")
        # 2026-09-28 新增/恢复的两项**在 gaitfree 里也要有**（它们是"步态质量/姿态"，不是手工步态风格）：
        # `feet_slide −0.05`（治拖行）与 `flat_orientation_l2 −5.0`（掩码）。
        self.assertEqual(self.effective["feet_slide"], -0.05)
        self.assertEqual(self.effective["flat_orientation_l2"], -5.0)
        # 2026-10-01 还原：flat-only 严格高度项已禁用（0.0、不在生效表），有符号高度诊断保留。
        self.assertEqual(self.weights["base_height_flat_l2"], 0.0)
        self.assertNotIn("base_height_flat_l2", self.effective)
        self.assertEqual(self.effective["diag_base_height"], 1e-6)

    def test_classifiers_and_diagnostics_are_kept(self):
        """分类器/诊断项必须留着 —— 否则再也看不见"先验漂没漂"。"""
        for name in ("gait_metric_trot", "gait_metric_bound", "gait_metric_pace",
                     "diag_air_time", "diag_bounce", "diag_pair_mismatch", "diag_base_height"):
            self.assertIn(name, self.effective, f"{name} 是度量项，不能跟着归零")
            self.assertLessEqual(abs(float(self.effective[name])), 1e-5)

    def test_task_level_terms_are_untouched(self):
        for name, value in (("track_world_vel_xy_exp", 5.0), ("base_height_l2", -10.0),
                            ("lin_vel_z_l2", -2.0), ("lin_pos_y", -0.4), ("yaw_abs", -0.2)):
            self.assertEqual(self.effective[name], value, f"{name} 不该被这条链改动")
        self.assertIn("MaskedLinVelZ", self.funcs["lin_vel_z_l2"])

    def test_zeroing_is_reported_not_silent(self):
        """归零必须出现在"被清零"提示里（每一项归零都是一次有意识的选择）。"""
        for term in ("joint_mirror", "feet_air_time", "feet_height_body",
                     "feet_air_time_variance", "feet_gait"):
            self.assertTrue(any(term in note and "清零" in note for note in self.notes),
                            f"{term} 的归零没有被报出来：{self.notes}")


if __name__ == "__main__":
    unittest.main()
