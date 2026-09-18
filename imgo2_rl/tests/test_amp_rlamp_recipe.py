"""rl_amp（fan-ziqi）配方的离线契约测试（纯标准库，不需要 GPU / Isaac Sim）。

用户 2026-09-18 的决定：粗糙地形版要「忠于 AMP」——按 `rl_amp` 的设置，
**任务奖励只留线速度/角速度跟踪**（其余全 0），AMP 侧比例也照它的 `coef 2.0 / lerp 0.3`。
为让"平地 vs 粗糙"成为单变量对照，同一配方同时做成 `Imgo2-basemove-flat-amp-rlamp` 与
`Imgo2-basemove-rough-amp-rlamp`。

参考数值（`~/Desktop/AMP/rl_amp/legged_gym/legged_gym/envs/a1/a1_amp_config.py`）：
    tracking_lin_vel = 1.5 * 1./(.005*6)  = 50.0     → 每步 50.0 × 0.02 = 1.0
    tracking_ang_vel = 0.5 * 1./(.005*6)  = 16.6667  → 每步 16.6667 × 0.02 = 0.3333
    其余（lin_vel_z/ang_vel_xy/orientation/torques/dof_vel/dof_acc/base_height/
          feet_air_time/collision/feet_stumble/action_rate/stand_still/dof_pos_limits）全 0

Run: python3 -m unittest discover -s tests -p test_amp_rlamp_recipe.py
"""

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
AMP_CFG = ROOT / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/base_move/amp_env_cfg.py"
AGENT_CFG = ROOT / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/base_move/agents/amp_rsl_rl_cfg.py"
TASKS_INIT = ROOT / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/base_move/__init__.py"
REWARDS_CFG = ROOT / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py"
RLAMP_CANDIDATES = [
    Path.home() / "Desktop/AMP/rl_amp/legged_gym/legged_gym/envs/a1/a1_amp_config.py",
    Path.home() / "RL/isaac/AMP/rl_amp/legged_gym/legged_gym/envs/a1/a1_amp_config.py",
]

# rl_amp 里**非零**的跟踪项 → 我们的项名
RLAMP_REFERENCE_MAP = {
    "tracking_lin_vel": "track_lin_vel_xy_exp",
    "tracking_ang_vel": "track_ang_vel_z_exp",
}
# 参考里显式为 0 的项（我们这边必须被清空，不得残留任何权重）
RLAMP_REFERENCE_ZEROED = (
    "lin_vel_z", "ang_vel_xy", "orientation", "torques", "dof_vel", "dof_acc",
    "base_height", "feet_air_time", "collision", "feet_stumble", "action_rate",
    "stand_still", "dof_pos_limits", "termination",
)


def _module(path):
    return ast.parse(Path(path).read_text(encoding="utf-8-sig"))


def _num(node):
    """把「只含数字与四则运算」的 AST 求值（参考里写的是 `1.5 * 1./(.005*6)` 这类表达式）。"""
    return eval(compile(ast.Expression(body=node), "<cfg>", "eval"), {"__builtins__": {}}, {})


def _class(tree, name):
    hits = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == name]
    assert hits, f"找不到类 {name}"
    return hits[0]


def _func_src(tree, cls_name, fn_name):
    cls = _class(tree, cls_name)
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == fn_name)
    return ast.unparse(fn)


def _reward_term_names():
    cfg = _class(_module(REWARDS_CFG), "RewardsCfg")
    names = set()
    for node in cfg.body:
        if isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


class RLAmpRecipeTests(unittest.TestCase):
    def setUp(self):
        self.tree = _module(AMP_CFG)
        self.helper = _func_src(self.tree, "Imgo2AmpRLAmpEnvCfg", "__post_init__")

    def test_only_two_rewards_are_kept(self):
        """只留线/角速度跟踪；其余项不得出现在 keep 名单里。"""
        keep = next(n for n in self.tree.body if isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "RLAMP_KEEP_REWARDS" for t in n.targets))
        self.assertEqual(ast.literal_eval(keep.value),
                         ("track_lin_vel_xy_exp", "track_ang_vel_z_exp"))

    def test_helper_nulls_everything_else(self):
        """helper 必须遍历清空非保留项，并把两项权重按『原始权重 × dt 换算』写进去。"""
        src = _func_src(self.tree, "_dummy", "_dummy") if False else ast.unparse(
            next(n for n in self.tree.body
                 if isinstance(n, ast.FunctionDef) and n.name == "apply_rlamp_task_rewards"))
        self.assertIn("attr not in RLAMP_KEEP_REWARDS", src)
        self.assertIn("setattr(cfg.rewards, attr, None)", src)
        self.assertIn("RLAMP_TRACK_LIN_VEL_RAW * RLAMP_DT / step_dt", src)
        self.assertIn("RLAMP_TRACK_ANG_VEL_RAW * RLAMP_DT / step_dt", src)
        # 不得偷偷加回高度/姿态/步态项
        for forbidden in ("base_height_l2", "lin_vel_z_l2", "ang_vel_xy_l2", "joint_pos_limits",
                          "feet_air_time", "action_rate_l2", "undesired_contacts"):
            self.assertNotIn(f"rewards.{forbidden}", src)

    def test_kept_terms_exist_in_rewards_cfg(self):
        available = _reward_term_names()
        for name in ("track_lin_vel_xy_exp", "track_ang_vel_z_exp"):
            self.assertIn(name, available)

    def test_raw_weights_match_reference_source(self):
        """本机有参考项目时直接读它的 scales，逐项比对（防止我抄错数）。"""
        ref = next((p for p in RLAMP_CANDIDATES if p.exists()), None)
        if ref is None:
            self.skipTest("本机没有 rl_amp 参考项目，跳过与参考源码的直接比对")
        scales = _class(_module(ref), "scales")
        values = {t.id: _num(node.value) for node in scales.body if isinstance(node, ast.Assign)
                  for t in node.targets if isinstance(t, ast.Name)}
        # 参考里非零的只有两项
        nonzero = {k: v for k, v in values.items() if v != 0}
        self.assertEqual(sorted(nonzero), sorted(RLAMP_REFERENCE_MAP), "参考的非零项变了")
        # 我们的常量必须等于参考表达式算出的值
        consts = {n.targets[0].id: _num(n.value) for n in self.tree.body
                  if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                  and n.targets[0].id.startswith("RLAMP_")}
        self.assertAlmostEqual(consts["RLAMP_TRACK_LIN_VEL_RAW"], values["tracking_lin_vel"], places=9)
        self.assertAlmostEqual(consts["RLAMP_TRACK_ANG_VEL_RAW"], values["tracking_ang_vel"], places=9)
        self.assertAlmostEqual(consts["RLAMP_DT"], 0.02, places=12)
        # 参考里显式为 0 的那些项：我们的实现必须靠"清空"来对齐（不能给非零权重）
        for ref_name in RLAMP_REFERENCE_ZEROED:
            self.assertIn(ref_name, values, f"参考 scales 里应有 {ref_name}=0")
            self.assertEqual(values[ref_name], 0, f"{ref_name} 在参考里应为 0（配方变了？）")

    def test_per_step_values_and_style_task_ratio(self):
        """每步系数：线 1.0 / 角 0.3333；AMP 比例沿用参考的 coef 2.0 / lerp 0.3。"""
        consts = {n.targets[0].id: _num(n.value) for n in self.tree.body
                  if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                  and n.targets[0].id.startswith("RLAMP_")}
        lin = consts["RLAMP_TRACK_LIN_VEL_RAW"] * consts["RLAMP_DT"]
        ang = consts["RLAMP_TRACK_ANG_VEL_RAW"] * consts["RLAMP_DT"]
        self.assertAlmostEqual(lin, 1.0, places=9)
        self.assertAlmostEqual(ang, 1.0 / 3.0, places=6)
        # AMP 比例：风格上限 2.0×(1−0.3)=1.40/步，任务上限 0.3×(1.0+0.333)=0.40/步
        style_cap = 2.0 * (1 - 0.3)
        task_cap = 0.3 * (lin + ang)
        self.assertAlmostEqual(style_cap, 1.40, places=9)
        self.assertAlmostEqual(task_cap, 0.40, places=6)
        self.assertGreater(style_cap / task_cap, 3.0)   # 风格明显主导

    def test_amp_runner_uses_reference_ratio(self):
        """两个 rlamp 任务都用 `AMPRunnerCfg`（coef 2.0 / lerp 0.3），即参考的比例。"""
        agent = _module(AGENT_CFG)
        src = ast.unparse(_class(agent, "AMPRunnerCfg"))
        self.assertIn("amp_reward_coef = 2.0", src)
        self.assertIn("amp_task_reward_lerp = 0.3", src)
        tasks = Path(TASKS_INIT).read_text(encoding="utf-8")
        for task in ("Imgo2-basemove-flat-amp-rlamp", "Imgo2-basemove-flat-amp-rlamp-play",
                     "Imgo2-basemove-rough-amp-rlamp", "Imgo2-basemove-rough-amp-rlamp-play"):
            self.assertIn(task, tasks)

    def test_flat_and_rough_share_the_same_recipe(self):
        """平地版与粗糙版必须用同一配方（地形才是单变量）。"""
        for cls in ("Imgo2AmpRLAmpEnvCfg", "Imgo2AmpRoughEnvCfg"):
            self.assertIn("apply_rlamp_task_rewards(self)", _func_src(self.tree, cls, "__post_init__"),
                          f"{cls} 没有套用 rl_amp 配方")
        # 粗糙版里不得再单独设置 base_height（该奖励已被清空）
        rough = _func_src(self.tree, "Imgo2AmpRoughEnvCfg", "__post_init__")
        self.assertNotIn("base_height_l2", rough)

    def test_inherited_six_term_recipe_is_still_intact(self):
        """改成 rlamp 不得动到原来的 AMP-only 配方（24500 轮那次）。"""
        keep = _func_src(self.tree, "Imgo2AmpMoveEnvCfg", "_keep_only_amp_task_rewards")
        for needle in ("TRACK_LIN_VEL_PER_STEP = 4.0", "TRACK_ANG_VEL_PER_STEP = 2.0",
                       "BASE_HEIGHT_PER_STEP = -5.0"):
            self.assertIn(needle, keep)


if __name__ == "__main__":
    unittest.main()
