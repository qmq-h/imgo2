"""amp_go2 配方移植的离线契约测试（纯标准库，不需要 GPU / Isaac Sim）。

背景：2026-09-18 按用户要求"尽可能参考 amp_go2"新增了
`Imgo2-basemove-flat-amp-go2`（env: `Imgo2AmpGo2StyleEnvCfg` / agent: `AMPGo2RunnerCfg`）。
这类"照抄参考数值"的改动最容易在后续编辑里被悄悄改坏（漏项、写错项名、写错单位、
把 `SceneEntityCfg` 的正则留成 `""` 导致运行期才报错），而 Isaac Lab 的配置类**必须**在
Isaac Sim 应用里才能实例化（`import omni.log`），本机沙箱里跑不起来。因此这里用 AST
在源码层面把契约钉住；本机若有参考项目，还会**直接读参考源码逐项比对**。

Run: python3 -m unittest discover -s tests -p test_amp_go2_recipe.py
"""

import ast
import os
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
AMP_CFG = ROOT / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/base_move/amp_env_cfg.py"
AGENT_CFG = ROOT / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/base_move/agents/amp_rsl_rl_cfg.py"
TASKS_INIT = ROOT / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/base_move/__init__.py"
REWARDS_CFG = ROOT / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py"
# 参考项目：本机在 ~/Desktop/AMP/amp_go2-main（旧文档里写的 ~/RL/isaac/AMP 是另一台机器）
REFERENCE_CANDIDATES = [
    Path.home() / "Desktop/AMP/amp_go2-main/legged_gym/legged_gym/envs/go2/go2_amp_config.py",
    Path.home() / "RL/isaac/AMP/amp_go2-main/legged_gym/legged_gym/envs/go2/go2_amp_config.py",
]

# amp_go2 的 scales（本文件写死一份，作为"参考不在场时"的判据；在场时会与参考源码比对）
EXPECTED_RAW = {
    "track_lin_vel_xy_exp": 4.0,
    "track_ang_vel_z_exp": 2.0,
    "lin_vel_z_l2": -1.0,
    "ang_vel_xy_l2": -0.05,
    "joint_acc_l2": -2.5e-7,
    "joint_torques_l2": -1e-4,
    "base_height_l2": -1.0,
    "action_rate_l2": -0.01,
    "undesired_contacts": -1.0,
    "joint_pos_limits": -2.0,
    "feet_air_time": 1.0,
}
# 参考里的 legged_gym 项名 → 我们的项名（用于直接读参考源码比对）
REFERENCE_NAME_MAP = {
    "tracking_lin_vel": "track_lin_vel_xy_exp",
    "tracking_ang_vel": "track_ang_vel_z_exp",
    "lin_vel_z": "lin_vel_z_l2",
    "ang_vel_xy": "ang_vel_xy_l2",
    "dof_acc": "joint_acc_l2",
    "torques": "joint_torques_l2",
    "base_height": "base_height_l2",
    "action_rate": "action_rate_l2",
    "collision": "undesired_contacts",
    "dof_pos_limits": "joint_pos_limits",
    "feet_air_time": "feet_air_time",
}


def _module(path):
    return ast.parse(Path(path).read_text(encoding="utf-8-sig"))


def _class(tree, name):
    """按名字找类（**递归**：参考项目里 `class scales` 嵌在 `class rewards` 内）。"""
    hits = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == name]
    assert hits, f"源码里找不到类 {name}"
    return hits[0]


def _literal_assign(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"找不到字面量赋值 {name}")


def _reward_term_names():
    cfg = _class(_module(REWARDS_CFG), "RewardsCfg")
    names = set()
    for node in cfg.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


class Go2RecipeTests(unittest.TestCase):
    def setUp(self):
        self.tree = _module(AMP_CFG)
        self.raw = _literal_assign(self.tree, "AMP_GO2_RAW_WEIGHTS")

    def test_weight_table_is_the_eleven_reference_terms(self):
        self.assertEqual(self.raw, EXPECTED_RAW)
        # 未保留的项（参考里这两个都是 0，别顺手加进来）
        for absent in ("dof_vel", "orientation", "feet_stumble", "stand_still"):
            self.assertNotIn(absent, self.raw)

    def test_reference_source_matches_when_available(self):
        """本机有参考项目时，直接读它的 scales 逐项比对（防止我抄错数）。"""
        ref = next((p for p in REFERENCE_CANDIDATES if p.exists()), None)
        if ref is None:
            self.skipTest("本机没有 amp_go2 参考项目，跳过与参考源码的直接比对")
        scales = _class(_module(ref), "scales")
        ref_values = {}
        for node in scales.body:
            if isinstance(node, ast.Assign):          # 形如 `feet_air_time =  1.0`
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        ref_values[t.id] = ast.literal_eval(node.value)
        for ref_name, our_name in REFERENCE_NAME_MAP.items():
            self.assertIn(ref_name, ref_values, f"参考 scales 里没有 {ref_name}")
            self.assertAlmostEqual(
                self.raw[our_name], ref_values[ref_name], places=12,
                msg=f"{our_name} 与参考 {ref_name} 不一致（参考={ref_values[ref_name]}）")
        # 参考里其余非零项必须已经被我们覆盖（否则就是漏抄）
        covered = {REFERENCE_NAME_MAP[n] for n in REFERENCE_NAME_MAP}
        for ref_name, value in ref_values.items():
            if ref_name in REFERENCE_NAME_MAP or value == 0:
                continue
            self.fail(f"参考里有非零项 {ref_name}={value} 没被映射到我们的奖励集")

    def test_all_mapped_terms_exist_in_rewards_cfg(self):
        """项名必须真实存在于 RewardsCfg —— 写错名只会在 Isaac Sim 里才炸。"""
        available = _reward_term_names()
        for name in self.raw:
            self.assertIn(name, available, f"{name} 不在 velocity_env_cfg.RewardsCfg 里")

    def test_rebuilt_terms_get_non_empty_patterns(self):
        """被清空后重建的 5 项必须给出能匹配到 body/joint 的正则（`""` 会在构造环境时报错）。"""
        fn = next(n for n in _class(self.tree, "Imgo2AmpGo2StyleEnvCfg").body
                  if isinstance(n, ast.FunctionDef) and n.name == "_apply_amp_go2_rewards")
        src = ast.unparse(fn)
        self.assertIn("_amp_go2_term(name, self.foot_body_names)", src)
        # 单位换算必须写成 `raw * AMP_GO2_DT / step_dt`
        self.assertIn("raw * AMP_GO2_DT / step_dt", src)
        # 重建项里不能出现空正则
        helper = next(n for n in self.tree.body
                      if isinstance(n, ast.FunctionDef) and n.name == "_amp_go2_term")
        hsrc = ast.unparse(helper)
        self.assertNotIn('body_names=""', hsrc)
        self.assertNotIn('joint_names=""', hsrc)
        self.assertEqual(_literal_assign(self.tree, "AMP_GO2_CONTACT_PENALTY_BODIES"), ".*_THIGH")
        self.assertIn("body_names=AMP_GO2_CONTACT_PENALTY_BODIES", hsrc)   # amp_go2 只惩罚 thigh
        self.assertIn('body_names=foot_body_names', hsrc)     # feet_air_time 必须点名四只足
        self.assertIn("AMP_GO2_FEET_AIR_TIME_THRESHOLD", hsrc)

    def test_per_step_equivalence(self):
        """weight_il × step_dt_il 必须等于 raw × dt_lg（两边 weight 语义相同）。"""
        dt_lg = _literal_assign(self.tree, "AMP_GO2_DT")
        self.assertAlmostEqual(dt_lg, 0.02, places=12)   # amp_go2: 0.005 × 4
        for step_dt in (0.01, 0.02, 0.04):
            for name, raw in self.raw.items():
                weight_il = raw * dt_lg / step_dt
                self.assertAlmostEqual(weight_il * step_dt, raw * dt_lg, places=12)

    def test_two_documented_deviations(self):
        """与参考的两处有意偏离必须写在代码里（目标高度、feet_air_time 阈值）。"""
        target = _literal_assign(self.tree, "AMP_GO2_BASE_HEIGHT_TARGET")
        thresh = _literal_assign(self.tree, "AMP_GO2_FEET_AIR_TIME_THRESHOLD")
        self.assertAlmostEqual(target, 0.30, places=12)   # 参考 0.38 是 Go2 的站高
        self.assertAlmostEqual(thresh, 0.2, places=12)    # 参考硬编码 0.5，比我们参考周期还慢

    def test_command_ranges_match_reference(self):
        """指令范围照抄 amp_go2 的 go2 配置（x[-1.2,1.5] / y±0.8 / yaw±1.0）。"""
        fn = next(n for n in _class(self.tree, "Imgo2AmpGo2StyleEnvCfg").body
                  if isinstance(n, ast.FunctionDef) and n.name == "_apply_amp_go2_rewards")
        src = ast.unparse(fn)
        self.assertIn("ranges.lin_vel_x = (-1.2, 1.5)", src)
        self.assertIn("ranges.lin_vel_y = (-0.8, 0.8)", src)
        self.assertIn("ranges.ang_vel_z = (-1.0, 1.0)", src)

    def test_go2_reference_source_command_ranges(self):
        """本机有参考项目时，直接读它的 commands.ranges 比对。"""
        ref = next((p for p in REFERENCE_CANDIDATES if p.exists()), None)
        if ref is None:
            self.skipTest("本机没有 amp_go2 参考项目，跳过")
        cmds = _class(_module(ref), "commands")
        ranges = next(n for n in cmds.body if isinstance(n, ast.ClassDef) and n.name == "ranges")
        got = {}
        for node in ranges.body:
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        got[t.id] = ast.literal_eval(node.value)
        self.assertEqual(got["lin_vel_x"], [-1.2, 1.5])
        self.assertEqual(got["lin_vel_y"], [-0.8, 0.8])
        self.assertEqual(got["ang_vel_yaw"], [-1.0, 1.0])

    def test_runner_cfg_matches_reference_style_weights(self):
        agent = _module(AGENT_CFG)
        cls = _class(agent, "AMPGo2RunnerCfg")
        src = ast.unparse(cls)
        self.assertIn("amp_reward_coef = 0.2", src)
        self.assertIn("amp_task_reward_lerp = 0.8", src)
        # 风格项上限必须真的小（coef × (1-lerp) = 0.04/步），否则说明抄错了
        coef, lerp = 0.2, 0.8
        self.assertAlmostEqual(coef * (1 - lerp), 0.04, places=12)

    def test_tasks_are_registered_and_isolated(self):
        src = Path(TASKS_INIT).read_text(encoding="utf-8")
        for task in ("Imgo2-basemove-flat-amp-go2", "Imgo2-basemove-flat-amp-go2-play"):
            self.assertIn(task, src)
        self.assertIn("amp_env_cfg:Imgo2AmpGo2StyleEnvCfg", src)
        self.assertIn("amp_env_cfg:Imgo2AmpGo2StylePlayEnvCfg", src)
        self.assertIn("amp_rsl_rl_cfg:AMPGo2RunnerCfg", src)
        # 旧任务必须原样保留（对照与回滚都要靠它）
        self.assertIn("amp_env_cfg:Imgo2AmpMoveEnvCfg", src)
        self.assertIn("amp_rsl_rl_cfg:AMPRunnerCfg", src)

    def test_original_amp_recipe_is_untouched(self):
        """AMP-only 配方的每步系数不能被我这次改动带偏。"""
        fn = next(n for n in _class(self.tree, "Imgo2AmpMoveEnvCfg").body
                  if isinstance(n, ast.FunctionDef) and n.name == "_keep_only_amp_task_rewards")
        src = ast.unparse(fn)
        for const, value in (("TRACK_LIN_VEL_PER_STEP", "4.0"), ("TRACK_ANG_VEL_PER_STEP", "2.0"),
                             ("BASE_HEIGHT_PER_STEP", "-5.0"), ("LIN_VEL_Z_PER_STEP", "-1.0"),
                             ("ANG_VEL_XY_PER_STEP", "-0.05"), ("JOINT_POS_LIMITS_PER_STEP", "-2.0")):
            self.assertIn(f"{const} = {value}", src)


if __name__ == "__main__":
    unittest.main()
