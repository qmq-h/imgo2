"""粗糙地形版 AMP（`Imgo2-basemove-rough-amp`）的离线契约测试。

设计意图（2026-09-18）：**配方不变、只改地形相关的最小四处**——这样才能把"地形"当成
单变量，与平地版 `Imgo2-basemove-flat-amp` 直接对照。这些约束都写在
`amp_env_cfg.Imgo2AmpRoughEnvCfg.__post_init__` 里，而 Isaac Lab 的配置类**必须**在 Isaac Sim
应用内才能实例化（`import omni.log`），本机沙箱跑不起来 ⇒ 用 AST 在源码层面钉住。

Run: python3 -m unittest discover -s tests -p test_amp_rough_recipe.py
"""

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
AMP_CFG = ROOT / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/base_move/amp_env_cfg.py"
OBS_PY = ROOT / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/mdp/observations.py"
TASKS_INIT = ROOT / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/base_move/__init__.py"
ROUGH_CFG = ROOT / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/base_move/rough_env_cfg.py"


def _tree(path):
    return ast.parse(Path(path).read_text(encoding="utf-8-sig"))


def _class(tree, name):
    hits = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == name]
    assert hits, f"找不到类 {name}"
    return hits[0]


def _init_src(cls):
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__post_init__")
    return ast.unparse(fn)


class RoughRecipeTests(unittest.TestCase):
    def setUp(self):
        self.tree = _tree(AMP_CFG)
        self.rough = _class(self.tree, "Imgo2AmpRoughEnvCfg")
        self.src = _init_src(self.rough)

    def test_terrain_and_curriculum_restored(self):
        self.assertIn("terrain_type = 'generator'", self.src)
        self.assertIn("terrain_generator = ROUGH_TERRAINS_CFG", self.src)
        self.assertIn("terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)", self.src)

    def test_sub_terrain_ranges_match_existing_rough_task(self):
        """子地形范围必须与仓库已有 PPO rough 任务一致（机器人只有 0.30 m 站高）。"""
        ppo = _init_src(_class(_tree(ROUGH_CFG), "Imgo2RoughEnvCfg"))
        for needle in ("grid_height_range = (0.025, 0.1)",
                       "noise_range = (0.01, 0.06)",
                       "noise_step = 0.01",
                       "step_height_range = (0.025, 0.08)"):
            self.assertIn(needle, self.src, f"AMP rough 里缺少 {needle}")
            self.assertIn(needle, ppo, f"PPO rough 里没有 {needle}（两边应一致）")

    def test_height_reward_is_terrain_relative(self):
        """奖励用的 9 射线扫描器要恢复，且 base_height_l2 必须指向它（地形相对高度）。"""
        self.assertIn("scene.height_scanner_base = MySceneCfg.height_scanner_base", self.src)
        self.assertIn(
            "rewards.base_height_l2.params['sensor_cfg'] = SceneEntityCfg('height_scanner_base')",
            self.src)

    def test_amp_root_z_is_terrain_relative(self):
        """AMP 观测的根高必须传同一个扫描器，否则这一维被地形起伏主导。"""
        self.assertIn(
            "observations.amp.root_z.params['sensor_cfg'] = SceneEntityCfg('height_scanner_base')",
            self.src)
        # 观测函数本身要支持这个可选参数，且不改平地行为（默认 None）
        obs = _tree(OBS_PY)
        fn = next(n for n in obs.body if isinstance(n, ast.FunctionDef) and n.name == "amp_root_z")
        args = [a.arg for a in fn.args.args]
        self.assertIn("sensor_cfg", args)
        body = ast.unparse(fn)
        self.assertIn("if sensor_cfg is not None:", body)
        self.assertIn("ray_hits_w", body)
        # 默认值必须是 None（平地版一行不改）
        defaults = [ast.unparse(d) for d in fn.args.defaults]
        self.assertIn("None", defaults)

    def test_reference_state_init_disabled_on_rough(self):
        """参考状态初始化写的是绝对根高、没有地形补偿 ⇒ 粗糙地形必须关掉。"""
        self.assertIn("events.reference_state_initialization = None", self.src)

    def test_actor_stays_blind_45d(self):
        """不把 187 维高度网格加进 actor：45 维盲走，部署契约不变。"""
        self.assertNotIn("observations.policy.height_scan = ObsTerm", self.src)
        self.assertNotIn("scene.height_scanner = MySceneCfg.height_scanner", self.src)
        self.assertNotIn("observations.critic.height_scan = ObsTerm", self.src)

    def test_recipe_unchanged_from_flat_amp(self):
        """粗糙版不得改动奖励配方（6 项、每步系数、AMP 权重）—— 那是平地版的责任。"""
        flat_cls = _class(self.tree, "Imgo2AmpMoveEnvCfg")
        keep = next(n for n in flat_cls.body
                    if isinstance(n, ast.FunctionDef) and n.name == "_keep_only_amp_task_rewards")
        flat = ast.unparse(keep)
        for needle in ("TRACK_LIN_VEL_PER_STEP = 4.0", "TRACK_ANG_VEL_PER_STEP = 2.0",
                       "BASE_HEIGHT_PER_STEP = -5.0"):
            self.assertIn(needle, flat)
        # 粗糙版的 __post_init__ 里不能出现任何 weight 赋值
        self.assertNotIn(".weight =", self.src)
        # 也不得引入 amp_go2 的任务项
        for term in ("feet_air_time", "undesired_contacts", "action_rate_l2", "joint_acc_l2"):
            self.assertNotIn(term, self.src, f"粗糙 AMP 版不应引入 {term}（那是 go2 路线）")

    def test_play_variant_reuses_play_overrides(self):
        play = _class(self.tree, "Imgo2AmpRoughPlayEnvCfg")
        self.assertIn("apply_amp_play_overrides(self)", _init_src(play))

    def test_tasks_registered(self):
        src = Path(TASKS_INIT).read_text(encoding="utf-8")
        for task in ("Imgo2-basemove-rough-amp", "Imgo2-basemove-rough-amp-play"):
            self.assertIn(task, src)
        self.assertIn("amp_env_cfg:Imgo2AmpRoughEnvCfg", src)
        self.assertIn("amp_env_cfg:Imgo2AmpRoughPlayEnvCfg", src)
        # 平地版与 go2 版必须仍在
        self.assertIn("amp_env_cfg:Imgo2AmpMoveEnvCfg", src)
        self.assertIn("amp_env_cfg:Imgo2AmpGo2StyleEnvCfg", src)


if __name__ == "__main__":
    unittest.main()
