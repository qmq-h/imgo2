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

    def test_height_scanner_restored_for_amp_obs_only(self):
        """9 射线扫描器要恢复，但**只服务 AMP 观测的根高**：rl_amp 配方没有高度奖励。"""
        self.assertIn("scene.height_scanner_base = MySceneCfg.height_scanner_base", self.src)
        self.assertNotIn("base_height_l2", self.src)

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

    def test_privileged_height_scan_goes_to_critic_only(self):
        """187 维高度网格**只喂 critic**（非对称 actor-critic），actor 必须仍是 45 维盲走。"""
        self.assertIn("scene.height_scanner = MySceneCfg.height_scanner", self.src)
        self.assertIn("observations.critic.height_scan = ObsTerm", self.src)
        # actor 侧不得出现 height_scan 的 ObsTerm（保持 None ⇒ 45 维、部署契约不变）
        self.assertNotIn("observations.policy.height_scan = ObsTerm", self.src)
        self.assertIn("observations.policy.height_scan is None", self.src)   # 断言守住这条
        # 扫描项的裁剪/缩放要与基类 CriticCfg 一致（clip ±1、scale 1.0）
        self.assertIn("clip=(-1.0, 1.0)", self.src)

    def test_critic_dim_contract_235(self):
        """critic = 48 + 187 = 235：与 amp_go2 的 num_privileged_obs 相同（静态核算）。"""
        vel = _tree(ROOT / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py")
        scene = _class(vel, "MySceneCfg")
        scanner = next(n for n in scene.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "height_scanner" for t in n.targets))
        grid = next(n for n in ast.walk(scanner) if isinstance(n, ast.Call)
                    and getattr(n.func, "attr", None) == "GridPatternCfg")
        kw = {k.arg: ast.literal_eval(k.value) for k in grid.keywords}
        nx = int(round(kw["size"][0] / kw["resolution"])) + 1
        ny = int(round(kw["size"][1] / kw["resolution"])) + 1
        self.assertEqual(nx * ny, 187, "高度网格维度变了，privileged obs 维度也要跟着更新")
        self.assertEqual(48 + nx * ny, 235, "critic 维度应为 235（= amp_go2 的 num_privileged_obs）")

    def test_flat_variants_keep_critic_blind(self):
        """平地版（AMP-only / go2 / rlamp）不得加 height scan：保持 48 维与干净配对。"""
        for cls in ("Imgo2AmpMoveEnvCfg", "Imgo2AmpGo2StyleEnvCfg", "Imgo2AmpRLAmpEnvCfg"):
            src = _init_src(_class(self.tree, cls))
            self.assertNotIn("height_scanner = MySceneCfg", src)
            self.assertNotIn("critic.height_scan = ObsTerm", src)

    def test_rough_uses_rlamp_recipe(self):
        """粗糙版的任务奖励 = rl_amp 的两项（由 helper 套用），不得自己写 weight、不得引入 go2 项。"""
        self.assertIn("apply_rlamp_task_rewards(self)", self.src)
        self.assertNotIn(".weight =", self.src)
        for term in ("feet_air_time", "undesired_contacts", "action_rate_l2", "joint_acc_l2",
                     "lin_vel_z_l2", "ang_vel_xy_l2", "joint_pos_limits"):
            self.assertNotIn(term, self.src, f"粗糙 AMP 版不应引入 {term}")
        # 原来的 AMP-only 6 项配方必须原封不动（它属于平地版）
        keep = next(n for n in _class(self.tree, "Imgo2AmpMoveEnvCfg").body
                    if isinstance(n, ast.FunctionDef) and n.name == "_keep_only_amp_task_rewards")
        flat = ast.unparse(keep)
        for needle in ("TRACK_LIN_VEL_PER_STEP = 4.0", "BASE_HEIGHT_PER_STEP = -5.0"):
            self.assertIn(needle, flat)

    def test_play_variant_reuses_play_overrides(self):
        play = _class(self.tree, "Imgo2AmpRoughPlayEnvCfg")
        self.assertIn("apply_amp_play_overrides(self)", _init_src(play))

    def test_tasks_registered(self):
        src = Path(TASKS_INIT).read_text(encoding="utf-8")
        for task in ("Imgo2-basemove-rough-amp-rlamp", "Imgo2-basemove-rough-amp-rlamp-play"):
            self.assertIn(task, src)
        self.assertIn("amp_env_cfg:Imgo2AmpRoughEnvCfg", src)
        self.assertIn("amp_env_cfg:Imgo2AmpRoughPlayEnvCfg", src)
        # 平地版（AMP-only 6 项）、go2 版、以及成对的平地 rlamp 版都必须仍在
        self.assertIn("amp_env_cfg:Imgo2AmpMoveEnvCfg", src)
        self.assertIn("amp_env_cfg:Imgo2AmpGo2StyleEnvCfg", src)
        self.assertIn("amp_env_cfg:Imgo2AmpRLAmpEnvCfg", src)


if __name__ == "__main__":
    unittest.main()
