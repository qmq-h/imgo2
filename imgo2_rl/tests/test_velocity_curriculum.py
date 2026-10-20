"""速度指令课程（低层 PPO 训练用）的离线回归。

课程**是仓库里一直存在的**（upstream Isaac Lab 的 `command_levels_lin_vel` /
`command_levels_ang_vel`，实现在 `locomotion/velocity/mdp/curriculums.py`），
`base_move/rough_env_cfg.py` 之前把它置成 `None` 关掉了；2026-10-17 重新启用。

背景（`docs/towing/towing_ppo_lowlevel_swap_2026-10-17.md` §11）：跟踪奖励是
`exp(-err²/std²)`（std=0.5、权重 1.5），err ≳ 1.0 m/s 之后梯度掉到 1e-3 ⇒ 一次性采样到
(−1.5, 3.0) 的策略在高速档学不动。课程把起点压到 `range_multiplier[0] × 静态域`，
按跟踪奖励逐级放大。
"""

import ast
import importlib.util
from pathlib import Path
import unittest

import torch
from types import SimpleNamespace

RL = Path(__file__).resolve().parents[1]
PKG = RL / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity"
ROUGH = PKG / "base_move/rough_env_cfg.py"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


curriculum = load("velocity_curriculums_test", PKG / "mdp/curriculums.py")


class BuiltInCurriculumTests(unittest.TestCase):
    """`command_levels_lin_vel` / `command_levels_ang_vel` 的行为（带 episode 边界的假 env）。

    实现细节（决定测试怎么写）：初始化只在 `common_step_counter == 0` 那一步做；而 0 也是
    episode 边界 ⇒ **首步可能「初始化 + 立即升级」各一次**，所以下面用「低奖励起步」把初始化
    与升级拆开测。
    """

    def make_env(self, *, max_episode_length=400, episode_length_s=20.0, weight=1.5,
                 reward_per_s=1.4):
        ranges = SimpleNamespace(lin_vel_x=(-1.5, 3.0), lin_vel_y=(-1.5, 1.5),
                                 ang_vel_z=(-1.5, 1.5))
        term = SimpleNamespace(cfg=SimpleNamespace(ranges=ranges))
        env = SimpleNamespace(
            command_manager=SimpleNamespace(get_term=lambda name: term),
            reward_manager=SimpleNamespace(
                # 真实环境里 `_episode_sums` 每项是 (num_envs,) 的 **episode 累计和**
                # （= 每秒奖励 × episode 长度），课程再除以 `max_episode_length_s` 得每秒均值。
                _episode_sums={"track_lin_vel_xy_exp": torch.tensor([reward_per_s * episode_length_s]),
                               "track_ang_vel_z_exp": torch.tensor([reward_per_s * episode_length_s])},
                get_term_cfg=lambda name: SimpleNamespace(weight=weight)),
            max_episode_length=max_episode_length,
            max_episode_length_s=episode_length_s,
            common_step_counter=0,
            device="cpu",       # 内建课程用 env.device 建张量
        )
        return env, ranges

    def test_starts_at_range_multiplier_of_the_static_domain(self):
        env, ranges = self.make_env(reward_per_s=0.3)     # 不达标 ⇒ 首步只初始化
        curriculum.command_levels_lin_vel(env, [0], reward_term_name="track_lin_vel_xy_exp",
                                          range_multiplier=(0.2, 1.0))
        self.assertAlmostEqual(ranges.lin_vel_x[0], -0.3, places=5)
        self.assertAlmostEqual(ranges.lin_vel_x[1], 0.6, places=5)
        self.assertAlmostEqual(ranges.lin_vel_y[0], -0.3, places=5)
        self.assertAlmostEqual(ranges.lin_vel_y[1], 0.3, places=5)

    def test_high_reward_widens_by_one_step_at_episode_boundary(self):
        env, ranges = self.make_env()                     # 达标
        env.common_step_counter = 0
        curriculum.command_levels_lin_vel(env, [0], reward_term_name="track_lin_vel_xy_exp",
                                          range_multiplier=(0.2, 1.0))
        # 首步 = 初始化 + （因为是边界且达标）一次升级 ⇒ 起点各外扩 0.1
        self.assertAlmostEqual(ranges.lin_vel_x[0], -0.4, places=5)
        before = tuple(ranges.lin_vel_x)
        env.common_step_counter = 7                       # 非边界 ⇒ 不评估
        curriculum.command_levels_lin_vel(env, [0], reward_term_name="track_lin_vel_xy_exp")
        self.assertEqual(tuple(ranges.lin_vel_x), before)
        env.common_step_counter = 400                     # 边界 + 达标 ⇒ 再 ±0.1
        curriculum.command_levels_lin_vel(env, [0], reward_term_name="track_lin_vel_xy_exp")
        self.assertAlmostEqual(ranges.lin_vel_x[0], before[0] - 0.1, places=5)
        self.assertAlmostEqual(ranges.lin_vel_x[1], before[1] + 0.1, places=5)

    def test_low_reward_keeps_the_range(self):
        env, ranges = self.make_env(reward_per_s=0.3)     # 0.3 < 0.8×1.5 ⇒ 不达标
        curriculum.command_levels_lin_vel(env, [0], reward_term_name="track_lin_vel_xy_exp",
                                          range_multiplier=(0.2, 1.0))
        before = tuple(ranges.lin_vel_x)
        for step in (400, 800, 1200):
            env.common_step_counter = step
            curriculum.command_levels_lin_vel(env, [0], reward_term_name="track_lin_vel_xy_exp")
        self.assertEqual(tuple(ranges.lin_vel_x), before)

    def test_range_is_clamped_at_the_static_domain(self):
        env, ranges = self.make_env()
        curriculum.command_levels_lin_vel(env, [0], reward_term_name="track_lin_vel_xy_exp",
                                          range_multiplier=(0.2, 1.0))
        for step in range(1, 60):                         # 远超所需次数
            env.common_step_counter = 400 * step
            curriculum.command_levels_lin_vel(env, [0], reward_term_name="track_lin_vel_xy_exp")
        self.assertAlmostEqual(ranges.lin_vel_x[0], -1.5, places=5)
        self.assertAlmostEqual(ranges.lin_vel_x[1], 3.0, places=5)
        self.assertAlmostEqual(ranges.lin_vel_y[1], 1.5, places=5)
        self.assertAlmostEqual(ranges.ang_vel_z[1], 1.5, places=5)   # lin_vel 课程不动 ang_vel

    def test_ang_vel_curriculum_uses_its_own_term(self):
        env, ranges = self.make_env()
        curriculum.command_levels_ang_vel(env, [0], reward_term_name="track_ang_vel_z_exp",
                                          range_multiplier=(0.2, 1.0))
        # 首步：初始化 ±0.3 + 一次升级 ⇒ ±0.4；lin_vel_* 不受影响
        self.assertAlmostEqual(ranges.ang_vel_z[1], 0.4, places=5)
        self.assertAlmostEqual(ranges.lin_vel_x[1], 3.0, places=5)



class RoughCfgWiringTests(unittest.TestCase):
    def test_rough_task_enables_both_built_in_curricula(self):
        text = ROUGH.read_text(encoding="utf-8")
        # 不再置 None，而是给出 range_multiplier
        self.assertNotIn("self.curriculum.command_levels_lin_vel = None", text)
        self.assertNotIn("self.curriculum.command_levels_ang_vel = None", text)
        self.assertIn('self.curriculum.command_levels_lin_vel.params["range_multiplier"] = (0.2, 1.0)', text)
        self.assertIn('self.curriculum.command_levels_ang_vel.params["range_multiplier"] = (0.2, 1.0)', text)

    def test_curriculum_functions_are_the_repo_ones(self):
        """课程函数必须来自 `velocity/mdp/curriculums.py`（不另写一套）。"""
        text = ROUGH.read_text(encoding="utf-8")
        self.assertIn("command_levels_lin_vel", text)
        tree = ast.parse(text)
        self.assertIsInstance(tree, ast.Module)
        # 没有自造的并行课程项
        self.assertNotIn("velocity_range_curriculum", text)
        module = curriculum
        for name in ("command_levels_lin_vel", "command_levels_ang_vel"):
            self.assertTrue(callable(getattr(module, name)))


if __name__ == "__main__":
    unittest.main()
