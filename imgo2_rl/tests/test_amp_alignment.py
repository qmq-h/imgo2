"""Offline kinematics checks plus an optional CPU Torch AMP update regression.

Run: python -m unittest discover -s tests -p test_amp_alignment.py -v
"""

import ast
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/tools"))
from audit_amp_dataset import audit, axis_rotation, forward_kinematics, read_chain, rotate


class KinematicsTests(unittest.TestCase):
    def test_axis_rotation_known_right_angle(self):
        result = rotate(axis_rotation([0, 0, 1], math.pi / 2), [1, 0, 0])
        for actual, expected in zip(result, [0, 1, 0]):
            self.assertAlmostEqual(actual, expected)

    def test_zero_pose_leg_length(self):
        urdf = ET.parse(ROOT.parent / "imgo2_description/urdf/imgo2.urdf").getroot()
        result = forward_kinematics(read_chain(urdf, "FL_FOOT"), [0, 0, 0])
        for actual, expected in zip(result, [0.2205, 0.16185, -0.42575]):
            self.assertAlmostEqual(actual, expected)

    def test_dataset_matches_urdf_without_middle_leg_swap(self):
        result = audit(ROOT / "datasets/imgo2_motion", ROOT.parent / "imgo2_description/urdf/imgo2.urdf")
        for row in result["motions"]:
            with self.subTest(motion=row["file"]):
                error = row["fk_coordinate_rmse_m_by_order"]
                self.assertLess(error["FL_FR_RL_RR"], 0.005)
                self.assertGreater(error["FL_RL_FR_RR"], 0.20)


class RewardContractTests(unittest.TestCase):
    # These two guards exist because the AMP task was unconstructible without them:
    # (a) mdp.reset_amp_reference_state did not exist (amp_events was not imported anywhere),
    #     so importing amp_env_cfg raised AttributeError before any run could start;
    # (b) base_height_l2 kept RewardsCfg's body_names="" and Isaac Lab raises
    #     "Not all regular expressions are matched!" while resolving it at env construction.
    # Neither is reachable from a simulator-free test, so they are asserted statically here.

    CONFIG = ROOT / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/base_move/amp_env_cfg.py"
    AMP_EVENTS = ROOT / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/mdp/amp_events.py"

    def test_runners_keep_checkpoint_iter_in_sync(self):
        """checkpoint 的 'iter' 必须来自「当前轮」，否则 resume 会从错误位置继续。

        AMP 与 PPO 的 learn() 曾在循环内从不更新 self.current_learning_iteration，
        于是它一直是该次运行的起点（通常 0）：save() 把它写进 checkpoint，
        load() 又用它作为续训起点 —— 结果是续训从 0 重跑，且第一次保存覆盖 model_0.pt。
        HIM runner 一直是对的（循环内 self.current_learning_iteration = it）。
        2026-09-17 实测 2026-09-17_17-19-04 的 20 个 checkpoint 全部 iter=0，即此缺陷。
        """
        runners = ROOT / "scripts/rl_lab/rl_lab/runners"
        for name in ("amp_on_policy_runner.py", "ppo_on_policy_runner.py", "him_on_policy_runner.py"):
            with self.subTest(runner=name):
                src = (runners / name).read_text(encoding="utf-8")
                self.assertIn("self.current_learning_iteration = it", src,
                              f"{name} 的 learn() 循环内未同步迭代计数，checkpoint 的 iter 会失真")
                # save() 必须写这个字段，否则 load() 无从恢复
                self.assertIn("'iter': self.current_learning_iteration", src)

    def test_amp_reference_reset_term_is_imported_and_defined(self):
        cfg_src = self.CONFIG.read_text(encoding="utf-8-sig")
        events_src = self.AMP_EVENTS.read_text(encoding="utf-8-sig")
        self.assertIn("def reset_amp_reference_state", events_src)
        # the config must import amp_events itself; the mdp package does not re-export it
        self.assertIn("from imgo2_rl.tasks.manager_based.locomotion.velocity.mdp import amp_events", cfg_src)
        # and must reference the function through that module, not through `mdp`
        self.assertIn("func=mdp_amp.reset_amp_reference_state", cfg_src)
        self.assertNotIn("func=mdp.reset_amp_reference_state", cfg_src)
        self.assertIn("# from .amp_events import *", (self.AMP_EVENTS.parent / "__init__.py").read_text(
            encoding="utf-8-sig"), "if mdp re-exports amp_events, this guard needs revisiting")

    def test_amp_task_terms_match_reference_per_step_scale(self):
        """AMP-06：任务项的每步系数必须精确，高度项必须仍是任务里最强的项。

        单位语义：AMP 风格奖励是**每步**（runner 里直接算），而 Isaac Lab 的 RewardManager
        计算 `term × weight × dt`，weight 的语义是「每秒」。因此 weight 一律写成
        `每步系数 / step_dt`，由代码换算。

        2026-09-17 第二次调参（当前值）：速度项 1.0→4.0、高度项 -10→-5，并补
        `lin_vel_z_l2` / `ang_vel_xy_l2` / `joint_pos_limits` 三个轻量姿态约束。
        依据：a1 论文代码的真实每步系数是 1.67（配置里的 50 还会被 legged_robot 再乘 dt），
        go2 用 4.0/2.0 + 辅助项。历史见 README AMP-06 与 docs/amp_experiments_2026-09-17.md。
        """
        path = self.CONFIG
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))  # 27 个上游文件带 UTF-8 BOM
        cfg_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Imgo2AmpMoveEnvCfg")
        method = next(n for n in cfg_class.body if isinstance(n, ast.FunctionDef) and n.name == "_keep_only_amp_task_rewards")
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)

        expected_per_step = {
            "track_lin_vel_xy_exp": 4.0,
            "track_ang_vel_z_exp": 2.0,
            "base_height_l2": -5.0,
            "lin_vel_z_l2": -1.0,
            "ang_vel_xy_l2": -0.05,
            "joint_pos_limits": -2.0,
        }
        for dt in (0.01, 0.02, 0.04):
            with self.subTest(step_dt=dt):
                terms = SimpleNamespace(**{name: SimpleNamespace(weight=0.0, params={}) for name in (
                    *expected_per_step, "feet_slide")})
                terms.base_height_l2.params["asset_cfg"] = SimpleNamespace(body_names="")
                cfg = SimpleNamespace(rewards=terms, sim=SimpleNamespace(dt=dt / 4), decimation=4,
                                      base_link_name="base")
                namespace[method.name](cfg)

                # 其它奖励项仍被过滤掉
                self.assertIsNone(terms.feet_slide)

                # 高度项保留：目标 0.30 m、无高度扫描传感器、body 名已解析成具体 body
                self.assertAlmostEqual(terms.base_height_l2.params["target_height"], 0.30)
                self.assertIsNone(terms.base_height_l2.params["sensor_cfg"])
                self.assertEqual(terms.base_height_l2.params["asset_cfg"].body_names, ["base"])

                # weight × step_dt 必须精确还原每步系数（防止写回手算魔数或漏乘 dt）
                for name, per_step in expected_per_step.items():
                    self.assertAlmostEqual(getattr(terms, name).weight * dt, per_step, places=6)

                # 关键比例：高度项必须**强于**单个速度项（压住贴地爬行），
                # 但不能强过两个速度项之和（否则会像 Run1 那样压住移动）。
                lin_ps = terms.track_lin_vel_xy_exp.weight * dt
                ang_ps = terms.track_ang_vel_z_exp.weight * dt
                height_ps = abs(terms.base_height_l2.weight * dt)
                self.assertGreater(height_ps, lin_ps)
                self.assertLess(height_ps, lin_ps + ang_ps)

    def test_amp_reward_weights_are_derived_not_hardcoded(self):
        """防止回归成手写魔数：weight 必须由「每步系数 / step_dt」表达。"""
        src = self.CONFIG.read_text(encoding="utf-8-sig")
        for const in ("TRACK_LIN_VEL_PER_STEP", "TRACK_ANG_VEL_PER_STEP", "BASE_HEIGHT_PER_STEP",
                      "LIN_VEL_Z_PER_STEP", "ANG_VEL_XY_PER_STEP", "JOINT_POS_LIMITS_PER_STEP"):
            self.assertIn(const, src)
        # 不应出现直接写死的 weight 数值
        for magic in ("weight = 50.0", "weight = 2500", "weight = -500", "1.0 / step_dt",
                      "0.3 / step_dt", "-10.0 / step_dt"):
            self.assertNotIn(magic, src)


HAS_TORCH = all(importlib.util.find_spec(name) is not None for name in ("torch", "numpy"))


@unittest.skipUnless(HAS_TORCH, "CPU regression needs the training environment's torch and numpy")
class AMPUpdateTests(unittest.TestCase):
    def test_raw_statistics_and_normalized_gradient_penalty(self):
        import numpy as np
        import torch

        def load_cpu_module(relative, **dependencies):
            # The package __init__ imports Isaac Lab export utilities. Load the
            # real CPU implementations without those simulator-only imports.
            path = ROOT / "scripts/rl_lab/rl_lab" / relative
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))  # 27 个上游文件带 UTF-8 BOM
            tree.body = [node for node in tree.body if not (
                isinstance(node, ast.ImportFrom) and (node.module or "").startswith("rl_lab"))]
            scope = {"__name__": "amp_cpu_regression", **dependencies}
            exec(compile(tree, str(path), "exec"), scope)
            return scope

        utils = load_cpu_module("utils/utils.py")
        ActorCritic = load_cpu_module("modules/actor_critic.py")["ActorCritic"]
        storage = load_cpu_module("storage/rollout_storage.py",
                                  split_and_pad_trajectories=utils["split_and_pad_trajectories"])
        replay = load_cpu_module("storage/replay_buffer.py")
        AMPPPO = load_cpu_module("algorithms/amp_ppo.py", ActorCritic=ActorCritic,
                                 RolloutStorage=storage["RolloutStorage"],
                                 ReplayBuffer=replay["ReplayBuffer"])["AMPPPO"]
        AMPDiscriminator = load_cpu_module("algorithms/amp_discriminator.py")["AMPDiscriminator"]
        Normalizer = utils["Normalizer"]

        torch.manual_seed(0)
        normalizer = Normalizer(43)
        normalizer.mean[:] = 2.0
        normalizer.var[:] = 4.0
        expert = torch.full((4, 43), 0.3)
        next_expert = torch.full((4, 43), 0.4)

        class ExpertData:
            def feed_forward_generator(self, num_mini_batch, mini_batch_size):
                for _ in range(num_mini_batch):
                    yield expert, next_expert

        actor = ActorCritic(3, 3, 2, actor_hidden_dims=[8], critic_hidden_dims=[8])
        discriminator = AMPDiscriminator(86, 2.0, [8], "cpu", 0.1)
        algorithm = AMPPPO(actor, discriminator, ExpertData(), normalizer,
                           device="cpu", amp_replay_buffer_size=16)
        algorithm.init_storage(2, 2, [3], [3], [2])
        obs = torch.zeros(2, 3)
        for _ in range(2):
            algorithm.act(obs, obs, torch.full((2, 43), 0.1))
            algorithm.process_env_step(torch.ones(2), torch.zeros(2, dtype=torch.bool), {},
                                       torch.full((2, 43), 0.2))
        algorithm.compute_returns(obs)
        expected_expert = normalizer.normalize_torch(expert, "cpu")
        expected_next_expert = normalizer.normalize_torch(next_expert, "cpu")
        with patch.object(normalizer, "update", wraps=normalizer.update) as update, \
             patch.object(discriminator, "compute_grad_pen", wraps=discriminator.compute_grad_pen) as penalty:
            losses = algorithm.update()
        self.assertTrue(all(math.isfinite(loss) for loss in losses))
        self.assertEqual(update.call_count, 2)
        np.testing.assert_allclose(update.call_args_list[0].args[0], 0.1)
        np.testing.assert_allclose(update.call_args_list[1].args[0], 0.3)
        torch.testing.assert_close(penalty.call_args.args[0], expected_expert)
        torch.testing.assert_close(penalty.call_args.args[1], expected_next_expert)


if __name__ == "__main__":
    unittest.main()
