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
        urdf = ET.parse(ROOT / "source/imgo2_rl/data/Imgo2/Imgo2_urdf/urdf/imgo2.urdf").getroot()
        result = forward_kinematics(read_chain(urdf, "FL_FOOT"), [0, 0, 0])
        for actual, expected in zip(result, [0.2205, 0.16185, -0.42575]):
            self.assertAlmostEqual(actual, expected)

    def test_dataset_matches_urdf_without_middle_leg_swap(self):
        result = audit(ROOT / "datasets/imgo2_motion", ROOT / "source/imgo2_rl/data/Imgo2/Imgo2_urdf/urdf/imgo2.urdf")
        for row in result["motions"]:
            with self.subTest(motion=row["file"]):
                error = row["fk_coordinate_rmse_m_by_order"]
                self.assertLess(error["FL_FR_RL_RR"], 0.005)
                self.assertGreater(error["FL_RL_FR_RR"], 0.20)


class RewardContractTests(unittest.TestCase):
    def test_height_survives_filter_and_per_step_scale_is_preserved(self):
        # Execute the actual config method without importing Isaac Sim.
        path = ROOT / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/base_move/amp_env_cfg.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        cfg_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Imgo2AmpMoveEnvCfg")
        method = next(n for n in cfg_class.body if isinstance(n, ast.FunctionDef) and n.name == "_keep_only_amp_task_rewards")
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
        for dt in (0.01, 0.02, 0.04):
            with self.subTest(step_dt=dt):
                terms = SimpleNamespace(**{name: SimpleNamespace(weight=0.0, params={}) for name in (
                    "track_lin_vel_xy_exp", "track_ang_vel_z_exp", "base_height_l2", "feet_slide")})
                cfg = SimpleNamespace(rewards=terms, sim=SimpleNamespace(dt=dt / 4), decimation=4)
                namespace[method.name](cfg)
                self.assertIsNone(terms.feet_slide)
                self.assertIsNotNone(terms.base_height_l2)
                target = terms.base_height_l2.params["target_height"]
                self.assertAlmostEqual(target, 0.30)
                self.assertIsNone(terms.base_height_l2.params["sensor_cfg"])
                standing_reward = dt * (terms.track_lin_vel_xy_exp.weight + terms.track_ang_vel_z_exp.weight)
                crawling_reward = standing_reward + dt * terms.base_height_l2.weight * (0.10 - target)**2
                self.assertAlmostEqual(standing_reward, 1.3)
                self.assertAlmostEqual(standing_reward - crawling_reward, 0.4)


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
            tree = ast.parse(path.read_text(encoding="utf-8"))
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
