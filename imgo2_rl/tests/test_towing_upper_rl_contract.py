"""Offline contract tests for the first upper-RL environment milestone."""

import ast
import importlib.util
from pathlib import Path
import sys
import unittest

try:
    import torch
except ImportError:  # The repository's offline contract checks also run outside the training env.
    torch = None


RL = Path(__file__).resolve().parents[1]
PKG = RL / "source/imgo2_rl/imgo2_rl/tasks/manager_based/towing"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


logic = load("towing_upper_logic_test", PKG / "upper_logic.py")


class UpperLogicTests(unittest.TestCase):
    def test_actor_contract_matches_paper_plan(self):
        spec = logic.UpperObservationSpec()
        self.assertEqual([name for name, _ in spec.terms],
                         ["cmd_vel", "last_action", "base_ang_vel", "projected_gravity",
                          "last_loco_action", "joint_pos", "joint_vel"])
        self.assertEqual(spec.frame_dim, 48)
        self.assertEqual(spec.actor_dim, 96)
        self.assertEqual(dict(spec.terms)["cmd_vel"], 3)
        self.assertEqual(dict(spec.terms)["last_action"], 3)

    def test_decoder_is_training_only_cart_mass(self):
        decoder = logic.DecoderSpec()
        self.assertEqual(decoder.dim, 1)
        self.assertEqual(decoder.terms, (("cart_mass", 1),))
        self.assertEqual(logic.normalize_decoder_targets((5,)), (-1.0,))
        self.assertEqual(logic.normalize_decoder_targets((10,)), (0.0,))
        self.assertEqual(logic.normalize_decoder_targets((15,)), (1.0,))

    def test_asymmetric_action_mapping_and_speed_limits(self):
        spec = logic.UpperActionSpec()
        self.assertEqual(logic.normalized_acceleration((-1, -1, -1), spec), (-1.0, -0.5, -1.0))
        self.assertEqual(logic.normalized_acceleration((1, 1, 1), spec), (0.5, 0.5, 1.0))
        self.assertEqual(logic.integrate_reference_speed((0, -0.3, -1), (-1, -1, -1), spec),
                         (0.0, -0.3, -1.0))
        self.assertEqual(logic.integrate_reference_speed((1, 0.3, 1), (1, 1, 1), spec),
                         (1.0, 0.3, 1.0))

    def test_command_schedule_contains_settle_tow_and_explicit_zero(self):
        self.assertEqual(logic.scheduled_command(0.5, 0.7, tow_start_s=1.0, stop_time_s=5.0), 0.0)
        self.assertEqual(logic.scheduled_command(2.0, 0.7, tow_start_s=1.0, stop_time_s=5.0), 0.7)
        self.assertEqual(logic.scheduled_command(5.0, 0.7, tow_start_s=1.0, stop_time_s=5.0), 0.0)

    def test_environment_is_not_registered_before_physics_adapter_is_complete(self):
        source = (PKG / "__init__.py").read_text("utf-8")
        self.assertNotIn("Imgo2-towing-upper", source)

    def test_policy_observation_excludes_privileged_load_signals(self):
        tree = ast.parse((PKG / "upper_env_cfg.py").read_text("utf-8"))
        policy = next(node for node in ast.walk(tree)
                      if isinstance(node, ast.ClassDef) and node.name == "PolicyCfg")
        names = {target.id for node in policy.body if isinstance(node, ast.Assign)
                 for target in node.targets if isinstance(target, ast.Name)}
        self.assertFalse(names & {"cart_mass", "cart_velocity", "rope_state", "ground_friction"})

    @unittest.skipIf(torch is None, "PyTorch is not installed in the offline-check interpreter")
    def test_mass_decoder_reward_is_masked_and_detached(self):
        decoder_module = load(
            "towing_mass_decoder_test",
            RL / "scripts/rl_lab/rl_lab/modules/towing_decoder.py")
        decoder = decoder_module.TowingMassDecoder(history_dim=96, hidden_dims=(8,))
        history = torch.zeros(3, 96, requires_grad=True)
        target = decoder(history).detach()
        reward = decoder_module.mass_identification_reward(
            decoder, history, target, torch.tensor([True, False, True]))
        self.assertEqual(tuple(reward.shape), (3,))
        self.assertTrue(torch.equal(reward, torch.tensor([1.0, 0.0, 1.0])))
        self.assertFalse(reward.requires_grad)

    @unittest.skipIf(torch is None, "PyTorch is not installed in the offline-check interpreter")
    def test_mass_decoder_trainer_updates_only_decoder(self):
        decoder_module = load(
            "towing_mass_decoder_train_test",
            RL / "scripts/rl_lab/rl_lab/modules/towing_decoder.py")
        decoder = decoder_module.TowingMassDecoder(history_dim=96, hidden_dims=(8,))
        trainer = decoder_module.MassDecoderTrainer(decoder)
        history = torch.randn(4, 96, requires_grad=True)
        loss = trainer.update(history, torch.zeros(4, 1))
        self.assertEqual(loss.ndim, 0)
        self.assertIsNone(history.grad)
        self.assertFalse(decoder.training)

    def test_reset_event_contract_has_all_v0_work_condition_axes(self):
        cfg = (PKG / "upper_env_cfg.py").read_text("utf-8")
        mdp = (PKG / "upper_mdp.py").read_text("utf-8")
        for fragment in (
            '"speed_range": (0.2, 1.0)',
            '"stop_time_range": (4.0, 6.0)',
            '"mass_range": (5.0, 15.0)',
            '"friction_range": (0.4, 1.2)',
            '"wheel_damping_range": (0.008, 0.032)',
        ):
            self.assertIn(fragment, cfg)
        self.assertIn("default_masses[ids_cpu] * scale[:, None]", mdp)
        self.assertIn("default_inertias[ids_cpu] * scale[:, None, None]", mdp)
        self.assertIn("permutation[: count // 2]", mdp)
        self.assertIn("elapsed_s < self.stop_time_s", mdp)
        self.assertIn("self._apply_towing_physics()", mdp)
        self.assertIn("self._physics_step % low_level_decimation", mdp)
        self.assertIn("-self.wheel_damping * self._cart.data.joint_vel", mdp)
        self.assertIn("SplitRopeModel(", mdp)
        self.assertIn("(towing & taut).float()", mdp)
        self.assertIn("(_term(env).cart_mass - 5.0) / 5.0 - 1.0", mdp)


if __name__ == "__main__":
    unittest.main()
