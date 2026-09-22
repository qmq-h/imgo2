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
                         ["cmd_vel", "reference_command", "last_action", "base_ang_vel", "projected_gravity",
                          "last_loco_action", "joint_pos", "joint_vel"])
        self.assertEqual(spec.frame_dim, 51)
        self.assertEqual(spec.decoder_dim, 5)
        self.assertEqual(spec.actor_dim, 56)
        self.assertEqual(dict(spec.terms)["cmd_vel"], 3)
        self.assertEqual(dict(spec.terms)["last_action"], 3)

    def test_decoder_targets_velocity_mass_and_towing_force(self):
        decoder = logic.DecoderSpec()
        self.assertEqual(decoder.dim, 5)
        self.assertEqual(
            decoder.terms,
            (("robot_velocity_xy", 2), ("cart_mass", 1), ("towing_force_xy", 2)))
        self.assertEqual(
            logic.normalize_decoder_targets((0, 0, 5, 0, 0)),
            (0.0, 0.0, -1.0, 0.0, 0.0))
        self.assertEqual(
            logic.normalize_decoder_targets((0.5, -0.25, 10, 10, -10)),
            (0.5, -0.5, 0.0, 0.5, -0.5))
        self.assertEqual(
            logic.normalize_decoder_targets((2, -2, 15, 30, -30)),
            (1.0, -1.0, 1.0, 0.75, -0.75))
        self.assertAlmostEqual(logic.denormalize_force(0.5), 10.0)
        self.assertAlmostEqual(logic.denormalize_force(-0.75), -30.0)

    def test_asymmetric_action_mapping_and_speed_limits(self):
        spec = logic.UpperActionSpec()
        self.assertEqual(logic.normalized_acceleration((-1, -1, -1), spec), (-1.0, -0.5, -1.0))
        self.assertEqual(logic.normalized_acceleration((1, 1, 1), spec), (0.5, 0.5, 1.0))
        self.assertEqual(logic.normalized_acceleration((0, 0, 0), spec), (0.0, 0.0, 0.0))
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
    def test_decoder_output_is_detached_before_actor(self):
        decoder_module = load(
            "towing_dynamics_decoder_test",
            RL / "scripts/rl_lab/rl_lab/modules/towing_decoder.py")
        decoder = decoder_module.TowingDynamicsDecoder(
            frame_dim=51, feature_dim=8, hidden_dim=8)
        frames = torch.zeros(3, 51, requires_grad=True)
        prediction, hidden = decoder(frames)
        self.assertEqual(tuple(prediction.shape), (3, 5))
        self.assertEqual(tuple(hidden.shape), (1, 3, 8))
        actor_obs = decoder_module.augment_actor_observation(frames, prediction)
        self.assertEqual(tuple(actor_obs.shape), (3, 56))
        actor_obs.sum().backward()
        self.assertTrue(all(parameter.grad is None for parameter in decoder.parameters()))

    @unittest.skipIf(torch is None, "PyTorch is not installed in the offline-check interpreter")
    def test_decoder_force_weighted_mass_supervision_and_update(self):
        decoder_module = load(
            "towing_dynamics_decoder_train_test",
            RL / "scripts/rl_lab/rl_lab/modules/towing_decoder.py")
        force = torch.tensor([[0.0, 0.0], [1.0, 0.0], [11.0, 0.0]])
        weight = decoder_module.mass_supervision_weight(force)
        self.assertTrue(torch.equal(weight, torch.tensor([0.0, 0.0, 0.5])))

        decoder = decoder_module.TowingDynamicsDecoder(
            frame_dim=51, feature_dim=8, hidden_dim=8)
        trainer = decoder_module.DynamicsDecoderTrainer(decoder)
        frames = torch.randn(4, 3, 51, requires_grad=True)
        targets = torch.zeros(4, 3, 5)
        weights = torch.zeros(4, 3)
        weights[:, 1] = 1.0
        loss = trainer.update(frames, targets, weights)
        self.assertEqual(loss.ndim, 0)
        self.assertIsNone(frames.grad)
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
            '"no_cart_fraction": 0.125',
            '"no_cart_lateral_offset": 2.0',
        ):
            self.assertIn(fragment, cfg)
        self.assertIn("default_masses[ids_cpu] * scale[:, None]", mdp)
        self.assertIn("default_inertias[ids_cpu] * scale[:, None, None]", mdp)
        self.assertIn("torch.randint(0, 2, (count,)", mdp)
        self.assertIn("elapsed_s < self.stop_time_s", mdp)
        self.assertIn("self._apply_towing_physics()", mdp)
        self.assertIn("self._physics_step % low_level_decimation", mdp)
        self.assertIn("-self.wheel_damping * self._cart.data.joint_vel", mdp)
        self.assertIn("SplitRopeModel(", mdp)
        self.assertIn("force / (force.abs() + 10.0)", mdp)
        self.assertIn("(force - minimum_force).clamp_min(0.0)", mdp)
        self.assertIn('params={"minimum_force": 1.0, "force_scale": 10.0}', cfg)
        self.assertIn("(term.cart_mass - 5.0) / 5.0 - 1.0", mdp)

    def test_no_cart_environments_are_physically_and_semantically_masked(self):
        mdp = (PKG / "upper_mdp.py").read_text("utf-8")
        self.assertIn("self.cart_present = torch.ones", mdp)
        self.assertIn("force_on_robot), dim=-1) * present", mdp)
        self.assertIn("force_on_cart), dim=-1) * present", mdp)
        self.assertIn("cart_state[~cart_present, 1] += no_cart_lateral_offset", mdp)
        self.assertIn("* term.cart_present[:, 0]", mdp)
        self.assertIn("term.cart_present.float()", mdp)

    def test_safety_and_history_producers_are_live(self):
        cfg = (PKG / "upper_env_cfg.py").read_text("utf-8")
        mdp = (PKG / "upper_mdp.py").read_text("utf-8")
        self.assertIn('prim_path="{ENV_REGEX_NS}/Cart/base_link"', cfg)
        for body in ("wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"):
            self.assertIn(f'prim_path="{{ENV_REGEX_NS}}/Cart/{body}"', cfg)
        self.assertEqual(cfg.count('filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/.*"]'), 5)
        self.assertIn("sensor.data.force_matrix_w", mdp)
        self.assertIn("maximum_force > self.cfg.collision_force_threshold", mdp)
        self.assertIn("self.rope_state[:, 0] =", mdp)
        self.assertIn("self.towing_force_b[:] = force_robot_b[:, 0, :2]", mdp)
        self.assertIn("frame = ObsTerm(func=mdp.policy_frame)", cfg)
        self.assertNotIn("cmd_vel = ObsTerm", cfg)
        self.assertIn('obs_groups = {"actor": ["policy"], "critic": ["critic"]}',
                      (PKG / "agents/upper_ppo_cfg.py").read_text("utf-8"))

    def test_upper_ppo_normalizes_only_privileged_critic_input(self):
        cfg = (PKG / "agents/upper_ppo_cfg.py").read_text("utf-8")
        self.assertIn("obs_normalization=False", cfg)
        self.assertIn("obs_normalization=True", cfg)
        self.assertEqual(cfg.count('rnn_type="gru"'), 2)
        self.assertIn("RslRlRNNModelCfg", cfg)
        self.assertNotIn("empirical_normalization", cfg)

    def test_hierarchical_frequency_contract_is_200_50_20_hz(self):
        env_cfg = (PKG / "upper_env_cfg.py").read_text("utf-8")
        mdp = (PKG / "upper_mdp.py").read_text("utf-8")
        policy_cfg = (PKG / "utils/policy_cfg.py").read_text("utf-8")
        self.assertIn("self.sim.dt = 0.005", env_cfg)
        self.assertIn("self.decimation = 10", env_cfg)
        self.assertIn("low_level_control_dt=0.02", env_cfg)
        self.assertIn("control_dt=0.02", policy_cfg)
        self.assertIn("cfg.physics_dt - env.physics_dt", mdp)
        self.assertIn("cfg.upper_control_dt - env.step_dt", mdp)
        self.assertIn("cfg.low_level_control_dt - self._policy_cfg.control_dt", mdp)

    def test_post_stop_distance_excludes_initial_settle_phase(self):
        mdp = (PKG / "upper_mdp.py").read_text("utf-8")
        self.assertIn("elapsed_s >= term.stop_time_s", mdp)

    def test_post_stop_towing_force_is_gated_and_bounded(self):
        env_cfg = (PKG / "upper_env_cfg.py").read_text("utf-8")
        mdp = (PKG / "upper_mdp.py").read_text("utf-8")
        self.assertIn("stop_towing_force = RewTerm(func=mdp.post_stop_towing_force", env_cfg)
        self.assertIn('params={"force_scale": 10.0}', env_cfg)
        self.assertIn("post_stop = (elapsed_s >= term.stop_time_s).float()", mdp)
        self.assertIn("normalized_force = force / (force + force_scale)", mdp)
        self.assertIn("normalized_force * post_stop * term.cart_present[:, 0]", mdp)


if __name__ == "__main__":
    unittest.main()
