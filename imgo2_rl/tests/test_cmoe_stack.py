"""Tensor-level regression tests for the repository-local CMoE port."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


RL_LAB_ROOT = Path(__file__).resolve().parents[1] / "scripts" / "rl_lab"
if str(RL_LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_LAB_ROOT))

try:
    import torch

    from rl_lab.modules.cmoe_actor_critic import CMoEActorCritic
    from rl_lab.storage.cmoe_rollout_storage import CMoERolloutStorage
    from rl_lab.utils.export_policy import _CMoEPolicyExporter
except ModuleNotFoundError as error:
    torch = None
    IMPORT_ERROR = error
else:
    IMPORT_ERROR = None


@unittest.skipIf(torch is None, f"PyTorch unavailable: {IMPORT_ERROR}")
class TestCMoEStack(unittest.TestCase):
    def setUp(self):
        self.batch = 8
        self.proprio_dim = 45
        self.history_steps = 10
        self.terrain_dim = 77
        self.actor_obs_dim = self.history_steps * self.proprio_dim + self.terrain_dim
        self.critic_dim = self.proprio_dim + 3 + self.terrain_dim
        self.model = CMoEActorCritic(
            num_actor_obs=self.actor_obs_dim,
            num_critic_obs=self.critic_dim,
            num_one_step_obs=self.proprio_dim,
            num_actions=12,
            history_steps=self.history_steps,
            terrain_obs_dim=self.terrain_dim,
            num_experts=5,
        )

    def test_forward_value_and_auxiliary_losses(self):
        observations = torch.randn(self.batch, self.actor_obs_dim)
        critic_observations = torch.randn(self.batch, self.critic_dim)
        next_critic_observations = torch.randn(self.batch, self.critic_dim)

        actions = self.model.act(observations)
        values = self.model.evaluate(critic_observations)
        contrastive_loss = self.model.compute_contrastive_loss(observations)
        estimator_losses = self.model.update_estimators(
            observations,
            critic_observations,
            next_critic_observations,
            lr=1.0e-3,
        )

        self.assertEqual(tuple(actions.shape), (self.batch, 12))
        self.assertEqual(tuple(values.shape), (self.batch, 1))
        self.assertEqual(tuple(self.model.gate_weights.shape), (self.batch, 5))
        self.assertTrue(torch.allclose(self.model.gate_weights.sum(dim=-1), torch.ones(self.batch)))
        self.assertTrue(torch.isfinite(contrastive_loss))
        self.assertEqual(len(estimator_losses), 8)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in estimator_losses))

    def test_storage_uses_configured_actor_dimension(self):
        storage = CMoERolloutStorage(
            num_envs=4,
            num_transitions_per_env=3,
            obs_shape=[self.actor_obs_dim],
            privileged_obs_shape=[self.critic_dim],
            actions_shape=[12],
            device="cpu",
        )
        self.assertEqual(tuple(storage.observations.shape), (3, 4, self.actor_obs_dim))
        self.assertEqual(tuple(storage.privileged_observations.shape), (3, 4, self.critic_dim))

    def test_exporter_matches_deterministic_policy(self):
        self.model.eval()
        observations = torch.randn(self.batch, self.actor_obs_dim)
        exporter = _CMoEPolicyExporter(self.model).eval()
        scripted_exporter = torch.jit.script(exporter)
        with torch.inference_mode():
            expected = self.model.act_inference(observations)
            actual = exporter(observations)
            scripted_actual = scripted_exporter(observations)
        self.assertTrue(torch.allclose(actual, expected, atol=1.0e-6, rtol=1.0e-5))
        self.assertTrue(torch.allclose(scripted_actual, expected, atol=1.0e-6, rtol=1.0e-5))


if __name__ == "__main__":
    unittest.main()
