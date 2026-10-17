"""CE Net regression checks（2026-10-17 合并后：显式 **5 维**、契约 **v5**）。

结构：GRU 并列「显式 5 维（vel 2 + force 3）」与「latent 16」，21 维拼接后重建下一帧；
本文件覆盖源码契约（v5 版本号、`reconstruction_coef`）与需要 torch 的张量行为
（并列头梯度、显式量不依赖 latent 采样、下一帧损失与 done 掩码、全终止 batch、
GRU 按环境重置、真实 runner 循环的帧/下一帧配对与缓冲区快照）。
"""

import ast
from collections import deque
import importlib.util
from pathlib import Path
import time
from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:
    torch = None


ROOT = Path(__file__).resolve().parents[1]
MODULES = ROOT / "scripts/rl_lab/rl_lab"


class SourceContractTests(unittest.TestCase):
    def test_checkpoint_versions_and_reconstruction_config(self):
        for path, name in (
            (MODULES / "runners/towing_on_policy_runner.py", "TOWING_CONTRACT_VERSION"),
            (ROOT / "scripts/towing/upper_policy_runtime.py", "CHECKPOINT_CONTRACT_VERSION"),
            (ROOT / "scripts/towing/play_towing_test.py", "UPPER_CONTRACT_VERSION"),
        ):
            tree = ast.parse(path.read_text("utf-8"))
            value = next(node.value.value for node in tree.body
                         if isinstance(node, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == name for t in node.targets))
            self.assertEqual(value, 5)
        cfg = ast.parse((MODULES / "config/towing_algorithm_cfg.py").read_text("utf-8"))
        decoder = next(n for n in cfg.body if isinstance(n, ast.ClassDef)
                       and n.name == "TowingDecoderCfg")
        values = {n.target.id: n.value.value for n in decoder.body
                  if isinstance(n, ast.AnnAssign) and isinstance(n.value, ast.Constant)}
        self.assertEqual(values["reconstruction_coef"], 1.0)
        self.assertEqual(values["frame_dim"], 58)
        self.assertEqual(values["latent_dim"], 16)

    def test_no_mass_head_survives_the_ce_net_merge(self):
        """合并只拿 CE Net 结构，**不**恢复质量头（2026-10-16 的用户准则）。"""
        source = (MODULES / "modules/towing_decoder.py").read_text("utf-8")
        self.assertNotIn("mass_head", source)
        self.assertNotIn("MASS_DIM", source)
        self.assertIn("OUTPUT_DIM = VELOCITY_DIM + FORCE_DIM", source)


@unittest.skipIf(torch is None, "Requires torch on the training machine")
class ContextEstimatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "ce_net_test_module", MODULES / "modules/towing_decoder.py")
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def setUp(self):
        torch.manual_seed(19)
        self.net = self.module.TowingDynamicsDecoder(feature_dim=8, hidden_dim=12)

    def test_parallel_heads_and_reconstruction_gradient(self):
        frames = torch.randn(3, 2, 58)
        explicit, mu, log_var, state = self.net.encode(frames)
        self.assertEqual(explicit.shape, (3, 2, 5))
        self.assertEqual(mu.shape, (3, 2, 16))
        self.assertEqual(state.shape, (1, 2, 12))
        reconstruction = self.net.decode(explicit, mu)
        self.assertEqual(reconstruction.shape, frames.shape)
        reconstruction.square().mean().backward()
        for head in (self.net.velocity_head, self.net.force_head, self.net.fc_mu):
            self.assertGreater(head.weight.grad.abs().sum().item(), 0)
        # Only the latent branch has a Gaussian prior.
        self.net.zero_grad()
        _, mu, log_var, _ = self.net.encode(frames)
        self.net.kl_loss(mu, log_var).backward()
        self.assertIsNone(self.net.velocity_head.weight.grad)
        self.assertIsNone(self.net.force_head.weight.grad)

    def test_explicit_prediction_does_not_depend_on_latent_sampling(self):
        frames = torch.randn(2, 58)
        first, z1, _ = self.net.forward_with_latent(frames, sample=True)
        second, z2, _ = self.net.forward_with_latent(frames, sample=True)
        torch.testing.assert_close(first, second)
        self.assertFalse(torch.equal(z1, z2))
        _, deterministic, _ = self.net.forward_with_latent(frames, sample=False)
        torch.testing.assert_close(deterministic, self.net.encode(frames)[1])

    def trainer(self):
        return self.module.DynamicsDecoderTrainer(self.net, learning_rate=0.0)

    def test_next_frame_loss_and_reset_target_mask(self):
        frames = torch.randn(3, 2, 58, requires_grad=True)
        targets = torch.randn(3, 2, 5)
        next_frames = torch.randn_like(frames, requires_grad=True)
        dones = torch.tensor([[False, True], [False, False], [True, True]])
        trainer = self.trainer()
        torch.manual_seed(71)
        loss = trainer.update(frames, targets, dones=dones, next_frames=next_frames)
        expected = (sum(trainer.last_parts) + trainer.last_weighted_kl
                    + trainer.last_weighted_reconstruction)
        self.assertAlmostEqual(loss.item(), expected, places=5)
        self.assertGreater(trainer.last_reconstruction, 0)
        # Returned reset observations may be arbitrary, even non-finite.
        poisoned = next_frames.detach().clone()
        poisoned[dones] = float("nan")
        torch.manual_seed(71)
        repeated = trainer.update(frames, targets, dones=dones, next_frames=poisoned)
        torch.testing.assert_close(loss, repeated)
        self.assertIsNone(frames.grad)
        self.assertIsNone(next_frames.grad)
        self.assertFalse(self.net.training)

    def test_all_terminal_batch_has_zero_reconstruction_and_finite_loss(self):
        frames = torch.randn(2, 2, 58)
        trainer = self.trainer()
        loss = trainer.update(frames, torch.zeros(2, 2, 5),
                              dones=torch.ones(2, 2, dtype=torch.bool),
                              next_frames=torch.full_like(frames, float("nan")))
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(trainer.last_reconstruction, 0)

    def test_hidden_state_resets_only_ended_environments(self):
        frames = torch.randn(2, 2, 58)
        hidden = torch.randn(1, 2, 12, requires_grad=True)
        original = hidden.detach().clone()
        encode = self.net.encode
        states = []

        def capture(frame, state=None):
            states.append(state.detach().clone())
            return encode(frame, state)

        self.net.encode = capture
        self.trainer().update(frames, torch.zeros(2, 2, 5),
                              dones=torch.tensor([[True, False], [False, False]]),
                              hidden_state=hidden, next_frames=frames)
        self.assertEqual(states[1][:, 0].abs().sum().item(), 0)
        expected = encode(frames[0], original)[-1]
        torch.testing.assert_close(states[1][:, 1], expected[:, 1])
        torch.testing.assert_close(hidden.detach(), original)
        self.assertIsNone(hidden.grad)

    def test_real_rollout_pairs_frames_and_snapshots_reused_buffers(self):
        # Execute the actual runner loop with a tiny reusable-buffer environment.
        # This catches off-by-one supervision and loss of the last rollout target.
        tree = ast.parse((MODULES / "runners/towing_on_policy_runner.py").read_text("utf-8"))
        runner = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        learn = next(n for n in runner.body if isinstance(n, ast.FunctionDef) and n.name == "learn")
        namespace = dict(torch=torch, time=time, deque=deque,
                         augment_actor_observation=self.module.augment_actor_observation,
                         reset_gru_hidden=self.module.reset_gru_hidden)
        exec(compile(ast.Module(body=[learn], type_ignores=[]), "<runner.learn>", "exec"), namespace)
        obs, targets = torch.zeros(2, 58), torch.full((2, 5), 10.)
        critic = torch.zeros(2, 73)

        def step(actions):
            obs.add_(1)
            targets.add_(1)
            return obs, critic, torch.zeros(2), torch.zeros(2, dtype=torch.bool), {}

        captured = {}

        def update(frames, explicit_targets, **kwargs):
            captured.update(frames=frames, targets=explicit_targets, **kwargs)
            return torch.tensor(0.)

        env = SimpleNamespace(num_envs=2, get_observations=lambda: obs,
                              get_privileged_observations=lambda: critic,
                              get_decoder_supervision=lambda: targets, step=step)
        alg = SimpleNamespace(actor_critic=SimpleNamespace(train=lambda: None),
                              act=lambda *args: torch.zeros(2, 13),
                              process_env_step=lambda *args: None,
                              compute_returns=lambda *args: None, update=lambda: (0., 0.))
        subject = SimpleNamespace(env=env, alg=alg, decoder=self.net.eval(), decoder_hidden=None,
                                  decoder_trainer=SimpleNamespace(update=update), device="cpu",
                                  writer=None, log_dir=None, current_learning_iteration=0,
                                  num_steps_per_env=2, save_interval=1,
                                  _critic_obs=lambda x, **kwargs: x,
                                  _log=lambda *args: None)
        namespace["learn"](subject, 1)
        for step_index in range(2):
            self.assertTrue((captured["frames"][step_index] == step_index).all())
            self.assertTrue((captured["next_frames"][step_index] == step_index + 1).all())
            self.assertTrue((captured["targets"][step_index] == step_index + 10).all())


if __name__ == "__main__":
    unittest.main()
