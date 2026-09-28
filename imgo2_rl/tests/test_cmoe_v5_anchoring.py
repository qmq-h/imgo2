"""v5（2026-09-28）三项新机制的可离线验证：

1. **门控初始偏置**（`init_gate_bias`）：`mode="first"` 时让第 0 步的混合动作仍≈先验；
2. **强制专家**（`actor_critic.forced_expert`）：混合输出严格等于被选中专家的输出（`play.py --force_expert`）；
3. **只在 flat 上锚定**：storage 的逐样本 `anchor_weight` + 算法的锚损失
   * 只有被锚的那个专家收到梯度（其余专家参数逐位不变）；
   * 只对权重非零的样本生效（按地形掩码）；
   * 权重全 0 / 没有教师 ⇒ 锚损失恒 0（与旧行为完全一致）。

全部用合成数据，不需要 GPU / Isaac Sim（模块按文件路径加载，避开 `rl_lab.utils.__init__`
对 isaaclab 的依赖，做法与 `tests/test_cmoe_expert_init.py` 一致）。

Run: python3 -m unittest discover -s tests -p test_cmoe_v5_anchoring.py
"""

import importlib.util
import sys
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
RL_LAB = REPO / "scripts" / "rl_lab" / "rl_lab"


def _private_package():
    """建一个**私有包命名空间**，把 `rl_lab/{utils,storage}` 下的纯 torch 模块按包成员导入。

    直接按文件路径加载会踩两个坑：`terrain_masks` 里有相对导入（`from .gait_dump import ...`），
    而 `rl_lab/utils/__init__.py` 又会拉 isaaclab（本机没有 omni.log）。用一个自己的包壳绕开两者
    —— 与 `tests/test_cmoe_expert_init.py` 的私有命名空间做法同一目的。
    """
    import types

    root = types.ModuleType("v5rl")
    root.__path__ = [str(RL_LAB)]
    utils = types.ModuleType("v5rl.utils")
    utils.__path__ = [str(RL_LAB / "utils")]
    storage = types.ModuleType("v5rl.storage")
    storage.__path__ = [str(RL_LAB / "storage")]
    sys.modules.update({"v5rl": root, "v5rl.utils": utils, "v5rl.storage": storage})
    import importlib

    return (
        importlib.import_module("v5rl.utils.pretrained_prior"),
        importlib.import_module("v5rl.utils.terrain_masks"),
        importlib.import_module("v5rl.storage.cmoe_rollout_storage"),
        importlib.import_module("v5rl.utils.anchor"),
    )


prior_mod, masks_mod, storage_mod, anchor_mod = _private_package()


class _ToyExpert(torch.nn.Module):
    """专家替身：单层线性（输入 dim → 动作 dim），便于断言"只改了哪个专家"。"""

    def __init__(self, in_dim, out_dim, scale):
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)
        with torch.no_grad():
            self.linear.weight.mul_(0.0).add_(scale)

    def act_inference(self, x):
        return self.linear(x)

    def act(self, x):
        return self.linear(x)


class _ToyActorCritic(torch.nn.Module):
    """带 `gating_network`（末层 softmax）与 `experts` 的极简替身，接口与 CMoEActorCritic 的关键部分一致。"""

    def __init__(self, in_dim=6, out_dim=2, num_experts=3):
        super().__init__()
        self.experts = torch.nn.ModuleList(
            [_ToyExpert(in_dim, out_dim, scale=float(k + 1)) for k in range(num_experts)]
        )
        self.gating_network = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 8), torch.nn.ELU(), torch.nn.Linear(8, num_experts), torch.nn.Softmax(dim=-1)
        )
        self.gate_weights = None
        self.forced_expert = None
        # 替身里的"共享编码器"（对应真实 AC 的 state/terrain 估计器）：锚**不应**更新它
        self.encoder = torch.nn.Linear(in_dim, in_dim)
        self.build_actor_input = lambda obs: self.encoder(obs)

    def mixture(self, obs):
        actor_input = self.build_actor_input(obs)
        self.gate_weights = self.gating_network(actor_input)
        # 与 CMoEActorCritic 相同的强制逻辑（内联，避免依赖 isaaclab 才能加载的模块文件）
        if self.forced_expert is not None:
            one_hot = torch.zeros_like(self.gate_weights)
            one_hot[:, int(self.forced_expert)] = 1.0
            self.gate_weights = one_hot
        means = torch.stack([expert.act_inference(actor_input) for expert in self.experts], dim=1)
        return (means * self.gate_weights.unsqueeze(-1)).sum(dim=1)


class TestGateBias(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.ac = _ToyActorCritic()

    def test_bias_makes_the_target_expert_dominate(self):
        torch.manual_seed(0)
        obs = torch.randn(64, 6)
        before = self.ac.mixture(obs).clone()
        report = prior_mod.init_gate_bias(self.ac, expert_index=1, margin=6.0)
        after = self.ac.mixture(obs)
        weights = self.ac.gate_weights
        self.assertEqual(report["expert"], 1)
        self.assertGreater(float(weights[:, 1].min()), 0.9, "偏置后目标专家的门控权重应接近 1")
        # 混合输出应当≈专家 1 单独的（而不再是 3 个专家的平均）
        solo = self.ac.experts[1].act_inference(self.ac.build_actor_input(obs))
        self.assertGreater(float(weights[:, 1].min()), 0.95, "偏置后目标专家应几乎独占")
        # 残差来自另外 1~5% 的权重；按输出幅度相对化判断（绝对容差会随专家输出尺度变化）
        self.assertLess(float((after - solo).abs().max()),
                        0.05 * float(solo.abs().max()) + 1e-6)
        self.assertGreater(float((before - after).abs().max()), 1e-3, "偏置必须真的改变混合输出")

    def test_shrink_makes_the_initial_gate_input_independent(self):
        """`shrink_last_layer` 的语义：把末层权重缩小 ⇒ 初始门控**几乎只由 bias 决定**（不随输入变化）。

        ⚠️ 注意它**不是**"让目标专家更占优"：缩小权重会同时压掉 logits 的输入相关部分，
        若 margin 很小反而更接近均匀。真正的"独占"靠 `margin`。
        """
        torch.manual_seed(0)
        obs = torch.randn(32, 6)
        prior_mod.init_gate_bias(self.ac, expert_index=2, margin=6.0, shrink_last_layer=0.0)
        self.ac.mixture(obs)
        spread_without = float(self.ac.gate_weights.std(dim=0).max())
        prior_mod.init_gate_bias(self.ac, expert_index=2, margin=6.0, shrink_last_layer=0.01)
        self.ac.mixture(obs)
        spread_with = float(self.ac.gate_weights.std(dim=0).max())
        self.assertLess(spread_with, 1e-3, "缩小末层后门控应几乎不随输入变化")
        self.assertLess(spread_with, spread_without)

    def test_bad_index_and_missing_gate_raise(self):
        with self.assertRaises(ValueError):
            prior_mod.init_gate_bias(self.ac, expert_index=99)
        with self.assertRaises(ValueError):
            prior_mod.init_gate_bias(torch.nn.Linear(3, 3), expert_index=0)


class TestForcedExpert(unittest.TestCase):
    def test_forced_expert_output_equals_that_expert(self):
        """`--force_expert k` 的语义：混合输出**严格等于**专家 k（其他专家完全不参与）。"""
        torch.manual_seed(0)
        ac = _ToyActorCritic()
        obs = torch.randn(16, 6)
        for k in range(len(ac.experts)):
            ac.forced_expert = k
            mixed = ac.mixture(obs)
            solo = ac.experts[k].act_inference(ac.build_actor_input(obs))
            self.assertLess(float((mixed - solo).abs().max()), 1e-6, f"专家 {k} 未生效")
            self.assertAlmostEqual(float(ac.gate_weights[:, k].min()), 1.0, places=6)
            self.assertAlmostEqual(float(ac.gate_weights.sum(dim=-1).max()), 1.0, places=6)
        ac.forced_expert = None
        self.assertGreater(float(ac.gate_weights.max()), 0.0)


class TestFlatOnlyAnchorMask(unittest.TestCase):
    KEYS = ["pyramid_stairs", "pyramid_stairs_inv", "boxes", "random_rough",
            "hf_pyramid_slope", "hf_pyramid_slope_inv", "gap", "hurdle", "mix",
            "narrow_stairs", "flat"]
    PROPS = [0.10, 0.10, 0.10, 0.05, 0.05, 0.05, 0.30, 0.10, 0.10, 0.10, 0.10]

    def test_flat_columns_and_weights(self):
        columns = masks_mod.terrain_columns(self.KEYS, self.PROPS, 40, ("flat",))
        self.assertEqual(columns, [37, 38, 39], "flat 占最后三列（按比例分配）")
        terrain_types = torch.tensor([0, 37, 20, 38, 39, 11], dtype=torch.long)
        weights = masks_mod.anchor_weights(terrain_types, keys=self.KEYS, proportions=self.PROPS,
                                          num_cols=40, names=("flat",), scale=0.3)
        expected = [0.0, 0.3, 0.0, 0.3, 0.3, 0.0]
        for got, want in zip(weights.tolist(), expected):
            self.assertAlmostEqual(got, want, places=6)   # float32 ⇒ 0.3 会存成 0.30000001

    def test_unknown_terrain_name_raises(self):
        with self.assertRaises(ValueError):
            masks_mod.terrain_columns(self.KEYS, self.PROPS, 40, ("flatt",))

    def test_empty_selection_is_all_zero(self):
        weights = masks_mod.anchor_weights(torch.tensor([0, 1]), keys=self.KEYS, proportions=self.PROPS,
                                          num_cols=40, names=())
        self.assertEqual(weights.tolist(), [0.0, 0.0])

    def test_storage_round_trip_and_default_zero(self):
        storage = storage_mod.CMoERolloutStorage(num_envs=2, num_transitions_per_env=3, obs_shape=[4],
                                                privileged_obs_shape=[4], actions_shape=[2])
        for step in range(3):
            transition = storage.Transition()
            transition.observations = torch.zeros(2, 4)
            transition.critic_observations = torch.zeros(2, 4)
            transition.next_critic_observations = torch.zeros(2, 4)
            transition.actions = torch.zeros(2, 2)
            transition.rewards = torch.zeros(2)
            transition.dones = torch.zeros(2)
            transition.values = torch.zeros(2, 1)
            transition.actions_log_prob = torch.zeros(2)
            transition.action_mean = torch.zeros(2, 2)
            transition.action_sigma = torch.ones(2, 2)
            if step == 1:
                transition.anchor_weight = torch.tensor([1.0, 0.0])
            storage.add_transitions(transition)
        self.assertEqual(storage.anchor_weight[:, :, 0].tolist(),
                         [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]], "未设 anchor_weight 的步必须是 0")
        batch = next(storage.mini_batch_generator(num_mini_batches=1, num_epochs=1))
        self.assertEqual(len(batch), 11, "generator 应多出 anchor_weight 一项")
        self.assertEqual(batch[-1].shape[0], batch[0].shape[0])


class TestAnchorLossSemantics(unittest.TestCase):
    """锚损失的三条语义（用算法里那段逻辑的等价实现来断言，避免构造完整 PPO/Isaac 依赖）。"""

    def _loss(self, actor_critic, obs, weight, teacher, expert=0):
        weight_sum = weight.sum()
        if teacher is None or float(weight_sum) <= 0.0:
            return torch.zeros(()), None
        with torch.inference_mode():
            teacher_actions = teacher(obs)
        actor_input = actor_critic.build_actor_input(obs).detach()   # 与 cmoe_ppo.update() 一致
        expert_actions = actor_critic.experts[expert].act_inference(actor_input)
        mse = torch.square(expert_actions - teacher_actions).mean(dim=-1)
        return (weight.squeeze(-1) * mse).sum() / weight_sum.clamp_min(1e-6), expert_actions

    def test_only_the_anchored_expert_gets_gradient(self):
        torch.manual_seed(0)
        ac = _ToyActorCritic()
        for expert in ac.experts:
            expert.linear.weight.requires_grad_(True)
            expert.linear.bias.requires_grad_(True)
        obs = torch.randn(8, 6)
        teacher = lambda x: torch.zeros(x.shape[0], 2)          # noqa: E731 - 目标恒 0，梯度方向明确
        weight = torch.ones(8, 1)
        loss, _ = self._loss(ac, obs, weight, teacher, expert=1)
        ac.zero_grad()
        loss.backward()
        grads = [float(expert.linear.weight.grad.abs().sum()) if expert.linear.weight.grad is not None else 0.0
                 for expert in ac.experts]
        self.assertGreater(grads[1], 0.0, "被锚的专家必须有梯度")
        self.assertEqual(grads[0], 0.0, "其它专家**不能**有梯度")
        self.assertEqual(grads[2], 0.0, "其它专家**不能**有梯度")
        self.assertIsNone(ac.gating_network[0].weight.grad, "锚损失不应触碰门控")
        self.assertIsNone(ac.encoder.weight.grad,
                          "锚损失不应穿过 build_actor_input（否则会与估计器自己的损失打架）")

    def test_weight_zero_or_no_teacher_gives_zero_loss(self):
        torch.manual_seed(0)
        ac = _ToyActorCritic()
        obs = torch.randn(4, 6)
        zero_weight = torch.zeros(4, 1)
        loss, _ = self._loss(ac, obs, zero_weight, lambda x: torch.zeros(x.shape[0], 2))
        self.assertEqual(float(loss), 0.0, "权重全 0（非 flat 地形）⇒ 锚损失必须为 0")
        loss_none, _ = self._loss(ac, obs, torch.ones(4, 1), None)
        self.assertEqual(float(loss_none), 0.0, "没有教师 ⇒ 锚损失必须为 0")

    def test_only_weighted_samples_contribute(self):
        """一半样本权重 0（＝不在 flat 上）：损失＝只对有权重那一半求的 MSE。"""
        torch.manual_seed(0)
        ac = _ToyActorCritic()
        obs = torch.randn(6, 6)
        teacher_actions = torch.zeros(6, 2)
        weight = torch.tensor([[1.0], [1.0], [1.0], [0.0], [0.0], [0.0]])
        loss, expert_actions = self._loss(ac, obs, weight, lambda x: teacher_actions)
        masked_mse = torch.square(expert_actions[:3] - teacher_actions[:3]).mean(dim=-1).mean()
        self.assertAlmostEqual(float(loss), float(masked_mse), places=6)


class TestAnchorLossFormula(unittest.TestCase):
    """`utils/anchor.py` 的**真实算式**（算法里就是调它，不是测试里另写一份）。"""

    def test_masking_and_normalization(self):
        pred = torch.tensor([[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]])
        teacher = torch.zeros(3, 2)
        weights = torch.tensor([1.0, 0.0, 1.0])          # 中间那条**完全不参与**
        loss = anchor_mod.weighted_mse(pred, teacher, weights)
        # 逐样本 MSE 是**对动作维取均值**：(1²+1²)/2=1、(5²+5²)/2=25 ⇒ (1+25)/2 = 13
        self.assertAlmostEqual(float(loss), 13.0, places=6)

    def test_zero_weight_or_empty_predictions(self):
        pred = torch.ones(4, 2)
        self.assertEqual(float(anchor_mod.weighted_mse(pred, torch.zeros(4, 2), torch.zeros(4))), 0.0)
        with self.assertRaises(ValueError):
            anchor_mod.weighted_mse([], torch.zeros(4, 2), torch.ones(4))

    def test_both_targets_are_averaged(self):
        teacher = torch.zeros(2, 1)
        expert = torch.full((2, 1), 2.0)                 # MSE 4
        mixture = torch.full((2, 1), 1.0)                # MSE 1
        weights = torch.ones(2)
        self.assertAlmostEqual(float(anchor_mod.weighted_mse([expert, mixture], teacher, weights)),
                               2.5, places=6)

    def test_shape_errors_raise(self):
        with self.assertRaises(ValueError):
            anchor_mod.weighted_mse(torch.ones(3, 2), torch.zeros(3, 2), torch.ones(2))
        with self.assertRaises(ValueError):      # 权重条数对不上批大小
            anchor_mod.weighted_mse(torch.ones(3, 2), torch.zeros(3, 2), torch.ones(4))
        with self.assertRaises(ValueError):      # 预测与教师形状不一致
            anchor_mod.weighted_mse(torch.ones(3, 5), torch.zeros(3, 2), torch.ones(3))

    def test_resolve_targets(self):
        self.assertEqual(anchor_mod.resolve_targets("expert"), (True, False))
        self.assertEqual(anchor_mod.resolve_targets("mixture"), (False, True))
        self.assertEqual(anchor_mod.resolve_targets("both"), (True, True))
        with self.assertRaises(ValueError):
            anchor_mod.resolve_targets("everything")

    def test_mix_experts_matches_weighted_sum(self):
        means = [torch.tensor([[1.0, 1.0]]), torch.tensor([[3.0, 3.0]])]
        gate = torch.tensor([[0.25, 0.75]])
        mixed = anchor_mod.mix_experts(means, gate)
        self.assertTrue(torch.allclose(mixed, torch.tensor([[2.5, 2.5]])))


class TestAnchorTargetGradientRouting(unittest.TestCase):
    """`anchor_target` 的三种模式：梯度该进谁、不该进谁（对应算法里那段选择逻辑）。"""

    def _predictions(self, ac, obs, target, expert=1):
        actor_input = ac.build_actor_input(obs).detach()          # 与算法一致：不穿过估计器
        expert_flag, mixture_flag = anchor_mod.resolve_targets(target)
        out = []
        if expert_flag:
            out.append(ac.experts[expert].act_inference(actor_input))
        if mixture_flag:
            gate = ac.gating_network(actor_input)
            means = [e.act_inference(actor_input) for e in ac.experts]
            out.append(anchor_mod.mix_experts(means, gate))
        return out

    def _grads(self, target):
        torch.manual_seed(0)
        ac = _ToyActorCritic()
        obs = torch.randn(8, 6)
        teacher = torch.zeros(8, 2)
        loss = anchor_mod.weighted_mse(self._predictions(ac, obs, target), teacher, torch.ones(8))
        ac.zero_grad()
        loss.backward()
        def g(module):
            return float(module.linear.weight.grad.abs().sum()) if module.linear.weight.grad is not None else 0.0
        return (
            [g(e) for e in ac.experts],
            ac.gating_network[0].weight.grad is not None,
            ac.encoder.weight.grad is not None,
        )

    def test_expert_mode_only_touches_that_expert(self):
        grads, gate_touched, encoder_touched = self._grads("expert")
        self.assertGreater(grads[1], 0.0)
        self.assertEqual(grads[0], 0.0)
        self.assertEqual(grads[2], 0.0)
        self.assertFalse(gate_touched, "expert 模式不应触碰门控")
        self.assertFalse(encoder_touched, "任何模式都不应穿过 build_actor_input")

    def test_mixture_mode_touches_gate_and_all_experts(self):
        grads, gate_touched, encoder_touched = self._grads("mixture")
        self.assertTrue(gate_touched, "输出层锚必须能训练门控（这正是它的目的）")
        self.assertTrue(all(g > 0.0 for g in grads), "混合输出对所有专家都有梯度（按门控权重）")
        self.assertFalse(encoder_touched)

    def test_both_mode_is_the_union(self):
        grads, gate_touched, _ = self._grads("both")
        self.assertTrue(gate_touched)
        self.assertTrue(all(g > 0.0 for g in grads))


class TestAlgorithmWiring(unittest.TestCase):
    """源码级锁定：算法确实用同一套算式与三种模式，且默认仍是 `expert`（向后兼容）。"""

    def setUp(self):
        self.src = (RL_LAB / "algorithms" / "cmoe_ppo.py").read_text(encoding="utf-8")
        self.cfg = (RL_LAB / "config" / "cmoe_algorithm_cfg.py").read_text(encoding="utf-8")

    def test_algorithm_uses_shared_formula_and_resolves_targets(self):
        for needle in ("from ..utils.anchor import mix_experts, resolve_targets, weighted_mse",
                       "resolve_targets(self.anchor_target)",
                       "weighted_mse(predictions, teacher_actions, anchor_weight_batch)",
                       "mix_experts(expert_means, gate_weights)"):
            self.assertIn(needle, self.src, needle)

    def test_default_target_is_expert_for_backward_compat(self):
        self.assertIn('anchor_target: str = "expert"', self.cfg)
        self.assertIn('self.anchor_target = str(anchor_target)', self.src)

    def test_gate_weights_in_anchor_are_recomputed_from_detached_input(self):
        """混合锚必须用 detach 过的输入重算（不能用 mu_batch，它的图里带估计器）。"""
        block = self.src.split("# 先验锚定（v5）", 1)[1].split("# Surrogate loss", 1)[0]
        self.assertIn(".detach()", block)
        # 只看**代码行**：注释里会提到 mu_batch（说明"为什么不用它"），那不算使用
        code = "\n".join(
            line for line in block.splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
        self.assertNotIn("mu_batch", code, "混合锚不要复用 mu_batch（它的图里带估计器）")


if __name__ == "__main__":
    unittest.main()
