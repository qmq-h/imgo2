"""Offline regression tests for the 45-D prior → CMoE expert transplant.

这些测试锁定 `scripts/tools/check_cmoe_expert_init.py` 与
`rl_lab/utils/pretrained_prior.py` 的结论：契约解析（含 77/187 两种扫描几何）与
零填充移植的等价性。全部**不需要 Isaac Lab**（用私有包名导入，绕开
`rl_lab/modules/__init__.py` 与 `rl_lab/utils/__init__.py` 对 `isaaclab` 的连锁依赖）。

torch 或本机 checkpoint 缺失时自动 skip（训练机/本机表现一致，不会因缺文件而红）。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "imgo2_rl" / "scripts" / "tools" / "check_cmoe_expert_init.py"

_spec = importlib.util.spec_from_file_location("check_cmoe_expert_init", TOOL_PATH)
tool = importlib.util.module_from_spec(_spec)
sys.modules["check_cmoe_expert_init"] = tool  # @dataclass 需要能按 __module__ 反查
_spec.loader.exec_module(tool)

try:
    import torch

    _PRIOR = tool.import_offline("utils.pretrained_prior")
    CMoEActorCritic = tool.import_offline("modules.cmoe_actor_critic").CMoEActorCritic
except ModuleNotFoundError as error:  # pragma: no cover - 取决于解释器
    torch = None
    _PRIOR = None
    CMoEActorCritic = None
    IMPORT_ERROR = error
else:
    IMPORT_ERROR = None

AMP_CHECKPOINT = REPO_ROOT / "imgo2_rl/logs/amp_rsl_rl/base_move_amp/2026-09-17_21-25-57/model_24500.pt"
AMP_SKIP = None if AMP_CHECKPOINT.is_file() else f"AMP checkpoint 不在本机: {AMP_CHECKPOINT}"


class TestContractParsing(unittest.TestCase):
    """纯解析层：不依赖 torch，锁住 77/187 两种几何与部署 yaml 的列表解析。"""

    def test_ray_count_mirrors_arange_tolerance(self):
        # 现行 77 维几何（前移 0.25 m、1.0x0.6 @ 0.1）
        self.assertEqual(tool.ray_count(0.1, (1.0, 0.6)), (11, 7, 77))
        # 旧 187 维几何：写成 int(span/res)+1 会被 0.6/0.1=5.999... 骗成 6
        self.assertEqual(tool.ray_count(0.1, (1.6, 1.0)), (17, 11, 187))
        self.assertEqual(tool.ray_count(0.05, (0.1, 0.1)), (3, 3, 9))

    def test_parse_scanner_geometry_picks_the_right_block(self):
        # height_scanner 与 height_scanner_base 几何不同，解析必须按块隔离
        text = (
            "scene:\n"
            "  height_scanner:\n"
            "    prim_path: /World/envs/env_.*/Robot/base\n"
            "    pattern_cfg:\n"
            "      resolution: 0.1\n"
            "      size: !!python/tuple\n"
            "      - 1.0\n"
            "      - 0.6\n"
            "  height_scanner_base:\n"
            "    pattern_cfg:\n"
            "      resolution: 0.05\n"
            "      size: !!python/tuple\n"
            "      - 0.1\n"
            "      - 0.1\n"
        )
        self.assertEqual(tool.parse_scanner_geometry(text), (0.1, (1.0, 0.6)))

    def test_parse_obs_groups_skips_group_flags_and_wraps(self):
        text = (
            "observations:\n"
            "  policy:\n"
            "    concatenate_terms: true\n"
            "    history_length: null\n"
            "    base_lin_vel: null\n"
            "    base_ang_vel:\n"
            "      func: isaaclab.envs.mdp.observations:base_ang_vel\n"
            "      scale: 0.25\n"
            "    height_scan: null\n"
        )
        groups = tool.parse_obs_groups(text)
        self.assertEqual(groups["policy"], [("base_lin_vel", None, True), ("base_ang_vel", "0.25", False),
                                            ("height_scan", None, True)])

    def test_parse_simple_yaml_multiline_lists(self):
        text = (
            "# comment\n"
            "imgo2/ppo:\n"
            "  num_observations: 45\n"
            "  observations: [\"ang_vel\", \"gravity_vec\",\n"
            "                 \"commands\", \"dof_pos\"]\n"
            "  action_scale: [0.125, 0.25, 0.25,\n"
            "                 0.125, 0.25, 0.25]\n"
        )
        parsed = tool.parse_simple_yaml(text)
        self.assertEqual(tool.as_float(parsed["num_observations"]), 45.0)
        self.assertEqual(tool.as_strings(parsed["observations"]),
                         ["ang_vel", "gravity_vec", "commands", "dof_pos"])
        self.assertEqual(tool.as_floats(parsed["action_scale"]),
                         [0.125, 0.25, 0.25, 0.125, 0.25, 0.25])


class TestContractAgainstRealRun(unittest.TestCase):
    """用仓库里最新的 CMoE run 自带的 env.yaml 反查契约（没有 run 就 skip）。"""

    def test_newest_run_derives_77_dim_contract(self):
        run_yaml = tool.newest_env_yaml()
        if run_yaml is None:
            self.skipTest("两个 logs 树里都没有 cmoe/*/*/params/env.yaml")
        text = Path(run_yaml).read_text(encoding="utf-8", errors="replace")
        geometry = tool.parse_scanner_geometry(text)
        self.assertIsNotNone(geometry, f"没解析到 height_scanner 几何: {run_yaml}")
        resolution, size = geometry
        rows, cols, rays = tool.ray_count(resolution, size)
        self.assertEqual((rows, cols), (11, 7))
        groups = tool.parse_obs_groups(text)
        policy = [name for name, _, is_null in groups["policy"] if not is_null]
        self.assertEqual(policy, tool.POLICY_TERMS)
        self.assertEqual([name for name, _, is_null in groups["terrain"] if not is_null], ["height_scan"])
        critic = [name for name, _, is_null in groups["critic"] if not is_null]
        self.assertEqual(critic, tool.CRITIC_TERMS)


@unittest.skipIf(torch is None, f"PyTorch unavailable: {IMPORT_ERROR}")
@unittest.skipIf(AMP_SKIP is not None, AMP_SKIP)
class TestZeroPadTransplant(unittest.TestCase):
    """真实 CMoE 模块 + AMP 24500：零填充移植的等价性与"通路接上了"的证据。"""

    @classmethod
    def setUpClass(cls):
        cls.prior = _PRIOR.load_prior(AMP_CHECKPOINT)
        torch.manual_seed(0)
        cls.one_step = 45
        cls.terrain = 77
        cls.critic_dim = 125
        cls.expert_dim = 157
        cls.actor_dim = 10 * cls.one_step + cls.terrain
        cls.observations = torch.randn(8, cls.actor_dim) * 0.5

    def _fixture(self, num_experts=2):
        return CMoEActorCritic(
            num_actor_obs=self.actor_dim,
            num_critic_obs=self.critic_dim,
            num_one_step_obs=self.one_step,
            num_actions=12,
            history_steps=10,
            terrain_obs_dim=self.terrain,
            num_experts=num_experts,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
            init_noise_std=1.0,
        )

    def test_prior_shapes(self):
        self.assertEqual(self.prior.actor_obs_dim, 45)
        self.assertEqual(self.prior.action_dim, 12)
        self.assertEqual(self.prior.critic_obs_dim, 48)
        self.assertIsNotNone(self.prior.std)
        self.assertEqual(tuple(self.prior.std.shape), (12,))

    def test_expert_zero_pad_is_bitwise_equal_and_tail_is_wired(self):
        fixture = self._fixture()
        _PRIOR.install_prior(fixture, self.prior, mode="all")
        weight = fixture.experts[0].actor[0].weight.detach()
        self.assertEqual(tuple(weight.shape), (512, self.expert_dim))
        # 新增 112 列恰为 0
        self.assertEqual(float(weight[:, self.one_step :].abs().max().item()), 0.0)

        teacher = _PRIOR.teacher_actor(self.prior)
        with torch.no_grad():
            reference = teacher(self.observations[:, : self.one_step])
            actor_input = fixture.build_actor_input(self.observations)
            random_tail = torch.cat(
                [actor_input[:, : self.one_step], torch.randn_like(actor_input[:, self.one_step :])], dim=-1
            )
            # 真实路径与随机非零尾输入都必须与先验逐位相同
            self.assertEqual(float((fixture.experts[0].act(actor_input) - reference).abs().max().item()), 0.0)
            self.assertEqual(float((fixture.experts[0].act(random_tail) - reference).abs().max().item()), 0.0)

        # 零权重不等于输入被丢弃：梯度必须流到新增列
        fixture.zero_grad(set_to_none=True)
        fixture.experts[0].act(random_tail).sum().backward()
        grad_tail = fixture.experts[0].actor[0].weight.grad.detach()[:, self.one_step :]
        self.assertGreater(float(grad_tail.abs().max().item()), 0.0)

    def test_critic_zero_pad_is_bitwise_equal(self):
        fixture = self._fixture()
        _PRIOR.install_prior(fixture, self.prior, mode="all")
        teacher = _PRIOR.teacher_critic(self.prior)
        critic_obs = torch.randn(8, self.critic_dim)
        with torch.no_grad():
            out = fixture.experts[0].evaluate(critic_obs)
            ref = teacher(critic_obs[:, :48])
        self.assertEqual(float((out - ref).abs().max().item()), 0.0)

    def test_include_critic_false_leaves_critic_untouched(self):
        """只搬 actor（`init_experts_critic=false`）时 critic 必须保持随机初始化。"""
        fixture = self._fixture()
        before = [expert.critic[0].weight.detach().clone() for expert in fixture.experts]
        report = _PRIOR.install_prior(fixture, self.prior, mode="all", include_critic=False)
        self.assertFalse(report["critic_installed"])
        for snapshot, expert in zip(before, fixture.experts):
            self.assertEqual(float((expert.critic[0].weight.detach() - snapshot).abs().max().item()), 0.0)
        # actor 仍然照装
        self.assertEqual(float(fixture.experts[0].actor[0].weight[:, 45:].abs().max().item()), 0.0)

    def test_std_is_copied(self):
        fixture = self._fixture()
        _PRIOR.install_prior(fixture, self.prior, mode="all")
        self.assertAlmostEqual(float(fixture.std.mean().item()), float(self.prior.std.mean().item()), places=6)
        self.assertNotAlmostEqual(float(fixture.std.mean().item()), 1.0, places=3)

    def test_include_std_false_keeps_default_noise(self):
        """`--init_experts_std=false`：噪声 std 保持 CMoE 默认（1.0），不搬先验的 0.334。"""
        fixture = self._fixture()
        _PRIOR.install_prior(fixture, self.prior, mode="all", include_std=False)
        self.assertAlmostEqual(float(fixture.std.mean().item()), 1.0, places=6)

    def test_jitter_keeps_prior_but_breaks_symmetry(self):
        """`--init_experts_jitter=0.01`：混合仍≈先验（RMSE<0.1），但专家已分化、第 0 个保持纯净。

        为什么需要它：`mode="all"` 让 5 个专家**完全相同** ⇒ 初始混合严格等于先验（好），
        但门控没有任何理由分化（浪费 5 专家容量）。只给**新增 112 列**加小扰动即可两头兼顾。
        """
        torch.manual_seed(0)
        observations = torch.randn(8, 10 * 45 + 77) * 0.5
        fixture = self._fixture(num_experts=5)
        _PRIOR.install_prior(fixture, self.prior, mode="all")
        teacher = _PRIOR.teacher_actor(self.prior)
        report = _PRIOR.jitter_new_columns(fixture, 45, 0.01)
        self.assertEqual(report["jittered_experts"], 4, "第 0 个专家应保持纯净")
        with torch.no_grad():
            rmse = float(_PRIOR.prior_action_rmse(fixture, teacher, observations, 45).item())
        self.assertLess(rmse, 0.1, f"抖动后混合应仍≈先验：RMSE={rmse}")
        # 第 0 个专家：新增列仍全 0、前 45 列仍是先验
        self.assertEqual(float(fixture.experts[0].actor[0].weight[:, 45:].abs().max().item()), 0.0)
        # 被抖动过的专家之间已经不同
        tail_1 = fixture.experts[1].actor[0].weight[:, 45:]
        tail_2 = fixture.experts[2].actor[0].weight[:, 45:]
        self.assertGreater(float((tail_1 - tail_2).abs().max().item()), 0.0)
        # 但它们的**先验本体**（前 45 列）一字未动
        self.assertEqual(float((fixture.experts[1].actor[0].weight[:, :45] - self.prior.actor["0.weight"]).abs().max().item()), 0.0)

    def test_first_mode_does_not_start_from_the_prior(self):
        """负向对照：只装第 0 个专家时，混合**远远不是**先验（实测 RMSE≈2.3）⇒ "只装一个"≠"从先验起步"。"""
        torch.manual_seed(0)
        observations = torch.randn(8, 10 * 45 + 77) * 0.5
        fixture = self._fixture(num_experts=5)
        _PRIOR.install_prior(fixture, self.prior, mode="first")
        teacher = _PRIOR.teacher_actor(self.prior)
        with torch.no_grad():
            rmse = float(_PRIOR.prior_action_rmse(fixture, teacher, observations, 45).item())
        self.assertGreater(rmse, 1.0, f"只装 1 个专家时混合被 4 个随机专家稀释：RMSE={rmse}")

    def test_all_experts_same_prior_makes_mixture_gate_independent(self):
        fixture = self._fixture(num_experts=3)
        _PRIOR.install_prior(fixture, self.prior, mode="all")
        teacher = _PRIOR.teacher_actor(self.prior)
        with torch.no_grad():
            reference = teacher(self.observations[:, : self.one_step])
            outputs = []
            for _ in range(2):
                for parameter in fixture.gating_network.parameters():
                    parameter.data.normal_()
                outputs.append(fixture.act_inference(self.observations).clone())
        # 5（这里是 3）个相同专家的加权和：float32 softmax 权重和 = 1 ± 1e-7
        tolerance = 1.0e-5
        for output in outputs:
            self.assertLessEqual(float((output - reference).abs().max().item()), tolerance)
        self.assertLessEqual(float((outputs[0] - outputs[1]).abs().max().item()), tolerance)

    def test_first_only_mode_is_not_gate_independent(self):
        """负向对照：只装一个专家时混合**不**等于先验（证明上面的等价性来自全专家初始化）。"""
        fixture = self._fixture(num_experts=3)
        _PRIOR.install_prior(fixture, self.prior, mode="first")
        teacher = _PRIOR.teacher_actor(self.prior)
        with torch.no_grad():
            reference = teacher(self.observations[:, : self.one_step])
            output = fixture.act_inference(self.observations)
        self.assertGreater(float((output - reference).abs().max().item()), 1.0e-3)


@unittest.skipIf(torch is None, f"PyTorch unavailable: {IMPORT_ERROR}")
@unittest.skipIf(AMP_SKIP is not None, AMP_SKIP)
class TestPriorActions(unittest.TestCase):
    """`play.py --prior` 的核心逻辑：取前 45 维（当前帧）+ 动作裁 ±3。"""

    @classmethod
    def setUpClass(cls):
        cls.prior = _PRIOR.load_prior(AMP_CHECKPOINT)
        cls.policy = _PRIOR.teacher_actor(cls.prior)

    def test_picks_the_current_frame(self):
        torch.manual_seed(0)
        observations = torch.randn(4, 10 * 45 + 77)
        with torch.no_grad():
            actions = _PRIOR.prior_actions(self.policy, observations, 45, clip=None)
            reference = self.policy(observations[:, :45])
        self.assertEqual(float((actions - reference).abs().max().item()), 0.0)

    def test_clip_matches_deploy_semantics(self):
        torch.manual_seed(0)
        observations = torch.randn(64, 10 * 45 + 77) * 3.0
        with torch.no_grad():
            raw = self.policy(observations[:, :45])
            actions = _PRIOR.prior_actions(self.policy, observations, 45, clip=3.0)
        torch.testing.assert_close(actions, raw.clamp(-3.0, 3.0))
        # 先验原始输出确实会超过 ±3（与部署 config 的 clip_actions 同理，裁剪不是装饰）
        self.assertTrue(bool((raw.abs() > 3.0).any()), "本批先验输出没有超过 ±3，裁剪用例失去意义")

    def test_rejects_too_short_observation(self):
        with self.assertRaises(ValueError):
            _PRIOR.prior_actions(self.policy, torch.randn(2, 12), 45)


@unittest.skipIf(torch is None, f"PyTorch unavailable: {IMPORT_ERROR}")
@unittest.skipIf(AMP_SKIP is not None, AMP_SKIP)
class TestPriorDriftMetric(unittest.TestCase):
    """`Policy/prior_action_rmse` 的语义：全专家初始化后应为 0，被改写后立刻 >0。"""

    @classmethod
    def setUpClass(cls):
        cls.prior = _PRIOR.load_prior(AMP_CHECKPOINT)
        cls.policy = _PRIOR.teacher_actor(cls.prior)
        torch.manual_seed(0)
        cls.observations = torch.randn(8, 10 * 45 + 77) * 0.5

    def _fixture(self):
        return CMoEActorCritic(
            num_actor_obs=527, num_critic_obs=125, num_one_step_obs=45, num_actions=12,
            history_steps=10, terrain_obs_dim=77, num_experts=3,
            actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128],
            activation="elu", init_noise_std=1.0,
        )

    def test_zero_after_full_install(self):
        fixture = self._fixture()
        _PRIOR.install_prior(fixture, self.prior, mode="all")
        rmse = _PRIOR.prior_action_rmse(fixture, self.policy, self.observations, 45)
        self.assertLessEqual(float(rmse.item()), 1.0e-5)

    def test_positive_after_perturbation(self):
        fixture = self._fixture()
        _PRIOR.install_prior(fixture, self.prior, mode="all")
        with torch.no_grad():
            for expert in fixture.experts:
                expert.actor[6].bias += 0.5  # 直接改最后一层偏置 ⇒ 动作均值整体偏移
        rmse = _PRIOR.prior_action_rmse(fixture, self.policy, self.observations, 45)
        self.assertGreater(float(rmse.item()), 0.1)

    def test_positive_for_uninitialised_mixture(self):
        """未装先验的随机混合策略与先验的差距必然很大（度量不是恒 0 的摆设）。"""
        fixture = self._fixture()
        rmse = _PRIOR.prior_action_rmse(fixture, self.policy, self.observations, 45)
        self.assertGreater(float(rmse.item()), 0.5)


class TestRunnerCfgDefaults(unittest.TestCase):
    """新加的初始化开关**默认关** ⇒ 现有 run 的行为一字不变。"""

    def test_init_defaults_are_off(self):
        cfg_cls = tool.import_offline("config.cmoe_algorithm_cfg").CMoEOnPolicyRunnerCfg
        cfg = cfg_cls()
        self.assertIsNone(cfg.init_experts_from, "默认必须不装先验（否则现有 run 会被改写）")
        self.assertEqual(cfg.init_experts_mode, "all")
        self.assertTrue(cfg.log_prior_rmse)
        self.assertEqual(cfg.history_steps, 10)

    def test_blank_prior_path_is_treated_as_missing(self):
        """空串/空白必须被判成"没给先验"（2026-09-25 实跑就栽在这里：静默不装）。"""
        normalize = _PRIOR.normalize_prior_path
        self.assertIsNone(normalize(None))
        self.assertIsNone(normalize(""))
        self.assertIsNone(normalize("   "))
        self.assertEqual(normalize(" /tmp/model.pt "), "/tmp/model.pt")


if __name__ == "__main__":
    unittest.main()
