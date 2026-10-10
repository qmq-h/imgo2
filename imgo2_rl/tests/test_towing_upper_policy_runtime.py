"""`upper_policy_runtime.py` 的离线测试：帧契约、残差、checkpoint 契约校验。

本机（以及任何没有 GPU 的机器）只能验证**纯 torch** 的那一半：

- 57 维 policy 帧的布局/缩放与训练侧 `upper_mdp.policy_frame` 及
  `upper_logic.UpperObservationSpec.terms` 逐项一致；actor 输入 79、critic 72；
- `delta = clamp(action, ±1) ⊙ action_scale`、`|delta| ≤ action_scale`、
  `_last_action` 的时序（首拍为零，次拍等于上一拍 clamp 后的动作）；
- checkpoint 的 `towing_contract` 硬校验（旧 56/63 维 checkpoint 必须被拒绝）。

**不覆盖**：把残差接到 PhysX 上的仿真行为（上层节拍、held 下发、指标口径）——
只能在训练机实跑，见 README TOW-20，不要把本文件的通过写成「开关已验证」。
"""
import ast
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

try:
    import torch
except ImportError:  # 离线契约检查也会在没有 torch 的解释器里跑（此时只保留静态用例）
    torch = None


RL = Path(__file__).resolve().parents[1]
PKG = RL / "source/imgo2_rl/imgo2_rl/tasks/manager_based/towing"
TOWING_SCRIPTS = RL / "scripts/towing"
#: 静态检查（AST / 源码文本）不需要 torch，所以这两个模块总是加载。
logic = None
policy_cfg_module = None
runtime = None


def _module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # 注册进 sys.modules：模块里有 dataclass + `from __future__ import annotations`。
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


logic = _module_at("towing_upper_logic_for_runtime_test", PKG / "upper_logic.py")
policy_cfg_module = _module_at("towing_policy_cfg_for_runtime_test",
                               PKG / "utils/policy_cfg.py")
if torch is not None:   # `upper_policy_runtime` 顶层 import torch
    runtime = _module_at("towing_upper_policy_runtime_test",
                         TOWING_SCRIPTS / "upper_policy_runtime.py")

#: 冻结策略契约的 12 维残差尺度（hip 0.125、thigh/shank 0.25）——运行时直接吃这一份。
AMP_ACTION_SCALE = tuple(float(value) for value in policy_cfg_module.get_policy("amp").action_scale)

_NEEDS_TORCH = unittest.skipIf(torch is None, "PyTorch is not installed in the offline-check interpreter")

_MODULES = None


def network_modules():
    """加载真实 actor/decoder 类（离线走 `upper_policy_runtime` 的按路径兜底）。"""
    global _MODULES
    if _MODULES is None:
        _MODULES = runtime.load_network_modules()
    return _MODULES


def registered_agent_cfg_stub():
    """与 `agents/upper_ppo_cfg.py::UpperTowingPPORunnerCfg` 同构的对象。

    真实注册 cfg 需要 import isaaclab（离线没有），所以这里按源码的 kwargs 建同构对象，
    并由 `RegisteredConfigTests` 用 AST 逐字核对，避免两处漂移。
    """
    from types import SimpleNamespace
    return SimpleNamespace(
        clip_actions=1.0,
        policy=SimpleNamespace(
            actor_hidden_dims=[256, 128, 64], critic_hidden_dims=[256, 128, 64],
            activation="elu", init_noise_std=0.5, fixed_std=False,
            rnn_type="gru", rnn_hidden_size=256, rnn_num_layers=1),
        decoder=SimpleNamespace(
            frame_dim=57, feature_dim=128, hidden_dim=128, num_layers=1, latent_dim=16,
            force_scale=10.0, kld_weight=0.005, learning_rate=1.0e-3, max_grad_norm=1.0,
            velocity_coef=5.0, force_coef=10.0, mass_coef=1.0),
    )


def build_actor(spec, *, num_actor_obs=None, zero_last=True):
    modules = network_modules()
    actor = modules.actor_critic(
        num_actor_obs=spec.actor_obs_dim if num_actor_obs is None else num_actor_obs,
        num_critic_obs=spec.num_critic_obs, num_actions=spec.num_actions,
        actor_hidden_dims=list(spec.actor_hidden_dims),
        critic_hidden_dims=list(spec.critic_hidden_dims), activation=spec.activation,
        rnn_type=spec.rnn_type, rnn_hidden_size=spec.rnn_hidden_size,
        rnn_num_layers=spec.rnn_num_layers, init_noise_std=spec.init_noise_std)
    if zero_last:
        last = [layer for layer in actor.actor if isinstance(layer, torch.nn.Linear)][-1]
        torch.nn.init.zeros_(last.weight)
        torch.nn.init.zeros_(last.bias)
    return actor


def build_decoder(spec):
    modules = network_modules()
    return modules.decoder(
        frame_dim=spec.frame_dim, feature_dim=spec.decoder_feature_dim,
        hidden_dim=spec.decoder_hidden_dim, num_layers=spec.decoder_num_layers,
        latent_dim=spec.latent_dim, force_scale=spec.decoder_force_scale)


def write_checkpoint(directory, *, actor, decoder, contract, iteration=42):
    payload = {"towing_contract": contract, "model_state_dict": actor.state_dict(),
               "decoder_state_dict": decoder.state_dict(), "iter": iteration}
    path = Path(directory) / "model_synthetic.pt"
    torch.save(payload, path)
    return path


def make_runtime(directory, *, spec=None, contract=None, actor=None, decoder=None,
                 num_envs=3, action_scale=AMP_ACTION_SCALE, deterministic=True,
                 zero_last=True, iteration=42):
    spec = spec or runtime.UpperNetworkSpec.from_agent_cfg(registered_agent_cfg_stub())
    modules = network_modules()
    if actor is None:
        actor = build_actor(spec, zero_last=zero_last)
    if decoder is None:
        decoder = build_decoder(spec)
    if contract is None:
        contract = runtime.expected_towing_contract(spec)
    path = write_checkpoint(directory, actor=actor, decoder=decoder, contract=contract,
                            iteration=iteration)
    instance = runtime.UpperPolicyRuntime(
        path, num_envs=num_envs, action_scale=action_scale, device="cpu",
        deterministic=deterministic, spec=spec, network_modules=modules)
    return instance, actor, decoder, path


def set_actor_output(instance, actor, value):
    """把 actor 末层设成常数输出（weight=0、bias=value），用来做确定性断言。"""
    last = [layer for layer in actor.actor if isinstance(layer, torch.nn.Linear)][-1]
    with torch.no_grad():
        last.weight.zero_()
        last.bias.fill_(value)
    instance.actor.load_state_dict(actor.state_dict(), strict=True)


def act_kwargs(num_envs, *, loco_command=None, base_ang_vel=None, projected_gravity=None,
               last_loco_action=None, joint_pos_rel=None, joint_vel=None):
    return {
        "loco_command": (torch.zeros(num_envs, 3) if loco_command is None else loco_command),
        "base_ang_vel": (torch.zeros(num_envs, 3) if base_ang_vel is None else base_ang_vel),
        "projected_gravity": (torch.tensor([[0.0, 0.0, -1.0]]).expand(num_envs, 3)
                              if projected_gravity is None else projected_gravity),
        "last_loco_action": (torch.zeros(num_envs, 12) if last_loco_action is None
                             else last_loco_action),
        "joint_pos_rel": (torch.zeros(num_envs, 12) if joint_pos_rel is None else joint_pos_rel),
        "joint_vel": (torch.zeros(num_envs, 12) if joint_vel is None else joint_vel),
    }


# ---------------------------------------------------------------- 注册配置来源


def _class_def(tree, name):
    return next(node for node in ast.walk(tree)
                if isinstance(node, ast.ClassDef) and node.name == name)


def _call_kwargs(source_tree, class_name, call_name):
    for node in _class_def(source_tree, class_name).body:
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name) and node.value.func.id == call_name):
            return {keyword.arg: ast.literal_eval(keyword.value) for keyword in node.value.keywords}
    raise AssertionError(f"{class_name} 里找不到 {call_name}(...) 调用")


def _class_scalar_assignments(source_tree, class_name):
    result = {}
    for node in _class_def(source_tree, class_name).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            try:
                result[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                continue
    return result


@_NEEDS_TORCH
class RegisteredConfigTests(unittest.TestCase):
    """运行时的默认超参必须与注册的 agent cfg（镜像训练超参）逐项一致。"""

    def setUp(self):
        self.tree = ast.parse((PKG / "agents/upper_ppo_cfg.py").read_text(encoding="utf-8"))

    def test_policy_and_decoder_kwargs_match_registered_cfg(self):
        policy = _call_kwargs(self.tree, "UpperTowingPPORunnerCfg", "TowingActorCriticCfg")
        decoder = _call_kwargs(self.tree, "UpperTowingPPORunnerCfg", "TowingDecoderCfg")
        spec = runtime.UpperNetworkSpec()
        self.assertEqual(list(spec.actor_hidden_dims), policy["actor_hidden_dims"])
        self.assertEqual(list(spec.critic_hidden_dims), policy["critic_hidden_dims"])
        self.assertEqual(spec.activation, policy["activation"])
        self.assertEqual(spec.init_noise_std, policy["init_noise_std"])
        self.assertEqual(spec.rnn_type, policy["rnn_type"])
        self.assertEqual(spec.rnn_hidden_size, policy["rnn_hidden_size"])
        self.assertEqual(spec.rnn_num_layers, policy["rnn_num_layers"])
        self.assertEqual(spec.frame_dim, decoder["frame_dim"])
        self.assertEqual(spec.decoder_feature_dim, decoder["feature_dim"])
        self.assertEqual(spec.decoder_hidden_dim, decoder["hidden_dim"])
        self.assertEqual(spec.decoder_num_layers, decoder["num_layers"])
        self.assertEqual(spec.latent_dim, decoder["latent_dim"])
        self.assertEqual(spec.decoder_force_scale, decoder["force_scale"])

    def test_clip_actions_matches_the_hardcoded_upper_clamp(self):
        """环境侧 `process_actions` 写死 `clamp(±1)`；注册 cfg 的 `clip_actions` 必须同为 1.0，
        否则「策略 vs 基线」两轮的残差权限不是同一个（本运行时不读它，只做一致性守卫）。"""
        values = _class_scalar_assignments(self.tree, "UpperTowingPPORunnerCfg")
        self.assertEqual(float(values["clip_actions"]), runtime.ACTION_CLIP)

    def test_stub_matches_the_registered_cfg(self):
        spec = runtime.UpperNetworkSpec.from_agent_cfg(registered_agent_cfg_stub())
        self.assertEqual(spec, runtime.UpperNetworkSpec())

    def test_critic_dim_matches_upper_env_cfg_terms(self):
        """critic 观测维数从 `upper_env_cfg.CriticCfg` 的项推出来（不是随手写的 72）。"""
        tree = ast.parse((PKG / "upper_env_cfg.py").read_text(encoding="utf-8"))
        terms = []
        for node in _class_def(tree, "CriticCfg").body:
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            for keyword in node.value.keywords:
                if (keyword.arg == "func" and isinstance(keyword.value, ast.Attribute)
                        and isinstance(keyword.value.value, ast.Name)
                        and keyword.value.value.id == "mdp"):
                    terms.append(keyword.value.attr)
        dims = {"policy_frame": runtime.FRAME_DIM, "robot_velocity": 2, "cart_velocity": 2,
                "rope_privileged_state": 4, "towing_force": 3,
                "cart_privileged_parameters": 4}
        self.assertEqual(sorted(terms), sorted(dims))
        self.assertEqual(sum(dims[name] for name in terms), runtime.DEFAULT_CRITIC_OBS_DIM)


class FrameContractTests(unittest.TestCase):
    """57 维帧：顺序、维数、缩放都与训练侧源码一致（纯 AST，不需要 torch）。"""

    def test_frame_terms_match_upper_observation_spec(self):
        spec = logic.UpperObservationSpec()
        self.assertEqual(tuple((name, dim) for name, dim, _ in runtime.FRAME_TERMS), spec.terms)
        self.assertEqual(runtime.FRAME_DIM, spec.frame_dim)
        self.assertEqual(runtime.FRAME_DIM, 57)
        self.assertEqual(spec.actor_dim, 79)

    def test_frame_terms_match_policy_frame_source(self):
        """从 `upper_mdp.policy_frame` 的 cat(...) 里读顺序与缩放，逐项比对。"""
        tree = ast.parse((PKG / "upper_mdp.py").read_text(encoding="utf-8"))
        function = next(node for node in ast.walk(tree)
                        if isinstance(node, ast.FunctionDef) and node.name == "policy_frame")
        returned = next(node for node in ast.walk(function) if isinstance(node, ast.Return))
        cat_call = returned.value
        name_map = {"loco_command": "loco_command", "upper_last_action": "last_action",
                    "base_angular_velocity": "base_ang_vel",
                    "projected_gravity": "projected_gravity",
                    "last_locomotion_action": "last_loco_action",
                    "joint_pos_rel_policy_order": "joint_pos",
                    "joint_vel_policy_order": "joint_vel"}
        parsed = []
        for element in cat_call.args[0].elts:
            if isinstance(element, ast.Call):
                parsed.append((name_map[element.func.id], 1.0))
            elif isinstance(element, ast.BinOp) and isinstance(element.op, ast.Mult):
                parsed.append((name_map[element.left.func.id], float(element.right.value)))
            else:  # 帧布局换了写法就必须让这条测试失败，而不是静默跳过
                self.fail(f"policy_frame 里出现未识别的项：{ast.dump(element)}")
        self.assertEqual(parsed, [(name, scale) for name, _, scale in runtime.FRAME_TERMS])


@_NEEDS_TORCH
class RuntimeActTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def test_actor_and_decoder_dims_after_loading(self):
        instance, _, _, _ = make_runtime(self._tmp.name)
        self.assertEqual(instance.actor.memory_a.rnn.input_size, 79)
        self.assertEqual(instance.actor.memory_c.rnn.input_size, runtime.DEFAULT_CRITIC_OBS_DIM)
        self.assertEqual(instance.actor.memory_a.rnn.hidden_size, 256)
        self.assertEqual(instance.decoder.frame_dim, 57)
        self.assertEqual(instance.decoder.output_dim, 6)
        self.assertEqual(instance.decoder.latent_dim, 16)
        self.assertEqual(instance.iteration, 42)
        self.assertEqual(instance.contract,
                         {"version": 2, "frame_dim": 57, "explicit_dim": 6, "latent_dim": 16})

    def test_zero_initialised_actor_starts_at_zero_residual(self):
        """训练侧 towing runner 把 actor 末层零初始化 ⇒ 首拍残差精确为 0（起点 = 冻结步态）。"""
        instance, _, _, _ = make_runtime(self._tmp.name)
        delta, processed = instance.act(**act_kwargs(3))
        self.assertEqual(tuple(delta.shape), (3, 12))
        self.assertEqual(float(delta.abs().max()), 0.0)
        self.assertEqual(float(processed.abs().max()), 0.0)
        self.assertEqual(float(instance._last_action.abs().max()), 0.0)

    def test_delta_is_clamped_action_times_action_scale(self):
        instance, actor, _, _ = make_runtime(self._tmp.name)
        set_actor_output(instance, actor, 5.0)      # 饱和 ⇒ processed 必须被 clamp 到 ±1
        delta, processed = instance.act(**act_kwargs(3))
        self.assertTrue(torch.allclose(processed, torch.ones(3, 12)))
        scale = instance.action_scale.expand(3, 12)
        self.assertTrue(torch.allclose(delta, scale))
        self.assertTrue(bool((delta.abs() <= scale + 1e-7).all()))
        set_actor_output(instance, actor, -5.0)     # 负向饱和同样被 clamp（−5 → −1）
        delta, processed = instance.act(**act_kwargs(3))
        self.assertTrue(torch.allclose(processed, -torch.ones(3, 12)))
        self.assertTrue(torch.allclose(delta, -scale))

    def test_action_scale_is_the_frozen_policy_contract(self):
        instance, _, _, _ = make_runtime(self._tmp.name)
        self.assertEqual(tuple(instance.action_scale.tolist()), AMP_ACTION_SCALE)
        self.assertEqual(sorted(set(instance.action_scale.tolist())), [0.125, 0.25])
        # 髋关节是 0.125、大腿/小腿是 0.25（策略关节顺序逐腿 hip/thigh/shank）
        self.assertEqual([float(value) for value in instance.action_scale[:6]],
                         [0.125, 0.25, 0.25, 0.125, 0.25, 0.25])

    def test_last_action_is_the_previous_clamped_action(self):
        """帧里的 `last_action` 位 = 上一拍 clamp 后的动作（不是本拍、也不是未裁剪值）。"""
        instance, actor, _, _ = make_runtime(self._tmp.name, num_envs=2)
        kwargs = act_kwargs(2)
        frame_before = instance.build_frame(**kwargs)
        self.assertEqual(float(frame_before[:, 3:15].abs().max()), 0.0)
        set_actor_output(instance, actor, 0.4)       # 0.4 < 1 ⇒ 不触发 clamp，能区分"上一拍"
        _, processed_first = instance.act(**kwargs)
        self.assertTrue(torch.allclose(processed_first, torch.full((2, 12), 0.4)))
        self.assertTrue(torch.allclose(instance._last_action, processed_first))
        frame_second = instance.build_frame(**kwargs)
        self.assertTrue(torch.allclose(frame_second[:, 3:15], processed_first))

    def test_frame_slices_follow_the_declared_layout(self):
        instance, _, _, _ = make_runtime(self._tmp.name, num_envs=2)
        kwargs = act_kwargs(
            2, loco_command=torch.tensor([[0.5, -0.25, 0.1]] * 2),
            base_ang_vel=torch.full((2, 3), 2.0),
            projected_gravity=torch.tensor([[0.0, 0.0, -1.0]] * 2),
            last_loco_action=torch.full((2, 12), 3.0),
            joint_pos_rel=torch.full((2, 12), 0.2),
            joint_vel=torch.full((2, 12), 4.0))
        frame = instance.build_frame(**kwargs)
        self.assertEqual(tuple(frame.shape), (2, 57))
        self.assertTrue(torch.allclose(frame[:, 0:3], kwargs["loco_command"]))
        self.assertTrue(torch.allclose(frame[:, 3:15], torch.zeros(2, 12)))
        self.assertTrue(torch.allclose(frame[:, 15:18], torch.full((2, 3), 0.5)))   # ×0.25
        self.assertTrue(torch.allclose(frame[:, 18:21], kwargs["projected_gravity"]))
        self.assertTrue(torch.allclose(frame[:, 21:33], kwargs["last_loco_action"]))
        self.assertTrue(torch.allclose(frame[:, 33:45], kwargs["joint_pos_rel"]))
        self.assertTrue(torch.allclose(frame[:, 45:57], torch.full((2, 12), 0.2)))   # ×0.05

    def test_build_frame_rejects_wrong_shapes(self):
        instance, _, _, _ = make_runtime(self._tmp.name, num_envs=3)
        with self.assertRaises(ValueError):
            instance.build_frame(**act_kwargs(2))               # 批量数不对
        with self.assertRaises(ValueError):
            instance.build_frame(**act_kwargs(3, joint_vel=torch.zeros(3, 11)))

    def test_reset_clears_action_and_hidden_states(self):
        instance, actor, _, _ = make_runtime(self._tmp.name, num_envs=3)
        set_actor_output(instance, actor, 0.5)
        instance.act(**act_kwargs(3))
        self.assertIsNotNone(instance.decoder_hidden)
        self.assertIsNotNone(instance.actor.memory_a.hidden_states)
        instance.reset()
        self.assertEqual(float(instance._last_action.abs().max()), 0.0)
        self.assertIsNone(instance.decoder_hidden)
        self.assertIsNone(instance.actor.memory_a.hidden_states)
        # 全量 reset 后又能得到同一动作（GRU 从零起步）
        _, processed = instance.act(**act_kwargs(3))
        self.assertTrue(torch.allclose(processed, torch.full((3, 12), 0.5)))
        # 只重置部分 env：其余环境的上一拍动作保留
        instance.reset(env_ids=[0])
        self.assertEqual(float(instance._last_action[0].abs().max()), 0.0)
        self.assertTrue(torch.allclose(instance._last_action[1], torch.full((12,), 0.5)))

    def test_stochastic_mode_still_respects_the_clamp(self):
        instance, actor, _, _ = make_runtime(self._tmp.name, num_envs=4, deterministic=False)
        set_actor_output(instance, actor, 3.0)
        delta, processed = instance.act(**act_kwargs(4))
        self.assertLessEqual(float(processed.abs().max()), 1.0 + 1e-6)
        self.assertTrue(bool((delta.abs() <= instance.action_scale + 1e-6).all()))

    def test_action_scale_is_validated(self):
        spec = runtime.UpperNetworkSpec.from_agent_cfg(registered_agent_cfg_stub())
        actor = build_actor(spec)
        decoder = build_decoder(spec)
        path = write_checkpoint(self._tmp.name, actor=actor, decoder=decoder,
                                contract=runtime.expected_towing_contract(spec))
        for bad in ([0.25] * 11, [0.25] * 11 + [0.0], [0.25] * 11 + [float("nan")]):
            with self.assertRaises(ValueError):
                runtime.UpperPolicyRuntime(path, num_envs=1, action_scale=bad, spec=spec,
                                           network_modules=network_modules())


@_NEEDS_TORCH
class CheckpointContractTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.spec = runtime.UpperNetworkSpec.from_agent_cfg(registered_agent_cfg_stub())
        self.actor = build_actor(self.spec)
        self.decoder = build_decoder(self.spec)

    def _runtime_for_contract(self, contract):
        path = write_checkpoint(self._tmp.name, actor=self.actor, decoder=self.decoder,
                                contract=contract)
        return runtime.UpperPolicyRuntime(path, num_envs=2, action_scale=AMP_ACTION_SCALE,
                                          spec=self.spec, network_modules=network_modules())

    def test_legacy_and_mismatched_contracts_are_rejected(self):
        """旧 56/63 维（frame 51/56）与任何字段漂移都必须在加载前被拒。"""
        for contract in (
            {"version": 2, "frame_dim": 56, "explicit_dim": 5, "latent_dim": 16},
            {"version": 2, "frame_dim": 57, "explicit_dim": 6, "latent_dim": 8},
            {"version": 1, "frame_dim": 57, "explicit_dim": 6, "latent_dim": 16},
            {"version": 2, "frame_dim": 63, "explicit_dim": 6, "latent_dim": 16},
            "not-a-contract",
            None,
        ):
            with self.assertRaises(runtime.TowingCheckpointContractError):
                self._runtime_for_contract(contract)

    def test_error_message_names_the_contract(self):
        with self.assertRaises(runtime.TowingCheckpointContractError) as caught:
            self._runtime_for_contract({"version": 2, "frame_dim": 56, "explicit_dim": 6,
                                        "latent_dim": 16})
        message = str(caught.exception)
        self.assertIn("towing_contract", message)
        self.assertIn("frame_dim", message)
        self.assertIn("56", message)

    def test_missing_keys_are_rejected(self):
        path = Path(self._tmp.name) / "incomplete.pt"
        torch.save({"towing_contract": runtime.expected_towing_contract(self.spec)}, path)
        with self.assertRaises(runtime.TowingCheckpointContractError):
            runtime.UpperPolicyRuntime(path, num_envs=1, action_scale=AMP_ACTION_SCALE,
                                       spec=self.spec, network_modules=network_modules())

    def test_missing_file_is_reported(self):
        with self.assertRaises(FileNotFoundError):
            runtime.UpperPolicyRuntime(Path(self._tmp.name) / "nope.pt", num_envs=1,
                                       action_scale=AMP_ACTION_SCALE, spec=self.spec,
                                       network_modules=network_modules())

    def test_strict_load_catches_a_forged_contract(self):
        """契约可以骗过校验，但旧 63 维权重躲不过 `load_state_dict(strict=True)`。"""
        old_actor = build_actor(self.spec, num_actor_obs=63)   # 旧 63 维 actor（56 帧 + 7）
        path = write_checkpoint(self._tmp.name, actor=old_actor, decoder=self.decoder,
                                contract=runtime.expected_towing_contract(self.spec))
        with self.assertRaises(RuntimeError):
            runtime.UpperPolicyRuntime(path, num_envs=1, action_scale=AMP_ACTION_SCALE,
                                       spec=self.spec, network_modules=network_modules())


class OfflineImportTests(unittest.TestCase):
    """运行时要保持「纯 torch + 标准库、惰性 import rl_lab」——否则 --dry-run 会失去可运行性。"""

    def setUp(self):
        self.source = (TOWING_SCRIPTS / "upper_policy_runtime.py").read_text(encoding="utf-8")
        self.tree = ast.parse(self.source)

    def test_no_isaaclab_import_anywhere(self):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                self.assertFalse(any(alias.name.split(".")[0] == "isaaclab"
                                     for alias in node.names), ast.dump(node))
            elif isinstance(node, ast.ImportFrom):
                self.assertNotEqual((node.module or "").split(".")[0], "isaaclab")

    def test_module_level_imports_are_stdlib_or_torch_only(self):
        allowed = {"__future__", "dataclasses", "pathlib", "importlib", "sys", "types",
                   "typing", "torch"}
        roots = set()
        for node in self.tree.body:
            if isinstance(node, ast.Import):
                roots |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                roots.add((node.module or "").split(".")[0])
        self.assertTrue(roots <= allowed, f"模块顶部出现额外 import：{sorted(roots - allowed)}")
        self.assertIn("torch", roots)

    def test_play_script_never_imports_torch_at_module_level(self):
        """`play_towing_test.py` 的仿真 import 必须在 `main()` 里面（含新增的 torch 使用者）。"""
        tree = ast.parse((TOWING_SCRIPTS / "play_towing_test.py").read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Import):
                self.assertNotIn("torch", {alias.name.split(".")[0] for alias in node.names})
            elif isinstance(node, ast.ImportFrom):
                self.assertNotIn((node.module or "").split(".")[0], {"torch", "isaaclab"})


if __name__ == "__main__":
    unittest.main()
