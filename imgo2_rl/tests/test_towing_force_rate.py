"""离线契约测试：拉力变化率惩罚（TOW-25，2026-10-10 用户要求）。

本机**没有 Isaac Lab**（`upper_mdp` 顶层 import isaaclab，离线 import 不了），所以这里是
纯逻辑 + 源码 AST 两层验证，与 `test_towing_upper_rl_contract.py` 里的既有做法一致：

- **纯逻辑**：把 `upper_mdp.towing_force_rate_penalty` / `obs_force_rate_max` 的函数体从源码
  里抽出来（`ast.get_source_segment`），在只含 `torch` 与 `_term` 的 stub 命名空间里实跑；
- **源码守卫**：缓存更新位置（`process_actions`，不是 200 Hz 子步）、`reset()` 清零、
  `UpperRewardsCfg` 的注册行与参数、函数签名默认值。

覆盖用户点名的 9 条：①快速绷紧被罚且量级符合公式；②卸载到 0 精确放行；③负载态内断崖掉力
被 `down` 罚；④`rate_limit` 内精确为 0；⑤杆行过零放行（同一 `slack` 掩码）；⑥reset 后第一拍
rate 精确为 0；⑦`clip_max` 生效；⑧cfg 参数默认值与类型；⑨200 Hz 诊断读后清零。

**不覆盖**：把奖励接到 `ManagerBasedRLEnv` 上的运行时行为（20 Hz 节拍的缓存是否真的落在
两个控制步之间、`force_rate_max` 在真实子步上的读数）——只能在训练机实跑，见 README TOW-25。
"""

import ast
from pathlib import Path
import textwrap
import unittest

try:
    import torch
except ImportError:  # 离线契约检查也可能在没有 torch 的解释器里跑（此时只保留静态用例）
    torch = None


RL = Path(__file__).resolve().parents[1]
PKG = RL / "source/imgo2_rl/imgo2_rl/tasks/manager_based/towing"

MDP_SOURCE = (PKG / "upper_mdp.py").read_text("utf-8")
CFG_SOURCE = (PKG / "upper_env_cfg.py").read_text("utf-8")

STEP_DT = 0.05

_NEEDS_TORCH = unittest.skipIf(torch is None, "PyTorch is not installed in the offline-check interpreter")


# ---------------------------------------------------------------------------
# 从源码里抽函数体，在 stub 上实跑
# ---------------------------------------------------------------------------

def _function_node(source, name):
    tree = ast.parse(source)
    return next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == name), tree


def _function_source(source, name):
    node, _tree = _function_node(source, name)
    return textwrap.dedent(ast.get_source_segment(source, node)), node


_FUNCTION_CACHE: dict = {}


def _function(name):
    """把 `upper_mdp.<name>` 抽出来在 stub 命名空间里 exec（按名缓存）。

    ⚠ 返回的是**普通函数**，测试里必须经 `_function(name)(env)` 调用；不要把它挂到
    `TestCase` 的类属性上——那会被 Python 当成方法绑定，第一个实参变成 `self`。
    """
    if name not in _FUNCTION_CACHE:
        body, _node = _function_source(MDP_SOURCE, name)
        namespace = {"torch": torch, "_term": lambda env: env.term}
        exec(body, namespace)  # noqa: S102 - 源码来自本仓库，测试专用
        _FUNCTION_CACHE[name] = namespace[name]
    return _FUNCTION_CACHE[name]


def _literal(node):
    """字面量（含负号）求值。"""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_literal(node.operand)
    return ast.literal_eval(node)


def _class_body(source, class_name):
    """返回 `UpperRewardsCfg` 的直接子节点列表（用于解析奖励注册行）。"""
    tree = ast.parse(source)
    cls = next(node for node in ast.walk(tree)
               if isinstance(node, ast.ClassDef) and node.name == class_name)
    return cls.body


def _reward_term(class_body, name):
    """从类体解析 `name = RewTerm(func=mdp.X, weight=W, params={...})`。"""
    node = next(node for node in class_body
                if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in node.targets))
    kwargs = {keyword.arg: keyword.value for keyword in node.value.keywords}
    if not isinstance(kwargs["func"], ast.Attribute):
        raise AssertionError("func 必须是 mdp.<name>")
    params = {}
    if "params" in kwargs:
        params = {key.value: _literal(value) for key, value in zip(kwargs["params"].keys,
                                                                  kwargs["params"].values)}
    return {"func": kwargs["func"].attr, "weight": _literal(kwargs["weight"]), "params": params}


def _signature_defaults(node):
    """返回 `def f(env, a=..., b=...)` 里带默认值的形参 → 默认值。"""
    args = node.args.args[-len(node.args.defaults):]
    return {arg.arg: _literal(default) for arg, default in zip(args, node.args.defaults)}


# ---------------------------------------------------------------------------
# stub：动作项（term）与环境
# ---------------------------------------------------------------------------

class _StubTerm:
    """`HierarchicalVelocityAction` 的最小替身：只放本项读的那几个量。

    `towing_force_b` 按 `sign` 只加在机体系 x 上（模长 = `mag`），用来模拟杆行"拉/推"
    两个方向——`RigidLink` 的 `rope_tension` 有符号（`mdp/rope_model.py`），力向量整体反向。
    """

    def __init__(self, mag, prev_mag=None, has_prev=True, sign=1.0):
        mag = torch.as_tensor(mag, dtype=torch.float32).reshape(-1)
        n = mag.numel()
        self.towing_force_b = torch.zeros(n, 3)
        self.towing_force_b[:, 0] = torch.as_tensor(sign, dtype=torch.float32) * mag
        self.prev_force_mag = (torch.zeros(n) if prev_mag is None
                               else torch.as_tensor(prev_mag, dtype=torch.float32).reshape(-1))
        if isinstance(has_prev, (list, tuple)):
            self.force_rate_has_prev = torch.tensor(has_prev, dtype=torch.bool)
        else:
            self.force_rate_has_prev = torch.full((n,), bool(has_prev), dtype=torch.bool)


class _StubEnv:
    step_dt = STEP_DT

    def __init__(self, term):
        self.term = term


def _penalty(mag, prev_mag, has_prev=True, sign=1.0, **kwargs):
    return _function("towing_force_rate_penalty")(
        _StubEnv(_StubTerm(mag, prev_mag, has_prev, sign)), **kwargs)


class ForceRateFormulaTests(unittest.TestCase):
    """①–⑤⑦：公式与掩码的方向语义（纯逻辑，实跑抽出来的函数体）。"""

    @_NEEDS_TORCH
    def test_fast_tautening_is_penalised_with_the_documented_magnitude(self):
        """① 0 → 大（快速绷紧 = 被猛拽/绷直冲击）**必罚**，且量级与公式一致。

        `rate = (20 − 0)/0.05 = 400 N/s = 2 × rate_limit` ⇒ `up = (400 − 200)/200 = 1.0`。
        """
        value = float(_penalty([20.0], [0.0])[0])
        self.assertAlmostEqual(value, 1.0, places=5)
        self.assertGreater(value, 0.0)

    @_NEEDS_TORCH
    def test_documented_calibration_yields_minus_0_025_per_step(self):
        """标定依据（写进 docstring 的那条）：`up = 1` ⇒ 每步 `−0.5 × 1.0 × 0.05 = −0.025`，
        ≈ `tracking_velocity` 满额（+0.05/步）的一半。权重从 cfg 读出，不手抄。"""
        term = _reward_term(_class_body(CFG_SOURCE, "UpperRewardsCfg"), "towing_force_rate")
        weight = float(term["weight"])
        self.assertEqual(weight, -0.5)
        up = float(_penalty([20.0], [0.0])[0])
        self.assertAlmostEqual(weight * up * STEP_DT, -0.025, places=6)

    @_NEEDS_TORCH
    def test_unloading_to_slack_is_allowed_exactly(self):
        """② 大 → ≈0（绳松弛 / 杆卸载）**放行**：`down` 被 `slack` 掩码关掉，**精确 0**。"""
        value = float(_penalty([0.5], [800.0])[0])
        self.assertEqual(value, 0.0)
        # 这里的 rate 极大（−15990 N/s），若不掩码 `down` 会远大于 clip —— 证明是掩码在放行；
        # 对照：同一段掉力只落到 5 N（仍 > slack_eps）时被罚满 clip。
        unmasked = (15990.0 - 600.0) / 600.0
        self.assertGreater(unmasked, 3.0)
        self.assertEqual(float(_penalty([5.0], [800.0])[0]), 3.0)

    @_NEEDS_TORCH
    def test_cliff_within_load_is_penalised_by_the_down_branch(self):
        """③ 负载态内断崖掉力（大 → 中等，仍 > `slack_eps_n`）被 `down` 罚。

        `rate = (60 − 100)/0.05 = −800 N/s` ⇒ `down = (800 − 600)/600 = 1/3`，`up = 0`。
        """
        value = float(_penalty([60.0], [100.0])[0])
        self.assertAlmostEqual(value, 1.0 / 3.0, places=5)
        self.assertGreater(value, 0.0)

    @_NEEDS_TORCH
    def test_slack_mask_uses_the_exact_threshold(self):
        """掩码阈值是**严格小于** `slack_eps_n`：`mag_t == 1 N` 不算松弛（与源码头一致）。"""
        self.assertEqual(float(_penalty([0.999], [800.0])[0]), 0.0)
        self.assertGreater(float(_penalty([1.0], [800.0])[0]), 0.0)

    @_NEEDS_TORCH
    def test_rates_inside_the_limits_are_exactly_zero(self):
        """④ 两侧都在免罚速率内 ⇒ **精确 0**（`relu` 的负半轴精确为 0，不是小量）。"""
        # 加载 100 N/s < 200
        self.assertEqual(float(_penalty([15.0], [10.0])[0]), 0.0)
        # 卸载 −400 N/s：绝对值 < 600，且 mag=20 > slack_eps ⇒ 两侧都免罚
        self.assertEqual(float(_penalty([20.0], [40.0])[0]), 0.0)
        # 力不变
        self.assertEqual(float(_penalty([30.0], [30.0])[0]), 0.0)
        # 恰好落在限值上：浮点除法下不是严格 0，但也必须远小于一个 penalty 单位
        self.assertLess(float(_penalty([20.0], [10.0])[0]), 1.0e-6)
        self.assertLess(float(_penalty([10.0], [40.0])[0]), 1.0e-6)

    @_NEEDS_TORCH
    def test_rigid_zero_crossing_is_allowed_in_both_directions(self):
        """⑤ 杆行过零（拉 → 推 / 推 → 拉，穿越 ≈0）**放行**——同一个 `slack` 掩码覆盖。

        `RigidLink` 的张力有符号（`rope_tension` 负值 = 推力），`towing_force_b` 整个反向，
        而 `mag = ‖F‖` 只看到"掉到 ≈0" ⇒ 过零那一下必落在 `slack` 掩码里。
        对照：同一段掉力若**没有掉到 ≈0**（仍 5 N > 1 N），就会被 `down` 罚到 clip。
        """
        # 拉 +400 N → 推 −0.2 N（模长 0.2 < 1）；方向反过来同理
        self.assertEqual(float(_penalty([0.2], [400.0], sign=-1.0)[0]), 0.0)
        self.assertEqual(float(_penalty([0.2], [400.0], sign=1.0)[0]), 0.0)
        # 没有过零的同量卸载（仍处于负载态）则被罚满 clip
        self.assertEqual(float(_penalty([5.0], [400.0])[0]), 3.0)

    @_NEEDS_TORCH
    def test_clip_max_caps_the_penalty(self):
        """⑦ 极端峰值被 `clip_max` 截住（`up`/`down` 都不会超过 3.0）。"""
        self.assertEqual(float(_penalty([800.0], [100.0])[0]), 3.0)
        # 参数化后的 clip 也必须生效（不是把默认值写死在函数里）
        self.assertEqual(float(_penalty([800.0], [100.0], clip_max=1.25)[0]), 1.25)

    @_NEEDS_TORCH
    def test_penalty_is_non_negative_and_bounded_over_a_sweep(self):
        """全域扫一遍：越界/NaN 都不许出现，且卸载到松弛侧恒为 0。"""
        magnitudes = [0.0, 0.2, 0.5, 1.0, 2.0, 5.0, 20.0, 60.0, 200.0, 800.0, 3000.0]
        for current in magnitudes:
            for previous in magnitudes:
                value = float(_penalty([current], [previous])[0])
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 3.0)
                if current < 1.0 < previous:
                    self.assertEqual(value, 0.0,
                                     f"卸载到松弛必须精确放行：{previous}→{current}")


class ResetAndCacheTests(unittest.TestCase):
    """⑥ + 缓存位置（20 Hz 控制步，不在 200 Hz 子步里）。"""

    @staticmethod
    def _method_body(name):
        body = MDP_SOURCE[MDP_SOURCE.index(f"    def {name}(self"):]
        return body[:body.index("\n    def ", 1)]

    @_NEEDS_TORCH
    def test_first_tick_after_reset_has_exactly_zero_rate(self):
        """⑥ reset 后第一拍 `rate` **精确为 0**：没有上一拍样本时不拿 0 当上一拍去算。

        实测：`has_prev=False` 时，哪怕出生那一拍力就有 5000 N，本项也必须精确为 0
        （否则等于在 spawn 处凭空扣分，破坏仓库「spawn 精确为 0」的纪律）。
        """
        self.assertEqual(float(_penalty([5000.0], [0.0], has_prev=False)[0]), 0.0)
        # 逐 env 门控：同一批里未复位的 env 照罚，刚复位的 env 放行
        values = _penalty([5000.0, 5000.0], [0.0, 0.0], has_prev=[True, False])
        self.assertEqual(float(values[0]), 3.0)
        self.assertEqual(float(values[1]), 0.0)

    def test_first_tick_gate_does_not_mask_the_real_takeup(self):
        """首拍门只盖住 settle 段的**第一拍**，不是掩护真实猛拽的漏洞。

        发力（`tow_start_s = SETTLE_TIME_S`）在 1.0 s 之后 ⇒ 起步绷直发生在第 20 拍左右，
        早就过了首拍门；所以"第一拍不罚"只保证出生那拍不误罚，不会放过起步冲击。
        """
        geometry = (PKG / "mdp/episode_geometry.py").read_text("utf-8")
        settle = float(next(line.split("=")[1] for line in geometry.splitlines()
                            if line.startswith("SETTLE_TIME_S")))
        self.assertEqual(settle, 1.0)
        self.assertIn("tow_start_s: float = SETTLE_TIME_S", MDP_SOURCE)
        self.assertGreater(settle / STEP_DT, 1.0,
                           "发力时刻必须远晚于第一拍，否则首拍门会掩护起步绷直")

    def test_reset_clears_every_piece_of_rate_state(self):
        """`reset()` 必须把缓存、首拍门与子步诊断全部清零。"""
        reset = self._method_body("reset")
        for fragment in ("self.prev_force_mag[env_ids] = 0",
                         "self.force_rate_has_prev[env_ids] = False",
                         "self._force_rate_tick_seen[env_ids] = False",
                         "self.force_rate_max[env_ids] = 0",
                         "self._phys_prev_force_mag[env_ids] = 0"):
            self.assertIn(fragment, reset)

    def test_previous_mag_is_updated_once_per_control_step(self):
        """「上一拍」只在 `process_actions`（20 Hz）更新，**不在** `_apply_towing_physics`。

        顺序也必须对：先定 `force_rate_has_prev`（取上一拍是否开始过），再取当前 mag，
        最后置位 `_force_rate_tick_seen`——写反会让第一拍就拿到"上一拍"。
        """
        process = self._method_body("process_actions")
        self.assertIn(
            "self.prev_force_mag.copy_(torch.linalg.vector_norm(self.towing_force_b, dim=1))",
            process)
        self.assertIn("self.force_rate_has_prev.copy_(self._force_rate_tick_seen)", process)
        self.assertIn("self._force_rate_tick_seen.fill_(True)", process)
        self.assertLess(process.index("self.force_rate_has_prev.copy_(self._force_rate_tick_seen)"),
                        process.index("self._force_rate_tick_seen.fill_(True)"))
        self.assertLess(process.index("self.prev_force_mag.copy_("),
                        process.index("self._force_rate_tick_seen.fill_(True)"))
        # 物理子步里只有 200 Hz 诊断自己的上一拍缓存（`_phys_prev_force_mag`），
        # 不得混进 20 Hz 的 `prev_force_mag`
        physics = self._method_body("_apply_towing_physics")
        self.assertNotIn("self.prev_force_mag", physics)
        self.assertNotIn("self.force_rate_has_prev", physics)
        self.assertIn("self._phys_prev_force_mag.copy_(force_mag)", physics)

    def test_state_is_allocated_in_init(self):
        start = MDP_SOURCE.index("    def __init__(self, cfg, env):")
        init = MDP_SOURCE[start:MDP_SOURCE.index("\n    def ", start + 10)]
        for fragment in ("self.prev_force_mag = torch.zeros(env.num_envs, device=env.device)",
                         "self.force_rate_has_prev = torch.zeros(env.num_envs, dtype=torch.bool",
                         "self._force_rate_tick_seen = torch.zeros(env.num_envs, dtype=torch.bool",
                         "self.force_rate_max = torch.zeros(env.num_envs, device=env.device)",
                         "self._phys_prev_force_mag = torch.zeros(env.num_envs, device=env.device)"):
            self.assertIn(fragment, init)


class DiagnosticTests(unittest.TestCase):
    """⑨ 200 Hz 子步诊断：读后清零 + 每控制步取最大。"""

    @_NEEDS_TORCH
    def test_read_clears_and_returns_the_pre_clear_value(self):
        term = _StubTerm([0.0])
        term.force_rate_max = torch.tensor([1234.5, 7.0])
        env = _StubEnv(term)
        first = _function("obs_force_rate_max")(env)
        self.assertTrue(torch.allclose(first, torch.tensor([1234.5, 7.0])))
        # 读后清零 ⇒ 同一个控制步内再读一次是 0（不会把同一个尖峰记两遍）
        second = _function("obs_force_rate_max")(env)
        self.assertTrue(torch.allclose(second, torch.zeros(2)))
        # 返回的是清零前的副本，不是同一个张量对象（否则返回值会被一起清掉）
        term2 = _StubTerm([0.0])
        term2.force_rate_max = torch.tensor([9.0])
        value = _function("obs_force_rate_max")(_StubEnv(term2))
        term2.force_rate_max.zero_()
        self.assertEqual(float(value[0]), 9.0)

    def test_substep_accumulator_takes_the_max_over_the_control_step(self):
        """`_apply_towing_physics` 里必须用 `max` 累积 `|Δmag| / dt_phys`。"""
        body = MDP_SOURCE[MDP_SOURCE.index("    def _apply_towing_physics(self"):]
        body = body[:body.index("\n    def ", 1)]
        self.assertIn("self.force_rate_max.copy_(torch.maximum(", body)
        self.assertIn("(force_mag - self._phys_prev_force_mag).abs() / self.cfg.physics_dt", body)
        # 必须在 `towing_force_b` 刷新**之后**算（否则用的是上一子步的力）
        self.assertLess(body.index("self.towing_force_b[:] = force_robot_b[:, 0, :3]"),
                        body.index("force_mag = torch.linalg.vector_norm(self.towing_force_b"))


class RewardCfgContractTests(unittest.TestCase):
    """⑧ 注册项、权重、params 默认值与类型，以及函数签名默认值。"""

    @classmethod
    def setUpClass(cls):
        cls.rewards = _class_body(CFG_SOURCE, "UpperRewardsCfg")
        cls.term = _reward_term(cls.rewards, "towing_force_rate")
        _, cls.node = _function_source(MDP_SOURCE, "towing_force_rate_penalty")

    def test_reward_term_is_registered_with_weight_minus_half(self):
        self.assertEqual(self.term["func"], "towing_force_rate_penalty")
        self.assertEqual(self.term["weight"], -0.5)

    def test_params_match_the_documented_defaults_and_types(self):
        self.assertEqual(self.term["params"], {"rate_limit_n_per_s": 200.0,
                                               "release_limit_n_per_s": 600.0,
                                               "slack_eps_n": 1.0,
                                               "clip_max": 3.0})
        for name, value in self.term["params"].items():
            self.assertIsInstance(value, float, f"{name} 必须是 float")
        # `slack_eps_n` 与既有 active 诊断的 1 N 阈值同口径（源码里两处都是 1.0）
        self.assertIn("towing_force_norm(env) > 1.0", MDP_SOURCE)
        self.assertEqual(self.term["params"]["slack_eps_n"], 1.0)

    def test_function_defaults_equal_the_registered_params(self):
        """函数签名默认值必须与 cfg 注册的 params **逐项相等**（防两处漂移）。"""
        defaults = _signature_defaults(self.node)
        self.assertEqual(defaults, self.term["params"])
        for name, value in defaults.items():
            self.assertIsInstance(value, float, f"{name} 默认值必须是 float")

    def test_limits_are_used_from_the_arguments_not_hardcoded(self):
        """函数体不得把默认值写死（全部走形参）——调参只需改 `params`。"""
        body, _ = _function_source(MDP_SOURCE, "towing_force_rate_penalty")
        code = body[body.index('"""', body.index('"""') + 3) + 3:]
        for literal in ("200.0", "600.0", "3.0"):
            self.assertNotIn(literal, code)
        for name in ("rate_limit_n_per_s", "release_limit_n_per_s", "slack_eps_n", "clip_max"):
            self.assertIn(name, code)

    def test_penalty_shape_matches_the_specification_line_by_line(self):
        body, _ = _function_source(MDP_SOURCE, "towing_force_rate_penalty")
        code = body[body.index('"""', body.index('"""') + 3) + 3:]
        self.assertIn("mag = torch.linalg.vector_norm(term.towing_force_b, dim=1)", code)
        self.assertIn("rate = (mag - term.prev_force_mag) / env.step_dt", code)
        self.assertIn("up = torch.relu(rate - rate_limit_n_per_s) / rate_limit_n_per_s", code)
        self.assertIn("slack = mag < slack_eps_n", code)
        self.assertIn(
            "down = (torch.relu(-rate - release_limit_n_per_s) / release_limit_n_per_s", code)
        self.assertIn("* (~slack).to(rate.dtype))", code)
        self.assertIn("return torch.clamp(up + down, 0.0, clip_max)", code)
        # 首拍门必须在 rate 算出来之后、up/down 之前生效
        self.assertIn("rate = torch.where(term.force_rate_has_prev, rate, torch.zeros_like(rate))",
                      code)

    def test_diagnostic_is_registered_with_the_tiny_weight(self):
        diag = _reward_term(self.rewards, "obs_force_rate_max")
        self.assertEqual(diag["func"], "obs_force_rate_max")
        self.assertEqual(diag["weight"], 1.0e-6)
        self.assertEqual(diag["params"], {})
        # 诊断不得读 `prev_force_mag` 那套 20 Hz 缓存（它量的是子步内的跳变）
        body, _ = _function_source(MDP_SOURCE, "obs_force_rate_max")
        self.assertIn("value = term.force_rate_max.clone()", body)
        self.assertIn("term.force_rate_max.zero_()", body)
        self.assertNotIn("prev_force_mag", body)

    def test_new_terms_never_read_loco_command(self):
        """与 `test_reward_terms_never_read_loco_command` 同一条防作弊红线（本文件自查）。"""
        for name in ("towing_force_rate_penalty", "obs_force_rate_max"):
            _, node = _function_source(MDP_SOURCE, name)
            offenders = [sub.lineno for sub in ast.walk(node)
                         if isinstance(sub, ast.Attribute) and sub.attr == "loco_command"]
            self.assertEqual(offenders, [], f"{name} 不得读 loco_command")

    def test_existing_stop_and_clearance_terms_are_untouched(self):
        """用户明确要求：不动 `stop_towing_force`（−1.0）与 `min_clearance`（−5.0/0.85）。"""
        stop = _reward_term(self.rewards, "stop_towing_force")
        self.assertEqual((stop["func"], stop["weight"]), ("post_stop_towing_force", -1.0))
        clearance = _reward_term(self.rewards, "min_clearance")
        self.assertEqual((clearance["func"], clearance["weight"]),
                         ("min_clearance_violation", -5.0))
        self.assertEqual(clearance["params"], {"spawn_margin": 0.85, "softness": 0.02})
        self.assertNotIn("extra_distance = RewTerm", CFG_SOURCE)


if __name__ == "__main__":
    unittest.main()
