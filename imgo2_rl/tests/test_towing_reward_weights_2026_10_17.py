"""离线契约测试：2026-10-17 的三处奖励权重改动（**2026-10-19 已部分覆盖**）。

2026-10-19 用户随验收口径收束又改了三条：`clearance` 与 `min_clearance` 权重 **→ 0.0**、
`collision` **全域 → −1.0**（不再按下坡缩放）。本文件保留原 10-17 的 `action_rate` 与
`cart_collision_cost` 函数级测试，并把已覆盖的注册行断言更新到 10-19 口径。


用户 2026-10-17 要求（原文）：

1. `clearance` 权重 **−1.0 → −10.0**；
2. `action_rate` 权重 **−0.1 → −0.05**；
3. `collision` **只在下坡路段（只在下坡段落）降到 −1.0**，其他情况不变（仍 −5.0）。

本机**没有 Isaac Lab**（`upper_mdp` 顶层 import isaaclab，离线 import 不了），所以这里是
纯逻辑 + 源码两层验证，与 `test_towing_force_rate.py` / `test_towing_upper_rl_contract.py`
的既有做法一致：

- **纯逻辑**：把 `upper_mdp.downhill_segment` / `cart_collision_cost` 的函数体从源码里抽出来
  （`ast.get_source_segment`），在只含 `torch` / `_term` / 两个剖面孔常量的 stub 命名空间里实跑；
- **几何交叉核对**：下坡 mask 必须与 `slope_geometry.profile_slope_degrees(grade, x) < 0`
  在**整条剖面**上逐点一致（这是"下坡段"的定义，不是手抄的区间）；
- **源码/注册行守卫**：三个 `RewTerm` 的权重与参数、`−5.0 × 0.2 == −1.0`、旧字面量不得回潮、
  单条 `collision` 曲线（不得拆成第二条 `RewTerm`）、mask 不读 `terrain_profile`（rolling 档的
  已知近似，见 `downhill_segment` 的 docstring）。

**不覆盖**：把奖励接到 `ManagerBasedRLEnv` 上的运行时行为（真实轨迹上下坡 mask 的触发占比、
`Episode_Reward/clearance` 与 `collision` 的实际量级）——只能在训练机实跑，见 README TOW-REW-01。
"""

import ast
from pathlib import Path
import sys
import textwrap
import unittest

try:
    import torch
except ImportError:  # 没有 torch 的解释器里只保留静态用例
    torch = None


RL = Path(__file__).resolve().parents[1]
PKG = RL / "source/imgo2_rl/imgo2_rl/tasks/manager_based/towing"
MDP_DIR = PKG / "mdp"

if str(MDP_DIR) not in sys.path:
    sys.path.insert(0, str(MDP_DIR))
import slope_geometry  # noqa: E402  （纯标准库，离线可用）

#: 与 `UpperRewardsCfg` 的 step_dt 一致（`sim.dt 0.005 × decimation 10`）。
STEP_DT = 0.05

MDP_SOURCE = (PKG / "upper_mdp.py").read_text("utf-8")
CFG_SOURCE = (PKG / "upper_env_cfg.py").read_text("utf-8")

_NEEDS_TORCH = unittest.skipIf(
    torch is None, "PyTorch is not installed in the offline-check interpreter")


# ---------------------------------------------------------------------------
# 从源码里抽函数体，在 stub 上实跑
# ---------------------------------------------------------------------------

def _function_source(source, name):
    tree = ast.parse(source)
    node = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == name)
    return textwrap.dedent(ast.get_source_segment(source, node)), node


_NAMESPACE: dict = {}


def _mdp_functions():
    """`downhill_segment` + `cart_collision_cost` 的 stub 命名空间（按需构建一次）。"""
    if not _NAMESPACE:
        namespace = {
            "torch": torch,
            "_term": lambda env: env.term,
            "DOWN_START_M": slope_geometry.DOWN_START_M,
            "FLAT_OUT_START_M": slope_geometry.FLAT_OUT_START_M,
            # `cart_collision(env)` 在真实现里要刷安全状态；stub 直接读 env.collision。
            "cart_collision": lambda env: env.term.collision,
        }
        for name in ("downhill_segment", "cart_collision_cost"):
            body, _node = _function_source(MDP_SOURCE, name)
            exec(body, namespace)  # noqa: S102 - 源码来自本仓库，测试专用
        _NAMESPACE.update(namespace)
    return _NAMESPACE


def _downhill_mask(xs, speedbump=False, wave=False):
    fn = _mdp_functions()["downhill_segment"]
    x = torch.as_tensor(xs, dtype=torch.float32).reshape(-1, 1)
    n = x.shape[0]
    term = type("Term", (), {})()
    term._asset = type("Asset", (), {})()
    term._asset.data = type("Data", (), {})()
    term._asset.data.root_pos_w = torch.cat((x, torch.zeros(n, 2)), dim=1)
    term.terrain_speedbump = torch.full((n,), bool(speedbump), dtype=torch.bool)
    term.terrain_wave = torch.full((n,), bool(wave), dtype=torch.bool)
    env = type("Env", (), {})()
    env.term = term
    env.scene = type("Scene", (), {})()
    env.scene.env_origins = torch.zeros(n, 3)
    return fn(env)


def _cost(xs, collision, downhill_scale=None):
    fn = _mdp_functions()["cart_collision_cost"]
    n = len(xs)
    x = torch.as_tensor(xs, dtype=torch.float32).reshape(-1, 1)
    term = type("Term", (), {})()
    term._asset = type("Asset", (), {})()
    term._asset.data = type("Data", (), {})()
    term._asset.data.root_pos_w = torch.cat((x, torch.zeros(n, 2)), dim=1)
    term.terrain_speedbump = torch.zeros(n, dtype=torch.bool)
    term.terrain_wave = torch.zeros(n, dtype=torch.bool)
    term.collision = torch.as_tensor(collision, dtype=torch.bool)
    env = type("Env", (), {})()
    env.term = term
    env.scene = type("Scene", (), {})()
    env.scene.env_origins = torch.zeros(n, 3)
    if downhill_scale is None:
        return fn(env)
    return fn(env, downhill_scale=downhill_scale)


# ---------------------------------------------------------------------------
# 解析奖励注册行
# ---------------------------------------------------------------------------

def _literal(node):
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_literal(node.operand)
    return ast.literal_eval(node)


def _reward_term(name):
    tree = ast.parse(CFG_SOURCE)
    cls = next(node for node in ast.walk(tree)
               if isinstance(node, ast.ClassDef) and node.name == "UpperRewardsCfg")
    node = next(node for node in cls.body
                if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in node.targets))
    kwargs = {keyword.arg: keyword.value for keyword in node.value.keywords}
    assert isinstance(kwargs["func"], ast.Attribute)
    params = {}
    if "params" in kwargs:
        params = {key.value: _literal(value)
                  for key, value in zip(kwargs["params"].keys, kwargs["params"].values)}
    return {"func": kwargs["func"].attr, "weight": _literal(kwargs["weight"]),
            "params": params}


def _reward_term_names():
    tree = ast.parse(CFG_SOURCE)
    cls = next(node for node in ast.walk(tree)
               if isinstance(node, ast.ClassDef) and node.name == "UpperRewardsCfg")
    return [target.id for node in cls.body if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
            and isinstance(node.value, ast.Call)
            and getattr(node.value.func, "id", None) == "RewTerm"]


# ---------------------------------------------------------------------------
# ① clearance −10.0
# ---------------------------------------------------------------------------

class ClearanceWeightTests(unittest.TestCase):
    def test_weight_is_zero_since_2026_10_19(self):
        term = _reward_term("clearance")
        self.assertEqual(term["func"], "clearance_barrier")
        self.assertEqual(term["weight"], 0.0)      # 2026-10-19: −10.0 → 0.0
        self.assertEqual(term["params"], {"warning_distance": 0.20, "scale": 0.05})

    def test_old_weights_do_not_come_back(self):
        self.assertNotIn("clearance = RewTerm(func=mdp.clearance_barrier, weight=-10.0",
                         CFG_SOURCE)
        self.assertNotIn("clearance = RewTerm(func=mdp.clearance_barrier, weight=-1.0",
                         CFG_SOURCE)

    @_NEEDS_TORCH
    def test_magnitude_at_the_three_anchor_gaps(self):
        """每步 = weight × func × step_dt；三个锚点的函数值与每步值都要与注释一致。"""
        import math
        for gap, func, per_step in ((0.20, 0.6931, -0.3466),
                                    (0.10, 2.1269, -1.0635),
                                    (0.00, 4.0181, -2.0091)):
            with self.subTest(gap=gap):
                self.assertAlmostEqual(func, math.log1p(math.exp((0.20 - gap) / 0.05)),
                                       places=4)
                self.assertAlmostEqual(per_step, -10.0 * func * STEP_DT, places=3)

    @_NEEDS_TORCH
    def test_spawn_gap_is_outside_the_soft_barrier(self):
        """出生那拍不该被提权后的软障碍扣分：最短绳行 0.365 m、最短杆行 0.473 m。"""
        import math
        for gap in (0.365, 0.473):
            with self.subTest(gap=gap):
                cost = -10.0 * math.log1p(math.exp((0.20 - gap) / 0.05)) * STEP_DT
                self.assertGreater(cost, -0.02)


# ---------------------------------------------------------------------------
# ② action_rate −0.05
# ---------------------------------------------------------------------------

class ActionRateWeightTests(unittest.TestCase):
    def test_weight_is_halved(self):
        term = _reward_term("action_rate")
        self.assertEqual(term["func"], "action_rate_l2")
        self.assertEqual(term["weight"], -0.05)

    def test_old_weight_does_not_come_back(self):
        self.assertNotIn("action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)",
                         CFG_SOURCE)

    def test_action_magnitude_untouched(self):
        """用户只点了 action_rate；action_magnitude 必须还是 −0.05（恰好同值，勿混淆）。"""
        self.assertEqual(_reward_term("action_magnitude")["weight"], -0.05)

    def test_per_step_magnitudes(self):
        """12 维残差：各变 1 ⇒ func 12 ⇒ −0.03/步；+1 翻到 −1 ⇒ func 48 ⇒ −0.12/步。"""
        self.assertAlmostEqual(-0.05 * 12 * STEP_DT, -0.03, places=12)
        self.assertAlmostEqual(-0.05 * 48 * STEP_DT, -0.12, places=12)


# ---------------------------------------------------------------------------
# ③ collision 只在下坡段 −1.0
# ---------------------------------------------------------------------------

class RegisteredCollisionTests(unittest.TestCase):
    def test_registered_weight_is_globally_minus_one(self):
        term = _reward_term("collision")
        self.assertEqual(term["func"], "cart_collision_cost")
        self.assertEqual(term["weight"], -1.0)       # 2026-10-19: 全域 −1.0
        self.assertEqual(term["params"], {})          # 不再传 downhill_scale
        self.assertAlmostEqual(term["weight"] * STEP_DT, -0.05, places=12)

    def test_single_collision_curve(self):
        """不得拆成第二条 RewTerm：`Episode_Reward/collision` 必须仍是一条曲线。"""
        names = _reward_term_names()
        self.assertEqual([n for n in names if "collision" in n], ["collision"])
        self.assertNotIn("RewTerm(func=mdp.cart_collision_cost, weight=-5.0", CFG_SOURCE)

    def test_downhill_section_uses_the_profile_constants(self):
        body, _node = _function_source(MDP_SOURCE, "downhill_segment")
        self.assertIn("DOWN_START_M", body)
        self.assertIn("FLAT_OUT_START_M", body)
        self.assertNotIn("terrain_profile", body)       # rolling 档的已知近似（docstring 已注明）
        self.assertIn("terrain_speedbump", body)        # 减速带列排除
        self.assertIn("terrain_wave", body)             # 波浪列排除
        # 常量确实来自 slope_geometry（不是本地硬编码）
        self.assertIn("DOWN_START_M, FLAT_OUT_START_M", MDP_SOURCE)

    @_NEEDS_TORCH
    def test_mask_equals_descending_profile(self):
        """整条剖面上 mask 必须与 `profile_slope_degrees < 0` 逐点一致（10° 档）。

        用 float32 张量里的**实际**取值做参考比较（`x.tolist()` 回读），否则 Python 侧
        累加出来的 9.000000000000002 与 float32 的 9.0 会在边界上假失败。
        """
        grid = torch.as_tensor([-1.0 + 0.01 * i for i in range(1301)], dtype=torch.float32)
        xs = grid.tolist()
        mask = _downhill_mask(xs)
        for x, flag in zip(xs, mask.tolist()):
            with self.subTest(x=x):
                self.assertEqual(flag, slope_geometry.profile_slope_degrees(10, x) < 0)

    @_NEEDS_TORCH
    def test_boundaries(self):
        mask = _downhill_mask([0.0, 2.25, 5.25, 5.99, 6.0, 6.001, 9.0, 9.001, 12.0])
        self.assertEqual(mask.tolist(),
                         [False, False, False, False, False, True, True, False, False])

    @_NEEDS_TORCH
    def test_non_smooth_columns_are_never_downhill(self):
        xs = [7.5, 7.5, 7.5]
        self.assertEqual(_downhill_mask(xs, speedbump=True).tolist(), [False] * 3)
        self.assertEqual(_downhill_mask(xs, wave=True).tolist(), [False] * 3)
        self.assertEqual(_downhill_mask(xs).tolist(), [True] * 3)

    @_NEEDS_TORCH
    def test_cost_scaling(self):
        xs = [0.0, 5.0, 5.99, 6.001, 7.5, 9.0, 9.001]
        collision = [1] * len(xs)
        cost = _cost(xs, collision, downhill_scale=0.2)
        # float32 的 0.2 是 0.20000000298… ⇒ 逐项近似比较，别用位串相等
        for got, want in zip(cost.tolist(), [1.0, 1.0, 1.0, 0.2, 0.2, 0.2, 1.0]):
            self.assertAlmostEqual(got, want, places=6)

    @_NEEDS_TORCH
    def test_no_collision_is_exactly_zero(self):
        xs = [5.0, 7.5, 9.5]
        self.assertEqual(_cost(xs, [0, 0, 0], downhill_scale=0.2).tolist(), [0.0, 0.0, 0.0])

    @_NEEDS_TORCH
    def test_default_scale_is_bit_identical_to_old_behaviour(self):
        """默认 `downhill_scale=1.0`（不传）⇒ 与旧实现逐位一致，且不读下坡相关字段。"""
        xs = [5.0, 7.5]
        legacy = _cost(xs, [1, 0])                      # 不传参数
        self.assertEqual(legacy.tolist(), [1.0, 0.0])
        self.assertEqual(legacy.dtype, torch.float32)

    def test_signature_keeps_default(self):
        _body, node = _function_source(MDP_SOURCE, "cart_collision_cost")
        defaults = node.args.args[-len(node.args.defaults):]
        self.assertEqual([a.arg for a in defaults], ["downhill_scale"])
        self.assertEqual(ast.literal_eval(node.args.defaults[0]), 1.0)


if __name__ == "__main__":
    unittest.main()
