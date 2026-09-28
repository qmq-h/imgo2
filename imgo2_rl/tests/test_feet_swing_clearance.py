"""`feet_swing_clearance`（相对支撑面的抬脚高度奖励，2026-09-28 新增）的离线回归。

分三层：

1. **纯函数**（`mdp/clearance_math.py`，只依赖 torch，按文件路径加载）：
   满分点/带外为零/只对非接触足计分/`band=0` 边界/**压身体不变性**；
2. **`stance_reference` 的参考面语义**：全接触 ⇒ 均值；无接触 ⇒ 沿用缓冲；超过
   `max_hold_steps` ⇒ 无效（`valid=False` 且参考为 NaN ⇒ 奖励为 0）；
3. **接线（源码级）**：CMoE 链里权重 > 0、两个旧高度项仍为 0、逐地形参数名真实存在于
   `sub_terrains`（写错名字必须让测试失败 —— 用生产代码里的 `clearance_terrain_params()`
   做负向对照）、`-gaitfree` 不把它归零、诊断项与课程聚合标签都接上；

Run: /opt/conda/envs/isaaclab/bin/python -m unittest discover -s tests -p test_feet_swing_clearance.py -v
"""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MDP = REPO / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/mdp"
CLEARANCE_MATH = MDP / "clearance_math.py"
REWARDS = MDP / "rewards.py"
CMOE_CFG = REPO / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/base_move/CMoE_env_cfg.py"
VELOCITY_CFG = REPO / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py"
TOOLS = REPO / "scripts" / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

try:
    import torch
except ModuleNotFoundError as error:  # pragma: no cover - 本机装了 torch
    torch = None
    IMPORT_ERROR: object = error
else:
    IMPORT_ERROR = None


def _load_clearance_math():
    """按**文件路径**加载纯数学模块（`mdp/__init__.py` 会拉 isaaclab，不能走包导入）。"""
    spec = importlib.util.spec_from_file_location("imgo2_clearance_math", CLEARANCE_MATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cfg_source() -> str:
    return CMOE_CFG.read_text(encoding="utf-8-sig")


def _module_constants(path: Path) -> dict[str, object]:
    """取模块级 `NAME = <字面量>` 常量（如 `SWING_CLEARANCE_TERRAIN_GROUPS`）。"""
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    out: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    out[target.id] = ast.literal_eval(node.value)
                except Exception:
                    continue
    return out


def _load_production_validator():
    """AST 抽出 `clearance_terrain_params()` 的**真实源码**并 exec（不 import isaaclab）。

    返回值 `(fn, namespace)`：namespace 里带着真实的地形分组表，可以按需覆盖成"打错名字"的
    版本做负向对照 —— 被测的就是**生产代码里的那段校验**，不是测试里另写一份。
    """
    source = _cfg_source()
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "clearance_terrain_params")
    namespace = {"SWING_CLEARANCE_TERRAIN_GROUPS": _module_constants(CMOE_CFG)["SWING_CLEARANCE_TERRAIN_GROUPS"]}
    exec(compile(ast.get_source_segment(source, node), str(CMOE_CFG), "exec"), namespace)  # noqa: S102
    return namespace["clearance_terrain_params"], namespace


def _terrain_keys() -> tuple[str, ...]:
    """CMoE 的 `sub_terrains` 键（基类顺序 + 任务侧覆盖/新增），复用现成的离线工具。"""
    import check_terrain_columns as columns  # noqa: PLC0415

    props, _cols = columns.cmoe_overrides()
    base = [name for name, _p in columns.base_sub_terrains()]
    return tuple(dict.fromkeys([*base, *props.keys()]))


def _params_keys_from_rewards_cfg(path: Path, term: str) -> set[str]:
    """读 `velocity_env_cfg.py::RewardsCfg` 里 `<term> = RewTerm(..., params={...})` 的键。"""
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "RewardsCfg")
    for node in cls.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        if not (isinstance(node.targets[0], ast.Name) and node.targets[0].id == term):
            continue
        for kw in node.value.keywords:
            if kw.arg == "params" and isinstance(kw.value, ast.Dict):
                return {key.value for key in kw.value.keys if isinstance(key, ast.Constant)}
    raise AssertionError(f"velocity_env_cfg.py::RewardsCfg 里找不到 {term} 的 params")


def _signature(path: Path, name: str, class_name: str | None = None) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    if class_name is None:
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    else:
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
        node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return [arg.arg for arg in node.args.args if arg.arg not in ("self", "env")]


@unittest.skipIf(torch is None, f"PyTorch unavailable: {IMPORT_ERROR}")
class TestSwingClearanceMath(unittest.TestCase):
    """`swing_clearance_reward` 的逐条语义（用户定稿的公式）。"""

    @classmethod
    def setUpClass(cls):
        cls.math = _load_clearance_math()

    def _reward(self, foot_z, contact, target, band, reference, **kwargs):
        return self.math.swing_clearance_reward(
            torch.tensor(foot_z, dtype=torch.float32),
            torch.tensor(contact, dtype=torch.bool),
            target=target,
            band=band,
            reference=torch.tensor(reference, dtype=torch.float32),
            **kwargs,
        )

    def test_full_score_at_target(self):
        """摆动足正好在 `h*` 上 ⇒ 该足满分（接触足不计分）。"""
        out = self._reward([[0.0, 0.07, 0.0, 0.0]], [[True, False, True, True]], 0.07, 0.05, [0.0])
        self.assertAlmostEqual(float(out[0]), 1.0, places=6)

    def test_two_swing_feet_sum_and_scale_linearly(self):
        """两只摆动足各差半个 band ⇒ 各 0.5 ⇒ 合计 1.0（本项是**求和**，不是求平均）。"""
        out = self._reward([[0.035, 0.035, 0.0, 0.0]], [[False, False, True, True]], 0.0, 0.07, [0.0])
        self.assertAlmostEqual(float(out[0]), 1.0, places=6)

    def test_zero_at_and_beyond_the_band(self):
        """`|h − h*| ≥ band ⇒ 0`（带外**不加分**，也就不会靠高抬腿刷分）。"""
        out = self._reward(
            [[0.0, 0.0, 0.0], [0.0, 0.05, 0.0], [0.0, 0.10, 0.0]],
            [[True, False, True]] * 3,
            0.0, 0.05, [0.0, 0.0, 0.0],
        )
        self.assertAlmostEqual(float(out[0]), 1.0, places=6, msg="h=0=h* ⇒ 满分")
        self.assertAlmostEqual(float(out[1]), 0.0, places=6, msg="|h−h*|=band ⇒ 0")
        self.assertAlmostEqual(float(out[2]), 0.0, places=6, msg="带外 ⇒ 0")

    def test_only_swing_feet_are_scored(self):
        """只对**非接触**足计分：四只脚都在满分高度、只有 1 只接触 ⇒ 只算另外 3 只。"""
        full = [[0.07, 0.07, 0.07, 0.07]]
        out = self._reward(full, [[True, False, False, False]], 0.07, 0.05, [0.0])
        self.assertAlmostEqual(float(out[0]), 3.0, places=6, msg="接触的那只不许计分")
        out_all_swing = self._reward(full, [[False, False, False, False]], 0.07, 0.05, [0.0])
        self.assertAlmostEqual(float(out_all_swing[0]), 4.0, places=6)

    def test_all_feet_in_contact_is_zero(self):
        out = self._reward([[0.0, 0.0, 0.0, 0.0]], [[True] * 4], 0.07, 0.05, [0.0])
        self.assertAlmostEqual(float(out[0]), 0.0, places=6)

    def test_band_zero_is_exact_match_indicator(self):
        """`band = 0` 边界：`h == h*` ⇒ 1，否则 0（不是 NaN、也不是满分）。"""
        hit = self._reward([[0.0, 0.07, 0.0, 0.0]], [[True, False, True, True]], 0.07, 0.0, [0.0])
        miss = self._reward([[0.0, 0.0700001, 0.0, 0.0]], [[True, False, True, True]], 0.07, 0.0, [0.0])
        self.assertAlmostEqual(float(hit[0]), 1.0, places=6)
        self.assertAlmostEqual(float(miss[0]), 0.0, places=6)

    def test_invalid_reference_gives_zero(self):
        """无参考面（NaN，例如长时间无接触/基座悬空）⇒ 输出 0，绝不按"错误参考"付钱。"""
        out = self._reward([[0.0, 0.07, 0.0, 0.0]], [[True, False, True, True]], 0.07, 0.05,
                           [float("nan")])
        self.assertAlmostEqual(float(out[0]), 0.0, places=6)
        out_inf = self._reward([[0.0, 0.07, 0.0, 0.0]], [[True, False, True, True]], 0.07, 0.05,
                               [float("inf")])
        self.assertAlmostEqual(float(out_inf[0]), 0.0, places=6)

    def test_body_press_invariance_is_exact(self):
        """**压身体不变性（严格版）**：足端世界 z 与 `z_ref` 同步 −0.05 ⇒ 输出**逐元素完全不变**。

        物理含义：旧 `feet_height_body` 用机体系 z，可以把基座压低 5 cm 换到"脚相对更高"的
        reward；这里 `h_i = z_foot − z_ref` 对"基座+足一起平移"是不变量 ⇒ 该捷径在数学上不存在。
        ⚠️ 取一组**差值精确可表示**的数值：`0.05` 不是二进制精确数，一般数值下
        `(z−d)−(z_ref−d)` 与 `z−z_ref` 会差 1 ulp（实测 ~1e-8）⇒ "逐元素完全不变"只能用这种
        数值钉（另有 `test_body_press_invariance_tolerance` 用随机值钉 1e-6 一致）。
        """
        foot_z = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, 0.07, 0.05, 0.14]])
        contact = torch.tensor([[True, False, True, False], [True, False, False, False]])
        reference = torch.tensor([0.0, 0.0])
        kwargs = dict(target=0.07, band=0.05)
        base = self.math.swing_clearance_reward(foot_z, contact, reference=reference, **kwargs)
        pressed = self.math.swing_clearance_reward(
            foot_z - 0.05, contact, reference=reference - 0.05, **kwargs
        )
        self.assertTrue(torch.equal(base, pressed), f"{base.tolist()} vs {pressed.tolist()}")
        self.assertGreater(float(base[1]), 0.0, "这组样例应当有非零分数，否则测试是空转")

    def test_body_press_invariance_tolerance(self):
        """随机（真实量级）足高的压身体不变性：浮点下 1e-6 级一致（1 ulp 舍入，见上）。"""
        torch.manual_seed(0)
        foot_z = torch.rand(16, 4) * 0.25 + 0.02
        contact = torch.rand(16, 4) > 0.4
        contact[:, 0] = True                     # 保证每行至少一只支撑足
        reference = torch.where(contact, foot_z, torch.zeros_like(foot_z)).sum(dim=1) / contact.sum(dim=1)
        kwargs = dict(target=0.07, band=0.05)
        base = self.math.swing_clearance_reward(foot_z, contact, reference=reference, **kwargs)
        pressed = self.math.swing_clearance_reward(foot_z - 0.05, contact, reference=reference - 0.05, **kwargs)
        self.assertTrue(torch.allclose(base, pressed, atol=1e-6, rtol=0.0),
                        f"最大差 {(base - pressed).abs().max().item():.3e}")

    def test_per_terrain_target_and_band(self):
        """`target`/`band` 支持逐环境 `[N]`（按地形给不同档），并且标量/`[N]` 等价。"""
        foot_z = torch.tensor([[0.0, 0.12, 0.0, 0.0], [0.0, 0.09, 0.0, 0.0]])
        contact = torch.tensor([[True, False, True, True]] * 2)
        reference = torch.zeros(2)
        out = self.math.swing_clearance_reward(
            foot_z, contact, target=torch.tensor([0.12, 0.09]), band=torch.tensor([0.10, 0.06]),
            reference=reference,
        )
        self.assertTrue(torch.allclose(out, torch.ones(2), atol=1e-6), out.tolist())

    def test_optional_velocity_gate_is_off_by_default(self):
        """可选速度门默认关（`tanh_mult=None` ⇒ 严格算式）；打开时慢速足拿不到分。"""
        foot_z = torch.tensor([[0.0, 0.07, 0.0, 0.0]])
        contact = torch.tensor([[True, False, True, True]])
        reference = torch.zeros(1)
        plain = self.math.swing_clearance_reward(foot_z, contact, target=0.07, band=0.05, reference=reference)
        gated = self.math.swing_clearance_reward(
            foot_z, contact, target=0.07, band=0.05, reference=reference,
            foot_lin_vel_xy=torch.zeros(1, 4, 2), tanh_mult=2.0,
        )
        self.assertAlmostEqual(float(plain[0]), 1.0, places=6)
        self.assertAlmostEqual(float(gated[0]), 0.0, places=6)

    def test_shape_errors_raise(self):
        with self.assertRaises(ValueError):
            self.math.swing_clearance_reward(torch.zeros(4), torch.zeros(4, dtype=torch.bool),
                                             target=0.0, band=1.0, reference=torch.zeros(4))
        with self.assertRaises(ValueError):
            self.math.swing_clearance_reward(torch.zeros(2, 4), torch.zeros(2, 3, dtype=torch.bool),
                                             target=0.0, band=1.0, reference=torch.zeros(2))
        with self.assertRaises(ValueError):
            self.math.swing_clearance_reward(torch.zeros(2, 4), torch.zeros(2, 4, dtype=torch.bool),
                                             target=0.0, band=1.0, reference=torch.zeros(3))


@unittest.skipIf(torch is None, f"PyTorch unavailable: {IMPORT_ERROR}")
class TestStanceReference(unittest.TestCase):
    """`stance_reference` 的参考面语义（含飞行相/沟壑的"无效"标记）。"""

    @classmethod
    def setUpClass(cls):
        cls.math = _load_clearance_math()

    def test_all_contact_equals_mean(self):
        foot_z = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.10, 0.12, 0.14, 0.16]])
        contact = torch.ones(2, 4, dtype=torch.bool)
        reference, valid = self.math.stance_reference(foot_z, contact, torch.full((2,), float("nan")), 3)
        self.assertTrue(torch.allclose(reference, torch.tensor([0.0, 0.13]), atol=1e-6), reference.tolist())
        self.assertTrue(bool(valid.all()))

    def test_partial_contact_uses_contacting_feet_only(self):
        """只有两只脚接触时，参考＝这两只脚的均值（抬起的那只**不能**算进去）。"""
        foot_z = torch.tensor([[0.0, 0.0, 0.20, 0.30]])
        contact = torch.tensor([[True, True, False, False]])
        reference, valid = self.math.stance_reference(foot_z, contact, torch.full((1,), float("nan")), 3)
        self.assertAlmostEqual(float(reference[0]), 0.0, places=6)
        self.assertTrue(bool(valid[0]))

    def test_no_contact_reuses_the_buffer(self):
        """无接触但在 `max_hold_steps` 之内 ⇒ 沿用最近一次有效参考（飞行相抖动保护）。"""
        reference, valid = self.math.stance_reference(
            torch.full((1, 4), 0.5), torch.zeros(1, 4, dtype=torch.bool),
            torch.tensor([0.13]), 3, torch.tensor([2]),
        )
        self.assertAlmostEqual(float(reference[0]), 0.13, places=6)
        self.assertTrue(bool(valid[0]))

    def test_no_contact_beyond_max_hold_is_invalid(self):
        """超过 `max_hold_steps` 仍无接触 ⇒ 无效（NaN + `valid=False` ⇒ 奖励为 0）。"""
        reference, valid = self.math.stance_reference(
            torch.full((1, 4), 0.5), torch.zeros(1, 4, dtype=torch.bool),
            torch.tensor([0.13]), 3, torch.tensor([3]),
        )
        self.assertTrue(bool(torch.isnan(reference[0])))
        self.assertFalse(bool(valid[0]))

    def test_no_history_is_invalid(self):
        reference, valid = self.math.stance_reference(
            torch.full((1, 4), 0.5), torch.zeros(1, 4, dtype=torch.bool),
            torch.full((1,), float("nan")), 3,
        )
        self.assertFalse(bool(valid[0]))
        self.assertTrue(bool(torch.isnan(reference[0])))

    def test_max_hold_zero_means_flight_is_always_invalid(self):
        """`max_hold_steps=0` ⇒ 只要没有脚接触就直接无效（不做缓冲）。"""
        reference, valid = self.math.stance_reference(
            torch.full((1, 4), 0.5), torch.zeros(1, 4, dtype=torch.bool), torch.tensor([0.13]), 0,
        )
        self.assertFalse(bool(valid[0]))

    def test_hold_does_not_apply_when_a_foot_touches(self):
        """有脚接触时以**新测量**为准，与缓冲/hold 计数无关。"""
        reference, valid = self.math.stance_reference(
            torch.tensor([[0.07, 0.07, 0.25, 0.25]]),
            torch.tensor([[True, True, False, False]]),
            torch.tensor([9.99]), 3, torch.tensor([0]),
        )
        self.assertAlmostEqual(float(reference[0]), 0.07, places=6)
        self.assertTrue(bool(valid[0]))


class TestCmoeClearanceWiring(unittest.TestCase):
    """源码级接线：CMoE 链里新项生效、两个旧高度项保持 0、逐地形参数名真实存在。"""

    @classmethod
    def setUpClass(cls):
        cls.src = _cfg_source()
        cls.constants = _module_constants(CMOE_CFG)
        cls.groups = cls.constants["SWING_CLEARANCE_TERRAIN_GROUPS"]

    # ---------------------------------------------------------------- 权重
    def test_enabled_with_user_weight(self):
        self.assertIn("self.rewards.feet_swing_clearance.weight = 0.5", self.src)
        weight = re.search(r"self\.rewards\.feet_swing_clearance\.weight = ([0-9.]+)", self.src)
        self.assertIsNotNone(weight)
        self.assertGreater(float(weight.group(1)), 0.0, "CMoE 链必须真的启用它（>0）")

    def test_legacy_foot_height_terms_stay_off(self):
        """用户强调：两个旧项**必须保持 0**（机体系可被"压身体"满足；固定世界高度在高差瓦片上误罚）。"""
        feet_height = re.search(r"self\.rewards\.feet_height\.weight = ([0-9.\-]+)", self.src)
        body = re.search(r"self\.rewards\.feet_height_body\.weight = ([0-9.\-]+)", self.src)
        self.assertIsNotNone(feet_height)
        self.assertIsNotNone(body)
        self.assertEqual(float(feet_height.group(1)), 0.0, "feet_height 必须保持 0")
        self.assertEqual(float(body.group(1)), 0.0, "feet_height_body 必须保持 0")

    def test_gaitfree_does_not_zero_it(self):
        """`-gaitfree` 子类**不能**把它归零（它正是用来补 gaitfree 缺失的抬脚压力）。"""
        block = re.search(r"class Imgo2CMoEGaitFreeEnvCfg\(.*", self.src, re.S)
        self.assertIsNotNone(block)
        body = block.group(0)
        self.assertNotIn("self.rewards.feet_swing_clearance.weight = 0.0", body)
        self.assertIn("feet_swing_clearance", body, "gaitfree 里应有一行注释说明它**不**归零")

    def test_contact_and_command_wiring(self):
        """接触判定用 contact 传感器；有命令门按 `feet_height` 的惯例（调用点在 `mdp/rewards.py`）。"""
        self.assertIn('self.rewards.feet_swing_clearance.params["sensor_cfg"].body_names = [self.foot_link_name]',
                      self.src)
        self.assertIn('self.rewards.feet_swing_clearance.params["asset_cfg"].body_names = [self.foot_link_name]',
                      self.src)
        self.assertIn('self.rewards.feet_swing_clearance.params["max_hold_steps"] = 3', self.src)
        self.assertIn('self.rewards.feet_swing_clearance.params["tanh_mult"] = None', self.src)
        rewards_src = REWARDS.read_text(encoding="utf-8")
        self.assertIn("net_forces_w_history", rewards_src, "接触判定必须走 ContactSensor 的力史（与 feet_slide 同口径）")
        self.assertIn("_foot_contacts", rewards_src)

    def test_default_reward_term_is_off_in_the_shared_config(self):
        """`velocity_env_cfg.py` 里默认权重 0（不改变任何现有任务）。"""
        src = VELOCITY_CFG.read_text(encoding="utf-8")
        block = re.search(r"feet_swing_clearance = RewTerm\((.*?)\n    \)", src, re.S)
        self.assertIsNotNone(block, "公共 RewardsCfg 里没有 feet_swing_clearance")
        self.assertIn("weight=0.0", block.group(1))
        self.assertIn("func=mdp.feet_swing_clearance", block.group(1))

    # ---------------------------------------------------------------- params ↔ 签名
    def test_param_names_match_the_function_signature(self):
        """`params` 的键必须逐字出现在函数签名里（Isaac Lab `manager_base` 的集合校验）。"""
        params = _params_keys_from_rewards_cfg(VELOCITY_CFG, "feet_swing_clearance")
        signature = set(_signature(REWARDS, "feet_swing_clearance"))
        self.assertEqual(params - signature, set(),
                         f"params 里有签名中没有的键（Isaac Lab 会直接报错）：{sorted(params - signature)}")
        # 无默认值的参数必须由 params 给全
        tree = ast.parse(REWARDS.read_text(encoding="utf-8"))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "feet_swing_clearance")
        names = [arg.arg for arg in node.args.args]
        mandatory = names[1:len(names) - len(node.args.defaults)]       # 去掉 env，再去掉带默认值的尾部
        self.assertEqual(set(mandatory) - params, set(), f"缺必填参数：{sorted(set(mandatory) - params)}")
        # 类版本的 `__call__` 必须能接住同一张 params 表
        call_signature = set(_signature(REWARDS, "__call__", class_name="FeetSwingClearance"))
        self.assertEqual(params - call_signature, set(),
                         f"FeetSwingClearance.__call__ 接不住这些参数：{sorted(params - call_signature)}")

    # ---------------------------------------------------------------- 逐地形参数
    def test_terrain_group_names_all_exist(self):
        """表里的地形名必须真实存在于 `sub_terrains`（写错 ⇒ 这一项在那一列上静默用默认档）。"""
        keys = _terrain_keys()
        for names, _t, _b in self.groups:
            for name in names:
                self.assertIn(name, keys, f"{name} 不在 sub_terrains 里（打字错？）")

    def test_terrain_groups_cover_everything_exactly_once(self):
        keys = set(_terrain_keys())
        flat = [name for names, _t, _b in self.groups for name in names]
        self.assertEqual(len(flat), len(set(flat)), f"有重复归类：{flat}")
        self.assertEqual(set(flat), keys, f"分组与 sub_terrains 不一致：{sorted(set(flat) ^ keys)}")

    def test_per_terrain_targets_and_bands(self):
        """逐档数值就是用户定稿的那张表（平地/粗糙/斜坡 0.07/0.05、台阶 0.09/0.06、障碍 0.12/0.10）。"""
        table = {name: (t, b) for names, t, b in self.groups for name in names}
        expected = {
            "flat": (0.07, 0.05), "random_rough": (0.07, 0.05),
            "hf_pyramid_slope": (0.07, 0.05), "hf_pyramid_slope_inv": (0.07, 0.05),
            "pyramid_stairs": (0.09, 0.06), "pyramid_stairs_inv": (0.09, 0.06),
            "narrow_stairs": (0.09, 0.06),
            "boxes": (0.12, 0.10), "gap": (0.12, 0.10),
            "hurdle": (0.12, 0.10), "mix": (0.12, 0.10),
        }
        self.assertEqual(table, expected)
        for name, (_t, band) in table.items():
            self.assertGreater(band, 0.0, f"{name} 的 band 必须为正（band=0 只在边界测试里用）")

    def test_production_validator_accepts_real_keys_and_rejects_typos(self):
        """用**生产代码里的校验函数**做正/负向对照：打错名字必须 raise（这条要能"吵"）。"""
        validate, namespace = _load_production_validator()
        keys = _terrain_keys()
        target, band = validate(keys)
        self.assertEqual(set(target), set(keys))
        self.assertEqual(set(band), set(keys))
        # ① 表里把 flat 写成 flatt ⇒ 少 `flat`、多 `flatt` ⇒ 必须报错
        namespace["SWING_CLEARANCE_TERRAIN_GROUPS"] = tuple(
            (tuple("flatt" if name == "flat" else name for name in names), t, b)
            for names, t, b in self.groups
        )
        with self.assertRaises(RuntimeError):
            validate(keys)
        # ② 新增一类地形但忘了归类 ⇒ 必须报错
        namespace["SWING_CLEARANCE_TERRAIN_GROUPS"] = self.groups
        with self.assertRaises(RuntimeError):
            validate((*keys, "brand_new_terrain"))

    def test_config_calls_the_validator(self):
        self.assertIn("clearance_terrain_params(", self.src)
        self.assertIn('self.rewards.feet_swing_clearance.params["target_height_by_terrain"] = _clearance_target',
                      self.src)
        self.assertIn('self.rewards.feet_swing_clearance.params["band_by_terrain"] = _clearance_band', self.src)

    # ---------------------------------------------------------------- 诊断项
    def test_diagnostics_are_wired_to_terrain_aggregation(self):
        """两条 1e-6 诊断项（`clearance_mean`、`base_height`）必须同时出现在奖励表与逐列聚合名单里。"""
        for term in ("diag_clearance_mean", "diag_base_height"):
            block = re.search(rf"{term} = RewTerm\((.*?)\n    \)", self.src, re.S)
            self.assertIsNotNone(block, f"{term} 没有定义")
            self.assertIn("weight=1e-6", block.group(1), f"{term} 必须是 1e-6 诊断权重（只写 TB）")
        self.assertIn('("clearance", "diag_clearance_mean")', self.src)
        self.assertIn('("base_height", "diag_base_height")', self.src)

    def test_no_new_terrain_baseline_hardcoding(self):
        """不要写死旧列数（11/20/40）—— 逐地形参数只按**名字**给，列数由比例现算。"""
        block = re.search(r"SWING_CLEARANCE_TERRAIN_GROUPS = \((.*?)\n\)", self.src, re.S)
        self.assertIsNotNone(block)
        self.assertNotIn("num_cols", block.group(1))


# ---------------------------------------------------------------------------------------------
# `FeetSwingClearance` 本体（有状态那部分）：用**桩 env** 跑真实类源码
# ---------------------------------------------------------------------------------------------

class _TermBaseStub:
    """`ManagerTermBase` 的桩（真类在 isaaclab 里，本机 import 不了）。"""

    def __init__(self, cfg, env):
        self._cfg = cfg
        self._env = env


class _CfgShim:
    def __init__(self, params):
        self.params = params


class _Data:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _Entity:
    def __init__(self, **data):
        self.data = _Data(**data)


class _SceneStub:
    def __init__(self, robot, sensors):
        self._robot = robot
        self.sensors = sensors

    def __getitem__(self, key):
        if key == "robot":
            return self._robot
        if key == "height_scanner_base":
            return self.sensors["height_scanner_base"]
        raise KeyError(key)


class _CommandManagerStub:
    def __init__(self, command):
        self._command = command

    def get_command(self, _name):
        return self._command


class _EnvStub:
    def __init__(self, terrain_names, foot_z, contacts, *, command=1.0, rays_valid=None,
                 gravity=None, episode_length=None):
        n, f = foot_z.shape
        self.num_envs = n
        self.device = "cpu"
        self.terrain_names = list(terrain_names)
        self.contacts = contacts                    # `_foot_contacts` 桩直接读它
        self.rays_valid = (torch.ones(n, dtype=torch.bool) if rays_valid is None else rays_valid)
        # 默认给一个"回合中途"的值：`<= 1` 才会被当成刚重置（见 `_clear_respawned` 的注释）
        self.episode_length_buf = (torch.full((n,), 100, dtype=torch.long) if episode_length is None
                                   else episode_length)
        zero = torch.zeros(n, f)
        robot = _Entity(
            body_pos_w=torch.stack([zero, zero, foot_z], dim=-1),
            body_lin_vel_w=torch.zeros(n, f, 3),
            projected_gravity_b=(torch.tensor([[0.0, 0.0, -1.0]] * n) if gravity is None else gravity),
        )
        scanner = _Entity(ray_hits_w=torch.zeros(n, 9, 3))
        self.scene = _SceneStub(robot, {"contact_forces": _Entity(net_forces_w=torch.zeros(1)), 
                                        "height_scanner_base": scanner})
        self.command_manager = _CommandManagerStub(
            torch.tensor([[command, 0.0, 0.0]] * n) if not torch.is_tensor(command) else command
        )


def _load_term_class():
    """AST 抽出 `FeetSwingClearance` 的真实源码，在桩命名空间里 exec。"""
    source = REWARDS.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FeetSwingClearance")
    namespace = {
        "torch": torch,
        "ManagerTermBase": _TermBaseStub,
        "ManagerBasedRLEnv": object,
        "RewTerm": object,
        "SceneEntityCfg": object,
        "RigidObject": object,
        "SWING_CLEARANCE_DEFAULT_TARGET": 0.07,
        "SWING_CLEARANCE_DEFAULT_BAND": 0.05,
        "stance_reference": _load_clearance_math().stance_reference,
        "swing_clearance_reward": _load_clearance_math().swing_clearance_reward,
        # 桩：地形掩码按 env.terrain_names 现算；接触/射线有效性直接读 env 上的张量
        "_terrain_type_mask": lambda env, names: torch.tensor(
            [name in names for name in env.terrain_names], dtype=torch.bool
        ),
        "_foot_contacts": lambda env, sensor_cfg, threshold=1.0: env.contacts,
        "_base_rays_valid": lambda env, sensor_name="height_scanner_base": env.rays_valid,
    }
    exec(compile(ast.get_source_segment(source, node), str(REWARDS), "exec"), namespace)  # noqa: S102
    return namespace["FeetSwingClearance"]


@unittest.skipIf(torch is None, f"PyTorch unavailable: {IMPORT_ERROR}")
class TestFeetSwingClearanceTerm(unittest.TestCase):
    """有状态部分：逐地形 `h*`/`band`、飞行相缓冲、沟壑置零、命令门、直立门、回合重置。"""

    @classmethod
    def setUpClass(cls):
        cls.cls = _load_term_class()
        cls.asset_cfg = type("Cfg", (), {"name": "robot", "body_ids": [0, 1, 2, 3]})()
        cls.sensor_cfg = type("Cfg", (), {"name": "contact_forces"})()
        cls.targets = {"flat": 0.07, "pyramid_stairs": 0.09}
        cls.bands = {"flat": 0.05, "pyramid_stairs": 0.06}

    def _term(self, env):
        return self.cls(_CfgShim({}), env)

    def _call(self, term, env, *, free=(), k=1.0, max_hold_steps=3):
        return term(
            env, "base_velocity", self.asset_cfg, self.sensor_cfg,
            self.targets, self.bands, free, k, None, max_hold_steps,
        )

    def test_per_terrain_target_is_resolved(self):
        """两只摆动足分别落在**各自地形**的 `h*` 上：都要满分（逐环境参数真的生效）。"""
        env = _EnvStub(
            ["flat", "pyramid_stairs"],
            torch.tensor([[0.0, 0.07, 0.0, 0.0], [0.0, 0.09, 0.0, 0.0]]),
            torch.tensor([[True, False, True, True]] * 2),
        )
        out = self._call(self._term(env), env)
        self.assertTrue(torch.allclose(out, torch.ones(2), atol=1e-6), out.tolist())

    def test_default_band_applies_when_resolution_differs(self):
        """`band` 也逐地形：`flat` 用 0.05（0.07±0.025 ⇒ 半带），`stairs` 用 0.06（0.09±0.03）。"""
        env = _EnvStub(
            ["flat", "pyramid_stairs"],
            torch.tensor([[0.0, 0.07 + 0.025, 0.0, 0.0], [0.0, 0.09 + 0.03, 0.0, 0.0]]),
            torch.tensor([[True, False, True, True]] * 2),
        )
        out = self._call(self._term(env), env)
        self.assertTrue(torch.allclose(out, torch.full((2,), 0.5), atol=1e-5), out.tolist())

    def test_free_terrain_names_mask_the_term(self):
        """`free_terrain_names` 里的列整列为 0（与 `Masked*` 家族同一语义）。"""
        env = _EnvStub(
            ["flat", "pyramid_stairs"],
            torch.tensor([[0.0, 0.07, 0.0, 0.0], [0.0, 0.09, 0.0, 0.0]]),
            torch.tensor([[True, False, True, True]] * 2),
        )
        term = self._term(env)
        self.assertTrue(torch.allclose(self._call(term, env, free=("flat",)), torch.tensor([0.0, 1.0]), atol=1e-6))

    def test_flight_phase_holds_then_zeroes(self):
        """飞行相（没有任何脚接触）：沿用最近有效 `z_ref` 最多 `max_hold_steps` 步，之后为 0。"""
        foot = torch.tensor([[0.0, 0.07, 0.0, 0.0]])
        contact = torch.tensor([[True, False, True, True]])
        env = _EnvStub(["flat"], foot, contact)
        term = self._term(env)
        self.assertAlmostEqual(float(self._call(term, env)[0]), 1.0, places=6)     # 有接触 ⇒ 满分
        env.contacts = torch.zeros(1, 4, dtype=torch.bool)                          # 起飞：全部离地
        for step in range(3):                                                       # 1..3 步内沿用参考面
            self.assertAlmostEqual(float(self._call(term, env)[0]), 1.0, places=6, msg=f"第 {step + 1} 步应仍有效")
        self.assertAlmostEqual(float(self._call(term, env)[0]), 0.0, places=6, msg="超过 3 步必须置 0")

    def test_void_under_base_zeroes_the_term(self):
        """沟壑/悬空（基座下方射线全落空）⇒ 没有任何支撑面 ⇒ 该项为 0。"""
        env = _EnvStub(
            ["flat"], torch.tensor([[0.0, 0.07, 0.0, 0.0]]),
            torch.tensor([[True, False, True, True]]), rays_valid=torch.tensor([False]),
        )
        self.assertAlmostEqual(float(self._call(self._term(env), env)[0]), 0.0, places=6)

    def test_zero_command_and_upright_gates(self):
        """`|cmd| = 0` ⇒ 0（与 `feet_height` 同惯例）；翻倒（重力门 ≈ 0）⇒ 0。"""
        foot = torch.tensor([[0.0, 0.07, 0.0, 0.0]])
        contact = torch.tensor([[True, False, True, True]])
        standing = _EnvStub(["flat"], foot, contact)
        self.assertGreater(float(self._call(self._term(standing), standing)[0]), 0.0)
        idle = _EnvStub(["flat"], foot, contact, command=0.0)
        self.assertAlmostEqual(float(self._call(self._term(idle), idle)[0]), 0.0, places=6)
        fallen = _EnvStub(["flat"], foot, contact, gravity=torch.tensor([[0.0, 0.0, 1.0]]))
        self.assertAlmostEqual(float(self._call(self._term(fallen), fallen)[0]), 0.0, places=6)

    def test_episode_reset_clears_the_reference_buffer(self):
        """回合重置后不许沿用上一回合的参考面：`episode_length_buf <= 1` ＝本回合第一步。

        ⚠️ 这里用的是 **1** 而不是 0：Isaac Lab 在**算奖励之前**就 `episode_length_buf += 1`
        （`manager_based_rl_env.py:201`），清 0 的 `_reset_idx` 在算完奖励之后（:221/:396）。
        用 `== 0` 会一次也命中不了 —— 本测试就是钉这一点的。
        """
        env = _EnvStub(
            ["flat"], torch.tensor([[0.0, 0.07, 0.0, 0.0]]), torch.tensor([[True, False, True, True]]),
        )
        term = self._term(env)
        self._call(term, env)                                    # 先建立一次有效参考面
        env.episode_length_buf = torch.ones(1, dtype=torch.long)  # 新回合第一步（算奖励时是 1）
        env.contacts = torch.zeros(1, 4, dtype=torch.bool)        # 新回合第一步就全部离地
        self.assertAlmostEqual(float(self._call(term, env)[0]), 0.0, places=6,
                               msg="重置后应清掉缓冲（否则会拿上一回合的参考面付钱）")

    def test_reference_buffer_survives_a_mid_episode_step(self):
        """同一回合中途（`episode_length_buf >= 2`）**不能**清缓冲 —— 否则飞行相保护形同虚设。"""
        env = _EnvStub(
            ["flat"], torch.tensor([[0.0, 0.07, 0.0, 0.0]]), torch.tensor([[True, False, True, True]]),
            episode_length=torch.tensor([2], dtype=torch.long),
        )
        term = self._term(env)
        self._call(term, env)                                    # 建立参考面
        env.episode_length_buf = torch.tensor([3], dtype=torch.long)
        env.contacts = torch.zeros(1, 4, dtype=torch.bool)        # 全部离地（模拟起跳）
        self.assertAlmostEqual(float(self._call(term, env)[0]), 1.0, places=6,
                               msg="同回合内应沿用参考面（≤ max_hold_steps）")

    def test_k_scales_the_term(self):
        env = _EnvStub(
            ["flat"], torch.tensor([[0.0, 0.07, 0.0, 0.0]]), torch.tensor([[True, False, True, True]]),
        )
        out = self._call(self._term(env), env, k=2.0)
        self.assertAlmostEqual(float(out[0]), 2.0, places=6)


if __name__ == "__main__":
    unittest.main()
