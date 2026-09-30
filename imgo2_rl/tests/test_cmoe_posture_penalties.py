"""2026-09-30 「平地蹲 7 cm + 膝盖蹭地 + 抬脚过高」姿态退化修复的离线回归。

用户实测证据（run `cmoe_v5_7_lv12cap` @8500，`Episode_Reward/*` 是**每步加权值**：
真实每步贡献 = 该值 × dt 0.02；对照 @4000 是同一 run 退化前）：

===========================  ============  ============  ==========
分项                          @4000         @8500         退化倍数
===========================  ============  ============  ==========
`track_world_vel_xy_exp`     ——            **+3.71**     （任务主力）
`track_ang_vel_z_exp`        ——            **+1.37**     （任务主力）
`base_height_l2`             −0.0030       **−0.0484**   **14×**
`undesired_contacts`         −0.0008       **−0.1019**   **127×**
`lin_vel_z_l2`               ——            **−0.0585**   （vz RMS ≈ 0.17 m/s）
===========================  ============  ============  ==========

`√(0.0484/10) = **0.070 m**` ⇒ 平地上蹲 7 cm，而代价只占任务项（+3.71/+1.37）的 ≈1%。

四项改动（权重按用户给定，未自行调整）：

① `mdp.diag_base_height`（1e-6 诊断）⇒ 逐列 `gait_base_height_<地形>`，**有符号**；
② `base_height_flat_l2`（`mdp.MaskedBaseHeightL2Strict`，−35，**只在 flat**、去重力门）；
③ `lin_vel_z_l2` −2.0 → **−4.0**（掩码仍 `("boxes", "gap")`）；
④ `undesired_contacts` −0.5 → **−5.0**、`contact_forces` −0.02 → **−0.1**、
   新增 DoneTerm `illegal_contact_body`（非足端、**50 N**、一行可关）。

2026-10-01 追加第五项（本文件下半部分的 `TestKneeHeightSoftFloor` 等）：

⑤ `knee_height_flat`（`mdp.MaskedKneeHeightFlat`，**−20**，**只在 flat**）：**膝关节高度软地板**
   `Σ_{4 条小腿} relu(knee_min − z_SHANK)`，阈值 `knee_min = 0.111 m` 由**离线 FK** 标定；
   另有 1e-6 探针 `mdp.diag_knee_height_min` ⇒ 逐列 `gait_knee_height_<地形>`（4 条腿里**最低**的 z）。
   证据（run `cmoe_v5_8b_posture` @4527）：`gait_base_height_flat = **−0.085 m**`（平地上基座仍低 8.5 cm），
   而 flat-only 的 −35 高度项**没压住**（蹲姿代价只占任务 ~5%）；用户观察"膝只是**低**、不是压在地上"
   ⇒ 既有的接触口径（`undesired_contacts`／已关闭的 50 N 终止）抓不到，必须用高度口径。

本文件与 `test_masked_terrain_terms.py` 同一套路：`mdp/rewards.py` 顶层 import isaaclab
（缺 `omni.log`）无法整模块 import，所以用 AST 抽出**真实源码**、在桩命名空间里 exec；
纯函数部分则用与 `test_base_height_per_env.py` 同构的桩 env（只有 `scene[name].data`）。
阈值标定复用仓库既有 FK（`scripts/tools/audit_amp_dataset.py`，stdlib only），不引入 isaaclab。
"""

from __future__ import annotations

import ast
import re
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
REWARDS = ROOT / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/mdp/rewards.py"
CMOE_CFG = ROOT / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/base_move/CMoE_env_cfg.py"
ROUGH_CFG = ROOT / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/base_move/rough_env_cfg.py"
URDF = REPO / "imgo2_description/urdf/imgo2.urdf"
TOOLS = ROOT / "scripts" / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import audit_amp_dataset as amp_audit  # noqa: E402  （仓库既有 FK，stdlib only）
import check_reward_overrides as chk  # noqa: E402
import check_terrain_columns as columns  # noqa: E402

try:
    import torch
except ModuleNotFoundError as error:  # pragma: no cover - 本机 isaaclab 解释器有 torch
    torch = None
    IMPORT_ERROR: object = error
else:
    IMPORT_ERROR = None


# --------------------------------------------------------------------------- 桩与源码装载
class _Data:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class _Entity:
    def __init__(self, name, **data):
        self.name = name
        self.data = _Data(**data)


class _Scene:
    def __init__(self, entities):
        self._entities = entities

    def __getitem__(self, key):
        return self._entities[key]


class _Env:
    def __init__(self, terrain_names, entities):
        self.num_envs = len(terrain_names)
        self.device = "cpu"
        self.terrain_names = list(terrain_names)
        self.scene = _Scene(entities)


class _BaseStub:
    """`ManagerTermBase`／`GaitReward` 的桩：`__init__` 只记 cfg/env。"""

    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env


class _StubSceneEntityCfg:
    """`SceneEntityCfg` 的桩（真实类在 isaaclab 里）。"""

    def __init__(self, name="robot", **kwargs):
        self.name = name
        for key, value in kwargs.items():
            setattr(self, key, value)


def _terrain_mask(env, terrain_names):
    return torch.tensor([name in terrain_names for name in env.terrain_names], dtype=torch.bool)


def _load_defs(names, extra_ns=None):
    """AST 抽出 `rewards.py` 的顶层函数/类真实源码，在桩命名空间里 exec，返回 {名字: 对象}。"""
    source = REWARDS.read_text(encoding="utf-8")
    tree = ast.parse(source)
    found = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            found[node.name] = ast.get_source_segment(source, node)
    missing = set(names) - set(found)
    if missing:
        raise AssertionError(f"rewards.py 里找不到这些顶层定义：{sorted(missing)}")
    body = "from __future__ import annotations\n" + "\n\n".join(found[name] for name in names)
    ns = {
        "torch": torch,
        "ManagerTermBase": _BaseStub,
        "ManagerBasedRLEnv": object,
        "RewTerm": object,
        "SceneEntityCfg": _StubSceneEntityCfg,
        "Articulation": object,
        "RigidObject": object,
        "RayCaster": object,
        "ContactSensor": object,
        "_terrain_type_mask": _terrain_mask,
    }
    ns.update(extra_ns or {})
    exec(compile(body, str(REWARDS), "exec"), ns)  # noqa: S102
    return {name: ns[name] for name in names}


def _make_env(terrain_names, root_z, ray_hits, gravity_z=-1.0):
    """root_z: (N,) / ray_hits: (N,R)。`root_pos_w` 与 `root_link_pos_w` 取同一 z（真机里都是 base）。"""
    num_envs = len(root_z)
    z = torch.as_tensor(root_z, dtype=torch.float32).reshape(num_envs)
    zeros = torch.zeros(num_envs)
    robot = _Entity(
        "robot",
        root_pos_w=torch.stack([zeros, zeros, z], dim=1),
        root_link_pos_w=torch.stack([zeros, zeros, z], dim=1),
        projected_gravity_b=torch.tensor([[0.0, 0.0, gravity_z]] * num_envs),
    )
    scanner = _Entity(
        "height_scanner_base",
        ray_hits_w=torch.stack(
            [torch.zeros_like(ray_hits), torch.zeros_like(ray_hits), ray_hits], dim=-1
        ),
    )
    return _Env(terrain_names, {"robot": robot, "height_scanner_base": scanner})


def _cfg(params):
    return type("Cfg", (), {"params": params})()


def _make_knee_env(terrain_names, body_z, body_ids=(0, 1, 2, 3)):
    """`body_z`: (N, B) 世界系 z（B = 全部 body）。掩码/膝高测试只用到 `body_pos_w`。"""
    num_envs = len(terrain_names)
    body_z = torch.as_tensor(body_z, dtype=torch.float32).reshape(num_envs, -1)
    zeros = torch.zeros_like(body_z)
    robot = _Entity("robot", body_pos_w=torch.stack([zeros, zeros, body_z], dim=-1))
    env = _Env(terrain_names, {"robot": robot})
    asset_cfg = _StubSceneEntityCfg("robot", body_names=".*_SHANK", body_ids=list(body_ids))
    return env, asset_cfg


def _knee_cfg(active=("flat",), body_ids=(0, 1, 2, 3), min_height=0.111):
    return _cfg({
        "min_height": min_height,
        "asset_cfg": _StubSceneEntityCfg("robot", body_names=".*_SHANK", body_ids=list(body_ids)),
        "active_terrain_names": active,
    })


def _reward_term_param_literal(term: str, key: str):
    """`CMoERewardsCfg` 里 `<term>` 的 `params[key]` 字面量（非字面量会抛异常 ⇒ 测试本就该红）。"""
    source = _cmoe_source()
    tree = ast.parse(source)
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "CMoERewardsCfg")
    for node in cls.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and getattr(node.targets[0], "id", None) == term):
            continue
        for kw in node.value.keywords:
            if kw.arg == "params" and isinstance(kw.value, ast.Dict):
                for k, v in zip(kw.value.keys, kw.value.values):
                    if isinstance(k, ast.Constant) and k.value == key:
                        return ast.literal_eval(v)
        raise AssertionError(f"{term} 的 params 里没有 {key!r}")
    raise AssertionError(f"CMoERewardsCfg 里没有 {term} = RewTerm(...)")


def _gait_metric_pairs() -> list[tuple[str, str]]:
    """从 `CMoE_env_cfg.py` 源码里取出 `gait_metric_terms` 的逐对 `(label, term_name)`。"""
    src = _cmoe_source()
    key = '"gait_metric_terms": ('
    start = src.index(key) + len(key) - 1  # 指向那个 '('
    depth = 0
    end = None
    for i in range(start, len(src)):
        if src[i] == "(":
            depth += 1
        elif src[i] == ")":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        raise AssertionError("gait_metric_terms 的括号没有闭合")
    node = ast.parse(src[start:end], mode="eval").body
    return [(ast.literal_eval(elt.elts[0]), ast.literal_eval(elt.elts[1])) for elt in node.elts]


def _all_terrain_names():
    """工具里合并出的**真实**地形键（新增地形忘了归类时本测试会自动覆盖到它）。"""
    props, _ = columns.cmoe_overrides()
    base = [name for name, _p in columns.base_sub_terrains()]
    return list(dict.fromkeys([*base, *props.keys()]))


def _cmoe_source() -> str:
    return CMOE_CFG.read_text(encoding="utf-8-sig")


def _cmoe_reward_term_source(term: str) -> str:
    """取 `CMoERewardsCfg` 类体里 `<term> = RewTerm(...)` 那段源码。"""
    source = _cmoe_source()
    tree = ast.parse(source)
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "CMoERewardsCfg")
    for node in cls.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == term:
                return ast.get_source_segment(source, node)
    raise AssertionError(f"CMoERewardsCfg 里没有 {term} = RewTerm(...)")


def _scene_entity_cfg(node: ast.AST) -> dict:
    """`SceneEntityCfg("contact_forces", body_names=[...])` → 逐字段字典（非字面量留源码字符串）。"""
    out: dict[str, object] = {}
    if not isinstance(node, ast.Call):
        return {"raw": ast.unparse(node)}
    if node.args:
        out["name"] = ast.literal_eval(node.args[0])
    for kw in node.keywords:
        try:
            out[kw.arg] = ast.literal_eval(kw.value)
        except Exception:
            out[kw.arg] = ast.unparse(kw.value)
    return out


def _class_call_param_names(class_name: str) -> set[str]:
    """`rewards.py` 里某个 `ManagerTermBase` 子类 `__call__` 的参数名（去掉 self/env）。"""
    source = REWARDS.read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__call__")
    names = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
    return names - {"self", "env"}


def _reward_term_param_keys(term: str) -> set[str]:
    """`CMoERewardsCfg` 里 `<term> = RewTerm(..., params={...})` 的 params 键集合。"""
    source = _cmoe_source()
    tree = ast.parse(source)
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "CMoERewardsCfg")
    for node in cls.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == term):
            continue
        for kw in node.value.keywords:
            if kw.arg == "params" and isinstance(kw.value, ast.Dict):
                return {k.value for k in kw.value.keys if isinstance(k, ast.Constant)}
        return set()
    raise AssertionError(f"CMoERewardsCfg 里没有 {term} = RewTerm(...)")


def _done_term_params(term: str) -> dict:
    """读 `self.terminations.<term> = DoneTerm(...)` 的 func/params（literal + 原始源码）。"""
    source = _cmoe_source()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        if ast.unparse(node.targets[0]) != f"self.terminations.{term}":
            continue
        if not isinstance(node.value, ast.Call):
            continue
        out: dict[str, object] = {"raw": ast.get_source_segment(source, node.value)}
        for kw in node.value.keywords:
            if kw.arg == "func":
                out["func"] = ast.unparse(kw.value)
            elif kw.arg == "params" and isinstance(kw.value, ast.Dict):
                for key, value in zip(kw.value.keys, kw.value.values):
                    if not isinstance(key, ast.Constant):
                        continue
                    if key.value == "sensor_cfg":
                        out["sensor_cfg"] = _scene_entity_cfg(value)
                    else:
                        try:
                            out[key.value] = ast.literal_eval(value)
                        except Exception:
                            out[key.value] = ast.unparse(value)
        return out
    raise AssertionError(f"配置里没有 self.terminations.{term} = DoneTerm(...)")


# --------------------------------------------------------------------------- ① diag_base_height
@unittest.skipIf(torch is None, f"PyTorch unavailable: {IMPORT_ERROR}")
class TestDiagBaseHeight(unittest.TestCase):
    """① 有符号高度误差探针（正=偏高、负=偏矮；全落空 ⇒ 0；不平方、不乘门）。"""

    TARGET = 0.30
    TERRAINS = ["flat", "boxes", "gap"]

    def setUp(self):
        (self.diag,) = _load_defs(["diag_base_height"]).values()
        self._asset = _StubSceneEntityCfg("robot")
        self._sensor = _StubSceneEntityCfg("height_scanner_base")

    def _call(self, env, sensor=True):
        return self.diag(
            env,
            target_height=self.TARGET,
            asset_cfg=self._asset,
            sensor_cfg=self._sensor if sensor else None,
        )

    def test_sign_is_positive_when_too_high(self):
        """抬高 5 cm ⇒ **+0.05**（正）。"""
        env = _make_env(self.TERRAINS, [0.35, 0.35, 0.35], torch.zeros(3, 9))
        out = self._call(env)
        self.assertTrue(torch.allclose(out, torch.full((3,), 0.05), atol=1e-7), out)

    def test_sign_is_negative_when_squatting(self):
        """压矮 7 cm（本次退化实测值）⇒ **−0.07**（负）。"""
        env = _make_env(self.TERRAINS, [0.23, 0.23, 0.23], torch.zeros(3, 9))
        out = self._call(env)
        self.assertTrue(torch.allclose(out, torch.full((3,), -0.07), atol=1e-7), out)

    def test_is_signed_not_squared(self):
        """同一误差的两种符号必须给出**相反**读数 —— 平方版本（`base_height_l2`）做不到这件事。"""
        high = _make_env(self.TERRAINS, [0.35, 0.35, 0.35], torch.zeros(3, 9))
        low = _make_env(self.TERRAINS, [0.25, 0.25, 0.25], torch.zeros(3, 9))
        self.assertAlmostEqual(float(self._call(high)[0]), 0.05, places=7)
        self.assertAlmostEqual(float(self._call(low)[0]), -0.05, places=7)

    def test_ground_offset_uses_the_local_ground(self):
        """射线读到地面 −0.05 m ⇒ 局部地面 = 0.25 ⇒ 基座 0.30 时误差 +0.05（不是 0）。"""
        env = _make_env(self.TERRAINS, [0.30, 0.30, 0.30], torch.full((3, 9), -0.05))
        out = self._call(env)
        self.assertTrue(torch.allclose(out, torch.full((3,), 0.05), atol=1e-7), out)

    def test_all_rays_miss_falls_back_to_zero_error(self):
        """整束射线落空（悬在沟/洞上方）⇒ 退回自身根高 ⇒ **误差 0**（与 `base_height_l2` 同口径）。"""
        ray_hits = torch.full((3, 9), float("inf"))
        ray_hits[1, :] = float("nan")
        env = _make_env(self.TERRAINS, [0.30, 0.23, 0.40], ray_hits)
        out = self._call(env)
        self.assertTrue(torch.isfinite(out).all(), out)
        self.assertTrue(torch.allclose(out, torch.zeros(3), atol=1e-7), out)

    def test_partial_miss_uses_valid_rays_only(self):
        """逐环境：只有该环境自己的有效射线参与均值（3 条未命中不影响其余 6 条）。"""
        ray_hits = torch.full((3, 9), -0.05)
        ray_hits[0, :3] = float("nan")
        env = _make_env(self.TERRAINS, [0.30, 0.30, 0.30], ray_hits)
        out = self._call(env)
        self.assertTrue(torch.allclose(out, torch.full((3,), 0.05), atol=1e-7), out)

    def test_one_env_over_void_does_not_affect_the_others(self):
        """0 号环境整束落空、1 号正常 ⇒ 1 号必须仍被量到（逐环境判定，不是整批）。"""
        ray_hits = torch.zeros(2, 9)
        ray_hits[0, :] = float("inf")
        env = _make_env(["flat", "boxes"], [0.35, 0.40], ray_hits)
        out = self._call(env)
        self.assertAlmostEqual(float(out[0]), 0.0, places=7)
        self.assertAlmostEqual(float(out[1]), 0.10, places=7)

    def test_no_sensor_uses_the_world_target(self):
        """没有扫描器时退化成"世界系固定目标"（与 `base_height_l2` 的 flat 分支一致）。"""
        env = _make_env(self.TERRAINS, [0.35, 0.25, 0.30], torch.zeros(3, 9))
        out = self._call(env, sensor=False)
        self.assertTrue(torch.allclose(out, torch.tensor([0.05, -0.05, 0.0]), atol=1e-7), out)

    def test_no_gravity_gate(self):
        """倒立（`projected_gravity_b.z = +1`）时读数**不变** —— 诊断不做直立门控。"""
        env = _make_env(self.TERRAINS, [0.23, 0.23, 0.23], torch.zeros(3, 9), gravity_z=1.0)
        out = self._call(env)
        self.assertTrue(torch.allclose(out, torch.full((3,), -0.07), atol=1e-7), out)

    def test_weight_is_diagnostic_only(self):
        """配 **1e-6** 权重 ⇒ 每步贡献 ≤1e-6（相对整回合 ~4/s 可忽略）⇒ 不进策略的梯度。"""
        weights, _funcs, _notes = chk.run_chain(chk.CHAINS["cmoe"][0][1])
        self.assertEqual(weights.get("diag_base_height"), 1e-6)
        self.assertLessEqual(abs(float(weights["diag_base_height"])), 1e-5)

    def test_is_wired_into_the_per_terrain_log(self):
        """`gait_metric_terms` 必须把它接进逐列聚合 ⇒ 日志出现 `gait_base_height_<地形>`。"""
        self.assertIn('("base_height", "diag_base_height")', _cmoe_source())

    def test_cfg_term_points_at_the_base_scanner(self):
        """cfg 里的 RewTerm 必须用 `height_scanner_base`（基座正下方），不是前移的观测扫描器。"""
        block = _cmoe_reward_term_source("diag_base_height")
        self.assertIn("mdp.diag_base_height", block)
        self.assertIn("weight=1e-6", block)
        self.assertIn('SceneEntityCfg("height_scanner_base")', block)
        self.assertIn('"target_height": 0.30', block)


# --------------------------------------------------------------------------- ② 严格高度项
@unittest.skipIf(torch is None, f"PyTorch unavailable: {IMPORT_ERROR}")
class TestMaskedBaseHeightL2Strict(unittest.TestCase):
    """② flat-only、去重力门的 L2 高度项（用桩 env 跑**真实类源码**）。"""

    TARGET = 0.30
    CLASS_NAME = "MaskedBaseHeightL2Strict"

    def setUp(self):
        defs = _load_defs(
            ["_local_ground_target_height", "base_height_l2", "base_height_l2_strict", self.CLASS_NAME]
        )
        self.strict = defs["base_height_l2_strict"]
        self.gated = defs["base_height_l2"]
        self.cls = defs[self.CLASS_NAME]
        self._asset = _StubSceneEntityCfg("robot")
        self._sensor = _StubSceneEntityCfg("height_scanner_base")

    def _term_out(self, terrain_names, root_z, ray_hits, active=("flat",), gravity_z=-1.0):
        env = _make_env(terrain_names, root_z, ray_hits, gravity_z=gravity_z)
        term = self.cls(_cfg({"active_terrain_names": active}), env)
        out = term(
            env,
            target_height=self.TARGET,
            asset_cfg=self._asset,
            sensor_cfg=self._sensor,
            active_terrain_names=active,
        )
        return env, out

    # ---------------------------------------------------------------- 掩码正确性
    def test_non_flat_terrains_are_exactly_zero(self):
        """**非 flat 地形必须为 0**（白名单外恒 0）—— 障碍地形只受全地形 `base_height_l2 −10` 约束。"""
        terrains = [name for name in _all_terrain_names() if name != "flat"]
        self.assertGreaterEqual(len(terrains), 10, f"地形集太小，测试没意义：{terrains}")
        # 蹲 7 cm（会触发 −35 的姿态），仍必须在每一个非 flat 列上恰好为 0
        _env, out = self._term_out(terrains, [0.23] * len(terrains), torch.zeros(len(terrains), 9))
        self.assertEqual(out.tolist(), [0.0] * len(terrains), f"非 flat 列不是 0：{out}")

    def test_flat_columns_equal_the_ungated_l2_error(self):
        """flat 上 = **未加门的 L2 平方误差**（不乘权重；权重由 RewardManager 乘）。"""
        _env, out = self._term_out(["flat", "flat"], [0.23, 0.35], torch.zeros(2, 9))
        expected = torch.tensor([(0.23 - self.TARGET) ** 2, (0.35 - self.TARGET) ** 2])
        self.assertTrue(torch.allclose(out, expected, atol=1e-7), (out, expected))

    def test_weight_times_term_gives_the_expected_step_penalty(self):
        """−35 × (0.07)² = **−0.1715**/步（蹲 7 cm 的代价；对照旧 `−10 × 0.0049 = −0.049`）。"""
        _env, out = self._term_out(["flat"], [0.23], torch.zeros(1, 9))
        self.assertAlmostEqual(float(out[0]), 0.07 ** 2, places=7)
        self.assertAlmostEqual(-35.0 * float(out[0]), -0.1715, places=7)

    def test_mixed_terrain_batch_masks_per_env(self):
        """同一批里逐环境掩码（不是整批开关）：flat 有值、boxes/gap/rough 为 0。"""
        _env, out = self._term_out(
            ["flat", "boxes", "gap", "random_rough"], [0.23] * 4, torch.zeros(4, 9)
        )
        self.assertEqual(out.tolist()[1:], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(float(out[0]), 0.07 ** 2, places=7)

    def test_local_ground_logic_is_shared_with_base_height_l2(self):
        """地面偏移 −0.05 m 时目标变 0.25 ⇒ 误差 (0.23−0.25)²（证明用了同一套局部地面逻辑）。"""
        _env, out = self._term_out(["flat"], [0.23], torch.full((1, 9), -0.05))
        self.assertAlmostEqual(float(out[0]), (0.23 - 0.25) ** 2, places=7)

    def test_all_rays_miss_reverts_to_zero_error(self):
        """整束落空 ⇒ 退回自身根高 ⇒ 误差 0（与 `base_height_l2` 同口径，不产生 NaN）。"""
        env, out = self._term_out(["flat"], [0.23], torch.full((1, 9), float("nan")))
        self.assertTrue(torch.isfinite(out).all())
        self.assertAlmostEqual(float(out[0]), 0.0, places=7)
        self.assertEqual(env.num_envs, 1)

    def test_default_active_terrain_is_flat(self):
        """不传 `active_terrain_names` 时默认 `("flat",)`（cfg 里省略也不会静默变成全地形）。"""
        env = _make_env(["flat", "boxes"], [0.23, 0.23], torch.zeros(2, 9))
        term = self.cls(_cfg({}), env)  # cfg.params 里没有这个键
        out = term(env, target_height=self.TARGET, asset_cfg=self._asset, sensor_cfg=self._sensor)
        self.assertGreater(float(out[0]), 0.0)
        self.assertEqual(float(out[1]), 0.0)

    # ---------------------------------------------------------------- 去重力门
    def test_gravity_gate_removed_but_l2_logic_identical(self):
        """机身倾斜（−g_z = 0.35 ⇒ 门 = 0.5）时：严格项 = **2 ×** 带门项。

        这同时证明两件事：① 门确实被去掉了；② 其余算式（射线 → 局部地面 → 平方）逐字共用。
        """
        env = _make_env(["flat"], [0.23], torch.zeros(1, 9), gravity_z=-0.35)
        strict = self.strict(env, target_height=self.TARGET, asset_cfg=self._asset, sensor_cfg=self._sensor)
        gated = self.gated(env, target_height=self.TARGET, asset_cfg=self._asset, sensor_cfg=self._sensor)
        self.assertAlmostEqual(float(strict[0]), 0.07 ** 2, places=7)
        self.assertAlmostEqual(float(gated[0]), 0.5 * 0.07 ** 2, places=7)
        self.assertAlmostEqual(float(strict[0]), 2.0 * float(gated[0]), places=7)

    def test_upright_env_matches_the_gated_version(self):
        """直立（−g_z = 1 ⇒ 门 = 1）时两者**数值相同** ⇒ 去门控只在倾斜姿态上产生差别。"""
        env = _make_env(["flat"], [0.23], torch.zeros(1, 9), gravity_z=-1.0)
        strict = self.strict(env, target_height=self.TARGET, asset_cfg=self._asset, sensor_cfg=self._sensor)
        gated = self.gated(env, target_height=self.TARGET, asset_cfg=self._asset, sensor_cfg=self._sensor)
        self.assertAlmostEqual(float(strict[0]), float(gated[0]), places=7)


# --------------------------------------------------------------------------- ③④/接线（源码级）
class TestPosturePenaltyWiring(unittest.TestCase):
    """③ lin_vel_z_l2 −4、④ 接触罚 −5/−0.1＋50 N 终止项、② 的 −35 启用与 flat-only。"""

    @classmethod
    def setUpClass(cls):
        cls.src = _cmoe_source()

    # ------------------------------------------------------------------ ② 严格高度项接线
    def test_strict_height_is_enabled_with_minus_35(self):
        self.assertIn("self.rewards.base_height_flat_l2.func = mdp.MaskedBaseHeightL2Strict", self.src)
        self.assertIn("self.rewards.base_height_flat_l2.weight = -35.0", self.src)

    def test_strict_height_default_weight_is_zero_in_the_base_cfg(self):
        """`CMoERewardsCfg` 默认 **0.0** ⇒ 不影响其它任务；只在 CMoE rough 的 `__post_init__` 启用。"""
        block = _cmoe_reward_term_source("base_height_flat_l2")
        self.assertIn("mdp.MaskedBaseHeightL2Strict", block)
        self.assertIn("weight=0.0", block)
        self.assertIn('"active_terrain_names": ("flat",)', block)

    def test_strict_height_is_flat_only(self):
        """白名单必须显式写 `("flat",)`，且**不是** `free_terrain_names` 豁免语义。"""
        self.assertIn(
            'self.rewards.base_height_flat_l2.params["active_terrain_names"] = ("flat",)', self.src
        )
        self.assertNotIn('base_height_flat_l2.params["free_terrain_names"]', self.src)

    def test_strict_height_reuses_base_height_params(self):
        """`target_height`/`sensor_cfg` 从既有 `base_height_l2` 现取/同值 ⇒ 不会两处漂移。"""
        self.assertIn(
            'self.rewards.base_height_flat_l2.params["target_height"] = self.rewards.base_height_l2.params[',
            self.src,
        )
        self.assertIn(
            'self.rewards.base_height_flat_l2.params["sensor_cfg"] = SceneEntityCfg("height_scanner_base")',
            self.src,
        )

    def test_strict_height_keeps_a_10_plus_35_stack_on_flat(self):
        """flat 上两项同时生效 ⇒ 总强度 ≈ −45；障碍地形只有 −10（`base_height_l2` 在 rough 基类）。"""
        self.assertIn("self.rewards.base_height_l2.weight = -10.0", ROUGH_CFG.read_text(encoding="utf-8"))
        self.assertIn("self.rewards.base_height_flat_l2.weight = -35.0", self.src)

    def test_strict_height_call_signature_accepts_every_cfg_param(self):
        """`__call__` 的参数名必须覆盖 cfg 的**全部** params 键。

        Isaac Lab 的 `RewardManager` 对 class-term 是 `cfg.func(env, **cfg.params)` ⇒ 少一个同名参数
        就是**运行期 TypeError**（离线 AST 审计看不到），多一个无用参数只会被静默忽略。这里把
        `active_terrain_names`（本项新增的白名单参数）一起钉住。
        """
        params = _reward_term_param_keys("base_height_flat_l2")
        accepts = _class_call_param_names("MaskedBaseHeightL2Strict")
        self.assertEqual(params - accepts, set(), f"cfg 传了 {params - accepts}，类里没有同名参数")
        self.assertIn("active_terrain_names", accepts)

    def test_diag_base_height_call_signature_matches_the_function(self):
        """`diag_base_height(env, target_height, asset_cfg, sensor_cfg)` 必须与 cfg 的 params 键同名。"""
        source = REWARDS.read_text(encoding="utf-8")
        tree = ast.parse(source)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "diag_base_height")
        accepts = {a.arg for a in fn.args.args} - {"env"}
        self.assertEqual(_reward_term_param_keys("diag_base_height") - accepts, set())

    def test_masked_class_uses_the_house_pattern(self):
        """新掩码类必须与其它 `Masked*` 同类：`__init__` 缓存静态掩码、`__call__` 乘掩码。"""
        source = REWARDS.read_text(encoding="utf-8")
        block = re.search(r"class MaskedBaseHeightL2Strict\(ManagerTermBase\):(.*?)\ndef ", source, re.S)
        self.assertIsNotNone(block, "mdp/rewards.py 里没有 MaskedBaseHeightL2Strict")
        body = block.group(1)
        self.assertIn("_terrain_type_mask(env", body)
        self.assertIn("base_height_l2_strict(env, target_height, asset_cfg, sensor_cfg)", body)
        self.assertIn("self._active_mask.float()", body)

    # ------------------------------------------------------------------ ③ 竖直速度罚
    def test_lin_vel_z_is_doubled_and_mask_keeps_the_jump_exemption(self):
        self.assertIn("self.rewards.lin_vel_z_l2.weight = -4.0", self.src)
        self.assertNotIn("self.rewards.lin_vel_z_l2.weight = -2.0", self.src)
        # 掩码保持 boxes/gap 豁免（2026-09-29 取消豁免 ⇒ "gap/boxes 过不去" ⇒ 当日回退 f7d1dc3）
        self.assertIn('self.rewards.lin_vel_z_l2.params["free_terrain_names"] = ("boxes", "gap")', self.src)

    # ------------------------------------------------------------------ ④ 接触罚
    def test_undesired_contacts_is_minus_5(self):
        self.assertIn("self.rewards.undesired_contacts.weight = -5.0", self.src)
        self.assertNotIn("self.rewards.undesired_contacts.weight = -0.5", self.src)

    def test_contact_forces_is_minus_point_1(self):
        self.assertIn("self.rewards.contact_forces.weight = -0.1", self.src)

    def test_illegal_contact_body_done_term(self):
        params = _done_term_params("illegal_contact_body")
        self.assertEqual(params["func"], "mdp.illegal_contact")
        self.assertEqual(params["threshold"], 50.0, "阈值必须是 50 N（轻擦容忍、称重跪地才终止）")
        self.assertEqual(params["sensor_cfg"]["name"], "contact_forces")
        self.assertEqual(params["sensor_cfg"]["body_names"], [r"^(?!.*_FOOT).*"])
        self.assertIn("SceneEntityCfg", params["raw"])

    def test_illegal_contact_body_has_a_one_line_switch(self):
        """默认**关闭**（2026-09-30 实测：50 N 把 100% 回合都终止了）+ 保留一行开关与开启说明。

        证据（run `cmoe_v5_8_posture`）：`Episode_Termination/illegal_contact_body = 1.0000`、
        回合长度 6.3 步（0.13 s）、`mean_reward −0.058`、`level_mean 0` ⇒ 阈值 50 N 不可用。
        若将来重启，阈值至少 150~250 N 且要求持续接触，并先离线标定接触力量级。
        """
        self.assertIn("ENABLE_ILLEGAL_CONTACT_BODY_TERMINATION = False", self.src,
                      "默认应为 False（50 N 实测不可用）")
        self.assertIn("if ENABLE_ILLEGAL_CONTACT_BODY_TERMINATION:", self.src,
                      "开关必须包住 DoneTerm 的创建，才能一行关掉")

    def test_base_contact_termination_unchanged(self):
        """既有基座触地终止（1 N）不动 —— 新项是**追加**的，不是替换。"""
        params = _done_term_params("illegal_contact")
        self.assertEqual(params["threshold"], 1.0)
        self.assertIn("self.base_link_name", str(params["sensor_cfg"]["body_names"]))

    # ------------------------------------------------------------------ -gaitfree 不得归零
    def test_gaitfree_does_not_zero_the_posture_term(self):
        block = re.search(r"class Imgo2CMoEGaitFreeEnvCfg(.*)", self.src, re.S)
        self.assertIsNotNone(block)
        body = block.group(1)
        # 只断言"没有任何对它的赋值"（docstring 里提到它是允许的、也正是要写清楚的）
        self.assertNotIn("self.rewards.base_height_flat_l2", body,
                         "严格高度项是姿态项（不是步态形状先验），-gaitfree 里不该有对它的赋值")
        self.assertNotIn("base_height_flat_l2.weight = 0.0", body)
        # 反向对照：五项手工步态 shaping 确实仍被归零（没把归零逻辑改坏）
        for term in ("joint_mirror", "feet_air_time", "feet_height_body",
                     "feet_air_time_variance", "feet_gait"):
            self.assertIn(f"if self.rewards.{term} is not None:", body)


class TestEffectiveWeightsViaAudit(unittest.TestCase):
    """用仓库自己的 AST 审计链核对**最终生效值**（后赋值覆盖前赋值，静态源码看不出谁赢）。"""

    @classmethod
    def setUpClass(cls):
        cls.cmoe, _f, _n = chk.run_chain(chk.CHAINS["cmoe"][0][1])
        cls.gaitfree, _f2, _n2 = chk.run_chain(chk.CHAINS["cmoe-gaitfree"][0][1])

    def test_cmoe_effective_values(self):
        expected = {
            "base_height_flat_l2": -35.0,
            "knee_height_flat": -20.0,
            "base_height_l2": -10.0,
            "undesired_contacts": -5.0,
            "contact_forces": -0.1,
            "lin_vel_z_l2": -4.0,
            "diag_base_height": 1e-6,
            "diag_knee_height_min": 1e-6,
        }
        for term, value in expected.items():
            self.assertEqual(self.cmoe[term], value, f"{term} 的最终生效权重不对")

    def test_gaitfree_keeps_the_posture_terms(self):
        for term, value in (("base_height_flat_l2", -35.0), ("knee_height_flat", -20.0),
                            ("lin_vel_z_l2", -4.0), ("undesired_contacts", -5.0),
                            ("contact_forces", -0.1), ("diag_knee_height_min", 1e-6)):
            self.assertEqual(self.gaitfree[term], value, f"{term} 在 -gaitfree 链里被改动了")

    def test_effective_term_counts(self):
        """cmoe 29 → **31**（＋膝高软地板、＋`diag_knee_height_min`）；gaitfree 25 → **27**。"""
        eff = {t: v for t, v in self.cmoe.items() if isinstance(v, (int, float)) and v != 0}
        eff_gf = {t: v for t, v in self.gaitfree.items() if isinstance(v, (int, float)) and v != 0}
        self.assertEqual(len(eff), 31, f"生效项数变了：{sorted(eff)}")
        self.assertEqual(len(eff_gf), 27, f"gaitfree 生效项数变了：{sorted(eff_gf)}")


class TestTerrainColumnsAuditCoversTheNewMask(unittest.TestCase):
    """新掩码（白名单 flat）必须进 `check_terrain_columns.MASKED_NAMES`，否则 0 列时静默失效。"""

    def test_mask_registered(self):
        self.assertEqual(columns.MASKED_NAMES["base_height_flat_l2.active_terrain_names"], ("flat",))
        self.assertEqual(columns.MASKED_NAMES["knee_height_flat.active_terrain_names"], ("flat",))

    def test_flat_has_columns_and_the_audit_passes(self):
        self.assertEqual(columns.main([]), 0, "check_terrain_columns 必须返回 0（全部掩码名字都有列）")


# =========================================================================== ⑤ 膝关节高度软地板
KNEE_MIN_HEIGHT = 0.111   # 与 cfg 同步（`test_cfg_min_height_matches_the_fk_calibration` 锁定）
KNEE_WEIGHT = -20.0       # 与 cfg 同步（接线测试锁定）
SHANK_LEGS = ("FL", "FR", "RL", "RR")
STAND_JOINT_ANGLES = (0.0, 0.87, -1.82)  # hip / thigh / shank（`assets/imgo2.py` 的默认姿态）
BASE_STAND_HEIGHT = 0.30                 # `base_height_l2` 的目标高度


def _fk_leg_origin_z(leg: str, tip: str, angles) -> float:
    """用仓库既有 FK（`audit_amp_dataset.read_chain`/`forward_kinematics`）算 `tip` link
    原点在**基座系**里的 z（m）。URDF 每次从盘上重读 ⇒ 断言直接绑在模型文件上。

    `angles` 必须与该链的**可动关节数**一致（SHANK 链 = hip/thigh/shank 3 个；
    THIGH 链 = hip/thigh 2 个）——数量不符时 `forward_kinematics` 会抛 ValueError。
    """
    urdf = ET.parse(URDF).getroot()
    return amp_audit.forward_kinematics(amp_audit.read_chain(urdf, f"{leg}_{tip}"), list(angles))[2]


def _cmoe_block_context(anchor: str, before: int = 3000, after: int = 200) -> str:
    """取 `CMoE_env_cfg.py` 里 `anchor` **之前** `before` 字符的上下文（含标定注释）。"""
    src = _cmoe_source()
    idx = src.index(anchor)
    return src[max(0, idx - before):idx + after]


@unittest.skipIf(torch is None, f"PyTorch unavailable: {IMPORT_ERROR}")
class TestKneeHeightSoftFloor(unittest.TestCase):
    """⑤ 平地膝关节高度软地板（内核 + 掩码类，用桩 env 跑**真实源码**）。"""

    CLASS_NAME = "MaskedKneeHeightFlat"
    KERNEL = "knee_height_soft_floor"
    MIN = KNEE_MIN_HEIGHT
    WEIGHT = KNEE_WEIGHT

    def setUp(self):
        defs = _load_defs([self.KERNEL, self.CLASS_NAME])
        self.kernel = defs[self.KERNEL]
        self.cls = defs[self.CLASS_NAME]

    def _kernel_out(self, terrain_names, body_z, body_ids=(0, 1, 2, 3), min_height=None):
        env, asset_cfg = _make_knee_env(terrain_names, body_z, body_ids)
        out = self.kernel(env, min_height if min_height is not None else self.MIN, asset_cfg)
        return env, out

    def _term_out(self, terrain_names, body_z, active=("flat",), body_ids=(0, 1, 2, 3)):
        env, asset_cfg = _make_knee_env(terrain_names, body_z, body_ids)
        term = self.cls(_knee_cfg(active=active, body_ids=body_ids), env)
        out = term(env, min_height=self.MIN, asset_cfg=asset_cfg, active_terrain_names=active)
        return env, out

    # ---------------------------------------------------------------- 语义（逐环境 Σ relu）
    def test_all_knees_above_the_threshold_is_exactly_zero(self):
        """全部膝高于阈值 ⇒ **0**（软地板只罚"低"，不奖励"高"）。"""
        _env, out = self._kernel_out(["flat", "flat"], [[0.20] * 4, [0.1111] * 4])
        self.assertEqual(out.tolist(), [0.0, 0.0])

    def test_one_knee_low_by_3cm_costs_exactly_3cm_times_weight(self):
        """**一条**膝低 3 cm ⇒ 项值恰为 `0.03`、加权后恰为 `3 × weight`（用户给的验收点）。"""
        _env, out = self._term_out(["flat"], [[0.20, 0.20, 0.20, self.MIN - 0.03]])
        self.assertAlmostEqual(float(out[0]), 0.03, places=7)
        self.assertAlmostEqual(float(out[0]) * self.WEIGHT, 3.0 * 0.01 * self.WEIGHT, places=7)
        self.assertAlmostEqual(float(out[0]) * self.WEIGHT, -0.6, places=7)

    def test_four_knees_low_by_2cm_cost_four_times(self):
        """**四条**各低 2 cm ⇒ `4 × 2 cm × weight`（逐腿各自计一次，不是取均值/取最差）。"""
        _env, out = self._term_out(["flat"], [[self.MIN - 0.02] * 4])
        self.assertAlmostEqual(float(out[0]), 4 * 0.02, places=6)
        self.assertAlmostEqual(float(out[0]) * self.WEIGHT, 4 * 0.02 * self.WEIGHT, delta=1e-5)
        self.assertAlmostEqual(float(out[0]) * self.WEIGHT, -1.6, delta=1e-5)

    def test_is_linear_not_quadratic(self):
        """低 2 cm 与低 4 cm 的读数比必须是 **1:2**（线性）—— 平方版会给出 1:4。"""
        _env, two = self._term_out(["flat"], [[self.MIN - 0.02] * 4])
        _env2, four = self._term_out(["flat"], [[self.MIN - 0.04] * 4])
        self.assertAlmostEqual(float(four[0]) / float(two[0]), 2.0, places=5)

    def test_exactly_at_the_threshold_is_zero(self):
        """恰好等于阈值 ⇒ 0（`relu(0) = 0`，边界不产生惩罚）。"""
        _env, out = self._kernel_out(["flat"], [[self.MIN] * 4])
        self.assertAlmostEqual(float(out[0]), 0.0, places=7)

    # ---------------------------------------------------------------- 掩码（白名单 flat）
    def test_non_flat_terrains_are_exactly_zero(self):
        """**非 flat 地形必须为 0**：即使膝低到 −5 cm，障碍列也不该被罚（越障要屈膝）。"""
        terrains = [name for name in _all_terrain_names() if name != "flat"]
        self.assertGreaterEqual(len(terrains), 10, f"地形集太小，测试没意义：{terrains}")
        _env, out = self._term_out(terrains, [[-0.05] * 4] * len(terrains))
        self.assertEqual(out.tolist(), [0.0] * len(terrains), f"非 flat 列不是 0：{out}")

    def test_mixed_terrain_batch_masks_per_env(self):
        """同一批里**逐环境**掩码（不是整批开关）：flat 有值、boxes/gap/rough 为 0。"""
        _env, out = self._term_out(
            ["flat", "boxes", "gap", "random_rough"],
            [[0.20, 0.20, 0.20, self.MIN - 0.03], [self.MIN - 0.09] * 4, [0.0] * 4, [-1.0] * 4],
        )
        self.assertEqual(out.tolist()[1:], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(float(out[0]), 0.03, places=6)

    def test_default_active_terrain_is_flat(self):
        """不传 `active_terrain_names` 时默认 `("flat",)`（cfg 里省略也不会静默变成全地形）。"""
        env, asset_cfg = _make_knee_env(["flat", "boxes"], [[self.MIN - 0.03] * 4] * 2)
        term = self.cls(_cfg({}), env)   # cfg.params 里没有这个键
        out = term(env, min_height=self.MIN, asset_cfg=asset_cfg)
        self.assertGreater(float(out[0]), 0.0)
        self.assertEqual(float(out[1]), 0.0)

    def test_terrain_mask_matches_the_audit_registry(self):
        """本项的白名单必须与 `check_terrain_columns.MASKED_NAMES` 登记的一致（否则审计看不出打错字）。"""
        self.assertEqual(columns.MASKED_NAMES["knee_height_flat.active_terrain_names"], ("flat",))

    # ---------------------------------------------------------------- body_ids（读的是 SHANK）
    def test_uses_the_resolved_shank_body_ids(self):
        """`body_ids` 必须被真正用来取数：同一批 body 里只有 SHANK 那 4 列算进 Σ。"""
        # 8 个 body，前 4 个（HIP/THIGH）远低于阈值、后 4 个（SHANK）都高 ⇒ 期望 0
        _env, out = self._term_out(["flat"], [[-1.0] * 4 + [0.30] * 4], body_ids=(4, 5, 6, 7))
        self.assertAlmostEqual(float(out[0]), 0.0, places=7)
        # 反过来：SHANK 是前 4 列且低 3 cm 中的一条 ⇒ 0.03
        _env2, out2 = self._term_out(
            ["flat"], [[0.30, 0.30, 0.30, self.MIN - 0.03] + [0.30] * 4], body_ids=(0, 1, 2, 3)
        )
        self.assertAlmostEqual(float(out2[0]), 0.03, places=7)

    def test_wrong_knee_count_raises_instead_of_silently_scaling(self):
        """`body_names` 匹配到 ≠4 个 link 时必须**显式报错**，不能静默少算/多算。"""
        env, _asset_cfg = _make_knee_env(["flat"], [[0.1] * 8])
        with self.assertRaises(ValueError):
            self.cls(_knee_cfg(body_ids=(0, 1, 2)), env)

    # ---------------------------------------------------------------- 家法（与其它 Masked* 同类）
    def test_masked_class_uses_the_house_pattern(self):
        source = REWARDS.read_text(encoding="utf-8")
        block = re.search(r"class MaskedKneeHeightFlat\(ManagerTermBase\):(.*?)\ndef ", source, re.S)
        self.assertIsNotNone(block, "mdp/rewards.py 里没有 MaskedKneeHeightFlat")
        body = block.group(1)
        self.assertIn("_terrain_type_mask(env", body)
        self.assertIn("knee_height_soft_floor(env, min_height, asset_cfg)", body)
        self.assertIn("self._active_mask.float()", body)

    def test_kernel_uses_world_frame_body_positions(self):
        """内核必须读 `asset.data.body_pos_w[:, body_ids, 2]`（**世界系 z**），不是 root_pos_w。"""
        source = REWARDS.read_text(encoding="utf-8")
        block = re.search(r"def knee_height_soft_floor\((.*?)\n\n\n", source, re.S)
        self.assertIsNotNone(block)
        self.assertIn("body_pos_w[:, asset_cfg.body_ids, 2]", block.group(1))
        self.assertIn("torch.clamp(min_height - knee_z, min=0.0).sum(dim=1)", block.group(1))


@unittest.skipIf(torch is None, f"PyTorch unavailable: {IMPORT_ERROR}")
class TestDiagKneeHeightMin(unittest.TestCase):
    """⑤ 的读数探针：4 条小腿里**最低**那条的世界系 z（有符号、只记录、不做门控）。"""

    def setUp(self):
        (self.diag,) = _load_defs(["diag_knee_height_min"]).values()

    def _call(self, terrain_names, body_z, body_ids=(0, 1, 2, 3)):
        env, asset_cfg = _make_knee_env(terrain_names, body_z, body_ids)
        return self.diag(env, asset_cfg=asset_cfg)

    def test_is_min_not_mean(self):
        """(0.16, 0.16, 0.16, 0.09) ⇒ **0.09**（min）；mean 会给 0.1425，两者相差很大。"""
        out = self._call(["flat"], [[0.16, 0.16, 0.16, 0.09]])
        self.assertAlmostEqual(float(out[0]), 0.09, places=7)

    def test_is_signed(self):
        """膝被压到地面以下也如实读**负值**（不做 clamp、不平方、有符号）。"""
        out = self._call(["flat", "boxes"], [[0.13, 0.12, 0.11, -0.02], [0.20, 0.20, 0.20, 0.20]])
        self.assertAlmostEqual(float(out[0]), -0.02, places=7)
        self.assertAlmostEqual(float(out[1]), 0.20, places=7)

    def test_uses_the_resolved_shank_body_ids(self):
        """8 个 body、只取第 4–7 列（SHANK）⇒ 最低值来自那 4 列，而不是全 body 的最小值。"""
        out = self._call(["flat"], [[-1.0] * 4 + [0.09, 0.16, 0.16, 0.16]], body_ids=(4, 5, 6, 7))
        self.assertAlmostEqual(float(out[0]), 0.09, places=7)

    def test_weight_is_diagnostic_only(self):
        """配 **1e-6** 权重 ⇒ 每步贡献 ≤1e-6（相对整回合 ~4/s 可忽略）⇒ 不进策略的梯度。"""
        weights, _funcs, _notes = chk.run_chain(chk.CHAINS["cmoe"][0][1])
        self.assertEqual(weights.get("diag_knee_height_min"), 1e-6)
        self.assertLessEqual(abs(float(weights["diag_knee_height_min"])), 1e-5)

    def test_is_wired_into_the_per_terrain_log(self):
        """`gait_metric_terms` 必须把它接进逐列聚合 ⇒ 日志出现 `gait_knee_height_<地形>`。"""
        self.assertIn(("knee_height", "diag_knee_height_min"), _gait_metric_pairs())

    def test_labels_do_not_collide(self):
        """三个高度类标签必须互不相同（`gait_height_*` / `gait_base_height_*` / `gait_knee_height_*`）。"""
        labels = [label for label, _term in _gait_metric_pairs()]
        self.assertEqual(len(labels), len(set(labels)), f"逐列 metric label 有重复：{labels}")
        for wanted in ("height", "base_height", "knee_height"):
            self.assertIn(wanted, labels)
        # 同一个 label 也不能被两个 term 复用（`gait_<label>_<地形>` 会互相覆盖）
        pairs = _gait_metric_pairs()
        self.assertEqual(len(pairs), len({term for _l, term in pairs}), f"term 被复用：{pairs}")

    def test_cfg_term_points_at_the_shank_bodies(self):
        """cfg 里的 RewTerm 必须用 `.*_SHANK`（膝关节）且权重 1e-6。"""
        block = _cmoe_reward_term_source("diag_knee_height_min")
        self.assertIn("mdp.diag_knee_height_min", block)
        self.assertIn("weight=1e-6", block)
        self.assertIn('body_names=".*_SHANK"', block)

    def test_call_signature_matches_the_function(self):
        """`diag_knee_height_min(env, asset_cfg)` 必须与 cfg 的 params 键同名（否则运行期 TypeError）。"""
        source = REWARDS.read_text(encoding="utf-8")
        tree = ast.parse(source)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "diag_knee_height_min")
        accepts = {a.arg for a in fn.args.args} - {"env"}
        self.assertEqual(_reward_term_param_keys("diag_knee_height_min") - accepts, set())


class TestKneeHeightThresholdFromFK(unittest.TestCase):
    """阈值 `knee_min` 必须由**离线 FK**标定（站姿膝高 × 0.70，四舍五入到 mm），不许拍脑袋。"""

    RATIO = 0.70
    TOL = 5e-3

    @classmethod
    def setUpClass(cls):
        cls.cfg_min_height = _reward_term_param_literal("knee_height_flat", "min_height")

    def _stand_knee_z(self) -> list[float]:
        return [
            BASE_STAND_HEIGHT + _fk_leg_origin_z(leg, "SHANK", STAND_JOINT_ANGLES)
            for leg in SHANK_LEGS
        ]

    def test_standing_knee_height_is_symmetric_across_the_four_legs(self):
        """站姿下四条小腿同高（模型左右/前后对称）⇒ 标定不依赖选哪条腿。"""
        heights = self._stand_knee_z()
        for h in heights:
            self.assertAlmostEqual(h, heights[0], places=9, msg=f"四条腿膝高不一致：{heights}")

    def test_shank_origin_is_the_knee_not_the_thigh(self):
        """负向对照：`*_THIGH` 原点的 z 明显**高于** `*_SHANK`（站姿下差 `0.22·cos 0.87 = 0.142 m`）
        ⇒ 证明取的是膝关节而不是大腿根（也证明 FK 的链路确实读到了 SHANK 那段）。"""
        knee = self._stand_knee_z()[0]
        thigh = BASE_STAND_HEIGHT + _fk_leg_origin_z("FL", "THIGH", (0.0, 0.87))
        self.assertAlmostEqual(thigh, BASE_STAND_HEIGHT, delta=2e-3, msg="THIGH 原点应大致在髋轴高度上")
        self.assertGreater(thigh - knee, 0.10, f"thigh={thigh}, knee={knee}")
        self.assertLess(thigh - knee, 0.20, f"thigh={thigh}, knee={knee}")

    def test_the_documented_standing_knee_height(self):
        """cfg 注释里写的算式用的是这三个数字：hip 0 / thigh 0.87 / shank −1.82、base 0.30。"""
        self.assertEqual(tuple(STAND_JOINT_ANGLES), (0.0, 0.87, -1.82))
        self.assertAlmostEqual(BASE_STAND_HEIGHT, 0.30, places=9)
        knee_z_stand = self._stand_knee_z()[0]
        self.assertAlmostEqual(knee_z_stand, 0.158388, delta=2e-3,
                               msg="站姿膝高与 cfg 注释里记的 0.158388 m 不符（URDF 变了？）")

    def test_cfg_min_height_matches_the_fk_calibration(self):
        """**核心断言**：cfg 里的 `min_height` == `0.70 × 站姿膝高`（容差 5e-3 m）。"""
        knee_z_stand = self._stand_knee_z()[0]
        expected = self.RATIO * knee_z_stand
        self.assertLessEqual(
            abs(float(self.cfg_min_height) - expected), self.TOL,
            f"cfg min_height={self.cfg_min_height}，FK 标定 0.70×{knee_z_stand:.6f}={expected:.6f}",
        )

    def test_cfg_min_height_is_rounded_to_the_millimetre(self):
        """`knee_min` 四舍五入到 mm（0.110872 ⇒ 0.111 m）。"""
        self.assertAlmostEqual(float(self.cfg_min_height), round(float(self.cfg_min_height), 3), places=9)

    def test_min_height_is_a_fn_of_a_real_model_file(self):
        """FK 的输入必须是仓库里**存在**的那份 URDF（路径写错时本测试要报错而不是静默通过）。"""
        self.assertTrue(URDF.is_file(), f"找不到 {URDF}")
        self.assertIn("<link name=\"FL_SHANK\">", URDF.read_text(encoding="utf-8"))
        for leg in SHANK_LEGS:
            self.assertIn(f'<link name="{leg}_SHANK">', URDF.read_text(encoding="utf-8"))


class TestKneeHeightWiring(unittest.TestCase):
    """⑤ 的配置级接线：−20、白名单 flat、一行可调、`-gaitfree` 不归零、探针 1e-6。"""

    @classmethod
    def setUpClass(cls):
        cls.src = _cmoe_source()

    def test_knee_height_is_enabled_with_minus_20(self):
        self.assertIn("self.rewards.knee_height_flat.weight = -20.0", self.src)

    def test_default_weight_is_zero_in_the_base_cfg(self):
        """`CMoERewardsCfg` 默认 **0.0** ⇒ 不影响其它任务；只在 CMoE rough 的 `__post_init__` 启用。"""
        block = _cmoe_reward_term_source("knee_height_flat")
        self.assertIn("mdp.MaskedKneeHeightFlat", block)
        self.assertIn("weight=0.0", block)
        self.assertIn('"min_height": 0.111', block)
        self.assertIn('"active_terrain_names": ("flat",)', block)
        self.assertIn('body_names=".*_SHANK"', block)

    def test_is_flat_only_and_uses_the_whitelist_param_name(self):
        """白名单必须显式写 `("flat",)`，且**不是** `free_terrain_names` 豁免语义。"""
        block = _cmoe_reward_term_source("knee_height_flat")
        self.assertIn('"active_terrain_names": ("flat",)', block)
        self.assertNotIn("free_terrain_names", block)

    def test_cfg_comment_documents_the_fk_calibration(self):
        """算式与结果必须写在 cfg 注释里（**证据链**：站姿膝高、0.70、四舍五入到 mm）。"""
        context = _cmoe_block_context("knee_height_flat = RewTerm(", before=3200)
        self.assertIn("knee_z_stand", context)
        self.assertIn("0.158388", context)
        self.assertIn("0.70", context)
        self.assertIn("0.111", context)
        self.assertIn("hip 0 / thigh 0.87 / shank −1.82", context)

    def test_call_signature_accepts_every_cfg_param(self):
        """`__call__` 的参数名必须覆盖 cfg 的**全部** params 键（否则运行期 TypeError）。"""
        params = _reward_term_param_keys("knee_height_flat")
        accepts = _class_call_param_names("MaskedKneeHeightFlat")
        self.assertEqual(params - accepts, set(), f"cfg 传了 {params - accepts}，类里没有同名参数")
        self.assertIn("min_height", accepts)
        self.assertIn("active_terrain_names", accepts)

    def test_kernel_signature_accepts_the_geometry_params(self):
        """内核只接**几何**参数（`min_height`/`asset_cfg`）；掩码参数由 `MaskedKneeHeightFlat` 持有。"""
        source = REWARDS.read_text(encoding="utf-8")
        tree = ast.parse(source)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "knee_height_soft_floor")
        accepts = {a.arg for a in fn.args.args} - {"env"}
        self.assertEqual(_reward_term_param_keys("knee_height_flat") - accepts, {"active_terrain_names"},
                         "内核只该比 cfg 少一个掩码参数（它不做掩码）")

    def test_weight_line_documents_how_to_switch_it_off(self):
        """权重那一段必须写明回退方式（一行可关 / 阈值一行可调）。"""
        idx = self.src.index("self.rewards.knee_height_flat.weight = -20.0")
        context = self.src[max(0, idx - 2500):idx + 400]
        self.assertIn("一行可关", context)
        self.assertIn("一行可调", context)

    def test_gaitfree_does_not_zero_the_knee_term(self):
        """膝高地板是**姿态**项（不是步态形状先验）⇒ `-gaitfree` 里不该有对它的赋值。"""
        block = re.search(r"class Imgo2CMoEGaitFreeEnvCfg(.*)", self.src, re.S)
        self.assertIsNotNone(block)
        body = block.group(1)
        self.assertNotIn("self.rewards.knee_height_flat", body,
                         "膝高软地板是姿态项，-gaitfree 里不该有对它的赋值")
        self.assertNotIn("knee_height_flat.weight = 0.0", body)

    def test_diag_term_is_weighted_at_1e_6(self):
        block = _cmoe_reward_term_source("diag_knee_height_min")
        self.assertIn("weight=1e-6", block)


if __name__ == "__main__":
    unittest.main()
