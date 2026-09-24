"""`feet_gait`（`TrotWithoutGapReward`）的按地形豁免 —— 离线回归。

2026-09-24 用户："那就开 feet gait，同样加掩码"。`TrotWithoutGapReward` 继承 `GaitReward`
（6 核乘积：2 个"对角对内同步"核 × 4 个"对角对之间反相"核），只把结果乘上**与其余四项步态 shaping
同一套**的地形掩码（豁免 `boxes`/`gap`）。它是全配方里**唯一**显式要求"对角反相"、因而能排除
**pronk** 的项（`joint_mirror` 与 `feet_air_time_variance` 都被 pronk 满足）。

`mdp/rewards.py` 顶层 `import isaaclab...`（缺 `omni.log`）无法整模块 import，所以用 AST 抽真实源码 +
桩 exec：桩只替代基类 `GaitReward`（返回常数 1.0，便于观察"乘法"是否生效）与 `_terrain_type_mask`。
"""

from __future__ import annotations

import ast
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REWARDS = ROOT / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/mdp/rewards.py"
CMOE_CFG = ROOT / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/base_move/CMoE_env_cfg.py"

try:
    import torch
except ModuleNotFoundError as error:
    torch = None
    IMPORT_ERROR: object = error
else:
    IMPORT_ERROR = None


class _Env:
    def __init__(self, terrain_names):
        self.num_envs = len(terrain_names)
        self.device = "cpu"
        self.terrain_names = list(terrain_names)
        self.scene = {}


class _BaseStub:
    """`GaitReward` 的桩：`__call__` 恒返回 1.0，`__init__` 不做事。"""

    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env

    def __call__(self, *args, **kwargs):
        return torch.ones(args[0].num_envs)


def _load_class(terrain_mask):
    source = REWARDS.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "TrotWithoutGapReward"
    )
    ns = {
        "torch": torch,
        "GaitReward": _BaseStub,
        "ManagerTermBase": _BaseStub,
        "ManagerBasedRLEnv": object,
        "RewTerm": object,
        "Articulation": object,
        "SceneEntityCfg": object,
        "_terrain_type_mask": terrain_mask,
    }
    exec(compile(ast.get_source_segment(source, node), str(REWARDS), "exec"), ns)  # noqa: S102
    return ns["TrotWithoutGapReward"]


def _terrain_mask(env, terrain_names):
    return torch.tensor([name in terrain_names for name in env.terrain_names], dtype=torch.bool)


@unittest.skipIf(torch is None, f"PyTorch unavailable: {IMPORT_ERROR}")
class TestFeetGaitMask(unittest.TestCase):
    TERRAINS = ["pyramid_stairs", "gap", "boxes", "flat"]

    def _reward(self, free=("boxes", "gap")):
        env = _Env(self.TERRAINS)
        cfg = type("Cfg", (), {"params": {"free_terrain_names": free}})()
        cls = _load_class(_terrain_mask)
        term = cls(cfg, env)
        return term(env, 0.707, "base_velocity", 0.2, 0.5, 0.1, (("a", "b"), ("c", "d")), None, None, free)

    def test_exempt_terrain_columns_are_zero(self):
        out = self._reward()
        self.assertAlmostEqual(float(out[0]), 1.0, places=6, msg="非豁免列应保留 GaitReward 的值")
        self.assertAlmostEqual(float(out[1]), 0.0, places=6, msg="gap 列应豁免（被掩掉）")
        self.assertAlmostEqual(float(out[2]), 0.0, places=6, msg="boxes 列应豁免（被掩掉）")
        self.assertAlmostEqual(float(out[3]), 1.0, places=6, msg="非豁免列应保留 GaitReward 的值")

    def test_default_free_names(self):
        """不传 `free_terrain_names` 时默认豁免 `boxes`/`gap`（与其余四项一致）。"""
        env = _Env(self.TERRAINS)
        cfg = type("Cfg", (), {"params": {}})()
        term = _load_class(_terrain_mask)(cfg, env)
        out = term(env, 0.707, "base_velocity", 0.2, 0.5, 0.1, (("a", "b"), ("c", "d")), None, None)
        self.assertEqual(out.int().tolist(), [1, 0, 0, 1])

    def test_is_a_multiplicative_mask_not_a_replacement(self):
        """掩码是"乘法"：非豁免列必须等于基类 `GaitReward` 的输出（这里桩为 1.0）。"""
        self.assertAlmostEqual(float(self._reward(free=())[0]), 1.0, places=6)


class TestCmoeCfgWiring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = CMOE_CFG.read_text(encoding="utf-8-sig")

    def test_enabled_with_weight_one(self):
        self.assertIn("self.rewards.feet_gait.func = mdp.TrotWithoutGapReward", self.src)
        self.assertIn("self.rewards.feet_gait.weight = 1.0", self.src)
        self.assertIn('self.rewards.feet_gait.params["free_terrain_names"] = ("boxes", "gap")', self.src)

    def test_pairs_are_diagonal_trot(self):
        block = re.search(r'self\.rewards\.feet_gait\.params\["synced_feet_pair_names"\] = \((.*?)\)\n',
                          self.src, re.S)
        self.assertIsNotNone(block, "配置里没有设置 synced_feet_pair_names")
        pairs = re.findall(r'"(FL|FR|RL|RR)_FOOT"', block.group(1))
        self.assertEqual(len(pairs), 4, f"应当给全两组对角对：{pairs}")
        self.assertEqual(set(pairs), {"FL", "FR", "RL", "RR"})
        self.assertIn('("FL_FOOT", "RR_FOOT")', block.group(1))
        self.assertIn('("FR_FOOT", "RL_FOOT")', block.group(1))


if __name__ == "__main__":
    unittest.main()
