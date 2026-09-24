"""`MaskedJointMirror` 的分地形对子切换（trot 对角对 / gap 左右对＝bound）离线回归。

为什么要用这种"抽取源码再 exec"的写法：`mdp/rewards.py` 在模块顶层 `import isaaclab...`，
裸解释器与被 launcher 调起的解释器都缺 `omni.log`（要给 `omni` 引导起 Isaac Sim 才行），
所以整模块无法 import。这里用 AST 从**真实源文件**里取出 `_pairwise_joint_mirror` 与
`MaskedJointMirror` 的源码片段，在带了桩的命名空间里 exec —— 测的仍是**仓库里的真代码**，
不是复制品。桩只替代 `ManagerTermBase`／`_terrain_type_mask`／场景对象。

锁定的缺陷模式：掩码/对子配错时训练照跑、日志无异常（2026-09-24 已有一次 `joint_mirror`
被晚赋值静默清零的先例）。另外也是上游一个真陷阱的回归：`mdp.joint_mirror` 把对子缓存在
**env** 上且只解析一次，因此"两套对子"不能用它连调两次。

步态语义（本文件用的姿态）：
* **trot 姿态**：`FL == RR = A`、`FR == RL = B`（A≠B）⇒ 满足对角对、违反左右对；
* **bound 姿态**：`FL == FR = A`、`RL == RR = B`（A≠B）⇒ 满足左右对、违反对角对。
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
except ModuleNotFoundError as error:  # 没有 torch 就跳过（纯标准库机器）
    torch = None
    IMPORT_ERROR: object = error
else:
    IMPORT_ERROR = None


LEGS = ("FL", "FR", "RL", "RR")
PARTS = ("hip", "thigh", "shank")
JOINTS = [f"{leg}_{part}_joint" for leg in LEGS for part in PARTS]

# 两条腿的"姿势"（hip, thigh, shank），差异刻意做得大，便于判定配错
A = (0.10, -0.50, 0.40)
B = (0.30, 0.90, -0.20)
# 单对、全关节的平方差之和 ⇒ 期望奖励（两对相同 ⇒ 平均后仍是它），门控为 1 时
PAIR_SQ_DIFF = sum((a - b) ** 2 for a, b in zip(A, B))


# ---------------------------------------------------------------- 假对象

class _Data:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class _FakeAsset:
    """只实现 `find_joints` 与 `data.{joint_pos,projected_gravity_b}`。"""

    def __init__(self, num_envs: int, gravity_z: float = -1.0, joint_pos=None):
        self.data = _Data(
            joint_pos=joint_pos if joint_pos is not None else torch.zeros(num_envs, len(JOINTS)),
            projected_gravity_b=torch.tensor([[0.0, 0.0, gravity_z]] * num_envs),
        )

    def find_joints(self, pattern: str):
        ids = [i for i, name in enumerate(JOINTS) if re.fullmatch(pattern, name)]
        return ids, [JOINTS[i] for i in ids]


class _Scene(dict):
    pass


class _Env:
    def __init__(self, names, gravity_z: float = -1.0):
        self.num_envs = len(names)
        self.device = "cpu"
        self.terrain_names = list(names)
        self.scene = _Scene()
        self.scene["robot"] = _FakeAsset(self.num_envs, gravity_z)

    def set_pose(self, poses):
        """poses：每个环境一个 {leg: (hip, thigh, shank)}。"""
        rows = []
        for pose in poses:
            rows.append([pose[leg][i] for leg in LEGS for i in range(3)])
        self.scene["robot"].data.joint_pos = torch.tensor(rows, dtype=torch.float32)


def trot_pose():
    """FL == RR、FR == RL。"""
    return {"FL": A, "RR": A, "FR": B, "RL": B}


def bound_pose():
    """FL == FR、RL == RR。"""
    return {"FL": A, "FR": A, "RL": B, "RR": B}


def _load_masked_joint_mirror(terrain_mask):
    """从真实源文件抽出两个定义的源码，在桩命名空间里 exec，返回类对象。"""
    source = REWARDS.read_text(encoding="utf-8")
    tree = ast.parse(source)
    wanted = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_pairwise_joint_mirror":
            wanted.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "MaskedJointMirror":
            wanted.append(node)
    assert len(wanted) == 2, "没能在 rewards.py 里找到 _pairwise_joint_mirror / MaskedJointMirror"

    class _StubTermBase:
        def __init__(self, cfg, env):
            self.cfg = cfg
            self.env = env

    ns = {
        "torch": torch,
        "ManagerTermBase": _StubTermBase,
        "ManagerBasedRLEnv": object,
        "RewTerm": object,
        "Articulation": object,
        "_terrain_type_mask": terrain_mask,
    }
    body = "\n\n".join(ast.get_source_segment(source, node) for node in wanted)
    exec(compile(body, str(REWARDS), "exec"), ns)  # noqa: S102 - 只执行仓库自己的两个定义
    return ns["MaskedJointMirror"]


def _terrain_mask(env, terrain_names):
    return torch.tensor([name in terrain_names for name in env.terrain_names], dtype=torch.bool)


DIAG = [["FR_(hip|thigh|shank).*", "RL_(hip|thigh|shank).*"], ["FL_(hip|thigh|shank).*", "RR_(hip|thigh|shank).*"]]
BOUND = [["FL_(hip|thigh|shank).*", "FR_(hip|thigh|shank).*"], ["RL_(hip|thigh|shank).*", "RR_(hip|thigh|shank).*"]]
FREE = ("boxes",)
GAP = ("gap",)


@unittest.skipIf(torch is None, f"PyTorch unavailable: {IMPORT_ERROR}")
class TestMaskedJointMirrorPerTerrain(unittest.TestCase):
    def _eval(self, env, *, bound_joints=BOUND, weight_joints=DIAG, free=FREE, bound=GAP):
        cfg = type("Cfg", (), {"params": {
            "asset_cfg": None, "mirror_joints": weight_joints,
            "free_terrain_names": free, "bound_terrain_names": bound,
            "bound_mirror_joints": bound_joints,
        }})()
        asset_cfg = type("A", (), {"name": "robot"})()
        term = _load_masked_joint_mirror(_terrain_mask)(cfg, env)
        return term(env, asset_cfg, weight_joints, free, bound, bound_joints)

    def test_trot_terrain_uses_diagonal_pairs(self):
        # trot 地形上：trot 姿态 ⇒ 0；bound 姿态 ⇒ >0（说明用的确实是对角对）。
        env = _Env(["pyramid_stairs", "random_rough"])
        env.set_pose([trot_pose(), bound_pose()])
        out = self._eval(env)
        self.assertAlmostEqual(float(out[0]), 0.0, places=6, msg="trot 地形上 trot 姿态不该被罚")
        self.assertAlmostEqual(float(out[1]), PAIR_SQ_DIFF, places=5,
                               msg="trot 地形用了错误的对子（应当是对角对）")

    def test_gap_switches_to_bound_pairs(self):
        # 沟壑上：bound 姿态 ⇒ 0；trot 姿态 ⇒ >0（说明换成了左右对）。
        env = _Env(["gap", "gap"])
        env.set_pose([bound_pose(), trot_pose()])
        out = self._eval(env)
        self.assertAlmostEqual(float(out[0]), 0.0, places=6, msg="沟壑没有换成 bound 左右对")
        self.assertAlmostEqual(float(out[1]), PAIR_SQ_DIFF, places=5,
                               msg="沟壑仍在对角对（那样这里会是 0）")

    def test_boxes_is_fully_exempt(self):
        env = _Env(["boxes"])
        env.set_pose([bound_pose()])
        self.assertAlmostEqual(float(self._eval(env)[0]), 0.0, places=6)

    def test_empty_bound_pairs_falls_back_to_exempt(self):
        env = _Env(["gap"])
        env.set_pose([bound_pose()])
        self.assertAlmostEqual(float(self._eval(env, bound_joints=())[0]), 0.0, places=6)

    def test_gravity_gate_zeroes_everything(self):
        env = _Env(["pyramid_stairs"], gravity_z=0.0)  # 完全躺平 ⇒ 门控为 0
        env.set_pose([bound_pose()])
        self.assertAlmostEqual(float(self._eval(env)[0]), 0.0, places=6)

    def test_two_pair_sets_do_not_share_a_cache(self):
        # 上游 `mdp.joint_mirror` 会把对子缓存在 env 上且只解析一次 ⇒ 两套对子会互相污染。
        # 本实现把缓存挂在 term 实例上，所以同一个 env 连续两次调用结果必须各按各自地形算。
        env = _Env(["pyramid_stairs", "gap"])
        env.set_pose([trot_pose(), bound_pose()])
        first = self._eval(env)
        second = self._eval(env)  # 新建实例、同一 env
        self.assertAlmostEqual(float(first[0]), 0.0, places=6)
        self.assertAlmostEqual(float(first[1]), 0.0, places=6)
        self.assertAlmostEqual(float(second[0]), 0.0, places=6)
        self.assertAlmostEqual(float(second[1]), 0.0, places=6)


class TestCmoeCfgWiring(unittest.TestCase):
    """确认配方把三套名单/对子真的写进了配置（纯 AST，不需要 torch）。"""

    @classmethod
    def setUpClass(cls):
        cls.src = CMOE_CFG.read_text(encoding="utf-8-sig")

    @staticmethod
    def _params_assignments(src):
        out = {}
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = ast.unparse(node.targets[0])
                if target.startswith("self.rewards.joint_mirror.params["):
                    key = target.split("[", 1)[1].rstrip("]").strip("'\"")
                    out[key] = ast.literal_eval(node.value)
        return out

    def test_terrain_name_lists(self):
        params = self._params_assignments(self.src)
        self.assertEqual(tuple(params.get("free_terrain_names", ())), ("boxes",))
        self.assertEqual(tuple(params.get("bound_terrain_names", ())), ("gap",))

    def test_bound_pairs_are_left_right(self):
        params = self._params_assignments(self.src)
        bound = params.get("bound_mirror_joints")
        self.assertTrue(bound, "配置里没有设置 bound_mirror_joints")
        legs = [(pair[0][:3], pair[1][:3]) for pair in bound]
        self.assertIn(("FL_", "FR_"), legs)
        self.assertIn(("RL_", "RR_"), legs)

    def test_trot_pairs_are_diagonal(self):
        params = self._params_assignments(self.src)
        legs = [(pair[0][:3], pair[1][:3]) for pair in params.get("mirror_joints", [])]
        self.assertIn(("FR_", "RL_"), legs)
        self.assertIn(("FL_", "RR_"), legs)

    def test_pairs_include_hip(self):
        params = self._params_assignments(self.src)
        for key in ("mirror_joints", "bound_mirror_joints"):
            for pair in params.get(key, []):
                for pattern in pair:
                    self.assertIn("(hip|thigh|shank)", pattern, f"{key} 应含 hip：{pair}")


if __name__ == "__main__":
    sys.exit(unittest.main())
