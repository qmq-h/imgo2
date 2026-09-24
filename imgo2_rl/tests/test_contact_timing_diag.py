"""接触时序诊断项（`diag_air_time` / `diag_pair_mismatch`）的离线回归。

2026-09-24 夜新增的两个 **1e-6 纯记录**项（用户同意）。为什么需要：

* `Episode_Reward/feet_air_time` 的时间平均＝`N_落地 · (ā − threshold)`（展开 `Σ_落地(last_air_time − c)`
  即得）⇒ 它**只约束这一个乘积**，单看它"长步幅慢步（ā≈0.45 s、4 次落地/s）"与"短步快蹭
  （ā≈0.05 s、23 次落地/s）"给的是同一个数 ⇒ 必须再记录 `ā` 自身才能分辨（docs §29.31.2）。
* 相位核 `GaitReward` 看的是**每一对脚的时间差**，差一超过 `max_err` 就贴地板 ⇒ 要按实测选
  `max_err`，就得知道"各对脚到底差多少"。本项给出六对 `|Δair| + |Δcon|` 的均值（同步侧尺度）：
  **四足近似同步（lockstep）时它≈0，2+2 分开（trot/bound）时它≈半个周期** ⇒ 与 `ā` 一起回答
  "是不是短而整齐的步子"（那正是让三个分类器都≈0.93、相位项失效的形态）。

`mdp/rewards.py` 顶层 `import isaaclab...`（缺 `omni.log`）无法整模块 import ⇒ 用 AST 抽真实函数源码 exec。
"""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REWARDS = ROOT / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/mdp/rewards.py"

try:
    import torch
except ModuleNotFoundError as error:
    torch = None
    IMPORT_ERROR: object = error
else:
    IMPORT_ERROR = None


class _Sensor:
    def __init__(self, last_air_time, current_air_time, current_contact_time):
        self.data = type(
            "_Data",
            (),
            {
                "last_air_time": torch.tensor([last_air_time], dtype=torch.float32),
                "current_air_time": torch.tensor([current_air_time], dtype=torch.float32),
                "current_contact_time": torch.tensor([current_contact_time], dtype=torch.float32),
            },
        )()


class _StubSceneEntityCfg:
    def __init__(self, name="contact_forces", **kwargs):
        self.name = name
        # 真 `SceneEntityCfg` 会把 body_names 解析成 body_ids；这里默认四足全取。
        self.body_ids = [0, 1, 2, 3]


class _Env:
    """最小 env 桩：`env.scene.sensors['contact_forces']` 返回给定接触时序的传感器。"""

    def __init__(self, last_air_time, current_air_time, current_contact_time):
        self.num_envs = 1
        self.device = "cpu"
        self.scene = type("_Scene", (), {})()
        self.scene.sensors = {
            "contact_forces": _Sensor(last_air_time, current_air_time, current_contact_time)
        }


def _load_functions():
    """AST 抽出两个诊断函数的真实源码，在只含 torch 的命名空间里 exec。"""
    src = REWARDS.read_text(encoding="utf-8")
    tree = ast.parse(src)
    wanted = {"diag_air_time", "diag_pair_mismatch"}
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    module = ast.Module(body=body, type_ignores=[])
    ns = {"torch": torch, "SceneEntityCfg": _StubSceneEntityCfg, "ContactSensor": object}
    exec(compile(module, str(REWARDS), "exec"), ns)  # noqa: S102 - 只执行仓库自己的函数
    return ns


@unittest.skipIf(torch is None, f"PyTorch unavailable: {IMPORT_ERROR}")
class TestContactTimingDiag(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ns = _load_functions()
        # 注意：直接 `cls.f = ns['f']` 会让它变成**绑定方法**（多传一个 self）⇒ 必须包成 staticmethod。
        cls.air_time = staticmethod(ns["diag_air_time"])
        cls.mismatch = staticmethod(ns["diag_pair_mismatch"])
        cls.sensor_cfg = _StubSceneEntityCfg()

    def test_mean_last_air_time(self):
        """`ā` ＝ 四足 `last_air_time` 均值（判定"长步幅 vs 短步碎步"的那个量）。"""
        env = _Env([0.40, 0.05, 0.05, 0.40], [0.0] * 4, [0.0] * 4)
        self.assertAlmostEqual(float(self.air_time(env, self.sensor_cfg)[0]), 0.225, places=5)

    def test_locked_feet_have_zero_mismatch(self):
        """四足完全同步（lockstep／pronk）⇒ 六对时间差全 0。"""
        env = _Env([0.3] * 4, [0.3] * 4, [0.0] * 4)
        self.assertAlmostEqual(float(self.mismatch(env, self.sensor_cfg)[0]), 0.0, places=5)

    def test_trot_snapshot_has_large_mismatch(self):
        """trot 的一瞬：FL/RR 悬空 0.3 s、FR/RL 支撑 0.3 s ⇒ 4 对相反（各 0.6）、2 对相同（0）。

        均值＝(4×0.6)/6＝0.4 ⇒ 与 lockstep 的 0 形成对照，正是"2+2 分开 vs 四足同步"的判据。
        """
        env = _Env([0.3] * 4, [0.30, 0.0, 0.0, 0.30], [0.0, 0.30, 0.30, 0.0])
        self.assertAlmostEqual(float(self.mismatch(env, self.sensor_cfg)[0]), 0.4, places=5)

    def test_pairs_cover_all_six_combinations(self):
        """六对必须都算进去：只改一只脚时，均值＝(它到另三只的差)/(6×2 项)。

        只让 RR 的 air 加 0.6（其余不动）⇒ 与 RR 相关的 3 对各贡献 0.6 ⇒ (3×0.6)/6＝0.3。
        若实现漏成"只有相邻对"或"只算 3 对"，这个数会变 ⇒ 钉住分母与配对覆盖。
        """
        env = _Env([0.0] * 4, [0.0, 0.0, 0.0, 0.6], [0.0] * 4)
        self.assertAlmostEqual(float(self.mismatch(env, self.sensor_cfg)[0]), 0.3, places=5)

    def test_uses_configured_body_ids(self):
        """`sensor_cfg.body_ids` 必须真的用来取列（而不是默认取全部/前四个）。"""
        env = _Env([0.1, 0.2, 0.3, 0.4], [0.0] * 4, [0.0] * 4)
        cfg = _StubSceneEntityCfg()
        cfg.body_ids = [3]
        self.assertAlmostEqual(float(self.air_time(env, cfg)[0]), 0.4, places=5)


if __name__ == "__main__":
    unittest.main()
