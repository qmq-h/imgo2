"""地形课程晋级判据的离线回归（前向进度 + 速度跟踪门控）。

2026-09-24 用户要求："地形等级提升还是需要考虑速度跟踪效果"。据此 `terrain_levels_vel_logged`
相对上游有**两处有意偏离**：
  ① 晋级看**沿 +x 的前向进度**（`root_x − origin_x`），不再用含横向分量的欧氏距离
     —— 上游那条把"横移绕开障碍"也算通过（docs §29.15/§29.16 实测过）；
  ② 晋级还要求**本回合平均速度跟踪核 > `tracking_move_up`**，低于 `tracking_move_down` 直接降级。

跟踪均值从奖励管理器的 `_episode_sums` 反解（`/weight/(回合步数·step_dt)`）。时机已被核对：
`_reset_idx` 里 `curriculum_manager.compute()` 在最前、`reward_manager.reset()`（读走并清零）在后、
`episode_length_buf` 最后才清零 ⇒ 拿到的是刚结束那一回合的值。

`mdp/curriculums.py` 顶层 `import isaaclab...`（缺 `omni.log`）无法整模块 import，这里用
`sys.modules` 塞桩后 exec **真实源码**（含函数体内的 `from .utils import ...` 相对导入）。
"""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CURRICULUMS = ROOT / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/mdp/curriculums.py"

try:
    import torch
except ModuleNotFoundError as error:
    torch = None
    IMPORT_ERROR: object = error
else:
    IMPORT_ERROR = None


class _Data:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class _Scene(dict):
    pass


class _Terrain:
    def __init__(self, size, levels, terrain_names=("flat",)):
        # 逐列日志会遍历 `sub_terrains.keys()`，所以桩里要把用例用到的地形名都放进去
        self.cfg = _Data(terrain_generator=_Data(
            size=size, sub_terrains={name: None for name in dict.fromkeys(terrain_names)}
        ))
        self.terrain_levels = levels
        self.calls = []

    def update_env_origins(self, env_ids, move_up, move_down):
        self.calls.append((list(env_ids), move_up.clone(), move_down.clone()))


class _CommandManager:
    def __init__(self, command):
        self._cmd = command

    def get_command(self, name):
        return self._cmd


class _TermCfg:
    def __init__(self, weight):
        self.weight = weight


class _RewardManager:
    def __init__(self, sums, weights):
        self._episode_sums = sums
        self._weights = weights

    def get_term_cfg(self, name):
        return _TermCfg(self._weights[name])


class _Env:
    def __init__(self, rows, command=(0.3, 0.0), track_avg=None, episode_steps=1000, term="track_world_vel_xy_exp",
                 terrains=None, metrics=None):
        """rows: [(x, y)] 相对出生点的位移；track_avg: 每个环境本回合的平均跟踪核（None＝不提供分项）。

        terrains: 每个环境所属的**地形列名**（用于 relaxed 阈值那组用例；None ⇒ 全部算 "flat"）。
        """
        n = len(rows)
        self.num_envs = n
        self.device = "cpu"
        self.step_dt = 0.02
        self.max_episode_length_s = 20.0
        self.episode_length_buf = torch.full((n,), episode_steps, dtype=torch.long)
        self.scene = _Scene()
        self.scene.env_origins = torch.zeros(n, 3)
        offsets = torch.tensor(rows, dtype=torch.float32)
        pos = torch.zeros(n, 3)
        pos[:, 0] = offsets[:, 0]
        pos[:, 1] = offsets[:, 1]
        self.scene["robot"] = _Data(data=_Data(root_pos_w=pos))
        self.terrain_names = list(terrains) if terrains is not None else ["flat"] * n
        self.scene.terrain = _Terrain((8.0, 4.0), torch.zeros(n, dtype=torch.long), self.terrain_names)
        cmd = torch.zeros(n, 3)
        cmd[:, 0], cmd[:, 1] = command[0], command[1]
        self.command_manager = _CommandManager(cmd)
        if track_avg is None and not metrics:
            self.reward_manager = None
        else:
            sums: dict[str, torch.Tensor] = {}
            weights: dict[str, float] = {}
            if track_avg is not None:
                weights[term] = 5.0
                sums[term] = torch.tensor(track_avg, dtype=torch.float32) * 5.0 * (episode_steps * self.step_dt)
            # 额外"度量为"（如 1e-6 权重的 GaitReward 度量）：反解回原始乘积值
            for name, (weight, values) in (metrics or {}).items():
                weights[name] = weight
                sums[name] = torch.tensor(values, dtype=torch.float32) * weight * (episode_steps * self.step_dt)
            self.reward_manager = _RewardManager(sums, weights)


def _load_functions():
    """用桩模块 exec 真实源码，返回命名空间（含两个函数）。"""
    import sys
    import types

    src = CURRICULUMS.read_text(encoding="utf-8")
    pkg_name = "cmoe_curriculums_under_test"
    keys = ("isaaclab", "isaaclab.managers", pkg_name, f"{pkg_name}.utils")
    saved = {k: sys.modules.get(k) for k in keys}

    class _StubSceneEntityCfg:
        def __init__(self, name="robot", **kwargs):
            self.name = name

    isaaclab = types.ModuleType("isaaclab")
    isaaclab.__path__ = []
    managers = types.ModuleType("isaaclab.managers")
    managers.SceneEntityCfg = _StubSceneEntityCfg
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = []
    utils = types.ModuleType(f"{pkg_name}.utils")
    utils.is_env_assigned_to_terrain = lambda env, name: torch.tensor(
        [t == name for t in getattr(env, "terrain_names", ["flat"] * env.num_envs)], dtype=torch.bool
    )
    sys.modules.update({"isaaclab": isaaclab, "isaaclab.managers": managers, pkg_name: pkg,
                        f"{pkg_name}.utils": utils})
    try:
        ns = pkg.__dict__
        ns.update({"torch": torch})
        exec(compile(src, str(CURRICULUMS), "exec"), ns)  # noqa: S102 - 只执行仓库自己的函数
    finally:
        # 只还原 isaaclab 的桩；**测试包本身保留在 sys.modules**，因为函数体内有
        # `from .utils import is_env_assigned_to_terrain` 的相对导入，调用时才解析。
        for key in ("isaaclab", "isaaclab.managers"):
            value = saved[key]
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
    return ns


@unittest.skipIf(torch is None, f"PyTorch unavailable: {IMPORT_ERROR}")
class TestTerrainCurriculum(unittest.TestCase):
    # (x 位移, y 位移, 跟踪核) —— 命令 0.3 m/s ⇒ 降级距离阈值 = 0.3*20*0.5 = 3 m；晋级阈值 = size[0]/2 = 4 m
    ROWS = [
        (5.0, 0.0, 0.90),   # 0：进度够 + 跟踪好 ⇒ 晋级
        (5.0, 0.0, 0.50),   # 1：进度够但跟踪一般（0.35~0.80）⇒ 冻结
        (5.0, 0.0, 0.20),   # 2：跟踪太差 ⇒ 降级
        (1.0, 0.0, 0.90),   # 3：跟踪好但没走够 ⇒ 降级
        (3.0, 3.0, 0.90),   # 4：横向走了 3 m（欧氏 4.24 > 4）但前向只 3 m ⇒ **不许晋级**（防"横移绕开"）
    ]

    def _run(self, ns, rows, with_tracking=True, terrains=None, metrics=None, metric_terms=()):
        env = _Env(
            [(r[0], r[1]) for r in rows],
            track_avg=[r[2] for r in rows] if with_tracking else None,
            terrains=terrains,
            metrics=metrics,
        )
        # 阈值显式传入 ⇒ 用例不随默认值漂移（默认值另有 cfg 断言覆盖）
        out = ns["terrain_levels_vel_logged"](
            env, torch.arange(len(rows)),
            **{"tracking_move_up": 0.80, "tracking_move_up_relaxed": 0.50,
               "gait_metric_terms": metric_terms},
        )
        _ids, up, down = env.scene.terrain.calls[0]
        return up, down, out

    @classmethod
    def setUpClass(cls):
        cls.ns = _load_functions()

    def test_progress_and_tracking_both_required(self):
        up, down, out = self._run(self.ns, self.ROWS)
        self.assertEqual(up.int().tolist(), [1, 0, 0, 0, 0], "晋级的组合不对")
        self.assertEqual(down.int().tolist(), [0, 0, 1, 1, 0], "降级的组合不对")
        self.assertAlmostEqual(float(out["tracking_is_used"]), 1.0)
        self.assertAlmostEqual(float(out["tracking_pass_frac"]), 3 / 5)
        self.assertAlmostEqual(float(out["tracking_fail_frac"]), 1 / 5)

    def test_lateral_travel_does_not_promote(self):
        """关键回归：env4 的欧氏距离 4.24 m 超过 4 m，但前向只 3 m ⇒ 旧判据会晋级，现在不许。"""
        up, _down, _out = self._run(self.ns, self.ROWS)
        self.assertEqual(int(up[4]), 0)

    def test_falls_back_to_distance_only_when_term_missing(self):
        up, down, out = self._run(self.ns, self.ROWS, with_tracking=False)
        self.assertAlmostEqual(float(out["tracking_is_used"]), 0.0)
        self.assertEqual(up.int().tolist(), [1, 1, 1, 0, 0])  # env1/2 只看进度 ⇒ 晋级
        self.assertEqual(down.int().tolist(), [0, 0, 0, 1, 0])
        self.assertNotIn("tracking_mean", out)

    def test_relaxed_threshold_for_step_terrains(self):
        """2026-09-24 晚：台阶/独立块这三类放宽到 0.65（否则长期卡在 frozen 带）。

        两个环境同为 进度 5 m、跟踪 0.70：在 `boxes` 上应**晋级**（0.70 > 0.65），
        在 `flat` 上应**冻结**（0.70 < 0.80）。
        """
        rows = [(5.0, 0.0, 0.55), (5.0, 0.0, 0.55), (5.0, 0.0, 0.45)]
        up, down, out = self._run(self.ns, rows, terrains=["boxes", "flat", "boxes"])
        self.assertEqual(up.int().tolist(), [1, 0, 0],
                         "boxes 放宽到 0.50 ⇒ 0.55 晋级；flat 保持 0.80 ⇒ 0.55 冻结；0.45 在 boxes 上也不够")
        self.assertEqual(down.int().tolist(), [0, 0, 0])
        self.assertAlmostEqual(float(out["tracking_up_threshold_mean"]), (0.50 + 0.80 + 0.50) / 3, places=6)

    def test_per_column_tracking_is_logged(self):
        """逐列跟踪均值：用来判断某列卡住是"跟踪不达标"还是"真过不去"。"""
        rows = [(5.0, 0.0, 0.90), (5.0, 0.0, 0.50), (5.0, 0.0, 0.70)]
        _up, _down, out = self._run(self.ns, rows, terrains=["boxes", "boxes", "flat"])
        self.assertAlmostEqual(float(out["tracking_boxes"]), 0.70, places=6)
        self.assertAlmostEqual(float(out["tracking_flat"]), 0.70, places=6)

    def test_per_column_gait_metrics_are_logged(self):
        """逐列步态度量（用户要求）：三成对方式的 GaitReward 度量按列取"本回合平均"。

        度量项权重极小（这里 1e-6）⇒ 反解后应还原成**原始乘积值**，且不能影响晋级判据。
        """
        rows = [(5.0, 0.0, 0.90)] * 3
        metrics = {
            "m_trot": (1e-6, [0.80, 0.60, 0.70]),
            "m_bound": (1e-6, [0.55, 0.90, 0.60]),
        }
        up, _down, out = self._run(
            self.ns, rows, terrains=["boxes", "boxes", "flat"],
            metrics=metrics, metric_terms=(("trot", "m_trot"), ("bound", "m_bound")),
        )
        self.assertEqual(up.int().tolist(), [1, 1, 1], "度量项不参与晋级判据")
        self.assertAlmostEqual(float(out["gait_trot_boxes"]), 0.70, places=6)
        self.assertAlmostEqual(float(out["gait_bound_boxes"]), 0.725, places=6)
        self.assertAlmostEqual(float(out["gait_trot_flat"]), 0.70, places=6)
        self.assertAlmostEqual(float(out["gait_trot_mean"]), 0.70, places=6)

    def test_promotion_wins_over_demotion(self):
        """进度与跟踪都很好时不降级（`move_down *= ~move_up`）。"""
        up, down, _out = self._run(self.ns, [(5.0, 0.0, 0.95)])
        self.assertEqual(up.int().tolist(), [1])
        self.assertEqual(down.int().tolist(), [0])


if __name__ == "__main__":
    unittest.main()
