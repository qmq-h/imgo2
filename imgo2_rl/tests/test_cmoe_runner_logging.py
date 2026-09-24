"""`CMoEOnPolicyRunner.log` 对「同一轮里各条 `ep_infos` 键集合不同」的回归（2026-09-24）。

背景：逐列指标 `Curriculum/terrain_levels/tracking_<地形>` / `gait_<标签>_<地形>` 只在"这一步
**确实**有该列的回合力样本"时才写（`env_ids` 只是这一步刚结束的环境 —— 4096 环境 × 24 步 ÷
平均回合长度 ≈ 150 个/轮，再摊到 24 次 `_reset_idx`，每次只剩约 6 个）。于是同一轮里各条
`ep_infos` 的**键集合不同**：

* 旧代码 `for key in ep_infos[0]` 驱动 ⇒ 第一条缺、后面有的键会被**整轮丢掉**；
  反过来第一条有、后面缺 ⇒ `ep_info[key]` **KeyError**，足以打断一次几十小时的训练。
* 新代码按键的**并集**遍历、缺键的条目跳过 ⇒ 键不丢、也不会 KeyError；
  该轮的值＝该轮里有该键的那几次的均值。

`rl_lab.runners.__init__` 会 import 需要 Isaac 的 AMP runner，所以这里**不能直接 import**，
改为用桩模块 exec **真实 runner 源码**（与 `test_terrain_curriculum.py` 同一套路），只调用 `log()`。
"""

from __future__ import annotations

import contextlib
import io
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/rl_lab/rl_lab/runners/cmoe_on_policy_runner.py"

try:
    import torch
except ModuleNotFoundError as error:  # pragma: no cover - 本机若缺 torch 就整体跳过
    torch = None
    IMPORT_ERROR: object = error
else:
    IMPORT_ERROR = None


class _Writer:
    """最小 SummaryWriter 桩：只记录 add_scalar。"""

    def __init__(self):
        self.scalars: list[tuple[str, float, int]] = []

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, float(value), int(step)))


class _Std:
    def mean(self):
        return torch.tensor(0.9)


class _ActorCritic:
    gate_weights = None
    std = _Std()


class _Alg:
    learning_rate = 1.0e-3
    actor_critic = _ActorCritic()


class _Env:
    num_envs = 4


def _load_runner_class():
    """用桩子模块 exec 真实 runner 源码，返回 `CMoEOnPolicyRunner`。"""
    src = RUNNER.read_text(encoding="utf-8")
    pkg = "cmoe_runner_under_test"
    subs = (pkg, f"{pkg}.runners", f"{pkg}.algorithms", f"{pkg}.algorithms.cmoe_ppo",
            f"{pkg}.envs", f"{pkg}.envs.vec_env", f"{pkg}.modules", f"{pkg}.modules.cmoe_actor_critic")
    saved = {name: sys.modules.get(name) for name in subs}

    def _module(name, **attrs):
        mod = types.ModuleType(name)
        mod.__path__ = []
        for key, value in attrs.items():
            setattr(mod, key, value)
        return mod

    sys.modules.update({
        pkg: _module(pkg),
        f"{pkg}.runners": _module(f"{pkg}.runners"),
        f"{pkg}.algorithms": _module(f"{pkg}.algorithms"),
        f"{pkg}.algorithms.cmoe_ppo": _module(f"{pkg}.algorithms.cmoe_ppo", CMoEPPO=object),
        f"{pkg}.envs": _module(f"{pkg}.envs"),
        f"{pkg}.envs.vec_env": _module(f"{pkg}.envs.vec_env", VecEnv=object),
        f"{pkg}.modules": _module(f"{pkg}.modules"),
        f"{pkg}.modules.cmoe_actor_critic": _module(
            f"{pkg}.modules.cmoe_actor_critic", CMoEActorCritic=object
        ),
    })
    module = types.ModuleType(f"{pkg}.runners.cmoe_on_policy_runner")
    module.__package__ = f"{pkg}.runners"
    sys.modules[module.__name__] = module
    try:
        exec(compile(src, str(RUNNER), "exec"), module.__dict__)  # noqa: S102 - 只执行仓库自己的代码
    finally:
        for name in subs:
            value = saved[name]
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    return module.CMoEOnPolicyRunner


@unittest.skipIf(torch is None, f"PyTorch unavailable: {IMPORT_ERROR}")
class TestCMoERunnerLogging(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner_cls = _load_runner_class()

    def _log(self, ep_infos, it=7):
        runner = self.runner_cls.__new__(self.runner_cls)
        runner.device = "cpu"
        runner.tot_timesteps = 0
        runner.tot_time = 0.0
        runner.current_learning_iteration = 0
        runner.num_steps_per_env = 24
        runner.env = _Env()
        runner.alg = _Alg()
        runner.writer = _Writer()
        locs = {
            "ep_infos": ep_infos,
            "it": it,
            "collection_time": 1.0,
            "learn_time": 0.5,
            "mean_value_loss": 0.1,
            "mean_surrogate_loss": 0.2,
            "rewbuffer": [],
            "lenbuffer": [],
            "num_learning_iterations": 10,
        }
        with contextlib.redirect_stdout(io.StringIO()):
            self.runner_cls.log(runner, locs, *[torch.tensor(0.0)] * 9)
        logged = {
            tag[len("Episode/"):]: value
            for tag, value, _step in runner.writer.scalars
            if tag.startswith("Episode/")
        }
        return logged

    def test_keys_missing_from_first_entry_are_not_dropped(self):
        """第一条缺、后面有的键不能被整轮丢掉（旧代码 `ep_infos[0]` 驱动会丢）。"""
        logged = self._log([
            {"Curriculum/terrain_levels/level_flat": 1.0},
            {"Curriculum/terrain_levels/level_flat": 2.0,
             "Curriculum/terrain_levels/tracking_boxes": 0.70},
        ])
        self.assertIn("Curriculum/terrain_levels/tracking_boxes", logged)
        self.assertAlmostEqual(logged["Curriculum/terrain_levels/tracking_boxes"], 0.70, places=6)
        self.assertAlmostEqual(logged["Curriculum/terrain_levels/level_flat"], 1.5, places=6)

    def test_keys_missing_from_later_entries_do_not_raise(self):
        """第一条有、后面缺的键不许 KeyError（旧代码会直接打断训练）。"""
        logged = self._log([
            {"Curriculum/terrain_levels/tracking_gap": 0.30,
             "Curriculum/terrain_levels/level_flat": 1.0},
            {"Curriculum/terrain_levels/level_flat": 3.0},
            {"Curriculum/terrain_levels/level_flat": 5.0},
        ])
        self.assertAlmostEqual(logged["Curriculum/terrain_levels/tracking_gap"], 0.30, places=6)
        self.assertAlmostEqual(logged["Curriculum/terrain_levels/level_flat"], 3.0, places=6)

    def test_no_nan_in_logged_scalars(self):
        """逐列指标缺样本时**不该出现键**（由 curriculums.py 保证），这里守住"写出去的不含 NaN"。"""
        logged = self._log([
            {"Curriculum/terrain_levels/level_gap": 2.0},
            {"Curriculum/terrain_levels/tracking_gap": 0.4},
        ])
        for key, value in logged.items():
            self.assertFalse(value != value, f"{key} 被写成 NaN")


if __name__ == "__main__":
    unittest.main()
