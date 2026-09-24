"""`base_height_l2` 的逐环境兜底回归（2026-09-24 修复：原为整批判定）。

Run（需要 Isaac Sim 引导过的解释器，因为 `mdp/rewards.py` 会 import isaaclab）:
    cd imgo2_rl
    bash scripts/run_isaaclab.sh -m unittest discover -s tests -p 'test_base_height_per_env.py' -v

锁定的缺陷：原实现是
    if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
        adjusted_target_height = asset.data.root_link_pos_w[:, 2]     # 全场误差记 0
4096 个环境里只要有**任意一个**的射线落空（悬在沟壑上方），**所有环境**的高度惩罚都会被关掉，
奖励因此依赖其它环境的状态、并随 `num_envs` 变化。修复后为逐环境判定：只用该环境自己的有效
射线求局部地面高度，全部落空才退回 `root_z`。
"""

from __future__ import annotations

import unittest

try:
    import torch

    from imgo2_rl.tasks.manager_based.locomotion.velocity.mdp import rewards as mdp_rewards
except ModuleNotFoundError as error:  # 本机没有 Isaac Sim 引导时跳过
    torch = None
    IMPORT_ERROR = error
else:
    IMPORT_ERROR = None


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
    def __init__(self, entities):
        self.scene = _Scene(entities)


def _make_env(root_z, ray_hits):
    """root_z/ray_hits: (N,) / (N,R) torch 张量；projected_gravity 固定直立（gating = 1）。"""
    num_envs = root_z.shape[0]
    robot = _Entity(
        "robot",
        root_pos_w=torch.stack([torch.zeros(num_envs), torch.zeros(num_envs), root_z], dim=1),
        root_link_pos_w=torch.stack([torch.zeros(num_envs), torch.zeros(num_envs), root_z], dim=1),
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]] * num_envs),
    )
    scanner = _Entity("height_scanner_base", ray_hits_w=torch.stack(
        [torch.zeros_like(ray_hits), torch.zeros_like(ray_hits), ray_hits], dim=-1
    ))
    return _Env({"robot": robot, "height_scanner_base": scanner})


def _cfg(name):
    return type("Cfg", (), {"name": name})()


@unittest.skipIf(torch is None, f"PyTorch/Isaac Lab unavailable: {IMPORT_ERROR}")
class TestBaseHeightPerEnv(unittest.TestCase):
    TARGET = 0.30

    def _reward(self, env):
        return mdp_rewards.base_height_l2(
            env,
            target_height=self.TARGET,
            asset_cfg=_cfg("robot"),
            sensor_cfg=_cfg("height_scanner_base"),
        )

    def test_flat_ground_is_zero_error(self):
        env = _make_env(torch.tensor([0.30, 0.30]), torch.zeros(2, 9))
        reward = self._reward(env)
        self.assertTrue(torch.allclose(reward, torch.zeros(2), atol=1e-8))

    def test_height_deviation_is_penalized(self):
        env = _make_env(torch.tensor([0.35, 0.25]), torch.zeros(2, 9))
        reward = self._reward(env)
        self.assertTrue(torch.allclose(reward, torch.tensor([0.0025, 0.0025]), atol=1e-8))

    def test_one_env_over_void_does_not_zero_the_others(self):
        """核心回归：0 号环境射线全落空（悬在沟上），1 号环境正常 —— 1 号必须仍被惩罚。"""
        ray_hits = torch.zeros(2, 9)
        ray_hits[0, :] = float("inf")           # 0 号完全悬空
        env = _make_env(torch.tensor([0.30, 0.35]), ray_hits)
        reward = self._reward(env)
        self.assertAlmostEqual(float(reward[0]), 0.0, places=8)          # 悬空环境不罚（原意）
        self.assertAlmostEqual(float(reward[1]), 0.0025, places=8)       # 旧实现下这里会是 0
        self.assertGreater(float(reward[1]), 0.0)

    def test_partial_miss_uses_valid_rays_only(self):
        """部分落空 → 用该环境有效射线的均值（地面 −0.05 m ⇒ 目标 0.25 m），而不是整批归零。

        根高 0.30 m ⇒ 误差 = (0.30 − 0.25)² = 0.0025；旧实现下这里会是 0。
        """
        ray_hits = torch.full((2, 9), -0.05)
        ray_hits[0, :3] = float("nan")          # 3 条未命中
        env = _make_env(torch.tensor([0.30, 0.30]), ray_hits)
        reward = self._reward(env)
        expected = (0.30 - (self.TARGET - 0.05)) ** 2
        self.assertTrue(torch.allclose(reward, torch.full((2,), expected), atol=1e-8))
        self.assertGreater(float(reward[0]), 0.0)

    def test_upright_gating_still_applies(self):
        env = _make_env(torch.tensor([0.35]), torch.zeros(1, 9))
        env.scene["robot"].data.projected_gravity_b = torch.tensor([[0.0, 0.0, 1.0]])  # 翻倒
        self.assertAlmostEqual(float(self._reward(env)[0]), 0.0, places=8)


if __name__ == "__main__":
    unittest.main()
