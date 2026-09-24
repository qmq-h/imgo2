"""`track_world_vel_xy_exp`（世界系速度跟踪）的离线回归 —— 锁定「转身/横移绕开」这个漏洞。

背景（2026-09-24 回放实测）：机体系版本 `track_lin_vel_xy_exp` 配合 `heading_command` 时，
yaw 指令由 heading 控制器**按当前误差实时生成**，而奖励只看机体系速度 ⇒ 机器人可以
"一边转身一边在机体系里前进"，在**世界系里横移/绕行**却仍拿满跟踪奖励。
本文件的核心用例 `test_turned_robot_sideways_is_penalized` 就是这条：机器人朝向 90°、
机体系速度 (0.6, 0)，机体系版本会给满分，世界系版本必须接近 0。

写法同 `test_masked_joint_mirror.py`：`mdp/rewards.py` 顶层 `import isaaclab...`（缺 `omni.log`）
无法整模块 import，所以用 AST 从**真实源文件**里取出该函数的源码，在带桩的命名空间里 exec。
测的仍是仓库里的真代码。
"""

from __future__ import annotations

import ast
import math
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


class _Data:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class _Entity:
    def __init__(self, **data):
        self.data = _Data(**data)


class _CommandTerm:
    def __init__(self, command, heading_target):
        self.command = command
        self.heading_target = heading_target


class _CommandManager:
    def __init__(self, command, heading_target):
        self._term = _CommandTerm(command, heading_target)

    def get_command(self, name):
        return self._term.command

    def get_term(self, name):
        return self._term


class _Env:
    def __init__(self, command, heading_target, world_vel, yaw=0.0, gravity_z=-1.0):
        n = command.shape[0]
        self.num_envs = n
        self.device = "cpu"
        self.command_manager = _CommandManager(command, heading_target)
        quat = torch.zeros(n, 4)
        quat[:, 0] = math.cos(0.5 * yaw)
        quat[:, 3] = math.sin(0.5 * yaw)
        self.scene = {
            "robot": _Entity(
                root_lin_vel_w=world_vel,
                root_quat_w=quat,
                projected_gravity_b=torch.tensor([[0.0, 0.0, gravity_z]] * n),
            )
        }


def _load_track_world_vel(std: float):
    """从真实源文件抽出函数源码，在桩命名空间里 exec。"""
    source = REWARDS.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "track_world_vel_xy_exp"
    )

    class _StubSceneEntityCfg:
        def __init__(self, name="robot", **kwargs):
            self.name = name

    ns = {
        "torch": torch,
        "math": math,
        "math_utils": None,
        "SceneEntityCfg": _StubSceneEntityCfg,
        "ManagerBasedRLEnv": object,
        "RigidObject": object,
    }
    exec(compile(ast.get_source_segment(source, node), str(REWARDS), "exec"), ns)  # noqa: S102
    return lambda env: ns["track_world_vel_xy_exp"](env, std, "base_velocity", _StubSceneEntityCfg("robot"))


@unittest.skipIf(torch is None, f"PyTorch unavailable: {IMPORT_ERROR}")
class TestWorldVelTracking(unittest.TestCase):
    STD = 0.5  # 与配置一致：std = sqrt(0.25)

    def test_target_is_rotated_by_commanded_heading(self):
        # 朝向目标 0°、命令 (0.6, 0)：世界系目标 = (0.6, 0)，机器人世界速度一致 ⇒ 满分
        env = _Env(
            command=torch.tensor([[0.6, 0.0, 0.0]]),
            heading_target=torch.tensor([0.0]),
            world_vel=torch.tensor([[0.6, 0.0]]),
        )
        self.assertAlmostEqual(float(_load_track_world_vel(self.STD)(env)[0]), 1.0, places=5)

    def test_heading_90_rotates_target_to_plus_y(self):
        # 朝向目标 90°（+y）、命令 (0.6, 0)：目标 = (0, 0.6)
        env = _Env(
            command=torch.tensor([[0.6, 0.0, 0.0]]),
            heading_target=torch.tensor([math.pi / 2]),
            world_vel=torch.tensor([[0.0, 0.6]]),
        )
        self.assertAlmostEqual(float(_load_track_world_vel(self.STD)(env)[0]), 1.0, places=5)

    def test_turned_robot_sideways_is_penalized(self):
        """核心回归：障碍列上机器人转身 90°、机体系"前进"，世界系里却在横移 ⇒ 必须被罚。

        机体系版本对这一帧给的是 exp(0) = 1.0（这就是"横移绕开"的来源）；
        世界系版本给的是 exp(-(0.6²+0.6²)/0.25) ≈ 0.056。
        """
        env = _Env(
            command=torch.tensor([[0.6, 0.0, 0.0]]),
            heading_target=torch.tensor([0.0]),  # 障碍列：目标朝向固定 +x
            world_vel=torch.tensor([[0.0, 0.6]]),  # 但实际在世界系里往 +y 走
            yaw=math.pi / 2,
        )
        reward = float(_load_track_world_vel(self.STD)(env)[0])
        self.assertLess(reward, 0.1, f"世界系版本仍放过了横移：{reward}")

    def test_backwards_world_velocity_is_penalized(self):
        env = _Env(
            command=torch.tensor([[0.6, 0.0, 0.0]]),
            heading_target=torch.tensor([0.0]),
            world_vel=torch.tensor([[-0.6, 0.0]]),
        )
        reward = float(_load_track_world_vel(self.STD)(env)[0])
        # 误差 = (0.6 − (−0.6))² = 1.44 ⇒ exp(−1.44/0.25) = 0.00315
        self.assertLess(reward, 0.01)
        self.assertAlmostEqual(reward, math.exp(-1.44 / self.STD**2), places=6)

    def test_tilt_gate_zeroes_upside_down(self):
        env = _Env(
            command=torch.tensor([[0.6, 0.0, 0.0]]),
            heading_target=torch.tensor([0.0]),
            world_vel=torch.tensor([[0.6, 0.0]]),
            gravity_z=0.0,
        )
        self.assertAlmostEqual(float(_load_track_world_vel(self.STD)(env)[0]), 0.0, places=6)


class TestCmoeCfgWiring(unittest.TestCase):
    """确认配置里世界系版本已启用、机体系版本已归零、两项软约束只作用于障碍列。"""

    @classmethod
    def setUpClass(cls):
        cls.src = (
            ROOT / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/base_move/CMoE_env_cfg.py"
        ).read_text(encoding="utf-8-sig")

    def test_world_vel_enabled_and_body_vel_disabled(self):
        self.assertIn("self.rewards.track_world_vel_xy_exp.weight = 5.0", self.src)
        self.assertIn("self.rewards.track_lin_vel_xy_exp.weight = 0.0", self.src)

    def test_parkour_soft_terms_are_global(self):
        """2026-09-24 用户："所有场景……都给脱离中心的惩罚" ⇒ `terrain_names=()`（空＝全局）。"""
        self.assertIn("self.rewards.lin_pos_y.weight = -0.4", self.src)
        self.assertIn("self.rewards.yaw_abs.weight = -0.2", self.src)
        self.assertIn('self.rewards.lin_pos_y.params["terrain_names"] = ()', self.src)
        self.assertIn('self.rewards.yaw_abs.params["terrain_names"] = ()', self.src)

    def test_every_terrain_is_forward_only(self):
        """所有地形列都要是"只给超前速度"：名单必须覆盖 sub_terrains 的全部键。"""
        import re
        block = re.search(
            r"self\.commands\.base_velocity\.forward_only_terrain_names = \((.*?)\)", self.src, re.S
        )
        self.assertIsNotNone(block, "配置里没有设置 forward_only_terrain_names")
        listed = set(re.findall(r'"([a-z_0-9]+)"', block.group(1)))
        expected = {"pyramid_stairs", "pyramid_stairs_inv", "boxes", "random_rough",
                    "hf_pyramid_slope", "hf_pyramid_slope_inv", "gap", "flat"}
        self.assertEqual(listed, expected,
                         f"forward_only 名单缺 {sorted(expected - listed)}／多 {sorted(listed - expected)}")


if __name__ == "__main__":
    sys.exit(unittest.main())
