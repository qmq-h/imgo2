"""Rope (unilateral spring-damper) physics tests; standard Python, no simulator.

对应研究计划 P3：虚拟绳力的定义、松弛/张紧边界、力对与力臂力矩。用标准库复现，
另外在 numpy / torch 可用时核对同一套算术在数组后端上一致（不可用则跳过）。
"""

import ast
import importlib.util
import math
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest

RL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RL / "scripts/tools"))


def module_at(name, relative):
    spec = importlib.util.spec_from_file_location(name, RL / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rope = module_at("towing_rope_test",
                 "source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/mdp/rope.py")


class RopeTensionTests(unittest.TestCase):
    """T = 0 (δ ≤ 0) / max(0, kδ + cḋ) (δ > 0)。"""

    def test_slack_rope_carries_no_tension(self):
        for rate in (-1.0, 0.0, 1.0):
            self.assertEqual(rope.rope_tension(0.9, rate, rest_length=1.0,
                                               stiffness=5000.0, damping=50.0), 0.0)

    def test_boundary_at_rest_length_is_zero_even_when_stretching(self):
        """δ = 0 时不连续：计划定义为 δ ≤ 0 ⇒ T = 0，故 ḋ > 0 也不产生阻尼力。"""
        self.assertEqual(rope.rope_tension(1.0, 2.0, rest_length=1.0,
                                           stiffness=5000.0, damping=50.0), 0.0)
        # 略微越过 δ = 0 就立刻按弹簧+阻尼给出张力（同一 ḋ）
        just_taut = rope.rope_tension(1.0 + 1e-9, 2.0, rest_length=1.0,
                                      stiffness=5000.0, damping=50.0)
        self.assertAlmostEqual(just_taut, 5000.0 * 1e-9 + 50.0 * 2.0, places=6)

    def test_taut_tension_is_spring_plus_damper(self):
        tension = rope.rope_tension(1.1, 0.3, rest_length=1.0,
                                    stiffness=400.0, damping=20.0)
        self.assertAlmostEqual(tension, 400.0 * 0.1 + 20.0 * 0.3, places=12)

    def test_pure_damper_when_extension_vanishes_but_positive(self):
        tension = rope.rope_tension(1.0 + 1e-12, 0.5, rest_length=1.0,
                                    stiffness=1000.0, damping=10.0)
        self.assertAlmostEqual(tension, 10.0 * 0.5, places=6)

    def test_rope_cannot_push(self):
        """ḋ 大幅为负时 kδ + cḋ 变负，但单侧约束把它截到 0。"""
        tension = rope.rope_tension(1.001, -100.0, rest_length=1.0,
                                    stiffness=10.0, damping=50.0)
        self.assertEqual(tension, 0.0)
        self.assertGreaterEqual(tension, 0.0)

    def test_tension_is_monotone_in_extension(self):
        values = [rope.rope_tension(1.0 + x, 0.0, rest_length=1.0,
                                    stiffness=300.0, damping=1.0)
                  for x in (0.0, 1e-6, 0.01, 0.05, 0.2)]
        self.assertEqual(values[0], 0.0)
        self.assertTrue(all(b >= a for a, b in zip(values, values[1:])),
                        f"tension must not decrease with extension: {values}")

    def test_zero_rest_length_means_any_separation_is_taut(self):
        self.assertGreater(rope.rope_tension(0.001, 0.0, rest_length=0.0,
                                             stiffness=1000.0, damping=0.0), 0.0)

    def test_extension_sign(self):
        self.assertAlmostEqual(rope.rope_extension(0.8, rest_length=1.0), -0.2, places=12)
        self.assertAlmostEqual(rope.rope_extension(1.3, rest_length=1.0), 0.3, places=12)


class RopeParameterValidationTests(unittest.TestCase):
    """坏参数必须显式报错，而不是静默产生非物理力。"""

    def test_rejects_bad_stiffness(self):
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                rope.rope_tension(1.1, 0.0, rest_length=1.0, stiffness=bad, damping=1.0)

    def test_rejects_bad_damping(self):
        for bad in (-1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                rope.rope_tension(1.1, 0.0, rest_length=1.0, stiffness=1.0, damping=bad)

    def test_rejects_bad_rest_length(self):
        for bad in (-0.5, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                rope.rope_tension(1.1, 0.0, rest_length=bad, stiffness=1.0, damping=1.0)

    def test_rejects_non_numeric_and_boolean_parameters(self):
        for bad in ("1.0", None, True):
            with self.assertRaises(TypeError):
                rope.rope_tension(1.1, 0.0, rest_length=1.0, stiffness=bad, damping=1.0)

    def test_rejects_negative_distance_and_non_finite_rate(self):
        with self.assertRaises(ValueError):
            rope.rope_tension(-0.1, 0.0, rest_length=1.0, stiffness=1.0, damping=1.0)
        with self.assertRaises(ValueError):
            rope.rope_tension(1.1, float("nan"), rest_length=1.0, stiffness=1.0, damping=1.0)

    def test_rejects_malformed_vectors(self):
        with self.assertRaises(ValueError):
            rope.cross((1.0, 2.0), (0.0, 0.0, 1.0))
        with self.assertRaises(ValueError):
            rope.offset_torque((1.0, 2.0), (0.0, 0.0, 1.0))
        with self.assertRaises(ValueError):
            rope.point_velocity((0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            rope.cross((0.0, float("nan"), 0.0), (1.0, 0.0, 0.0))


class RopeWrenchTests(unittest.TestCase):
    """力对、方向与力臂力矩。"""

    def test_slack_pair_is_zero_and_direction_still_defined(self):
        state = rope.rope_state((0.0, 0.0, 0.0), (0.9, 0.0, 0.0),
                                (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                                rest_length=1.0, stiffness=1000.0, damping=10.0)
        self.assertEqual(state.tension, 0.0)
        self.assertEqual(state.force_on_robot, (0.0, 0.0, 0.0))
        self.assertEqual(state.force_on_cart, (0.0, 0.0, 0.0))
        self.assertAlmostEqual(state.direction, (1.0, 0.0, 0.0), places=12)

    def test_taut_pair_is_equal_and_opposite_and_points_to_the_cart(self):
        state = rope.rope_state((0.0, 0.0, 0.0), (1.2, 0.0, 0.0),
                                (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                                rest_length=1.0, stiffness=500.0, damping=0.0)
        self.assertAlmostEqual(state.extension, 0.2, places=12)
        self.assertAlmostEqual(state.tension, 100.0, places=12)
        self.assertAlmostEqual(state.force_on_robot[0], 100.0, places=12)
        self.assertAlmostEqual(state.force_on_cart[0], -100.0, places=12)
        for a, b in zip(state.force_on_robot, state.force_on_cart):
            self.assertAlmostEqual(a, -b, places=12)

    def test_direction_is_unit_and_tension_equals_force_magnitude(self):
        state = rope.rope_state((-1.0, 2.0, 0.5), (2.0, -1.0, 1.5),
                                (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                                rest_length=1.0, stiffness=250.0, damping=5.0)
        self.assertAlmostEqual(math.dist((0.0, 0.0, 0.0), state.direction), 1.0, places=12)
        self.assertAlmostEqual(math.dist((0.0, 0.0, 0.0), state.force_on_robot),
                               state.tension, places=12)

    def test_distance_rate_is_projection_of_relative_velocity(self):
        # 沿绳方向分离 1.0 m/s ⇒ ḋ = +1.0；横向速度不贡献伸长率
        state = rope.rope_state((0.0, 0.0, 0.0), (2.0, 0.0, 0.0),
                                (0.0, 0.0, 0.0), (1.0, 3.0, 0.0),
                                rest_length=1.0, stiffness=100.0, damping=10.0)
        self.assertAlmostEqual(state.distance_rate, 1.0, places=12)
        self.assertAlmostEqual(state.tension, 100.0 * 1.0 + 10.0 * 1.0, places=12)

    def test_rejects_coincident_attachments(self):
        with self.assertRaises(ValueError):
            rope.rope_state((0.5, 0.0, 0.0), (0.5, 0.0, 0.0),
                            (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                            rest_length=1.0, stiffness=100.0, damping=1.0)

    def test_offset_torque_matches_cross_product(self):
        torque = rope.offset_torque((0.0, 3.0, 4.0), (0.25, 0.0, 0.0))
        # (0.25,0,0) × (0,3,4) = (0*4-0*3, 0*0-0.25*4, 0.25*3-0*0) = (0, -1.0, 0.75)
        self.assertAlmostEqual(torque[0], 0.0, places=12)
        self.assertAlmostEqual(torque[1], -1.0, places=12)
        self.assertAlmostEqual(torque[2], 0.75, places=12)

    def test_offset_torque_vanishes_for_parallel_vectors(self):
        torque = rope.offset_torque((0.0, 0.0, 5.0), (0.0, 0.0, 0.25))
        self.assertEqual(torque, (0.0, 0.0, 0.0))

    def test_point_velocity_adds_omega_cross_offset(self):
        # ω = +z、r = +x ⇒ ω×r = +y
        velocity = rope.point_velocity((1.0, 0.0, 0.0), (0.0, 0.0, 2.0), (0.5, 0.0, 0.0))
        self.assertAlmostEqual(velocity[0], 1.0, places=12)
        self.assertAlmostEqual(velocity[1], 1.0, places=12)
        self.assertAlmostEqual(velocity[2], 0.0, places=12)


class RopeConservationTests(unittest.TestCase):
    """力对是内力：整体动量与总角动量必须守恒（数值积分下到舍入误差）。"""

    def test_internal_pair_conserves_total_force_and_torque(self):
        state = rope.rope_state((0.0, 0.0, 0.0), (1.4, 0.3, 0.1),
                                (1.0, 0.0, 0.0), (0.4, 0.0, 0.0),
                                rest_length=1.0, stiffness=800.0, damping=30.0)
        total = [a + b for a, b in zip(state.force_on_robot, state.force_on_cart)]
        for component in total:
            self.assertAlmostEqual(component, 0.0, places=12)

    def test_two_body_tow_conserves_momentum_and_energy(self):
        """两个质点用绳连接、除绳力外无外力：动量守恒，总机械能不增（阻尼只耗散）。

        注意**不要**断言「末态两体同速」：单侧绳在松弛段不施力，相对速度会被冻结，
        欠/过阻尼都会留下持续的 slack ↔ taut 循环。这是模型性质，见
        `docs/towing_p3_rope_2026-09-20.md`；P4 的真实系统里机器人受速度控制器驱动，
        不能由这个自由质点模型外推。
        """
        robot_mass, cart_mass, rest = 5.5339402, 10.0, 1.0
        robot_p, cart_p = [1.05, 0.0, 0.0], [0.0, 0.0, 0.0]   # 机器人在前、小车在后
        robot_v, cart_v = [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]
        dt, stiffness, damping = 1e-4, 4000.0, 100.0
        p0 = robot_mass * robot_v[0] + cart_mass * cart_v[0]

        def total_energy():
            """动能 + **弹性势能**。t=0 绳已伸长 0.05 m ⇒ 势能 5.0 J 是主要项，
            只算动能会把「弹簧释放」误读成能量注入。"""
            kinetic = 0.5 * robot_mass * robot_v[0] ** 2 + 0.5 * cart_mass * cart_v[0] ** 2
            extension = abs(cart_p[0] - robot_p[0]) - rest
            return kinetic + 0.5 * stiffness * max(extension, 0.0) ** 2

        e0 = total_energy()
        peak = e0
        slack_violations = 0
        clamped = 0
        slack_kinetic_drift = 0.0
        previous_kinetic = 0.5 * robot_mass * robot_v[0] ** 2 + 0.5 * cart_mass * cart_v[0] ** 2
        for _ in range(20000):
            state = rope.rope_state(tuple(robot_p), tuple(cart_p), tuple(robot_v), tuple(cart_v),
                                    rest_length=rest, stiffness=stiffness, damping=damping)
            if state.extension <= 0.0 and state.tension != 0.0:
                slack_violations += 1                      # 松弛绝不能有张力
            if state.extension > 0.0 and state.tension == 0.0:
                clamped += 1                               # 张紧但快速接近：被 max(0,·) 截断
            for axis in range(3):
                robot_v[axis] += state.force_on_robot[axis] / robot_mass * dt
                cart_v[axis] += state.force_on_cart[axis] / cart_mass * dt
                robot_p[axis] += robot_v[axis] * dt
                cart_p[axis] += cart_v[axis] * dt
            peak = max(peak, total_energy())
            kinetic = 0.5 * robot_mass * robot_v[0] ** 2 + 0.5 * cart_mass * cart_v[0] ** 2
            if state.tension == 0.0:
                slack_kinetic_drift = max(slack_kinetic_drift, abs(kinetic - previous_kinetic))
            previous_kinetic = kinetic

        p1 = robot_mass * robot_v[0] + cart_mass * cart_v[0]
        self.assertLess(abs(p1 - p0) / abs(p0), 1e-9)       # 力对是内力 ⇒ 动量守恒
        self.assertEqual(slack_violations, 0)              # δ ≤ 0 ⇒ T = 0（单向蕴含）
        self.assertGreater(clamped, 0)                     # 该工况确实触发了单侧截断
        self.assertLessEqual(peak, e0 * (1.0 + 1e-6))      # 阻尼不会注入能量
        self.assertLess(total_energy(), e0)                # 确实发生了耗散
        self.assertLess(slack_kinetic_drift, 1e-12)        # 松弛段不做功：KE 逐帧不变

    def test_slack_interval_freezes_relative_velocity(self):
        """松弛段（T = 0）两体均匀运动 ⇒ 相对速度必须逐帧相同。"""
        robot_mass, cart_mass = 5.5339402, 10.0
        robot_p, cart_p = [1.05, 0.0, 0.0], [0.0, 0.0, 0.0]
        robot_v, cart_v = [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]
        dt = 1e-4
        frozen = []
        for _ in range(20000):
            state = rope.rope_state(tuple(robot_p), tuple(cart_p), tuple(robot_v), tuple(cart_v),
                                    rest_length=1.0, stiffness=4000.0, damping=100.0)
            if state.tension == 0.0 and state.extension < -1e-9:
                frozen.append(cart_v[0] - robot_v[0])
            for axis in range(3):
                robot_v[axis] += state.force_on_robot[axis] / robot_mass * dt
                cart_v[axis] += state.force_on_cart[axis] / cart_mass * dt
                robot_p[axis] += robot_v[axis] * dt
                cart_p[axis] += cart_v[axis] * dt
        self.assertGreater(len(frozen), 100)               # 确有松弛区间可供检验
        span = max(frozen) - min(frozen)
        self.assertLess(span, 1e-9, f"slack 段相对速度漂移了 {span}")


class RopeBackendTests(unittest.TestCase):
    """同一套算术在 numpy / torch 上必须与标量结果一致。"""

    def test_numpy_backend(self):
        try:
            import numpy as np
        except ImportError:  # pragma: no cover - 环境无 numpy 时跳过
            self.skipTest("numpy not available")
        state = rope.rope_state(np.array([0.0, 0.0, 0.0]), np.array([1.2, 0.0, 0.0]),
                                np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
                                rest_length=1.0, stiffness=500.0, damping=10.0)
        self.assertAlmostEqual(float(state.tension), 500.0 * 0.2 + 10.0 * 1.0, places=10)
        self.assertAlmostEqual(float(state.force_on_cart[0]), -110.0, places=10)

    def test_numpy_batches_match_scalars(self):
        try:
            import numpy as np
        except ImportError:  # pragma: no cover
            self.skipTest("numpy not available")
        distances = np.array([0.9, 1.0, 1.1, 1.2])
        tension = rope.rope_tension(distances, 0.0, rest_length=1.0,
                                    stiffness=100.0, damping=1.0)
        expected = [0.0, 0.0, 10.0, 20.0]
        for got, want in zip(tension.tolist(), expected):
            self.assertAlmostEqual(got, want, places=10)

    def test_torch_backend(self):
        try:
            import torch
        except ImportError:  # pragma: no cover - 环境无 torch 时跳过
            self.skipTest("torch not available")
        state = rope.rope_state(torch.tensor([0.0, 0.0, 0.0]), torch.tensor([1.2, 0.0, 0.0]),
                                torch.tensor([0.0, 0.0, 0.0]), torch.tensor([1.0, 0.0, 0.0]),
                                rest_length=1.0, stiffness=500.0, damping=10.0)
        # torch 默认 float32，比较精度按单精度给
        self.assertAlmostEqual(float(state.tension), 110.0, places=3)
        self.assertAlmostEqual(float(state.force_on_cart[0]), -110.0, places=3)
        # 松弛段在张量后端同样被截成 0（全靠掩码与 max(·,0)，没有 Python 分支）
        slack = rope.rope_tension(torch.tensor([0.5, 1.5]), torch.tensor([0.0, 0.0]),
                                  rest_length=1.0, stiffness=100.0, damping=1.0)
        self.assertAlmostEqual(float(slack[0]), 0.0, places=6)
        self.assertAlmostEqual(float(slack[1]), 50.0, places=6)

    def test_torch_parameter_validation_is_still_scalar(self):
        try:
            import torch
        except ImportError:  # pragma: no cover
            self.skipTest("torch not available")
        with self.assertRaises(ValueError):
            rope.rope_tension(torch.tensor([1.2]), 0.0, rest_length=1.0,
                              stiffness=-1.0, damping=1.0)


class MdpPackageContractTests(unittest.TestCase):
    """`towing/mdp` 必须能脱离仿真器导入：物理模块保持纯算术。

    这条契约来自架构记录 §2「导入不得启动仿真器」；离线可复现是 P3 能在没有 GPU 的
    机器上验证的前提。注意 `imgo2_rl.tasks` 包本身会 import isaaclab（需要 Kit 的
    `omni.log`），所以这里按路径构造合成包来导入，绕开父包。
    """

    MDP_RELATIVE = "source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/mdp"

    def test_physics_modules_import_only_stdlib(self):
        allowed = {"__future__", "math", "typing"}
        for name in ("rope.py", "resistance.py"):
            source = (RL / self.MDP_RELATIVE / name).read_text(encoding="utf-8")
            imported = set()
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.Import):
                    imported.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    imported.add(node.module.split(".")[0])
            self.assertTrue(imported <= allowed,
                            f"{name} 引入了非标准库模块: {sorted(imported - allowed)}")

    def test_mdp_package_reexports_without_simulator(self):
        script = textwrap.dedent(f"""
            import importlib.util, sys
            from pathlib import Path
            package = Path({self.MDP_RELATIVE!r}).resolve()
            spec = importlib.util.spec_from_file_location(
                "towing_mdp", package / "__init__.py",
                submodule_search_locations=[str(package)])
            module = importlib.util.module_from_spec(spec)
            sys.modules["towing_mdp"] = module
            spec.loader.exec_module(module)
            print("exports:" + ",".join(sorted(module.__all__)))
            heavy = [m for m in ("torch", "numpy", "isaaclab", "omni") if m in sys.modules]
            print("heavy:" + (",".join(heavy) if heavy else "none"))
            print("tension:" + repr(module.rope_tension(1.2, 0.0, rest_length=1.0,
                                                        stiffness=500.0, damping=10.0)))
        """)
        done = subprocess.run([sys.executable, "-c", script], cwd=str(RL),
                              capture_output=True, text=True, timeout=120)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("heavy:none", done.stdout)
        tension = [line for line in done.stdout.splitlines() if line.startswith("tension:")]
        self.assertEqual(len(tension), 1, done.stdout)
        self.assertAlmostEqual(float(tension[0].split(":", 1)[1]), 100.0, places=6)
        for exported in ("rope_state", "rope_tension", "offset_torque", "point_velocity",
                         "viscous_resistance"):
            self.assertIn(exported, done.stdout)


if __name__ == "__main__":
    unittest.main()
