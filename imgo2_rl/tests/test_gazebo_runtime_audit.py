"""Offline checks for direction commands and URDF inertia auditing."""
import ast
import importlib.util
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("physics_audit", REPO / "imgo2_rl/scripts/tools/audit_gazebo_model_physics.py")
physics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(physics)


class PhysicsTests(unittest.TestCase):
    def test_rotated_principal_inertias_and_invalid_tensor(self):
        eig = physics.eigenvalues3([[1.5, -0.5, 0], [-0.5, 1.5, 0], [0, 0, 3]])
        for actual, expected in zip(eig, (1, 2, 3)):
            self.assertAlmostEqual(actual, expected)
        self.assertLess(min(physics.eigenvalues3([[1, 2, 0], [2, 1, 0], [0, 0, 1]])), 0)

    def test_bad_inertia_fails_and_origin_change_is_detected(self):
        good = '<robot name="test"><link name="base"><inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="1"/><inertia ixx="1" iyy="2" izz="3" ixy="0" ixz="0" iyz="0"/></inertial></link></robot>'
        with tempfile.TemporaryDirectory() as folder:
            train, gazebo = (Path(folder) / n for n in ("train.urdf", "gazebo.urdf"))
            train.write_text(good)
            gazebo.write_text(good)
            self.assertTrue(physics.audit(train, gazebo)["passed"])
            gazebo.write_text(good.replace('rpy="0 0 0"', 'rpy="0 0 1"'))
            self.assertEqual(physics.audit(train, gazebo)["link_differences"][0]["fields"], ["inertial"])
            gazebo.write_text(good.replace('izz="3"', 'izz="4"'))
            self.assertFalse(physics.audit(train, gazebo)["passed"])
            self.assertFalse(physics.audit(train, gazebo)["invalid_inertias"][0]["triangle_inequality"])


class DirectionTests(unittest.TestCase):
    def build_eval(self, vx=0, vy=0, wz=0):
        class Publisher:
            def publish(self, message):
                self.message = message

        class Node:
            def __init__(self, *args):
                pass

            def create_subscription(self, *args):
                pass

            def create_publisher(self, *args):
                return Publisher()

        clock = SimpleNamespace(value=0)
        source = ast.parse((REPO / "imgo2_deploy/scripts/eval_gazebo_policy.py").read_text())
        cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "Eval")
        namespace = {"Node": Node, "QoSProfile": lambda **kwargs: None,
                     "ReliabilityPolicy": SimpleNamespace(BEST_EFFORT=0), "Joy": SimpleNamespace,
                     "JointState": object, "Odometry": object,
                     "time": SimpleNamespace(time=lambda: clock.value)}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), "<actual Eval class>", "exec"), namespace)
        node = namespace["Eval"](vx, key=3, vy=vy, wz=wz)
        return node, clock

    def test_backward_lateral_yaw_commands_preserve_sign_and_getup_zeroes(self):
        node, clock = self.build_eval(-0.5, 0.2, -0.8)
        node.tick()
        self.assertEqual(node.joy_pub.message.axes, [0.0] * 8)
        self.assertEqual(node.joy_pub.message.buttons[0], 1)
        clock.value = 7
        node.tick()
        self.assertEqual(node.joy_pub.message.axes[0:4], [0.2, -0.5, 0.0, -0.8])
        self.assertEqual(node.joy_pub.message.axes[7], -1)
        self.assertEqual(node.joy_pub.message.buttons[5], 1)

    def test_existing_forward_default_still_has_zero_lateral_and_yaw(self):
        node, clock = self.build_eval(0.5)
        clock.value = 7
        node.tick()
        self.assertEqual(node.joy_pub.message.axes[1], 0.5)
        self.assertEqual(node.joy_pub.message.axes[0], 0)
        self.assertEqual(node.joy_pub.message.axes[3], 0)


if __name__ == "__main__":
    unittest.main()
