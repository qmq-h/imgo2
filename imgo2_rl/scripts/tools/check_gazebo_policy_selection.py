#!/usr/bin/env python3
"""Exercise the evaluator's real tick method without ROS; check controller bindings."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

REPO = Path(__file__).resolve().parents[3]
EVAL = REPO / "imgo2_deploy/ros2_ws/scripts/eval_gazebo_policy.py"
CONTROL = REPO / "imgo2_deploy/ros2_ws/src/imgo2_deploy/src/rl_sim.cpp"


class PolicySelectionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse(EVAL.read_text())
        evaluator = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Eval")
        tick = next(n for n in evaluator.body if isinstance(n, ast.FunctionDef) and n.name == "tick")
        namespace = {"time": SimpleNamespace(time=lambda: 100.0)}
        exec(compile(ast.Module(body=[tick], type_ignores=[]), str(EVAL), "exec"), namespace)
        cls.tick = staticmethod(namespace["tick"])

    def run_tick(self, key, getup=False, t0=0.0):
        sent = []
        state = SimpleNamespace(key=key, getup=getup, t0=t0, state="passive",
                                vx=.3, vy=-.1, wz=.2,
                                send_joy=lambda buttons, axes: sent.append((buttons, axes)))
        self.tick(state)
        return state, sent[0]

    def test_three_policies_match_controller_bindings(self):
        source = CONTROL.read_text()
        for key, axis, sign, name in [(1, 7, 1, "Up"), (2, 6, -1, "Right"), (3, 7, -1, "Down")]:
            with self.subTest(key=key):
                state, (buttons, axes) = self.run_tick(key)
                self.assertEqual(state.state, "policy")
                self.assertEqual(buttons[5], 1)
                self.assertEqual(axes[axis], sign)
                other = 6 if axis == 7 else 7
                self.assertEqual(axes[other], 0)
                comparison = ">" if sign > 0 else "<"
                self.assertIn(f"this->joy_msg.axes[{axis}] {comparison} 0) this->control.SetGamepad(Input::Gamepad::RB_DPad{name})", source)
                self.assertEqual([axes[1], axes[0], axes[3]], [.3, -.1, .2])

    def test_getup_precedes_policy(self):
        state, (buttons, axes) = self.run_tick(2, getup=True, t0=98.0)
        self.assertEqual(state.state, "getup")
        self.assertEqual(buttons[0], 1)
        self.assertEqual(axes, [0.0] * 8)


if __name__ == "__main__":
    unittest.main()
