"""`tow_clearance.py` 的离线测试：几何、FK、以及「缺列必须报错」。

这些数都由 `imgo2_description/urdf/imgo2.urdf` 与 `cart/cart.urdf` 的几何决定，
一条命令即可复现：`python3 imgo2_rl/scripts/tools/tow_clearance.py --selftest`。
"""

import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

RL = Path(__file__).resolve().parents[1]
REPO = RL.parent
sys.path.insert(0, str(RL / "scripts/tools"))

import tow_clearance  # noqa: E402


def _module_at(name, path):
    """按路径直接加载 recording.py：走包导入会把 Isaac Lab 的 `omni.*` 一起拉进来。"""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


recording = _module_at("towing_recording_clearance_test",
                       RL / "source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/utils/recording.py")

JOINT_NAMES = ["FL_hip_joint", "FL_thigh_joint", "FL_shank_joint",
               "FR_hip_joint", "FR_thigh_joint", "FR_shank_joint",
               "RL_hip_joint", "RL_thigh_joint", "RL_shank_joint",
               "RR_hip_joint", "RR_thigh_joint", "RR_shank_joint"]
DEFAULT_STANCE = [0.0, 0.87, -1.82] * 4


def _row(time_s, phase, *, robot_x, load_x, joints=DEFAULT_STANCE):
    row = {"phase": phase, "time_s": time_s,
           "robot_x_m": robot_x, "robot_z_m": 0.2735,
           "load_x_m": load_x, "load_z_m": 0.15,
           "robot_quat_x": 0.0, "robot_quat_y": 0.0, "robot_quat_z": 0.0, "robot_quat_w": 1.0,
           "load_quat_x": 0.0, "load_quat_y": 0.0, "load_quat_z": 0.0, "load_quat_w": 1.0,
           "rope_distance_m": (robot_x - 0.16) - (load_x + 0.25)}
    row.update(recording.joint_position_fields(joints))
    return row


def _write_case(directory: Path, rows, joint_names=JOINT_NAMES):
    directory.mkdir(parents=True, exist_ok=True)
    config = {"user_command_mps": 0.5}
    if joint_names is not None:
        config["policy_joint_names"] = joint_names
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    fields = sorted({key for row in rows for key in row})
    with (directory / "tow.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class GeometryTests(unittest.TestCase):
    def test_selftest_passes(self):
        self.assertEqual(tow_clearance.selftest(), 0)

    def test_rear_extent_matches_independent_fk(self):
        """后向极值必须与独立 FK 算得的数一致（文档 §5.17 用的同一组值）。

        这几个数是另一套 numpy FK 扫出来的：默认站姿 −0.3984 m；足端着地（base 0.2735 m）
        约束下后腿伸到最远时 −0.5913 m；收腿时 −0.2698 m。
        """
        robot, _ = tow_clearance.build_bodies(REPO)
        for joints, expected in ((DEFAULT_STANCE, -0.3984),
                                 ([1.16, 0.0, 0.0] * 0 + [0.0, 1.16, -0.52] * 4, -0.5913),
                                 ([0.0, -0.06, -1.50] * 4, -0.2698)):
            points = robot.points(dict(zip(JOINT_NAMES, joints)),
                                  [0.0, 0.0, 0.2735], tow_clearance.IDENTITY)
            self.assertAlmostEqual(min(p[0] for p in points), expected, delta=0.005)


class ScanCaseTests(unittest.TestCase):
    def test_clearance_at_known_attachment_gap(self):
        """挂点间距 0.80 m、默认站姿 ⇒ 车头到机器人后表面 0.5616 m。

        间隙 = 挂点间距 − (|后伸| − 0.16) = 0.80 − (0.3984 − 0.16) = 0.5616。
        车斗前表面与小车挂点同在 base 前 0.25 m，两项相消。
        """
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "case_00"
            _write_case(case, [_row(0.005, "station", robot_x=1.21, load_x=0.0),
                               _row(0.010, "tow", robot_x=1.21, load_x=0.0),
                               _row(1.000, "coast", robot_x=1.21, load_x=0.0)])
            result = tow_clearance.scan_case(case, REPO)
            self.assertAlmostEqual(result["final_gap_x_m"], 0.5616, delta=0.005)
            self.assertAlmostEqual(result["final_rope_distance_m"], 0.80, delta=1e-9)
            self.assertAlmostEqual(result["min_gap_x_coast_m"], 0.5616, delta=0.005)

    def test_retracted_leg_gives_a_larger_clearance(self):
        """收腿（thigh −0.06 / shank −1.50，后伸 0.2698 m）⇒ 间隙 0.80 − 0.1098 = 0.6902。"""
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "case_00"
            retracted = [0.0, -0.06, -1.50] * 4
            _write_case(case, [_row(0.005, "tow", robot_x=1.21, load_x=0.0, joints=retracted)])
            result = tow_clearance.scan_case(case, REPO)
            self.assertAlmostEqual(result["final_gap_x_m"], 0.6902, delta=0.01)

    def test_same_gap_is_clearance_or_contact_depending_on_pose(self):
        """**同一个挂点间距**既可能「还差十几厘米」也可能是「已经压进去」——取决于腿的位形。

        这就是不能用挂点间距判追尾的原因，也是本工具存在的理由：0.3639 m 正是三个
        b=0.008 run 实测发生碰撞时的挂点间距。
        """
        gap_at_impact = 0.3639
        robot_x = gap_at_impact + 0.41          # 挂点间距 = (x_R − 0.16) − (x_L + 0.25)
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "case_00"
            _write_case(case, [_row(0.005, "tow", robot_x=robot_x, load_x=0.0)])
            default = tow_clearance.scan_case(case, REPO)["final_gap_x_m"]
            extended_leg = [0.0, 1.16, -0.52] * 4
            case2 = Path(tmp) / "case_01"
            _write_case(case2, [_row(0.005, "tow", robot_x=robot_x, load_x=0.0,
                                    joints=extended_leg)])
            extended = tow_clearance.scan_case(case2, REPO)["final_gap_x_m"]
            self.assertAlmostEqual(default, 0.3639 + 0.16 - 0.3984, delta=0.005)   # +0.1255 未接触
            self.assertLess(extended, 0.0)                                         # 已经压进去
            self.assertGreater(default - extended, 0.15)

    def test_missing_joint_columns_raise(self):
        """旧 run 没有关节列时必须报错，不能静默把间隙当 0（那会变成假结论）。"""
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "case_00"
            row = _row(0.005, "tow", robot_x=1.21, load_x=0.0)
            for key in list(row):
                if key.startswith("robot_jp_"):
                    row.pop(key)
            _write_case(case, [row])
            with self.assertRaises(KeyError):
                tow_clearance.scan_case(case, REPO)

    def test_missing_joint_names_in_config_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "case_00"
            _write_case(case, [_row(0.005, "tow", robot_x=1.21, load_x=0.0)], joint_names=None)
            with self.assertRaises(KeyError):
                tow_clearance.scan_case(case, REPO)

    def test_cli_runs_on_a_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "case_00"
            _write_case(case, [_row(0.005, "tow", robot_x=1.21, load_x=0.0)])
            done = subprocess.run([sys.executable, str(RL / "scripts/tools/tow_clearance.py"), str(case)],
                                  capture_output=True, text=True, check=False)
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertIn("case_00", done.stdout)


if __name__ == "__main__":
    unittest.main()
