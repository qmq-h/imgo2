"""`play_towing_test.py` 的离线测试：判据与网格逻辑不依赖 Isaac Sim。

本机（以及任何没有 GPU/Isaac Lab 的机器）只能验证**纯逻辑**部分：网格生成、坡度→重力、
阶段划分、指令整形、五项指标的计算与判定、报告与矩阵、CLI 校验、记录字段契约。
仿真相位（PhysX 步进、绳力施加、重力写入）不在覆盖范围内 —— 那部分只能在训练机实跑，
不要在文档里把本文件的通过写成「测试台已验证」。
"""

import ast
import importlib.util
import math
import sys
from pathlib import Path
import unittest

RL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RL / "scripts/towing"))
sys.path.insert(0, str(RL / "scripts/tools"))


def _module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # 必须先注册进 sys.modules：模块里有 `@dataclass` + `from __future__ import annotations`，
    # dataclass 处理字符串注解时会去 sys.modules[cls.__module__] 找命名空间。
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


play = _module_at("play_towing_test_under_test", RL / "scripts/towing/play_towing_test.py")
connection_grid = play.connection_grid
recording = _module_at(
    "play_test_recording",
    RL / "source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/utils/recording.py")


def load_rope_model_module():
    """`mdp/rope_model.py` 需要 torch；没有就跳过（与仓库其它 towing 测试同一约定）。

    按文件路径加载时没有包上下文，`rope_model.py` 会退回 `from rope import ...`，
    所以要把 `mdp/` 自己加进 sys.path。
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        return None
    mdp_dir = RL / "source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/mdp"
    if str(mdp_dir) not in sys.path:
        sys.path.insert(0, str(mdp_dir))
    return _module_at("play_test_rope_model", mdp_dir / "rope_model.py")


def base_row(**overrides):
    """一条字段齐全（全是 TEST_FIELDS）的合成记录行，默认数值全 0、字符串列有值。"""
    row = {name: 0.0 for name in play.TEST_FIELDS}
    row["phase"] = "tow"
    row["connection"] = "compliant"
    row["slope_backend"] = "gravity"
    row["time_s"] = 0.005
    return row | overrides


def joint_columns(error_by_joint, *, target=0.0, torque=0.0):
    """按逐关节误差生成 jp/jt/tau 三组列。"""
    columns = {}
    for joint in range(12):
        error = error_by_joint[joint] if isinstance(error_by_joint, (list, tuple)) else error_by_joint
        columns[f"robot_jp_{joint:02d}"] = target + error
        columns[f"robot_jt_{joint:02d}"] = target
        columns[f"robot_tau_{joint:02d}"] = torque
    return columns


def build_episode(*, command=1.0, dt=0.005, station_steps=20, tow_steps=100, coast_steps=500,
                  startup_error=0.02, coast_error=0.02, deck_fx_coast=0.0, deck_fx_tow=0.0,
                  load_speed=1.0, load_travel=0.6, gap_start=0.8, gap_closed=0.05,
                  robot_height=0.27, pitch=0.0, robot_vx_b=None, torque=0.0):
    """合成三段轨迹：station（静止）→ tow（跟速）→ coast（小车黏性滑行 + 追尾见证）。

    滑行段用 `v(t) = v0·exp(−t/τ)`、`τ = 滑行距离 / v0`（与 P2 的黏性衰减解析式同形）：
    这样「滑行总距离 ≈ load_travel」「速度降到 0.02 m/s 的时刻」都是解析可算的。
    """
    rows = []

    def add(phase, index, step_time, **overrides):
        row = base_row(phase=phase, time_s=(index + 1) * dt, user_cmd_mps=command,
                       ref_cmd_mps=command, velocity_cmd_mps=command,
                       slope_deg=0.0, gravity_x_mps2=0.0, gravity_z_mps2=-9.81,
                       slope_backend="gravity", cart_mass_kg=10.0,
                       robot_z_m=robot_height, body_pitch_rad=pitch,
                       body_pitch_rel_rad=pitch, robot_surface_height_m=robot_height,
                       load_surface_height_m=0.15, rope_taut=1.0,
                       velocity_cmd_vy_mps=0.0, velocity_cmd_wz_radps=0.0,
                       lane_offset_m=0.0, lane_heading_rad=0.0,
                       robot_vy_b_mps=0.0, robot_wz_b_radps=0.0, load_offset_m=0.0)
        row.update(overrides)
        rows.append(row)

    for index in range(station_steps):
        add("station", index, index * dt, robot_vx_b_mps=0.0, robot_vx_mps=0.0,
            load_vx_mps=0.0, load_x_m=-0.81, load_progress_m=-0.81, robot_x_m=0.0, robot_progress_m=0.0,
            rope_distance_m=gap_start,
            cart_deck_fx_n=0.0, **joint_columns(startup_error, torque=torque))
    for index in range(tow_steps):
        speed = command if robot_vx_b is None else robot_vx_b[index]
        error = startup_error if index < 10 else coast_error
        add("tow", index, index * dt, robot_vx_b_mps=speed, robot_vx_mps=speed,
            load_vx_mps=speed, load_x_m=-0.81 + load_speed * index * dt,
            load_progress_m=-0.81 + load_speed * index * dt,
            robot_x_m=speed * index * dt, robot_progress_m=speed * index * dt,
            rope_distance_m=gap_start,
            cart_deck_fx_n=deck_fx_tow, **joint_columns(error, torque=torque))
    decay_tau = load_travel / max(1e-9, load_speed)
    for index in range(coast_steps):
        elapsed = index * dt
        step_speed = load_speed * math.exp(-elapsed / decay_tau)
        travelled = load_speed * decay_tau * (1.0 - math.exp(-elapsed / decay_tau))
        fraction = min(1.0, travelled / max(1e-9, load_travel))
        add("coast", index, index * dt, robot_vx_b_mps=0.0, robot_vx_mps=0.0,
            load_vx_mps=step_speed, load_x_m=-0.81 + 0.5 + travelled,
            load_progress_m=-0.81 + 0.5 + travelled,
            robot_x_m=0.5, robot_progress_m=0.5,
            rope_distance_m=gap_start - gap_closed * fraction,
            cart_deck_fx_n=deck_fx_coast, **joint_columns(coast_error, torque=torque))
    return rows


def schedules():
    return play.make_schedule(settle_steps=20, tow_duration=0.5, coast_duration=0.5, dt=0.005)


class SlopeTests(unittest.TestCase):
    def test_flat_slope_is_nominal_gravity(self):
        self.assertEqual(play.slope_gravity(0.0), (0.0, 0.0, -9.81))
        self.assertAlmostEqual(play.slope_gravity(0.0)[2], -play.GRAVITY_MPS2)

    def test_uphill_tilts_gravity_towards_minus_x(self):
        gx, gy, gz = play.slope_gravity(10.0)
        self.assertLess(gx, 0.0)                      # +x 上坡 ⇒ 重力有 −x 分量
        self.assertEqual(gy, 0.0)
        self.assertLess(gz, 0.0)
        self.assertAlmostEqual(math.sqrt(gx * gx + gz * gz), play.GRAVITY_MPS2, places=9)

    def test_downhill_is_mirror(self):
        uphill = play.slope_gravity(5.0)
        downhill = play.slope_gravity(-5.0)
        self.assertAlmostEqual(uphill[0], -downhill[0], places=12)
        self.assertAlmostEqual(uphill[2], downhill[2], places=12)

    def test_slope_magnitude_is_angle_invariant(self):
        for slope in (-20.0, -3.0, 0.0, 7.5, 30.0):
            gx, _, gz = play.slope_gravity(slope)
            self.assertAlmostEqual(math.hypot(gx, gz), play.GRAVITY_MPS2, places=9)

    def test_invalid_slope_rejected(self):
        for slope in (math.nan, math.inf, 46.0, -46.0):
            with self.assertRaises(ValueError):
                play.slope_gravity(slope)

    def test_plan_note_and_terrain_backend(self):
        plan = play.slope_ground_plan(5.0, "gravity")
        self.assertEqual(plan["backend"], "gravity")
        self.assertIn("旋转重力", plan["note"])
        self.assertEqual(plan["gravity_mps2"], play.slope_gravity(5.0))
        terrain = play.slope_ground_plan(5.0, "terrain")
        self.assertEqual(terrain["backend"], "terrain")
        # terrain 后端用**世界竖直**重力，坡度由真实 mesh 提供
        self.assertEqual(terrain["gravity_mps2"], (0.0, 0.0, -play.GRAVITY_MPS2))
        self.assertIn("坡面", terrain["note"])
        self.assertEqual(terrain["frame"]["tangent"], terrain["frame"]["tangent"])
        with self.assertRaises(ValueError):
            play.slope_ground_plan(5.0, "nope")


class GridTests(unittest.TestCase):
    def test_default_grid_is_complete(self):
        cases = play.build_case_grid(play.DEFAULT_VELOCITIES, play.CONNECTIONS,
                                     play.DEFAULT_CART_MASSES, play.DEFAULT_SLOPES_DEG)
        self.assertEqual(len(cases), 3 * 3 * 5 * 5)
        self.assertEqual(len(play.cases_for_slope(cases, 0.0)), 45)
        self.assertEqual(len(play.cases_for_slope(cases, 10.0)), 45)

    def test_grid_order_is_slope_velocity_connection_mass(self):
        cases = play.build_case_grid((0.5, 1.0), ("compliant", "rigid"), (5.0, 25.0), (0.0, 5.0))
        self.assertEqual([case.slope_deg for case in cases],
                         [0.0] * 8 + [5.0] * 8)
        self.assertEqual([case.cart_mass_kg for case in cases][:4], [5.0, 25.0, 5.0, 25.0])
        self.assertEqual([case.connection for case in cases][:2], ["compliant", "compliant"])
        self.assertEqual(cases[0].slug, "slope+0_v0.5_compliant_m5kg")
        self.assertEqual(cases[-1].slug, "slope+5_v1_rigid_m25kg")

    def test_slug_is_filename_safe(self):
        for case in play.build_case_grid((0.5, 1.5), ("inextensible",), (5.0,), (-10.0, 5.0)):
            self.assertNotIn("/", case.slug)

    def test_connections_match_rope_model_module(self):
        module = load_rope_model_module()
        if module is None:
            self.skipTest("需要 torch 才能 import mdp/rope_model.py")
        self.assertEqual(tuple(play.CONNECTIONS), tuple(module.CONNECTION_MODELS))
        self.assertIn(module.RIGID_MODEL, play.CONNECTIONS)


class ScheduleTests(unittest.TestCase):
    def test_phase_boundaries(self):
        schedule = play.make_schedule(settle_steps=10, tow_duration=0.05, coast_duration=0.05,
                                      dt=0.005)
        self.assertEqual(schedule.total_steps, 30)
        self.assertEqual([schedule.phase_of(step) for step in (0, 9, 10, 19, 20, 29)],
                         ["station"] * 2 + ["tow"] * 2 + ["coast"] * 2)
        self.assertEqual([schedule.step_in_phase(step) for step in (0, 9, 10, 11, 20, 21)],
                         [0, 9, 0, 1, 0, 1])

    def test_non_multiple_duration_rejected(self):
        with self.assertRaises(ValueError):
            play.make_schedule(settle_steps=0, tow_duration=0.0501, coast_duration=0.05, dt=0.005)

    def test_negative_settle_rejected(self):
        with self.assertRaises(ValueError):
            play.make_schedule(settle_steps=-1, tow_duration=0.5, coast_duration=0.5, dt=0.005)

    def test_command_shaping(self):
        self.assertEqual(play.shaped_command(phase="station", step_in_phase=0, velocity=1.0), 0.0)
        self.assertEqual(play.shaped_command(phase="coast", step_in_phase=3, velocity=1.0), 0.0)
        self.assertEqual(play.shaped_command(phase="tow", step_in_phase=0, velocity=1.5), 1.5)
        ramp = play.shaped_command(phase="tow", step_in_phase=99, velocity=1.0,
                                   shaping="ramp", ramp_time_s=1.0, dt=0.005)
        self.assertAlmostEqual(ramp, 0.5, places=9)          # 第 100 步 = 0.5 s
        self.assertEqual(play.shaped_command(phase="tow", step_in_phase=500, velocity=1.0,
                                             shaping="ramp", ramp_time_s=1.0, dt=0.005), 1.0)
        with self.assertRaises(ValueError):
            play.shaped_command(phase="tow", step_in_phase=0, velocity=1.0, shaping="bang")

    def test_min_env_spacing_scales_with_speed(self):
        slow = play.min_env_spacing(0.5, tow_duration=5.0, coast_duration=5.0)
        fast = play.min_env_spacing(1.5, tow_duration=5.0, coast_duration=5.0)
        self.assertGreater(fast, slow)
        self.assertGreater(play.min_env_spacing(1.5, tow_duration=5.0, coast_duration=5.0),
                           1.5 * 5.0 + 2.0 - 1e-9)


class FieldContractTests(unittest.TestCase):
    def test_recording_columns_are_a_subset(self):
        self.assertTrue(set(recording.TOW_FIELDS) <= set(play.TEST_FIELDS),
                        "自定义记录必须包含 summarize_tow 需要的全部列")

    def test_no_duplicate_columns(self):
        self.assertEqual(len(play.TEST_FIELDS), len(set(play.TEST_FIELDS)))

    def test_every_row_builder_field_is_declared(self):
        row = base_row()
        self.assertEqual(set(row), set(play.TEST_FIELDS))

    def test_joint_columns_cover_all_joints(self):
        for index in range(12):
            self.assertIn(f"robot_jp_{index:02d}", play.TEST_FIELDS)
            self.assertIn(f"robot_jt_{index:02d}", play.TEST_FIELDS)
            self.assertIn(f"robot_tau_{index:02d}", play.TEST_FIELDS)


class RowContractTests(unittest.TestCase):
    """记录列契约：``make_row`` 每步写的字段集必须与 ``TEST_FIELDS`` **完全一致**。

    ``make_row`` 只有跑仿真才会执行，本机跑不了，所以用 AST 抽出它字典字面量的键，
    再把两个「生成列名」的纯函数实际调用一次合起来比对：加列/漏列在离线就暴露，
    不用烧一次 Isaac Sim 才在 ``TowRecorder.append`` 那里报错。
    """

    def test_make_row_field_set_matches_contract(self):
        source = (RL / "scripts/towing/play_towing_test.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(node for node in ast.walk(tree)
                        if isinstance(node, ast.FunctionDef) and node.name == "make_row")
        keys = set()
        for node in ast.walk(function):
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                                                    and target.id == "row"
                                                    for target in node.targets):
                if isinstance(node.value, ast.Dict):
                    keys |= {key.value for key in node.value.keys
                             if isinstance(key, ast.Constant)}
        self.assertTrue(keys, "没有从 make_row 的字典字面量里读到任何键")
        generated = set(play.wheel_omega_fields([0.0, 0.0, 0.0, 0.0], [0, 1, 2, 3]))
        generated |= set(play.joint_state_fields(pos=[0.0] * 12, target=[0.0] * 12,
                                                 torque=[0.0] * 12))
        self.assertEqual(keys | generated, set(play.TEST_FIELDS))

    def test_helper_columns_use_recording_conventions(self):
        fields = play.joint_state_fields(pos=list(range(12)), target=[1.0] * 12,
                                         torque=[2.0] * 12)
        for index, name in enumerate(recording.ROBOT_JOINT_POSITION_FIELDS):
            self.assertEqual(fields[name], float(index))
            self.assertEqual(fields[f"robot_jt_{index:02d}"], 1.0)
            self.assertEqual(fields[f"robot_tau_{index:02d}"], 2.0)

    def test_wheel_columns_follow_joint_ids(self):
        fields = play.wheel_omega_fields([10.0, 20.0, 30.0, 40.0], [3, 2, 1, 0])
        self.assertEqual(fields["wheel_fl_omega_radps"], 40.0)
        self.assertEqual(fields["wheel_rr_omega_radps"], 10.0)


class JointStatsTests(unittest.TestCase):
    def test_rms_and_max_and_worst_joint(self):
        names = [f"j{index}" for index in range(12)]
        errors = [0.1] * 11 + [0.5]
        rows = [base_row(**joint_columns(errors)) for _ in range(4)]
        stats = play.joint_track_stats(rows, joint_names=names, torque_limits=[23.7] * 12)
        expected_rms = math.sqrt((11 * 0.1 ** 2 + 0.5 ** 2) / 12)
        self.assertAlmostEqual(stats["joint_rms_rad"], expected_rms, places=12)
        self.assertAlmostEqual(stats["joint_max_rad"], 0.5, places=12)
        self.assertEqual(stats["worst_joint"], "j11")
        self.assertEqual(stats["samples"], 4)

    def test_torque_saturation_fraction(self):
        rows = [base_row(**joint_columns(0.01, torque=23.0)),   # 23.0 > 0.95*23.7
                base_row(**joint_columns(0.01, torque=1.0))]
        stats = play.joint_track_stats(rows, torque_limits=[23.7] * 12)
        self.assertAlmostEqual(stats["torque_saturated_frac"], 0.5, places=12)

    def test_empty_window_is_reported_not_crashed(self):
        stats = play.joint_track_stats([])
        self.assertFalse(stats["available"])
        self.assertNotIn("joint_rms_rad", stats)

    def test_joint_name_count_is_validated(self):
        with self.assertRaises(ValueError):
            play.joint_track_stats([base_row()], joint_names=["a", "b"])


class SpeedStatsTests(unittest.TestCase):
    def test_known_error_series(self):
        rows = []
        for index, speed in enumerate((0.0, 0.5, 1.0, 1.0)):
            rows.append(base_row(time_s=(index + 1) * 0.005, robot_vx_b_mps=speed))
        stats = play.speed_track_stats(rows, 1.0, steady_fraction=0.5)
        # 误差 [−1, −0.5, 0, 0] ⇒ MAE 0.375，RMSE sqrt(1.25/4)=0.559
        self.assertAlmostEqual(stats["mae_mps"], 0.375, places=12)
        self.assertAlmostEqual(stats["rmse_mps"], math.sqrt(1.25 / 4), places=12)
        self.assertAlmostEqual(stats["bias_mps"], -0.375, places=12)
        self.assertAlmostEqual(stats["ratio_mean"], 0.625, places=12)
        self.assertAlmostEqual(stats["frac_within_10pct"], 0.5, places=12)
        # 稳态窗 = 后一半（[0,0] 误差）⇒ ratio 1.0
        self.assertAlmostEqual(stats["steady_ratio_mean"], 1.0, places=12)

    def test_command_must_be_positive(self):
        with self.assertRaises(ValueError):
            play.speed_track_stats([base_row(robot_vx_b_mps=1.0)], 0.0)

    def test_p95_and_max(self):
        rows = [base_row(robot_vx_b_mps=value) for value in (1.0, 1.1, 1.2, 1.4)]
        stats = play.speed_track_stats(rows, 1.0, steady_fraction=1.0)
        self.assertAlmostEqual(stats["max_abs_err_mps"], 0.4, places=12)
        self.assertGreaterEqual(stats["p95_abs_err_mps"], 0.2)


class LaneKeepingTests(unittest.TestCase):
    """横向/朝向 PD：把机器人压回 lane 中线（y=0）并保持超前（yaw=0）；vx 不参与。"""

    def _pd(self, **overrides):
        kwargs = dict(kp_y=1.0, kd_y=0.3, kp_yaw=1.5, kd_yaw=0.3,
                      vy_limit=0.4, wz_limit=0.8, lane_y=0.0, lane_yaw=0.0,
                      body_vy=0.0, body_wz=0.0)
        kwargs.update(overrides)
        return play.lane_keeping_command(**kwargs)

    def test_zero_error_gives_zero_correction(self):
        self.assertEqual(self._pd(), (0.0, 0.0))

    def test_lateral_error_pushes_back_to_the_centreline(self):
        vy_left_of_centre, _ = self._pd(lane_y=0.2)          # 在中线左侧 ⇒ 往右（vy<0）
        self.assertAlmostEqual(vy_left_of_centre, -0.2, places=9)
        vy_right_of_centre, _ = self._pd(lane_y=-0.2)        # 在中线右侧 ⇒ 往左（vy>0）
        self.assertAlmostEqual(vy_right_of_centre, 0.2, places=9)

    def test_heading_error_turns_back_towards_forward(self):
        _, wz_pointing_left = self._pd(lane_yaw=0.2)         # 朝左偏 ⇒ 往右转（wz<0）
        self.assertAlmostEqual(wz_pointing_left, -0.3, places=9)
        _, wz_pointing_right = self._pd(lane_yaw=-0.2)
        self.assertAlmostEqual(wz_pointing_right, 0.3, places=9)

    def test_heading_error_wraps_across_pi(self):
        # 误差在 ±π 附近不能跳变：yaw = π−0.1、目标 0 ⇒ 误差 −(π−0.1) ≈ −3.04（不是 +3.24）
        # 放开 wz 限幅以便看原始值（默认 0.8 rad/s 会把它夹住）
        _, wz = self._pd(lane_yaw=math.pi - 0.1, body_wz=0.0, wz_limit=10.0)
        self.assertLess(wz, 0.0)
        self.assertAlmostEqual(wz, 1.5 * (-(math.pi - 0.1)), places=6)
        self.assertAlmostEqual(play.wrap_to_pi(math.pi + 0.1), -math.pi + 0.1, places=9)
        self.assertAlmostEqual(play.wrap_to_pi(-math.pi - 0.1), math.pi - 0.1, places=9)

    def test_damping_uses_body_rates(self):
        vy, wz = self._pd(body_vy=0.5, body_wz=0.5)
        self.assertAlmostEqual(vy, -0.15, places=9)
        self.assertAlmostEqual(wz, -0.15, places=9)

    def test_commands_are_clamped_to_the_training_range(self):
        vy, _ = self._pd(lane_y=5.0)                         # 横向误差 5 m ⇒ P 项 −5 m/s
        self.assertAlmostEqual(vy, -0.4, places=9)           # vy 限幅
        _, wz = self._pd(lane_yaw=math.pi / 2)               # 朝向误差 π/2 ⇒ P 项 −2.36 rad/s
        self.assertAlmostEqual(wz, -0.8, places=9)           # wz 限幅

    def test_lateral_error_uses_the_body_lateral_axis(self):
        # 朝向偏 90° 时「机体系横向」与 lane 横向正交 ⇒ 误差投影为 0
        vy, _ = self._pd(lane_y=0.3, lane_yaw=math.pi / 2)
        self.assertAlmostEqual(vy, 0.0, places=9)

    def test_invalid_arguments_rejected(self):
        for bad in (float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                self._pd(lane_y=bad)
        for bad in (0.0, -1.0):
            with self.assertRaises(ValueError):
                self._pd(vy_limit=bad)


class LaneStatsTests(unittest.TestCase):
    def test_reports_offset_heading_and_saturation(self):
        rows = [base_row(lane_offset_m=0.1, lane_heading_rad=0.05,
                         velocity_cmd_vy_mps=-0.4, velocity_cmd_wz_radps=0.0),
                base_row(lane_offset_m=-0.3, lane_heading_rad=-0.15,
                         velocity_cmd_vy_mps=0.1, velocity_cmd_wz_radps=0.8)]
        stats = play.lane_stats(rows, vy_limit=0.4, wz_limit=0.8)
        self.assertAlmostEqual(stats["y_max_abs_m"], 0.3, places=12)
        self.assertAlmostEqual(stats["heading_max_abs_rad"], 0.15, places=12)
        self.assertAlmostEqual(stats["vy_saturated_frac"], 0.5, places=12)
        self.assertAlmostEqual(stats["wz_saturated_frac"], 0.5, places=12)
        self.assertAlmostEqual(stats["y_rms_m"], math.sqrt((0.1**2 + 0.3**2) / 2), places=12)

    def test_lane_deviation_is_flagged_only_when_enabled(self):
        rows = build_episode()
        for row in rows:
            row["lane_offset_m"] = 0.5
        metrics = play.compute_case_metrics(
            rows, command_mps=1.0, slope_deg=0.0, connection="compliant", cart_mass_kg=10.0,
            schedule=schedules(), record_dt=0.005,
            tow_summary={"valid": True, "failures": [], "min_clearance_coast_m": 0.3},
            thresholds=play.DEFAULT_THRESHOLDS, transition_window_s=0.05,
            lane_keeping="pd", lane_vy_limit=0.4, lane_wz_limit=0.8)
        self.assertIn("lane_deviation", metrics["verdict"]["reasons"])
        self.assertEqual(metrics["verdict"]["code"], "LAT")
        metrics_off = play.compute_case_metrics(
            rows, command_mps=1.0, slope_deg=0.0, connection="compliant", cart_mass_kg=10.0,
            schedule=schedules(), record_dt=0.005,
            tow_summary={"valid": True, "failures": [], "min_clearance_coast_m": 0.3},
            thresholds=play.DEFAULT_THRESHOLDS, transition_window_s=0.05, lane_keeping="off")
        self.assertNotIn("lane_deviation", metrics_off["verdict"]["reasons"])


class ContactWitnessTests(unittest.TestCase):
    def test_deck_force_triggers(self):
        rows = build_episode(deck_fx_coast=12.0)
        coast = play.phase_rows(rows, "coast")
        witness = play.contact_witness(coast, record_dt=0.005)
        self.assertTrue(witness["contact"])
        self.assertIn("deck_contact_force", witness["channels"])

    def test_clean_run_has_no_witness(self):
        rows = build_episode()
        coast = play.phase_rows(rows, "coast")
        witness = play.contact_witness(coast, record_dt=0.005)
        self.assertFalse(witness["contact"])
        self.assertEqual(witness["channels"], [])

    def test_velocity_jump_limit_scales_with_record_step(self):
        # 10 ms 记录时，正常滑行（0.0087 m/s / 5 ms ⇒ ~0.0174 / 10 ms）不应被误判为撞击
        rows = []
        for index in range(10):
            rows.append(base_row(time_s=(index + 1) * 0.01, load_vx_mps=1.0 - 0.0087 * 2 * index,
                                 cart_deck_fx_n=0.0))
        coarse = play.contact_witness(rows, record_dt=0.01)
        fine = play.contact_witness(rows, record_dt=0.005)
        self.assertFalse(coarse["contact"])
        self.assertAlmostEqual(coarse["load_dv_limit_mps"], 0.030, places=12)
        self.assertAlmostEqual(fine["load_dv_limit_mps"], 0.015, places=12)


class EpisodeMetricTests(unittest.TestCase):
    def test_full_episode_metrics_are_present(self):
        rows = build_episode(command=1.0, load_travel=0.6)
        tow_summary = {"valid": True, "failures": [], "min_clearance_coast_m": 0.25,
                       "time_to_contact_after_stop_s": None, "clearance_at_stop_m": 0.35,
                       "final_clearance_m": 0.28}
        metrics = play.compute_case_metrics(
            rows, command_mps=1.0, slope_deg=0.0, connection="compliant", cart_mass_kg=10.0,
            schedule=schedules(), record_dt=0.005, tow_summary=tow_summary,
            thresholds=play.DEFAULT_THRESHOLDS, joint_names=[f"j{i}" for i in range(12)],
            torque_limits=[23.7] * 12, transition_window_s=0.05)
        self.assertEqual(metrics["verdict"]["code"], "OK")
        self.assertAlmostEqual(metrics["startup"]["joint_rms_rad"], 0.02, places=12)
        self.assertAlmostEqual(metrics["stop"]["joint_rms_rad"], 0.02, places=12)
        self.assertAlmostEqual(metrics["speed"]["mae_mps"], 0.0, places=9)
        self.assertAlmostEqual(metrics["stop"]["cart_coast_distance_m"],
                               metrics["stop"]["cart_coast_distance_m"])
        self.assertGreater(metrics["stop"]["cart_coast_distance_m"], 0.0)
        self.assertLess(metrics["stop"]["cart_coast_distance_m"], 0.7)
        self.assertIsNotNone(metrics["stop"]["cart_coast_time_to_rest_s"])
        self.assertFalse(metrics["stability"]["fell"])
        self.assertEqual(metrics["samples"]["station"], 20)

    def test_slow_startup_raises_joint_flag(self):
        rows = build_episode(startup_error=0.4)
        tow_summary = {"valid": True, "failures": [], "min_clearance_coast_m": 0.3}
        metrics = play.compute_case_metrics(
            rows, command_mps=1.0, slope_deg=0.0, connection="compliant", cart_mass_kg=10.0,
            schedule=schedules(), record_dt=0.005, tow_summary=tow_summary,
            thresholds=play.DEFAULT_THRESHOLDS, transition_window_s=0.05)
        self.assertEqual(metrics["verdict"]["code"], "JNT")
        self.assertIn("startup_joint_error", metrics["verdict"]["reasons"])

    def test_speed_shortfall_is_flagged(self):
        rows = build_episode(robot_vx_b=[0.5] * 100)
        metrics = play.compute_case_metrics(
            rows, command_mps=1.0, slope_deg=0.0, connection="compliant", cart_mass_kg=10.0,
            schedule=schedules(), record_dt=0.005,
            tow_summary={"valid": True, "failures": [], "min_clearance_coast_m": 0.3},
            thresholds=play.DEFAULT_THRESHOLDS, transition_window_s=0.05)
        self.assertIn("speed_track_error", metrics["verdict"]["reasons"])

    def test_coast_collision_is_flagged(self):
        rows = build_episode(deck_fx_coast=15.0)
        metrics = play.compute_case_metrics(
            rows, command_mps=1.0, slope_deg=0.0, connection="compliant", cart_mass_kg=10.0,
            schedule=schedules(), record_dt=0.005,
            tow_summary={"valid": True, "failures": [], "min_clearance_coast_m": 0.05},
            thresholds=play.DEFAULT_THRESHOLDS, transition_window_s=0.05)
        self.assertEqual(metrics["verdict"]["code"], "COL")
        self.assertTrue(metrics["stop"]["contact"])

    def test_low_margin_is_flagged_without_contact(self):
        rows = build_episode()
        metrics = play.compute_case_metrics(
            rows, command_mps=1.0, slope_deg=0.0, connection="compliant", cart_mass_kg=10.0,
            schedule=schedules(), record_dt=0.005,
            tow_summary={"valid": True, "failures": [], "min_clearance_coast_m": 0.05},
            thresholds=play.DEFAULT_THRESHOLDS, transition_window_s=0.05)
        self.assertEqual(metrics["verdict"]["code"], "LOW")
        self.assertTrue(metrics["stop"]["gap_margin_low"])

    def test_fallen_robot_is_flagged(self):
        rows = build_episode(robot_height=0.10, pitch=1.2)
        metrics = play.compute_case_metrics(
            rows, command_mps=1.0, slope_deg=5.0, connection="rigid", cart_mass_kg=25.0,
            schedule=schedules(), record_dt=0.005,
            tow_summary={"valid": False, "failures": ["body_pitch_excessive"],
                         "min_clearance_coast_m": None},
            thresholds=play.DEFAULT_THRESHOLDS, transition_window_s=0.05)
        self.assertEqual(metrics["verdict"]["code"], "FALL")
        self.assertEqual(metrics["stability"]["fell_phase"], "station")

    def test_startup_time_to_90pct(self):
        rows = build_episode(robot_vx_b=[0.0] * 50 + [1.0] * 50)
        metrics = play.compute_case_metrics(
            rows, command_mps=1.0, slope_deg=0.0, connection="compliant", cart_mass_kg=10.0,
            schedule=schedules(), record_dt=0.005,
            tow_summary={"valid": True, "failures": [], "min_clearance_coast_m": 0.3},
            thresholds=play.DEFAULT_THRESHOLDS, transition_window_s=0.05)
        # 第 50 个 tow 样本（index 50）起误差为 0，保持 0.25 s = 50 样本刚好够 ⇒ 0.25 s
        self.assertAlmostEqual(metrics["startup"]["time_to_90pct_s"], 0.25, places=9)
        # 过冲是「窗口内最大速度 / 指令 − 1」：该窗口（0.05 s）内还没到指令 ⇒ −1
        self.assertAlmostEqual(metrics["startup"]["overshoot_ratio"], -1.0, places=12)


class ClassificationTests(unittest.TestCase):
    def _metrics(self, **sections):
        metrics = {"case": {"velocity_mps": 1.0}, "startup": {}, "speed": {}, "stop": {},
                   "stability": {}}
        for name, values in sections.items():
            metrics[name].update(values)
        return metrics

    def test_severity_ordering_prefers_fall_over_collision(self):
        metrics = self._metrics(stability={"fell": True}, stop={"contact": True},
                               speed={"available": True, "mae_mps": 1.0},
                               startup={"joint_rms_rad": 1.0})
        verdict = play.classify_case(metrics, play.DEFAULT_THRESHOLDS)
        self.assertEqual(verdict["code"], "FALL")
        self.assertIn("stop_collision", verdict["reasons"])
        self.assertIn("robot_fell", verdict["reasons"])

    def test_ok_when_all_within_limits(self):
        metrics = self._metrics(stability={"fell": False, "invalid": False},
                                stop={"contact": False, "gap_margin_low": False,
                                      "joint_rms_rad": 0.01, "joint_max_rad": 0.05},
                                speed={"available": True, "mae_mps": 0.05},
                                startup={"joint_rms_rad": 0.01, "joint_max_rad": 0.05})
        verdict = play.classify_case(metrics, play.DEFAULT_THRESHOLDS)
        self.assertEqual(verdict, {"code": "OK", "reasons": [], "severity": 0})

    def test_invalid_beats_fall(self):
        metrics = self._metrics(stability={"invalid": True, "fell": True},
                                stop={"contact": True}, speed={"available": False},
                                startup={})
        verdict = play.classify_case(metrics, play.DEFAULT_THRESHOLDS)
        self.assertEqual(verdict["code"], "INV")


class ReportTests(unittest.TestCase):
    def _summary(self, slope, velocity, connection, mass, code, reasons):
        metrics = {
            "case": {"slope_deg": slope, "velocity_mps": velocity, "connection": connection,
                     "cart_mass_kg": mass},
            "startup": {"joint_rms_rad": 0.02 if code == "OK" else 0.4, "joint_max_rad": 0.1,
                        "worst_joint": "j0", "torque_saturated_frac": 0.0,
                        "body_vx_rms_err_mps": 0.1, "time_to_90pct_s": 0.2,
                        "overshoot_ratio": 0.0},
            "speed": {"available": True, "mae_mps": 0.05, "rmse_mps": 0.06, "ratio_mean": 1.0,
                      "steady_mae_mps": 0.04},
            "stop": {"available": True, "cart_coast_distance_m": 0.7, "cart_coast_to_rest_m": 0.6,
                     "cart_speed_at_stop_mps": 1.0, "min_clearance_coast_m": 0.2,
                     "final_clearance_m": 0.3, "time_to_contact_after_stop_s": None,
                     "contact": "stop_collision" in reasons, "joint_rms_rad": 0.03,
                     "joint_max_rad": 0.1, "settle_time_s": 0.3, "robot_travel_after_stop_m": 0.1},
            "stability": {"min_robot_z_m": 0.27, "max_abs_pitch_rad": 0.05,
                          "fell": "robot_fell" in reasons, "invalid": False},
            "verdict": {"code": code, "reasons": reasons, "severity": play.VERDICT_SEVERITY[code]},
            "summarize_tow": {"valid": code == "OK", "failures": []},
        }
        return {"case": metrics["case"], "metrics": metrics, "verdict": metrics["verdict"]}

    def test_group_statistics_split(self):
        summaries = [
            self._summary(0.0, 0.5, "compliant", 5.0, "OK", []),
            self._summary(0.0, 1.5, "rigid", 25.0, "COL", ["stop_collision"]),
            self._summary(5.0, 1.0, "rigid", 20.0, "FALL", ["robot_fell"]),
            self._summary(-5.0, 1.0, "compliant", 10.0, "SPD", ["speed_track_error"]),
        ]
        groups = play.group_statistics(summaries)
        self.assertEqual(groups["flat"]["cases"], 2)
        self.assertEqual(groups["flat"]["cases_ok"], 1)
        self.assertEqual(groups["uphill"]["cases"], 1)
        self.assertEqual(groups["downhill"]["cases"], 1)
        conclusion = play.necessity_conclusion(groups, play.DEFAULT_THRESHOLDS)
        self.assertEqual(conclusion["verdict"]["flat"], "baseline_insufficient")
        self.assertEqual(conclusion["verdict"]["uphill"], "baseline_insufficient")
        self.assertEqual(len(conclusion["lines"]), 3)

    def test_all_ok_group_is_sufficient(self):
        summaries = [self._summary(0.0, 0.5, "compliant", 5.0, "OK", [])]
        groups = play.group_statistics(summaries)
        conclusion = play.necessity_conclusion(groups, play.DEFAULT_THRESHOLDS)
        self.assertEqual(conclusion["verdict"]["flat"], "baseline_sufficient")
        self.assertIn("不构成上层任务必要性的证据", conclusion["lines"][0])
        # 没有 case 的组不编结论
        self.assertEqual(conclusion["verdict"]["uphill"], "no_cases")

    def test_stop_dominated_hint(self):
        summaries = [self._summary(5.0, 1.0, "compliant", 10.0, "COL", ["stop_collision"])]
        groups = play.group_statistics(summaries)
        conclusion = play.necessity_conclusion(groups, play.DEFAULT_THRESHOLDS)
        self.assertIn("停车段", conclusion["lines"][1])

    def test_verdict_matrix_shape(self):
        summaries = [self._summary(0.0, 0.5, "compliant", 5.0, "OK", []),
                     self._summary(0.0, 1.0, "rigid", 5.0, "COL", ["stop_collision"]),
                     self._summary(5.0, 0.5, "compliant", 25.0, "OK", [])]
        matrix = play.format_verdict_matrix(summaries, slopes=[0.0, 5.0])
        self.assertIn("坡度 +0°", matrix)
        self.assertIn("坡度 +5°", matrix)
        lines = [line for line in matrix.splitlines()
                 if line.startswith("| 5 |") or line.startswith("| 25 |")]
        self.assertEqual(len(lines), 2)
        self.assertIn("COL", matrix)

    def test_markdown_report_contains_sections(self):
        summaries = [self._summary(0.0, 1.5, "rigid", 25.0, "COL", ["stop_collision"])]
        groups = play.group_statistics(summaries)
        conclusion = play.necessity_conclusion(groups, play.DEFAULT_THRESHOLDS)
        report = play.build_markdown_report(
            case_summaries=summaries, groups=groups, conclusion=conclusion,
            args_dict={"velocities": [1.5], "connections": ["rigid"], "cart_masses": [25.0],
                       "ground_friction": 0.8, "command_shaping": "direct", "ramp_time_s": 1.0,
                       "slope_backend": "gravity", "slope_settle": "hold", "record_every": 1,
                       "write_csv": "failed", "env_spacing": 16.0,
                       "lane_keeping": "pd", "lane_kp_y": 1.0, "lane_kd_y": 0.3,
                       "lane_kp_yaw": 1.5, "lane_kd_yaw": 0.3,
                       "lane_vy_limit": 0.4, "lane_wz_limit": 0.8},
            thresholds=play.DEFAULT_THRESHOLDS,
            schedule=play.make_schedule(settle_steps=200, tow_duration=5.0,
                                        coast_duration=5.0, dt=0.005),
            slopes=[0.0], git={"commit": "deadbeef", "working_tree": None})
        self.assertIn("## 逐坡度判定矩阵", report)
        self.assertIn("## 结论（任务是否有必要）", report)
        self.assertIn("## 限制", report)
        self.assertIn("deadbeef", report)

    def test_case_report_row_keys(self):
        summary = self._summary(5.0, 1.0, "rigid", 20.0, "COL", ["stop_collision"])
        row = play.case_report_row(summary["metrics"])
        for key in ("slope_deg", "velocity_mps", "connection", "cart_mass_kg", "verdict",
                    "reasons", "startup_joint_rms_rad", "speed_mae_mps",
                    "stop_cart_coast_distance_m", "stop_min_clearance_coast_m"):
            self.assertIn(key, row)
        self.assertEqual(row["verdict"], "COL")
        self.assertEqual(row["reasons"], "stop_collision")


class CliTests(unittest.TestCase):
    def test_defaults_match_documented_grid(self):
        args = play.parse_args([])
        self.assertEqual(list(args.velocities), list(play.DEFAULT_VELOCITIES))
        self.assertEqual(list(args.cart_masses), list(play.DEFAULT_CART_MASSES))
        self.assertEqual(list(args.slopes), list(play.DEFAULT_SLOPES_DEG))
        self.assertEqual(args.ground_friction, play.FACTORY_FLOOR_FRICTION)
        self.assertEqual(args.slope_settle, "hold")
        self.assertEqual(args.command_shaping, "direct")
        self.assertEqual(len(args.cases), 225)
        self.assertEqual(args.schedule.total_steps, 2200)

    def test_dry_run_plan_lines(self):
        args = play.parse_args(["--dry-run"])
        text = "\n".join(play.planned_grid_lines(args))
        self.assertIn("225 case", text)
        self.assertIn("45 环境并行", text)
        self.assertIn("0.8", text)

    def test_env_spacing_guard(self):
        with self.assertRaises(SystemExit):
            play.parse_args(["--env-spacing", "5"])

    def test_max_envs_guard(self):
        with self.assertRaises(SystemExit):
            play.parse_args(["--max-envs", "10"])

    def test_bad_slope_and_mass_rejected(self):
        for argv in (["--slopes", "50"], ["--slopes", "-46"], ["--cart-masses", "1"],
                     ["--cart-masses", "60"], ["--velocities", "0"], ["--velocities", "3"]):
            with self.assertRaises(SystemExit):
                play.parse_args(argv)

    def test_rope_geometry_guard(self):
        with self.assertRaises(SystemExit):
            play.parse_args(["--rope-length", "0.3", "--slack", "0.3"])

    def test_terrain_backend_accepts_a_grid_that_fits_the_training_cells(self):
        args = play.parse_args(["--slope-backend", "terrain", "--slopes", "5",
                                "--velocities", "1.0", "--connections", "rigid",
                                "--cart-masses", "10"])
        self.assertEqual(args.slope_backend, "terrain")
        self.assertEqual(len(args.cases), 1)

    def test_terrain_backend_rejects_more_envs_than_cells(self):
        # 5° 档在训练网格里只有 200 块 lane：14 速度 × 3 连接 × 5 质量 = 210 > 200
        with self.assertRaises(SystemExit):
            play.parse_args(["--slope-backend", "terrain", "--slopes", "5",
                             "--velocities", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9",
                             "1.0", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6",
                             "--connections", "compliant", "inextensible", "rigid",
                             "--cart-masses", "5", "10", "15", "20", "25",
                             "--env-spacing", "16", "--max-envs", "210"])

    def test_single_connection_subset_is_allowed(self):
        args = play.parse_args(["--connections", "rigid", "--slopes", "0",
                                "--velocities", "1.0", "--cart-masses", "10"])
        self.assertEqual(len(args.cases), 1)
        self.assertEqual(len(play.cases_for_slope(args.cases, 0.0)), 1)


class JsonSanitiseTests(unittest.TestCase):
    def test_non_finite_floats_become_none(self):
        payload = {"a": 1.5, "b": float("nan"), "c": [1.0, float("inf"), {"d": -float("inf")}],
                   "e": (float("nan"), 2.0)}
        clean = play._finite_only(payload)
        self.assertEqual(clean["a"], 1.5)
        self.assertIsNone(clean["b"])
        self.assertEqual(clean["c"][0], 1.0)
        self.assertIsNone(clean["c"][1])
        self.assertIsNone(clean["c"][2]["d"])
        self.assertEqual(clean["e"], [None, 2.0])

    def test_sanitised_payload_is_json_serialisable(self):
        import json as _json
        text = _json.dumps(play._finite_only({"x": float("nan")}), allow_nan=False)
        self.assertIn("null", text)

    def test_other_types_pass_through(self):
        payload = {"s": "rigid", "n": None, "i": 3, "b": True}
        self.assertEqual(play._finite_only(payload), payload)


class SimLoopStaticTests(unittest.TestCase):
    """仿真相位跑不了时的静态守卫：只钉住几条「顺序错了就一定错」的约定。

    这些**不是**行为测试：仿真行为证据只能来自训练机实跑。它们防的是把 `tow_drag.py`
    已经踩过的坑再踩一遍（力必须在步进前写入、位姿必须在 scene.reset 前写死、
    观测里的重力必须是单位向量、PhysX 主机构建缓冲不能和 CUDA 张量相乘）。
    """

    @classmethod
    def setUpClass(cls):
        cls.source = (RL / "scripts/towing/play_towing_test.py").read_text(encoding="utf-8")

    def test_gravity_is_written_through_physx_view(self):
        self.assertIn("physics_sim_view.set_gravity(carb.Float3(", self.source)

    def test_observation_gravity_is_normalised(self):
        self.assertIn("gravity_world = gravity_vector / gravity_vector.norm()", self.source)
        self.assertIn("per_env(gravity_world)", self.source)

    def test_terrain_reuses_the_first_scene_field(self):
        # 地形必须占住 `ground` 这个**原位**字段，否则 dataclass 会把新增字段排到最后，
        # 地形就比 robot/cart 晚建（与训练场景的 terrain-first 顺序不一致）。
        self.assertIn("class PlayTestTerrainSceneCfg(TowSceneCfg):", self.source)
        self.assertIn("scene_cfg.ground = TerrainImporterCfg(", self.source)
        self.assertIn("class_type=PlayTestTileImporter", self.source)
        self.assertIn("terrain_type=\"generator\"", self.source)
        self.assertNotIn("scene_cfg.terrain =", self.source)

    def test_mass_scaling_happens_on_host_buffer(self):
        # PhysX 的 get_masses()/get_inertias() 是 CPU 缓冲 ⇒ 缩放前必须先搬到 CPU
        self.assertIn("scale_cart_mass_inertia(", self.source)
        self.assertIn('mass_scales.detach().to("cpu")', self.source)
        # 回归守卫：不能再把已经是 (N, nb) 的缓冲 `.unsqueeze(0)` 去乘 (N,1)
        self.assertNotIn("nominal_masses.unsqueeze(0)", self.source)
        self.assertNotIn("nominal_inertias.unsqueeze(0)", self.source)

    def test_step_order_is_force_then_write_then_step_then_update(self):
        body = self.source.split("def apply_rope_and_resistance")[1]
        # 从调用点往后切：`scene.update(dt)` 在 reset_episode() 里也出现一次，不能用全局首个
        loop = body[body.index("state = apply_rope_and_resistance("):]
        order = [loop.index("scene.write_data_to_sim()"), loop.index("sim.step()"),
                 loop.index("scene.update(dt)")]
        self.assertEqual(order, sorted(order))

    def test_episode_reset_writes_poses_before_scene_reset(self):
        body = self.source.split("def reset_episode(*")[1].split("def make_row")[0]
        self.assertLess(body.index("write_root_pose_to_sim"), body.index("scene.reset()"))

    def test_records_start_at_the_physical_step(self):
        # record_every 默认必须是 1：summarize_tow 的负载速度跃变见证按 5 ms 标定
        self.assertEqual(play.parse_args([]).record_every, 1)


class SlopeCellTests(unittest.TestCase):
    """terrain 后端的 tile 来源：训练侧 40×20 网格里同坡度的 cell（不另写一张坡度表）。"""

    def test_cell_counts_per_grade(self):
        # 20 平列 × 20 行；5° 与 10° 各 10 列 × 20 行
        self.assertEqual(len(play.slope_cells(0.0)), 400)
        self.assertEqual(len(play.slope_cells(5.0)), 200)
        self.assertEqual(len(play.slope_cells(10.0)), 200)
        self.assertGreaterEqual(len(play.slope_cells(10.0)), 45)

    def test_every_cell_has_the_requested_grade(self):
        for grade in (0.0, 5.0, 10.0):
            for row, column in play.slope_cells(grade):
                self.assertEqual(connection_grid.slope_degrees(column, row), grade)

    def test_rejects_grade_outside_the_training_grid(self):
        # 剖面自带上下坡 ⇒ 没有负档，也没有 7° 这类量级
        for grade in (7.0, -5.0, -10.0):
            with self.assertRaises(ValueError):
                play.slope_cells(grade)


class CompactLayoutTests(unittest.TestCase):
    def test_layout_is_centred_and_spaced(self):
        origins = play.compact_tile_origins(45, row_spacing=22.0, column_spacing=8.0)
        self.assertEqual(len(origins), 45)
        xs = sorted({round(origin[0], 6) for origin in origins})
        ys = sorted({round(origin[1], 6) for origin in origins})
        self.assertEqual(len(xs), 7)                    # ceil(sqrt(45)) = 7 列 × 7 行
        self.assertEqual(len(ys), 7)
        self.assertAlmostEqual(sum(xs) / len(xs), 0.0, places=9)
        self.assertAlmostEqual(sum(ys) / len(ys), 0.0, places=9)
        self.assertAlmostEqual(xs[1] - xs[0], 22.0, places=9)
        self.assertAlmostEqual(ys[1] - ys[0], 8.0, places=9)
        self.assertTrue(all(origin[2] == 0.0 for origin in origins))

    def test_layout_keeps_the_forward_lane_clear(self):
        # 一块 tile 前向 17 m（FORWARD_M），行距必须大于「前进距离 + 后退距离」才不串场
        origins = play.compact_tile_origins(45, row_spacing=22.0, column_spacing=8.0)
        xs = sorted({origin[0] for origin in origins})
        self.assertGreaterEqual(xs[1] - xs[0], 17.0 + 3.0)

    def test_block_footprints_do_not_overlap(self):
        # 一块 tile 的足迹 = 沿坡 [−3, +17] m、横向 ±3 m；相邻 tile 至少在 x 或 y 上分开
        origins = play.compact_tile_origins(
            45, row_spacing=play.slope_geometry.ROW_SPACING_M,
            column_spacing=play.slope_geometry.COLUMN_SPACING_M)
        back, forward = play.slope_geometry.BACK_M, play.slope_geometry.FORWARD_M
        half = play.slope_geometry.HALF_WIDTH_M
        boxes = [(x - back, x + forward, y - half, y + half) for x, y, _ in origins]
        for index, first in enumerate(boxes):
            for second in boxes[index + 1:]:
                overlap_x = min(first[1], second[1]) - max(first[0], second[0])
                overlap_y = min(first[3], second[3]) - max(first[2], second[2])
                self.assertFalse(overlap_x > 0 and overlap_y > 0,
                                 f"tile 足迹重叠：{first} vs {second}")

    def test_rejects_bad_arguments(self):
        with self.assertRaises(ValueError):
            play.compact_tile_origins(0, row_spacing=1.0, column_spacing=1.0)
        with self.assertRaises(ValueError):
            play.compact_tile_origins(3, row_spacing=0.0, column_spacing=1.0)


class TranslateTileTests(unittest.TestCase):
    def test_top_surface_follows_the_profile_after_translation(self):
        profile = play.slope_geometry
        for column in (0, 30):                       # 平地 lane + 10° lane
            vertices, faces, source = profile.tile_mesh(0, column)
            target = (5.0, -3.0, 0.0)
            moved = play.translate_tile(vertices, source, target)
            self.assertEqual(len(moved), len(vertices))
            grade = play.connection_grid.slope_degrees(column, 0)
            half = len(moved) // 2                   # 每个剖面点先 −y 侧、后 +y 侧
            top = [(x, z) for x, _, z in moved if z > -0.5*profile.THICKNESS_M]
            self.assertTrue(top)
            for x, z in top:
                self.assertAlmostEqual(z, target[2] + profile.profile_height(grade, x - target[0]),
                                       places=9)
            # 平移不改变相对几何，且 lane 原点（剖面平地段）仍在表面上
            for old_vertex, new_vertex in zip(vertices, moved):
                self.assertAlmostEqual(new_vertex[0] - old_vertex[0], target[0] - source[0], places=9)
                self.assertAlmostEqual(new_vertex[1] - old_vertex[1], target[1] - source[1], places=9)
                self.assertAlmostEqual(new_vertex[2], old_vertex[2], places=9)
            self.assertAlmostEqual(profile.profile_height(grade, 0.0), 0.0, places=12)
            self.assertGreater(len(vertices), 0)
            self.assertEqual(half, len(vertices) // 2)

    def test_mesh_faces_stay_in_range(self):
        vertices, faces, source = play.slope_geometry.tile_mesh(0, 31)
        moved = play.translate_tile(vertices, source, (0.0, 0.0, 0.0))
        for face in faces:
            for index in face:
                self.assertLess(index, len(moved))


class MassScalingTests(unittest.TestCase):
    """逐 env 质量/惯量缩放的形状契约。

    这条测试是 2026-10-09 默认 45 环境首跑的回归守卫：
    `nominal_masses.unsqueeze(0) * (N,1)` 会在 dim 2 上撞 `num_bodies`(小车 5) 与 `N`，
    报 `size of tensor a (5) must match the size of tensor b (45)`；N=1 时侥幸通过，
    所以必须按 N>1 的形状测。
    """

    def setUp(self):
        try:
            import torch
        except ImportError:
            self.skipTest("需要 torch 才能做形状回归")
        self.torch = torch

    def test_shapes_for_several_env_counts(self):
        torch = self.torch
        for num_envs in (1, 8, 45):
            bodies = 5                      # base_link + 四轮（fixed joint 已合并）
            masses = torch.ones(num_envs, bodies)
            inertias = torch.ones(num_envs, bodies, 9)
            scales = torch.linspace(0.5, 2.5, num_envs)
            scaled_masses, scaled_inertias = play.scale_cart_mass_inertia(masses, inertias, scales)
            self.assertEqual(tuple(scaled_masses.shape), (num_envs, bodies))
            self.assertEqual(tuple(scaled_inertias.shape), (num_envs, bodies, 9))

    def test_scaling_is_applied_per_environment(self):
        torch = self.torch
        masses = torch.tensor([[10.0, 1.0, 1.0, 1.0, 1.0],
                               [10.0, 1.0, 1.0, 1.0, 1.0]])
        inertias = torch.ones(2, 5, 9)
        scales = torch.tensor([0.5, 2.5])
        scaled_masses, scaled_inertias = play.scale_cart_mass_inertia(masses, inertias, scales)
        self.assertAlmostEqual(float(scaled_masses[0, 0]), 5.0, places=6)
        self.assertAlmostEqual(float(scaled_masses[1, 0]), 25.0, places=6)
        self.assertAlmostEqual(float(scaled_inertias[1, 3, 4]), 2.5, places=6)
        # 惯量必须与质量同比例（AGENTS.md：只改质量不改惯量会让模型不自洽）
        # 惯量与质量必须用**同一个**比例（AGENTS.md：只改质量不改惯量会让模型不自洽）
        self.assertAlmostEqual(float(scaled_inertias[0, 0, 0] / inertias[0, 0, 0]),
                               float(scales[0]), places=9)
        self.assertAlmostEqual(float(scaled_inertias[1, 0, 8] / inertias[1, 0, 8]),
                               float(scales[1]), places=9)
    def test_the_old_unsqueeze_pattern_is_really_broken(self):
        """把 bug 的形状写进测试：N>1 时它必须抛广播错误，防止再写回去。"""
        torch = self.torch
        num_envs, bodies = 45, 5
        inertias = torch.ones(num_envs, bodies, 9)
        scales = torch.linspace(0.5, 2.5, num_envs).unsqueeze(1)
        with self.assertRaises(RuntimeError) as caught:
            _ = inertias.unsqueeze(0) * scales
        self.assertIn("dimension 2", str(caught.exception))

    def test_bad_shapes_are_rejected(self):
        torch = self.torch
        with self.assertRaises(ValueError):
            play.scale_cart_mass_inertia(torch.ones(5), torch.ones(1, 5, 9), torch.ones(1))
        with self.assertRaises(ValueError):
            play.scale_cart_mass_inertia(torch.ones(2, 5), torch.ones(3, 5, 9), torch.ones(2))
        with self.assertRaises(ValueError):
            play.scale_cart_mass_inertia(torch.ones(2, 5), torch.ones(2, 5, 9), torch.ones(3))


class TerrainLayoutTests(unittest.TestCase):
    """整块 mesh 的组装（只有仿真里才建，但布局与自洽性可以离线核对）。"""

    GRADES = (0.0, 5.0, 10.0)

    @classmethod
    def setUpClass(cls):
        cls.layout = play.build_terrain_layout(cls.GRADES, 45)

    def test_block_shape_and_counts(self):
        self.assertEqual(len(self.layout["blocks"]), len(self.GRADES))
        for grade in self.GRADES:
            block = self.layout["blocks"][grade]
            self.assertEqual(len(block["cells"]), 45)
            self.assertEqual(len(block["origins"]), 45)
        # 平地 lane 的剖面退化成矩形（8 顶点 / 12 面），坡道 lane 是 16 顶点 / 28 面
        expected_verts = expected_faces = 0
        for grade in self.GRADES:
            per_lane = (8, 12) if grade == 0.0 else (16, 28)
            expected_verts += 45 * per_lane[0]
            expected_faces += 45 * per_lane[1]
        self.assertEqual(len(self.layout["vertices"]), expected_verts)
        self.assertEqual(len(self.layout["faces"]), expected_faces)

    def test_mesh_is_finite_and_faces_are_in_range(self):
        for vertex in self.layout["vertices"]:
            self.assertTrue(all(math.isfinite(value) for value in vertex))
        face_count = len(self.layout["vertices"])
        for face in self.layout["faces"]:
            self.assertEqual(len(face), 3)
            self.assertTrue(all(0 <= index < face_count for index in face))

    def test_every_origin_lies_on_its_tile_surface(self):
        for grade in self.GRADES:
            block = self.layout["blocks"][grade]
            for cell, origin in zip(block["cells"], block["origins"]):
                vertices, _, source = play.slope_geometry.tile_mesh(*cell)
                moved = play.translate_tile(vertices, source, origin)
                for x, _, z in moved:
                    if z > origin[2] - 0.5*play.slope_geometry.THICKNESS_M:
                        self.assertAlmostEqual(
                            z, play.slope_geometry.profile_height(grade, x - origin[0]), places=9)

    def test_origins_grid_matches_the_mesh_and_the_terrain_contract(self):
        # Isaac Lab 的 `terrain_origins` 契约是 (num_rows, num_cols, 3)，必须覆盖整块 mesh
        grid = self.layout["origins_grid"]
        self.assertEqual(len(grid), len(self.GRADES))
        for block in grid:
            self.assertEqual(len(block), 45)
            for origin in block:
                self.assertEqual(len(origin), 3)
        self.assertEqual(len(grid) * len(grid[0]), len(self.layout["blocks"]) * 45)
        # 块顺序与 slopes 顺序一致（generator 用 num_rows = 坡度量级数来报这张表）
        self.assertEqual([self.layout["blocks"][grade]["origins"] for grade in self.GRADES],
                         [list(block) for block in grid])

    def test_blocks_are_separated_along_y(self):
        pitch = self.layout["block_y_pitch_m"]
        half = play.slope_geometry.HALF_WIDTH_M
        shifts = [self.layout["blocks"][grade]["block_y_shift_m"] for grade in self.GRADES]
        self.assertEqual(shifts, sorted(shifts))
        for previous, following in zip(shifts, shifts[1:]):
            self.assertGreater(following - previous, 2.0 * half)   # 两块 y 上不重叠

    def test_too_few_cells_is_rejected(self):
        with self.assertRaises(ValueError):
            play.build_terrain_layout((5.0,), 201)          # 5° 只有 200 块 lane

    def test_empty_or_zero_env_rejected(self):
        with self.assertRaises(ValueError):
            play.build_terrain_layout((), 1)
        with self.assertRaises(ValueError):
            play.build_terrain_layout((0.0,), 0)


class SurfaceFrameTests(unittest.TestCase):
    """两个后端的度量系都是 lane 系（+x 前 / +z 上）；terrain 多带一个「坡面档位」。"""

    def test_gravity_backend_is_the_flat_frame(self):
        frame = play.surface_frame(-5.0, "gravity")          # 负号 = 下坡（重力倾斜方向）
        self.assertEqual(frame["tangent"], (1.0, 0.0, 0.0))
        self.assertEqual(frame["normal"], (0.0, 0.0, 1.0))
        self.assertEqual(frame["profile_grade_deg"], 0.0)

    def test_terrain_backend_carries_the_profile_grade(self):
        for grade in (0.0, 5.0, 10.0):
            frame = play.surface_frame(grade, "terrain")
            self.assertEqual(frame["tangent"], (1.0, 0.0, 0.0))
            self.assertEqual(frame["normal"], (0.0, 0.0, 1.0))
            self.assertEqual(frame["profile_grade_deg"], grade)
        # 剖面自带上下坡 ⇒ 负档不成立
        for bad in (-5.0, -10.0):
            with self.assertRaises(ValueError):
                play.surface_frame(bad, "terrain")

    def test_frame_is_orthonormal(self):
        for backend, slope in (("terrain", 0.0), ("terrain", 10.0), ("gravity", -10.0),
                               ("gravity", 5.0)):
            frame = play.surface_frame(slope, backend)
            tangent, normal = frame["tangent"], frame["normal"]
            self.assertAlmostEqual(math.dist(tangent, (0, 0, 0)), 1.0, places=12)
            self.assertAlmostEqual(math.dist(normal, (0, 0, 0)), 1.0, places=12)
            self.assertAlmostEqual(sum(t * n for t, n in zip(tangent, normal)), 0.0, places=12)

    def test_unknown_backend_rejected(self):
        with self.assertRaises(ValueError):
            play.surface_frame(5.0, "ramp")

    def test_frame_coordinates_are_lane_x_and_z(self):
        # 点在剖面表面上：lane 系高度 = profile_height，减掉它才是「离面高度」
        profile = play.slope_geometry
        frame = play.surface_frame(10.0, "terrain")
        for x in (2.0, 7.5, 10.0):
            height = profile.profile_height(10.0, x)
            progress, z = play.frame_coordinates((x, 0.0, height), (0.0, 0.0, 0.0), frame)
            self.assertAlmostEqual(progress, x, places=9)
            self.assertAlmostEqual(z, height, places=9)
            self.assertAlmostEqual(z - profile.profile_height(10.0, progress), 0.0, places=9)

    def test_gravity_frame_reduces_to_x_and_z(self):
        frame = play.surface_frame(-5.0, "gravity")
        progress, height = play.frame_coordinates((3.0, 1.0, 0.27), (1.0, 1.0, 0.0), frame)
        self.assertAlmostEqual(progress, 2.0, places=12)
        self.assertAlmostEqual(height, 0.27, places=12)



class SpawnOnSurfaceTests(unittest.TestCase):
    ROBOT_OFFSET = (-0.16, 0.0, 0.0)
    CART_OFFSET = (0.25, 0.0, 0.0)

    def _spawn(self, slope, target=0.4, robot_height=0.35, cart_height=0.15):
        return play.spawn_on_surface(slope_deg=slope, robot_height=robot_height,
                                     cart_height=cart_height, robot_offset=self.ROBOT_OFFSET,
                                     cart_offset=self.CART_OFFSET, target_distance=target)

    def test_flat_case_matches_the_flat_spawn_solution(self):
        spec = self._spawn(0.0)
        expected_x = play._initial_cart_x(0.8, 0.4, spawn_height=0.35, cart_height=0.15,
                                         robot_offset=self.ROBOT_OFFSET,
                                         cart_offset=self.CART_OFFSET)
        self.assertAlmostEqual(spec["cart_root"][0], expected_x, places=9)
        self.assertAlmostEqual(spec["cart_root"][2], 0.15, places=9)
        self.assertEqual(spec["quat_wxyz"], (1.0, 0.0, 0.0, 0.0))

    def test_attachment_distance_is_exact_on_every_slope(self):
        for slope in (0.0, 5.0, -5.0, 10.0, -10.0):
            spec = self._spawn(slope, target=0.4)
            self.assertAlmostEqual(spec["attachment_distance_m"], 0.4, places=9)

    def test_birth_attitude_aligns_body_z_with_the_plane_normal(self):
        spec = self._spawn(10.0)
        normal = play.rotate_vector(spec["quat_wxyz"], (0.0, 0.0, 1.0))
        self.assertAlmostEqual(normal[0], spec["normal"][0], places=9)
        self.assertAlmostEqual(normal[2], spec["normal"][2], places=9)
        forward = play.rotate_vector(spec["quat_wxyz"], (1.0, 0.0, 0.0))
        self.assertAlmostEqual(forward[0], spec["tangent"][0], places=9)

    def test_robot_height_is_a_normal_clearance_and_cart_sits_behind(self):
        # 恒定坡度档的解（剖面出生用的是 slope=0，这条方程本身仍要能离线复核）
        for slope in (0.0, 5.0, -5.0, 10.0):
            tangent, normal = play.slope_geometry.slope_frame(slope)
            frame = {"tangent": tangent, "normal": normal}
            spec = self._spawn(slope)
            robot_progress, robot_height = play.frame_coordinates(spec["robot_root"], (0, 0, 0),
                                                                  frame)
            cart_progress, cart_height = play.frame_coordinates(spec["cart_root"], (0, 0, 0), frame)
            self.assertAlmostEqual(robot_progress, 0.0, places=9)
            self.assertAlmostEqual(robot_height, 0.35, places=9)
            self.assertLess(cart_progress, robot_progress - 0.3)   # 小车在后方
            self.assertGreater(cart_height, 0.0)

    def test_impossible_geometry_is_rejected(self):
        with self.assertRaises(ValueError):
            self._spawn(5.0, target=0.1)

    def test_invalid_heights_rejected(self):
        for value in (0.0, -0.2, float("nan")):
            with self.assertRaises(ValueError):
                self._spawn(0.0, robot_height=value)


class RotationTests(unittest.TestCase):
    def test_identity_and_axis_mapping(self):
        self.assertEqual(play.rotation_y(0.0), (1.0, 0.0, 0.0, 0.0))
        quat = play.rotation_y(-math.radians(10.0))
        forward = play.rotate_vector(quat, (1.0, 0.0, 0.0))
        tangent = (math.cos(math.radians(10.0)), 0.0, math.sin(math.radians(10.0)))
        for mine, theirs in zip(forward, tangent):
            self.assertAlmostEqual(mine, theirs, places=9)

    def test_rotation_preserves_length(self):
        quat = play.rotation_y(0.37)
        vector = (0.3, -0.4, 0.5)
        rotated = play.rotate_vector(quat, vector)
        self.assertAlmostEqual(math.dist(rotated, (0, 0, 0)), math.dist(vector, (0, 0, 0)),
                               places=12)


class GeometryTests(unittest.TestCase):
    def test_initial_cart_x_matches_attachment_distance(self):
        robot_offset = (-0.16, 0.0, 0.0)
        cart_offset = (0.25, 0.0, 0.0)
        spawn_height, cart_height = 0.35, 0.15
        cart_x = play._initial_cart_x(0.8, 0.4, spawn_height=spawn_height,
                                      cart_height=cart_height, robot_offset=robot_offset,
                                      cart_offset=cart_offset)
        robot_point = (0.0 + robot_offset[0], 0.0, spawn_height + robot_offset[2])
        cart_point = (cart_x + cart_offset[0], 0.0, cart_height + cart_offset[2])
        distance = math.dist(robot_point, cart_point)
        self.assertAlmostEqual(distance, 0.4, places=9)     # L0 − slack

    def test_impossible_rope_length_rejected(self):
        with self.assertRaises(ValueError):
            play._initial_cart_x(0.2, 0.19, spawn_height=0.35, cart_height=0.15,
                                 robot_offset=(-0.16, 0.0, 0.0), cart_offset=(0.25, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
