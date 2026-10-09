"""`play_towing_test.py` 的离线测试：判据与网格逻辑不依赖 Isaac Sim。

本机（以及任何没有 GPU/Isaac Lab 的机器）只能验证**纯逻辑**部分：训练网格映射与工作条件
轮转、出生几何、阶段划分、指令整形、五项指标与横向 PD 的计算与判定、报告与矩阵、CLI 校验、
记录字段契约。仿真相位（场景构造、PhysX 步进、绳力施加、接触传感器）不在覆盖范围内 ——
那部分只能在训练机实跑，不要在文档里把本文件的通过写成「测试台已验证」。
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
    row["time_s"] = 0.005
    return row | overrides


def synthetic_case(*, grade_deg=0.0, velocity_mps=1.0, connection="compliant",
                   cart_mass_kg=10.0, connection_length_m=0.8, env_index=0,
                   column=0, row=0):
    """与 `EnvCase.to_dict()` 同构的合成 case（指标函数只读这个字典）。"""
    return {"env_index": env_index, "cell_index": row * 40 + column, "column": column,
            "row": row, "connection": connection,
            "connection_length_m": connection_length_m, "grade_deg": grade_deg,
            "velocity_mps": velocity_mps, "cart_mass_kg": cart_mass_kg}


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
                       grade_deg=0.0, connection_length_m=0.8,
                       gravity_x_mps2=0.0, gravity_z_mps2=-9.81, cart_mass_kg=10.0,
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
            rows, case=synthetic_case(grade_deg=0.0, connection="compliant", cart_mass_kg=10.0),
            schedule=schedules(), record_dt=0.005,
            tow_summary={"valid": True, "failures": [], "min_clearance_coast_m": 0.3},
            thresholds=play.DEFAULT_THRESHOLDS, transition_window_s=0.05,
            lane_keeping="pd", lane_vy_limit=0.4, lane_wz_limit=0.8)
        self.assertIn("lane_deviation", metrics["verdict"]["reasons"])
        self.assertEqual(metrics["verdict"]["code"], "LAT")
        metrics_off = play.compute_case_metrics(
            rows, case=synthetic_case(grade_deg=0.0, connection="compliant", cart_mass_kg=10.0),
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
            rows, case=synthetic_case(grade_deg=0.0, connection="compliant", cart_mass_kg=10.0),
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
            rows, case=synthetic_case(grade_deg=0.0, connection="compliant", cart_mass_kg=10.0),
            schedule=schedules(), record_dt=0.005, tow_summary=tow_summary,
            thresholds=play.DEFAULT_THRESHOLDS, transition_window_s=0.05)
        self.assertEqual(metrics["verdict"]["code"], "JNT")
        self.assertIn("startup_joint_error", metrics["verdict"]["reasons"])

    def test_speed_shortfall_is_flagged(self):
        rows = build_episode(robot_vx_b=[0.5] * 100)
        metrics = play.compute_case_metrics(
            rows, case=synthetic_case(grade_deg=0.0, connection="compliant", cart_mass_kg=10.0),
            schedule=schedules(), record_dt=0.005,
            tow_summary={"valid": True, "failures": [], "min_clearance_coast_m": 0.3},
            thresholds=play.DEFAULT_THRESHOLDS, transition_window_s=0.05)
        self.assertIn("speed_track_error", metrics["verdict"]["reasons"])

    def test_coast_collision_is_flagged(self):
        rows = build_episode(deck_fx_coast=15.0)
        metrics = play.compute_case_metrics(
            rows, case=synthetic_case(grade_deg=0.0, connection="compliant", cart_mass_kg=10.0),
            schedule=schedules(), record_dt=0.005,
            tow_summary={"valid": True, "failures": [], "min_clearance_coast_m": 0.05},
            thresholds=play.DEFAULT_THRESHOLDS, transition_window_s=0.05)
        self.assertEqual(metrics["verdict"]["code"], "COL")
        self.assertTrue(metrics["stop"]["contact"])

    def test_low_margin_is_flagged_without_contact(self):
        rows = build_episode()
        metrics = play.compute_case_metrics(
            rows, case=synthetic_case(grade_deg=0.0, connection="compliant", cart_mass_kg=10.0),
            schedule=schedules(), record_dt=0.005,
            tow_summary={"valid": True, "failures": [], "min_clearance_coast_m": 0.05},
            thresholds=play.DEFAULT_THRESHOLDS, transition_window_s=0.05)
        self.assertEqual(metrics["verdict"]["code"], "LOW")
        self.assertTrue(metrics["stop"]["gap_margin_low"])

    def test_fallen_robot_is_flagged(self):
        rows = build_episode(robot_height=0.10, pitch=1.2)
        metrics = play.compute_case_metrics(
            rows, case=synthetic_case(grade_deg=5.0, connection="rigid", cart_mass_kg=25.0),
            schedule=schedules(), record_dt=0.005,
            tow_summary={"valid": False, "failures": ["body_pitch_excessive"],
                         "min_clearance_coast_m": None},
            thresholds=play.DEFAULT_THRESHOLDS, transition_window_s=0.05)
        self.assertEqual(metrics["verdict"]["code"], "FALL")
        self.assertEqual(metrics["stability"]["fell_phase"], "station")

    def test_startup_time_to_90pct(self):
        rows = build_episode(robot_vx_b=[0.0] * 50 + [1.0] * 50)
        metrics = play.compute_case_metrics(
            rows, case=synthetic_case(grade_deg=0.0, connection="compliant", cart_mass_kg=10.0),
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
    def _summary(self, grade, velocity, connection, mass, code, reasons, *, length=0.8,
                 env_index=0):
        metrics = {
            "case": synthetic_case(grade_deg=grade, velocity_mps=velocity,
                                   connection=connection, cart_mass_kg=mass,
                                   connection_length_m=length, env_index=env_index),
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

    def test_group_statistics_split_by_grade(self):
        summaries = [
            self._summary(0.0, 0.5, "compliant", 5.0, "OK", []),
            self._summary(0.0, 1.5, "rigid", 25.0, "COL", ["stop_collision"]),
            self._summary(5.0, 1.0, "rigid", 20.0, "FALL", ["robot_fell"]),
            self._summary(10.0, 1.0, "compliant", 10.0, "SPD", ["speed_track_error"]),
        ]
        groups = play.group_statistics(summaries)
        self.assertEqual(list(groups), ["grade0", "grade5", "grade10"])
        self.assertEqual(groups["grade0"]["cases"], 2)
        self.assertEqual(groups["grade0"]["cases_ok"], 1)
        self.assertEqual(groups["grade5"]["cases"], 1)
        self.assertEqual(groups["grade10"]["cases"], 1)
        self.assertEqual(groups["grade5"]["label"], "5° 坡")
        self.assertEqual(groups["grade0"]["label"], "平地")
        conclusion = play.necessity_conclusion(groups, play.DEFAULT_THRESHOLDS)
        self.assertEqual(conclusion["verdict"]["grade0"], "baseline_insufficient")
        self.assertEqual(conclusion["verdict"]["grade5"], "baseline_insufficient")
        self.assertEqual(len(conclusion["lines"]), 3)

    def test_all_ok_group_is_sufficient(self):
        summaries = [self._summary(0.0, 0.5, "compliant", 5.0, "OK", [])]
        groups = play.group_statistics(summaries)
        conclusion = play.necessity_conclusion(groups, play.DEFAULT_THRESHOLDS)
        self.assertEqual(conclusion["verdict"]["grade0"], "baseline_sufficient")
        self.assertIn("不构成上层任务必要性的证据", conclusion["lines"][0])
        # 没有 case 的档位根本不进分组（不是编一条 no_cases）
        self.assertNotIn("grade5", groups)

    def test_stop_dominated_hint(self):
        summaries = [self._summary(5.0, 1.0, "compliant", 10.0, "COL", ["stop_collision"])]
        groups = play.group_statistics(summaries)
        conclusion = play.necessity_conclusion(groups, play.DEFAULT_THRESHOLDS)
        self.assertIn("停车段", conclusion["lines"][0])

    def test_lane_dominated_hint(self):
        summaries = [self._summary(5.0, 1.0, "compliant", 10.0, "LAT", ["lane_deviation"])]
        groups = play.group_statistics(summaries)
        conclusion = play.necessity_conclusion(groups, play.DEFAULT_THRESHOLDS)
        self.assertIn("横向", conclusion["lines"][0])

    def test_verdict_matrix_shape_and_env_counts(self):
        summaries = [self._summary(0.0, 0.5, "compliant", 5.0, "OK", []),
                     self._summary(0.0, 1.0, "rigid", 5.0, "COL", ["stop_collision"]),
                     self._summary(0.0, 1.0, "rigid", 5.0, "OK", [], env_index=1),
                     self._summary(5.0, 0.5, "compliant", 25.0, "OK", [])]
        matrix = play.format_verdict_matrix(summaries, grades=[0.0, 5.0])
        self.assertIn("坡度量级 平地", matrix)
        self.assertIn("坡度量级 5° 坡", matrix)
        # 同一 (速度, 连接, 质量) 组合里的多个 env 合成一格：最严重判定码 + env 数
        self.assertIn("COL×2", matrix)
        self.assertIn("OK×1", matrix)
        lines = [line for line in matrix.splitlines()
                 if line.startswith("| 5 |") or line.startswith("| 25 |")]
        self.assertEqual(len(lines), 2)

    def test_markdown_report_contains_sections(self):
        summaries = [self._summary(0.0, 1.5, "rigid", 25.0, "COL", ["stop_collision"])]
        groups = play.group_statistics(summaries)
        conclusion = play.necessity_conclusion(groups, play.DEFAULT_THRESHOLDS)
        report = play.build_markdown_report(
            case_summaries=summaries, groups=groups, conclusion=conclusion,
            args_dict={"num_envs": 800, "velocities": [1.5], "cart_masses": [25.0],
                       "ground_friction": 0.8, "wheel_damping": 0.032,
                       "command_shaping": "direct", "ramp_time_s": 1.0,
                       "record_every": 5, "write_csv": "failed",
                       "lane_keeping": "pd", "lane_kp_y": 1.0, "lane_kd_y": 0.3,
                       "lane_kp_yaw": 1.5, "lane_kd_yaw": 0.3,
                       "lane_vy_limit": 0.4, "lane_wz_limit": 0.8},
            thresholds=play.DEFAULT_THRESHOLDS,
            schedule=play.make_schedule(settle_steps=200, tow_duration=5.0,
                                        coast_duration=5.0, dt=0.005),
            grades=[0.0], git={"commit": "deadbeef", "working_tree": None})
        self.assertIn("## 逐坡度量级判定矩阵", report)
        self.assertIn("## 结论（任务是否有必要）", report)
        self.assertIn("## 限制", report)
        self.assertIn("deadbeef", report)
        self.assertIn("训练场景", report)

    def test_case_report_row_keys(self):
        summary = self._summary(5.0, 1.0, "rigid", 20.0, "COL", ["stop_collision"])
        row = play.case_report_row(summary["metrics"])
        for key in ("env_index", "cell_index", "column", "row", "grade_deg",
                    "connection_length_m", "velocity_mps", "connection", "cart_mass_kg",
                    "verdict", "reasons", "startup_joint_rms_rad", "speed_mae_mps",
                    "stop_cart_coast_distance_m", "stop_min_clearance_coast_m",
                    "lane_y_max_abs_m"):
            self.assertIn(key, row)
        self.assertEqual(row["verdict"], "COL")
        self.assertEqual(row["reasons"], "stop_collision")


class NoCartTests(unittest.TestCase):
    """`--no-cart-fraction`：无负载参考测量的分配与物理屏蔽（纯逻辑 + 静态守卫）。

    用途：`startup_joint_error` 是「实测关节角 − 当拍下发的关节目标」，它的**静差部分 ≈ τ/kp**
    （PD 顺从性），拖曳与否都会有 —— 所以要拿**完全不拖车**的一轮当参考统计量比较，
    而不是给测试台加一条"站定基线"判据（用户 2026-10-09 决定）。这一轮只改场景：
    无小车 env 的小车横向停到 2 m 外、绳力与轮阻置 0，判定时跳过与小车有关的 `COL`/`LOW`。
    """

    def test_default_is_all_carts(self):
        for index in range(20):
            self.assertTrue(play.cart_present_for(index, 0.0))
        cases = play.build_env_cases(8, [1.0], [10.0])
        self.assertTrue(all(case.cart_present for case in cases))
        self.assertNotIn("_nocart", cases[0].slug)

    def test_fraction_one_means_no_carts_at_all(self):
        self.assertFalse(any(play.cart_present_for(index, 1.0) for index in range(50)))
        cases = play.build_env_cases(8, [1.0], [10.0], no_cart_fraction=1.0)
        self.assertFalse(any(case.cart_present for case in cases))
        self.assertTrue(cases[0].slug.endswith("_nocart"))
        self.assertEqual(cases[0].to_dict()["cart_present"], False)

    def test_middle_fraction_is_deterministic_and_close_to_requested(self):
        cases = play.build_env_cases(800, play.DEFAULT_VELOCITIES, play.DEFAULT_CART_MASSES,
                                     no_cart_fraction=0.125)
        absent = sum(1 for case in cases if not case.cart_present)
        self.assertEqual(absent, 100)                      # 800 × 0.125 = 100（等距抽样）
        again = play.build_env_cases(800, play.DEFAULT_VELOCITIES, play.DEFAULT_CART_MASSES,
                                     no_cart_fraction=0.125)
        self.assertEqual([c.cart_present for c in cases], [c.cart_present for c in again])

    def test_rejects_out_of_range_fraction(self):
        with self.assertRaises(ValueError):
            play.cart_present_for(0, -0.1)
        with self.assertRaises(ValueError):
            play.cart_present_for(0, 1.5)

    def test_physics_and_verdicts_are_masked_for_absent_carts(self):
        source = (RL / "scripts/towing/play_towing_test.py").read_text(encoding="utf-8")
        # 绳力/轮阻必须乘 present（只靠横向距离不够稳；训练侧 cart_present 同义）
        self.assertIn("for component in state.force_on_robot], dim=-1) * present", source)
        self.assertIn("for component in state.force_on_cart], dim=-1) * present", source)
        self.assertIn("wheel_damping) * present.unsqueeze(1)", source)
        # 不拖车的 env 必须在判定前把与小车有关的量中性化（否则会报 COL/LOW）
        self.assertIn("if not cart_present:", source)
        self.assertIn('stop["gap_margin_low"] = False', source)
        # 无小车 env 的文件名带 _nocart，两轮 run 可以共存并区分
        self.assertIn('"_nocart"', source)


class WorkConditionTests(unittest.TestCase):
    """工作条件的确定性轮转：`slot = row + column` ⇒ 质量 `slot % len(masses)`、速度每 len(masses) 个 slot 换一档。"""

    def test_slot_rotation_is_deterministic(self):
        first = play.work_conditions(30, [0.5, 1.0], [5.0, 10.0, 20.0], columns=10)
        again = play.work_conditions(30, [0.5, 1.0], [5.0, 10.0, 20.0], columns=10)
        self.assertEqual(first, again)
        self.assertEqual(len(first), 30)
        # slot = row + column（columns=10）：env0=(0,0)→slot0、env1=(0,1)→1、env10=(1,0)→1 …
        self.assertEqual(play.work_conditions(4, [0.5], [5.0, 10.0, 20.0], columns=2),
                         [(0.5, 5.0), (0.5, 10.0), (0.5, 10.0), (0.5, 20.0)])

    def test_bucket_counts_are_balanced_within_a_few(self):
        conditions = play.work_conditions(800, [0.5, 1.0, 1.5], [5.0, 10.0, 15.0, 20.0, 25.0])
        counts = {}
        for condition in conditions:
            counts[condition] = counts.get(condition, 0) + 1
        self.assertEqual(len(counts), 15)
        self.assertEqual(sorted(counts), sorted(
            (velocity, mass) for velocity in (0.5, 1.0, 1.5)
            for mass in (5.0, 10.0, 15.0, 20.0, 25.0)))
        # slot 的取值个数在 20×40 网格里两头少中间多 ⇒ 不是严格 ±1，但都靠近 800/15 = 53.3
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 5)
        self.assertTrue(all(50 <= value <= 55 for value in counts.values()), sorted(counts.values()))
        self.assertEqual(sum(counts.values()), 800)

    def test_mass_is_balanced_across_connection_and_grade(self):
        """回归：质量**不能**只由列决定，否则 (连接 × 质量)/(坡度量级 × 质量) 会混淆。"""
        cases = play.build_env_cases(800, [0.5, 1.0, 1.5], [5.0, 10.0, 15.0, 20.0, 25.0])
        by_connection, by_grade, per_column = {}, {}, {}
        for case in cases:
            by_connection.setdefault(case.connection, {}).setdefault(case.cart_mass_kg, 0)
            by_connection[case.connection][case.cart_mass_kg] += 1
            by_grade.setdefault(case.grade_deg, {}).setdefault(case.cart_mass_kg, 0)
            by_grade[case.grade_deg][case.cart_mass_kg] += 1
            per_column.setdefault(case.column, set()).add((case.velocity_mps, case.cart_mass_kg))
        for connection, counts in by_connection.items():
            self.assertEqual(len(set(counts.values())), 1, (connection, counts))
            self.assertEqual(len(counts), len(play.DEFAULT_CART_MASSES))
        for grade, counts in by_grade.items():
            self.assertEqual(len(set(counts.values())), 1, (grade, counts))
            self.assertEqual(len(counts), len(play.DEFAULT_CART_MASSES))
        # 每一列都覆盖全部 15 个 (速度, 质量) 组合
        self.assertTrue(all(len(combos) == 15 for combos in per_column.values()))

    def test_bad_arguments_rejected(self):
        for argv in ((0, [1.0], [5.0]), (-1, [1.0], [5.0]), (4, [], [5.0]),
                     (4, [1.0], []), (4, [0.0], [5.0]), (4, [1.0], [0.0]),
                     (4, [float("nan")], [5.0])):
            with self.assertRaises(ValueError):
                play.work_conditions(*argv)


class EnvCaseTests(unittest.TestCase):
    """逐 env 的 case 必须与训练网格逐位一致（env i → env_spec(i)）。"""

    def test_full_grid_maps_every_cell_once(self):
        cases = play.build_env_cases(800, play.DEFAULT_VELOCITIES, play.DEFAULT_CART_MASSES)
        self.assertEqual(len(cases), 800)
        self.assertEqual(sorted(case.cell_index for case in cases), list(range(800)))
        for case in cases:
            spec = connection_grid.env_spec(case.env_index)
            self.assertEqual(case.cell_index, spec["grid_index"])
            self.assertEqual(case.column, spec["column"])
            self.assertEqual(case.row, spec["row"])
            self.assertEqual(case.connection, spec["model_name"])
            self.assertEqual(case.length_m, spec["length"])
            self.assertEqual(case.grade_deg, spec["slope_degrees"])

    def test_cell_distribution_matches_the_training_grid(self):
        cases = play.build_env_cases(800, play.DEFAULT_VELOCITIES, play.DEFAULT_CART_MASSES)
        connections = {}
        grades = {}
        for case in cases:
            connections[case.connection] = connections.get(case.connection, 0) + 1
            grades[case.grade_deg] = grades.get(case.grade_deg, 0) + 1
        # 20 平列 + 10 个 5° + 10 个 10°；连接 8/8/4（平列）与 4/4/2（坡列）
        self.assertEqual(grades, {0.0: 400, 5.0: 200, 10.0: 200})
        self.assertEqual(connections, {"compliant": 320, "inextensible": 160, "rigid": 320})

    def test_multiple_of_the_grid_repeats_cells_with_other_work_conditions(self):
        # 轮转周期 = 3 速度 × 5 质量 = 15，800 % 15 = 5 ≠ 0 ⇒ 同一 cell 的第二个 env 落在别的组合
        cases = play.build_env_cases(1600, play.DEFAULT_VELOCITIES, play.DEFAULT_CART_MASSES)
        self.assertEqual(len(cases), 1600)
        self.assertEqual(cases[0].cell_index, cases[800].cell_index)
        self.assertNotEqual((cases[0].velocity_mps, cases[0].cart_mass_kg),
                            (cases[800].velocity_mps, cases[800].cart_mass_kg))

    def test_slug_is_unique_and_filename_safe(self):
        cases = play.build_env_cases(800, play.DEFAULT_VELOCITIES, play.DEFAULT_CART_MASSES)
        slugs = [case.slug for case in cases]
        self.assertEqual(len(set(slugs)), 800)
        for slug in slugs:
            self.assertNotIn("/", slug)
        self.assertEqual(cases[0].slug, "env0000_c00r00_g0_v0.5_compliant_m5kg")

    def test_to_dict_is_json_friendly(self):
        import json as _json
        payload = play.build_env_cases(3, [1.0], [10.0])[2].to_dict()
        self.assertEqual(set(payload), {"env_index", "cell_index", "column", "row",
                                        "connection", "connection_length_m", "grade_deg",
                                        "velocity_mps", "cart_mass_kg", "cart_present"})
        _json.dumps(payload)

    def test_env_case_summary_reports_the_distribution(self):
        cases = play.build_env_cases(800, play.DEFAULT_VELOCITIES, play.DEFAULT_CART_MASSES)
        summary = play.env_case_summary(cases)
        self.assertEqual(summary["envs"], 800)
        self.assertEqual(summary["cells"], 800)
        self.assertEqual(summary["cell_repeats"], 1.0)
        self.assertEqual(summary["grades_deg"], {"0": 400, "5": 200, "10": 200})
        self.assertEqual(summary["length_m"], {"min": 0.6, "max": 1.2})
        self.assertEqual(len(summary["work_condition_buckets"]), 15)
        self.assertLessEqual(max(summary["bucket_env_counts"]) -
                             min(summary["bucket_env_counts"]), 5)
        # 「连接 × 质量」计数用来核对 slot 轮转的均衡性：每个连接类型内各质量档相等
        # （compliant/rigid 各 64、inextensible 各 32），15 个组合全部非空
        masses = summary["mass_counts_by_connection"]
        self.assertEqual(sum(sum(counts.values()) for counts in masses.values()), 800)
        self.assertEqual(sum(masses["compliant"].values()), 320)
        self.assertEqual(sum(masses["rigid"].values()), 320)
        self.assertEqual(sum(masses["inextensible"].values()), 160)
        self.assertEqual(sum(len(counts) for counts in masses.values()), 15)
        self.assertEqual(sorted(masses["inextensible"]), ["10", "15", "20", "25", "5"])
        self.assertEqual({count for counts in masses.values() for count in counts.values()},
                         {32, 64})


class SpawnGeometryTests(unittest.TestCase):
    """出生几何：勾股解 + 逐 env 的挂点距自检（spawn 第一拍不能有约束力）。"""

    ROBOT_OFFSET = (-0.16, 0.0, 0.0)
    CART_OFFSET = (0.25, 0.0, 0.0)

    def _spawn(self, distances, *, robot_height=0.35, cart_height=0.18):
        return play.spawn_offsets(distances, robot_height=robot_height,
                                  cart_height=cart_height, robot_offset=self.ROBOT_OFFSET,
                                  cart_offset=self.CART_OFFSET)

    def test_attachment_along_is_the_pythagorean_leg(self):
        self.assertAlmostEqual(play.attachment_along(0.5, 0.3), 0.4, places=12)
        self.assertAlmostEqual(play.attachment_along(1.3, 0.5), 1.2, places=12)
        self.assertAlmostEqual(play.attachment_along(0.17, 0.0), 0.17, places=12)

    def test_impossible_geometry_is_rejected(self):
        for distance, difference in ((0.2, 0.3), (0.3, 0.3), (0.1, -0.5)):
            with self.assertRaises(ValueError):
                play.attachment_along(distance, difference)
        for bad in ((0.5, float("nan")), (float("inf"), 0.1), (0.0, 0.0), (-1.0, 0.0)):
            with self.assertRaises(ValueError):
                play.attachment_along(*bad)

    def test_spawn_keeps_the_attachment_distance_for_every_env(self):
        distances = [0.3, 0.575, 0.6, 0.9, 1.2]
        spec = self._spawn(distances)
        self.assertEqual(len(spec["cart_root"]), len(distances))
        for target, measured, along in zip(distances, spec["attachment_distance_m"],
                                           spec["along_m"]):
            self.assertAlmostEqual(measured, target, places=9)
            self.assertAlmostEqual(along, math.sqrt(target ** 2 -
                                                    spec["normal_difference_m"] ** 2), places=9)

    def test_robot_and_cart_roots_match_the_training_geometry(self):
        spec = self._spawn([0.5])
        # 法向高差 = (机器人挂点高 − 小车挂点高) = (0.35 + 0) − (0.18 + 0)
        self.assertAlmostEqual(spec["normal_difference_m"], 0.17, places=12)
        self.assertEqual(spec["robot_root"], (0.0, 0.0, 0.35))
        cart_x, cart_y, cart_z = spec["cart_root"][0]
        self.assertAlmostEqual(cart_z, 0.18, places=12)          # 小车仍落在平面上
        self.assertAlmostEqual(cart_y, self.ROBOT_OFFSET[1] - self.CART_OFFSET[1], places=12)
        expected_along = math.sqrt(0.5 ** 2 - 0.17 ** 2)
        self.assertAlmostEqual(cart_x, self.ROBOT_OFFSET[0] - self.CART_OFFSET[0] - expected_along,
                               places=12)
        self.assertLess(cart_x, 0.0)                             # 小车在机器人后方（−x）

    def test_rope_and_rigid_targets_differ_for_the_same_length(self):
        # env_spec: 绳 = 0.5·L0、刚体 = L ⇒ 同一行的两类连接挂点距差一倍
        rope_spec = connection_grid.env_spec(0)
        rigid_spec = next(connection_grid.env_spec(index) for index in range(800)
                          if connection_grid.env_spec(index)["model_name"] == "rigid"
                          and abs(connection_grid.env_spec(index)["length"]
                                  - rope_spec["length"]) < 1e-12)
        self.assertAlmostEqual(rigid_spec["initial_distance"], rope_spec["length"], places=12)
        self.assertAlmostEqual(rope_spec["initial_distance"], 0.5 * rope_spec["length"],
                               places=12)
        spec = self._spawn([rope_spec["initial_distance"], rigid_spec["initial_distance"]])
        self.assertLess(spec["along_m"][0], spec["along_m"][1])

    def test_bad_arguments_rejected(self):
        for kwargs in ({"robot_height": 0.0}, {"cart_height": -0.1},
                       {"robot_height": float("nan")}):
            with self.assertRaises(ValueError):
                self._spawn([0.5], **kwargs)
        with self.assertRaises(ValueError):
            self._spawn([])
        with self.assertRaises(ValueError):
            play.spawn_offsets([0.5], robot_height=0.35, cart_height=0.18,
                               robot_offset=(0.0, 0.0), cart_offset=(0.0, 0.0))
        with self.assertRaises(ValueError):
            play.spawn_offsets([0.5], robot_height=0.35, cart_height=0.18,
                               robot_offset=(0.0, 0.0, float("nan")),
                               cart_offset=(0.0, 0.0, 0.0))
        # 目标距小于法向高差 ⇒ 逐 env 报错（不能静默 NaN）
        with self.assertRaises(ValueError):
            self._spawn([0.1])


class CliTests(unittest.TestCase):
    def test_defaults_match_the_training_grid(self):
        args = play.parse_args([])
        self.assertEqual(list(args.velocities), list(play.DEFAULT_VELOCITIES))
        self.assertEqual(list(args.cart_masses), list(play.DEFAULT_CART_MASSES))
        self.assertEqual(args.num_envs, connection_grid.GRID_SIZE)
        self.assertEqual(args.ground_friction, play.FACTORY_FLOOR_FRICTION)
        self.assertEqual(args.wheel_damping, play.TRAINING_WHEEL_DAMPING)
        self.assertEqual(args.command_shaping, "direct")
        self.assertEqual(args.lane_keeping, "pd")
        self.assertEqual(args.record_every, play.DEFAULT_RECORD_EVERY)
        self.assertEqual(len(args.cases), connection_grid.GRID_SIZE)
        self.assertEqual(args.schedule.total_steps, 2200)

    def test_removed_scan_options_are_gone(self):
        for flag in ("--connections", "--slopes", "--slope-backend", "--slope-settle",
                     "--hold-damping", "--env-spacing", "--max-envs", "--rope-length",
                     "--slack", "--stiffness", "--damping", "--position-gain",
                     "--max-correction-rate", "--spawn-height", "--cart-drop"):
            with self.assertRaises(SystemExit):
                play.parse_args([flag, "1"])

    def test_dry_run_plan_lines(self):
        args = play.parse_args(["--dry-run"])
        text = "\n".join(play.planned_grid_lines(args))
        self.assertIn("训练场景", text)
        self.assertIn("800", text)
        self.assertIn("compliant 320", text)
        self.assertIn("0° 400", text)
        self.assertIn("env0000", text)
        self.assertIn("20, 25 kg 超出", text)
        self.assertIn("轮轴阻尼 0.032", text)
        self.assertIn("「连接 × 质量」env 数", text)

    def test_bad_values_rejected(self):
        for argv in (["--num-envs", "0"], ["--num-envs", "-5"],
                     ["--cart-masses", "1"], ["--cart-masses", "60"],
                     ["--velocities", "0"], ["--velocities", "3"],
                     ["--ground-friction", "0"], ["--ground-friction", "3"],
                     ["--record-every", "0"], ["--wheel-damping", "0"],
                     ["--transition-window", "10"], ["--lane-vy-limit", "0"],
                     ["--lane-kp-y", "-1"], ["--joint-rms-limit-rad", "0"]):
            with self.assertRaises(SystemExit):
                play.parse_args(argv)

    def test_partial_grid_prefix_is_allowed_with_a_warning(self):
        args = play.parse_args(["--num-envs", "40"])
        self.assertEqual(len(args.cases), 40)
        self.assertEqual([case.cell_index for case in args.cases], list(range(40)))

    def test_num_envs_multiple_of_the_grid_repeats_cells(self):
        args = play.parse_args(["--num-envs", "1600", "--velocities", "1.0",
                                "--cart-masses", "10"])
        self.assertEqual(len(args.cases), 1600)
        self.assertEqual(args.cases[0].cell_index, args.cases[800].cell_index)


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

    def test_scene_comes_from_the_training_config(self):
        # 场景必须是训练场景本体：不再自建材质的平地、也不再有自建地形生成器
        self.assertIn("UpperTowingSceneCfg(num_envs=num_envs", self.source)
        self.assertIn("UpperTowingEnvCfg()", self.source)
        self.assertIn("scene = InteractiveScene(scene_cfg)", self.source)
        for gone in ("TowSceneCfg(", "TerrainImporterCfg(", "TerrainGeneratorCfg(",
                     "PlayTestTileImporter", "PlayTestTileGenerator",
                     "build_terrain_layout", "slope_cells"):
            self.assertNotIn(gone, self.source)

    def test_per_env_parameters_come_from_the_training_grid(self):
        self.assertIn("connection_grid.env_spec(index)", self.source)
        self.assertIn("model_ids=model_ids", self.source)
        # 三套模型都要按逐 env 的 rest_length 构造（与 upper_mdp 同一写法）
        self.assertEqual(self.source.count("rest_length=connection_length"), 3)

    def test_contact_sensors_come_from_the_training_scene(self):
        self.assertIn("action_cfg.collision_sensor_names", self.source)
        self.assertIn("for sensor in wheel_sensors", self.source)
        # 车斗必须是第 0 项、其余是轮子（顺序变了就报错，不要静默记错列）
        self.assertIn('if "deck" not in action_cfg.collision_sensor_names[0]', self.source)
        self.assertIn('if "wheel" not in name', self.source)
        # 训练传感器是**按机器人 body 过滤**的 ⇒ 只能读 force_matrix_w（训练侧
        # cart_collision 同口径）；net_forces_w 是传感器 body 的净法向力，含轮地接触
        self.assertIn("sensor.data.force_matrix_w", self.source)
        self.assertNotIn(".data.net_forces_w", self.source)
        # 旧版自建的未过滤传感器在训练场景里不存在，不能再引用
        self.assertNotIn('scene["wheel_contacts"]', self.source)
        self.assertNotIn('scene["deck_contacts"]', self.source)

    def test_spawn_geometry_is_per_env_and_upright(self):
        self.assertIn("spawn_offsets(", self.source)
        self.assertIn("initial_distance.tolist()", self.source)
        # 出生姿态竖直：不能再有坡面出生旋转
        self.assertNotIn("rotation_y(", self.source)
        self.assertNotIn("spawn_on_surface(", self.source)

    def test_observation_gravity_is_normalised(self):
        # 重力恒为世界竖直（-z）；观测里的 projected_gravity 必须用单位向量
        self.assertIn("gravity_world = torch.tensor([0.0, 0.0, -1.0]", self.source)
        self.assertIn("per_env(gravity_world)", self.source)

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
        order = [loop.index("scene.write_data_to_sim()"), loop.index("sim.step(render=False)"),
                 loop.index("scene.update(dt)")]
        self.assertEqual(order, sorted(order))

    def test_physics_step_must_not_render(self):
        """物理步进必须 `render=False`：回归守卫，对应 2026-10-09 实跑定位到的真实 bug。

        `SimulationContext.step()` 的 render 默认是 True，那条分支走 `self._app.update()`：
        物理由 app/帧时序驱动，**一帧可能推进多个物理 tick**（实测首拍 base 就掉 7 mm、
        小腿关节转 0.1 rad ⇒ 等效 dt ≈ 50 ms 而不是 5 ms）。于是「每 4 次调用 = 一个
        20 ms 控制周期」的前提失效，按 200 Hz 标定的 PD（kp 25 / kd 0.5、腿链惯量
        ~6e-4 kg·m²）发散：关节正负交替打到 ±23.7 N·m 并撞到自身限位（髋 ±0.523、
        小腿 −3.0），机器人在最初几十毫秒内塌掉（绳/杆张力与车斗接触都是**后果**）。
        Isaac Lab 的 `ManagerBasedRLEnv` 物理步进一律 `sim.step(render=False)`，
        渲染单独放在 `render_interval` 那一拍 `sim.render()`。
        """
        self.assertIn("sim.step(render=False)", self.source)
        self.assertNotIn("sim.step()\n", self.source)
        self.assertIn("sim.render()", self.source)
        self.assertIn("render_interval = max(1, int(getattr(sim_cfg", self.source)

    def test_episode_reset_writes_poses_before_scene_reset(self):
        body = self.source.split("def reset_episode(")[1].split("def make_row")[0]
        self.assertLess(body.index("write_root_pose_to_sim"), body.index("scene.reset()"))

    def test_record_every_scales_the_contact_witness_threshold(self):
        # 800 环境下默认 5（25 ms）：`contact_witness` 必须把 5 ms 标定的速度跃变阈值放大
        self.assertEqual(play.parse_args([]).record_every, play.DEFAULT_RECORD_EVERY)
        rows = [base_row(load_vx_mps=0.0), base_row(load_vx_mps=0.06)]
        # 单步跃变 0.06 m/s：5 ms 记录下超限（0.015），25 ms 记录下不超限（0.075）
        self.assertTrue(play.contact_witness(rows, record_dt=0.005)["contact"])
        self.assertFalse(play.contact_witness(rows, record_dt=0.025)["contact"])

    def test_compact_log_caps_the_detail_lines(self):
        """`--compact-log` 必须**限行**，不能只按「非 OK 才打印」——全网格实测 800/800 都是非 OK。

        `startup_joint_error`/`stop_joint_error` 用的是未标定的占位阈值（见记录「无负载参考」一节），
        于是 800 个 case 里几乎全部非 OK；只按"非 OK 才打印"起不到压缩作用（2026-10-09 实跑刷了
        800 行）。现在按 `--compact-log-detail`（默认 30）截断明细，并每 100 个 case 打一条带判定码
        计数的进度行；逐 env 全量指标始终写在 `summaries/<case>.json` 与 `report.*` 里。
        """
        self.assertIn("--compact-log-detail", self.source)
        self.assertIn("compact_detail_printed < args.compact_log_detail", self.source)
        self.assertIn("tally.most_common()", self.source)
        self.assertIn("逐 case 明细最多打印", self.source)

    def test_no_second_gravity_write_path(self):
        # 训练场景的重力就是世界竖直，不需要再往 PhysX 里写重力
        self.assertNotIn("set_gravity(", self.source)
        self.assertNotIn("carb.Float3", self.source)


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
