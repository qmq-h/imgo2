"""`play_towing_test.py` 的离线测试：判据与网格逻辑不依赖 Isaac Sim。

本机（以及任何没有 GPU/Isaac Lab 的机器）只能验证**纯逻辑**部分：训练网格映射与工作条件
轮转、出生几何、阶段划分、指令整形、五项指标与横向 PD 的计算与判定、报告与矩阵、CLI 校验、
记录字段契约。仿真相位（场景构造、PhysX 步进、绳力施加、接触传感器）不在覆盖范围内 ——
那部分只能在训练机实跑，不要在文档里把本文件的通过写成「测试台已验证」。
"""
import ast
import importlib.util
import math
import statistics
import sys
import tempfile
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


def build_impact_episode(*, impact_error=0.05, steady_error=0.01, n=100, record_dt=0.005,
                         takeup_index=10, takeup_force=5.0, contact_index=30,
                         contact_force=12.0):
    """为冲击窗口统计合成一段轨迹：t<takeup 无张力、t≥takeup 有张力、coast 里有一个撞击。

    `n` 是 tow 段的行数（coast 段 40 行）。窗口边界用 `.5` 行错位取值（`0.025 s` = 5 行），
    这样「时间过滤」与「按行数取整」两种实现会给出**不同**的关节 RMS，边界测试才有分辨力。
    """
    rows = []
    for index in range(n):
        tension = takeup_force if index >= takeup_index else 0.0
        rows.append(base_row(
            phase="tow", time_s=(index + 0.5) * record_dt, robot_vx_b_mps=1.0,
            rope_tension_n=tension, load_vx_mps=1.0,
            **joint_columns(impact_error if index >= takeup_index else steady_error)))
    for index in range(40):
        speed = 1.0 if index < contact_index else 0.2
        rows.append(base_row(
            phase="coast", time_s=(n + index + 1) * record_dt, robot_vx_b_mps=0.0,
            rope_tension_n=0.0, load_vx_mps=speed,
            cart_deck_fx_n=contact_force if index == contact_index else 0.0,
            **joint_columns(impact_error if index >= contact_index else steady_error)))
    return rows


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


class ImpactWindowTests(unittest.TestCase):
    """冲击窗口独立统计（`impact_stats`）：纯逻辑、离线可测。

    四类时刻（起拖 / 绷直 / 指令归零 / 停车撞击）+ 稳态各自独立统计，并给「冲击 vs 稳态」的
    比值/增量。**这些字段不参与判定码**（`classify_case` 只读既有 startup/stop），
    所以这里的断言只钉统计口径，不钉安全性结论。
    """

    def test_defaults_are_the_documented_defaults(self):
        self.assertAlmostEqual(play.DEFAULT_IMPACT_WINDOW_S, 0.2, places=12)
        self.assertAlmostEqual(play.DEFAULT_STEADY_MARGIN_S, 1.0, places=12)
        self.assertAlmostEqual(play.DEFAULT_TAKEUP_FORCE_THRESHOLD_N, 1.0, places=12)

    def test_four_windows_align_to_different_moments(self):
        rows = build_impact_episode()
        stats = play.impact_stats(rows, record_dt=0.005, impact_window_s=0.025,
                                  steady_margin_s=0.05)
        # 起拖 = tow 首行；归零 = coast 首行（不是绷直/撞击时刻）
        self.assertAlmostEqual(stats["startup"]["time_s"], 0.0025, places=12)
        self.assertAlmostEqual(stats["stop"]["time_s"], 0.505, places=12)
        # 绷直 = 首个 |张力| 越阈（takeup_index=10 => (10+0.5)·5 ms）
        self.assertAlmostEqual(stats["takeup"]["time_s"], 0.0525, places=12)
        self.assertAlmostEqual(stats["takeup"]["tension_n"], 5.0, places=12)
        self.assertAlmostEqual(stats["takeup"]["vx_mps"], 1.0, places=12)
        self.assertAlmostEqual(stats["takeup"]["time_since_tow_start_s"], 0.05, places=12)
        # 停车撞击 = 首个接触见证（coast 第 30 行），与「指令归零」相差整整 150 ms
        self.assertTrue(stats["stop_contact"]["contact"])
        self.assertAlmostEqual(stats["stop_contact"]["time_s"], 0.655, places=12)
        self.assertAlmostEqual(stats["stop_contact"]["time_since_stop_s"], 0.15, places=12)
        self.assertGreater(stats["stop_contact"]["time_s"], stats["stop"]["time_s"])
        self.assertIn("deck_contact_force", stats["stop_contact"]["channels"])

    def test_window_uses_time_filtering_not_index_rounding(self):
        """窗口按 `time_s` 闭区间取：0.025 s = 5 个记录步，两种口径的样本数不同。"""
        rows = build_impact_episode()
        stats = play.impact_stats(rows, record_dt=0.005, impact_window_s=0.025,
                                  steady_margin_s=0.05)
        # [0.0025, 0.0275] 住 0.0025..0.0275 共 6 行；按行数取整会得到 5 行
        self.assertEqual(stats["startup"]["samples"], 6)
        # 围绕 t 的窗口是 ±W ⇒ 总宽 2W，两端各住 5 行 + 中心 1 行 = 11
        self.assertEqual(stats["takeup"]["samples"], 11)
        self.assertAlmostEqual(stats["takeup"]["window_s"], 0.05, places=12)
        self.assertAlmostEqual(stats["takeup"]["half_window_s"], 0.025, places=12)
        self.assertEqual(stats["startup"]["aligned"], "forward")
        self.assertEqual(stats["takeup"]["aligned"], "centred")

    def test_steady_margin_trims_both_ends_of_the_tow_phase(self):
        rows = build_impact_episode()
        stats = play.impact_stats(rows, record_dt=0.005, impact_window_s=0.025,
                                  steady_margin_s=0.05)
        # tow 段 0.0025..0.4975（100 行），去首尾各 0.05 ⇒ 剩 79 行
        self.assertEqual(stats["steady"]["samples"], 79)
        self.assertAlmostEqual(stats["steady"]["margin_s"], 0.05, places=12)
        self.assertAlmostEqual(stats["steady"]["start_s"], 0.0525, places=12)
        self.assertAlmostEqual(stats["steady"]["end_s"], 0.4475, places=12)
        self.assertAlmostEqual(stats["steady"]["span_s"], 0.495, places=12)
        self.assertAlmostEqual(stats["steady"]["joint_rms_rad"], 0.05, places=9)
        # 稳态不与自己比
        self.assertIsNone(stats["steady"]["rms_over_steady"])
        self.assertIsNone(stats["steady"]["rms_delta_rad"])

    def test_ratios_and_deltas_are_reported_against_steady(self):
        rows = build_impact_episode(impact_error=0.05, steady_error=0.02)
        stats = play.impact_stats(rows, record_dt=0.005, impact_window_s=0.025,
                                  steady_margin_s=0.05)
        steady = stats["steady"]["joint_rms_rad"]
        self.assertAlmostEqual(steady, 0.05, places=9)
        # 起拖窗口整段还在 0.02 的平静期 ⇒ 比值 0.4、增量 −0.03
        self.assertAlmostEqual(stats["startup"]["joint_rms_rad"], 0.02, places=9)
        self.assertAlmostEqual(stats["startup"]["rms_over_steady"], 0.4, places=9)
        self.assertAlmostEqual(stats["startup"]["rms_delta_rad"], -0.03, places=9)
        # 绷直窗口横跨误差跳变 ⇒ RMS 落在两端之间，比值 > 起拖
        self.assertGreater(stats["takeup"]["joint_rms_rad"], stats["startup"]["joint_rms_rad"])
        self.assertGreater(stats["takeup"]["rms_over_steady"],
                           stats["startup"]["rms_over_steady"])
        self.assertAlmostEqual(stats["takeup"]["rms_delta_rad"],
                               stats["takeup"]["joint_rms_rad"] - steady, places=9)
        # 逐关节比值也在（与合并比值同量级）
        per_joint = stats["takeup"]["per_joint_rms_over_steady"]
        self.assertEqual(len(per_joint), 12)
        for value in per_joint.values():
            self.assertAlmostEqual(value, stats["takeup"]["rms_over_steady"], places=9)

    def test_joint_rms_is_not_averaged_away_by_the_steady_window(self):
        """核心诉求：一个只有 2 行（0.01 s）的尖峰，稳态窗里看不到，冲击窗里看得见。"""
        rows = build_impact_episode()
        for joint in range(12):
            rows[3][f"robot_jp_{joint:02d}"] = rows[3][f"robot_jt_{joint:02d}"] + 0.5
        stats = play.impact_stats(rows, record_dt=0.005, impact_window_s=0.05,
                                  steady_margin_s=0.05)
        # 尖峰在 tow 第 4 行（t=0.0225）⇒ 落在起拖窗口 [0.0025, 0.0525] 内（峰值与 RMS 都跳）
        self.assertAlmostEqual(stats["startup"]["joint_max_rad"], 0.5, places=12)
        self.assertGreater(stats["startup"]["joint_rms_rad"], 0.10)
        # 稳态窗从 0.0575 起 ⇒ 尖峰整根被去掉，只剩 0.05 的常态误差
        self.assertAlmostEqual(stats["steady"]["joint_max_rad"], 0.05, places=9)
        self.assertAlmostEqual(stats["steady"]["joint_rms_rad"], 0.05, places=9)
        self.assertGreater(stats["startup"]["rms_over_steady"], 2.0)

    def test_no_takeup_sample_is_reported_not_crashed(self):
        rows = build_impact_episode(takeup_force=0.5)          # 阈值 1 N 时永远不越阈
        stats = play.impact_stats(rows, record_dt=0.005, impact_window_s=0.025,
                                  steady_margin_s=0.05)
        self.assertFalse(stats["takeup"]["available"])
        self.assertEqual(stats["takeup"].get("samples", 0), 0)
        self.assertIsNone(stats["takeup"]["joint_rms_rad"])
        self.assertIsNone(stats["takeup"]["rms_over_steady"])
        self.assertIn("没有 |rope_tension_n|", stats["takeup"]["note"])
        self.assertAlmostEqual(stats["takeup"]["force_threshold_n"], 1.0, places=12)
        # 其它三个窗口不受影响
        self.assertTrue(stats["startup"]["available"])
        self.assertTrue(stats["stop"]["available"])
        self.assertTrue(stats["stop_contact"]["available"])

    def test_negative_tension_counts_as_takeup_for_rigid_links(self):
        """rigid 球铰连杆张力有符号（压缩为负）⇒ 判定必须用模长。"""
        rows = build_impact_episode(takeup_force=0.0)
        for row in rows:
            if row["phase"] == "tow":
                row["rope_tension_n"] = -8.0
        stats = play.impact_stats(rows, record_dt=0.005, impact_window_s=0.025,
                                  steady_margin_s=0.05)
        self.assertTrue(stats["takeup"]["available"])
        self.assertAlmostEqual(stats["takeup"]["tension_n"], -8.0, places=12)
        self.assertAlmostEqual(stats["takeup"]["time_s"], 0.0025, places=12)

    def test_no_contact_sample_is_reported_not_crashed(self):
        rows = build_impact_episode(contact_force=0.0, contact_index=0)
        stats = play.impact_stats(rows, record_dt=0.005, impact_window_s=0.025,
                                  steady_margin_s=0.05)
        self.assertFalse(stats["stop_contact"]["available"])
        self.assertEqual(stats["stop_contact"].get("samples", 0), 0)
        self.assertFalse(stats["stop_contact"]["contact"])
        self.assertEqual(stats["stop_contact"]["channels"], [])
        self.assertIsNone(stats["stop_contact"]["rms_over_steady"])
        self.assertIn("没有接触见证", stats["stop_contact"]["note"])

    def test_velocity_jump_alone_is_a_contact_witness(self):
        rows = build_impact_episode(contact_force=0.0)
        coast_index = next(i for i, row in enumerate(rows) if row["phase"] == "coast")
        for row in rows[coast_index:]:
            row["load_vx_mps"] = 0.2
        rows[coast_index + 4]["load_vx_mps"] = 0.4     # 单步跃变 0.2 m/s > 0.015
        stats = play.impact_stats(rows, record_dt=0.005, impact_window_s=0.025,
                                  steady_margin_s=0.05)
        self.assertTrue(stats["stop_contact"]["available"])
        self.assertEqual(stats["stop_contact"]["channels"], ["load_velocity_jump"])
        self.assertAlmostEqual(stats["stop_contact"]["load_vx_mps"], 0.4, places=12)

    def test_window_beyond_the_record_end_is_clipped(self):
        rows = build_impact_episode()
        stats = play.impact_stats(rows, record_dt=0.005, impact_window_s=10.0,
                                  steady_margin_s=0.05)
        # 起拖窗口比整段都宽 ⇒ 只有 tow 段那 100 行（窗口不越 phase 边界）
        self.assertEqual(stats["startup"]["available"], True)
        # 窗口只吃 tow 段那 100 行，不会把 coast 段的行算进来（窗口不越 phase 边界）
        self.assertEqual(stats["startup"]["samples"], 100)
        self.assertTrue(stats["stop_contact"]["available"])

    def test_single_step_record(self):
        rows = [base_row(phase="tow", time_s=0.005, robot_vx_b_mps=1.0,
                         rope_tension_n=4.0, load_vx_mps=1.0,
                         **joint_columns(0.03))]
        stats = play.impact_stats(rows, record_dt=0.005, impact_window_s=0.2)
        self.assertEqual(stats["startup"]["samples"], 1)
        self.assertAlmostEqual(stats["startup"]["joint_rms_rad"], 0.03, places=12)
        self.assertAlmostEqual(stats["takeup"]["samples"], 1)
        # 没有 coast 段：归零/撞击/稳态都不能是「有数据」，但也不能崩
        self.assertFalse(stats["stop"]["available"])
        self.assertFalse(stats["stop_contact"]["available"])
        self.assertFalse(stats["steady"]["available"])
        self.assertIsNone(stats["startup"]["rms_over_steady"])

    def test_empty_record(self):
        stats = play.impact_stats([], record_dt=0.005, impact_window_s=0.2)
        for group in play.IMPACT_GROUPS:
            self.assertFalse(stats[group]["available"], group)
        self.assertTrue(stats["available"])

    def test_zero_torque_saturation_and_saturated_case(self):
        rows = build_impact_episode()
        stats = play.impact_stats(rows, record_dt=0.005, impact_window_s=0.025,
                                  steady_margin_s=0.05)
        self.assertAlmostEqual(stats["startup"]["torque_saturated_frac"], 0.0, places=12)
        # |τ| > 0.95 · 23.7 = 22.515 N·m
        rows = build_impact_episode()
        for index, row in enumerate(rows):
            if row["phase"] == "tow" and index < 10:
                for joint in range(12):
                    row[f"robot_tau_{joint:02d}"] = 30.0
        stats = play.impact_stats(rows, record_dt=0.005, impact_window_s=0.025,
                                  steady_margin_s=0.05)
        self.assertAlmostEqual(stats["startup"]["torque_saturated_frac"], 1.0, places=12)
        self.assertAlmostEqual(stats["steady"]["torque_saturated_frac"], 0.0, places=12)

    def test_invalid_parameters_rejected(self):
        rows = build_impact_episode()
        for kwargs in ({"impact_window_s": 0.0}, {"impact_window_s": float("nan")},
                       {"steady_margin_s": -1.0}, {"steady_margin_s": float("inf")}):
            with self.assertRaises(ValueError):
                play.impact_stats(rows, record_dt=0.005, **kwargs)
        with self.assertRaises(ValueError):
            play._takeup_index(rows, 0.0)

    def test_bad_parameters_must_not_change_the_verdict(self):
        """新增的冲击窗口参数**不得**影响判定：同一条轨迹换窗口，判定码与既有指标逐位一致。"""
        rows = build_impact_episode(contact_force=15.0)
        base = play.compute_case_metrics(
            rows, case=synthetic_case(), schedule=schedules(), record_dt=0.005,
            tow_summary={"valid": True, "failures": [], "min_clearance_coast_m": 0.05},
            thresholds=play.DEFAULT_THRESHOLDS, transition_window_s=0.05)
        other = play.compute_case_metrics(
            rows, case=synthetic_case(), schedule=schedules(), record_dt=0.005,
            tow_summary={"valid": True, "failures": [], "min_clearance_coast_m": 0.05},
            thresholds=play.DEFAULT_THRESHOLDS, transition_window_s=0.05,
            impact_window_s=0.9, steady_margin_s=0.0, takeup_force_threshold_n=0.01)
        self.assertEqual(base["verdict"], other["verdict"])
        self.assertEqual(base["startup"], other["startup"])
        self.assertEqual(base["stop"], other["stop"])
        # 冲击统计本身确实随参数变了（否则这条测试没意义）
        self.assertNotEqual(base["impact"]["window_s"], other["impact"]["window_s"])
        self.assertNotEqual(base["impact"]["startup"]["samples"],
                            other["impact"]["startup"]["samples"])
        self.assertTrue(other["impact"]["takeup"]["available"])

    def test_case_report_row_adds_impact_columns(self):
        rows = build_impact_episode()
        metrics = play.compute_case_metrics(
            rows, case=synthetic_case(), schedule=schedules(), record_dt=0.005,
            tow_summary={"valid": True, "failures": [], "min_clearance_coast_m": 0.3},
            thresholds=play.DEFAULT_THRESHOLDS, transition_window_s=0.05,
            impact_window_s=0.025, steady_margin_s=0.05)
        row = play.case_report_row(metrics)
        for group in play.IMPACT_GROUPS:
            for suffix in ("available", "joint_rms_rad", "rms_over_steady", "rms_delta_rad",
                           "torque_sat_frac", "samples"):
                self.assertIn(f"impact_{group}_{suffix}", row)
        self.assertIn("impact_takeup_tension_n", row)
        self.assertIn("impact_takeup_force_threshold_n", row)
        self.assertIn("impact_stop_contact_hit", row)
        self.assertIn("impact_stop_contact_channels", row)
        self.assertIsNone(row["impact_steady_rms_over_steady"])
        # 新增列不能顶掉既有列
        for key in ("startup_joint_rms_rad", "stop_joint_rms_rad", "verdict"):
            self.assertIn(key, row)

    def test_impact_is_reported_for_every_episode(self):
        metrics = play.compute_case_metrics(
            build_episode(tow_steps=500, coast_steps=200), case=synthetic_case(),
            schedule=play.make_schedule(settle_steps=20, tow_duration=2.5,
                                        coast_duration=1.0, dt=0.005), record_dt=0.005,
            tow_summary={"valid": True, "failures": [], "min_clearance_coast_m": 0.3},
            thresholds=play.DEFAULT_THRESHOLDS, transition_window_s=0.05)
        impact = metrics["impact"]
        self.assertTrue(impact["available"])
        self.assertAlmostEqual(impact["window_s"], play.DEFAULT_IMPACT_WINDOW_S, places=12)
        self.assertAlmostEqual(impact["steady_margin_s"], play.DEFAULT_STEADY_MARGIN_S,
                               places=12)
        self.assertAlmostEqual(impact["takeup_force_threshold_n"],
                               play.DEFAULT_TAKEUP_FORCE_THRESHOLD_N, places=12)
        self.assertTrue(impact["startup"]["available"])
        self.assertTrue(impact["steady"]["available"])

    def test_short_tow_phase_makes_the_steady_window_degenerate(self):
        """默认 margin 1.0 s 在 0.5 s 的短回合上取不到稳态 ⇒ 标为不可用、比值 None（不抛错）。"""
        metrics = play.compute_case_metrics(
            build_episode(), case=synthetic_case(), schedule=schedules(), record_dt=0.005,
            tow_summary={"valid": True, "failures": [], "min_clearance_coast_m": 0.3},
            thresholds=play.DEFAULT_THRESHOLDS, transition_window_s=0.05)
        impact = metrics["impact"]
        self.assertFalse(impact["steady"]["available"])
        self.assertIsNone(impact["steady"]["joint_rms_rad"])
        self.assertIsNone(impact["startup"]["rms_over_steady"])
        self.assertIsNone(impact["startup"]["rms_delta_rad"])
        # 既有指标与判定不受影响
        self.assertEqual(metrics["verdict"]["code"], "OK")


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
        """默认口径：JNT **不进判定**（用户 2026-10-10 决定），但观测值照旧在。

        `--count-jnt`（`count_jnt=True`）时必须逐位复现旧口径（判定码 `JNT` +
        `startup_joint_error`）；两条口径下 `startup.joint_rms_rad` 完全相同（同一批观测）。
        """
        rows = build_episode(startup_error=0.4)
        tow_summary = {"valid": True, "failures": [], "min_clearance_coast_m": 0.3}
        common = dict(case=synthetic_case(grade_deg=0.0, connection="compliant",
                                          cart_mass_kg=10.0),
                      schedule=schedules(), record_dt=0.005, tow_summary=tow_summary,
                      thresholds=play.DEFAULT_THRESHOLDS, transition_window_s=0.05)
        metrics = play.compute_case_metrics(rows, **common)
        # 新口径：不产生失败原因、判定码不是 JNT；观测值仍在（超过阈值但只是观测）
        self.assertEqual(metrics["verdict"]["code"], "OK")
        self.assertNotIn("startup_joint_error", metrics["verdict"]["reasons"])
        self.assertGreater(metrics["startup"]["joint_rms_rad"],
                           play.DEFAULT_THRESHOLDS["joint_rms_limit_rad"])
        # 旧口径（--count-jnt）：逐位复现
        legacy = play.compute_case_metrics(rows, count_jnt=True, **common)
        self.assertEqual(legacy["verdict"]["code"], "JNT")
        self.assertIn("startup_joint_error", legacy["verdict"]["reasons"])
        self.assertEqual(metrics["startup"]["joint_rms_rad"],
                         legacy["startup"]["joint_rms_rad"])

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

    def test_joint_error_is_not_counted_by_default(self):
        """新口径：`startup_joint_error`/`stop_joint_error` 不产生失败原因与判定码。

        合成 case 的关节 RMS/max 双双超限，其余全好 ⇒ 默认必须判 `OK` 且 `reasons == []`；
        `JNT` 只是观测（字段仍在 metrics 里）。
        """
        metrics = self._metrics(stability={"fell": False, "invalid": False},
                                stop={"contact": False, "gap_margin_low": False,
                                      "joint_rms_rad": 0.9, "joint_max_rad": 2.0},
                                speed={"available": True, "mae_mps": 0.01},
                                startup={"joint_rms_rad": 0.9, "joint_max_rad": 2.0})
        self.assertGreater(0.9, play.DEFAULT_THRESHOLDS["joint_rms_limit_rad"])
        self.assertGreater(2.0, play.DEFAULT_THRESHOLDS["joint_max_limit_rad"])
        verdict = play.classify_case(metrics, play.DEFAULT_THRESHOLDS)
        self.assertEqual(verdict["code"], "OK")
        self.assertEqual(verdict["reasons"], [])
        # 观测字段仍在（不是把指标删了）
        self.assertEqual(metrics["startup"]["joint_rms_rad"], 0.9)
        self.assertEqual(metrics["stop"]["joint_max_rad"], 2.0)

    def test_count_jnt_reproduces_the_legacy_caliber_bit_for_bit(self):
        """`count_jnt=True`（`--count-jnt`）与旧口径逐位一致：原因顺序、判定码都不变。"""
        metrics = self._metrics(stability={"fell": False, "invalid": False},
                                stop={"contact": False, "gap_margin_low": False,
                                      "joint_rms_rad": 0.9, "joint_max_rad": 2.0},
                                speed={"available": True, "mae_mps": 0.01},
                                startup={"joint_rms_rad": 0.9, "joint_max_rad": 2.0})
        legacy = play.classify_case(metrics, play.DEFAULT_THRESHOLDS, count_jnt=True)
        self.assertEqual(legacy["code"], "JNT")
        # 旧实现的顺序：先 startup/stop RMS，再 startup/stop max（去重）
        self.assertEqual(legacy["reasons"], ["startup_joint_error", "stop_joint_error"])
        self.assertEqual(legacy["severity"], play.VERDICT_SEVERITY["JNT"])
        # 只超单关节阈值（RMS 未超）也要复现
        only_max = self._metrics(stability={"fell": False, "invalid": False},
                                 stop={"contact": False, "gap_margin_low": False,
                                       "joint_rms_rad": 0.01, "joint_max_rad": 0.5},
                                 speed={"available": True, "mae_mps": 0.01},
                                 startup={"joint_rms_rad": 0.01, "joint_max_rad": 0.05})
        self.assertEqual(
            play.classify_case(only_max, play.DEFAULT_THRESHOLDS, count_jnt=True)["reasons"],
            ["stop_joint_error"])
        self.assertEqual(
            play.classify_case(only_max, play.DEFAULT_THRESHOLDS)["reasons"], [])
        # 默认口径就是 count_jnt=False（省参等价）
        self.assertEqual(play.classify_case(metrics, play.DEFAULT_THRESHOLDS),
                         play.classify_case(metrics, play.DEFAULT_THRESHOLDS, count_jnt=False))


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

    def test_markdown_report_excludes_jnt_but_keeps_the_observation_rows(self):
        """③ 新口径报告：不再有 JNT 失败行，但关节观测的中位数行与口径说明都在。

        用一条**关节误差超限**的合成轨迹走完整链路（`compute_case_metrics` → `group_statistics`
        → `build_markdown_report`）：默认口径下判定是 OK、失败模式「无」，
        `startup_joint_error` 不出现在报告文本里；`--count-jnt` 时才出现。
        """
        rows = build_episode(startup_error=0.4)
        metrics = play.compute_case_metrics(
            rows, case=synthetic_case(grade_deg=0.0, connection="compliant", cart_mass_kg=10.0),
            schedule=schedules(), record_dt=0.005,
            tow_summary={"valid": True, "failures": [], "min_clearance_coast_m": 0.3},
            thresholds=play.DEFAULT_THRESHOLDS, transition_window_s=0.05)
        summary = {"case": metrics["case"], "metrics": metrics, "verdict": metrics["verdict"]}
        groups = play.group_statistics([summary])
        conclusion = play.necessity_conclusion(groups, play.DEFAULT_THRESHOLDS)
        comparison = play.format_baseline_comparison(play.case_metric_summary([summary]))
        report = play.build_markdown_report(
            case_summaries=[summary], groups=groups, conclusion=conclusion,
            args_dict={"num_envs": 1, "velocities": [1.0], "cart_masses": [10.0],
                       "ground_friction": 0.8, "wheel_damping": 0.032,
                       "command_shaping": "direct", "ramp_time_s": 1.0,
                       "record_every": 5, "write_csv": "failed",
                       "lane_keeping": "pd", "lane_kp_y": 1.0, "lane_kd_y": 0.3,
                       "lane_kp_yaw": 1.5, "lane_kd_yaw": 0.3,
                       "lane_vy_limit": 0.4, "lane_wz_limit": 0.8,
                       "count_jnt": False},
            thresholds=play.DEFAULT_THRESHOLDS | {"count_jnt": False},
            schedule=play.make_schedule(settle_steps=200, tow_duration=5.0,
                                        coast_duration=5.0, dt=0.005),
            grades=[0.0], git={"commit": "deadbeef", "working_tree": None},
            comparison=comparison)
        # 失败原因以 `reason×count` 形式出现；`startup_joint_error` 只应出现在口径说明句里
        self.assertNotIn("startup_joint_error×", report)
        self.assertNotIn("stop_joint_error×", report)
        self.assertIn("判定口径：**JNT 不计入**", report)
        self.assertIn("**JNT 已按用户决定（2026-10-10）移出判定统计量**", report)
        self.assertIn("PD 静差", report)
        # 观测行仍在（五项指标中位数里两个关节项）
        self.assertIn("起步关节响应 RMS", report)
        self.assertIn("停车关节响应 RMS", report)
        self.assertIn("失败模式：无", report)
        # 打开旧口径后同样的轨迹渲染出 JNT 失败原因
        legacy = play.compute_case_metrics(
            rows, case=synthetic_case(grade_deg=0.0, connection="compliant", cart_mass_kg=10.0),
            schedule=schedules(), record_dt=0.005,
            tow_summary={"valid": True, "failures": [], "min_clearance_coast_m": 0.3},
            thresholds=play.DEFAULT_THRESHOLDS, transition_window_s=0.05, count_jnt=True)
        legacy_summary = {"case": legacy["case"], "metrics": legacy,
                          "verdict": legacy["verdict"]}
        legacy_groups = play.group_statistics([legacy_summary])
        legacy_report = play.build_markdown_report(
            case_summaries=[legacy_summary], groups=legacy_groups,
            conclusion=play.necessity_conclusion(legacy_groups, play.DEFAULT_THRESHOLDS),
            args_dict={"num_envs": 1, "velocities": [1.0], "cart_masses": [10.0],
                       "ground_friction": 0.8, "wheel_damping": 0.032,
                       "command_shaping": "direct", "ramp_time_s": 1.0,
                       "record_every": 5, "write_csv": "failed",
                       "lane_keeping": "pd", "lane_kp_y": 1.0, "lane_kd_y": 0.3,
                       "lane_kp_yaw": 1.5, "lane_kd_yaw": 0.3,
                       "lane_vy_limit": 0.4, "lane_wz_limit": 0.8,
                       "count_jnt": True},
            thresholds=play.DEFAULT_THRESHOLDS,
            schedule=play.make_schedule(settle_steps=200, tow_duration=5.0,
                                        coast_duration=5.0, dt=0.005),
            grades=[0.0], git={"commit": "deadbeef", "working_tree": None})
        self.assertIn("startup_joint_error×", legacy_report)
        self.assertIn("判定口径：**JNT 计入**", legacy_report)

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
        # 绳力/轮阻必须乘 present（只靠横向距离不够稳；训练侧 cart_present 同义）。
        # ⚠️ 必须 `present.unsqueeze(1)`：力是 (N, 3)、掩码是 (N,)，直接乘会报
        #   "The size of tensor a (3) must match the size of tensor b (800) at
        #    non-singleton dimension 1"（2026-10-09 首跑 800 env 实测踩到）。
        self.assertIn("* present.unsqueeze(1)", source)
        self.assertEqual(source.count("* present.unsqueeze(1)"), 3)   # 两个力 + 轮阻
        self.assertNotIn("], dim=-1) * present\n", source)
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
        # 绳 0.5–1.5 与杆 0.5–1.0 的并集
        self.assertEqual(summary["length_m"], {"min": 0.5, "max": 1.5})
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

    def test_rope_and_rigid_targets_follow_their_own_length_ranges(self):
        """2026-10-09：绳与杆长度区间解耦（绳 0.5–1.5、杆 0.5–1.0），出生距规则也不同。

        绳 = `SLACK_RATIO · L`（0.8，仍是松弛 ⇒ 第一拍无约束力）；杆 = `L`（全长 ⇒ C=0）。
        两者不再是"同一长度按比例"，所以旧断言"同一行的两类挂点距差一倍"已作废。
        """
        def spec_for(model, row):
            return next(connection_grid.env_spec(i) for i in range(800)
                        if connection_grid.env_spec(i)["model_name"] == model
                        and connection_grid.env_spec(i)["row"] == row)
        for row in (0, connection_grid.ROWS // 2, connection_grid.ROWS - 1):
            rope, rigid = spec_for("compliant", row), spec_for("rigid", row)
            self.assertAlmostEqual(rope["initial_distance"],
                                   connection_grid.SLACK_RATIO * rope["length"], places=12)
            self.assertAlmostEqual(rigid["initial_distance"], rigid["length"], places=12)
            # 同一行：绳不短于杆（区间上端 1.5 > 1.0，下端都是 0.5）
            self.assertGreaterEqual(rope["length"], rigid["length"] - 1e-12)
        # 出生比 0.8 ⇒ 最短绳行的真实实体间隙不再贴脸（旧的 0.5 只有 9 mm）
        self.assertAlmostEqual(connection_grid.SLACK_RATIO, 0.8, places=12)

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
        # 默认速度下界与训练域下界（`episode_geometry.SPEED_RANGE[0]`）一致
        self.assertEqual(play.DEFAULT_VELOCITIES[0], 0.5)
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

    def test_count_jnt_defaults_off_and_plan_line_shows_the_caliber(self):
        """`--count-jnt` 默认关；plan 行必须打印当前口径（JNT 计入/不计入）。"""
        args = play.parse_args(["--dry-run"])
        self.assertFalse(args.count_jnt)
        self.assertFalse(play.DEFAULT_COUNT_JNT)
        text = "\n".join(play.planned_grid_lines(args))
        self.assertIn("判定口径：**JNT 不计入**", text)
        self.assertIn("`--count-jnt`", text)
        # 阈值参数保留（仍用于生成观测标记）
        self.assertIn("--joint-rms-limit-rad", text)
        self.assertIn("--joint-max-limit-rad", text)
        # 口径标记跟阈值一起落到 report.json / experiment.json
        thresholds = play.caliber_thresholds(args)
        self.assertIs(thresholds["count_jnt"], False)
        self.assertEqual(thresholds["joint_rms_limit_rad"], 0.10)
        self.assertEqual(thresholds["joint_max_limit_rad"], 0.30)
        # 打开后 plan 行与标记都翻面
        opened = play.parse_args(["--dry-run", "--count-jnt"])
        self.assertTrue(opened.count_jnt)
        self.assertIs(play.caliber_thresholds(opened)["count_jnt"], True)
        opened_text = "\n".join(play.planned_grid_lines(opened))
        self.assertIn("判定口径：**JNT 计入**", opened_text)
        self.assertIn("逐位一致", opened_text)

    def test_dry_run_plan_lines(self):
        args = play.parse_args(["--dry-run"])
        text = "\n".join(play.planned_grid_lines(args))
        self.assertIn("训练场景", text)
        self.assertIn("800", text)
        self.assertIn("compliant 320", text)
        self.assertIn("0° 400", text)
        self.assertIn("env0000", text)
        # 训练上限 2026-10-10 收紧为 20 kg ⇒ 默认档里的 25 kg 是域外外推检查
        self.assertEqual(play.TRAINING_MASS_RANGE_KG, (5.0, 20.0))
        self.assertIn("超出该范围的质量档 25 kg", text)
        self.assertIn("外推检查，判读要与分布内档位分开", text)
        self.assertNotIn("本网格的质量档全部落在训练分布内", text)
        self.assertIn("轮轴阻尼 0.032", text)
        self.assertIn("「连接 × 质量」env 数", text)

    def test_impact_window_defaults_and_plan_lines(self):
        args = play.parse_args(["--dry-run"])
        self.assertAlmostEqual(args.impact_window, play.DEFAULT_IMPACT_WINDOW_S, places=12)
        self.assertAlmostEqual(args.steady_margin_s, play.DEFAULT_STEADY_MARGIN_S, places=12)
        self.assertAlmostEqual(args.takeup_force_threshold,
                               play.DEFAULT_TAKEUP_FORCE_THRESHOLD_N, places=12)
        text = "\n".join(play.planned_grid_lines(args))
        self.assertIn("冲击窗口独立统计", text)
        self.assertIn("--impact-window", text)
        self.assertIn("--steady-margin-s", text)
        self.assertIn("--takeup-force-threshold", text)
        self.assertIn("`--impact-window` 0.2 s", text)
        self.assertIn("`--takeup-force-threshold` 1 N", text)
        # 口径并存的说明必须在 plan 里（避免读者以为既有 startup/stop 被换掉了）
        self.assertIn("既有 `startup.*`/`stop.*` 语义逐位不变", text)

    def test_impact_window_parameters_change_the_plan_line(self):
        args = play.parse_args(["--dry-run", "--impact-window", "0.05",
                                "--steady-margin-s", "0.5",
                                "--takeup-force-threshold", "2.5"])
        text = "\n".join(play.planned_grid_lines(args))
        self.assertIn("`--impact-window` 0.05 s", text)
        self.assertIn("`--steady-margin-s` 0.5 s", text)
        self.assertIn("`--takeup-force-threshold` 2.5 N", text)

    def test_impact_window_bad_values_rejected(self):
        for argv in (["--impact-window", "0"], ["--impact-window", "-0.1"],
                     ["--impact-window", "10"],          # > tow-duration 5.0
                     ["--steady-margin-s", "-1"],
                     ["--takeup-force-threshold", "0"],
                     ["--takeup-force-threshold", "-1"]):
            with self.assertRaises(SystemExit):
                play.parse_args(argv)
        # 0 是合法的「不设边距」
        self.assertEqual(play.parse_args(["--steady-margin-s", "0"]).steady_margin_s, 0.0)

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

    def test_upper_switch_defaults_off(self):
        """开关默认关：不给路径就是现在的基线（残差恒 0），确定性取均值。"""
        args = play.parse_args([])
        self.assertIsNone(args.upper_checkpoint)
        self.assertFalse(args.upper_stochastic)

    def test_upper_plan_lines_show_the_switch(self):
        baseline = play.parse_args(["--dry-run"])
        text = "\n".join(play.planned_grid_lines(baseline))
        self.assertIn("上层网络：**关闭**", text)
        self.assertIn("残差恒 0", text)
        # 打开：需要真实存在的 checkpoint（parse_args 会在启动 Isaac Sim 之前校验）
        with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
            args = play.parse_args(["--dry-run", "--upper-checkpoint", handle.name])
            self.assertEqual(args.upper_checkpoint, Path(handle.name).resolve())
            opened = "\n".join(play.planned_grid_lines(args))
            stochastic_args = play.parse_args(
                ["--upper-checkpoint", handle.name, "--upper-stochastic"])
            self.assertTrue(stochastic_args.upper_stochastic)
            stochastic = "\n".join(play.planned_grid_lines(stochastic_args))
        self.assertIn("上层网络：**开启**", opened)
        self.assertIn("确定性均值", opened)
        self.assertIn("towing_contract", opened)
        self.assertIn("frame_dim=57", opened)
        self.assertIn("57", opened)
        self.assertIn("随机采样", stochastic)

    def test_upper_switch_bad_values_rejected(self):
        with self.assertRaises(SystemExit):
            play.parse_args(["--upper-stochastic"])              # 没给 checkpoint
        with self.assertRaises(SystemExit):
            play.parse_args(["--upper-checkpoint", "/definitely/not/here.pt"])

    def test_compare_report_cli(self):
        self.assertIsNone(play.parse_args([]).compare_report)
        with self.assertRaises(SystemExit):
            play.parse_args(["--compare-report", "/definitely/not/here.json"])
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            bad.write_text('{"groups": {}}', encoding="utf-8")
            with self.assertRaises(SystemExit):
                play.parse_args(["--compare-report", str(bad)])
            good = Path(tmp) / "good.json"
            good.write_text('{"cases": []}', encoding="utf-8")
            args = play.parse_args(["--compare-report", str(good)])
            self.assertEqual(args.compare_report, good.resolve())
            plan = "\n".join(play.planned_grid_lines(args))
        self.assertIn("同版本基线对照", plan)
        self.assertIn(str(good.resolve()), plan)


class BaselineComparisonTests(unittest.TestCase):
    """「策略 vs 基线」一节：同口径汇总（不给策略新造指标/不改阈值）+ 归档基线参照。"""

    @staticmethod
    def _summary(grade, code, *, startup=0.1, speed=0.2, coast=0.3, clearance=0.4,
                 stop_joint=0.05):
        metrics = {
            "case": {"grade_deg": grade, "velocity_mps": 1.0, "connection": "compliant",
                     "cart_mass_kg": 10.0, "cart_present": True},
            "verdict": {"code": code, "reasons": []},
            "startup": {"joint_rms_rad": startup},
            "speed": {"mae_mps": speed},
            "stop": {"cart_coast_distance_m": coast, "min_clearance_coast_m": clearance,
                     "joint_rms_rad": stop_joint},
        }
        return {"case": metrics["case"], "metrics": metrics, "verdict": metrics["verdict"]}

    @staticmethod
    def _args_dict():
        return {"num_envs": 800, "velocities": [1.5], "cart_masses": [25.0],
                "ground_friction": 0.8, "wheel_damping": 0.032,
                "command_shaping": "direct", "ramp_time_s": 1.0,
                "record_every": 5, "write_csv": "failed",
                "lane_keeping": "pd", "lane_kp_y": 1.0, "lane_kd_y": 0.3,
                "lane_kp_yaw": 1.5, "lane_kd_yaw": 0.3,
                "lane_vy_limit": 0.4, "lane_wz_limit": 0.8}

    def test_summary_counts_codes_and_medians(self):
        cases = [self._summary(0.0, "OK", speed=0.1),
                 self._summary(0.0, "COL", speed=0.3),
                 self._summary(5.0, "SPD", speed=0.2, clearance=None)]
        summary = play.case_metric_summary(cases)
        self.assertEqual(summary["envs"], 3)
        self.assertEqual(summary["ok"], 1)
        self.assertEqual(summary["codes"], {"OK": 1, "COL": 1, "SPD": 1})
        self.assertAlmostEqual(summary["medians"]["speed.mae_mps"], 0.2, places=9)
        # None（无小车环境没有几何间隙）不能进中位数，但样本数要如实报
        self.assertAlmostEqual(summary["medians"]["stop.min_clearance_coast_m"], 0.4, places=9)
        self.assertEqual(summary["samples"]["stop.min_clearance_coast_m"], 2)
        self.assertEqual(summary["by_grade"]["grade0"]["cases"], 2)
        self.assertEqual(summary["by_grade"]["grade0"]["ok"], 1)
        self.assertEqual(summary["by_grade"]["grade5"]["ok"], 0)

    def test_archived_constants_match_the_archived_report(self):
        """参照列的常量必须等于归档 report.json 的复算值（防手抄漂移）。"""
        path = play.REPO / play.ARCHIVED_BASELINE_2026_10_09["report"]
        if not path.is_file():
            self.skipTest(f"归档基线不在工作区：{path}")
        import json as _json
        summary = play.case_metric_summary(
            _json.loads(path.read_text(encoding="utf-8"))["cases"])
        archived = play.ARCHIVED_BASELINE_2026_10_09
        self.assertEqual(summary["envs"], archived["num_envs"])
        self.assertEqual(summary["ok"], archived["ok"])
        self.assertEqual(summary["codes"], archived["codes"])
        for key, value in archived["medians"].items():
            self.assertAlmostEqual(summary["medians"][key], value, places=6)
        for key, value in archived["median_samples"].items():
            self.assertEqual(summary["samples"][key], value)
        self.assertEqual({name: (entry["cases"], entry["ok"])
                          for name, entry in summary["by_grade"].items()},
                         archived["by_grade"])

    def test_comparison_section_mentions_every_field_and_flags_jnt(self):
        current = play.case_metric_summary([self._summary(0.0, "COL")])
        reference = play.case_metric_summary([self._summary(0.0, "OK")])
        text = "\n".join(play.format_baseline_comparison(current, reference=reference,
                                                         upper_enabled=True))
        self.assertIn("## 策略 vs 基线", text)
        self.assertIn("上层策略驱动", text)
        self.assertIn("同版本基线", text)
        for _group, _field, label in play.FIVE_METRIC_FIELDS:
            self.assertIn(label, text)
        # 同版本基线列（通过 1/1）与本轮列（0/1）都在
        self.assertIn("通过 **0/1**", text)
        self.assertIn("通过 **1/1**", text)
        # 归档基线参照 + 出处 + 不可逐格硬比 + JNT 说明
        self.assertIn(play.ARCHIVED_BASELINE_2026_10_09["label"], text)
        self.assertIn("1bbb405", text)
        self.assertIn("6fac89a", text)
        self.assertIn("不可逐格硬比", text)
        # 默认口径：JNT 移出统计量 + 归档旧口径不可比 + 新口径重判读数
        self.assertIn("判定口径：**JNT 不计入**", text)
        self.assertIn("`JNT` 已移出判定统计量", text)
        self.assertIn("PD 静差", text)
        self.assertIn("不是同一个量", text)
        self.assertIn("口径不可比（JNT）", text)
        recount = play.JNT_EXCLUDED_RECOUNT_2026_10_10
        self.assertIn(f"基线 **{recount['ok']['baseline']}/{recount['num_envs']}**", text)
        self.assertIn(f"策略 **{recount['ok']['policy']}/{recount['num_envs']}**", text)
        # 关节观测的中位数行必须仍在（观测，不是判据）
        self.assertIn("起步关节响应 RMS", text)
        self.assertIn("停车关节响应 RMS", text)
        # 打开 --count-jnt 时口径行改成「计入」
        opened = "\n".join(play.format_baseline_comparison(current, count_jnt=True))
        self.assertIn("判定口径：**JNT 计入**", opened)

    def test_switch_does_not_add_metrics_or_thresholds(self):
        """验收口径必须与基线同一套：判据/指标/阈值函数里不得出现开关或新参数分支。"""
        source = (RL / "scripts/towing/play_towing_test.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        tokens = ("upper_checkpoint", "upper_delta", "UpperPolicyRuntime", "upper.act(",
                  "upper_stochastic", "compare_report",
                  # 冲击窗口的新参数同样不得进判据/阈值（只进 metrics["impact"] 与报告）；
                  # 它们的取值只允许出现在 compute_case_metrics 的签名/转发与报告层。
                  "impact_window", "steady_margin_s", "takeup_force_threshold")
        for name in ("classify_case", "joint_track_stats", "speed_track_stats", "stop_stats",
                     "stability_stats", "lane_stats", "group_statistics",
                     "necessity_conclusion"):
            node = next(node for node in ast.walk(tree)
                        if isinstance(node, ast.FunctionDef) and node.name == name)
            segment = ast.get_source_segment(source, node) or ""
            for token in tokens:
                self.assertNotIn(token, segment, f"{name} 里不应出现开关/新参数 {token}")
        # `compute_case_metrics` / `case_report_row` 只是**转发**新参数（签名默认值 + 传进
        # `impact_stats`）与**抄写**结果，所以只对「开关」token 设黑名单；新参数本身必须在那里
        # 出现（否则新统计永远拿不到 CLI 的值）。
        for name in ("compute_case_metrics", "case_report_row"):
            node = next(node for node in ast.walk(tree)
                        if isinstance(node, ast.FunctionDef) and node.name == name)
            segment = ast.get_source_segment(source, node) or ""
            for token in tokens[:6]:
                self.assertNotIn(token, segment, f"{name} 里不应出现开关相关代码 {token}")
        # 新参数在 `compute_case_metrics` 里只准出现在签名与一次转发里：数一数引用次数上限，
        # 防止有人把窗口/阈值拿去做分支（例如 `if impact_window_s > x:` 改判据）。
        compute_source = ast.get_source_segment(
            source, next(node for node in ast.walk(tree)
                         if isinstance(node, ast.FunctionDef)
                         and node.name == "compute_case_metrics")) or ""
        # 各 3 次 = 签名形参 + 转发关键字 + 转发值；多出来就说明有人拿去做了别的判断
        self.assertEqual(compute_source.count("impact_window_s"), 3)
        self.assertEqual(compute_source.count("steady_margin_s"), 3)
        self.assertEqual(compute_source.count("takeup_force_threshold_n"), 3)
        # 判定口径开关 `count_jnt` 同理：只准「签名 + 一次转发给 classify_case」，
        # 不得在指标计算里做分支（口径只影响 verdict，不影响任何观测值）。
        self.assertEqual(compute_source.count("count_jnt"), 3)
        # 口径标记只由 `caliber_thresholds()` 生成，且不塞进 DEFAULT_THRESHOLDS（阈值数值冻结）
        self.assertNotIn("count_jnt", play.DEFAULT_THRESHOLDS)
        self.assertIs(play.caliber_thresholds(play.parse_args([]))["count_jnt"], False)
        thresholds_node = next(node for node in tree.body if isinstance(node, ast.Assign)
                               and any(isinstance(target, ast.Name)
                                       and target.id == "DEFAULT_THRESHOLDS"
                                       for target in node.targets))
        thresholds_source = ast.get_source_segment(source, thresholds_node) or ""
        self.assertNotIn("upper", thresholds_source)
        # 阈值口径冻结：默认阈值一个字都不变（归档对照依赖它们）
        self.assertEqual(set(play.DEFAULT_THRESHOLDS), {
            "joint_rms_limit_rad", "joint_max_limit_rad", "speed_mae_ratio_limit",
            "gap_margin_limit_m", "fall_height_limit_m", "pitch_limit_rad",
            "pitch_fraction_limit", "lane_y_limit_m", "lane_heading_limit_deg"})
        self.assertEqual(play.DEFAULT_THRESHOLDS["joint_rms_limit_rad"], 0.10)
        self.assertEqual(play.DEFAULT_THRESHOLDS["joint_max_limit_rad"], 0.30)
        self.assertEqual(play.VERDICT_CODES, ("OK", "LOW", "LAT", "JNT", "SPD", "COL",
                                              "FALL", "INV"))
        # 冲击统计函数里不得读阈值字典（它是观测，不是判据）
        impact_node = next(node for node in ast.walk(tree)
                           if isinstance(node, ast.FunctionDef) and node.name == "impact_stats")
        impact_source = ast.get_source_segment(source, impact_node) or ""
        self.assertNotIn("thresholds", impact_source)
        # 五项指标就是 docstring 里那五项（顺序即报告顺序）
        self.assertEqual([field for _group, field, _label in play.FIVE_METRIC_FIELDS],
                         ["joint_rms_rad", "mae_mps", "cart_coast_distance_m",
                          "min_clearance_coast_m", "joint_rms_rad"])
        self.assertEqual([group for group, _field, _label in play.FIVE_METRIC_FIELDS],
                         ["startup", "speed", "stop", "stop", "stop"])

    def _impact_summary(self, grade, code, *, startup_rms=0.2, steady=0.1, takeup_available=True,
                        takeup_rms=0.3, contact_available=False, contact_rms=None):
        metrics = self._summary(grade, code)["metrics"]
        metrics["impact"] = {
            "available": True, "window_s": 0.2, "steady_margin_s": 1.0,
            "takeup_force_threshold_n": 1.0,
            "startup": {"available": True, "samples": 8, "joint_rms_rad": startup_rms,
                        "joint_max_rad": startup_rms, "worst_joint": "FL_hip_joint",
                        "per_joint_rms_rad": {"FL_hip_joint": startup_rms},
                        "torque_saturated_frac": 0.01, "rms_over_steady": startup_rms / steady,
                        "rms_delta_rad": startup_rms - steady},
            "takeup": {"available": takeup_available,
                       "samples": 12 if takeup_available else 0,
                       "joint_rms_rad": takeup_rms if takeup_available else None,
                       "joint_max_rad": takeup_rms if takeup_available else None,
                       "worst_joint": "FL_thigh_joint" if takeup_available else None,
                       "per_joint_rms_rad": None,
                       "torque_saturated_frac": 0.05 if takeup_available else None,
                       "rms_over_steady": (takeup_rms / steady) if takeup_available else None,
                       "rms_delta_rad": (takeup_rms - steady) if takeup_available else None},
            "stop": {"available": True, "samples": 8, "joint_rms_rad": startup_rms * 2,
                     "joint_max_rad": startup_rms * 3, "worst_joint": "RL_shank_joint",
                     "per_joint_rms_rad": None, "torque_saturated_frac": 0.02,
                     "rms_over_steady": 2.0 * startup_rms / steady,
                     "rms_delta_rad": 2.0 * startup_rms - steady},
            "stop_contact": {"available": contact_available,
                             "samples": 11 if contact_available else 0,
                             "joint_rms_rad": contact_rms if contact_available else None,
                             "joint_max_rad": contact_rms if contact_available else None,
                             "worst_joint": "RR_thigh_joint" if contact_available else None,
                             "per_joint_rms_rad": None, "torque_saturated_frac": None,
                             "rms_over_steady": ((contact_rms / steady)
                                                 if contact_available else None),
                             "rms_delta_rad": ((contact_rms - steady)
                                               if contact_available else None)},
            "steady": {"available": True, "samples": 300, "joint_rms_rad": steady,
                       "joint_max_rad": steady * 2, "worst_joint": "FL_hip_joint",
                       "per_joint_rms_rad": {"FL_hip_joint": steady},
                       "torque_saturated_frac": 0.0, "rms_over_steady": None,
                       "rms_delta_rad": None},
        }
        return {"case": metrics["case"], "metrics": metrics, "verdict": metrics["verdict"]}

    def test_impact_medians_and_samples_are_summarised(self):
        summaries = [self._impact_summary(0.0, "OK", startup_rms=0.2, steady=0.1),
                     self._impact_summary(0.0, "COL", startup_rms=0.4, steady=0.2),
                     self._impact_summary(5.0, "SPD", startup_rms=0.6, steady=0.3,
                                          takeup_available=False, contact_available=True,
                                          contact_rms=0.9)]
        summary = play.case_metric_summary(summaries)
        self.assertAlmostEqual(summary["impact_medians"]["impact.startup.joint_rms_rad"],
                               0.4, places=9)
        self.assertAlmostEqual(summary["impact_medians"]["impact.steady.joint_rms_rad"],
                               0.2, places=9)
        # 起拖比值 = 2.0（三个 case 都是 2.0）
        self.assertAlmostEqual(summary["impact_medians"]["impact.startup.rms_over_steady"],
                               2.0, places=9)
        # 绷直只有 2 个 case 有样本（第三个 available=False）⇒ 样本数如实报
        self.assertEqual(summary["impact_samples"]["impact.takeup.joint_rms_rad"], 2)
        self.assertEqual(summary["impact_samples"]["impact.stop_contact.joint_rms_rad"], 1)
        self.assertAlmostEqual(
            summary["impact_medians"]["impact.stop_contact.joint_rms_rad"], 0.9, places=9)
        # 最差关节众数（起拖 3/3 都是 FL_hip_joint）
        self.assertEqual(summary["impact_modal_worst_joint"]["startup"], "FL_hip_joint")
        self.assertEqual(summary["impact_modal_worst_joint"]["takeup"], "FL_thigh_joint")
        # 窗口参数随逐 case 的 impact 一起带出来
        self.assertAlmostEqual(summary["impact_window_s"], 0.2, places=12)
        self.assertAlmostEqual(summary["impact_steady_margin_s"], 1.0, places=12)
        self.assertAlmostEqual(summary["impact_takeup_force_threshold_n"], 1.0, places=12)

    def test_impact_medians_are_none_without_impact_metrics(self):
        """归档基线（旧版本 report.json）没有 impact.* ⇒ 必须显示 n/a 而不是崩。"""
        summary = play.case_metric_summary([self._summary(0.0, "OK")])
        self.assertIsNone(summary["impact_medians"]["impact.startup.joint_rms_rad"])
        self.assertEqual(summary["impact_samples"]["impact.startup.joint_rms_rad"], 0)
        self.assertIsNone(summary["impact_window_s"])
        self.assertIsNone(summary["impact_modal_worst_joint"]["startup"])

    def test_impact_section_renders_four_moments_plus_steady(self):
        summaries = [self._impact_summary(0.0, "COL", contact_available=True, contact_rms=0.9)]
        current = play.case_metric_summary(summaries)
        reference = play.case_metric_summary([self._impact_summary(0.0, "OK")])
        text = "\n".join(play.format_impact_section(
            current, reference=reference, archived=play.ARCHIVED_BASELINE_2026_10_09))
        self.assertIn("## 冲击窗口 vs 稳态（关节响应）", text)
        for token in ("起拖", "绷直", "指令归零", "停车撞击", "稳态", "RMS/稳态", "RMS−稳态",
                      "不参与判定", "最差关节（众数）", "FL_hip_joint"):
            self.assertIn(token, text)
        self.assertIn("W = 0.2 s", text)
        self.assertIn("稳态 margin = 1 s", text)
        self.assertIn("绷直阈值 = 1 N", text)
        # 同版本基线列与归档缺字段说明
        self.assertIn("同版本基线", text)
        self.assertIn("该归档没有 `impact.*` 字段", text)
        # 有/无数据的行都要出现（绷直不可用 => n/a）
        self.assertIn("n/a", text)

    def test_impact_section_without_any_impact_data_still_renders(self):
        current = play.case_metric_summary([self._summary(0.0, "OK")])
        text = "\n".join(play.format_impact_section(current))
        self.assertIn("## 冲击窗口 vs 稳态（关节响应）", text)
        self.assertIn("n/a", text)
        self.assertNotIn("同版本基线", text)

    def test_markdown_report_renders_the_impact_section(self):
        summaries = [self._impact_summary(0.0, "COL", contact_available=True, contact_rms=0.9)]
        groups = play.group_statistics(summaries)
        conclusion = play.necessity_conclusion(groups, play.DEFAULT_THRESHOLDS)
        current = play.case_metric_summary(summaries)
        comparison = play.format_baseline_comparison(current)
        impact = play.format_impact_section(current, archived=play.ARCHIVED_BASELINE_2026_10_09)
        report = play.build_markdown_report(
            case_summaries=summaries, groups=groups, conclusion=conclusion,
            args_dict=self._args_dict(), thresholds=play.DEFAULT_THRESHOLDS,
            schedule=play.make_schedule(settle_steps=200, tow_duration=5.0,
                                        coast_duration=5.0, dt=0.005),
            grades=[0.0], git={"commit": "deadbeef", "working_tree": None},
            comparison=comparison, impact=impact)
        self.assertIn("## 冲击窗口 vs 稳态（关节响应）", report)
        self.assertIn("impact-window", report)
        self.assertLess(report.index("## 分组统计"), report.index("## 冲击窗口"))
        self.assertLess(report.index("## 冲击窗口"), report.index("## 结论"))
        self.assertLess(report.index("## 结论"), report.index("## 策略 vs 基线"))
        # 不给 impact 时保持旧结构（向后兼容）
        without = play.build_markdown_report(
            case_summaries=summaries, groups=groups, conclusion=conclusion,
            args_dict=self._args_dict(), thresholds=play.DEFAULT_THRESHOLDS,
            schedule=play.make_schedule(settle_steps=200, tow_duration=5.0,
                                        coast_duration=5.0, dt=0.005),
            grades=[0.0], git={"commit": "deadbeef", "working_tree": None})
        self.assertNotIn("## 冲击窗口", without)

    def test_markdown_report_renders_the_comparison_section(self):
        summaries = [self._summary(0.0, "COL", speed=0.3)]
        groups = play.group_statistics(summaries)
        conclusion = play.necessity_conclusion(groups, play.DEFAULT_THRESHOLDS)
        comparison = play.format_baseline_comparison(play.case_metric_summary(summaries))
        report = play.build_markdown_report(
            case_summaries=summaries, groups=groups, conclusion=conclusion,
            args_dict=self._args_dict(), thresholds=play.DEFAULT_THRESHOLDS,
            schedule=play.make_schedule(settle_steps=200, tow_duration=5.0,
                                        coast_duration=5.0, dt=0.005),
            grades=[0.0], git={"commit": "deadbeef", "working_tree": None},
            comparison=comparison)
        self.assertIn("## 策略 vs 基线", report)
        self.assertIn("## 限制", report)
        # 对照节在结论之后、限制之前
        self.assertLess(report.index("## 结论"), report.index("## 策略 vs 基线"))
        self.assertLess(report.index("## 策略 vs 基线"), report.index("## 限制"))
        # 不给 comparison 时保持旧结构（向后兼容）
        without = play.build_markdown_report(
            case_summaries=summaries, groups=groups, conclusion=conclusion,
            args_dict=self._args_dict(), thresholds=play.DEFAULT_THRESHOLDS,
            schedule=play.make_schedule(settle_steps=200, tow_duration=5.0,
                                        coast_duration=5.0, dt=0.005),
            grades=[0.0], git={"commit": "deadbeef", "working_tree": None})
        self.assertNotIn("## 策略 vs 基线", without)


class JntExclusionRecountTests(unittest.TestCase):
    """**离线复算**（本变更最有力的验证，不依赖 Isaac Lab）。

    用两轮同版本、同几何、800 cell 的 `report.json`（`git.commit = df593f4…`、
    `--no-cart-fraction 0.125`）：

    1. 把 `cases[].verdict.reasons` 里的 `startup_joint_error` / `stop_joint_error` 剔除后
       重算通过数 / 逐坡度 / 逐连接 / 有负载-无负载 / 剩余失败原因，断言与
       `play.JNT_EXCLUDED_RECOUNT_2026_10_10`（README TOW-23 与 docs 记录里的数字）逐项一致；
    2. 反过来用 `classify_case(..., count_jnt=True)` 从 `cases[].metrics` 复算**旧口径**，
       断言与归档里存的 `verdict`（code + reasons）**逐位一致** —— 这就是
       「开 `--count-jnt` 时旧口径可复现」的证据；
    3. 用 `count_jnt=False` 复算新口径，断言等于剔除 JNT 后的 reasons。
    """

    JNT_REASONS = ("startup_joint_error", "stop_joint_error")

    @classmethod
    def setUpClass(cls):
        cls.reports = {}
        for tag in ("baseline", "policy"):
            path = (RL / "logs/towing/play_test" / f"upper_switch_{tag}" / "report.json")
            if not path.is_file():
                raise unittest.SkipTest(f"重判所用的 report.json 不在工作区：{path}")
            import json as _json
            cls.reports[tag] = _json.loads(path.read_text(encoding="utf-8"))

    @classmethod
    def _new_reasons(cls, case):
        return [reason for reason in case["verdict"]["reasons"]
                if reason not in cls.JNT_REASONS]

    def test_sources_are_the_documented_runs(self):
        expected = play.JNT_EXCLUDED_RECOUNT_2026_10_10
        for tag, payload in self.reports.items():
            self.assertEqual(payload["git"]["commit"], expected["git_commit"], tag)
            self.assertEqual(len(payload["cases"]), expected["num_envs"], tag)
            self.assertEqual(float(payload["arguments"]["no_cart_fraction"]),
                             expected["no_cart_fraction"], tag)
        # 策略轮确实开了上层开关、且契约是 v2 / iter 1000
        policy = self.reports["policy"]
        self.assertTrue(policy["upper_policy"]["enabled"])
        self.assertEqual(policy["upper_policy"]["iter"], 1000)
        self.assertEqual(policy["upper_policy"]["towing_contract"]["version"], 2)
        self.assertFalse(self.reports["baseline"]["upper_policy"]["enabled"])

    def test_recount_matches_the_documented_numbers(self):
        expected = play.JNT_EXCLUDED_RECOUNT_2026_10_10
        for tag, payload in self.reports.items():
            cases = payload["cases"]
            ok = sum(1 for case in cases if not self._new_reasons(case))
            self.assertEqual(ok, expected["ok"][tag], tag)
            self.assertEqual(len(cases) - ok, len(cases) - expected["ok"][tag], tag)
            # 旧口径：存下来的判定码里两轮都是 0 通过（JNT 800/800）
            self.assertEqual(expected["old_caliber_ok"][tag], 0)
            self.assertTrue(all("startup_joint_error" in case["verdict"]["reasons"]
                                and "stop_joint_error" in case["verdict"]["reasons"]
                                for case in cases), tag)
            # 逐坡度
            by_grade = {}
            for case in cases:
                key = play.grade_group_key(case["case"]["grade_deg"])
                total, passed = by_grade.get(key, (0, 0))
                by_grade[key] = (total + 1, passed + (0 if self._new_reasons(case) else 1))
            self.assertEqual(by_grade, expected["by_grade"][tag], tag)
            # 逐连接
            by_connection = {}
            for case in cases:
                key = case["case"]["connection"]
                total, passed = by_connection.get(key, (0, 0))
                by_connection[key] = (total + 1,
                                      passed + (0 if self._new_reasons(case) else 1))
            self.assertEqual(by_connection, expected["by_connection"][tag], tag)
            # 有负载 / 无负载
            by_cart = {}
            for case in cases:
                key = "cart" if case["case"].get("cart_present", True) else "nocart"
                total, passed = by_cart.get(key, (0, 0))
                by_cart[key] = (total + 1, passed + (0 if self._new_reasons(case) else 1))
            self.assertEqual(by_cart, expected["by_cart"][tag], tag)
            # 剩余失败原因（逐原因计数，一个 case 可有多条）
            reasons = {}
            for case in cases:
                for reason in self._new_reasons(case):
                    reasons[reason] = reasons.get(reason, 0) + 1
            self.assertEqual(reasons, expected["reasons"][tag], tag)
            # 剩余失败里不再有 JNT
            self.assertNotIn("startup_joint_error", reasons, tag)
            self.assertNotIn("stop_joint_error", reasons, tag)
            # 判定码分布里不再有 JNT（新口径）
            codes = {}
            for case in cases:
                code = max((play._REASON_TO_CODE[reason] for reason in self._new_reasons(case)),
                           key=lambda item: play.VERDICT_SEVERITY[item],
                           default="OK")
                codes[code] = codes.get(code, 0) + 1
            self.assertNotIn("JNT", codes, tag)

    def test_count_jnt_reproduces_the_archived_verdict_bit_for_bit(self):
        """`classify_case(count_jnt=True)` 对 800+800 个 case 逐位复现归档 verdict。"""
        for tag, payload in self.reports.items():
            thresholds = dict(payload["thresholds"])
            reproduced = 0
            for case in payload["cases"]:
                metrics = case["metrics"]
                stored = case["verdict"]
                legacy = play.classify_case(metrics, thresholds, count_jnt=True)
                self.assertEqual(legacy["reasons"], stored["reasons"], tag)
                self.assertEqual(legacy["code"], stored["code"], tag)
                self.assertEqual(legacy["severity"], stored["severity"], tag)
                # 新口径 = 旧口径剔除 JNT 两条（判定码取剩余最严重项）
                fresh = play.classify_case(metrics, thresholds, count_jnt=False)
                self.assertEqual(fresh["reasons"], self._new_reasons(case), tag)
                expected_code = max((play._REASON_TO_CODE[reason] for reason in fresh["reasons"]),
                                    key=lambda item: play.VERDICT_SEVERITY[item],
                                    default="OK")
                self.assertEqual(fresh["code"], expected_code, tag)
                reproduced += 1
            self.assertEqual(reproduced, play.JNT_EXCLUDED_RECOUNT_2026_10_10["num_envs"], tag)

    def test_open_question_channel_composition_is_from_the_baseline_first_witness(self):
        """待定问题（不在本轮实现）的数字出处：`impact.stop_contact.channels` 首个见证通道。

        用户给的 `load_velocity_jump 254 / deck_contact_force 70` 正好等于**基线**轮
        `cases[].metrics.impact.stop_contact.channels` 的计数；`deck |fx|` 中位 0 N。
        （若改按 `stop.contact_channels` 的**并集**统计会得到 264/71 —— 口径不同，不是矛盾。）
        """
        payload = self.reports["baseline"]
        channels = {"load_velocity_jump": 0, "deck_contact_force": 0}
        union = {"load_velocity_jump": 0, "deck_contact_force": 0}
        fx = []
        for case in payload["cases"]:
            if "stop_collision" not in case["verdict"]["reasons"]:
                continue
            contact = case["metrics"].get("impact", {}).get("stop_contact", {})
            for name in contact.get("channels") or []:
                channels[name] = channels.get(name, 0) + 1
            for name in case["metrics"]["stop"].get("contact_channels") or []:
                union[name] = union.get(name, 0) + 1
            if contact.get("deck_fx_n") is not None:
                fx.append(abs(float(contact["deck_fx_n"])))
        self.assertEqual(channels["load_velocity_jump"], 254)
        self.assertEqual(channels["deck_contact_force"], 70)
        self.assertEqual(union["load_velocity_jump"], 264)
        self.assertEqual(union["deck_contact_force"], 71)
        self.assertEqual(statistics.median(fx), 0.0)


class DryRunTests(unittest.TestCase):
    """`--dry-run` 只能在**标准库**下跑：子进程里跑一遍并检查 `torch` 从未进 `sys.modules`。

    这是「不启动 Isaac Sim 也能看 plan」这条契约的实测（不是静态检查）：子进程从
    `runpy` 走到 `main()` 返回，plan 行里必须体现新增的冲击窗口参数。
    """

    def test_dry_run_plan_lines_and_no_torch(self):
        import subprocess
        script = RL / "scripts/towing/play_towing_test.py"
        code = (
            "import runpy, sys\n"
            f"sys.argv = ['{script}', '--dry-run']\n"
            "try:\n"
            f"    runpy.run_path(r'{script}', run_name='__main__')\n"
            "except SystemExit as exc:\n"
            "    assert exc.code in (0, None), exc.code\n"
            "loaded = sorted(m for m in sys.modules if m == 'torch' or m.startswith('torch.'))\n"
            "assert not loaded, loaded\n"
            "print('DRY_RUN_OK')\n"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                timeout=300, cwd=str(RL.parent))
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertIn("DRY_RUN_OK", result.stdout)
        self.assertIn("--impact-window", result.stdout)
        self.assertIn("--steady-margin-s", result.stdout)
        self.assertIn("--takeup-force-threshold", result.stdout)
        self.assertIn("冲击窗口独立统计", result.stdout)
        # plan 里给出默认值
        self.assertIn("`--impact-window` 0.2 s", result.stdout)
        self.assertIn("`--steady-margin-s` 1 s", result.stdout)
        self.assertIn("`--takeup-force-threshold` 1 N", result.stdout)
        # ④ 默认判定口径必须打印（JNT 不计入），且 `--count-jnt` 开关可见
        self.assertIn("判定口径：**JNT 不计入**", result.stdout)
        self.assertIn("--count-jnt", result.stdout)


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

    def test_upper_switch_is_off_by_default_and_lazy(self):
        """`--upper-checkpoint` 开关的静态守卫（仿真相位跑不了，只钉住几条顺序约定）。

        1. 默认 `upper = None`，主循环只走基线路径（残差恒 0，与加开关前逐位一致）；
        2. 开关关时**不构造** runtime：构造点唯一，且落在
           `if args.upper_checkpoint is not None:` 块里（在 AppLauncher 之后的惰性块中）；
        3. 注入点唯一：`joint_targets = loco_joint_targets + upper_delta`（= held）只出现在
           `if upper is None: … else:` 的 **else（开关开）** 分支里，`held` 同时被下发、
           又被当 JNT 参考；
        4. 基线路径原样保留（`policy_step` 直接下发/返回冻结目标）；
        5. 模块顶部仍然没有 torch / isaaclab（`--dry-run` 可跑）。
        """
        self.assertIn("upper = None", self.source)
        tree = ast.parse(self.source)
        self.assertEqual(self.source.count("UpperPolicyRuntime("), 1)
        guarded = [node for node in ast.walk(tree) if isinstance(node, ast.If)
                   and "args.upper_checkpoint is not None" in ast.unparse(node.test)]
        self.assertTrue(guarded, "找不到 `if args.upper_checkpoint is not None:` 块")
        self.assertTrue(any("UpperPolicyRuntime(" in (ast.get_source_segment(self.source, node) or "")
                            for node in guarded),
                        "runtime 的构造点必须在开关分支里（关时不能构造）")
        # 开关状态进产物：experiment.json 的 upper_policy 段 + args 自动 dump
        self.assertIn('"upper_policy"', self.source)
        self.assertIn('"enabled": args.upper_checkpoint is not None', self.source)
        # 注入点：held = 冻结目标 + 残差，唯一，且只在「开关开」分支（`if upper is None: … else:`）
        held_assignments = [node for node in ast.walk(tree)
                            if isinstance(node, ast.Assign) and len(node.targets) == 1
                            and isinstance(node.targets[0], ast.Name)
                            and node.targets[0].id == "joint_targets"
                            and "loco_joint_targets + upper_delta" in ast.unparse(node.value)]
        self.assertEqual(len(held_assignments), 1)
        upper_blocks = [node for node in ast.walk(tree) if isinstance(node, ast.If)
                        and ast.unparse(node.test) == "upper is None"]
        self.assertTrue(upper_blocks, "找不到 `if upper is None:` 分支")
        for node in upper_blocks:
            baseline = "\n".join(ast.unparse(statement) for statement in node.body)
            # 关时基线路径：直接下发冻结目标，绝不碰残差
            self.assertIn("joint_targets = loco_joint_targets", baseline)
            self.assertNotIn("upper_delta", baseline)
            opened = "\n".join(ast.unparse(statement) for statement in node.orelse)
            # 开时：先推理（可选地刷新残差），再 held = 冻结目标 + 残差，并下发同一个张量
            self.assertIn("loco_joint_targets + upper_delta", opened)
            self.assertIn("robot.set_joint_position_target(joint_targets[:, asset_to_policy])",
                          opened)
            self.assertIn("upper.act(", opened)
        self.assertIn("step % upper_control_decimation == 0", self.source)
        # 基线路径的其余部分逐位保留
        self.assertIn("robot.set_joint_position_target(out.joint_targets[:, asset_to_policy])",
                      self.source)
        self.assertIn("return out.joint_targets", self.source)
        # 模块顶部不能 import torch / isaaclab
        for node in tree.body:
            if isinstance(node, ast.Import):
                self.assertNotIn("torch", {alias.name.split(".")[0] for alias in node.names})
            elif isinstance(node, ast.ImportFrom):
                self.assertNotIn((node.module or "").split(".")[0], {"torch", "isaaclab"})

    def test_upper_step_happens_at_the_training_rate(self):
        """上层每 `upper_control_decimation`（= upper_control_dt / dt）个物理步推理一次，
        两次之间残差保持不变（与 `HierarchicalVelocityAction` 同一结构）。"""
        self.assertIn('action_cfg.upper_control_dt', self.source)
        self.assertIn("step % upper_control_decimation == 0", self.source)
        # 冻结策略的帧项全部按训练口径取：策略关节顺序 + joint_pos 减默认角
        self.assertIn("robot.data.joint_pos[:, policy_to_asset]", self.source)
        self.assertIn("default_joint_pos_policy", self.source)
        self.assertIn("projected_gravity_body()", self.source)
        # held 只在冻结刷新那一拍重算（与训练侧 apply_actions 一致）
        self.assertIn("joint_targets = loco_joint_targets + upper_delta", self.source)
        self.assertIn("只在**冻结策略刷新**这一拍重算 held", self.source)

    def test_switch_state_is_recorded_for_traceability(self):
        """plan 行 / experiment.json / report.json 都要能查到开关状态与 checkpoint 契约。"""
        # experiment.json 的四要素（开关、路径、iter、完整契约）与 deterministic/stochastic
        for token in ('"enabled": args.upper_checkpoint is not None',
                      '"checkpoint": (str(args.upper_checkpoint)',
                      '"mode": "stochastic" if args.upper_stochastic else "deterministic"',
                      '"towing_contract": None',
                      '"iter": None'):
            self.assertIn(token, self.source)
        # 加载后打印成 plan 行（含 iter + 4 字段契约 + mode），并写回 experiment.json
        self.assertIn("[plan] 上层网络（checkpoint 已加载）：", self.source)
        self.assertIn("towing_contract={upper.contract}", self.source)
        self.assertIn("iter={upper.iteration}", self.source)
        self.assertIn("towing_contract=upper.contract, iter=upper.iteration", self.source)
        # report.json 带上层信息与对照节；report.md 用同一个汇总函数
        self.assertIn('"upper_policy": experiment["upper_policy"]', self.source)
        self.assertIn('"baseline_comparison": baseline_comparison', self.source)
        self.assertIn("comparison=comparison_lines", self.source)
        self.assertIn("case_metric_summary(case_summaries)", self.source)
        # 对照节里三项同口径：本轮 / 同版本基线 / 归档基线
        self.assertIn("reference=comparison_reference", self.source)
        self.assertIn("ARCHIVED_BASELINE_2026_10_09", self.source)


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
