#!/usr/bin/env python3
"""Gazebo 侧策略评估：位姿 / 速度跟随 / 腿部抖动 / 周期性。

用法（需先起 Gazebo，见 README 6.2）：
    ros2 launch imgo2_deploy gazebo.launch.py gui:=false     # 另一个终端
    python3 imgo2_deploy/scripts/eval_gazebo_policy.py --vx 0.5 --duration 22            # 键1 PPO
    python3 imgo2_deploy/scripts/eval_gazebo_policy.py --key 3 --vx 0.5 --duration 22    # 键3 AMP
    # 需要可靠步频时加 --sim-time：仿真时间戳、100 Hz 均匀重采样

脚本会自己发布 /joy（A=起身 → RB+方向键选策略，axes[1]=vx；1=DPadUp / 2=DPadRight / 3=DPadDown），
随后从 /odom（p3d 真值）与 /joint_states 统计四项指标；只依赖 rclpy + 标准库 math。

**每条速度指令请用一条全新的 Gazebo + rl_sim 栈**。在同一条已进入策略的栈上连跑多个 vx 时，
开头那 6s 的 A 会重新触发 GetUp，而测量窗口仍从本次按键算起，均值会被起身段稀释
（实测 0.5 命令下 0.380 m/s 被压到 0.039 m/s）。确实要复用同一条栈时加 --no-getup。

指标定义：
  位姿      z_mean/z_min/z_max（站立或行走高度）、roll/pitch 范围、yaw 漂移率、dx/dt
  速度跟随  身体坐标前进速度对命令 vx 的误差（另保留世界 x 方向速度）
  抖动      相邻控制周期的 |Δdq|/dt 均值（角加速度量级）与 dq 换向次数/秒；分 hip/thigh/shank
  周期性    大腿角自相关首个峰 → 步态周期与周期强度（0~1）、步频；FL↔FR 相位差（trot≈180°）
"""
import argparse, json, math, sys, time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState, Joy
from nav_msgs.msg import Odometry

LEGS = ["FL", "FR", "RL", "RR"]
NAMES = [f"{l}_{j}_joint" for l in LEGS for j in ("hip", "thigh", "shank")]

def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

def roll_of(q):
    return math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y))

def pitch_of(q):
    return math.asin(max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x))))

class Eval(Node):
    def __init__(self, vx, key=1, getup=True, sim_time=False, vy=0.0, wz=0.0, diagnostics=False):
        super().__init__("eval_gazebo_policy")
        q = QoSProfile(depth=2000, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.vx = vx
        self.vy = vy
        self.wz = wz
        self.commands = []
        self.motor_feedback = []
        self._last_motor_wall = 0.0
        # 策略按键与手柄组合：1 = PPO (RB+DPadUp)，2 = himloco (RB+DPadRight)，3 = amp (RB+DPadDown)
        # 见 imgo2_deploy/src/imgo2_deploy/fsm_robot/fsm_imgo2.hpp
        self.key = key
        self.getup = getup
        self.sim_time = sim_time
        self.odom_wall = []
        self.js_wall = []
        self.odom = []      # (t, x, z, yaw, roll, pitch)
        self.js = []        # (t, {name: (q, dq)})
        self.create_subscription(Odometry, "/odom", self.on_odom, q)
        self.create_subscription(JointState, "/joint_states", self.on_js, q)
        if diagnostics:
            from robot_msgs.msg import RobotCommand, RobotState
            self.create_subscription(RobotCommand, "/robot_joint_controller/command", self.on_command, q)
            self.create_subscription(RobotState, "/robot_joint_controller/state", self.on_motor, q)
        self.joy_pub = self.create_publisher(Joy, "/joy", 10)
        self.state = "passive"
        self.t0 = time.time()

    def on_odom(self, m):
        p, o = m.pose.pose.position, m.pose.pose.orientation
        wall = time.time()
        stamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        self.odom_wall.append(wall)
        self.odom.append((stamp if self.sim_time else wall, p.x, p.z, yaw_of(o), roll_of(o), pitch_of(o), p.y))

    def on_js(self, m):
        wall = time.time()
        stamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        self.js_wall.append(wall)
        self.js.append((stamp if self.sim_time else wall, dict(zip(m.name, zip(m.position, m.velocity)))))

    def send_joy(self, buttons, axes):
        j = Joy()
        j.buttons = list(buttons) + [0] * (11 - len(buttons))
        j.axes = list(axes) + [0.0] * (8 - len(axes))
        self.joy_pub.publish(j)

    def on_command(self, m):
        self.commands.append((time.time(), [(x.q, x.dq, x.kp, x.kd, x.tau) for x in m.motor_command]))

    def on_motor(self, m):
        wall = time.time()
        if wall - self._last_motor_wall >= 0.01:
            self._last_motor_wall = wall
            self.motor_feedback.append((wall, [(x.q, x.dq, x.tau_est) for x in m.motor_state]))

    def tick(self):
        """按时间推进按键序列：0~6s 起身(A)，6s 起按 --key 进策略并持续给 vx。"""
        el = time.time() - self.t0
        if self.getup and el < 6.0:
            self.state = "getup"
            self.send_joy([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0.0] * 8)
        else:
            if self.state in ("getup", "passive"):
                self.state = "policy"
                self.entry_time = time.time()
            # RB=buttons[5]；DPadUp=axes[7]>0（键1）/ DPadRight=axes[6]>0（键2）/ DPadDown=axes[7]<0（键3）
            dpad = {1: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    2: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    3: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]}[self.key]
            axes = list(dpad)
            axes[1] = self.vx
            axes[0] = self.vy
            axes[3] = self.wz
            self.send_joy([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], axes)

def autocorr_peak(sig, dt):
    """返回 (周期秒, 强度0~1, 步频Hz)；sig 需已去均值。"""
    n = len(sig)
    if n < 20:
        return (float("nan"), float("nan"), float("nan"))
    denom = sum(v * v for v in sig)
    if denom <= 1e-12:
        return (float("nan"), float("nan"), float("nan"))
    best_lag, best_val = None, -2.0
    min_lag = max(2, math.ceil(0.12 / dt - 1e-9))  # 最快 ~8 Hz；避免浮点截断到 0.11 s
    max_lag = min(n // 2, int(1.5 / dt))  # 最慢 ~0.67 Hz
    vals = []
    for lag in range(min_lag, max_lag + 1):
        c = sum(sig[i] * sig[i + lag] for i in range(n - lag)) / denom
        vals.append((lag, c))
    # 取自相关第一个局部极大
    for i in range(1, len(vals) - 1):
        if vals[i][1] > vals[i - 1][1] and vals[i][1] >= vals[i + 1][1] and vals[i][1] > 0.15:
            best_lag, best_val = vals[i]
            break
    if best_lag is None:
        # A monotone boundary maximum is not evidence of a gait cycle.
        return (float("nan"), float("nan"), float("nan"))
    period = best_lag * dt
    return (period, max(-1.0, min(1.0, best_val)), 1.0 / period if period > 0 else float("nan"))

def phase_deg(sig_a, sig_b, dt):
    """sig_a 与 sig_b 的相位差（度）：先用自相关定周期，再互相关找峰。"""
    period, _, _ = autocorr_peak([v - sum(sig_a) / len(sig_a) for v in sig_a], dt)
    if not period or math.isnan(period):
        return float("nan"), float("nan")
    a = [v - sum(sig_a) / len(sig_a) for v in sig_a]
    b = [v - sum(sig_b) / len(sig_b) for v in sig_b]
    max_lag = int(period * 0.75 / dt)
    best_lag, best = 0, -2.0
    for lag in range(-max_lag, max_lag + 1):
        s = 0.0
        for i in range(len(a)):
            j = i + lag
            if 0 <= j < len(b):
                s += a[i] * b[j]
        if s > best:
            best, best_lag = s, lag
    return (360.0 * best_lag * dt / period), period

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vx", type=float, default=0.0)
    ap.add_argument("--vy", type=float, default=0.0, help="Body lateral velocity command (m/s).")
    ap.add_argument("--wz", type=float, default=0.0, help="Yaw velocity command (rad/s).")
    ap.add_argument("--key", type=int, default=1, choices=[1, 2, 3], help="1=PPO 2=himloco 3=amp")
    ap.add_argument("--no-getup", dest="getup", action="store_false",
                    help="跳过前 6s 的 A 起身。仅在已经处于策略状态时使用："
                         "若在一条已经进入 AMP 的栈里再次按 A，会重新触发 GetUp，"
                         "而测量窗口仍从本次按键算起 ⇒ 均值被起身段稀释（实测 0.5 命令下"
                         "0.380 m/s 被压到 0.039 m/s）。默认每条命令用一条全新的栈。")
    ap.add_argument("--duration", type=float, default=22.0, help="含 6s 起身的总时长")
    ap.add_argument("--skip", type=float, default=3.0, help="进入策略后跳过的稳定时间")
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--sim-time", action="store_true", help="Use ROS message simulation timestamps and resample joints at 100 Hz.")
    ap.add_argument("--raw-json", type=str, default=None, help="Save stable-window pose and joint samples for offline review.")
    ap.add_argument("--transient-json", type=str, default=None, help="Save the full startup/transition samples and motor commands/feedback.")
    a = ap.parse_args()

    rclpy.init()
    n = Eval(a.vx, a.key, a.getup, a.sim_time, a.vy, a.wz, bool(a.transient_json))
    t_end = time.time() + a.duration
    while time.time() < t_end:
        n.tick()
        rclpy.spin_once(n, timeout_sec=0.01)

    # 取稳定窗口
    t_start = getattr(n, "entry_time", n.t0) + a.skip
    if a.transient_json:
        with open(a.transient_json, "w") as f:
            json.dump({"time_base": "simulation" if a.sim_time else "wall",
                       "policy_entry_wall": getattr(n, "entry_time", None),
                       "odom": n.odom, "odom_wall": n.odom_wall,
                       "joint_states": n.js, "joint_wall": n.js_wall,
                       "motor_commands_wall": n.commands,
                       "motor_feedback_wall": n.motor_feedback}, f)
    od = [e for wall, e in zip(n.odom_wall, n.odom) if wall >= t_start]
    js = [e for wall, e in zip(n.js_wall, n.js) if wall >= t_start]
    out = {"vx_cmd": a.vx, "vy_cmd": a.vy, "wz_cmd": a.wz,
           "samples": {"odom": len(od), "joint_states": len(js)}}
    if len(od) < 20 or len(js) < 50:
        print("样本不足：确认 Gazebo 在跑、策略已进入（--key 指定）且 /odom、/joint_states 有数据")
        print(json.dumps(out, ensure_ascii=False)); rclpy.shutdown(); sys.exit(1)

    if a.sim_time:
        # Callback arrival intervals vary with machine load; use a uniform
        # simulation-time grid instead of their median for autocorrelation.
        unique = []
        for entry in js:
            if all(name in entry[1] for name in NAMES) and (not unique or entry[0] > unique[-1][0]):
                unique.append(entry)
        if len(unique) < 2:
            raise RuntimeError("Insufficient increasing simulation timestamps")
        sampled = []
        index = 0
        t = unique[0][0]
        while t <= unique[-1][0]:
            while index + 1 < len(unique) - 1 and unique[index + 1][0] < t:
                index += 1
            left, right = unique[index], unique[index + 1]
            alpha = (t - left[0]) / (right[0] - left[0])
            values = {name: tuple(left[1][name][k] + alpha * (right[1][name][k] - left[1][name][k])
                                 for k in (0, 1)) for name in NAMES}
            sampled.append((t, values))
            t += 0.01
        js = sampled
        out["time_base"] = "simulation"
        out["samples"]["resampled_joint_states"] = len(js)
    else:
        out["time_base"] = "wall"
    if a.raw_json:
        with open(a.raw_json, "w") as f:
            json.dump({"time_base": out["time_base"], "odom": od, "joint_states": js}, f)

    # ---- 位姿 ----
    zs = [e[2] for e in od]
    dt_pose = max(1e-6, od[-1][0] - od[0][0])
    dx = od[-1][1] - od[0][1]
    yaw_un = [math.degrees(e[3]) for e in od]
    for i in range(1, len(yaw_un)):
        while yaw_un[i] - yaw_un[i - 1] > 180: yaw_un[i] -= 360
        while yaw_un[i] - yaw_un[i - 1] < -180: yaw_un[i] += 360
    out["pose"] = {
        "z_mean": sum(zs) / len(zs), "z_min": min(zs), "z_max": max(zs),
        "roll_range_deg": math.degrees(max(e[4] for e in od) - min(e[4] for e in od)),
        "pitch_range_deg": math.degrees(max(e[5] for e in od) - min(e[5] for e in od)),
        "yaw_drift_deg_per_s": (yaw_un[-1] - yaw_un[0]) / dt_pose,
        "dx_m": dx, "dx_per_s": dx / dt_pose,
    }
    # ---- 速度跟随 ----
    forward_distance = 0.0
    lateral_distance = 0.0
    for left, right in zip(od, od[1:]):
        # Commands are in body coordinates; world-x displacement undercounts
        # forward motion when the robot turns. Rotate planar displacement.
        dyaw = math.atan2(math.sin(right[3] - left[3]), math.cos(right[3] - left[3]))
        heading = left[3] + dyaw / 2
        dx_step, dy_step = right[1] - left[1], right[6] - left[6]
        forward_distance += math.cos(heading) * dx_step + math.sin(heading) * dy_step
        lateral_distance += -math.sin(heading) * dx_step + math.cos(heading) * dy_step
    forward_speed = forward_distance / dt_pose
    out["tracking"] = {
        "speed_mean": forward_speed,
        "lateral_speed_mean": lateral_distance / dt_pose,
        "world_x_speed_mean": dx / dt_pose,
        "coordinate_frame": "body",
        "error_abs": abs(forward_speed - a.vx),
        "error_pct": (abs(forward_speed - a.vx) / abs(a.vx) * 100.0) if abs(a.vx) > 1e-6 else 0.0,
        "lateral_error_abs": abs(lateral_distance / dt_pose - a.vy),
        "yaw_rate_mean": math.radians(out["pose"]["yaw_drift_deg_per_s"]),
        "yaw_error_abs": abs(math.radians(out["pose"]["yaw_drift_deg_per_s"]) - a.wz),
    }
    # ---- 抖动：dq 相邻变化（角加速度量级）与换向率 ----
    dt_js = []
    for i in range(1, len(js)):
        if js[i][0] > js[i - 1][0]: dt_js.append(js[i][0] - js[i - 1][0])
    dts = sorted(dt_js)[len(dt_js) // 2] if dt_js else 0.02
    groups = {"hip": [], "thigh": [], "shank": []}
    rever = {"hip": 0, "thigh": 0, "shank": 0}
    for j in ("hip", "thigh", "shank"):
        for leg in LEGS:
            name = f"{leg}_{j}_joint"
            seq = [d[name][1] for t, d in js if name in d]
            if len(seq) < 10: continue
            acc = [abs(seq[i] - seq[i - 1]) / dts for i in range(1, len(seq))]
            groups[j].extend(acc)
            sign = [1 if v > 1e-3 else (-1 if v < -1e-3 else 0) for v in seq]
            sign = [s for s in sign if s != 0]
            rever[j] += sum(1 for i in range(1, len(sign)) if sign[i] != sign[i - 1])
    dur = js[-1][0] - js[0][0]
    out["jitter"] = {
        j: {"mean_abs_ddq": (sum(v) / len(v)) if v else float("nan"),
            "reversals_per_s": rever[j] / (4 * dur) if dur > 0 else float("nan")}
        for j, v in groups.items()
    }
    # ---- 周期性 ----
    def series(group):
        return {f"{l}_{group}_joint": [d[f"{l}_{group}_joint"][0] for t, d in js if f"{l}_{group}_joint" in d] for l in LEGS}
    th = series("thigh")
    periods, scores, freqs = [], [], []
    for leg in LEGS:
        raw = th[f"{leg}_thigh_joint"]
        m = sum(raw) / max(1, len(raw))
        sig = [v - m for v in raw]
        p, sc, f = autocorr_peak(sig, dts)
        periods.append(p); scores.append(sc); freqs.append(f)
    def nanmean(x):
        v = [q for q in x if not math.isnan(q)]
        return sum(v) / len(v) if v else float("nan")
    ph, _ = phase_deg(th["FL_thigh_joint"], th["FR_thigh_joint"], dts)
    out["periodicity"] = {
        "period_s": nanmean(periods), "freq_hz": nanmean(freqs),
        "score": nanmean(scores), "FL_FR_phase_deg": ph,
        "per_leg": {leg: {"period_s": p, "freq_hz": f, "score": sc}
                    for leg, p, f, sc in zip(LEGS, periods, freqs, scores)},
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
