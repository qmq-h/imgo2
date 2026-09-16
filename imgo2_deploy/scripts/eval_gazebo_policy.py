#!/usr/bin/env python3
"""Gazebo 侧策略评估：位姿 / 速度跟随 / 腿部抖动 / 周期性。

用法（需先起 Gazebo，见 README 6.2）：
    ros2 launch imgo2_deploy gazebo.launch.py gui:=false     # 另一个终端
    python3 imgo2_deploy/scripts/eval_gazebo_policy.py --vx 0.5 --duration 22

脚本会自己发布 /joy（A=起身 → RB+DPadUp=键1 PPO，axes[1]=vx），随后从 /odom（p3d 真值）与
/joint_states 统计四项指标；只依赖 rclpy + 标准库 math。

指标定义：
  位姿      z_mean/z_min/z_max（站立或行走高度）、roll/pitch 范围、yaw 漂移率、dx/dt
  速度跟随  mean(dx/dt) 对命令 vx 的误差
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
    def __init__(self, vx):
        super().__init__("eval_gazebo_policy")
        q = QoSProfile(depth=2000, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.vx = vx
        self.odom = []      # (t, x, z, yaw, roll, pitch)
        self.js = []        # (t, {name: (q, dq)})
        self.create_subscription(Odometry, "/odom", self.on_odom, q)
        self.create_subscription(JointState, "/joint_states", self.on_js, q)
        self.joy_pub = self.create_publisher(Joy, "/joy", 10)
        self.state = "passive"
        self.t0 = time.time()

    def on_odom(self, m):
        p, o = m.pose.pose.position, m.pose.pose.orientation
        self.odom.append((time.time(), p.x, p.z, yaw_of(o), roll_of(o), pitch_of(o)))

    def on_js(self, m):
        self.js.append((time.time(), dict(zip(m.name, zip(m.position, m.velocity)))))

    def send_joy(self, buttons, axes):
        j = Joy()
        j.buttons = list(buttons) + [0] * (11 - len(buttons))
        j.axes = list(axes) + [0.0] * (8 - len(axes))
        self.joy_pub.publish(j)

    def tick(self):
        """按时间推进按键序列：0~6s 起身(A)，6s 起进 PPO 并持续给 vx。"""
        el = time.time() - self.t0
        if el < 6.0:
            self.state = "getup"
            self.send_joy([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0.0] * 8)
        else:
            if self.state != "ppo":
                self.state = "ppo"
                self.entry_time = time.time()
            self.send_joy([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0.0, self.vx, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

def autocorr_peak(sig, dt):
    """返回 (周期秒, 强度0~1, 步频Hz)；sig 需已去均值。"""
    n = len(sig)
    if n < 20:
        return (float("nan"), float("nan"), float("nan"))
    denom = sum(v * v for v in sig)
    if denom <= 1e-12:
        return (float("nan"), float("nan"), float("nan"))
    best_lag, best_val = None, -2.0
    min_lag = max(2, int(0.12 / dt))      # 最快 ~8 Hz
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
        best_lag, best_val = max(vals, key=lambda x: x[1])
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
    ap.add_argument("--duration", type=float, default=22.0, help="含 6s 起身的总时长")
    ap.add_argument("--skip", type=float, default=3.0, help="进入策略后跳过的稳定时间")
    ap.add_argument("--json", type=str, default=None)
    a = ap.parse_args()

    rclpy.init()
    n = Eval(a.vx)
    t_end = time.time() + a.duration
    while time.time() < t_end:
        n.tick()
        rclpy.spin_once(n, timeout_sec=0.01)

    # 取稳定窗口
    t_start = getattr(n, "entry_time", n.t0) + a.skip
    od = [e for e in n.odom if e[0] >= t_start]
    js = [e for e in n.js if e[0] >= t_start]
    out = {"vx_cmd": a.vx, "samples": {"odom": len(od), "joint_states": len(js)}}
    if len(od) < 20 or len(js) < 50:
        print("样本不足：确认 Gazebo 在跑、策略已进入（键1）且 /odom、/joint_states 有数据")
        print(json.dumps(out, ensure_ascii=False)); rclpy.shutdown(); sys.exit(1)

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
    out["tracking"] = {
        "speed_mean": dx / dt_pose,
        "error_abs": abs(dx / dt_pose - a.vx),
        "error_pct": (abs(dx / dt_pose - a.vx) / a.vx * 100.0) if a.vx > 1e-6 else 0.0,
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
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
