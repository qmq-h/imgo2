#!/usr/bin/env python3
"""Owned headless Gazebo/PPO run and raw IMU recording, timed by simulation clock.

No model, policy or controller parameters are changed. --collect is the ROS child.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]


def profile(t, ramp, hold):
    if t < 0:
        return 0.0, "getup"
    if t < hold:
        return 0.0, "initial_stand"
    u = t - hold
    stages = [(0.0, 0.5, "forward_accel"), (0.5, 0.0, "forward_decel"),
              (0.0, -0.5, "backward_accel"), (-0.5, 0.0, "backward_decel")]
    i = int(u / ramp)
    if i >= 4:
        return 0.0, "final_stand"
    a, b, label = stages[i]
    return a + (b - a) * (u / ramp - i), label


def rpy(q):
    w, x, y, z = q.w, q.x, q.y, q.z
    return (math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
            math.asin(max(-1.0, min(1.0, 2*(w*y-z*x)))),
            math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)))


def stamp(m):
    return m.header.stamp.sec + m.header.stamp.nanosec * 1e-9


def collect(args):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Imu, Joy
    from nav_msgs.msg import Odometry
    from rosgraph_msgs.msg import Clock

    class Recorder(Node):
        def __init__(self):
            super().__init__("ppo_ramp_imu_recorder")
            self.clock = self.origin = None
            self.last_send = -math.inf
            self.imu_rows, self.odom_rows, self.command_rows = [], [], []
            self.fallen = False
            self.pub = self.create_publisher(Joy, "/joy", 10)
            self.create_subscription(Clock, "/clock", self.on_clock, qos_profile_sensor_data)
            self.create_subscription(Imu, "/imu", self.on_imu, qos_profile_sensor_data)
            self.create_subscription(Odometry, "/odom", self.on_odom, qos_profile_sensor_data)

        def on_clock(self, msg):
            self.clock = msg.clock.sec + msg.clock.nanosec * 1e-9
            if self.origin is None:
                self.origin = self.clock

        def relative(self, t):
            return t - self.origin - 6.0

        def on_imu(self, msg):
            if self.origin is None:
                return
            t = self.relative(stamp(msg))
            roll, pitch, yaw = rpy(msg.orientation)
            vx, stage = profile(t, args.ramp_seconds, args.hold_seconds)
            self.imu_rows.append([stamp(msg), t, stage, vx, roll, pitch, yaw,
                msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
                msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
                msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z])
            if t > 1 and max(abs(roll), abs(pitch)) > math.radians(60):
                self.fallen = True

        def on_odom(self, msg):
            if self.origin is None:
                return
            p = msg.pose.pose.position
            self.odom_rows.append([stamp(msg), self.relative(stamp(msg)), p.x, p.y, p.z,
                                   rpy(msg.pose.pose.orientation)[2]])

        def tick(self):
            if self.clock is None or self.clock - self.last_send < 0.019:
                return
            self.last_send = self.clock
            t = self.relative(self.clock)
            vx, stage = profile(t, args.ramp_seconds, args.hold_seconds)
            joy = Joy()
            joy.buttons = [0]*11
            joy.axes = [0.0]*8
            if t < 0:
                joy.buttons[0] = 1  # GetUp
            else:
                joy.buttons[5] = 1
                joy.axes[7] = 1.0  # PPO, key 1
                joy.axes[1] = vx
            self.pub.publish(joy)
            self.command_rows.append([self.clock, t, stage, vx, 0.0, 0.0])

    rclpy.init()
    node = Recorder()
    end_t = 4*args.ramp_seconds + 2*args.hold_seconds
    deadline = time.monotonic() + args.wall_timeout
    complete = False
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.005)
            node.tick()
            if node.fallen:
                raise RuntimeError("Tilt exceeded 60 degrees; recording stopped")
            if node.clock is not None and node.relative(node.clock) >= end_t:
                complete = True
                break
        if not complete:
            raise RuntimeError("Simulation clock did not finish the schedule before timeout")
    finally:
        tables = [("imu.csv", ["sim_s", "t_s", "stage", "scheduled_vx_mps", "roll_rad", "pitch_rad", "yaw_rad",
                              "gyro_x_rads", "gyro_y_rads", "gyro_z_rads", "accel_x_mps2", "accel_y_mps2", "accel_z_mps2",
                              "quat_w", "quat_x", "quat_y", "quat_z"], node.imu_rows),
                  ("odom.csv", ["sim_s", "t_s", "x_m", "y_m", "z_m", "yaw_rad"], node.odom_rows),
                  ("commands.csv", ["sim_s", "t_s", "stage", "vx_mps", "vy_mps", "wz_rads"], node.command_rows)]
        for name, header, rows in tables:
            with (args.output / name).open("w") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(rows)
        node.destroy_node()
        rclpy.shutdown()
    if len(node.imu_rows) < end_t * 50 or len(node.odom_rows) < end_t * 50:
        raise RuntimeError("Too few IMU/odom samples; see raw CSVs")
    print(json.dumps({"complete": complete, "imu_samples": len(node.imu_rows),
                      "odom_samples": len(node.odom_rows), "policy_seconds": end_t}), flush=True)


def run(args):
    base = os.environ.copy()
    for k in ("PYTHONHOME", "PYTHONPATH", "LD_LIBRARY_PATH"):
        base.pop(k, None)
    raw = subprocess.check_output(["bash", "--noprofile", "--norc", "-c",
        "source /opt/ros/humble/setup.bash\nsource imgo2_deploy/install/setup.bash\nenv -0"], cwd=REPO, env=base)
    env = dict(x.decode().split("=", 1) for x in raw.split(b"\0") if b"=" in x)
    env.update(ROS_DOMAIN_ID=str(args.domain), ROS_LOCALHOST_ONLY="1",
               GAZEBO_MASTER_URI=f"http://127.0.0.1:{args.master_port}", LIBGL_ALWAYS_SOFTWARE="1",
               ROS_LOG_DIR=str(args.output / "ros_logs"),
               IMGO2_GAZEBO_URDF=str(REPO / "imgo2_description/urdf/imgo2_real.gazebo.urdf"))
    env.pop("IMGO2_POLICY_TRACE", None)
    sources = ["imgo2_deploy/policy/imgo2/ppo/policy.pt", "imgo2_deploy/policy/imgo2/ppo/config.yaml",
               "imgo2_description/urdf/imgo2_real.gazebo.urdf", "imgo2_deploy/install/lib/imgo2_deploy/rl_sim"]
    meta = {"algorithm": "PPO", "gui": False, "domain": args.domain, "master_port": args.master_port,
            "ramp_seconds": args.ramp_seconds, "hold_seconds": args.hold_seconds,
            "ramp_acceleration_mps2": .5 / args.ramp_seconds,
            "sequence_mps": [0, .5, 0, -.5, 0], "lateral_and_yaw_commands": [0, 0],
            "files": {s: hashlib.sha256((REPO/s).read_bytes()).hexdigest() for s in sources}}
    (args.output / "run_metadata.json").write_text(json.dumps(meta, indent=2)+"\n")
    processes, handles = [], []
    try:
        def launch(name, cmd):
            handle = (args.output / f"{name}.log").open("w")
            handles.append(handle)
            proc = subprocess.Popen(cmd, cwd=REPO / "imgo2_deploy", env=env, stdout=handle,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(proc)
            return proc
        launch("gazebo", ["ros2", "launch", "imgo2_deploy", "gazebo.launch.py", "gui:=false"])
        time.sleep(5)
        launch("rl_sim", [str(REPO / sources[-1])])
        deadline = time.monotonic() + 70
        while "loop_keyboard" not in (args.output / "rl_sim.log").read_text(errors="replace"):
            if any(p.poll() is not None for p in processes):
                raise RuntimeError("Gazebo/control exited during startup; see logs")
            if time.monotonic() > deadline:
                raise RuntimeError("Controller startup timed out")
            time.sleep(.5)
        print("Controller ready; starting six-second getup then PPO speed ramps", flush=True)
        child = launch("recorder", ["/usr/bin/python3", str(Path(__file__).resolve()), "--collect",
            "--output", str(args.output), "--ramp-seconds", str(args.ramp_seconds),
            "--hold-seconds", str(args.hold_seconds), "--wall-timeout", str(args.wall_timeout)])
        while child.poll() is None:
            if any(p.poll() is not None for p in processes[:-1]):
                raise RuntimeError("Gazebo/control exited during collection")
            time.sleep(.5)
        if child.returncode:
            raise RuntimeError("Recording failed; see recorder.log")
        if "RL Controller [ppo]" not in (args.output / "rl_sim.log").read_text(errors="replace"):
            raise RuntimeError("PPO state entry was not confirmed by controller log")
        print("Speed schedule complete; stopping owned processes", flush=True)
    finally:
        for p in reversed(processes):
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGINT)
        for p in reversed(processes):
            try:
                p.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait(timeout=5)
        for h in handles:
            h.close()
        unchanged = all(hashlib.sha256((REPO/s).read_bytes()).hexdigest() == meta["files"][s] for s in sources)
        meta["source_files_unchanged_after_run"] = unchanged
        (args.output / "run_metadata.json").write_text(json.dumps(meta, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ramp-seconds", type=float, default=10)
    parser.add_argument("--hold-seconds", type=float, default=5)
    parser.add_argument("--wall-timeout", type=float, default=240)
    parser.add_argument("--domain", type=int, default=61)
    parser.add_argument("--master-port", type=int, default=11371)
    parser.add_argument("--collect", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.ramp_seconds <= 0 or args.hold_seconds < 0:
        parser.error("ramp must be positive and hold nonnegative")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    collect(args) if args.collect else run(args)


if __name__ == "__main__":
    main()
