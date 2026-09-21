#!/usr/bin/env python3
"""检查同机 ROS 2 的"发现"和"数据"两条通路是否都通。

背景：rl_sim 启动时会等 /param_node（由 gazebo.launch.py 起）并调用它取 robot_name /
gazebo_model_name。**服务能被发现 ≠ 调用能成功**——同机 DDS 的发现走多播，数据走单播/共享内存，
共享内存端口异常（例如 /dev/shm 里的 fastrtps_port* 残留、或两个终端不在同一个 /dev/shm 视图里）
时就会出现"服务列出来了、请求回不来"，rl_sim 于是报 param_node 相关错误。
`ros2 node list` / `ros2 param get` 走的是 daemon，daemon 自己的缓存可能过期，看到空结果不代表
真的没有节点，所以这里直接用 rclpy 建节点、发真实请求。

用法（ROS 2 环境里，与运行 rl_sim 的终端保持一致）：

    python3 imgo2_deploy/scripts/check_ros2_dds.py                 # 默认查 /param_node
    python3 imgo2_deploy/scripts/check_ros2_dds.py /controller_manager

退出码：0 = 发现与数据都通；1 = 有一项不通（脚本会把是哪一项打印出来）。
可选环境变量：DDS_CHECK_TIMEOUT（发现等待秒数，默认 10）。
"""

import os
import sys
import time

import rclpy
from rcl_interfaces.srv import GetParameters


def log(message: str) -> None:
    # flush 是必须的：输出被重定向或经过 GUI 终端时，行缓冲会让人误以为脚本卡住了。
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> int:
    target = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "/param_node"
    discover_timeout = float(os.environ.get("DDS_CHECK_TIMEOUT", "10"))

    log(f"target            : {target}")
    log(f"ROS_DOMAIN_ID     : {os.environ.get('ROS_DOMAIN_ID', '<unset>')}")
    log(f"RMW_IMPLEMENTATION: {os.environ.get('RMW_IMPLEMENTATION', '<unset>')}")
    log(f"ROS_LOCALHOST_ONLY: {os.environ.get('ROS_LOCALHOST_ONLY', '<unset>')}")
    log(f"FASTRTPS_DEFAULT_PROFILES_FILE: {os.environ.get('FASTRTPS_DEFAULT_PROFILES_FILE', '<unset>')}")

    # 下面两步（rclpy.init / create_node）如果卡住，说明 DDS 参与者都建不起来——rl_sim 报
    # "只有 Registered type 一行" 时就是同一个位置。分开计时便于判断。
    started = time.time()
    rclpy.init()
    log(f"rclpy.init()      : ok（{time.time() - started:.2f}s）")

    started = time.time()
    node = rclpy.create_node("dds_check_node")
    log(f"create_node()     : ok（{time.time() - started:.2f}s）")

    deadline = time.time() + discover_timeout
    nodes, services = [], []
    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
        nodes = sorted(n for n in node.get_node_names() if "dds_check" not in n)
        services = sorted(s for s, _ in node.get_service_names_and_types() if s.startswith(target + "/"))
        if services:
            break

    log(f"discovery nodes   : {nodes if nodes else '（一个都没发现）'}")
    log(f"discovery services: {services if services else '（一个都没发现）'}")

    if not services:
        log("RESULT: discovery FAILED —— 对方进程可能不在、或两条通路的配置不一致")
        node.destroy_node()
        rclpy.shutdown()
        return 1

    client = node.create_client(GetParameters, f"{target}/get_parameters")
    if not client.wait_for_service(timeout_sec=5.0):
        log("RESULT: data FAILED —— 服务在列表里但 wait_for_service 也过不去")
        node.destroy_node()
        rclpy.shutdown()
        return 1

    request = GetParameters.Request()
    request.names = ["robot_name", "gazebo_model_name"]
    started = time.time()
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=8.0)
    elapsed = time.time() - started

    result = None
    if future.done():
        try:
            result = future.result()
        except Exception as exc:  # noqa: BLE001 - 只想把原因打出来
            log(f"call exception    : {exc}")

    if result is None:
        log(f"data              : FAILED（{elapsed:.2f}s 内没回来）")
        log("RESULT: data FAILED —— 发现正常但请求回不来，检查 ROS_LOCALHOST_ONLY 是否两端一致、")
        log("        清理 /dev/shm/fastrtps_* 残留与遗留进程，或用 fastdds_udp_only.xml 强制走 UDP")
        node.destroy_node()
        rclpy.shutdown()
        return 1

    log(f"data              : OK（{elapsed:.2f}s） {[v.string_value for v in result.values]}")
    log("RESULT: OK —— 发现与数据都通，rl_sim 不应该卡在这一步")
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
