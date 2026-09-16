# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
#
# ROS 2 / Gazebo 启动（Humble + Gazebo 11 + gazebo_ros2_control）。
# 参照 ~/RL/sim2sim/Imgo2_deploy/src/imgo2_deploy/launch/gazebo.launch.py，差异与理由：
#   1. 模型取 imgo2_description 里入库的生成物 urdf/imgo2.gazebo.urdf（已含 ros2_control
#      与 libgazebo_ros2_control.so 插件，见 imgo2_description/xacro/gazebo.xacro）；
#   2. 读进来后把 `package://imgo2_description` 换成本包的实际 share 路径，并把结果写到临时
#      URDF：gazebo_ros2_control 0.4.x 的 <parameters> 只认真实文件路径（给 package:// 会在
#      Load 里抛异常、插件不加载），换成绝对路径后网格与参数都不再依赖文件所在目录；
#   3. 因此 spawn 用 -file <临时 URDF>（比 -topic 少一层 QoS 依赖，实测更稳），
#      robot_state_publisher 用同一份替换后的文本；
#   4. 世界文件取本包 share 下的 worlds/<wname>.world（wname 可换 earth / stairs）；
#   5. gui:=false 可无头运行；实体名与 param_node 的 gazebo_model_name 一致（rl_sim 会读它）。
# robot_joint_controller 不在这里 spawn：rl_sim.cpp 启动时自己用 controller_manager
# spawner 起它，并把关节名单用临时 yaml 传进去。

import os
import tempfile
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, TextSubstitution, Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    rname = LaunchConfiguration("rname")
    wname = LaunchConfiguration("wname")
    gui = LaunchConfiguration("gui")

    robot_name = ParameterValue(Command(["echo -n ", rname]), value_type=str)
    gazebo_model_name = ParameterValue(Command(["echo -n ", rname, "_gazebo"]), value_type=str)

    description_share = get_package_share_directory("imgo2_description")
    urdf_path = os.path.join(description_share, "urdf", "imgo2.gazebo.urdf")
    with open(urdf_path, "r", encoding="utf-8") as f:
        text = f.read().replace("package://imgo2_description", description_share)
    # gazebo_ros2_control 0.4.x 把 URDF 以 `--param robot_description:=<xml>` 交给 controller_manager；
    # 带换行/XML 声明时 rcl 报 "Couldn't parse parameter override rule"（实测），CM 拿不到 URDF、
    # 控制器起不来。故压成单行并去掉声明。
    robot_description = " ".join(text.replace('<?xml version="1.0" ?>', '').split())
    # Gazebo 侧用写好的临时文件（绝对路径解析），RSP 用同一份文本
    resolved_urdf = os.path.join(tempfile.gettempdir(), "imgo2_gazebo_resolved.urdf")
    with open(resolved_urdf, "w", encoding="utf-8") as f:
        f.write(robot_description)

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description}],
    )

    world_path = PathJoinSubstitution([
        get_package_share_directory("imgo2_deploy"),
        "worlds",
        [wname, TextSubstitution(text=".world")],
    ])

    # 直接起 gzserver / gzclient（不经过 gazebo_ros 的 launch：本机实测它起的 gzserver
    # 没加载 libgazebo_ros_factory.so，/spawn_entity 始终不出现）。
    gzserver = ExecuteProcess(
        cmd=["gzserver", "-s", "libgazebo_ros_init.so", "-s", "libgazebo_ros_factory.so", world_path],
        output="both",
    )
    gzclient = ExecuteProcess(
        cmd=["gzclient"],
        output="both",
        condition=IfCondition(gui),
    )

    spawn_entity = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-file", resolved_urdf,
            "-entity", [rname, TextSubstitution(text="_gazebo")],
            "-x", "0.0",
            "-y", "0.0",
            "-z", "0.5",
        ],
        output="screen",
    )

    joint_state_broadcaster_node = Node(
        package="controller_manager",
        executable='spawner.py' if os.environ.get('ROS_DISTRO', '') == 'foxy' else 'spawner',
        arguments=["joint_state_broadcaster"],
        output="screen",
    )

    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        output='screen',
        parameters=[{
            'deadzone': 0.1,
            'autorepeat_rate': 0.0,
        }],
    )

    param_node = Node(
        package="demo_nodes_cpp",
        executable="parameter_blackboard",
        name="param_node",
        parameters=[{
            "robot_name": robot_name,
            "gazebo_model_name": gazebo_model_name,
        }],
    )

    # joint_state_broadcaster 必须等模型 spawn 完再起
    delayed_joint_state_broadcaster = RegisterEventHandler(
        OnProcessExit(target_action=spawn_entity, on_exit=[joint_state_broadcaster_node])
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "rname",
            description="Robot name",
            default_value=TextSubstitution(text="imgo2"),
        ),
        DeclareLaunchArgument(
            "wname",
            description="Gazebo world name (earth / stairs)",
            default_value=TextSubstitution(text="earth"),
        ),
        DeclareLaunchArgument(
            "gui",
            description="Start gzclient (set false for headless testing)",
            default_value=TextSubstitution(text="true"),
        ),
        robot_state_publisher_node,
        gzserver,
        gzclient,
        spawn_entity,
        delayed_joint_state_broadcaster,
        joy_node,
        param_node,
    ])
