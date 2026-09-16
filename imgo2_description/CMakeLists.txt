cmake_minimum_required(VERSION 3.10)
project(imgo2_description VERSION 2.0.0)

# 模型唯一源（见 README MODEL-02）：本包只装资源，不编译代码。
#   xacro/    模型源（core + 可选 transmission/gazebo/imu 模块）
#   urdf/     生成物：imgo2.urdf（纯）、imgo2.gazebo.urdf（+Gazebo/ROS 内容）
#   mjcf/     MuJoCo 模型（deploy 的 rl_sim_mujoco 经 IMGO2_MODEL_DIR 读取仓库路径）
#   config/   Gazebo / ros2_control 控制器配置
#   launch/   Gazebo 启动辅助 launch
#   meshes/   全仓唯一一套网格
if($ENV{ROS_DISTRO} MATCHES "noetic")
    find_package(catkin REQUIRED)
    catkin_package()
    foreach(dir config launch meshes urdf xacro mjcf)
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${dir})
            install(
                DIRECTORY ${dir}/
                DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/${dir}
            )
        endif()
    endforeach()
else()
    find_package(ament_cmake REQUIRED)
    find_package(urdf REQUIRED)
    find_package(xacro REQUIRED)
    find_package(robot_state_publisher REQUIRED)
    foreach(dir config launch meshes urdf xacro mjcf)
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${dir})
            install(
                DIRECTORY ${dir}/
                DESTINATION share/${PROJECT_NAME}/${dir}
            )
        endif()
    endforeach()
    ament_package()
endif()
