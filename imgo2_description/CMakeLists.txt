cmake_minimum_required(VERSION 3.10)
project(imgo2_description)

## Find catkin macros and libraries
find_package(catkin REQUIRED
        COMPONENTS
        )

###################################
## catkin specific configuration ##
###################################
catkin_package(
        CATKIN_DEPENDS
)

#############
## Install ##
#############

# Mark resource files for installation
# xacro/ 是模型源（core + 可选 gazebo/imu/transmission 模块），urdf/ 与 mjcf/ 是生成物，
# 都需要装进 share，否则 launch 与不吃 xacro 的消费者（RL/MuJoCo）拿不到模型。
install(
        DIRECTORY meshes urdf mjcf xacro launch
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
