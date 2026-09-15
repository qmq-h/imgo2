# imgo2_deploy

Deployment workspace for imgo2, derived from `rl_sar-main` and trimmed to a single robot.

## Layout

- `src/imgo2_deploy`: RL deployment package for ROS/CMake, including sim2sim, MuJoCo sim, and imgo2 real-robot entry points.
- `src/robot_msgs`: Shared motor/robot state messages.
- `src/robot_joint_controller`: Gazebo joint controller used by ROS simulation.
- `policy/imgo2`: imgo2 policy configuration and the copied Go2 HIM-Loco policy placeholder.
- `robot_description/imgo2_urdf`: URDF and meshes copied from `C:\Users\qmq\Desktop\Imgo2\Imgo2\Imgo2_urdf`.
- `robot_description/imgo2_mjcf`: Put generated MuJoCo XML files here, for example `robot_description/imgo2_mjcf/scene.xml`.

## Build

Install or download the inference runtime first:

```bash
bash build.sh --cmake
```

For MuJoCo sim2sim:

```bash
bash build.sh --mujoco
./cmake_build/bin/rl_sim_mujoco imgo2 scene
```

For ROS simulation, source ROS and run:

```bash
bash build.sh
ros2 launch imgo2_deploy gazebo.launch.py
```

For real robot deployment, populate `src/imgo2_deploy/library/thirdparty/robot_sdk/unitree/unitree_sdk2` first. The build will skip `rl_real_imgo2` while that SDK directory is empty.

## Notes

- `policy/imgo2/base.yaml` uses the URDF joint names (`*_shank_joint`) instead of Go2's `*_calf_joint`.
- The current policy files are adapted from the Go2 reference. Replace them with imgo2-trained policy/config files when available.
