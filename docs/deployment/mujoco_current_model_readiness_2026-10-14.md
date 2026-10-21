# MuJoCo 当前 AMP 对照的准备核查（2026-10-14）

## 结论

可以使用MuJoCo作为第三个引擎对照，但当前工程尚未满足同模型、同策略转换的比较前提。本轮只读源码/模型并运行标准库检查，未修改MJCF、训练或部署逻辑，未安装依赖、编译或运行MuJoCo。

## 已完成核查

1. 有独立C++入口 `rl_sim_mujoco.cpp`，CMake读取仓库根 `imgo2_description/mjcf/scene.xml`（include imgo2.xml），共享当前AMP策略配置。键0起身、键3选AMP，W/S/A/D/Q/E给速度、Space清零。
2. 当前MJCF仍是历史模型：四条腿的HIP、THIGH共8个body的质量、COM和显式惯量与最新 `imgo2_real.urdf` 不同。例如HIP为.66982842kg，最新URDF为.66kg；THIGH为.96995719kg，最新为1.044kg。
3. MJCF每个活动关节另有被动damping=1、armature=.1；这些不是共享策略配置里的电机PD25/.5。不同被动阻尼、惯量会改变实际响应，因此更平滑的画面不能单独证明Gazebo错误。
4. 当前MuJoCo `Forward()`仍先对raw action裁剪，再由共享 `ComputeOutput()` 做默认角+缩放，并反馈裁剪后的action；Gazebo AMP已改为默认角+缩放后裁剪目标、反馈raw action。该已知接口差异尚未同步到MuJoCo。
5. 本机 `imgo2_deploy/library/mujoco` 不存在，当前工作区未找到 `rl_sim_mujoco` 可执行文件。旧README的2026-09-17构建成功属于当时环境，不能替代当前依赖与模型核对。

[逐body核对结果](mujoco_current_model_readiness_2026-10-14.json) 可复现：

```bash
python3 imgo2_rl/scripts/tools/check_mjcf_current_robot.py \
  --json docs/deployment/mujoco_current_model_readiness_2026-10-14.json
```

当前返回1是“准备检查未通过”：8个body参数不同。检查仅比对当前对齐body frame中的显式惯量/COM/质量；不代替关节、碰撞转换或MuJoCo运行验证。

## 待修复／待确认（MODEL-MJCF-01 / DEPLOY-MJCF-01）

- 以最新core.xacro生成的real URDF为来源同步MJCF；逐项验证质量、COM、惯量、关节与碰撞、FK。需要有效转换与独立检查，不能复制旧模型或凭视觉确认。
- 为MuJoCo AMP同步已验证的目标裁剪/last_action契约，保留其它算法行为，验证同输入的actor、目标和反馈误差。
- 明确damping、armature和接触参数的基线；需要控制对照，不默认删除参数或将其当真实电机属性。
- 安装/定位MuJoCo开发库，构建当前入口，用同一策略、PD和固定0/+.4/−.4指令短测与现有Isaac/Gazebo结果比较。

**以下是准备完成后的入口说明，当前尚未验证可运行：**

```bash
cd /home/tommy0929/Documents/imgo2/imgo2_deploy
bash build.sh -mj
./cmake_build/bin/rl_sim_mujoco imgo2 scene
```

构建需要上述依赖，本轮没有执行下载。启动后的按0、按3和Space零指令仍需实测。来源代码见 [MuJoCo入口](../../imgo2_deploy/ros2_ws/src/imgo2_deploy/src/rl_sim_mujoco.cpp)、[共享动作转换](../../imgo2_deploy/ros2_ws/src/imgo2_deploy/library/core/rl_sdk/rl_sdk.cpp)。

本轮发现当前部署policy已换成cba59d44...，本轮没有对其运行验证；以下双引擎数值针对上轮9c6bf22b...，不作为新文件的测量结果。

已有双引擎结论见 [当前策略对照](amp_engine_comparison_2026-10-14.md)：静止自转在Isaac原条件约−.096rad/s、清洁摩擦对齐条件仍约−.092rad/s，Gazebo约−.057rad/s；命令均明确为0。因此“自转只发生在Gazebo”与本机测量不符，MuJoCo用于额外交叉验证，不作为先验正确答案。
