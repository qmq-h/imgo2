# IMU 终端调试（2026-10-10）

## 现象与根因

用户最初使用 ttyACM1 无输出，换为 ttyACM0 后获得 valid=15 的连续数据。
旧示例只打印有效帧，超时静默继续；逐帧滚屏也不便观察。

## 已修复

imu_reader 在交互终端清屏一次、隐藏光标，并在固定位置最多每秒刷新十次。
以设备同帧 Euler 字段显示 Roll/Pitch/Yaw（度），不显示四元数；角速度仍为 rad/s，加速度为 m/s²。
没有收到有效帧时显示等待；有效帧超过 500 ms 未更新显示 STALE，数值变为 --。
缺失字段显示 --，不把默认值当测量值。退出及异常时恢复光标。
非终端输出不包含清屏控制码。串口读取与解析协议未改。

## 验证

使用 Linux /usr/bin/c++（GCC 11.4）、CMake，构建目录 /tmp/imgo2_imu_debug_build。
编译通过，既有解析器与串口伪终端测试 2/2 通过。
使用 Python3 伪终端运行示例，确认等待提示、RPY 标签、固定刷新控制码和 SIGINT 后光标恢复，退出码 0。
测试辅助代码有 GCC stringop-overread 警告，未在本轮处理。

## 待确认与所缺条件

真实设备终端显示仍需用户在 IMU 所在电脑重新编译运行确认；本会话未连接 IMU。
设备安装到机器人后的坐标旋转、静止零偏、更新频率仍需真实设备实验。
用户数据与四元数方向基本一致，不等于已经完成机身坐标对齐。

仓库根目录运行：

```bash
cmake -S imgo2_deploy/imu_sensor -B imgo2_deploy/imu_sensor/build -DCMAKE_BUILD_TYPE=Release
cmake --build imgo2_deploy/imu_sensor/build -j
./imgo2_deploy/imu_sensor/build/imu_reader /dev/ttyACM0 460800
```
