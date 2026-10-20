# Standalone Linux hardware toolkit. No ROS or Unitree SDK dependency.
cmake_minimum_required(VERSION 3.10)
project(go_imu LANGUAGES CXX)
include(CTest)

# ==================== 本地构建开关（只改这里，不用每次敲 -D） ====================
# GO_IMU_BUILD_RL=ON：编译真机 PPO 部署（rl_real_imgo2 / fsm_check 及 RL 测试），
#   需要 LibTorch；仓库自带 library/inference_runtime/libtorch，会自动找到。
#   改成 OFF：整个工程不再依赖 LibTorch/yaml-cpp，只编电机/串口/IMU 等工具。
# GO_IMU_BUILD_APPS：go_monitor / go_stand / go_pd_demo 等终端工具与电机程序。
# 这两个都是 CMake 缓存变量，已经 configure 过的 build 目录不会因为这里改了而自动
# 跟随，需要 -DGO_IMU_BUILD_RL=ON/OFF 覆盖一次，或用 cmake --fresh 清缓存重来。
set(GO_IMU_BUILD_RL ON CACHE BOOL "Build independent PPO deployment (requires LibTorch and yaml-cpp)")
set(GO_IMU_BUILD_APPS ON CACHE BOOL "Build terminal tools and motor applications")
# ==============================================================================

include_directories(serialport/inc ${CMAKE_CURRENT_SOURCE_DIR})
add_library(yesense_parser STATIC Imu/src/yesense_parser.cpp)
target_include_directories(yesense_parser PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Imu/inc)
target_compile_features(yesense_parser PUBLIC cxx_std_14)

find_package(Threads REQUIRED)
add_library(go_math STATIC function/my_myth.cpp)
target_include_directories(go_math PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(go_math PUBLIC cxx_std_14)

# ---- 可移植部分：不依赖串口 ----
# go_loop：通用机制，定周期线程；只依赖 std::thread。
add_library(go_loop STATIC Motor/src/go_loop.cpp)
target_include_directories(go_loop PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Motor/inc ${CMAKE_CURRENT_SOURCE_DIR}/config)
target_link_libraries(go_loop PUBLIC Threads::Threads)
target_compile_features(go_loop PUBLIC cxx_std_14)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(go_loop PRIVATE -Wall -Wextra -Wconversion)
endif()
# go_protocol：协议编解码、CRC、输出侧/转子侧单位换算。
add_library(go_protocol STATIC Motor/src/go_protocol.cpp)
target_include_directories(go_protocol PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Motor/inc ${CMAKE_CURRENT_SOURCE_DIR}/config)
target_compile_features(go_protocol PUBLIC cxx_std_14)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(go_protocol PRIVATE -Wall -Wextra -Wconversion)
endif()
# go_trajectory：轨迹曲线（只描述"进度"，与具体关节无关）。纯数学，不依赖串口。
add_library(go_trajectory STATIC Motor/src/go_trajectory.cpp)
target_include_directories(go_trajectory PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Motor/inc ${CMAKE_CURRENT_SOURCE_DIR}/config)
target_compile_features(go_trajectory PUBLIC cxx_std_14)
target_link_libraries(go_trajectory PUBLIC go_math)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(go_trajectory PRIVATE -Wall -Wextra -Wconversion)
endif()
# go_config：整机配置（接线、offset、限幅、节拍、看门狗）。纯数据，不依赖串口。
add_library(go_config STATIC config/real_config.cpp)
target_include_directories(go_config PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Motor/inc ${CMAKE_CURRENT_SOURCE_DIR}/config)
target_compile_features(go_config PUBLIC cxx_std_14)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(go_config PRIVATE -Wall -Wextra -Wconversion)
endif()
# go_safety：限幅层（位置/速度/力矩/增益 + 限速斜坡）。纯函数，只看 Targets。
add_library(go_safety STATIC Motor/src/go_safety.cpp)
target_link_libraries(go_safety PUBLIC go_config)
target_compile_features(go_safety PUBLIC cxx_std_14)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(go_safety PRIVATE -Wall -Wextra -Wconversion)
endif()

# 纯目标生成FSM，不依赖串口或终端。
add_library(go_stand_fsm STATIC fsm/states/stand_state.cpp)
target_compile_features(go_stand_fsm PUBLIC cxx_std_17)
target_include_directories(go_stand_fsm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(go_stand_fsm PUBLIC go_trajectory go_safety)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  # Common serial transport; motor control and diagnostics share MotorTransport.
  add_library(shared_serial STATIC serialport/src/serial_port.cpp)
  target_include_directories(shared_serial PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/serialport/inc)
  target_compile_features(shared_serial PUBLIC cxx_std_14)
  add_library(imu_sensor STATIC Imu/src/imu_sensor.cpp Imu/src/imu_monitor.cpp)
  target_link_libraries(imu_sensor PUBLIC yesense_parser shared_serial go_config Threads::Threads)

  add_library(go_motor STATIC Motor/src/go_motor.cpp Motor/src/go_bus.cpp Motor/src/go_motor_monitor.cpp)
  target_link_libraries(go_motor PUBLIC go_protocol shared_serial)
  target_compile_options(go_motor PRIVATE -Wall -Wextra -Wconversion)
  # go_motor_array：把多条总线上的电机当成一组统一访问。
  add_library(go_motor_array STATIC Motor/src/go_motor_array.cpp)
  target_link_libraries(go_motor_array PUBLIC go_motor go_loop go_safety go_config Threads::Threads)
  target_compile_options(go_motor_array PRIVATE -Wall -Wextra -Wconversion)




  # Importable hardware library has no terminal/display dependency.
  add_library(real_hardware STATIC hardware/real_hardware.cpp)
  target_include_directories(real_hardware PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(real_hardware PUBLIC go_motor_array imu_sensor)
  add_library(go_imu::hardware ALIAS real_hardware)
  if(GO_IMU_BUILD_APPS)
    add_library(go_imu_display STATIC display/imu_panel.cpp display/motor_panel.cpp)
    target_include_directories(go_imu_display PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(go_imu_display PUBLIC go_motor imu_sensor)
    add_executable(go_monitor go_monitor.cpp)
    target_link_libraries(go_monitor PRIVATE go_imu_display)
    add_executable(go_pd_demo pd_demo_main.cpp)
    target_link_libraries(go_pd_demo PRIVATE go_motor_array)
    add_executable(go_stand stand_main.cpp)
    target_link_libraries(go_stand PRIVATE go_motor_array go_stand_fsm)
  endif()
  if(BUILD_TESTING)
    add_executable(test_shared_serial tests/test_shared_serial.cpp)
    target_link_libraries(test_shared_serial PRIVATE shared_serial util)
    add_test(NAME shared_serial_status COMMAND test_shared_serial)
    set_tests_properties(shared_serial_status PROPERTIES TIMEOUT 10)
    add_executable(test_serial_pty tests/test_serial_pty.cpp)
    target_link_libraries(test_serial_pty PRIVATE go_motor Threads::Threads util)
    add_test(NAME serial_pty COMMAND test_serial_pty)
    set_tests_properties(serial_pty PROPERTIES TIMEOUT 20)
    find_package(PythonInterp 3 QUIET)
    if(PYTHONINTERP_FOUND AND GO_IMU_BUILD_APPS)
      add_test(NAME library_import COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_library_import.py ${CMAKE_CURRENT_SOURCE_DIR})
      set_tests_properties(library_import PROPERTIES TIMEOUT 90)

      add_executable(hardware_pty_client tests/hardware_pty_client.cpp)
      target_link_libraries(hardware_pty_client PRIVATE real_hardware)
      add_test(NAME hardware_interface COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_hardware_pty.py $<TARGET_FILE:hardware_pty_client>)
      set_tests_properties(hardware_interface PROPERTIES TIMEOUT 30)

      add_executable(go_monitor_pty go_monitor.cpp)
      target_compile_definitions(go_monitor_pty PRIVATE GO_IMU_MONITOR_PTY_TEST)
      target_link_libraries(go_monitor_pty PRIVATE go_imu_display go_motor go_config imu_sensor Threads::Threads)
      add_test(NAME combined_monitor COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_combined_monitor.py $<TARGET_FILE:go_monitor_pty>)
      set_tests_properties(combined_monitor PROPERTIES TIMEOUT 30)

      add_executable(go_pd_demo_pty pd_demo_main.cpp)
      target_compile_definitions(go_pd_demo_pty PRIVATE GO_DEMO_PTY_TEST)
      # 伪终端测试胶水 pty_devices.h 放在 tests/，只给测试目标加这个 include 路径
      target_include_directories(go_pd_demo_pty PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
      target_link_libraries(go_pd_demo_pty PRIVATE go_motor_array Threads::Threads)
      add_test(NAME demo_pty COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_demo.py $<TARGET_FILE:go_pd_demo_pty>)
      set_tests_properties(demo_pty PROPERTIES TIMEOUT 30)
      add_executable(go_stand_pty stand_main.cpp)
      target_compile_definitions(go_stand_pty PRIVATE GO_STAND_PTY_TEST)
      target_include_directories(go_stand_pty PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
      target_link_libraries(go_stand_pty PRIVATE go_motor_array go_trajectory go_stand_fsm Threads::Threads)
      add_test(NAME stand_pty COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_stand.py $<TARGET_FILE:go_stand_pty>)
      set_tests_properties(stand_pty PROPERTIES TIMEOUT 60)
      # go_monitor 的启动配置摘要必须与 config/real_config.cpp 一致（不需要硬件；
      # 若检测到真实串口存在则自动跳过，避免 ctest 让实机卸力）
      add_test(NAME monitor COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_monitor.py $<TARGET_FILE:go_monitor>)
      set_tests_properties(monitor PROPERTIES TIMEOUT 30)
    endif()
  endif()
else()
  message(STATUS "Linux required for serial driver/demo; building portable tests only")
endif()

if(BUILD_TESTING)
  # 可移植测试
  add_executable(test_loop tests/test_loop.cpp)
  target_link_libraries(test_loop PRIVATE go_loop)
  add_test(NAME loop COMMAND test_loop)
  set_tests_properties(loop PROPERTIES TIMEOUT 30)
  add_executable(test_protocol tests/test_protocol.cpp)
  target_link_libraries(test_protocol PRIVATE go_protocol)
  add_test(NAME protocol COMMAND test_protocol)
  add_executable(test_trajectory tests/test_trajectory.cpp)
  target_link_libraries(test_trajectory PRIVATE go_trajectory)
  add_test(NAME trajectory COMMAND test_trajectory)
  set_tests_properties(trajectory PROPERTIES TIMEOUT 30)
  add_executable(test_motor_fake tests/test_motor_fake.cpp Motor/src/go_motor.cpp Motor/src/go_bus.cpp)
  target_link_libraries(test_motor_fake PRIVATE go_protocol)
  add_test(NAME motor_fake COMMAND test_motor_fake)
  # 看门狗测试：同样在链接期替换 SerialPort，额外编入 Loop 与 GoMotorArray
  # 才能验证“上层失联 → 主动卸力 → 恢复后自动接管”这条链路。
  add_executable(test_watchdog tests/test_watchdog.cpp Motor/src/go_motor.cpp Motor/src/go_bus.cpp
                 Motor/src/go_loop.cpp Motor/src/go_motor_array.cpp Motor/src/go_safety.cpp config/real_config.cpp)
  target_link_libraries(test_watchdog PRIVATE go_protocol Threads::Threads)
  add_test(NAME watchdog COMMAND test_watchdog)
  set_tests_properties(watchdog PROPERTIES TIMEOUT 30)
  # 限幅测试：纯函数部分 + 端到端部分（同样替换 SerialPort，检查线上字节确实被夹过）。
  add_executable(test_safety tests/test_safety.cpp Motor/src/go_motor.cpp Motor/src/go_bus.cpp
                 Motor/src/go_loop.cpp Motor/src/go_motor_array.cpp Motor/src/go_safety.cpp config/real_config.cpp)
  target_link_libraries(test_safety PRIVATE go_protocol Threads::Threads)
  add_test(NAME safety COMMAND test_safety)
  set_tests_properties(safety PROPERTIES TIMEOUT 30)
endif()

if(BUILD_TESTING)
  add_executable(imu_parser_test tests/imu/parser_test.cpp)
  target_link_libraries(imu_parser_test PRIVATE yesense_parser)
  add_test(NAME imu_parser COMMAND imu_parser_test)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(imu_serial_test tests/imu/serial_test.cpp)
    target_link_libraries(imu_serial_test PRIVATE imu_sensor)
    add_test(NAME imu_serial COMMAND imu_serial_test)
  endif()
endif()

if(BUILD_TESTING)
  add_executable(test_stand_fsm tests/test_stand_fsm.cpp)
  target_link_libraries(test_stand_fsm PRIVATE go_stand_fsm)
  add_test(NAME stand_fsm COMMAND test_stand_fsm)
endif()

if(BUILD_TESTING AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_executable(test_math_config tests/test_math_config.cpp)
  target_link_libraries(test_math_config PRIVATE go_motor_array go_math)
  add_test(NAME math_config COMMAND test_math_config)
  add_executable(test_trajectory_safety tests/test_trajectory_safety.cpp)
  target_link_libraries(test_trajectory_safety PRIVATE go_safety)
  add_test(NAME trajectory_safety COMMAND test_trajectory_safety)
  if(PYTHONINTERP_FOUND AND GO_IMU_BUILD_APPS)
    add_test(NAME application_preflight COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_application_preflight.py $<TARGET_FILE:go_monitor> $<TARGET_FILE:go_stand_pty>)
    set_tests_properties(application_preflight PROPERTIES TIMEOUT 15)
  endif()
endif()

# Standalone PPO deployment; the switch is defined in the build-options block at the
# top of this file. With it off the hardware tools still build without Torch/YAML.
if(GO_IMU_BUILD_RL)
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "RL hardware deployment currently requires Linux")
  endif()
  # LibTorch ships with the repository under library/, so resolve it from the
  # project root. A relative or stale CMAKE_PREFIX_PATH such as
  # "../library/inference_runtime/libtorch" then cannot break discovery.
  if("${Torch_DIR}" MATCHES "-NOTFOUND$")
    unset(Torch_DIR CACHE)
  endif()
  set(GO_IMU_BUNDLED_LIBTORCH "${CMAKE_CURRENT_SOURCE_DIR}/library/inference_runtime/libtorch")
  if(EXISTS "${GO_IMU_BUNDLED_LIBTORCH}/share/cmake/Torch/TorchConfig.cmake")
    list(APPEND CMAKE_PREFIX_PATH "${GO_IMU_BUNDLED_LIBTORCH}")
  endif()
  find_package(Torch REQUIRED)
  find_package(yaml-cpp REQUIRED)
  if(TORCH_CXX_FLAGS MATCHES "_GLIBCXX_USE_CXX11_ABI=0")
    message(FATAL_ERROR "Use a C++11 ABI LibTorch build compatible with system yaml-cpp and the hardware libraries")
  endif()
  add_library(go_rl_core STATIC
    sdk/policy_config.cpp sdk/rl_sdk.cpp sdk/inference_worker.cpp
    config/deployment_config.cpp hardware/robot_adapter.cpp fsm/deployment_states.cpp
    fsm/states/hold_state.cpp fsm/states/passive_state.cpp
    fsm/states/ppo_state.cpp fsm/states/fault_state.cpp)
  target_link_libraries(go_rl_core PUBLIC real_hardware go_stand_fsm go_math yaml-cpp)
  target_compile_features(go_rl_core PUBLIC cxx_std_17)
  target_include_directories(go_rl_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
  add_library(go_policy_runner STATIC sdk/policy_runner.cpp)
  target_link_libraries(go_policy_runner PUBLIC go_rl_core ${TORCH_LIBRARIES})
  add_library(go_imu::sdk ALIAS go_policy_runner)
  add_executable(fsm_check fsm_check.cpp)
  target_link_libraries(fsm_check PRIVATE go_policy_runner)
  add_executable(rl_real_imgo2 rl_real_imgo2.cpp apps/log_writer.cpp)
  target_link_libraries(rl_real_imgo2 PRIVATE go_policy_runner)
  # Keep the external runtime discoverable after installation; sibling lib also permits packaging it.
  set_target_properties(rl_real_imgo2 fsm_check PROPERTIES INSTALL_RPATH "$ORIGIN/../lib" INSTALL_RPATH_USE_LINK_PATH TRUE)
  foreach(app rl_real_imgo2 fsm_check)
  add_custom_command(TARGET ${app} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/policy $<TARGET_FILE_DIR:${app}>/policy
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${app}>/config
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/config/deployment.yaml $<TARGET_FILE_DIR:${app}>/config/deployment.yaml)
  endforeach()
  install(TARGETS rl_real_imgo2 fsm_check RUNTIME DESTINATION bin)
  install(DIRECTORY policy DESTINATION bin)
  install(FILES config/deployment.yaml DESTINATION bin/config)
endif()

if(GO_IMU_BUILD_RL AND BUILD_TESTING)
  foreach(name policy_core robot_fsm inference_worker)
    add_executable(test_${name} tests/test_${name}.cpp)
    target_link_libraries(test_${name} PRIVATE go_rl_core)
  endforeach()
  add_test(NAME policy_core COMMAND test_policy_core ${CMAKE_CURRENT_SOURCE_DIR}/policy/ppo)
  add_test(NAME robot_fsm COMMAND test_robot_fsm ${CMAKE_CURRENT_SOURCE_DIR}/policy/ppo ${CMAKE_CURRENT_SOURCE_DIR}/config/deployment.yaml)
  add_test(NAME inference_worker COMMAND test_inference_worker)
  set_tests_properties(inference_worker PROPERTIES TIMEOUT 5)
  add_executable(policy_inference_check tests/policy_inference_check.cpp)
  target_link_libraries(policy_inference_check PRIVATE go_policy_runner)
  add_test(NAME policy_preflight COMMAND rl_real_imgo2 --check-policy)
  add_executable(rl_real_imgo2_pty rl_real_imgo2.cpp apps/log_writer.cpp)
  target_compile_definitions(rl_real_imgo2_pty PRIVATE GO_RL_PTY_TEST)
  target_include_directories(rl_real_imgo2_pty PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  target_link_libraries(rl_real_imgo2_pty PRIVATE go_policy_runner)
  set(GO_IMU_TORCH_PYTHON "" CACHE FILEPATH "Optional Python with torch for numerical reference test")
  if(GO_IMU_TORCH_PYTHON)
    add_test(NAME policy_reference COMMAND ${GO_IMU_TORCH_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_policy_reference.py $<TARGET_FILE:policy_inference_check> ${CMAKE_CURRENT_SOURCE_DIR}/policy/ppo)
    set_tests_properties(policy_reference PROPERTIES TIMEOUT 45)
  endif()
endif()

if(GO_IMU_BUILD_RL AND BUILD_TESTING AND PYTHONINTERP_FOUND)
  add_executable(terminal_input_client tests/terminal_input_client.cpp)
  target_link_libraries(terminal_input_client PRIVATE Threads::Threads)
  add_test(NAME terminal_input COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_terminal_input.py $<TARGET_FILE:terminal_input_client>)
  set_tests_properties(terminal_input PROPERTIES TIMEOUT 15)
endif()
if(GO_IMU_BUILD_RL AND BUILD_TESTING AND PYTHONINTERP_FOUND)
  add_test(NAME rl_deployment_pty COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_rl_deployment_pty.py $<TARGET_FILE:rl_real_imgo2_pty> ${CMAKE_CURRENT_SOURCE_DIR}/policy)
  set_tests_properties(rl_deployment_pty PROPERTIES TIMEOUT 35)
endif()
if(GO_IMU_BUILD_RL AND BUILD_TESTING AND PYTHONINTERP_FOUND)
  add_test(NAME rl_invalid_preflight COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_rl_preflight.py $<TARGET_FILE:rl_real_imgo2> ${CMAKE_CURRENT_SOURCE_DIR}/policy ${CMAKE_CURRENT_SOURCE_DIR}/config/deployment.yaml)
  add_test(NAME rl_relocation COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_rl_relocation.py ${CMAKE_CURRENT_SOURCE_DIR} ${Torch_DIR})
  set_tests_properties(rl_invalid_preflight PROPERTIES TIMEOUT 90)
  set_tests_properties(rl_relocation PROPERTIES TIMEOUT 240 RUN_SERIAL TRUE)
endif()
if(GO_IMU_BUILD_RL AND BUILD_TESTING AND PYTHONINTERP_FOUND)
  add_test(NAME fsm_check COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_fsm_check.py $<TARGET_FILE:fsm_check> ${CMAKE_CURRENT_SOURCE_DIR}/policy ${CMAKE_CURRENT_SOURCE_DIR}/config/deployment.yaml)
  set_tests_properties(fsm_check PROPERTIES TIMEOUT 55)
endif()
