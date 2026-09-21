/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rl_sim.hpp"

#if defined(USE_ROS2)
#include <rmw/rmw.h>
#endif

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <set>
#include <thread>

RL_Sim::RL_Sim(int argc, char **argv)
{
#if defined(USE_ROS1)
    this->ang_vel_axis = "world";
    ros::NodeHandle nh;
    nh.param<std::string>("ros_namespace", this->ros_namespace, "");
    nh.param<std::string>("robot_name", this->robot_name, "");
#elif defined(USE_ROS2)
    // 这两行是为了把「卡在 DDS 参与者初始化」和「卡在等 param_node」区分开：前者连下面那条
    // Waiting for /param_node 都不会打印。历史上出现过一个只有 Registered type 一行、什么都没
    // 有的卡死报告，就是前者（共享内存端口初始化 / 同名节点残留）。
    std::cout << LOGGER::INFO << "Creating ROS 2 node (rl_sim_node) ..." << std::endl;
    ros2_node = std::make_shared<rclcpp::Node>("rl_sim_node");
    std::cout << LOGGER::INFO << "ROS 2 node created; rmw=" << rmw_get_implementation_identifier() << std::endl;
    this->ang_vel_axis = "body";
    this->ros_namespace = ros2_node->get_namespace();
    // get params from param_node
    // param_node 由 gazebo.launch.py 起（demo_nodes_cpp 的 parameter_blackboard），是整条 ROS 链路里
    // 第一个跨进程依赖：等不到它只有两种可能——Gazebo 没在跑，或者两个终端的 DDS 发现不一致
    // （ROS_DOMAIN_ID / RMW_IMPLEMENTATION / ROS_LOCALHOST_ONLY / 共享内存残留）。这里把这两类线索
    // 一次打印出来，并在 kParamNodeTimeoutS 秒后给出可读报错，避免"看起来卡死、什么都看不到"。
    // 注意：base.yaml 要到下面的 ReadYaml 才读，所以这里的超时只能是常量。
    param_client = ros2_node->create_client<rcl_interfaces::srv::GetParameters>("/param_node/get_parameters");
    constexpr double kParamNodeTimeoutS = 60.0;
    bool param_warned = false;
    const auto param_deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(kParamNodeTimeoutS);
    while (!param_client->wait_for_service(std::chrono::seconds(1)))
    {
        if (!rclcpp::ok()) {
            throw std::runtime_error("Interrupted while waiting for /param_node");
        }
        if (!param_warned)
        {
            param_warned = true;
            const char* domain_id = std::getenv("ROS_DOMAIN_ID");
            const char* rmw = std::getenv("RMW_IMPLEMENTATION");
            const char* localhost_only = std::getenv("ROS_LOCALHOST_ONLY");
            std::cout << LOGGER::WARNING << "Waiting for /param_node (max " << kParamNodeTimeoutS
                      << " s). It is started by gazebo.launch.py; is that launch still running? Current DDS:"
                      << " ROS_DOMAIN_ID=" << (domain_id ? domain_id : "<unset>")
                      << " RMW_IMPLEMENTATION=" << (rmw ? rmw : "<unset>")
                      << " ROS_LOCALHOST_ONLY=" << (localhost_only ? localhost_only : "<unset>")
                      << std::endl;
        }
        if (std::chrono::steady_clock::now() >= param_deadline)
        {
            std::cout << LOGGER::ERROR << "/param_node did not show up within " << kParamNodeTimeoutS
                      << " s; giving up. Start Gazebo first (ros2 launch imgo2_deploy gazebo.launch.py),"
                      << " and check both terminals share ROS_DOMAIN_ID/RMW_IMPLEMENTATION and can see"
                      << " each other (ros2 node list)." << std::endl;
            throw std::runtime_error("param_node is not available");
        }
    }
    auto request = std::make_shared<rcl_interfaces::srv::GetParameters::Request>();
    request->names = {"robot_name", "gazebo_model_name"};
    // 服务"能被发现"不等于"调用能成功"：同机 DDS 的发现（多播）与数据（单播/SHM）是两条路径，
    // 共享内存端口异常时会出现"服务在、请求回不来"。实测这种时候旧代码只打印一行错误就往下走，
    // 于是 robot_name 为空 → 去读 policy//base.yaml → 报文件不存在 → FSM 未注册 → 空关节名单，
    // 真正的原因被一串次生错误埋掉。这里改成重试若干次，仍失败就带着 DDS 线索明确退出。
    constexpr int kParamRetries = 3;
    bool got_params = false;
    for (int attempt = 1; attempt <= kParamRetries && !got_params; ++attempt)
    {
        if (!rclcpp::ok())
        {
            throw std::runtime_error("Interrupted while waiting for /param_node");
        }
        auto future = param_client->async_send_request(request);
        auto status = rclcpp::spin_until_future_complete(ros2_node->get_node_base_interface(), future, std::chrono::seconds(5));
        if (status != rclcpp::FutureReturnCode::SUCCESS)
        {
            std::cout << LOGGER::WARNING << "/param_node service call timed out (attempt " << attempt << "/"
                      << kParamRetries << ")" << std::endl;
            continue;
        }
        auto result = future.get();
        if (result->values.size() < 2)
        {
            std::cout << LOGGER::ERROR << "Failed to get all parameters from param_node" << std::endl;
            continue;
        }
        this->robot_name = result->values[0].string_value;
        this->gazebo_model_name = result->values[1].string_value;
        std::cout << LOGGER::INFO << "Get param robot_name: " << this->robot_name << std::endl;
        std::cout << LOGGER::INFO << "Get param gazebo_model_name: " << this->gazebo_model_name << std::endl;
        got_params = true;
    }
    if (!got_params)
    {
        std::cout << LOGGER::ERROR << "Could not read robot_name/gazebo_model_name from /param_node: the service"
                  << " was discovered but the call did not return. Same-host DDS discovery (multicast) and data"
                  << " (unicast/shared memory) are separate paths - check ROS_LOCALHOST_ONLY on both terminals,"
                  << " stale /dev/shm/fastrtps_* leftovers, and leftover ROS/Gazebo processes." << std::endl;
        throw std::runtime_error("param_node call failed");
    }
#endif

    // read params from yaml
    this->ReadYaml(this->robot_name, "base.yaml");
    // ReadYaml 只打印一行错误就返回，后面会以空参数一路走下去（FSM 未注册、关节名单为空），
    // 把一个"文件没读到"放大成看起来完全无关的失败。这里直接判定。
    if (!this->params.Has("num_of_dofs"))
    {
        std::cout << LOGGER::ERROR << "base.yaml for robot '" << this->robot_name << "' was not loaded; expected "
                  << std::string(POLICY_DIR) + "/" + this->robot_name + "/base.yaml" << std::endl;
        throw std::runtime_error("base.yaml not loaded");
    }

    // auto load FSM by robot_name
    if (FSMManager::GetInstance().IsTypeSupported(this->robot_name))
    {
        auto fsm_ptr = FSMManager::GetInstance().CreateFSM(this->robot_name, this);
        if (fsm_ptr)
        {
            this->fsm = *fsm_ptr;
        }
    }
    else
    {
        std::cout << LOGGER::ERROR << "[FSM] No FSM registered for robot: " << this->robot_name << std::endl;
    }

    // init robot
#if defined(USE_ROS1)
    this->joint_publishers_commands.resize(this->params.Get<int>("num_of_dofs"));
#elif defined(USE_ROS2)
    this->robot_command_publisher_msg.motor_command.resize(this->params.Get<int>("num_of_dofs"));
    this->robot_state_subscriber_msg.motor_state.resize(this->params.Get<int>("num_of_dofs"));
#endif
    this->InitJointNum(this->params.Get<int>("num_of_dofs"));
    this->InitOutputs();
    this->InitControl();

#if defined(USE_ROS1)
    auto joint_controller_names_vec = this->params.Get<std::vector<std::string>>("joint_controller_names");  // avoid dangling reference
    this->StartJointController(this->ros_namespace, joint_controller_names_vec);
    // publisher
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        const std::string &joint_controller_name = joint_controller_names_vec[i];
        const std::string topic_name = this->ros_namespace + joint_controller_name + "/command";
        this->joint_publishers[joint_controller_name] =
            nh.advertise<robot_msgs::MotorCommand>(topic_name, 10);
    }

    // subscriber
    this->cmd_vel_subscriber = nh.subscribe<geometry_msgs::Twist>("/cmd_vel", 10, &RL_Sim::CmdvelCallback, this);
    this->joy_subscriber = nh.subscribe<sensor_msgs::Joy>("/joy", 10, &RL_Sim::JoyCallback, this);
    this->model_state_subscriber = nh.subscribe<gazebo_msgs::ModelStates>("/gazebo/model_states", 10, &RL_Sim::ModelStatesCallback, this);
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        const std::string &joint_controller_name = joint_controller_names_vec[i];
        const std::string topic_name = this->ros_namespace + joint_controller_name + "/state";
        this->joint_subscribers[joint_controller_name] =
            nh.subscribe<robot_msgs::MotorState>(topic_name, 10,
                [this, joint_controller_name](const robot_msgs::MotorState::ConstPtr &msg)
                {
                    this->JointStatesCallback(msg, joint_controller_name);
                }
            );
        this->joint_positions[joint_controller_name] = 0.0f;
        this->joint_velocities[joint_controller_name] = 0.0f;
        this->joint_efforts[joint_controller_name] = 0.0f;
    }

    // service
    nh.param<std::string>("gazebo_model_name", this->gazebo_model_name, "");
    this->gazebo_pause_physics_client = nh.serviceClient<std_srvs::Empty>("/gazebo/pause_physics");
    this->gazebo_unpause_physics_client = nh.serviceClient<std_srvs::Empty>("/gazebo/unpause_physics");
    this->gazebo_reset_world_client = nh.serviceClient<std_srvs::Empty>("/gazebo/reset_world");
#elif defined(USE_ROS2)
    // ROS2/Gazebo 路径：controller（robot_joint_controller/RobotJointControllerGroup）的消息槽位顺序
    // = 这里传给它的 `joints` 参数顺序（controller 按名字解析 ros2_control 接口，与 URDF 里的声明
    // 顺序无关，所以顺序由我们给）。策略的顺序是模型顺序（URDF/MJCF 都是 FL,FR,RL,RR，也是训练
    // 顺序），而 base.yaml 的 joint_names 是 Unitree SDK 的电机顺序（FR,FL,RR,RL）；直接用
    // joint_names 会让恒等 joint_mapping 把策略的 FL 接到物理 FR（左右腿互换，给速度指令后翻成
    // 四脚朝天，实测见 docs/gazebo_ros2_bringup_2026-09-17.md 第 8 节）。故按 URDF 顺序重排。
    // MuJoCo 路径不读 joint_names，其 joint_mapping 语义与顺序保持原样。
    this->StartJointController(
        this->ros_namespace,
        this->OrderJointsByModelOrder(this->params.Get<std::vector<std::string>>("joint_names")));
    // publisher
    this->robot_command_publisher = ros2_node->create_publisher<robot_msgs::msg::RobotCommand>(
        this->ros_namespace + "robot_joint_controller/command", rclcpp::SystemDefaultsQoS());

    // subscriber
    this->cmd_vel_subscriber = ros2_node->create_subscription<geometry_msgs::msg::Twist>(
        "/cmd_vel", rclcpp::SystemDefaultsQoS(),
        [this] (const geometry_msgs::msg::Twist::SharedPtr msg) {this->CmdvelCallback(msg);}
    );
    this->joy_subscriber = ros2_node->create_subscription<sensor_msgs::msg::Joy>(
        "/joy", rclcpp::SystemDefaultsQoS(),
        [this] (const sensor_msgs::msg::Joy::SharedPtr msg) {this->JoyCallback(msg);}
    );
    this->gazebo_imu_subscriber = ros2_node->create_subscription<sensor_msgs::msg::Imu>(
        "/imu", rclcpp::SystemDefaultsQoS(), [this] (const sensor_msgs::msg::Imu::SharedPtr msg) {this->GazeboImuCallback(msg);}
    );
    this->robot_state_subscriber = ros2_node->create_subscription<robot_msgs::msg::RobotState>(
        this->ros_namespace + "robot_joint_controller/state", rclcpp::SystemDefaultsQoS(),
        [this] (const robot_msgs::msg::RobotState::SharedPtr msg) {this->RobotStateCallback(msg);}
    );

    // service
    this->gazebo_pause_physics_client = ros2_node->create_client<std_srvs::srv::Empty>("/pause_physics");
    this->gazebo_unpause_physics_client = ros2_node->create_client<std_srvs::srv::Empty>("/unpause_physics");
    this->gazebo_reset_world_client = ros2_node->create_client<std_srvs::srv::Empty>("/reset_world");

    auto empty_request = std::make_shared<std_srvs::srv::Empty::Request>();
    auto result = this->gazebo_reset_world_client->async_send_request(empty_request);
#endif

    // loop
    this->loop_control = std::make_shared<LoopFunc>("loop_control", this->params.Get<float>("dt"), std::bind(&RL_Sim::RobotControl, this));
    this->loop_rl = std::make_shared<LoopFunc>("loop_rl", this->params.Get<float>("dt") * this->params.Get<int>("decimation"), std::bind(&RL_Sim::RunModel, this));
    this->loop_control->start();
    this->loop_rl->start();

    // keyboard
    this->loop_keyboard = std::make_shared<LoopFunc>("loop_keyboard", 0.05, std::bind(&RL_Sim::KeyboardInterface, this));
    this->loop_keyboard->start();

#ifdef PLOT
    this->plot_t = std::vector<int>(this->plot_size, 0);
    this->plot_real_joint_pos.resize(this->params.Get<int>("num_of_dofs"));
    this->plot_target_joint_pos.resize(this->params.Get<int>("num_of_dofs"));
    for (auto &vector : this->plot_real_joint_pos) { vector = std::vector<float>(this->plot_size, 0); }
    for (auto &vector : this->plot_target_joint_pos) { vector = std::vector<float>(this->plot_size, 0); }
    this->loop_plot = std::make_shared<LoopFunc>("loop_plot", 0.001, std::bind(&RL_Sim::Plot, this));
    this->loop_plot->start();
#endif
#ifdef CSV_LOGGER
    this->CSVInit(this->robot_name);
#endif

    std::cout << LOGGER::INFO << "RL_Sim start" << std::endl;
}

RL_Sim::~RL_Sim()
{
    this->loop_keyboard->shutdown();
    this->loop_control->shutdown();
    this->loop_rl->shutdown();
#ifdef PLOT
    this->loop_plot->shutdown();
#endif
    std::cout << LOGGER::INFO << "RL_Sim exit" << std::endl;
}

#if defined(USE_ROS2)
// 把关节名单按**模型（URDF）里的关节声明顺序**重排，只保留 names 里有的关节。
// 为什么需要：ROS2 控制器槽位顺序 = 传给它的 `joints` 顺序，而 base.yaml 的 joint_names 是
// Unitree SDK 的电机顺序；策略（以及 MuJoCo 路径的 sensordata/ctrl）用的是模型顺序。
// 顺序来源是模型唯一源里的纯 URDF（IMGO2_MODEL_DIR/urdf/imgo2.urdf，见 CMakeLists 与 README
// MODEL-02）；读不到或数量对不上就回退到原名单并告警，不影响启动。
std::vector<std::string> RL_Sim::OrderJointsByModelOrder(const std::vector<std::string>& names)
{
    const std::string urdf_path = std::string(IMGO2_MODEL_DIR) + "/urdf/imgo2.urdf";
    std::ifstream file(urdf_path);
    if (!file.good())
    {
        std::cout << LOGGER::WARNING << "Cannot open " << urdf_path
                  << " to order joints; using joint_names as-is" << std::endl;
        return names;
    }
    const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    const std::set<std::string> wanted(names.begin(), names.end());
    std::vector<std::string> ordered;
    const std::string tag = "<joint name=\"";
    std::size_t pos = 0;
    while ((pos = text.find(tag, pos)) != std::string::npos)
    {
        pos += tag.size();
        const std::size_t end = text.find('"', pos);
        if (end == std::string::npos)
        {
            break;
        }
        const std::string joint_name = text.substr(pos, end - pos);
        if (wanted.count(joint_name) != 0 &&
            std::find(ordered.begin(), ordered.end(), joint_name) == ordered.end())
        {
            ordered.push_back(joint_name);
        }
        pos = end;
    }
    if (ordered.size() != names.size())
    {
        std::cout << LOGGER::WARNING << "Ordered " << ordered.size() << " of " << names.size()
                  << " joints from " << urdf_path << "; using joint_names as-is" << std::endl;
        return names;
    }
    std::cout << LOGGER::INFO << "Joint order for the ROS 2 controller follows the URDF:";
    for (const auto& name : ordered)
    {
        std::cout << " " << name;
    }
    std::cout << std::endl;
    return ordered;
}
#endif

void RL_Sim::StartJointController(const std::string& ros_namespace, const std::vector<std::string>& names)
{
#if defined(USE_ROS1)
    pid_t pid0 = fork();
    if (pid0 == 0)
    {
        std::string cmd = "rosrun controller_manager spawner joint_state_controller ";
        for (const auto& name : names)
        {
            cmd += name + " ";
        }
        cmd += "__ns:=" + ros_namespace;
        // cmd += " > /dev/null 2>&1";  // Comment this line to see the output
        execlp("sh", "sh", "-c", cmd.c_str(), nullptr);
        exit(1);
    }
#elif defined(USE_ROS2)
    const char* ros_distro = std::getenv("ROS_DISTRO");
    std::string spawner = (ros_distro && std::string(ros_distro) == "foxy") ? "spawner.py" : "spawner";

    // controller_manager 不是 gzserver 一起步就有的：它是 gazebo_ros2_control 插件在机器人实体
    // spawn 成功之后才创建的节点。Gazebo 还在加载网格/还没 spawn 完就 spawn 控制器时，spawner 会
    // 重试 3 次×10 s 后非 0 退出，rl_sim 只能抛 "Failed to start joint controller"（用户实测，
    // 现象是打印完 Joint order 就停住然后 abort）。这里先显式等它出现并把等待过程打印出来，
    // 既避免误判"卡死"，也让失败信息能指向 Gazebo 侧。超时可用 base.yaml 的
    // controller_manager_timeout 覆盖（秒）。
    const double cm_timeout_s = this->params.Get<double>("controller_manager_timeout", 60.0);
    auto cm_client = ros2_node->create_client<rcl_interfaces::srv::GetParameters>("/controller_manager/get_parameters");
    bool cm_ready = false;
    bool cm_warned = false;
    const auto cm_deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(cm_timeout_s);
    while (!cm_ready)
    {
        if (!rclcpp::ok())
        {
            throw std::runtime_error("Interrupted while waiting for /controller_manager");
        }
        try
        {
            cm_ready = cm_client->wait_for_service(std::chrono::seconds(1));
        }
        catch (const std::exception& e)
        {
            throw std::runtime_error(std::string("Interrupted while waiting for /controller_manager: ") + e.what());
        }
        if (!cm_ready && !cm_warned)
        {
            cm_warned = true;
            std::cout << LOGGER::WARNING << "Waiting for /controller_manager (max " << cm_timeout_s
                      << " s); is Gazebo running? ros2 launch imgo2_deploy gazebo.launch.py" << std::endl;
        }
        if (!cm_ready && std::chrono::steady_clock::now() >= cm_deadline)
        {
            std::cout << LOGGER::ERROR << "/controller_manager did not show up within " << cm_timeout_s
                      << " s; giving up. Check the Gazebo terminal for 'Loaded gazebo_ros2_control' and"
                      << " 'Successfully spawned entity [<robot>_gazebo]'." << std::endl;
            throw std::runtime_error("controller_manager is not available");
        }
    }
    std::cout << LOGGER::INFO << "controller_manager is up; spawning robot_joint_controller" << std::endl;

    std::filesystem::path tmp_path = std::filesystem::temp_directory_path() / "robot_joint_controller_params.yaml";
    {
        std::ofstream tmp_file(tmp_path);
        if (!tmp_file)
        {
            throw std::runtime_error("Failed to create temporary parameter file");
        }

        tmp_file << "/robot_joint_controller:\n";
        tmp_file << "    ros__parameters:\n";
        tmp_file << "        joints:\n";
        for (const auto& name : names)
        {
            tmp_file << "            - " << name << "\n";
        }
    }

    pid_t pid = fork();
    if (pid == 0)
    {
        // 自成进程组：spawner 实际是 sh → python3 ros2 → python3 spawner 三层，父进程只 kill 直接
        // 子进程会把它下面两层留成孤儿（实测）。自成一组后父进程可以按组整棵收掉，也不会被终端的
        // Ctrl+C 抢先去打断（那样父进程就不知道子进程死没死）。
        setpgid(0, 0);
        std::string cmd = "ros2 run controller_manager " + spawner + " robot_joint_controller ";
        cmd += "-p " + tmp_path.string() + " ";
        // spawner 默认 --controller-manager-timeout 0 = 无限等 controller_manager：rl_sim 一旦被强杀，
        // 这个 fork 出来的子进程就成了"永远重试"的遗留进程（2026-09-18 用户机器上那个从 14:58 挂到
        // 16:5x 的 spawner 就是这么来的，它又反过来占着 DDS/SHM 端口）。给个上限让它自己也会退出。
        cmd += "--controller-manager-timeout 30 ";
        // cmd += " > /dev/null 2>&1";  // Comment this line to see the output
        execlp("sh", "sh", "-c", cmd.c_str(), nullptr);
        exit(1);
    }
    else if (pid > 0)
    {
        // 与子进程竞争着设进程组，谁先设上都行，失败（子进程已 exec）可忽略。
        setpgid(pid, pid);

        int status = 0;
        // 不要用阻塞式 waitpid：Ctrl+C 时 rclcpp 先让 context 失效，而阻塞等待在信号竞态下可能一直
        // 等下去（实测：SIGINT 后 rl_sim 仍活着、等到 spawner 自己超时为止，那 30 s 里既没退出也没
        // 清掉子进程）。改成 WNOHANG 轮询，就能立刻响应 Ctrl+C 并按进程组收掉整棵 spawner 进程树。
        bool interrupted = false;
        pid_t waited = 0;
        const auto spawner_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
        while (true)
        {
            waited = waitpid(pid, &status, WNOHANG);
            if (waited != 0)
            {
                break;  // >0：子进程已退出；-1：出错
            }
            if (!rclcpp::ok() || std::chrono::steady_clock::now() >= spawner_deadline)
            {
                interrupted = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        if (interrupted)
        {
            // 按进程组收掉整棵 spawner 进程树（sh → python3 ros2 → python3 spawner），否则它会在
            // 后台无限重试。2026-09-18 用户机器上那个从 14:58 挂到 16:5x 的 spawner 就是这么来的。
            if (kill(-pid, SIGTERM) == 0)
            {
                std::cout << LOGGER::WARNING << "Terminated the joint controller spawner process group (pgid "
                          << pid << ")" << std::endl;
            }
            waitpid(pid, &status, 0);
            std::cout << LOGGER::WARNING << "Interrupted while waiting for the joint controller spawner;"
                      << " generated parameters kept at " << tmp_path.string() << std::endl;
            throw std::runtime_error("Interrupted while starting joint controller");
        }

        if (waited == -1)
        {
            std::cout << LOGGER::ERROR << "waitpid failed while waiting for the spawner; generated parameters kept at "
                      << tmp_path.string() << std::endl;
            throw std::runtime_error("Failed to start joint controller");
        }

        if (WIFEXITED(status) && WEXITSTATUS(status) != 0)
        {
            std::cout << LOGGER::ERROR << "spawner exited with code " << WEXITSTATUS(status)
                      << "; generated parameters kept at " << tmp_path.string() << std::endl;
            throw std::runtime_error("Failed to start joint controller");
        }

        std::filesystem::remove(tmp_path);
    }
    else
    {
        throw std::runtime_error("fork() failed");
    }
#endif
}

void RL_Sim::GetState(RobotState<float> *state)
{
#if defined(USE_ROS1)
    const auto &orientation = this->pose.orientation;
    const auto &angular_velocity = this->vel.angular;
#elif defined(USE_ROS2)
    const auto &orientation = this->gazebo_imu.orientation;
    const auto &angular_velocity = this->gazebo_imu.angular_velocity;
#endif

    state->imu.quaternion[0] = orientation.w;
    state->imu.quaternion[1] = orientation.x;
    state->imu.quaternion[2] = orientation.y;
    state->imu.quaternion[3] = orientation.z;

    state->imu.gyroscope[0] = angular_velocity.x;
    state->imu.gyroscope[1] = angular_velocity.y;
    state->imu.gyroscope[2] = angular_velocity.z;

    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
#if defined(USE_ROS1)
        state->motor_state.q[i] = this->joint_positions[this->params.Get<std::vector<std::string>>("joint_controller_names")[this->params.Get<std::vector<int>>("joint_mapping")[i]]];
        state->motor_state.dq[i] = this->joint_velocities[this->params.Get<std::vector<std::string>>("joint_controller_names")[this->params.Get<std::vector<int>>("joint_mapping")[i]]];
        state->motor_state.tau_est[i] = this->joint_efforts[this->params.Get<std::vector<std::string>>("joint_controller_names")[this->params.Get<std::vector<int>>("joint_mapping")[i]]];
#elif defined(USE_ROS2)
        state->motor_state.q[i] = this->robot_state_subscriber_msg.motor_state[this->params.Get<std::vector<int>>("joint_mapping")[i]].q;
        state->motor_state.dq[i] = this->robot_state_subscriber_msg.motor_state[this->params.Get<std::vector<int>>("joint_mapping")[i]].dq;
        state->motor_state.tau_est[i] = this->robot_state_subscriber_msg.motor_state[this->params.Get<std::vector<int>>("joint_mapping")[i]].tau_est;
#endif
    }
}

void RL_Sim::SetCommand(const RobotCommand<float> *command)
{
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
#if defined(USE_ROS1)
        this->joint_publishers_commands[this->params.Get<std::vector<int>>("joint_mapping")[i]].q = command->motor_command.q[i];
        this->joint_publishers_commands[this->params.Get<std::vector<int>>("joint_mapping")[i]].dq = command->motor_command.dq[i];
        this->joint_publishers_commands[this->params.Get<std::vector<int>>("joint_mapping")[i]].kp = command->motor_command.kp[i];
        this->joint_publishers_commands[this->params.Get<std::vector<int>>("joint_mapping")[i]].kd = command->motor_command.kd[i];
        this->joint_publishers_commands[this->params.Get<std::vector<int>>("joint_mapping")[i]].tau = command->motor_command.tau[i];
#elif defined(USE_ROS2)
        this->robot_command_publisher_msg.motor_command[this->params.Get<std::vector<int>>("joint_mapping")[i]].q = command->motor_command.q[i];
        this->robot_command_publisher_msg.motor_command[this->params.Get<std::vector<int>>("joint_mapping")[i]].dq = command->motor_command.dq[i];
        this->robot_command_publisher_msg.motor_command[this->params.Get<std::vector<int>>("joint_mapping")[i]].kp = command->motor_command.kp[i];
        this->robot_command_publisher_msg.motor_command[this->params.Get<std::vector<int>>("joint_mapping")[i]].kd = command->motor_command.kd[i];
        this->robot_command_publisher_msg.motor_command[this->params.Get<std::vector<int>>("joint_mapping")[i]].tau = command->motor_command.tau[i];
#endif
    }

#if defined(USE_ROS1)
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        this->joint_publishers[this->params.Get<std::vector<std::string>>("joint_controller_names")[i]].publish(this->joint_publishers_commands[i]);
    }
#elif defined(USE_ROS2)
    // Ctrl+C 时 rclcpp 的信号处理会先把 context 置为失效，而 loop_control 线程还会再跑几拍；
    // 那时 publish / 发服务请求会抛 RCLError，线程里没人接就 std::terminate（用户实测
    // "could not create publisher: rcl node's context is invalid" → Aborted）。直接跳过即可。
    if (!rclcpp::ok())
    {
        return;
    }
    this->robot_command_publisher->publish(this->robot_command_publisher_msg);
#endif
}

void RL_Sim::RobotControl()
{
#if defined(USE_ROS2)
    // 同 SetCommand：context 失效后这一拍什么都不做，避免 shutdown 期间抛 RCLError 变 Aborted。
    if (!rclcpp::ok())
    {
        return;
    }
#endif
    this->GetState(&this->robot_state);

    this->StateController(&this->robot_state, &this->robot_command);

    if (this->control.current_keyboard == Input::Keyboard::R || this->control.current_gamepad == Input::Gamepad::RB_Y)
    {
#if defined(USE_ROS1)
        std_srvs::Empty empty;
        this->gazebo_reset_world_client.call(empty);
#elif defined(USE_ROS2)
        auto empty_request = std::make_shared<std_srvs::srv::Empty::Request>();
        auto result = this->gazebo_reset_world_client->async_send_request(empty_request);
#endif
        this->control.current_keyboard = this->control.last_keyboard;
    }
    if (this->control.current_keyboard == Input::Keyboard::Enter || this->control.current_gamepad == Input::Gamepad::RB_X)
    {
        if (simulation_running)
        {
#if defined(USE_ROS1)
            std_srvs::Empty empty;
            this->gazebo_pause_physics_client.call(empty);
#elif defined(USE_ROS2)
            auto empty_request = std::make_shared<std_srvs::srv::Empty::Request>();
            auto result = this->gazebo_pause_physics_client->async_send_request(empty_request);
#endif
            std::cout << std::endl << LOGGER::INFO << "Simulation Stop" << std::endl;
        }
        else
        {
#if defined(USE_ROS1)
            std_srvs::Empty empty;
            this->gazebo_unpause_physics_client.call(empty);
#elif defined(USE_ROS2)
            auto empty_request = std::make_shared<std_srvs::srv::Empty::Request>();
            auto result = this->gazebo_unpause_physics_client->async_send_request(empty_request);
#endif
            std::cout << std::endl << LOGGER::INFO << "Simulation Start" << std::endl;
        }
        simulation_running = !simulation_running;
        this->control.current_keyboard = this->control.last_keyboard;
    }

    this->control.ClearInput();

    this->SetCommand(&this->robot_command);
}

#if defined(USE_ROS1)
void RL_Sim::ModelStatesCallback(const gazebo_msgs::ModelStates::ConstPtr &msg)
{
    this->vel = msg->twist[2];
    this->pose = msg->pose[2];
}
#elif defined(USE_ROS2)
void RL_Sim::GazeboImuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
{
    this->gazebo_imu = *msg;
}
#endif

void RL_Sim::CmdvelCallback(
#if defined(USE_ROS1)
    const geometry_msgs::Twist::ConstPtr &msg
#elif defined(USE_ROS2)
    const geometry_msgs::msg::Twist::SharedPtr msg
#endif
)
{
    this->cmd_vel = *msg;
}

void RL_Sim::JoyCallback(
#if defined(USE_ROS1)
    const sensor_msgs::Joy::ConstPtr &msg
#elif defined(USE_ROS2)
    const sensor_msgs::msg::Joy::SharedPtr msg
#endif
)
{
    this->joy_msg = *msg;

    // joystick control
    // Description of buttons and axes(F710):
    // |__ buttons[]: A=0, B=1, X=2, Y=3, LB=4, RB=5, back=6, start=7, power=8, stickL=9, stickR=10
    // |__ axes[]: Lx=0, Ly=1, Rx=3, Ry=4, LT=2, RT=5, DPadX=6, DPadY=7

    if (this->joy_msg.buttons[0]) this->control.SetGamepad(Input::Gamepad::A);
    if (this->joy_msg.buttons[1]) this->control.SetGamepad(Input::Gamepad::B);
    if (this->joy_msg.buttons[2]) this->control.SetGamepad(Input::Gamepad::X);
    if (this->joy_msg.buttons[3]) this->control.SetGamepad(Input::Gamepad::Y);
    if (this->joy_msg.buttons[4]) this->control.SetGamepad(Input::Gamepad::LB);
    if (this->joy_msg.buttons[5]) this->control.SetGamepad(Input::Gamepad::RB);
    if (this->joy_msg.buttons[9]) this->control.SetGamepad(Input::Gamepad::LStick);
    if (this->joy_msg.buttons[10]) this->control.SetGamepad(Input::Gamepad::RStick);
    if (this->joy_msg.axes[7] > 0) this->control.SetGamepad(Input::Gamepad::DPadUp);
    if (this->joy_msg.axes[7] < 0) this->control.SetGamepad(Input::Gamepad::DPadDown);
    if (this->joy_msg.axes[6] < 0) this->control.SetGamepad(Input::Gamepad::DPadLeft);
    if (this->joy_msg.axes[6] > 0) this->control.SetGamepad(Input::Gamepad::DPadRight);
    if (this->joy_msg.buttons[4] && this->joy_msg.buttons[0]) this->control.SetGamepad(Input::Gamepad::LB_A);
    if (this->joy_msg.buttons[4] && this->joy_msg.buttons[1]) this->control.SetGamepad(Input::Gamepad::LB_B);
    if (this->joy_msg.buttons[4] && this->joy_msg.buttons[2]) this->control.SetGamepad(Input::Gamepad::LB_X);
    if (this->joy_msg.buttons[4] && this->joy_msg.buttons[3]) this->control.SetGamepad(Input::Gamepad::LB_Y);
    if (this->joy_msg.buttons[4] && this->joy_msg.buttons[9]) this->control.SetGamepad(Input::Gamepad::LB_LStick);
    if (this->joy_msg.buttons[4] && this->joy_msg.buttons[10]) this->control.SetGamepad(Input::Gamepad::LB_RStick);
    if (this->joy_msg.buttons[4] && this->joy_msg.axes[7] > 0) this->control.SetGamepad(Input::Gamepad::LB_DPadUp);
    if (this->joy_msg.buttons[4] && this->joy_msg.axes[7] < 0) this->control.SetGamepad(Input::Gamepad::LB_DPadDown);
    if (this->joy_msg.buttons[4] && this->joy_msg.axes[6] < 0) this->control.SetGamepad(Input::Gamepad::LB_DPadRight);
    if (this->joy_msg.buttons[4] && this->joy_msg.axes[6] > 0) this->control.SetGamepad(Input::Gamepad::LB_DPadLeft);
    if (this->joy_msg.buttons[5] && this->joy_msg.buttons[0]) this->control.SetGamepad(Input::Gamepad::RB_A);
    if (this->joy_msg.buttons[5] && this->joy_msg.buttons[1]) this->control.SetGamepad(Input::Gamepad::RB_B);
    if (this->joy_msg.buttons[5] && this->joy_msg.buttons[2]) this->control.SetGamepad(Input::Gamepad::RB_X);
    if (this->joy_msg.buttons[5] && this->joy_msg.buttons[3]) this->control.SetGamepad(Input::Gamepad::RB_Y);
    if (this->joy_msg.buttons[5] && this->joy_msg.buttons[9]) this->control.SetGamepad(Input::Gamepad::RB_LStick);
    if (this->joy_msg.buttons[5] && this->joy_msg.buttons[10]) this->control.SetGamepad(Input::Gamepad::RB_RStick);
    if (this->joy_msg.buttons[5] && this->joy_msg.axes[7] > 0) this->control.SetGamepad(Input::Gamepad::RB_DPadUp);
    if (this->joy_msg.buttons[5] && this->joy_msg.axes[7] < 0) this->control.SetGamepad(Input::Gamepad::RB_DPadDown);
    if (this->joy_msg.buttons[5] && this->joy_msg.axes[6] < 0) this->control.SetGamepad(Input::Gamepad::RB_DPadRight);
    if (this->joy_msg.buttons[5] && this->joy_msg.axes[6] > 0) this->control.SetGamepad(Input::Gamepad::RB_DPadLeft);
    if (this->joy_msg.buttons[4] && this->joy_msg.buttons[5]) this->control.SetGamepad(Input::Gamepad::LB_RB);

    this->control.x = this->joy_msg.axes[1]; // LY
    this->control.y = this->joy_msg.axes[0]; // LX
    this->control.yaw = this->joy_msg.axes[3]; // RX
}

#if defined(USE_ROS1)
void RL_Sim::JointStatesCallback(const robot_msgs::MotorState::ConstPtr &msg, const std::string &joint_controller_name)
{
    this->joint_positions[joint_controller_name] = msg->q;
    this->joint_velocities[joint_controller_name] = msg->dq;
    this->joint_efforts[joint_controller_name] = msg->tau_est;
}
#elif defined(USE_ROS2)
void RL_Sim::RobotStateCallback(const robot_msgs::msg::RobotState::SharedPtr msg)
{
    this->robot_state_subscriber_msg = *msg;
}
#endif

void RL_Sim::RunModel()
{
    if (this->rl_init_done && simulation_running)
    {
        this->episode_length_buf += 1;
        this->obs.ang_vel = this->robot_state.imu.gyroscope;
        this->obs.commands = {this->control.x, this->control.y, this->control.yaw};
        if (this->control.navigation_mode)
        {
            this->obs.commands = {(float)this->cmd_vel.linear.x, (float)this->cmd_vel.linear.y, (float)this->cmd_vel.angular.z};
        }
        this->obs.base_quat = this->robot_state.imu.quaternion;
        this->obs.dof_pos = this->robot_state.motor_state.q;
        this->obs.dof_vel = this->robot_state.motor_state.dq;

        this->obs.actions = this->Forward();
        this->ComputeOutput(this->obs.actions, this->output_dof_pos, this->output_dof_vel, this->output_dof_tau);

        if (!this->output_dof_pos.empty())
        {
            output_dof_pos_queue.push(this->output_dof_pos);
        }
        if (!this->output_dof_vel.empty())
        {
            output_dof_vel_queue.push(this->output_dof_vel);
        }
        if (!this->output_dof_tau.empty())
        {
            output_dof_tau_queue.push(this->output_dof_tau);
        }

        // this->TorqueProtect(this->output_dof_tau);
        // this->AttitudeProtect(this->robot_state.imu.quaternion, 75.0f, 75.0f);

#ifdef CSV_LOGGER
        std::vector<float> tau_est(this->params.Get<int>("num_of_dofs"), 0.0f);
        for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
        {
            tau_est[i] = this->joint_efforts[this->params.Get<std::vector<std::string>>("joint_controller_names")[i]];
        }
        this->CSVLogger(this->output_dof_tau, tau_est, this->obs.dof_pos, this->output_dof_pos, this->obs.dof_vel);
#endif
    }
}

std::vector<float> RL_Sim::Forward()
{
    std::unique_lock<std::mutex> lock(this->model_mutex, std::try_to_lock);

    // If model is being reinitialized, return previous actions to avoid blocking
    if (!lock.owns_lock())
    {
        std::cout << LOGGER::WARNING << "Model is being reinitialized, using previous actions" << std::endl;
        return this->obs.actions;
    }

    std::vector<float> clamped_obs = this->ComputeObservation();

    std::vector<float> actions;
    if (this->params.Get<std::vector<int>>("observations_history").size() != 0)
    {
        this->history_obs_buf.insert(clamped_obs);
        this->history_obs = this->history_obs_buf.get_obs_vec(this->params.Get<std::vector<int>>("observations_history"));
        actions = this->model->forward({this->history_obs});
    }
    else
    {
        actions = this->model->forward({clamped_obs});
    }

    if (!this->params.Get<std::vector<float>>("clip_actions_upper").empty() && !this->params.Get<std::vector<float>>("clip_actions_lower").empty())
    {
        return clamp(actions, this->params.Get<std::vector<float>>("clip_actions_lower"), this->params.Get<std::vector<float>>("clip_actions_upper"));
    }
    else
    {
        return actions;
    }
}

void RL_Sim::Plot()
{
    this->plot_t.erase(this->plot_t.begin());
    this->plot_t.push_back(this->motiontime);
    plt::cla();
    plt::clf();
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        this->plot_real_joint_pos[i].erase(this->plot_real_joint_pos[i].begin());
        this->plot_target_joint_pos[i].erase(this->plot_target_joint_pos[i].begin());
#if defined(USE_ROS1)
        this->plot_real_joint_pos[i].push_back(this->joint_positions[this->params.Get<std::vector<std::string>>("joint_controller_names")[i]]);
        this->plot_target_joint_pos[i].push_back(this->joint_publishers_commands[i].q);
#elif defined(USE_ROS2)
        this->plot_real_joint_pos[i].push_back(this->robot_state_subscriber_msg.motor_state[i].q);
        this->plot_target_joint_pos[i].push_back(this->robot_command_publisher_msg.motor_command[i].q);
#endif
        plt::subplot(this->params.Get<int>("num_of_dofs"), 1, i + 1);
        plt::named_plot("_real_joint_pos", this->plot_t, this->plot_real_joint_pos[i], "r");
        plt::named_plot("_target_joint_pos", this->plot_t, this->plot_target_joint_pos[i], "b");
        plt::xlim(this->plot_t.front(), this->plot_t.back());
    }
    // plt::legend();
    plt::pause(0.01);
}

#if defined(USE_ROS1)
void signalHandler(int signum)
{
    ros::shutdown();
    exit(0);
}
#endif

int main(int argc, char **argv)
{
#if defined(USE_ROS1)
    signal(SIGINT, signalHandler);
    ros::init(argc, argv, "imgo2_deploy");
    RL_Sim imgo2_deploy(argc, argv);
    ros::spin();
#elif defined(USE_ROS2)
    rclcpp::init(argc, argv);
    try
    {
        auto imgo2_deploy = std::make_shared<RL_Sim>(argc, argv);
        rclcpp::spin(imgo2_deploy->ros2_node);
    }
    catch (const std::exception& e)
    {
        // 启动期失败（等不到 /controller_manager、spawner 起不来、Ctrl+C 打断）都走这里，
        // 打印一行可读原因后正常退出，而不是 uncaught exception → "terminate called" → Aborted。
        std::cout << LOGGER::ERROR << e.what() << std::endl;
        rclcpp::shutdown();
        return 1;
    }
    rclcpp::shutdown();
#endif
    return 0;
}
