/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rl_sim_mujoco.hpp"

RL_Sim* RL_Sim::instance = nullptr;
namespace { volatile std::sig_atomic_t exit_requested = 0; }

RL_Sim::RL_Sim(int argc, char **argv)
{
    // Set static instance pointer early for signal handler
    instance = this;
    try
    {

        if (argc < 3)
        {
            std::cout << LOGGER::ERROR << "Usage: " << argv[0] << " imgo2 scene_name" << std::endl;
            throw std::runtime_error("Invalid arguments");
        }
        else
        {
            this->robot_name = argv[1];
            this->scene_name = argv[2];
        }
        this->towing_scene = this->scene_name == "scene_tow_compliant" ||
                             this->scene_name == "scene_tow_inextensible";
        if (argc != 3 && (argc != 5 || std::string(argv[3]) != "--cart-mass" || !this->towing_scene))
        {
            throw std::runtime_error("Usage: rl_sim_mujoco imgo2 <scene> [--cart-mass 5..15]");
        }
        if (argc == 5)
        {
            std::size_t consumed = 0;
            this->cart_mass_kg = std::stod(argv[4], &consumed);
            if (consumed != std::string(argv[4]).size() || !std::isfinite(this->cart_mass_kg) ||
                this->cart_mass_kg < 5.0 || this->cart_mass_kg > 15.0)
            {
                throw std::runtime_error("--cart-mass must be between 5 and 15 kg");
            }
        }

        this->ang_vel_axis = "body";

        // now launch mujoco
        std::cout << LOGGER::INFO << "[MuJoCo] Launching..." << std::endl;

        // display an error if running on macOS under Rosetta 2
    #if defined(__APPLE__) && defined(__AVX__)
        if (rosetta_error_msg)
        {
            DisplayErrorDialogBox("Rosetta 2 is not supported", rosetta_error_msg);
            std::exit(1);
        }
    #endif

        // print version, check compatibility
        std::cout << LOGGER::INFO << "[MuJoCo] Version: " << mj_versionString() << std::endl;
        if (mjVERSION_HEADER != mj_version())
        {
            mju_error("Headers and library have different versions");
        }

        // scan for libraries in the plugin directory to load additional plugins
        scanPluginLibraries();

        mjvCamera cam;
        mjv_defaultCamera(&cam);

        mjvOption opt;
        mjv_defaultOption(&opt);

        mjvPerturb pert;
        mjv_defaultPerturb(&pert);

        // simulate object encapsulates the UI
        sim = std::make_unique<mj::Simulate>(
            std::make_unique<mj::GlfwAdapter>(),
            &cam, &opt, &pert, /* is_passive = */ false);

        // 模型唯一源：<repo>/imgo2_description/mjcf/<scene>.xml（含 imgo2.xml）。
        // 2026-09-17 统一前这里是 ../../robot_description/<robot>_mjcf/<scene>.xml。
        std::string filename = std::string(IMGO2_MODEL_DIR) + "/mjcf/" + this->scene_name + ".xml";

        // start physics thread
        // Keep physics paused until the first complete Passive command is ready.
        sim->run = 0;
        motor_pd = std::make_unique<MujocoMotorPd>();
        physics_feedback_hook = &RL_Sim::SampleFeedback;
        physics_thread = std::thread(&PhysicsThread, sim.get(), filename.c_str());

        const auto load_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (true)
        {
            if (exit_requested || std::chrono::steady_clock::now() > load_deadline)
                throw std::runtime_error("MuJoCo model loading interrupted or timed out");
            {
                std::lock_guard<std::recursive_mutex> lock(sim->mtx);
                if (d)
                {
                    this->mj_model = m;
                    this->mj_data = d;
                    mj_forward(m, d);
                    std::cout << LOGGER::INFO << "[MuJoCo] Data prepared" << std::endl;
                    break;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        if (this->towing_scene)
        {
            this->ConfigureCartMass();
        }
        this->SetupSysJoystick("/dev/input/js0", 16); // 16 bits joystick

        // read params from yaml
        this->ReadYaml(this->robot_name, "base.yaml");

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
        this->InitJointNum(this->params.Get<int>("num_of_dofs"));
        this->InitOutputs();
        this->InitControl();
        motor_feedback_delay_ms = params.Get<double>("mujoco_motor_feedback_delay_ms", 0);
        imu_feedback_delay_ms = params.Get<double>("mujoco_imu_feedback_delay_ms", 0);
        ConfigureMotorInterfaces();
        for (auto& kd : robot_command.motor_command.kd) kd = 8; // initial Passive, even during delay warmup
        SetCommand(&robot_command);
        std::cout << "[MuJoCo] feedback delays motor=" << motor_feedback_delay_ms
                  << " ms, imu=" << imu_feedback_delay_ms << " ms (simulation time)" << std::endl;
        RobotControl();  // publish initial Passive gains before allowing physics to run
        {
            std::lock_guard<std::recursive_mutex> lock(sim->mtx);
            sim->run = 1;
        }

        // loop
        this->loop_control = std::make_shared<LoopFunc>("loop_control", this->params.Get<float>("dt"), std::bind(&RL_Sim::RobotControl, this));
        this->loop_rl = std::make_shared<LoopFunc>("loop_rl", this->params.Get<float>("dt") * this->params.Get<int>("decimation"), std::bind(&RL_Sim::RunModel, this));
        this->loop_control->start(false);
        this->loop_rl->start(false);

        // keyboard
        this->loop_keyboard = std::make_shared<LoopFunc>("loop_keyboard", 0.05, std::bind(&RL_Sim::KeyboardInterface, this));
        this->loop_keyboard->start(false);

        // joystick
        this->loop_joystick = std::make_shared<LoopFunc>("loop_joystick", 0.01, std::bind(&RL_Sim::GetSysJoystick, this));
        this->loop_joystick->start(false);

    #ifdef PLOT
        this->plot_t = std::vector<int>(this->plot_size, 0);
        this->plot_real_joint_pos.resize(this->params.Get<int>("num_of_dofs"));
        this->plot_target_joint_pos.resize(this->params.Get<int>("num_of_dofs"));
        for (auto &vector : this->plot_real_joint_pos) { vector = std::vector<float>(this->plot_size, 0); }
        for (auto &vector : this->plot_target_joint_pos) { vector = std::vector<float>(this->plot_size, 0); }
        this->loop_plot = std::make_shared<LoopFunc>("loop_plot", 0.001, std::bind(&RL_Sim::Plot, this));
        this->loop_plot->start(false);
    #endif
    #ifdef CSV_LOGGER
        this->CSVInit(this->robot_name);
    #endif

        std::cout << LOGGER::INFO << "RL_Sim start" << std::endl;

        // start simulation UI loop (blocking call)
        sim->RenderLoop();
    }
    catch (...)
    {
        StopResources();
        instance = nullptr;
        throw;
    }
}

void RL_Sim::ConfigureCartMass()
{
    const std::lock_guard<std::recursive_mutex> lock(sim->mtx);
    // All masses and principal inertias scale together from the 10 kg URDF model.
    const double scale = this->cart_mass_kg / 10.0;
    for (const char* name : {"cart_base", "cart_wheel_fl", "cart_wheel_fr",
                             "cart_wheel_rl", "cart_wheel_rr"})
    {
        const int id = mj_name2id(this->mj_model, mjOBJ_BODY, name);
        if (id < 0)
        {
            throw std::runtime_error(std::string("Towing scene missing cart body: ") + name);
        }
        this->mj_model->body_mass[id] *= scale;
        for (int axis = 0; axis < 3; ++axis)
        {
            this->mj_model->body_inertia[3 * id + axis] *= scale;
        }
    }
    mj_setConst(this->mj_model, this->mj_data);
    mj_forward(this->mj_model, this->mj_data);
    std::cout << LOGGER::INFO << "[MuJoCo] Cart total mass: " << this->cart_mass_kg << " kg" << std::endl;
}

void RL_Sim::StopResources()
{
    if (sim) sim->exitrequest.store(1);
    if (loop_keyboard) loop_keyboard->shutdown();
    if (loop_joystick) loop_joystick->shutdown();
    if (loop_control) loop_control->shutdown();
    if (loop_rl) loop_rl->shutdown();
#ifdef PLOT
    if (loop_plot) loop_plot->shutdown();
#endif
    if (physics_thread.joinable()) physics_thread.join();
    // No physics/GUI evaluation can use the global callback after this point.
    physics_feedback_hook = nullptr;
    motor_pd.reset();
}

RL_Sim::~RL_Sim()
{
    StopResources();
    instance = nullptr;
    std::cout << LOGGER::INFO << "RL_Sim exit" << std::endl;
}

void RL_Sim::ConfigureMotorInterfaces()
{
    pd_mapping = params.Get<std::vector<int>>("joint_mapping");
    motor_pd->Bind(mj_model, pd_mapping);
    feedback.Configure(mj_model, pd_mapping, motor_feedback_delay_ms, imu_feedback_delay_ms);
    feedback_ready = false;
}

void RL_Sim::SampleFeedback(const mjModel* model, const mjData* data)
{
    if (instance) instance->feedback.Sample(model, data);
}

void RL_Sim::GetState(RobotState<float> *state)
{
    feedback.Sample(mj_model, mj_data); // also handles startup, GUI single-step and reset
    MujocoFeedback::State sample;
    feedback_ready = feedback.Read(mj_data->time, sample);
    if (!feedback_ready) return; // never leak fresh data during delay warmup
    for (int i=0;i<12;++i)
    {
        state->motor_state.q[i] = sample.motors.q[i];
        state->motor_state.dq[i] = sample.motors.dq[i];
        state->motor_state.tau_est[i] = sample.motors.tau[i];
    }
    for (int i=0;i<4;++i) state->imu.quaternion[i] = sample.imu.quat[i];
    for (int i=0;i<3;++i) state->imu.gyroscope[i] = sample.imu.gyro[i];
}

void RL_Sim::SetCommand(const RobotCommand<float> *command)
{
    if (pd_mapping != params.Get<std::vector<int>>("joint_mapping")) ConfigureMotorInterfaces();
    const auto& c = command->motor_command;
    motor_pd->Publish(c.q, c.dq, c.kp, c.kd, c.tau);

}

void RL_Sim::RobotControl()
{
    // Lock the sim mutex once for the entire control cycle to prevent race conditions
    const std::lock_guard<std::recursive_mutex> lock(sim->mtx);
    if (exit_requested) sim->exitrequest.store(1);
    if (sim->exitrequest.load() || !m || !d) return;
    // The viewer may reload its model. Rebind before touching cached pointers;
    // the callback ignores an unbound model during loading/compilation.
    if (mj_model != m || mj_data != d)
    {
        mj_model = m;
        mj_data = d;
        try { ConfigureMotorInterfaces(); }
        catch (const std::exception& e)
        {
            std::cerr << "MuJoCo reload rejected: " << e.what() << std::endl;
            sim->exitrequest.store(1);
            return;
        }
    }

    {
        std::lock_guard<std::mutex> state_lock(state_mutex);
        this->GetState(&this->robot_state);
    }

    if (feedback_ready) this->StateController(&this->robot_state, &this->robot_command);

    if (this->control.current_keyboard == Input::Keyboard::R || this->control.current_gamepad == Input::Gamepad::RB_Y)
    {
        if (this->mj_model && this->mj_data)
        {
            feedback.Reset();
            feedback_ready = false;
            mj_resetData(this->mj_model, this->mj_data);
            mj_forward(this->mj_model, this->mj_data);
        }
    }
    if (this->control.current_keyboard == Input::Keyboard::Enter || this->control.current_gamepad == Input::Gamepad::RB_X)
    {
        if (simulation_running)
        {
            sim->run = 0;
            std::cout << std::endl << LOGGER::INFO << "Simulation Stop" << std::endl;
        }
        else
        {
            sim->run = 1;
            std::cout << std::endl << LOGGER::INFO << "Simulation Start" << std::endl;
        }
        simulation_running = !simulation_running;
    }

    this->control.ClearInput();

    this->SetCommand(&this->robot_command);
}

void RL_Sim::SetupSysJoystick(const std::string& device, int bits)
{
    this->sys_js = std::make_unique<Joystick>(device);
    if (!this->sys_js->isFound())
    {
        std::cout << LOGGER::ERROR << "Joystick [" << device << "] open failed." << std::endl;
        // exit(1);
    }

    this->sys_js_max_value = (1 << (bits - 1));
}

void RL_Sim::GetSysJoystick()
{
    // Clear all button event states
    for (int i = 0; i < 20; ++i)
    {
        this->sys_js_button[i].on_press = false;
        this->sys_js_button[i].on_release = false;
    }

    // Check if joystick is valid before using
    if (!this->sys_js)
    {
        return;
    }

    while (this->sys_js->sample(&this->sys_js_event))
    {
        if (this->sys_js_event.isButton())
        {
            this->sys_js_button[this->sys_js_event.number].update(this->sys_js_event.value);
        }
        else if (this->sys_js_event.isAxis())
        {
            double normalized = double(this->sys_js_event.value) / this->sys_js_max_value;
            if (std::abs(normalized) < this->axis_deadzone)
            {
                this->sys_js_axis[this->sys_js_event.number] = 0;
            }
            else
            {
                this->sys_js_axis[this->sys_js_event.number] = this->sys_js_event.value;
            }
        }
    }

    if (this->sys_js_button[0].on_press) this->control.SetGamepad(Input::Gamepad::A);
    if (this->sys_js_button[1].on_press) this->control.SetGamepad(Input::Gamepad::B);
    if (this->sys_js_button[2].on_press) this->control.SetGamepad(Input::Gamepad::X);
    if (this->sys_js_button[3].on_press) this->control.SetGamepad(Input::Gamepad::Y);
    if (this->sys_js_button[4].on_press) this->control.SetGamepad(Input::Gamepad::LB);
    if (this->sys_js_button[5].on_press) this->control.SetGamepad(Input::Gamepad::RB);
    if (this->sys_js_button[9].on_press) this->control.SetGamepad(Input::Gamepad::LStick);
    if (this->sys_js_button[10].on_press) this->control.SetGamepad(Input::Gamepad::RStick);
    if (this->sys_js_axis[7] < 0) this->control.SetGamepad(Input::Gamepad::DPadUp);
    if (this->sys_js_axis[7] > 0) this->control.SetGamepad(Input::Gamepad::DPadDown);
    if (this->sys_js_axis[6] > 0) this->control.SetGamepad(Input::Gamepad::DPadLeft);
    if (this->sys_js_axis[6] < 0) this->control.SetGamepad(Input::Gamepad::DPadRight);
    if (this->sys_js_button[4].pressed && this->sys_js_button[0].on_press) this->control.SetGamepad(Input::Gamepad::LB_A);
    if (this->sys_js_button[4].pressed && this->sys_js_button[1].on_press) this->control.SetGamepad(Input::Gamepad::LB_B);
    if (this->sys_js_button[4].pressed && this->sys_js_button[2].on_press) this->control.SetGamepad(Input::Gamepad::LB_X);
    if (this->sys_js_button[4].pressed && this->sys_js_button[3].on_press) this->control.SetGamepad(Input::Gamepad::LB_Y);
    if (this->sys_js_button[4].pressed && this->sys_js_button[9].on_press) this->control.SetGamepad(Input::Gamepad::LB_LStick);
    if (this->sys_js_button[4].pressed && this->sys_js_button[10].on_press) this->control.SetGamepad(Input::Gamepad::LB_RStick);
    if (this->sys_js_button[4].pressed && this->sys_js_axis[7] < 0) this->control.SetGamepad(Input::Gamepad::LB_DPadUp);
    if (this->sys_js_button[4].pressed && this->sys_js_axis[7] > 0) this->control.SetGamepad(Input::Gamepad::LB_DPadDown);
    if (this->sys_js_button[4].pressed && this->sys_js_axis[6] > 0) this->control.SetGamepad(Input::Gamepad::LB_DPadRight);
    if (this->sys_js_button[4].pressed && this->sys_js_axis[6] < 0) this->control.SetGamepad(Input::Gamepad::LB_DPadLeft);
    if (this->sys_js_button[5].pressed && this->sys_js_button[0].on_press) this->control.SetGamepad(Input::Gamepad::RB_A);
    if (this->sys_js_button[5].pressed && this->sys_js_button[1].on_press) this->control.SetGamepad(Input::Gamepad::RB_B);
    if (this->sys_js_button[5].pressed && this->sys_js_button[2].on_press) this->control.SetGamepad(Input::Gamepad::RB_X);
    if (this->sys_js_button[5].pressed && this->sys_js_button[3].on_press) this->control.SetGamepad(Input::Gamepad::RB_Y);
    if (this->sys_js_button[5].pressed && this->sys_js_button[9].on_press) this->control.SetGamepad(Input::Gamepad::RB_LStick);
    if (this->sys_js_button[5].pressed && this->sys_js_button[10].on_press) this->control.SetGamepad(Input::Gamepad::RB_RStick);
    if (this->sys_js_button[5].pressed && this->sys_js_axis[7] < 0) this->control.SetGamepad(Input::Gamepad::RB_DPadUp);
    if (this->sys_js_button[5].pressed && this->sys_js_axis[7] > 0) this->control.SetGamepad(Input::Gamepad::RB_DPadDown);
    if (this->sys_js_button[5].pressed && this->sys_js_axis[6] > 0) this->control.SetGamepad(Input::Gamepad::RB_DPadRight);
    if (this->sys_js_button[5].pressed && this->sys_js_axis[6] < 0) this->control.SetGamepad(Input::Gamepad::RB_DPadLeft);
    if (this->sys_js_button[4].pressed && this->sys_js_button[5].on_press) this->control.SetGamepad(Input::Gamepad::LB_RB);

    float ly = -float(this->sys_js_axis[1]) / float(this->sys_js_max_value);
    float lx = -float(this->sys_js_axis[0]) / float(this->sys_js_max_value);
    float rx = -float(this->sys_js_axis[3]) / float(this->sys_js_max_value);

    bool has_input = (ly != 0.0f || lx != 0.0f || rx != 0.0f);

    if (has_input)
    {
        this->control.x = ly;
        this->control.y = lx;
        this->control.yaw = rx;
        this->sys_js_active = true;
    }
    else if (this->sys_js_active)
    {
        this->control.x = 0.0f;
        this->control.y = 0.0f;
        this->control.yaw = 0.0f;
        this->sys_js_active = false;
    }
}

void RL_Sim::RunModel()
{
    if (this->rl_init_done && simulation_running && feedback_ready)
    {
        RobotState<float> policy_state;
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            policy_state = robot_state;
        }
        this->episode_length_buf += 1;
        this->obs.ang_vel = policy_state.imu.gyroscope;
        this->obs.commands = {this->control.x, this->control.y, this->control.yaw};
        //not currently available for non-ros mujoco version
        // if (this->control.navigation_mode)
        // {
        //     this->obs.commands = {(float)this->cmd_vel.linear.x, (float)this->cmd_vel.linear.y, (float)this->cmd_vel.angular.z};
        // }
        this->obs.base_quat = policy_state.imu.quaternion;
        this->obs.dof_pos = policy_state.motor_state.q;
        this->obs.dof_vel = policy_state.motor_state.dq;

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
        // this->AttitudeProtect(policy_state.imu.quaternion, 75.0f, 75.0f);

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
        this->plot_real_joint_pos[i].push_back(mj_data->sensordata[i]);
        // this->plot_target_joint_pos[i].push_back();  // TODO
        plt::subplot(this->params.Get<int>("num_of_dofs"), 1, i + 1);
        plt::named_plot("_real_joint_pos", this->plot_t, this->plot_real_joint_pos[i], "r");
        plt::named_plot("_target_joint_pos", this->plot_t, this->plot_target_joint_pos[i], "b");
        plt::xlim(this->plot_t.front(), this->plot_t.back());
    }
    // plt::legend();
    plt::pause(0.01);
}

// Signal handler for Ctrl+C
void signalHandler(int)
{
    exit_requested = 1;
}

int main(int argc, char **argv)
{
    std::string robot_name(argc > 1 ? argv[1] : "");
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    try
    {
        RL_Sim imgo2_deploy(argc, argv);
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "MuJoCo initialization failed: " << e.what() << std::endl;
        return 1;
    }
}
