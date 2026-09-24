/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef IMGO2_FSM_HPP
#define IMGO2_FSM_HPP

#include "fsm.hpp"
#include "rl_sdk.hpp"

namespace imgo2_fsm
{

class RLFSMStatePassive : public RLFSMState
{
public:
    RLFSMStatePassive(RL *rl) : RLFSMState(*rl, "RLFSMStatePassive") {}

    void Enter() override
    {
        std::cout << LOGGER::NOTE << "Entered passive mode. Press '0' (Keyboard) or 'A' (Gamepad) to switch to RLFSMStateGetUp." << std::endl;
    }

    void Run() override
    {
        for (int i = 0; i < rl.params.Get<int>("num_of_dofs"); ++i)
        {
            // fsm_command->motor_command.q[i] = fsm_state->motor_state.q[i];
            fsm_command->motor_command.dq[i] = 0;
            fsm_command->motor_command.kp[i] = 0;
            fsm_command->motor_command.kd[i] = 8;
            fsm_command->motor_command.tau[i] = 0;
        }
    }

    void Exit() override {}

    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::Num0 || rl.control.current_gamepad == Input::Gamepad::A)
        {
            return "RLFSMStateGetUp";
        }
        return state_name_;
    }
};

class RLFSMStateGetUp : public RLFSMState
{
public:
    RLFSMStateGetUp(RL *rl) : RLFSMState(*rl, "RLFSMStateGetUp") {}

    float percent_pre_getup = 0.0f;
    float percent_getup = 0.0f;
    std::vector<float> pre_running_pos = {
        0.00, 1.36, -2.65,
        0.00, 1.36, -2.65,
        0.00, 1.36, -2.65,
        0.00, 1.36, -2.65,
        0.00, 0.00, 0.00, 0.00
    };
    bool stand_from_passive = true;

    void Enter() override
    {
        percent_pre_getup = 0.0f;
        percent_getup = 0.0f;
        if (rl.fsm.previous_state_->GetStateName() == "RLFSMStatePassive")
        {
            stand_from_passive = true;
        }
        else
        {
            stand_from_passive = false;
        }
        rl.now_state = *fsm_state;
        rl.start_state = rl.now_state;
    }

    void Run() override
    {
        if(stand_from_passive)
        {

            if (Interpolate(percent_pre_getup, rl.now_state.motor_state.q, pre_running_pos, 1.0f, "Pre Getting up", true)) return;
            if (Interpolate(percent_getup, pre_running_pos, rl.params.Get<std::vector<float>>("default_dof_pos"), 2.0f, "Getting up", true)) return;
        }
        else
        {
            if (Interpolate(percent_getup, rl.now_state.motor_state.q, rl.params.Get<std::vector<float>>("default_dof_pos"), 1.0f, "Getting up", true)) return;
        }
    }

    void Exit() override {}

    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::P || rl.control.current_gamepad == Input::Gamepad::LB_X)
        {
            return "RLFSMStatePassive";
        }
        if (percent_getup >= 1.0f)
        {
            // 策略按键统一：1 = PPO，2 = himloco，3 = amp
            // （与各 locomotion 状态的 CheckChange 保持一致）
            if (rl.control.current_keyboard == Input::Keyboard::Num1 || rl.control.current_gamepad == Input::Gamepad::RB_DPadUp)
            {
                return "RLFSMStatePPOLocomotion";
            }
            else if (rl.control.current_keyboard == Input::Keyboard::Num2 || rl.control.current_gamepad == Input::Gamepad::RB_DPadRight)
            {
                return "RLFSMStateRLLocomotion";
            }
            else if (rl.control.current_keyboard == Input::Keyboard::Num3 || rl.control.current_gamepad == Input::Gamepad::RB_DPadDown)
            {
                return "RLFSMStateAMPLocomotion";
            }
            else if (rl.control.current_keyboard == Input::Keyboard::Num4)
            {
                return "RLFSMStateCMoELocomotion";
            }
            else if (rl.control.current_keyboard == Input::Keyboard::Num9 || rl.control.current_gamepad == Input::Gamepad::B)
            {
                return "RLFSMStateGetDown";
            }
        }
        return state_name_;
    }
};

class RLFSMStateGetDown : public RLFSMState
{
public:
    RLFSMStateGetDown(RL *rl) : RLFSMState(*rl, "RLFSMStateGetDown") {}

    float percent_getdown = 0.0f;

    void Enter() override
    {
        percent_getdown = 0.0f;
        rl.now_state = *fsm_state;
    }

    void Run() override
    {
        Interpolate(percent_getdown, rl.now_state.motor_state.q, rl.start_state.motor_state.q, 2.0f, "Getting down", true);
    }

    void Exit() override {}

    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::P || rl.control.current_gamepad == Input::Gamepad::LB_X || percent_getdown >= 1.0f)
        {
            return "RLFSMStatePassive";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num0 || rl.control.current_gamepad == Input::Gamepad::A)
        {
            return "RLFSMStateGetUp";
        }
        return state_name_;
    }
};

// PPO policy, added the same way as the other policies: one key command plus one
// complete state class. Key 1 (gamepad RB+DPadUp) loads policy/imgo2/ppo (45-input
// base_move actor: ang_vel/gravity_vec/commands/dof_pos/dof_vel/actions).
class RLFSMStatePPOLocomotion : public RLFSMState
{
public:
    RLFSMStatePPOLocomotion(RL *rl) : RLFSMState(*rl, "RLFSMStatePPOLocomotion") {}

    float percent_transition = 0.0f;

    void Enter() override
    {
        percent_transition = 0.0f;
        rl.episode_length_buf = 0;

        // read params from yaml
        rl.config_name = "ppo";
        std::string robot_config_path = rl.robot_name + "/" + rl.config_name;
        try
        {
            rl.InitRL(robot_config_path);
            rl.now_state = *fsm_state;
        }
        catch (const std::exception& e)
        {
            std::cout << LOGGER::ERROR << "InitRL() failed: " << e.what() << std::endl;
            rl.rl_init_done = false;
            rl.fsm.RequestStateChange("RLFSMStatePassive");
        }
    }

    void Run() override
    {
        if (!rl.rl_init_done) rl.rl_init_done = true;

        std::cout << "\r\033[K" << std::flush << LOGGER::INFO << "RL Controller [" << rl.config_name << "] x:" << rl.control.x << " y:" << rl.control.y << " yaw:" << rl.control.yaw << std::flush;
        RLControl();
    }

    void Exit() override
    {
        rl.rl_init_done = false;
    }

    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::P || rl.control.current_gamepad == Input::Gamepad::LB_X)
        {
            return "RLFSMStatePassive";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num9 || rl.control.current_gamepad == Input::Gamepad::B)
        {
            return "RLFSMStateGetDown";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num0 || rl.control.current_gamepad == Input::Gamepad::A)
        {
            return "RLFSMStateGetUp";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num1 || rl.control.current_gamepad == Input::Gamepad::RB_DPadUp)
        {
            return "RLFSMStatePPOLocomotion";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num2 || rl.control.current_gamepad == Input::Gamepad::RB_DPadRight)
        {
            return "RLFSMStateRLLocomotion";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num3 || rl.control.current_gamepad == Input::Gamepad::RB_DPadDown)
        {
            return "RLFSMStateAMPLocomotion";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num4)
        {
            return "RLFSMStateCMoELocomotion";
        }
        return state_name_;
    }
};

// himloco policy (当前是 Go2 参考占位策略): key 2 (gamepad RB+DPadRight) loads policy/imgo2/himloco.
class RLFSMStateRLLocomotion : public RLFSMState
{
public:
    RLFSMStateRLLocomotion(RL *rl) : RLFSMState(*rl, "RLFSMStateRLLocomotion") {}

    float percent_transition = 0.0f;

    void Enter() override
    {
        percent_transition = 0.0f;
        rl.episode_length_buf = 0;

        // read params from yaml
        rl.config_name = "himloco";
        std::string robot_config_path = rl.robot_name + "/" + rl.config_name;
        try
        {
            rl.InitRL(robot_config_path);
            rl.now_state = *fsm_state;
        }
        catch (const std::exception& e)
        {
            std::cout << LOGGER::ERROR << "InitRL() failed: " << e.what() << std::endl;
            rl.rl_init_done = false;
            rl.fsm.RequestStateChange("RLFSMStatePassive");
        }
    }

    void Run() override
    {
        // position transition from last default_dof_pos to current default_dof_pos
        // if (Interpolate(percent_transition, rl.now_state.motor_state.q, rl.params.Get<std::vector<float>>("default_dof_pos"), 0.5f, "Policy transition", true)) return;

        if (!rl.rl_init_done) rl.rl_init_done = true;

        std::cout << "\r\033[K" << std::flush << LOGGER::INFO << "RL Controller [" << rl.config_name << "] x:" << rl.control.x << " y:" << rl.control.y << " yaw:" << rl.control.yaw << std::flush;
        RLControl();
    }

    void Exit() override
    {
        rl.rl_init_done = false;
    }

    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::P || rl.control.current_gamepad == Input::Gamepad::LB_X)
        {
            return "RLFSMStatePassive";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num9 || rl.control.current_gamepad == Input::Gamepad::B)
        {
            return "RLFSMStateGetDown";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num0 || rl.control.current_gamepad == Input::Gamepad::A)
        {
            return "RLFSMStateGetUp";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num1 || rl.control.current_gamepad == Input::Gamepad::RB_DPadUp)
        {
            return "RLFSMStatePPOLocomotion";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num2 || rl.control.current_gamepad == Input::Gamepad::RB_DPadRight)
        {
            return "RLFSMStateRLLocomotion";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num3 || rl.control.current_gamepad == Input::Gamepad::RB_DPadDown)
        {
            return "RLFSMStateAMPLocomotion";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num4)
        {
            return "RLFSMStateCMoELocomotion";
        }
        return state_name_;
    }
};

// AMP policy: key 3 (gamepad RB+DPadDown) loads policy/imgo2/amp.
class RLFSMStateAMPLocomotion : public RLFSMState
{
public:
    RLFSMStateAMPLocomotion(RL *rl) : RLFSMState(*rl, "RLFSMStateAMPLocomotion") {}

    float percent_transition = 0.0f;

    void Enter() override
    {
        percent_transition = 0.0f;
        rl.episode_length_buf = 0;

        // read params from yaml
        rl.config_name = "amp";
        std::string robot_config_path = rl.robot_name + "/" + rl.config_name;
        try
        {
            rl.InitRL(robot_config_path);
            rl.now_state = *fsm_state;
        }
        catch (const std::exception& e)
        {
            std::cout << LOGGER::ERROR << "InitRL() failed: " << e.what() << std::endl;
            rl.rl_init_done = false;
            rl.fsm.RequestStateChange("RLFSMStatePassive");
        }
    }

    void Run() override
    {
        if (!rl.rl_init_done) rl.rl_init_done = true;

        std::cout << "\r\033[K" << std::flush << LOGGER::INFO << "RL Controller [" << rl.config_name << "] x:" << rl.control.x << " y:" << rl.control.y << " yaw:" << rl.control.yaw << std::flush;
        RLControl();
    }

    void Exit() override
    {
        rl.rl_init_done = false;
    }

    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::P || rl.control.current_gamepad == Input::Gamepad::LB_X)
        {
            return "RLFSMStatePassive";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num9 || rl.control.current_gamepad == Input::Gamepad::B)
        {
            return "RLFSMStateGetDown";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num0 || rl.control.current_gamepad == Input::Gamepad::A)
        {
            return "RLFSMStateGetUp";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num1 || rl.control.current_gamepad == Input::Gamepad::RB_DPadUp)
        {
            return "RLFSMStatePPOLocomotion";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num2 || rl.control.current_gamepad == Input::Gamepad::RB_DPadRight)
        {
            return "RLFSMStateRLLocomotion";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num3 || rl.control.current_gamepad == Input::Gamepad::RB_DPadDown)
        {
            return "RLFSMStateAMPLocomotion";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num4)
        {
            return "RLFSMStateCMoELocomotion";
        }
        return state_name_;
    }
};

// CMoE needs a terrain ray grid supplied by the MuJoCo backend.
class RLFSMStateCMoELocomotion : public RLFSMState
{
public:
    RLFSMStateCMoELocomotion(RL *rl) : RLFSMState(*rl, "RLFSMStateCMoELocomotion") {}

    void Enter() override
    {
        rl.episode_length_buf = 0;
        if (!rl.SupportsCMoE())
        {
            std::cout << LOGGER::ERROR << "CMoE terrain scan is only available in MuJoCo" << std::endl;
            rl.fsm.RequestStateChange("RLFSMStatePassive");
            return;
        }
        rl.config_name = "cmoe";
        try
        {
            const std::string policy_dir = std::string(POLICY_DIR) + "/" + rl.robot_name + "/cmoe/";
            if (!std::ifstream(policy_dir + "config.yaml") || !std::ifstream(policy_dir + "policy.pt"))
                throw std::runtime_error("CMoE requires config.yaml and a play.py-exported policy.pt in " + policy_dir);
            rl.InitRL(rl.robot_name + "/cmoe");
            rl.ResetCMoEState();
            rl.now_state = *fsm_state;
        }
        catch (const std::exception& e)
        {
            std::cout << LOGGER::ERROR << "CMoE InitRL() failed: " << e.what() << std::endl;
            rl.rl_init_done = false;
            rl.fsm.RequestStateChange("RLFSMStatePassive");
        }
    }

    void Run() override
    {
        if (!rl.SupportsCMoE() || !rl.model) return;
        if (!rl.rl_init_done) rl.rl_init_done = true;
        std::cout << "\r\033[K" << std::flush << LOGGER::INFO << "RL Controller [cmoe] x:"
                  << rl.control.x << " y:" << rl.control.y << " yaw:" << rl.control.yaw << std::flush;
        RLControl();
    }

    void Exit() override { rl.rl_init_done = false; }

    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::P || rl.control.current_gamepad == Input::Gamepad::LB_X)
            return "RLFSMStatePassive";
        if (rl.control.current_keyboard == Input::Keyboard::Num9 || rl.control.current_gamepad == Input::Gamepad::B)
            return "RLFSMStateGetDown";
        if (rl.control.current_keyboard == Input::Keyboard::Num0 || rl.control.current_gamepad == Input::Gamepad::A)
            return "RLFSMStateGetUp";
        if (rl.control.current_keyboard == Input::Keyboard::Num1 || rl.control.current_gamepad == Input::Gamepad::RB_DPadUp)
            return "RLFSMStatePPOLocomotion";
        if (rl.control.current_keyboard == Input::Keyboard::Num2 || rl.control.current_gamepad == Input::Gamepad::RB_DPadRight)
            return "RLFSMStateRLLocomotion";
        if (rl.control.current_keyboard == Input::Keyboard::Num3 || rl.control.current_gamepad == Input::Gamepad::RB_DPadDown)
            return "RLFSMStateAMPLocomotion";
        return state_name_;
    }
};

} // namespace imgo2_fsm

class Imgo2FSMFactory : public FSMFactory
{
public:
    Imgo2FSMFactory(const std::string& initial) : initial_state_(initial) {}
    std::shared_ptr<FSMState> CreateState(void *context, const std::string &state_name) override
    {
        RL *rl = static_cast<RL *>(context);
        if (state_name == "RLFSMStatePassive")
            return std::make_shared<imgo2_fsm::RLFSMStatePassive>(rl);
        else if (state_name == "RLFSMStateGetUp")
            return std::make_shared<imgo2_fsm::RLFSMStateGetUp>(rl);
        else if (state_name == "RLFSMStateGetDown")
            return std::make_shared<imgo2_fsm::RLFSMStateGetDown>(rl);
        else if (state_name == "RLFSMStateRLLocomotion")
            return std::make_shared<imgo2_fsm::RLFSMStateRLLocomotion>(rl);
        else if (state_name == "RLFSMStatePPOLocomotion")
            return std::make_shared<imgo2_fsm::RLFSMStatePPOLocomotion>(rl);
        else if (state_name == "RLFSMStateAMPLocomotion")
            return std::make_shared<imgo2_fsm::RLFSMStateAMPLocomotion>(rl);
        else if (state_name == "RLFSMStateCMoELocomotion")
            return std::make_shared<imgo2_fsm::RLFSMStateCMoELocomotion>(rl);
        return nullptr;
    }
    std::string GetType() const override { return "imgo2"; }
    std::vector<std::string> GetSupportedStates() const override
    {
        return {
            "RLFSMStatePassive",
            "RLFSMStateGetUp",
            "RLFSMStateGetDown",
            "RLFSMStateRLLocomotion",
            "RLFSMStatePPOLocomotion",
            "RLFSMStateAMPLocomotion",
            "RLFSMStateCMoELocomotion"
        };
    }
    std::string GetInitialState() const override { return initial_state_; }
private:
    std::string initial_state_;
};

REGISTER_FSM_FACTORY(Imgo2FSMFactory, "RLFSMStatePassive")

#endif // IMGO2_FSM_HPP
