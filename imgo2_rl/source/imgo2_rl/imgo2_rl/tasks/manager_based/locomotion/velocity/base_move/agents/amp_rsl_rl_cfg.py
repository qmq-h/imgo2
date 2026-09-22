from isaaclab.utils import configclass

from imgo2_rl.assets.imgo2 import AMP_MOTION_FILES
from rl_lab.config import AMPActorCriticCfg, AMPAlgorithmCfg, AMPOnPolicyRunnerCfg


def _policy_cfg():
    return AMPActorCriticCfg(
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
        fixed_std=False,
    )


def _algorithm_cfg(*, clamp_noise_std):
    return AMPAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        amp_replay_buffer_size=1000000,
        clamp_noise_std=clamp_noise_std,
    )


@configclass
class AMPHeightRunnerCfg(AMPOnPolicyRunnerCfg):
    """Flat-terrain AMP with the Imgo2 height-stabilized task reward."""

    num_steps_per_env = 24
    max_iterations = 40000
    save_interval = 500
    experiment_name = "base_move_amp_height"
    include_history_steps = None
    amp_motion_files = AMP_MOTION_FILES
    amp_num_preload_transitions = 2000000
    amp_reward_coef = 2.0
    amp_discr_hidden_dims = [1024, 512]
    amp_task_reward_lerp = 0.3
    min_normalized_std = [0.05, 0.02, 0.05] * 4
    policy = _policy_cfg()
    algorithm = _algorithm_cfg(clamp_noise_std=False)


@configclass
class FanziqiAMPRunnerCfg(AMPOnPolicyRunnerCfg):
    """Runner values from Fanziqi's A1 AMP configuration."""

    num_steps_per_env = 24
    max_iterations = 500000
    save_interval = 50
    experiment_name = "base_move_amp_fanziqi"
    include_history_steps = None
    amp_motion_files = AMP_MOTION_FILES
    amp_num_preload_transitions = 2000000
    amp_reward_coef = 2.0
    amp_discr_hidden_dims = [1024, 512]
    amp_task_reward_lerp = 0.3
    min_normalized_std = [0.05, 0.02, 0.05] * 4
    policy = _policy_cfg()
    algorithm = _algorithm_cfg(clamp_noise_std=True)
