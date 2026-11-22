#!/usr/bin/env bash
# 5300平地低速分支；参数依据与限制见 docs/training/low5300_retrain_2026-11-10.md。
# 默认仅在用户执行本脚本时续训；--dry-run只显示命令，--baseline使用原足高/腾空参数。
set -euo pipefail

IMGO2_RL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
profile_baseline=false
profile_dry_run=false
forward_args=()
for arg in "$@"; do
    case "${arg}" in
        --baseline) profile_baseline=true ;;
        --dry-run) profile_dry_run=true ;;
        *) forward_args+=("${arg}") ;;
    esac
done

# 这两项是相对5300配方的实验变量；机身坐标足高不是离地净空。
foot_target=-0.25
airtime_threshold=0.20
run_suffix=low5300_h25_air20
if "${profile_baseline}"; then
    foot_target=-0.20
    airtime_threshold=0.50
    run_suffix=low5300_control
fi

checkpoint="${IMGO2_RL_ROOT}/logs/rsl_rl/base_move_rough/2026-11-05_15-19-20/model_5300.pt"
if [[ ! -f "${checkpoint}" ]]; then
    echo "缺少原始5300 checkpoint：${checkpoint}" >&2
    exit 1
fi
checkpoint_hash="$(sha256sum -- "${checkpoint}")"
if [[ "${checkpoint_hash%% *}" != e13653e7984dee6d01b0e2f7472879c06066e5a1c8f2b5fc31300affd46b92d9 ]]; then
    echo '5300 checkpoint内容与已核对来源不同，请先确认来源。' >&2
    exit 1
fi

# 保留rough任务的45维actor/235维critic，只把场景切回平地；不改共享rough配置。
command_args=(
    bash "${IMGO2_RL_ROOT}/scripts/run_isaaclab.sh"
    "${IMGO2_RL_ROOT}/scripts/rl_lab/ppo/train.py"
    --task Imgo2-basemove-rough-ppo --headless --seed 42
    --resume --warm_start --load_run '^2026-11-05_15-19-20$' --checkpoint '^model_5300\.pt$'
    --run_name "${run_suffix}" --num_envs 4096 --max_iterations 1000
    env.scene.terrain.terrain_type=plane env.scene.terrain.terrain_generator=null
    env.curriculum.terrain_levels=null
    env.curriculum.command_levels_lin_vel=null env.curriculum.command_levels_ang_vel=null
    env.commands.base_velocity.heading_command=false
    'env.commands.base_velocity.ranges.lin_vel_x=[-0.5,0.5]'
    'env.commands.base_velocity.ranges.lin_vel_y=[-0.4,0.4]'
    'env.commands.base_velocity.ranges.ang_vel_z=[-0.6,0.6]'
    env.rewards.lin_vel_z_l2.weight=-4.0
    env.rewards.ang_vel_xy_l2.weight=-0.3
    env.rewards.flat_orientation_l2.weight=-5.0
    env.rewards.base_height_l2.weight=-35.0
    env.rewards.base_height_l2.params.target_height=0.33
    env.rewards.joint_deviation_l1.weight=-0.5
    env.rewards.stand_still.weight=-0.2
    env.rewards.joint_pos_penalty=null
    env.rewards.joint_mirror.weight=-1.0
    env.rewards.contact_forces.weight=-0.02
    env.rewards.track_lin_vel_xy_exp.weight=1.5
    env.rewards.track_ang_vel_z_exp.weight=1.2
    env.rewards.feet_air_time.weight=1.6
    "env.rewards.feet_air_time.params.threshold=${airtime_threshold}"
    env.rewards.feet_air_time_variance.weight=-8.0
    env.rewards.feet_slide.weight=-0.15
    env.rewards.feet_height_body.weight=-4.0
    "env.rewards.feet_height_body.params.target_height=${foot_target}"
)
command_args+=("${forward_args[@]}")
if "${profile_dry_run}"; then
    printf '%q ' "${command_args[@]}"
    printf '\n'
    exit 0
fi

export PYTHONPATH="${IMGO2_RL_ROOT}/source/imgo2_rl:${IMGO2_RL_ROOT}/scripts/rl_lab${PYTHONPATH:+:${PYTHONPATH}}"
cd -- "${IMGO2_RL_ROOT}"
exec "${command_args[@]}"
