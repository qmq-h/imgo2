from __future__ import annotations

from typing import Any

import torch

from isaaclab.envs import ManagerBasedRLEnv


class AmpManagerBasedRLEnv(ManagerBasedRLEnv):
    """Manager-based Isaac Lab environment with AMP terminal observations.

    The AMP runner needs two additions beyond the standard manager-based RL
    step output:
    - current AMP observations through :meth:`get_amp_observations`
    - AMP observations captured before reset for terminated environments

    AMP observations are expected to be defined as an ``amp`` observation group
    in the environment configuration.
    """

    def __init__(self, cfg, render_mode: str | None = None, **kwargs) -> None:
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        if "amp" not in self.observation_manager.group_obs_dim:
            raise ValueError("AmpManagerBasedRLEnv requires an 'amp' observation group.")

    def step(
        self, action: torch.Tensor
    ) -> tuple[
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
        torch.Tensor,
        torch.Tensor,
    ]:
        self.action_manager.process_action(action.to(self.device))
        self.recorder_manager.record_pre_step()

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        terminal_amp_states = self.get_amp_observations()[reset_env_ids].clone()

        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            self.scene.write_data_to_sim()
            self.sim.forward()
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()
            self.recorder_manager.record_post_reset(reset_env_ids)

        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.obs_buf = self.observation_manager.compute(update_history=True)

        return (
            self.obs_buf,
            self.reward_buf,
            self.reset_terminated,
            self.reset_time_outs,
            self.extras,
            reset_env_ids,
            terminal_amp_states,
        )

    def get_amp_observations(self) -> torch.Tensor:
        """Return AMP observations for all environments."""
        amp_obs = self.observation_manager.compute_group("amp")
        if isinstance(amp_obs, dict):
            amp_obs = torch.cat(tuple(amp_obs.values()), dim=-1)
        return amp_obs

    @property
    def dof_pos_limits(self) -> torch.Tensor:
        asset = self.scene["robot"]
        if hasattr(asset.data, "soft_joint_pos_limits"):
            return asset.data.soft_joint_pos_limits[0]
        return asset.data.joint_pos_limits[0]

    @property
    def dt(self) -> float:
        return self.step_dt
