"""play / 导出入口共用的环境配置微调。

本文件与 ``cli_args.py`` 同目录（``imgo2_rl/scripts/rl_lab``），三个算法的 ``play.py``
都通过 ``PYTHONPATH`` 导入这里的模块，不额外安装。
"""


def disable_command_debug_visualizers(env_cfg) -> list[str]:
    """关闭指令项（``env_cfg.commands.*``）的调试可视化，返回被关闭的项名。

    为什么要关：``base_velocity.debug_vis=True``（``velocity_env_cfg.py`` 的默认值）会让
    ``UniformVelocityCommand._set_debug_vis_impl`` 在**建场景时**就 spawn 速度指令箭头，
    而 ``goal_vel_visualizer_cfg`` / ``current_vel_visualizer_cfg`` 的 ``usd_path`` 是远端资产
    ``.../Assets/Isaac/5.1/Isaac/Props/UIElements/arrow_x.usd``。离线、DNS 不通或代理坏掉时
    ``omni.client.stat`` 判定不可用，直接抛::

        FileNotFoundError: USD file not found at path at: 'https://.../arrow_x.usd'

    建场景随之中断——注意这时连 ``exported/`` 都不会生成，看起来像「导出不了」，其实
    checkpoint 还没被加载（2026-10-28 本机实测，见 docs/deployment/ppo_5499_export_2026-10-28.md）。

    headless 下没有视口，这些 marker 本来也看不到；导出更不需要它们。GUI 回放不受影响，
    因为本函数只在 headless / export_only 时才被调用。marker 只影响画面，不改动力学、不改
    观测，所以关掉不影响导出策略的数值。

    防守式写法：``env_cfg`` 没有 ``commands``（例如纯 AMP 配置）或某个项没有 ``debug_vis``
    （例如 AMP 的 ``amp`` 风格项）时跳过，不产生副作用。
    """
    disabled: list[str] = []
    commands = getattr(env_cfg, "commands", None)
    if commands is None:
        return disabled
    for name in dir(commands):
        if name.startswith("_"):
            continue
        term_cfg = getattr(commands, name, None)
        if term_cfg is None or not hasattr(term_cfg, "debug_vis"):
            continue
        if term_cfg.debug_vis:
            term_cfg.debug_vis = False
            disabled.append(name)
    return disabled
