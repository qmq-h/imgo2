"""Goal and deadline contract, independently checkable without the simulator.

2026-10-09（同日二次修改）：回合结构从「走到目标就结束」改回**三段制**
（settle → tow → STOP/滑行），因为两条 post_stop 奖励（卸力、说停就停）需要
"停车之后"这个真实相位才有依据：

    0 ── settle(指令 0，静止稳定) ── tow(指令 = tow_speed) ── STOP(指令 0) ── timeout 结束
        SETTLE_TIME_S                                    STOP_DISTANCE_M

`STOP_DISTANCE_M` 是**沿 lane 的水平前进距离**（从出生点起算），到达它就把指令置零；
它必须**大于坡面出口** `slope_geometry.FLAT_OUT_START_M = 9.0 m`——用户要求
「给 cmd vel 一定要在越过坡之后」，`upper_env_cfg.__post_init__` 里有显式断言。
触发用**进度**而不是时间：最慢速度（0.4 m/s）下光是走到坡出口就要约 23 s，若用固定
时间阈值，慢速环境会在坡上就被叫停，无法保证"越过坡"。

`POST_STOP_WINDOW_S` 是 STOP 之后的评分窗口，作为 `episode_timeout_s` 的 margin：
timeout = settle + 到 STOP 点的坡面弧长 / 最小速度 + 窗口，所以**最慢速度下**
STOP 之后仍有约 `POST_STOP_WINDOW_S` 秒的步供 post_stop 奖励作用（速度越快窗口越长）。
"""

import math

SPEED_RANGE = (0.4, 1.5)
# 指令归零的沿 lane 水平距离（m）。必须 > FLAT_OUT_START_M（坡面出口 9.0 m）。
STOP_DISTANCE_M = 10.0
SETTLE_TIME_S = 1.0
# STOP 之后至少要留下的评分窗口（s），用作 timeout 的 margin。
POST_STOP_WINDOW_S = 3.0
# 其余调用方（不显式传 margin 时）沿用的默认余量。
TIMEOUT_MARGIN_S = 2.0


def episode_timeout_s(distance, minimum_speed, settle_time=SETTLE_TIME_S,
                      margin=TIMEOUT_MARGIN_S):
    if not all(math.isfinite(x) for x in (distance, minimum_speed, settle_time, margin)):
        raise ValueError("episode parameters must be finite")
    if distance <= 0 or minimum_speed <= 0 or settle_time < 0 or margin <= 0:
        raise ValueError("positive distance, speed and timeout margin required")
    return settle_time + distance / minimum_speed + margin
