"""Goal and deadline contract, independently checkable without the simulator."""

import math

SPEED_RANGE = (0.4, 1.5)
GOAL_DISTANCE_M = 10.0
SETTLE_TIME_S = 1.0
TIMEOUT_MARGIN_S = 2.0


def episode_timeout_s(distance, minimum_speed, settle_time=SETTLE_TIME_S,
                      margin=TIMEOUT_MARGIN_S):
    if not all(math.isfinite(x) for x in (distance, minimum_speed, settle_time, margin)):
        raise ValueError("episode parameters must be finite")
    if distance <= 0 or minimum_speed <= 0 or settle_time < 0 or margin <= 0:
        raise ValueError("positive distance, speed and timeout margin required")
    return settle_time + distance / minimum_speed + margin
