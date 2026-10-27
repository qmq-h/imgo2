"""Shared deterministic commands for headless PPO maneuver diagnostics (SI units)."""

# duration, label, (vx_start, yaw_start), (vx_end, yaw_end)
SEGMENTS = (
    (6., "initial_stand", (0., 0.), (0., 0.)),
    (3., "low_accel", (0., 0.), (.1, 0.)),
    (6., "low_baseline", (.1, 0.), (.1, 0.)),
    (3., "high_accel", (.1, 0.), (.8, 0.)),
    (6., "high_before_fast", (.8, 0.), (.8, 0.)),
    (.2, "fast_decel", (.8, 0.), (.1, 0.)),
    (6., "after_fast_decel", (.1, 0.), (.1, 0.)),
    (3., "high_accel_again", (.1, 0.), (.8, 0.)),
    (6., "high_before_slow", (.8, 0.), (.8, 0.)),
    (3., "slow_decel", (.8, 0.), (.1, 0.)),
    (6., "after_slow_decel", (.1, 0.), (.1, 0.)),
    (1., "stop_straight", (.1, 0.), (0., 0.)),
    (5., "stand_before_turn", (0., 0.), (0., 0.)),
    (1., "left_accel", (0., 0.), (0., 1.)),
    (8., "left_turn", (0., 1.), (0., 1.)),
    (1., "left_stop", (0., 1.), (0., 0.)),
    (5., "stand_after_left", (0., 0.), (0., 0.)),
    (1., "right_accel", (0., 0.), (0., -1.)),
    (8., "right_turn", (0., -1.), (0., -1.)),
    (1., "right_stop", (0., -1.), (0., 0.)),
    (5., "stand_after_right", (0., 0.), (0., 0.)),
    (1., "arc_accel", (0., 0.), (.4, .8)),
    (8., "arc_turn", (.4, .8), (.4, .8)),
    (1., "arc_stop", (.4, .8), (0., 0.)),
    (6., "final_stand", (0., 0.), (0., 0.)),
)
TOTAL_SECONDS = sum(s[0] for s in SEGMENTS)


def command_at(t, low_speed=.1):
    if t < 0:
        return 0., 0., "getup"
    for duration, label, start, end in SEGMENTS:
        if t < duration:
            start = (low_speed if start[0] == .1 else start[0], start[1])
            end = (low_speed if end[0] == .1 else end[0], end[1])
            u = max(0., t / duration)
            return (start[0] + u * (end[0] - start[0]),
                    start[1] + u * (end[1] - start[1]), label)
        t -= duration
    return 0., 0., "final_stand"


def schedule(low_speed=.1):
    result = []
    t = 0.
    for duration, label, start, end in SEGMENTS:
        result.append(dict(stage=label, start_s=t, end_s=t+duration,
                           vx_start=low_speed if start[0] == .1 else start[0],
                           vx_end=low_speed if end[0] == .1 else end[0],
                           yaw_start=start[1], yaw_end=end[1]))
        t += duration
    return result
