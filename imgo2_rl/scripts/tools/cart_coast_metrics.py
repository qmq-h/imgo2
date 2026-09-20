"""Offline coast metrics; no Isaac Lab, torch, numpy or plotting dependency."""

import math

LEGS = ("fl", "fr", "rl", "rr")


def summarize_sweep(results):
    """Report engineering candidates at 1 m/s; do not infer real caster values."""
    groups = []
    for velocity in sorted({r["velocity_mps"] for r in results}):
        cases = sorted((r for r in results if r["velocity_mps"] == velocity), key=lambda r: r["damping"])
        stopped = [r for r in cases if r["valid"] and r["stopped"]]
        pairs = [(a, b) for a, b in zip(stopped, stopped[1:]) if b["damping"] > a["damping"]]
        monotone = all(b["stop_distance_m"] <= a["stop_distance_m"] + 0.01 for a, b in pairs)
        groups.append({"velocity_mps": velocity,
                       "valid_cases": sum(r["valid"] for r in cases),
                       "total_cases": len(cases),
                       "stopped_cases": len(stopped),
                       "distance_decreases_with_damping": monotone if pairs else None})
    candidates = sorted({r["damping"] for r in results
                         if math.isclose(r["velocity_mps"], 1.0) and r["valid"] and r["stopped"]
                         and r["damping"] > 0 and 0.5 <= r["stop_distance_m"] <= 2.0})
    return {"engineering_only": True, "target_velocity_mps": 1.0, "target_distance_m": [0.5, 2.0],
            "candidate_damping_nms_per_rad": candidates, "groups": groups,
            "note": "Candidates require simulator inspection and dt sensitivity checks; not measured caster parameters."}


def summarize(rows, *, radius, resting_height, mode="coast", stop_speed=0.02, stop_hold=0.5,
              max_lateral=0.1, max_tilt=0.4, max_yaw=0.3, min_contact=0.1):
    """Measure the final sustained low-speed interval; never equate timeout to stop.

    Torques in each row are the torques applied during the interval ending at
    that sample. Dissipation is therefore evaluated against the previous omega.
    """
    if mode not in ("coast", "drop"):
        raise ValueError("Unknown experiment mode")
    if not all(math.isfinite(v) and v > 0 for v in (radius, resting_height, stop_speed, stop_hold)):
        raise ValueError("Invalid geometry or stop thresholds")
    if len(rows) < 2:
        raise ValueError("At least two samples are required")
    required = ["time_s", "x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps",
                "roll_rad", "pitch_rad", "yaw_rad", "wx_radps", "wy_radps", "wz_radps",
                *(f"{leg}_{suffix}" for leg in LEGS for suffix in ("omega_radps", "tau_nm", "normal_n"))]
    for i, row in enumerate(rows):
        if any(k not in row or not math.isfinite(row[k]) for k in required):
            raise ValueError(f"Invalid sample {i}")
        if row["time_s"] < 0 or (i and row["time_s"] <= rows[i-1]["time_s"]):
            raise ValueError("Sample times must be nonnegative and strictly increasing")
    speed = [math.sqrt(sum(r[k]**2 for k in ("vx_mps", "vy_mps", "vz_mps"))) for r in rows]
    candidate = None
    for i, value in enumerate(speed):
        if value >= stop_speed:
            candidate = None
        elif candidate is None:
            candidate = i
    stopped = candidate is not None and rows[-1]["time_s"] - rows[candidate]["time_s"] >= stop_hold - 1e-9
    stop_row = rows[candidate] if stopped else None
    x0, y0 = rows[0]["x_m"], rows[0]["y_m"]
    max_dy = max(abs(r["y_m"] - y0) for r in rows)
    tilt = max(max(abs(r["roll_rad"]), abs(r["pitch_rad"])) for r in rows)
    yaw = max(abs(math.atan2(math.sin(r["yaw_rad"]), math.cos(r["yaw_rad"]))) for r in rows)
    final_contact = all(rows[-1][f"{leg}_normal_n"] > min_contact for leg in LEGS)
    contact_fraction = {leg: sum(r[f"{leg}_normal_n"] > min_contact for r in rows) / len(rows) for leg in LEGS}
    rolling_residual = math.sqrt(sum(
        (r["vx_mps"] * math.cos(r["yaw_rad"]) + r["vy_mps"] * math.sin(r["yaw_rad"])
         - radius * r[f"{leg}_omega_radps"])**2 for r in rows for leg in LEGS) / (4 * len(rows)))
    positive_work = sum(max(0.0, rows[i][f"{leg}_tau_nm"] * rows[i-1][f"{leg}_omega_radps"])
                        * (rows[i]["time_s"] - rows[i-1]["time_s"])
                        for i in range(1, len(rows)) for leg in LEGS)
    failures = []
    if max_dy > max_lateral:
        failures.append("lateral_drift")
    if tilt > max_tilt or yaw > max_yaw:
        failures.append("orientation_out_of_bounds")
    if not final_contact:
        failures.append("missing_final_wheel_contact")
    if abs(rows[-1]["z_m"] - resting_height) > 0.025:
        failures.append("incorrect_resting_height")
    if positive_work > 1e-8:
        failures.append("resistance_injects_energy")
    if mode == "coast":
        if rolling_residual > 0.15:
            failures.append("excessive_rolling_slip")
        if min(contact_fraction.values()) < 0.95:
            failures.append("intermittent_wheel_contact")
    elif speed[-1] > 0.05 or max(abs(rows[-1][k]) for k in ("wx_radps", "wy_radps", "wz_radps")) > 0.1:
        failures.append("drop_not_settled")
    return {
        "mode": mode, "valid": not failures, "failures": failures, "samples": len(rows),
        "stopped": bool(stopped), "stop_time_s": stop_row["time_s"] if stopped else None,
        "stop_distance_m": stop_row["x_m"] - x0 if stopped else None,
        "observed_duration_s": rows[-1]["time_s"] - rows[0]["time_s"],
        "observed_distance_m": rows[-1]["x_m"] - x0,
        "initial_speed_mps": speed[0], "final_speed_mps": speed[-1],
        "max_lateral_displacement_m": max_dy, "max_tilt_rad": tilt, "max_abs_yaw_rad": yaw,
        "rolling_residual_rms_mps": rolling_residual, "contact_fraction": contact_fraction,
        "positive_resistance_work_j": positive_work,
        "final_max_wheel_speed_radps": max(abs(rows[-1][f"{leg}_omega_radps"]) for leg in LEGS),
        "stop_speed_mps": stop_speed, "stop_hold_s": stop_hold,
    }


def svg_plot(rows):
    """Two independent axes: time vs velocity and longitudinal displacement."""
    width, left, plot_width = 900, 70, 790
    t0, span = rows[0]["time_s"], rows[-1]["time_s"] - rows[0]["time_s"]
    if span <= 0:
        raise ValueError("Plot requires a positive time span")
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="520" viewBox="0 0 900 520">',
             '<rect width="900" height="520" fill="white"/>',
             '<g font-family="sans-serif" font-size="14" fill="#243447">']
    for top, label, vals, color in (
        (40, "Longitudinal velocity (m/s)", [r["vx_mps"] for r in rows], "#1664aa"),
        (290, "Longitudinal displacement (m)", [r["x_m"] - rows[0]["x_m"] for r in rows], "#c05b11"),
    ):
        low, high = min(0.0, min(vals)), max(0.001, max(vals))
        height = 160
        parts.append(f'<text x="{left}" y="{top-12}">{label}</text>')
        for i in range(5):
            y, value = top + height * i / 4, high - (high-low) * i / 4
            parts.append(f'<path d="M{left},{y} H860" stroke="#dce1e6"/><text x="8" y="{y+5}">{value:.3g}</text>')
        # Decimate only the rendering; metrics always use every sample.
        indices = list(range(0, len(rows), max(1, len(rows)//1800)))
        if indices[-1] != len(rows)-1:
            indices.append(len(rows)-1)
        points = " ".join(f'{left+(rows[i]["time_s"]-t0)/span*plot_width:.2f},{top+(high-vals[i])/(high-low)*height:.2f}' for i in indices)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        for i in range(6):
            parts.append(f'<text x="{left+plot_width*i/5}" y="{top+height+23}" text-anchor="middle">{span*i/5:.2g}</text>')
        parts.append(f'<text x="435" y="{top+height+46}">Time since first sample (s)</text>')
    return "\n".join(parts + ["</g></svg>\n"])
