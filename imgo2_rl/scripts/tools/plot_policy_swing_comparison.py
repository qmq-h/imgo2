#!/usr/bin/env python3
"""Plot saved policy comparison metrics and optional pooled swing-height profiles.

Requires matplotlib only for rendering; no simulator or model loading.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output filename prefix")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    data = json.loads(args.report.read_text())
    results = {(r["policy"], r["case"]): r for r in data["main"]["results"]}
    policies = [("policy_1.pt", "#277da8", "Hardware policy_1"),
                ("policy.pt", "#d95f02", "New policy (6900)")]
    cases = ["forward03", "forward05", "backward03", "left03", "right03"]
    labels = ["Fwd .3", "Fwd .5", "Back -.3", "Left .3", "Right -.3"]
    definitions = [("vx_mean", 1., "Body forward speed (m/s)"),
                   ("height_mean_m", 100., "Mean base height (cm)"),
                   ("stride_frequency_hz", 1., "Same-foot stride frequency (Hz)"),
                   ("swing_clearance_peak_mean_m", 100., "Mean swing peak clearance (cm)"),
                   ("gyro_xy_rms_rads", 1., "Roll/pitch angular-speed RMS (rad/s)")]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    x = np.arange(len(cases))
    for ax, (key, scale, title) in zip(axes.flat, definitions):
        for i, (policy, color, label) in enumerate(policies):
            values = [results[policy, case][key] * scale for case in cases]
            ax.bar(x + (i - .5) * .36, values, .36, color=color, label=label)
        ax.set_xticks(x, labels, rotation=20)
        ax.set_title(title)
        ax.grid(axis="y", alpha=.25)
    ax = axes[1, 2]
    turns = ["turn_left03", "turn_right03"]
    for i, (policy, color, label) in enumerate(policies):
        ax.bar(np.arange(2) + (i - .5) * .36,
               [results[policy, c]["wz_mean"] for c in turns], .36, color=color, label=label)
    ax.scatter([0, 1], [.3, -.3], marker="x", color="black", label="Command")
    ax.set_xticks([0, 1], ["Left turn", "Right turn"])
    ax.set_title("Yaw tracking (rad/s)")
    ax.grid(axis="y", alpha=.25)
    axes[0, 0].legend(fontsize=8)
    ax.legend(fontsize=8)
    fig.suptitle("Same model / PD 25, 0.5 / 20 s cases / last 16 s measured", fontsize=13)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(str(args.output) + "." + suffix, dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    cases = ["forward03", "forward05", "backward03", "turn_left03", "turn_right03"]
    titles = ["Forward .3 m/s", "Forward .5 m/s", "Backward -.3 m/s",
              "Left turn .3 rad/s", "Right turn -.3 rad/s"]
    for ax, case, title in zip(axes.flat, cases, titles):
        for policy, color, label in policies:
            profile = data["swing_profiles"][Path(policy).stem + "_" + case]
            if not profile["mean_clearance_m"]:
                continue
            phase = np.array(profile["phase"])
            mean = np.array(profile["mean_clearance_m"]) * 100
            std = np.array(profile["std_clearance_m"]) * 100
            ax.plot(phase, mean, color=color, label=f"{label} (n={profile['air_run_count']})")
            ax.fill_between(phase, mean-std, mean+std, color=color, alpha=.15)
        ax.set_title(title)
        ax.set_xlabel("Normalized completed swing (offline only)")
        ax.set_ylabel("Foot-bottom clearance (cm)")
        ax.set_ylim(bottom=-.2)
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    axes[1, 2].axis("off")
    axes[1, 2].text(0, .9, "Four feet pooled; mean +/- one std.\n"
                   "Only complete air runs >=30 ms.\n\n"
                   "Phase is reconstructed after landing.\n"
                   "It is not available online as shown.\n\n"
                   "Height profiles do not fully describe\n"
                   "3D path smoothness or hardware impact.", va="top", fontsize=11)
    fig.suptitle("Measured swing-height profiles: new policy vs hardware policy_1", fontsize=13)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(str(args.output) + "_swing." + suffix, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
