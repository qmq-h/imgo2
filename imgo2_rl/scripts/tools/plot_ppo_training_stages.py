#!/usr/bin/env python3
"""Render saved three-run/four-checkpoint PPO comparisons (optional matplotlib)."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    report = json.loads(args.report.read_text())
    policies = [("stage1_4999.pt", "#277da8", "Initial 4999"),
                ("stage2_5498.pt", "#e9a23b", "Resume 5498"),
                ("stage3_6900.pt", "#43aa8b", "Resume 6900"),
                ("stage3_7497.pt", "#d95f62", "Resume 7497")]
    lookup = {(r["policy"], r["case"]): r for r in report["benchmark"]["results"]}
    cases = ["forward03", "forward05", "backward03", "left03", "right03"]
    labels = ["Fwd .3", "Fwd .5", "Back -.3", "Left .3", "Right -.3"]
    definitions = [("vx_mean", 1., "Actual forward velocity (m/s)"),
                   ("height_mean_m", 100., "Base height (cm)"),
                   ("stride_frequency_hz", 1., "Same-foot stride frequency (Hz)"),
                   ("swing_clearance_peak_mean_m", 100., "Mean swing peak clearance (cm)"),
                   ("gyro_xy_rms_rads", 1., "Roll/pitch angular-speed RMS (rad/s)")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    x = np.arange(len(cases))
    for ax, (key, scale, title) in zip(axes.flat, definitions):
        for index, (policy, color, label) in enumerate(policies):
            ax.bar(x + (index - 1.5) * .19,
                   [lookup[policy, c][key] * scale for c in cases], .19, color=color, label=label)
        ax.set_title(title)
        ax.set_xticks(x, labels, rotation=15)
        ax.grid(axis="y", alpha=.2)
    ax = axes[1, 2]
    for index, (policy, color, label) in enumerate(policies):
        ax.bar(np.arange(2) + (index - 1.5) * .19,
               [lookup[policy, c]["wz_mean"] for c in ("turn_left03", "turn_right03")],
               .19, color=color, label=label)
    ax.scatter([0, 1], [.3, -.3], marker="x", color="black", label="Command")
    ax.set_title("Yaw tracking (rad/s)")
    ax.set_xticks([0, 1], ["Left turn", "Right turn"])
    ax.grid(axis="y", alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Three training runs, four saved actors: identical MuJoCo conditions", fontsize=14)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(str(args.output) + "." + suffix, dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    tags = [("Train/mean_reward", "Episode return (weights differ between runs)"),
            ("Metrics/base_velocity/error_vel_xy", "Logged XY error (project normalization)"),
            ("Metrics/base_velocity/error_vel_yaw", "Logged yaw error (project normalization)"),
            ("Loss/value_function", "Value loss"),
            ("Episode_Reward/base_height_l2", "Height contribution (target / weights differ)")]
    colors = ["#277da8", "#e9a23b", "#43aa8b"]
    for index, (_, run) in enumerate(report["runs"].items()):
        color, label = colors[index], ["Initial run", "First resume", "Second resume"][index]
        for ax, (tag, title) in zip(axes.flat, tags):
            points = []
            for window in run["windows"]:
                value = window["metrics"].get(tag)
                if value and value["n"] >= 100:
                    points.append((window["start"] + (value["n"] - 1) / 2, value["mean"]))
            ax.plot([p[0] for p in points], [p[1] for p in points], color=color, label=label)
            ax.set_title(title)
        checkpoints = run["checkpoints"]
        axes[1, 2].plot([c["iter"] for c in checkpoints], [c["mean_std"] for c in checkpoints],
                        color=color, label=label)
    axes[1, 2].set_title("Mean action std from saved checkpoints")
    last = list(report["runs"].values())[-1]["last_iteration"]
    for ax in axes.flat:
        ax.axvline(4999, color="grey", ls=":", alpha=.6)
        ax.axvline(5498, color="grey", ls=":", alpha=.6)
        ax.axvspan(last, 7497, color="grey", alpha=.10)
        ax.set_xlabel("Saved training iteration")
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"250-iteration means; scalar logs end at {last}; shaded tail has checkpoints only", fontsize=13)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(str(args.output) + "_training." + suffix, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
