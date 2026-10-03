#!/usr/bin/env python3
"""Plot left/right wing commanded and actual angle distributions.

The default signal mapping is:
  command: PwmServoTask.input.sl_deg_in/sr_deg_in
  actual : PwmServoTask.output.left_deg/right_deg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataset_plot_utils import (
    COLORS, DEFAULT_DATASET_DIR, DEFAULT_FIGURE_DIR, add_panel_label,
    configure_style, resolve_path, save_figure, style_axis,
    save_panels,
)


SIGNALS = {
    "left_command": "PwmServoTask.input.sl_deg_in",
    "right_command": "PwmServoTask.input.sr_deg_in",
    "left_output": "PwmServoTask.output.left_deg",
    "right_output": "PwmServoTask.output.right_deg",
}


def source_files(dataset_dir: Path) -> list[tuple[Path, tuple[float, float]]]:
    paths: list[tuple[Path, tuple[float, float]]] = []
    for meta_path in sorted(dataset_dir.glob("*.meta.json")):
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        source = Path(metadata["source_csv"])
        interval = metadata.get("source_interval_us")
        if source.is_file() and interval and len(interval) == 2:
            paths.append((source, (float(interval[0]), float(interval[1]))))
    if not paths:
        raise FileNotFoundError("No readable source_csv entries found in metadata")
    return paths


def load_signal_columns(
    path: Path,
    interval_us: tuple[float, float],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    usecols = []
    for signal in SIGNALS.values():
        usecols.extend((f"{signal}_time_us", f"{signal}_value"))
    frame = pd.read_csv(path, comment="#", usecols=usecols, low_memory=False)
    result = {}
    for key, signal in SIGNALS.items():
        t = pd.to_numeric(frame[f"{signal}_time_us"], errors="coerce").to_numpy()
        v = pd.to_numeric(frame[f"{signal}_value"], errors="coerce").to_numpy()
        valid = (
            np.isfinite(t) & np.isfinite(v)
            & (t >= interval_us[0]) & (t <= interval_us[1])
        )
        result[key] = t[valid].astype(float), v[valid].astype(float)
    return result


def paired(command, output):
    command_t, command_v = command
    output_t, output_v = output
    if len(command_t) < 2 or len(output_t) < 2:
        return np.empty(0), np.empty(0)
    order = np.argsort(output_t)
    output_t, output_v = output_t[order], output_v[order]
    valid = (command_t >= output_t[0]) & (command_t <= output_t[-1])
    return command_v[valid], np.interp(command_t[valid], output_t, output_v)


def pooled_hist(ax, command, output, title, bins):
    lower = min(np.percentile(command, 0.5), np.percentile(output, 0.5))
    upper = max(np.percentile(command, 99.5), np.percentile(output, 99.5))
    edges = np.linspace(lower, upper, bins+1)
    ax.hist(command, bins=edges, density=True, histtype="stepfilled",
            color=COLORS["orange"], alpha=0.28, label="Commanded angle")
    ax.hist(command, bins=edges, density=True, histtype="step",
            color=COLORS["orange"], linewidth=1.35)
    ax.hist(output, bins=edges, density=True, histtype="stepfilled",
            color=COLORS["navy"], alpha=0.22, label="Actual angle")
    ax.hist(output, bins=edges, density=True, histtype="step",
            color=COLORS["navy"], linewidth=1.35)
    ax.set_xlabel("Servo angle [deg]")
    ax.set_ylabel("Probability density")
    ax.set_title(title)
    ax.legend(frameon=False)
    style_axis(ax)


def tracking_density(ax, command, output, title, gridsize):
    # Limit rendering to robust bounds while metrics retain all finite samples.
    lo = min(np.percentile(command, 0.5), np.percentile(output, 0.5))
    hi = max(np.percentile(command, 99.5), np.percentile(output, 99.5))
    ax.hexbin(command, output, gridsize=gridsize, mincnt=1, cmap="GnBu",
              extent=(lo, hi, lo, hi), linewidths=0)
    ax.plot([lo, hi], [lo, hi], color=COLORS["red"], linestyle="--",
            linewidth=1.2, label="Ideal tracking")
    rmse = float(np.sqrt(np.mean((output-command)**2)))
    bias = float(np.mean(output-command))
    ax.text(0.04, 0.94, f"RMSE: {rmse:.2f} deg\nBias: {bias:+.2f} deg",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.7,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=2.0))
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Commanded angle [deg]")
    ax.set_ylabel("Actual angle [deg]")
    ax.set_title(title)
    ax.legend(frameon=False, loc="lower right")
    style_axis(ax, "both")
    return rmse, bias


def render(args):
    dataset_dir = resolve_path(args.dataset_dir)
    output_dir = resolve_path(args.output_dir)
    sources = source_files(dataset_dir)
    collected = {key: [] for key in SIGNALS}
    paired_values = {"left": [[], []], "right": [[], []]}
    for path, interval_us in sources:
        signals = load_signal_columns(path, interval_us)
        for key, (_, values) in signals.items():
            if len(values):
                collected[key].append(values)
        for side in ("left", "right"):
            command, output = paired(signals[f"{side}_command"],
                                     signals[f"{side}_output"])
            if len(command):
                paired_values[side][0].append(command)
                paired_values[side][1].append(output)
    pooled = {key: np.concatenate(values) for key, values in collected.items()}
    pairs = {
        side: (np.concatenate(values[0]), np.concatenate(values[1]))
        for side, values in paired_values.items()
    }

    configure_style(args.axis_label_size, args.tick_size)
    fig, axes = plt.subplots(2, 2, figsize=(args.figure_width, args.figure_height))
    pooled_hist(axes[0, 0], pooled["left_command"], pooled["left_output"],
                "Left-wing angle distribution", args.bins)
    pooled_hist(axes[0, 1], pooled["right_command"], pooled["right_output"],
                "Right-wing angle distribution", args.bins)
    left_metrics = tracking_density(axes[1, 0], *pairs["left"],
                                    "Left-wing tracking", args.gridsize)
    right_metrics = tracking_density(axes[1, 1], *pairs["right"],
                                     "Right-wing tracking", args.gridsize)
    for ax, label in zip(axes.flat, "abcd"):
        add_panel_label(ax, f"({label})")
    fig.subplots_adjust(left=0.09, right=0.965, top=0.93, bottom=0.10,
                        wspace=0.30, hspace=0.50)
    save_panels(
        fig,
        {
            "a_left_distribution": axes[0, 0],
            "b_right_distribution": axes[0, 1],
            "c_left_tracking": axes[1, 0],
            "d_right_tracking": axes[1, 1],
        },
        output_dir / "panels",
        "actuation",
        args.dpi,
    )
    save_figure(fig, output_dir / "dataset_wing_actuation", args.dpi)
    print(f"Source files: {len(sources)}")
    print(f"Left  RMSE/bias: {left_metrics[0]:.3f}/{left_metrics[1]:+.3f} deg")
    print(f"Right RMSE/bias: {right_metrics[0]:.3f}/{right_metrics[1]:+.3f} deg")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_FIGURE_DIR))
    parser.add_argument("--bins", type=int, default=72)
    parser.add_argument("--gridsize", type=int, default=62)
    parser.add_argument("--figure-width", type=float, default=7.2)
    parser.add_argument("--figure-height", type=float, default=5.25)
    parser.add_argument("--axis-label-size", type=float, default=11.0)
    parser.add_argument("--tick-size", type=float, default=9.3)
    parser.add_argument("--dpi", type=int, default=400)
    render(parser.parse_args())


if __name__ == "__main__":
    main()
