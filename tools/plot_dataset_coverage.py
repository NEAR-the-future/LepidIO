#!/usr/bin/env python3
"""Plot motion and attitude coverage of the LepidIO flight dataset."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from dataset_plot_utils import (
    COLORS, DEFAULT_DATASET_DIR, DEFAULT_FIGURE_DIR, add_panel_label,
    balanced_values, configure_style, load_all_clean, resolve_path,
    save_figure, style_axis,
    save_panels,
)


def colored_violin(ax, datasets, positions, colors):
    parts = ax.violinplot(datasets, positions=positions, widths=0.72,
                          showextrema=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.52)
    for pos, values in zip(positions, datasets):
        q5, q25, med, q75, q95 = np.percentile(values, [5, 25, 50, 75, 95])
        ax.plot([pos, pos], [q5, q95], color=COLORS["dark"], linewidth=0.9)
        ax.add_patch(plt.Rectangle((pos-0.10, q25), 0.20, q75-q25,
                                   facecolor="white", edgecolor=COLORS["dark"],
                                   linewidth=0.75, zorder=4))
        ax.scatter([pos], [med], s=10, color=COLORS["dark"], zorder=5)


def render(args):
    dataset_dir = resolve_path(args.dataset_dir)
    output_dir = resolve_path(args.output_dir)
    sequences = load_all_clean(dataset_dir)
    columns = ["gt_px", "gt_py", "gt_pz", "roll_deg", "pitch_deg", "yaw_deg"]
    pooled = balanced_values(sequences, columns, args.samples_per_sequence)

    configure_style(args.axis_label_size, args.tick_size)
    fig = plt.figure(figsize=(args.figure_width, args.figure_height))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.16, 1.0],
                            left=0.075, right=0.96, top=0.93, bottom=0.11,
                            wspace=0.30, hspace=0.42)
    ax_a = fig.add_subplot(grid[:, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 1])

    # Light individual paths show sequence diversity; hexbin shows occupancy.
    for _, frame in sequences:
        ax_a.plot(frame["gt_px"], frame["gt_py"], color=COLORS["teal"],
                  alpha=0.10, linewidth=0.65)
    density = ax_a.hexbin(
        pooled["gt_px"], pooled["gt_py"], gridsize=args.hexbin_size,
        mincnt=1, cmap="GnBu", linewidths=0, alpha=0.90,
    )
    colorbar = fig.colorbar(density, ax=ax_a, fraction=0.045, pad=0.025)
    colorbar.ax.set_title("Count", fontsize=8.5, pad=4)
    colorbar.ax.tick_params(labelsize=args.tick_size)
    ax_a.scatter([0], [0], marker="o", s=22, color=COLORS["dark"], zorder=8,
                 label="Sequence start")
    ax_a.set_xlabel(r"Relative $x$ [m]")
    ax_a.set_ylabel(r"Relative $y$ [m]")
    ax_a.set_title(f"XY coverage across {len(sequences)} flights")
    ax_a.set_aspect("equal", adjustable="datalim")
    ax_a.legend(frameon=False, loc="upper right")
    style_axis(ax_a, "both")

    position_values = [pooled[name].to_numpy() for name in ("gt_px", "gt_py", "gt_pz")]
    colored_violin(ax_b, position_values, [1, 2, 3],
                   [COLORS["navy"], COLORS["teal"], COLORS["blue"]])
    ax_b.set_xticks([1, 2, 3], [r"$x$", r"$y$", r"$z$"])
    ax_b.set_ylabel("Relative position [m]")
    ax_b.set_title("Position distribution")
    style_axis(ax_b)

    attitude_values = [
        pooled[name].to_numpy() for name in ("roll_deg", "pitch_deg", "yaw_deg")
    ]
    colored_violin(ax_c, attitude_values, [1, 2, 3],
                   [COLORS["mint"], COLORS["red"], COLORS["orange"]])
    ax_c.set_xticks([1, 2, 3], ["Roll", "Pitch", "Yaw"])
    ax_c.set_ylabel("Attitude [deg]")
    ax_c.set_title("Recorded attitude distribution")
    style_axis(ax_c)

    for ax, label in zip((ax_a, ax_b, ax_c), ("(a)", "(b)", "(c)")):
        add_panel_label(ax, label)
    save_panels(
        fig,
        {
            "a_xy_coverage": (ax_a, colorbar.ax),
            "b_position_distribution": ax_b,
            "c_attitude_distribution": ax_c,
        },
        output_dir / "panels",
        "coverage",
        args.dpi,
    )
    save_figure(fig, output_dir / "dataset_motion_coverage", args.dpi)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_FIGURE_DIR))
    parser.add_argument("--samples-per-sequence", type=int, default=1200)
    parser.add_argument("--hexbin-size", type=int, default=46)
    parser.add_argument("--figure-width", type=float, default=7.2)
    parser.add_argument("--figure-height", type=float, default=4.65)
    parser.add_argument("--axis-label-size", type=float, default=11.0)
    parser.add_argument("--tick-size", type=float, default=9.3)
    parser.add_argument("--dpi", type=int, default=400)
    render(parser.parse_args())


if __name__ == "__main__":
    main()
