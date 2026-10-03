#!/usr/bin/env python3
"""Plot dataset-wide distributions and phase-locked FWAV oscillations."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dataset_plot_utils import (
    COLORS, DEFAULT_DATASET_DIR, DEFAULT_FIGURE_DIR, add_panel_label,
    balanced_values, configure_style, load_all_clean, moving_average,
    periodogram, resolve_path, save_figure, style_axis,
    save_panels,
)


def violin_with_box(ax, values, color, ylabel, title):
    parts = ax.violinplot(values, positions=[1], widths=0.72, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.55)
    q5, q25, median, q75, q95 = np.percentile(values, [5, 25, 50, 75, 95])
    ax.plot([1, 1], [q5, q95], color=COLORS["dark"], linewidth=1.0)
    ax.add_patch(plt.Rectangle((0.93, q25), 0.14, q75-q25,
                               facecolor="white", edgecolor=COLORS["dark"],
                               linewidth=0.8, zorder=4))
    ax.scatter([1], [median], s=13, color=COLORS["dark"], zorder=5)
    ax.set_xlim(0.45, 1.55)
    ax.set_xticks([])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    style_axis(ax)


def phase_statistics(phase, values, bins=36):
    phase = np.mod(phase, 2*np.pi)
    edges = np.linspace(0, 2*np.pi, bins+1)
    centers = (edges[:-1] + edges[1:]) / 2
    means = np.full(bins, np.nan)
    stds = np.full(bins, np.nan)
    for index in range(bins):
        selected = values[(phase >= edges[index]) & (phase < edges[index+1])]
        if len(selected):
            means[index] = np.mean(selected)
            stds[index] = np.std(selected)
    return centers, means, stds


def render(args):
    dataset_dir = resolve_path(args.dataset_dir)
    output_dir = resolve_path(args.output_dir)
    sequences = load_all_clean(dataset_dir)
    by_name = dict(sequences)
    if args.sequence not in by_name:
        raise FileNotFoundError(f"Sequence {args.sequence} not found in {dataset_dir}")
    representative = by_name[args.sequence]
    pooled = balanced_values(sequences, ["pitch_deg", "gt_pz"])

    time = representative["time_s"].to_numpy()
    pitch = representative["pitch_deg"].to_numpy()
    z = representative["gt_pz"].to_numpy()
    acc_z = representative["acc_z"].to_numpy()
    phase = representative["flap_phase_left_actual_rad"].to_numpy()
    wing = np.rad2deg(
        representative["flap_displacement_left_actual_rad"].to_numpy()
    )
    sample_hz = 1.0 / np.median(np.diff(time))
    z_osc = z - moving_average(z, round(args.detrend_s * sample_hz))

    pitch_zoom_start = args.pitch_zoom_start
    pitch_zoom_end = min(time[-1], pitch_zoom_start + args.pitch_zoom_duration)
    pitch_zoom = (time >= pitch_zoom_start) & (time <= pitch_zoom_end)
    vertical_zoom_start = args.vertical_zoom_start
    vertical_zoom_end = min(
        time[-1], vertical_zoom_start + args.vertical_zoom_duration
    )
    vertical_zoom = (time >= vertical_zoom_start) & (time <= vertical_zoom_end)
    if np.sum(pitch_zoom) < 10:
        raise ValueError("Selected pitch zoom interval contains fewer than ten samples")
    if np.sum(vertical_zoom) < 10:
        raise ValueError("Selected vertical zoom interval contains fewer than ten samples")

    configure_style(args.axis_label_size, args.tick_size)
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(args.figure_width, args.figure_height),
        gridspec_kw={"width_ratios": [0.90, 0.90, 2.50]},
    )
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes.flat

    violin_with_box(ax_a, pooled["pitch_deg"].to_numpy(), COLORS["red"],
                    "Pitch [deg]", "Dataset-wide pitch")
    violin_with_box(ax_b, pooled["gt_pz"].to_numpy(), COLORS["blue"],
                    r"Vertical position $p_z$ [m]", "Dataset-wide altitude")

    ax_c.plot(time, pitch, color=COLORS["red"], linewidth=1.0)
    ax_c.set_ylabel("Pitch [deg]", color=COLORS["red"])
    ax_c.tick_params(axis="y", colors=COLORS["red"])
    twin_c = ax_c.twinx()
    twin_c.plot(time, z_osc * 100.0, color=COLORS["blue"], linewidth=1.0)
    twin_c.set_ylabel(r"$z_{osc}$ [cm]", color=COLORS["blue"], labelpad=4)
    twin_c.tick_params(axis="y", colors=COLORS["blue"])
    twin_c.spines["top"].set_visible(False)
    ax_c.axvspan(
        pitch_zoom_start, pitch_zoom_end,
        facecolor=COLORS["orange"], edgecolor=COLORS["orange"],
        linewidth=0.8, alpha=0.18,
    )
    ax_c.axvspan(
        vertical_zoom_start, vertical_zoom_end,
        facecolor=COLORS["mint"], edgecolor=COLORS["teal"],
        linewidth=0.8, alpha=0.18,
    )
    ax_c.set_xlabel("Time [s]")
    ax_c.set_title("Representative flight")
    style_axis(ax_c)

    ax_d.plot(time[pitch_zoom], pitch[pitch_zoom], color=COLORS["red"], linewidth=1.35,
              label="Pitch")
    ax_d.set_xlabel("Time [s]")
    ax_d.set_ylabel("Pitch [deg]", color=COLORS["red"])
    ax_d.tick_params(axis="y", colors=COLORS["red"])
    twin_d = ax_d.twinx()
    twin_d.plot(time[pitch_zoom], wing[pitch_zoom], color=COLORS["navy"], linewidth=1.0,
                alpha=0.85, label="Left wing")
    twin_d.set_ylabel("Wing [deg]", color=COLORS["navy"], labelpad=2)
    twin_d.tick_params(axis="y", colors=COLORS["navy"])
    twin_d.spines["top"].set_visible(False)
    ax_d.set_title("Pitch and wing stroke")
    style_axis(ax_d)

    ax_e.plot(time[vertical_zoom], z_osc[vertical_zoom]*100.0, color=COLORS["blue"],
              linewidth=1.35, label=r"$z_{osc}$")
    ax_e.set_xlabel("Time [s]")
    ax_e.set_ylabel(r"$z_{osc}$ [cm]", color=COLORS["blue"], labelpad=2)
    ax_e.tick_params(axis="y", colors=COLORS["blue"])
    twin_e = ax_e.twinx()
    twin_e.plot(time[vertical_zoom], acc_z[vertical_zoom], color=COLORS["burnt"],
                linewidth=0.9, alpha=0.8)
    twin_e.set_ylabel(r"$a_z$ [m/s$^2$]", color=COLORS["burnt"], labelpad=2)
    twin_e.tick_params(axis="y", colors=COLORS["burnt"])
    twin_e.spines["top"].set_visible(False)
    ax_e.set_title("Vertical and inertial disturbance")
    style_axis(ax_e)

    for values, color, label in (
        (pitch, COLORS["red"], "Pitch"),
        (z_osc, COLORS["blue"], r"$z_{osc}$"),
        (acc_z, COLORS["burnt"], r"$a_z$"),
    ):
        frequency, power = periodogram(values, sample_hz)
        valid = (frequency >= 0.2) & (frequency <= args.max_frequency)
        normalized = power / max(np.nanmax(power[valid]), 1e-12)
        ax_f.plot(frequency[valid], normalized[valid], color=color,
                  linewidth=1.25, label=label)
    ax_f.axvline(args.flap_frequency, color=COLORS["orange"], linestyle="--",
                 linewidth=1.2, label=f"{args.flap_frequency:g} Hz")
    ax_f.set_xlabel("Frequency [Hz]")
    ax_f.set_ylabel("Normalized PSD")
    ax_f.set_title("Periodic disturbance spectrum")
    ax_f.set_ylim(0, 1.08)
    ax_f.legend(frameon=False, ncol=2)
    style_axis(ax_f)

    for ax, label in zip(axes.flat, "abcdef"):
        add_panel_label(ax, f"({label})")
    fig.subplots_adjust(left=0.075, right=0.965, top=0.93, bottom=0.10,
                        wspace=0.74, hspace=0.52)
    save_panels(
        fig,
        {
            "a_pitch_distribution": ax_a,
            "b_altitude_distribution": ax_b,
            "c_flight_overview": (ax_c, twin_c),
            "d_pitch_wing_zoom": (ax_d, twin_d),
            "e_vertical_inertial_zoom": (ax_e, twin_e),
            "f_disturbance_spectrum": ax_f,
        },
        output_dir / "panels",
        f"periodicity_{args.sequence}",
        args.dpi,
    )
    save_figure(fig, output_dir / f"dataset_periodicity_{args.sequence}", args.dpi)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_FIGURE_DIR))
    parser.add_argument("--sequence", default="20260903_171951")
    parser.add_argument("--pitch-zoom-start", type=float, default=8.0)
    parser.add_argument("--pitch-zoom-duration", type=float, default=3.0)
    parser.add_argument("--vertical-zoom-start", type=float, default=18.0)
    parser.add_argument("--vertical-zoom-duration", type=float, default=3.0)
    parser.add_argument("--detrend-s", type=float, default=1.0)
    parser.add_argument("--flap-frequency", type=float, default=3.0)
    parser.add_argument("--max-frequency", type=float, default=12.0)
    parser.add_argument("--figure-width", type=float, default=12.5)
    parser.add_argument("--figure-height", type=float, default=5.2)
    parser.add_argument("--axis-label-size", type=float, default=11.0)
    parser.add_argument("--tick-size", type=float, default=9.3)
    parser.add_argument("--dpi", type=int, default=400)
    render(parser.parse_args())


if __name__ == "__main__":
    main()
