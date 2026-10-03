#!/usr/bin/env python3
r"""Create a paper-ready four-panel XY trajectory comparison.

Each panel overlays one estimator on the same ground-truth trajectory.  The
LepidIO panel reads ``stamped_traj_network.txt``. XY ATE is evaluated on the common time interval with
no spatial alignment or scale correction.

Example (PowerShell)
--------------------
    .\.venv\Scripts\python.exe tools\plot_paper_xy_comparison.py `
        --sequence 20260903_171951
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEQUENCE = "20260903_171951"
DEFAULT_LEPID_RUN = REPO_ROOT / "results" / "lepid_io_tcn7_w150_20260912_080826"
DEFAULT_BASELINE_ROOT = REPO_ROOT / "results" / "Baselines"
DEFAULT_DATASET_NAME = "Dataset_clean_raw_attitude"
DEFAULT_DATASET_DIR = REPO_ROOT / "datasets" / DEFAULT_DATASET_NAME
DEFAULT_OUTPUT_DIR = REPO_ROOT / "figures" / "xy_comparison"


@dataclass(frozen=True)
class Method:
    name: str
    color: str
    path: Path


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_tum_positions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing trajectory: {path}")
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.shape[1] < 4:
        raise ValueError(f"Expected at least timestamp,x,y,z columns: {path}")
    timestamps = data[:, 0].astype(float)
    positions = data[:, 1:4].astype(float)
    valid = np.isfinite(timestamps) & np.all(np.isfinite(positions), axis=1)
    return timestamps[valid], positions[valid]


def load_gt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing ground-truth CSV: {path}")
    frame = pd.read_csv(path)
    required = ["timestamp_us", "gt_px", "gt_py", "gt_pz"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing GT columns {missing}: {path}")
    timestamps = frame["timestamp_us"].to_numpy(dtype=float) * 1e-6
    positions = frame[["gt_px", "gt_py", "gt_pz"]].to_numpy(dtype=float)
    valid = np.isfinite(timestamps) & np.all(np.isfinite(positions), axis=1)
    return timestamps[valid], positions[valid]


def crop_time(
    timestamps: np.ndarray,
    positions: np.ndarray,
    start: float,
    end: float,
) -> tuple[np.ndarray, np.ndarray]:
    mask = (timestamps >= start) & (timestamps <= end)
    timestamps, positions = timestamps[mask], positions[mask]
    if len(timestamps) < 2:
        raise ValueError("Trajectory has fewer than two samples in common interval")
    return timestamps, positions


def xy_ate(
    gt_t: np.ndarray,
    gt_p: np.ndarray,
    est_t: np.ndarray,
    est_p: np.ndarray,
) -> float:
    gt_xy = np.column_stack(
        [np.interp(est_t, gt_t, gt_p[:, axis]) for axis in range(2)]
    )
    error_xy = est_p[:, :2] - gt_xy
    return float(np.sqrt(np.mean(np.sum(error_xy**2, axis=1))))


def rotate_xy(positions: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate XY coordinates for display without changing trajectory errors."""
    angle = np.deg2rad(float(degrees))
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    rotated = positions.copy()
    rotated[:, :2] = positions[:, :2] @ rotation.T
    return rotated


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def build_methods(args: argparse.Namespace) -> list[Method]:
    sequence = args.sequence
    lepid_run = resolve_path(args.lepid_run)
    baseline_root = resolve_path(args.baseline_root)
    dataset_name = args.dataset_name
    methods_by_name = {
        "AirIMU": Method(
            "AirIMU", "#F2B84B",
            baseline_root / "AirIMU" / dataset_name / sequence
            / "stamped_traj_estimate.txt",
        ),
        "AirIO": Method(
            "AirIO", "#2458D3",
            baseline_root / "AirIO" / dataset_name / sequence
            / "stamped_traj_estimate.txt",
        ),
        "TLIO": Method(
            "TLIO", "#C07ABB",
            baseline_root / "TLIO" / dataset_name / sequence
            / "stamped_traj_estimate.txt",
        ),
        "LepidIO": Method(
            "LepidIO", "#E31A1C",
            lepid_run / sequence / "stamped_traj_network.txt",
        ),
    }
    unknown = [name for name in args.order if name not in methods_by_name]
    if unknown:
        raise ValueError(f"Unknown method(s) in --order: {', '.join(unknown)}")
    if len(set(args.order)) != len(args.order):
        raise ValueError("--order must not contain duplicate method names")
    return [methods_by_name[name] for name in args.order]


def derive_limits(
    gt_xy: np.ndarray,
    margin_ratio: float,
    xlim: list[float] | None,
    ylim: list[float] | None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    lower = np.min(gt_xy, axis=0)
    upper = np.max(gt_xy, axis=0)
    span = np.maximum(upper - lower, 1.0)
    margin = span * margin_ratio
    lower, upper = lower - margin, upper + margin
    if xlim is not None:
        lower[0], upper[0] = map(float, xlim)
    if ylim is not None:
        lower[1], upper[1] = map(float, ylim)
    if np.any(lower >= upper):
        raise ValueError("Each explicit axis range must satisfy MIN < MAX")
    return (float(lower[0]), float(upper[0])), (float(lower[1]), float(upper[1]))


def add_scale_bar(
    ax: plt.Axes,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    length_m: float,
    y_fraction: float,
) -> None:
    x_span, y_span = xlim[1] - xlim[0], ylim[1] - ylim[0]
    center_x = (xlim[0] + xlim[1]) / 2.0
    y = ylim[0] + y_fraction * y_span
    cap = 0.014 * y_span
    x0, x1 = center_x - length_m / 2.0, center_x + length_m / 2.0
    ax.plot(
        [x0, x1], [y, y], color="#222222", linewidth=1.1,
        zorder=20, clip_on=False,
    )
    ax.plot(
        [x0, x0], [y - cap, y + cap], color="#222222", linewidth=0.8,
        zorder=20, clip_on=False,
    )
    ax.plot(
        [x1, x1], [y - cap, y + cap], color="#222222", linewidth=0.8,
        zorder=20, clip_on=False,
    )
    ax.text(
        center_x, y + 0.022 * y_span, f"{length_m:g} m",
        ha="center", va="bottom", fontsize=7.5,
    )


def render(args: argparse.Namespace) -> list[Path]:
    methods = build_methods(args)
    dataset_dir = resolve_path(args.dataset_dir)
    output_dir = resolve_path(args.output_dir)
    gt_path = dataset_dir / f"{args.sequence}.csv"

    gt_t, gt_p = load_gt(gt_path)
    loaded: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    starts, ends = [float(gt_t[0])], [float(gt_t[-1])]
    for method in methods:
        timestamps, positions = load_tum_positions(method.path)
        loaded[method.name] = timestamps, positions
        starts.append(float(timestamps[0]))
        ends.append(float(timestamps[-1]))

    common_start, common_end = max(starts), min(ends)
    if common_end <= common_start:
        raise ValueError("Trajectories have no common time interval")
    gt_t, gt_p = crop_time(gt_t, gt_p, common_start, common_end)
    for method in methods:
        loaded[method.name] = crop_time(
            *loaded[method.name], common_start, common_end
        )

    metrics = {
        method.name: xy_ate(gt_t, gt_p, *loaded[method.name])
        for method in methods
    }
    # Apply one rigid display rotation to every curve and marker.  A negative
    # angle is clockwise, so the default -90 degrees matches the paper layout.
    gt_p = rotate_xy(gt_p, args.plot_rotation_deg)
    loaded = {
        name: (timestamps, rotate_xy(positions, args.plot_rotation_deg))
        for name, (timestamps, positions) in loaded.items()
    }
    # Keep a common metric scale across all panels, but expand the reference
    # bounds enough to show AirIO in full. AirIMU may still be clipped because
    # its very large divergence would make every useful trajectory unreadable.
    limit_reference = gt_p[:, :2]
    if "AirIO" in loaded:
        limit_reference = np.vstack((limit_reference, loaded["AirIO"][1][:, :2]))
    xlim, ylim = derive_limits(
        limit_reference, args.margin, args.xlim, args.ylim
    )

    configure_style()
    panel_count = len(methods)
    if panel_count != 4:
        raise ValueError("The compact paper layout requires exactly four methods")
    fig, axes = plt.subplots(
        2, 2,
        figsize=(args.figure_width, args.figure_height),
        squeeze=False,
    )
    axes = axes.ravel()
    method_linewidths = {
        "AirIMU": args.airimu_linewidth,
        "AirIO": args.airio_linewidth,
        "TLIO": args.tlio_linewidth,
        "LepidIO": args.lepid_linewidth,
    }

    panel_anchors = ("SE", "SW", "NE", "NW")
    for panel_index, (ax, method, anchor) in enumerate(
        zip(axes, methods, panel_anchors)
    ):
        heading_shift = -0.10 if panel_index % 2 == 0 else 0.10
        _, estimate = loaded[method.name]
        ax.plot(
            gt_p[:, 0], gt_p[:, 1],
            color="#111111", linewidth=args.gt_linewidth,
            solid_capstyle="round", zorder=1,
        )
        ax.plot(
            estimate[:, 0], estimate[:, 1],
            color=method.color,
            linewidth=(method_linewidths[method.name]
                       if method_linewidths[method.name] is not None
                       else args.method_linewidth),
            alpha=(1.0 if method.name == "LepidIO" else args.baseline_alpha),
            solid_capstyle="round", zorder=3,
        )

        # One shared start point; the two endpoints are intentionally drawn as
        # overlapping stars when the estimator terminates near ground truth.
        ax.scatter(
            gt_p[0, 0], gt_p[0, 1], marker="o", s=24,
            facecolor="black", edgecolor="black", linewidth=0.4, zorder=10,
        )
        ax.scatter(
            gt_p[-1, 0], gt_p[-1, 1], marker="*", s=58,
            facecolor="#111111", edgecolor="black", linewidth=0.4, zorder=11,
        )
        ax.scatter(
            estimate[-1, 0], estimate[-1, 1], marker="*", s=42,
            facecolor=method.color, edgecolor="black", linewidth=0.4,
            zorder=12, clip_on=True,
        )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_anchor(anchor)
        ax.axis("off")
        # All four panels share identical metric limits, so one scale bar is
        # sufficient. Keep it in the final (bottom-right) LepidIO panel.
        if method.name == "LepidIO":
            add_scale_bar(ax, xlim, ylim, args.scale_bar, args.scale_bar_y)
        # Compact paper-style heading: colored method swatch and black text on
        # the first row, followed by the ATE value on the second row.
        ax.plot(
            [0.27 + heading_shift, 0.42 + heading_shift],
            [1.035, 1.035], transform=ax.transAxes,
            color=method.color, linewidth=args.heading_linewidth,
            solid_capstyle="butt", clip_on=False,
        )
        ax.text(
            0.45 + heading_shift, 1.035, method.name,
            transform=ax.transAxes,
            ha="left", va="center", color="#111111",
            fontsize=args.title_fontsize,
        )
        ax.text(
            0.50 + heading_shift, 0.950,
            f"XY ATE: {metrics[method.name]:.{args.ate_precision}f}",
            transform=ax.transAxes, ha="center", va="center",
            color="#111111", fontsize=args.title_fontsize,
        )

    gt_handle = Line2D([], [], color="#111111", linewidth=args.gt_linewidth)
    start_handle = Line2D(
        [], [], linestyle="none", marker="o", markersize=5.2,
        markerfacecolor="black", markeredgecolor="black",
    )
    end_handles = (
        Line2D(
            [], [], linestyle="none", marker="*", markersize=10.0,
            markerfacecolor="#111111", markeredgecolor="black",
        ),
        Line2D(
            [], [], linestyle="none", marker="*", markersize=7.5,
            markerfacecolor="#E31A1C", markeredgecolor="black",
        ),
    )
    fig.legend(
        [gt_handle, start_handle, end_handles],
        ["Ground Truth", "Start Point", "End Points"],
        loc="lower center", bbox_to_anchor=(0.5, args.legend_y),
        ncol=3, frameon=False, fontsize=args.legend_fontsize,
        columnspacing=1.8, handletextpad=0.55,
        handler_map={tuple: HandlerTuple(ndivide=1, pad=0)},
    )
    fig.subplots_adjust(
        left=0.012, right=0.995, top=0.91, bottom=args.bottom_margin,
        wspace=args.panel_spacing, hspace=args.row_spacing,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"xy_comparison_{args.sequence}"
    outputs = [stem.with_suffix(ext) for ext in (".pdf", ".svg", ".png")]
    fig.savefig(outputs[0], bbox_inches="tight", pad_inches=0.025)
    fig.savefig(outputs[1], bbox_inches="tight", pad_inches=0.025)
    fig.savefig(outputs[2], dpi=args.dpi, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)

    metadata_path = stem.with_suffix(".json")
    metadata = {
        "sequence": args.sequence,
        "trajectory_type": {
            "LepidIO": "network-only stamped_traj_network.txt",
            "baselines": "stamped_traj_estimate.txt",
        },
        "metric": "raw XY ATE = sqrt(mean(dx^2 + dy^2))",
        "alignment": "none",
        "common_time_s": [common_start, common_end],
        "xy_ate_m": metrics,
        "xlim_m": list(xlim),
        "ylim_m": list(ylim),
        "scale_bar_m": args.scale_bar,
        "scale_bar_y_fraction": args.scale_bar_y,
        "plot_rotation_deg": args.plot_rotation_deg,
        "ground_truth": str(gt_path),
        "trajectories": {method.name: str(method.path) for method in methods},
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    outputs.append(metadata_path)

    print(f"Sequence    : {args.sequence}")
    print(f"Common time : {common_start:.3f}--{common_end:.3f} s")
    for method in methods:
        print(f"{method.name:8s} XY ATE: {metrics[method.name]:.3f} m")
    for output in outputs:
        print(f"Saved       : {output}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", default=DEFAULT_SEQUENCE)
    parser.add_argument("--lepid-run", default=str(DEFAULT_LEPID_RUN))
    parser.add_argument("--baseline-root", default=str(DEFAULT_BASELINE_ROOT))
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--order", nargs="+",
        default=["AirIMU", "AirIO", "TLIO", "LepidIO"],
        help="Panel order; choose among AirIMU AirIO TLIO LepidIO",
    )
    parser.add_argument("--xlim", nargs=2, type=float, metavar=("MIN", "MAX"))
    parser.add_argument("--ylim", nargs=2, type=float, metavar=("MIN", "MAX"))
    parser.add_argument("--margin", type=float, default=0.08)
    parser.add_argument(
        "--plot-rotation-deg", type=float, default=-90.0,
        help="Rigid display rotation in degrees; negative values rotate clockwise",
    )
    parser.add_argument("--scale-bar", type=float, default=1.0)
    parser.add_argument(
        "--scale-bar-y", type=float, default=-0.030,
        help="Scale-bar height as a fraction of plot height from the lower boundary",
    )
    parser.add_argument("--gt-linewidth", type=float, default=0.5)
    parser.add_argument("--method-linewidth", type=float, default=0.50)
    parser.add_argument("--airimu-linewidth", type=float, default=None)
    parser.add_argument("--airio-linewidth", type=float, default=None)
    parser.add_argument("--tlio-linewidth", type=float, default=None)
    parser.add_argument("--lepid-linewidth", type=float, default=None)
    parser.add_argument("--baseline-alpha", type=float, default=0.72)
    parser.add_argument("--figure-width", type=float, default=3.8)
    parser.add_argument("--figure-height", type=float, default=3.35)
    parser.add_argument("--title-fontsize", type=float, default=7.2)
    parser.add_argument("--heading-linewidth", type=float, default=1.2)
    parser.add_argument("--legend-fontsize", type=float, default=7.2)
    parser.add_argument("--legend-y", type=float, default=0.005)
    parser.add_argument("--bottom-margin", type=float, default=0.14)
    parser.add_argument("--panel-spacing", type=float, default=-0.08)
    parser.add_argument("--row-spacing", type=float, default=0.16)
    parser.add_argument("--ate-precision", type=int, default=3)
    parser.add_argument(
        "--dpi", type=int, default=800,
        help="Raster PNG resolution; PDF/SVG outputs remain vector graphics",
    )
    render(parser.parse_args())


if __name__ == "__main__":
    main()
