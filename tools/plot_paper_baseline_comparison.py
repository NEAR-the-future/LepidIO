#!/usr/bin/env python3
r"""Plot GT, LepidIO, AirIMU, AirIO, and TLIO in one paper-ready 3D view.

The default paths reproduce the current hardware comparison. Change only
``--sequence`` when all result roots keep the same directory structure.

Example
-------
    .\.venv\Scripts\python.exe tools\plot_paper_baseline_comparison.py `
        --sequence 20260903_191508 `
        --elevation 35 --azimuth -58

Outputs PDF/SVG/PNG plus an interactive HTML view under ``figures/``.
No trajectory alignment or scale correction is applied.
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
import plotly.graph_objects as go


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEQUENCE = "20260903_191508"
DEFAULT_LEPID_RUN = REPO_ROOT / "results" / "lepidnet_test_20260912_221816_window1.0"
DEFAULT_BASELINE_ROOT = REPO_ROOT / "results" / "Baselines"
DEFAULT_DATASET_NAME = "Dataset_clean_raw_attitude"
DEFAULT_DATASET_DIR = REPO_ROOT / "datasets" / DEFAULT_DATASET_NAME
DEFAULT_OUTPUT_DIR = REPO_ROOT / "figures" / "baseline_comparison"


@dataclass(frozen=True)
class Method:
    name: str
    color: str
    path: Path
    linewidth: float
    alpha: float


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_tum_positions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing trajectory: {path}")
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.shape[1] < 4:
        raise ValueError(f"Expected at least ts,x,y,z columns: {path}")
    timestamps = data[:, 0].astype(float)
    positions = data[:, 1:4].astype(float)
    valid = np.isfinite(timestamps) & np.all(np.isfinite(positions), axis=1)
    return timestamps[valid], positions[valid]


def load_gt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing GT CSV: {path}")
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
        raise ValueError("Trajectory has fewer than two samples in common time range")
    return timestamps, positions


def plot_limits(gt: np.ndarray, margin_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    lower = np.min(gt, axis=0)
    upper = np.max(gt, axis=0)
    span = np.maximum(upper - lower, np.array([0.5, 0.5, 0.3]))
    margin = np.maximum(margin_ratio * span, np.array([0.20, 0.20, 0.15]))
    return lower - margin, upper + margin


def segment_box_intersection(
    start: np.ndarray,
    end: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Clip one line segment against a 3D axis-aligned box."""
    direction = end - start
    t_min, t_max = 0.0, 1.0
    for axis in range(3):
        if abs(direction[axis]) < 1e-12:
            if start[axis] < lower[axis] or start[axis] > upper[axis]:
                return None
            continue
        t1 = (lower[axis] - start[axis]) / direction[axis]
        t2 = (upper[axis] - start[axis]) / direction[axis]
        near, far = min(t1, t2), max(t1, t2)
        t_min, t_max = max(t_min, near), min(t_max, far)
        if t_min > t_max:
            return None
    return start + t_min * direction, start + t_max * direction


def clip_polyline(
    positions: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return visible line segments and inside-to-outside boundary crossings."""
    visible: list[np.ndarray] = []
    exits: list[np.ndarray] = []
    inside = np.all((positions >= lower) & (positions <= upper), axis=1)
    for index in range(len(positions) - 1):
        clipped = segment_box_intersection(
            positions[index], positions[index + 1], lower, upper
        )
        if clipped is None:
            continue
        first, second = clipped
        visible.extend((first, second, np.full(3, np.nan)))
        if inside[index] and not inside[index + 1]:
            exits.append(second)
    if not visible:
        return np.empty((0, 3)), exits
    return np.vstack(visible), exits


def raw_ate(
    gt_t: np.ndarray,
    gt_p: np.ndarray,
    est_t: np.ndarray,
    est_p: np.ndarray,
) -> float:
    interpolated_gt = np.column_stack(
        [np.interp(est_t, gt_t, gt_p[:, axis]) for axis in range(3)]
    )
    return float(np.sqrt(np.mean(np.sum((est_p - interpolated_gt) ** 2, axis=1))))


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8.2,
            "axes.linewidth": 0.7,
            "grid.linewidth": 0.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def build_methods(
    sequence: str,
    lepid_run: Path,
    baseline_root: Path,
    dataset_name: str,
    args: argparse.Namespace,
) -> list[Method]:
    return [
        Method(
            "LepidIO", "#E31A1C",
            lepid_run / sequence / "stamped_traj_network.txt",
            args.lepid_linewidth, 1.0,
        ),
        Method(
            "AirIMU", "#F2B84B",
            baseline_root / "AirIMU" / dataset_name / sequence
            / "stamped_traj_estimate.txt",
            args.airimu_linewidth, args.baseline_alpha,
        ),
        Method(
            "AirIO", "#2458D3",
            baseline_root / "AirIO" / dataset_name / sequence
            / "stamped_traj_estimate.txt",
            args.airio_linewidth, args.baseline_alpha,
        ),
        Method(
            "TLIO", "#C07ABB",
            baseline_root / "TLIO" / dataset_name / sequence
            / "stamped_traj_estimate.txt",
            args.tlio_linewidth, args.baseline_alpha,
        ),
    ]


def render(args: argparse.Namespace) -> list[Path]:
    sequence = args.sequence
    lepid_run = resolve_path(args.lepid_run)
    baseline_root = resolve_path(args.baseline_root)
    dataset_dir = resolve_path(args.dataset_dir)
    output_dir = resolve_path(args.output_dir)
    gt_path = dataset_dir / f"{sequence}.csv"
    methods = build_methods(
        sequence, lepid_run, baseline_root, args.dataset_name, args
    )

    gt_t, gt_p = load_gt(gt_path)
    loaded: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    starts, ends = [float(gt_t[0])], [float(gt_t[-1])]
    end_marker_sizes = {
        "LepidIO": 11.0,
        "AirIMU": 9.0,
        "AirIO": 7.0,
        "TLIO": 5.0,
    }
    for method in methods:
        timestamps, positions = load_tum_positions(method.path)
        loaded[method.name] = (timestamps, positions)
        starts.append(float(timestamps[0]))
        ends.append(float(timestamps[-1]))
    common_start, common_end = max(starts), min(ends)
    if common_end <= common_start:
        raise ValueError("The five trajectories have no common time interval")
    gt_t, gt_p = crop_time(gt_t, gt_p, common_start, common_end)
    for method in methods:
        loaded[method.name] = crop_time(
            *loaded[method.name], common_start, common_end
        )

    # The paper figure should show GT and the proposed method in full. Methods
    # leaving this common region are clipped and annotated at their first exit.
    boundary_reference = np.vstack((gt_p, loaded["LepidIO"][1]))
    lower, upper = plot_limits(boundary_reference, args.boundary_margin)
    for axis, limits, name in (
        (0, args.xlim, "xlim"),
        (1, args.ylim, "ylim"),
        (2, args.zlim, "zlim"),
    ):
        if limits is None:
            continue
        minimum, maximum = map(float, limits)
        if minimum >= maximum:
            raise ValueError(
                f"--{name} requires MIN < MAX, received {minimum}, {maximum}"
            )
        lower[axis], upper[axis] = minimum, maximum
    plot_span = upper - lower
    overflow = args.overflow_margin * plot_span
    overflow_lower = lower - overflow
    overflow_upper = upper + overflow
    configure_style()
    fig = plt.figure(figsize=(args.figure_width, args.figure_height))
    ax = fig.add_subplot(111, projection="3d")

    gt_color = "#111111"
    gt_line, = ax.plot(
        *gt_p.T, color=gt_color, linewidth=args.gt_linewidth,
        alpha=1.0, label="Ground Truth",
    )
    line_handles: list[Line2D] = [gt_line]
    end_handles: list[Line2D] = []
    metrics: dict[str, float] = {}
    boundary_exits: dict[str, list[list[float]]] = {}

    # GT endpoint is always visible because the plotting bounds contain GT.
    ax.scatter(
        *gt_p[-1], marker="*", s=62, c=gt_color, edgecolors="black",
        linewidths=0.55, depthshade=False, zorder=12,
    )
    end_handles.append(Line2D(
        [], [], linestyle="none", marker="*", markersize=12.5,
        markerfacecolor=gt_color, markeredgecolor="black",
        markeredgewidth=0.4,
    ))

    for method in methods:
        timestamps, positions = loaded[method.name]
        metrics[method.name] = raw_ate(gt_t, gt_p, timestamps, positions)
        # Keep a limited amount of the real trajectory after it leaves the
        # nominal axes box. The axes limits remain unchanged, so the main
        # comparison retains a useful scale.
        clipped, _ = clip_polyline(positions, overflow_lower, overflow_upper)
        _, exits = clip_polyline(positions, lower, upper)
        boundary_exits[method.name] = [point.tolist() for point in exits]
        # Crossings remain available in JSON for later PPT annotations; the
        # paper figure deliberately does not draw boundary-crossing markers.
        if len(clipped):
            line, = ax.plot(
                *clipped.T, color=method.color, linewidth=method.linewidth,
                alpha=method.alpha, label=method.name, clip_on=False,
            )
        else:
            line = Line2D(
                [], [], color=method.color, linewidth=method.linewidth,
                alpha=method.alpha,
            )
        line_handles.append(line)

        final_inside = bool(
            np.all((positions[-1] >= lower) & (positions[-1] <= upper))
        )
        if final_inside:
            ax.scatter(
                *positions[-1], marker="*", s=54, c=method.color,
                edgecolors="black", linewidths=0.45,
                depthshade=False, zorder=10,
            )
        end_handles.append(Line2D(
            [], [], linestyle="none", marker="*",
            markersize=end_marker_sizes[method.name],
            markerfacecolor=method.color, markeredgecolor="black",
            markeredgewidth=0.35,
        ))

    start = gt_p[0]
    ax.scatter(
        *start, marker="o", s=35, c="black", edgecolors="white",
        linewidths=0.5, depthshade=False, zorder=12,
    )
    start_handle = Line2D(
        [], [], linestyle="none", marker="o", markersize=6,
        markerfacecolor="black", markeredgecolor="black",
    )

    ax.set_xlim(lower[0], upper[0])
    ax.set_ylim(lower[1], upper[1])
    ax.set_zlim(lower[2], upper[2])

    if args.axis_lengths is not None:
        box_aspect = np.asarray(args.axis_lengths, dtype=float)
        if np.any(box_aspect <= 0):
            raise ValueError("--axis-lengths values must all be positive")
    elif args.axis_lengths_from_range:
        # Make the physical lengths of X/Y/Z proportional to their displayed
        # numeric spans. An optional Z multiplier preserves vertical detail.
        box_aspect = (upper - lower).astype(float)
        box_aspect[2] *= args.z_length_scale
        box_aspect /= np.max(box_aspect)
    else:
        box_aspect = np.array([1.55, 1.25, 0.82], dtype=float)
    ax.set_box_aspect(tuple(box_aspect))

    if args.show_limit_ticks:
        for limits, setter in (
            (args.xlim, ax.set_xticks),
            (args.ylim, ax.set_yticks),
            (args.zlim, ax.set_zticks),
        ):
            if limits is None:
                continue
            minimum, maximum = map(float, limits)
            setter(np.linspace(minimum, maximum, args.axis_tick_count))
    ax.view_init(elev=args.elevation, azim=args.azimuth)
    ax.set_xlabel(r"$X$ [m]", labelpad=3)
    ax.set_ylabel(r"$Y$ [m]", labelpad=3)
    ax.set_zlabel(r"$Z$ [m]", labelpad=3)
    ax.tick_params(axis="both", which="major", labelsize=7.8, pad=0)
    ax.grid(True, color="#AFAFAF", alpha=0.68)
    pane_gray = float(np.clip(args.pane_gray, 0.0, 1.0))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((pane_gray, pane_gray, pane_gray, 1.0))
        axis.pane.set_edgecolor("#B8B8B8")

    legend_handles = line_handles + [start_handle, tuple(end_handles)]
    legend_labels = ["Ground Truth"] + [method.name for method in methods]
    legend_labels += ["Start Point", "End Points"]
    ax.legend(
        legend_handles,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1.21, 0.46),
        frameon=False,
        handlelength=2.7,
        handletextpad=0.55,
        borderaxespad=0,
        handler_map={tuple: HandlerTuple(ndivide=1, pad=0)},
    )
    fig.subplots_adjust(left=0.0, right=0.72, top=0.99, bottom=0.02)

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / f"trajectory_comparison_{sequence}"
    outputs = [prefix.with_suffix(ext) for ext in (".pdf", ".svg", ".png")]
    for output in outputs:
        options: dict[str, object] = {"bbox_inches": "tight", "pad_inches": 0.03}
        if output.suffix == ".png":
            options["dpi"] = 450
        fig.savefig(output, **options)
    plt.close(fig)

    interactive = go.Figure()
    interactive.add_trace(go.Scatter3d(
        x=gt_p[:, 0], y=gt_p[:, 1], z=gt_p[:, 2], mode="lines",
        name="Ground Truth", line=dict(color=gt_color, width=6),
    ))
    interactive.add_trace(go.Scatter3d(
        x=[gt_p[-1, 0]], y=[gt_p[-1, 1]], z=[gt_p[-1, 2]],
        mode="text", name="Ground Truth end", text=["★"],
        textfont=dict(color=gt_color, size=18), showlegend=False,
    ))
    for method in methods:
        _, positions = loaded[method.name]
        interactive.add_trace(go.Scatter3d(
            x=positions[:, 0], y=positions[:, 1], z=positions[:, 2], mode="lines",
            name=method.name,
            line=dict(color=method.color, width=max(2.0, 2.4 * method.linewidth)),
            opacity=method.alpha,
        ))
    elevation, azimuth = np.deg2rad([args.elevation, args.azimuth])
    interactive.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=25, b=55),
        font=dict(family="Times New Roman", size=15),
        legend=dict(orientation="h", y=-0.08),
        scene=dict(
            xaxis=dict(
                title="X [m]", range=[lower[0], upper[0]],
                showbackground=True,
                backgroundcolor=f"rgb({int(255*pane_gray)}, {int(255*pane_gray)}, {int(255*pane_gray)})",
            ),
            yaxis=dict(
                title="Y [m]", range=[lower[1], upper[1]],
                showbackground=True,
                backgroundcolor=f"rgb({int(255*pane_gray)}, {int(255*pane_gray)}, {int(255*pane_gray)})",
            ),
            zaxis=dict(
                title="Z [m]", range=[lower[2], upper[2]],
                showbackground=True,
                backgroundcolor=f"rgb({int(255*pane_gray)}, {int(255*pane_gray)}, {int(255*pane_gray)})",
            ),
            aspectmode="manual",
            aspectratio=dict(
                x=float(box_aspect[0]),
                y=float(box_aspect[1]),
                z=float(box_aspect[2]),
            ),
            camera=dict(eye=dict(
                x=float(2 * np.cos(elevation) * np.cos(azimuth)),
                y=float(2 * np.cos(elevation) * np.sin(azimuth)),
                z=float(2 * np.sin(elevation)),
            )),
            dragmode="orbit",
        ),
    )
    html_path = prefix.with_suffix(".html")
    interactive.write_html(
        html_path,
        include_plotlyjs=True,
        full_html=True,
        config=dict(
            scrollZoom=True,
            displaylogo=False,
            toImageButtonOptions=dict(
                format="png", filename=prefix.name,
                width=1800, height=1150, scale=2,
            ),
        ),
    )
    outputs.append(html_path)

    provenance = {
        "sequence": sequence,
        "common_time_s": [common_start, common_end],
        "alignment": "none",
        "lepidio_trajectory_type": (
            "network-only displacement accumulation from stamped_traj_network.txt"
        ),
        "plot_limits_xyz_m": {"lower": lower.tolist(), "upper": upper.tolist()},
        "overflow_margin_ratio": args.overflow_margin,
        "pane_gray": pane_gray,
        "box_aspect_xyz": box_aspect.tolist(),
        "ground_truth": str(gt_path),
        "methods": {
            method.name: {
                "trajectory": str(method.path),
                "color": method.color,
                "alpha": method.alpha,
                "linewidth": method.linewidth,
                "raw_ate_m": metrics[method.name],
                "boundary_exits": boundary_exits[method.name],
            }
            for method in methods
        },
    }
    metadata_path = prefix.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    outputs.append(metadata_path)

    print(f"Sequence   : {sequence}")
    print(f"Common time: {common_start:.3f}--{common_end:.3f} s")
    for method in methods:
        print(f"{method.name:8s} ATE: {metrics[method.name]:.3f} m")
    for output in outputs:
        print(f"Saved      : {output}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a multi-method 3D trajectory comparison figure."
    )
    parser.add_argument("--sequence", default=DEFAULT_SEQUENCE)
    parser.add_argument("--lepid-run", default=str(DEFAULT_LEPID_RUN))
    parser.add_argument("--baseline-root", default=str(DEFAULT_BASELINE_ROOT))
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--boundary-margin", type=float, default=0.12,
        help="Plot margin as a fraction of the GT span",
    )
    parser.add_argument(
        "--overflow-margin", type=float, default=0.25,
        help="Visible trajectory extension beyond the axes, as an axes-span ratio",
    )
    parser.add_argument(
        "--pane-gray", type=float, default=0.85,
        help="Axes-pane gray level: 0 is black, 1 is white",
    )
    parser.add_argument(
        "--xlim", type=float, nargs=2, metavar=("MIN", "MAX"),
        help="Explicit X-axis range in metres",
    )
    parser.add_argument(
        "--ylim", type=float, nargs=2, metavar=("MIN", "MAX"),
        help="Explicit Y-axis range in metres",
    )
    parser.add_argument(
        "--zlim", type=float, nargs=2, metavar=("MIN", "MAX"),
        help="Explicit Z-axis range in metres",
    )
    parser.add_argument(
        "--show-limit-ticks", action="store_true",
        help="Show the exact manual axis limits as endpoint tick labels",
    )
    parser.add_argument(
        "--axis-tick-count", type=int, default=5,
        help="Tick count per manually limited axis with --show-limit-ticks",
    )
    parser.add_argument(
        "--axis-lengths", type=float, nargs=3, metavar=("X", "Y", "Z"),
        help="Manual relative physical lengths of the displayed X/Y/Z axes",
    )
    parser.add_argument(
        "--axis-lengths-from-range", action="store_true",
        help="Make physical axis lengths proportional to xlim/ylim/zlim spans",
    )
    parser.add_argument(
        "--z-length-scale", type=float, default=2.5,
        help="Vertical length multiplier used with --axis-lengths-from-range",
    )
    parser.add_argument("--elevation", type=float, default=60.0)
    parser.add_argument("--azimuth", type=float, default=-45.0)
    parser.add_argument("--baseline-alpha", type=float, default=0.70)
    parser.add_argument("--gt-linewidth", type=float, default=1.90)
    parser.add_argument("--lepid-linewidth", type=float, default=1.90)
    parser.add_argument("--airimu-linewidth", type=float, default=1.55)
    parser.add_argument("--airio-linewidth", type=float, default=1.55)
    parser.add_argument("--tlio-linewidth", type=float, default=1.55)
    args = parser.parse_args()
    render(args)


if __name__ == "__main__":
    main()
