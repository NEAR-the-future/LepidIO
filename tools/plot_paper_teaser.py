#!/usr/bin/env python3
"""Generate the paper teaser trajectory in a compact AirIO-like style.

The figure uses a held-out GT trajectory and a LepidNet accumulated estimate in their
recorded common world frame. It performs timestamp cropping/interpolation only;
no post-hoc SE(3) or scale alignment is applied.

Typical use (only the sequence changes):

    python tools/plot_paper_teaser.py --sequence 20260903_191508

To use another trained run:

    python tools/plot_paper_teaser.py \
        --run-dir results/lepid_io_tcn7_w150_YYYYMMDD_HHMMSS \
        --sequence 20260903_191508
"""

from __future__ import annotations

import argparse
import json
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

# Change this once after the final model/configuration is locked. Afterwards,
# generating another flight requires changing only --sequence.
DEFAULT_RUN_DIR = REPO_ROOT / "results" / "lepid_io_tcn7_w150_20260912_072115"
# Keep exports inside this repository, grouped by run to avoid collisions.
DEFAULT_OUTPUT_DIR = REPO_ROOT / "figures"


def _resolve_from_repo(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _load_run_config(run_dir: Path) -> dict:
    config_path = run_dir / "config_used.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing run configuration: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _dataset_csv(run_dir: Path, sequence: str, explicit: str | None) -> Path:
    if explicit:
        dataset_dir = _resolve_from_repo(explicit)
    else:
        config = _load_run_config(run_dir)
        csv_dir = config.get("data", {}).get("csv_dir")
        if not csv_dir:
            raise KeyError(
                "config_used.json does not define data.csv_dir; pass --dataset-dir"
            )
        dataset_dir = _resolve_from_repo(csv_dir)
    return dataset_dir / f"{sequence}.csv"


def _load_estimate(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.shape[1] < 4:
        raise ValueError(f"Expected at least ts,x,y,z columns in {path}")
    return data[:, 0], data[:, 1:4]


def _load_ground_truth(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = pd.read_csv(path)
    required = ["timestamp_us", "gt_px", "gt_py", "gt_pz"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing GT columns in {path}: {missing}")
    timestamps = data["timestamp_us"].to_numpy(dtype=float) * 1e-6
    positions = data[["gt_px", "gt_py", "gt_pz"]].to_numpy(dtype=float)
    return timestamps, positions


def _crop_to_overlap(
    gt_t: np.ndarray,
    gt_p: np.ndarray,
    est_t: np.ndarray,
    est_p: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    start = max(float(gt_t[0]), float(est_t[0]))
    end = min(float(gt_t[-1]), float(est_t[-1]))
    if end <= start:
        raise ValueError("Ground truth and estimate have no common time interval")

    gt_mask = (gt_t >= start) & (gt_t <= end)
    est_mask = (est_t >= start) & (est_t <= end)
    gt_t, gt_p = gt_t[gt_mask], gt_p[gt_mask]
    est_t, est_p = est_t[est_mask], est_p[est_mask]
    if len(gt_t) < 2 or len(est_t) < 2:
        raise ValueError("Too few samples in the common time interval")
    return gt_t, gt_p, est_t, est_p


def _ate_at_estimate_times(
    gt_t: np.ndarray,
    gt_p: np.ndarray,
    est_t: np.ndarray,
    est_p: np.ndarray,
) -> float:
    gt_interp = np.column_stack(
        [np.interp(est_t, gt_t, gt_p[:, axis]) for axis in range(3)]
    )
    return float(np.sqrt(np.mean(np.sum((est_p - gt_interp) ** 2, axis=1))))


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.7,
            "grid.linewidth": 0.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def render(
    run_dir: Path,
    sequence: str,
    dataset_dir: str | None,
    output_dir: Path,
    elevation: float,
    azimuth: float,
    show_ate: bool,
) -> list[Path]:
    estimate_path = run_dir / sequence / "stamped_traj_network.txt"
    gt_path = _dataset_csv(run_dir, sequence, dataset_dir)
    if not estimate_path.is_file():
        raise FileNotFoundError(f"Missing LepidIO trajectory: {estimate_path}")
    if not gt_path.is_file():
        raise FileNotFoundError(f"Missing ground-truth CSV: {gt_path}")

    est_t, est_p = _load_estimate(estimate_path)
    gt_t, gt_p = _load_ground_truth(gt_path)
    gt_t, gt_p, est_t, est_p = _crop_to_overlap(gt_t, gt_p, est_t, est_p)
    ate = _ate_at_estimate_times(gt_t, gt_p, est_t, est_p)

    _configure_style()
    fig = plt.figure(figsize=(6.9, 3.75), constrained_layout=False)
    ax = fig.add_subplot(111, projection="3d")

    gt_color = "#1455D9"
    estimate_color = "#E31A1C"
    gt_line, = ax.plot(*gt_p.T, color=gt_color, linewidth=2.0, label="Ground Truth")
    estimate_line, = ax.plot(
        *est_p.T, color=estimate_color, linewidth=1.8, label="LepidIO"
    )

    ax.scatter(
        *gt_p[0], marker="*", s=78, c="black", edgecolors="white",
        linewidths=0.45, depthshade=False, zorder=10,
    )
    ax.scatter(
        *gt_p[-1], marker="*", s=72, c=gt_color, edgecolors="black",
        linewidths=0.55, depthshade=False, zorder=10,
    )
    ax.scatter(
        *est_p[-1], marker="*", s=72, c=estimate_color, edgecolors="black",
        linewidths=0.55, depthshade=False, zorder=10,
    )

    all_positions = np.vstack((gt_p, est_p))
    lower = np.nanmin(all_positions, axis=0)
    upper = np.nanmax(all_positions, axis=0)
    span = np.maximum(upper - lower, 1e-3)
    margin = np.maximum(0.07 * span, 0.08)
    ax.set_xlim(lower[0] - margin[0], upper[0] + margin[0])
    ax.set_ylim(lower[1] - margin[1], upper[1] + margin[1])
    ax.set_zlim(lower[2] - margin[2], upper[2] + margin[2])
    ax.set_box_aspect((1.75, 1.25, 0.58))
    ax.view_init(elev=elevation, azim=azimuth)

    ax.set_xlabel(r"$X$ [m]", labelpad=3)
    ax.set_ylabel(r"$Y$ [m]", labelpad=3)
    ax.set_zlabel(r"$Z$ [m]", labelpad=3)
    ax.tick_params(axis="both", which="major", labelsize=8, pad=0)
    ax.grid(True, color="#3B3939", alpha=0.72)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.97, 0.97, 0.97, 0.72))
        axis.pane.set_edgecolor("#3B3939")

    start_handle = Line2D(
        [], [], linestyle="none", marker="*", markersize=9,
        markerfacecolor="black", markeredgecolor="black",
    )
    # Draw a larger red star first and a smaller blue star on top. HandlerTuple
    # with one division places both artists at the same legend position.
    end_red_handle = Line2D(
        [], [], linestyle="none", marker="*", markersize=10,
        markerfacecolor=estimate_color, markeredgecolor="black",
        markeredgewidth=0.7,
    )
    end_blue_handle = Line2D(
        [], [], linestyle="none", marker="*", markersize=7,
        markerfacecolor=gt_color, markeredgecolor="black",
        markeredgewidth=0.6,
    )
    handles = [gt_line, estimate_line, start_handle,
               (end_red_handle, end_blue_handle)]
    labels = ["Ground Truth", "LepidIO", "Start Point", "End Points"]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.51, 0.015),
        ncol=4,
        frameon=False,
        handlelength=2.5,
        columnspacing=1.1,
        handletextpad=0.45,
        handler_map={tuple: HandlerTuple(ndivide=1, pad=0)},
    )
    if show_ate:
        ax.text2D(
            0.02, 0.96, f"ATE = {ate:.2f} m", transform=ax.transAxes,
            ha="left", va="top", fontsize=8.5,
        )

    fig.subplots_adjust(left=0.0, right=0.97, top=0.99, bottom=0.18)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / run_dir.name / f"lepidnet_trajectory_{sequence}"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = [prefix.with_suffix(extension) for extension in (".pdf", ".svg", ".png")]
    for output in outputs:
        save_options = {"bbox_inches": "tight", "pad_inches": 0.02}
        if output.suffix == ".png":
            save_options["dpi"] = 450
        fig.savefig(output, **save_options)
    plt.close(fig)

    interactive = go.Figure()
    for positions, color, label in (
        (gt_p, gt_color, "Ground Truth"), (est_p, estimate_color, "LepidIO")
    ):
        interactive.add_trace(go.Scatter3d(
            x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
            mode="lines", name=label, line=dict(color=color, width=5)))
        interactive.add_trace(go.Scatter3d(
            x=[positions[0, 0]], y=[positions[0, 1]], z=[positions[0, 2]],
            mode="markers", name=f"{label} start",
            marker=dict(color=color, size=6, symbol="circle")))
        # Scatter3d has no star marker; a text glyph remains crisp while zooming.
        interactive.add_trace(go.Scatter3d(
            x=[positions[-1, 0]], y=[positions[-1, 1]], z=[positions[-1, 2]],
            mode="text", name=f"{label} end", text=["★"],
            textfont=dict(color=color, size=18)))
    elev, azim = np.deg2rad([elevation, azimuth])
    interactive.update_layout(
        template="plotly_white", margin=dict(l=10, r=10, t=25, b=70),
        font=dict(family="Times New Roman", size=16),
        legend=dict(orientation="h", y=-0.08),
        scene=dict(xaxis_title="X [m]", yaxis_title="Y [m]", zaxis_title="Z [m]",
                   aspectmode="manual", aspectratio=dict(x=1.75, y=1.25, z=0.58),
                   camera=dict(eye=dict(x=float(2*np.cos(elev)*np.cos(azim)),
                                        y=float(2*np.cos(elev)*np.sin(azim)),
                                        z=float(2*np.sin(elev)))), dragmode="orbit"))
    html_path = prefix.with_suffix(".html")
    interactive.write_html(
        html_path, include_plotlyjs=True, full_html=True,
        config=dict(scrollZoom=True, displaylogo=False,
                    toImageButtonOptions=dict(format="png", filename=prefix.name,
                                              width=1800, height=1100, scale=2)))
    outputs.append(html_path)

    # Keep the interpretation of network accumulation alongside the exported plot.
    metadata_path = run_dir / sequence / "network_trajectory_metrics.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    provenance = dict(run=str(run_dir), sequence=sequence, estimate=str(estimate_path),
                      ground_truth=str(gt_path), network_metadata=metadata,
                      alignment="none", raw_ate_m=ate)
    prefix.with_suffix(".json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Sequence : {sequence}")
    print(f"Run      : {run_dir}")
    print(f"GT       : {gt_path}")
    print(f"Estimate : {estimate_path}")
    print(f"Raw ATE  : {ate:.3f} m (timestamp interpolation only)")
    for output in outputs:
        print(f"Saved    : {output}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LepidNet trajectories (PDF/SVG/PNG and interactive HTML)."
    )
    parser.add_argument("--sequence", required=True, help="Held-out sequence name")
    parser.add_argument(
        "--run-dir",
        default=str(DEFAULT_RUN_DIR),
        help="Completed LepidIO result directory",
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Dataset CSV directory; defaults to data.csv_dir in config_used.json",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for PDF, SVG, and PNG outputs",
    )
    parser.add_argument("--elevation", type=float, default=48.0)
    parser.add_argument("--azimuth", type=float, default=-58.0)
    parser.add_argument(
        "--show-ate", action="store_true", help="Print ATE inside the figure"
    )
    args = parser.parse_args()

    render(
        run_dir=_resolve_from_repo(args.run_dir),
        sequence=args.sequence,
        dataset_dir=args.dataset_dir,
        output_dir=_resolve_from_repo(args.output_dir),
        elevation=args.elevation,
        azimuth=args.azimuth,
        show_ate=args.show_ate,
    )


if __name__ == "__main__":
    main()
