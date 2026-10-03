#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render an estimated trajectory against its ground-truth reference."""

import argparse

import matplotlib

# Set MPLBACKEND=Agg for headless rendering.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_estimated_traj(path):
    """Load an estimated trajectory in TUM format."""
    data = np.loadtxt(path, comments="#")
    return data[:, 0], data[:, 1:4]      # ts, xyz


def load_gt(path):
    """Load world-frame ground-truth positions from CSV."""
    df = pd.read_csv(path)
    ts = df["timestamp_us"].to_numpy() * 1e-6
    xyz = df[["gt_px", "gt_py", "gt_pz"]].to_numpy()
    return ts, xyz


def crop_to_common_time_range(gt_ts, gt_xyz, est_ts, est_xyz):
    """Crop both trajectories to their overlap and align exact GT endpoints."""
    if len(est_ts) == 0 or len(gt_ts) == 0:
        raise ValueError("estimated and reference trajectories must not be empty")

    overlap_start = max(float(est_ts[0]), float(gt_ts[0]))
    overlap_end = min(float(est_ts[-1]), float(gt_ts[-1]))
    if overlap_start > overlap_end:
        raise ValueError("estimated trajectory and reference have no timestamp overlap")

    est_mask = (est_ts >= overlap_start) & (est_ts <= overlap_end)
    cropped_est_ts = est_ts[est_mask]
    cropped_est_xyz = est_xyz[est_mask]
    if len(cropped_est_ts) == 0:
        raise ValueError("estimated trajectory has no samples inside reference coverage")

    # Use retained estimate timestamps as the exact plotting boundaries. This
    # also handles a final estimate sample just outside reference coverage
    # the entire plot stage.
    start, end = float(cropped_est_ts[0]), float(cropped_est_ts[-1])
    interior = (gt_ts > start) & (gt_ts < end)
    cropped_gt_ts = np.concatenate(([start], gt_ts[interior], [end]))
    cropped_gt_xyz = np.column_stack([
        np.interp(cropped_gt_ts, gt_ts, gt_xyz[:, axis]) for axis in range(3)
    ])
    return cropped_gt_ts, cropped_gt_xyz, cropped_est_ts, cropped_est_xyz


def render_trajectory(traj_path, gt_path, output_prefix=None, estimate_label="LepidIO"):
    """Render one estimated trajectory against GT and return output paths."""
    est_ts, est_xyz = load_estimated_traj(traj_path)
    gt_ts, gt_xyz = load_gt(gt_path)
    gt_ts, gt_xyz, est_ts, est_xyz = crop_to_common_time_range(
        gt_ts, gt_xyz, est_ts, est_xyz
    )

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2],
            "b-", linewidth=2, label="Ground Truth")
    ax.plot(est_xyz[:, 0], est_xyz[:, 1], est_xyz[:, 2],
            "r-", linewidth=2, label=estimate_label)

    ax.scatter(*gt_xyz[0], c="b", marker="o", s=60, label="GT start")
    ax.scatter(*est_xyz[0], c="r", marker="o", s=60, label=f"{estimate_label} start")
    ax.scatter(*gt_xyz[-1], c="b", marker="x", s=60, label="GT end")
    ax.scatter(*est_xyz[-1], c="r", marker="x", s=60, label=f"{estimate_label} end")

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title(f"{estimate_label} vs Ground Truth")
    ax.legend()

    # Use equal axis spans for an undistorted spatial comparison.
    all_xyz = np.vstack([gt_xyz, est_xyz])
    center = all_xyz.mean(axis=0)
    spread = np.abs(all_xyz - center).max() * 1.2
    ax.set_xlim(center[0] - spread, center[0] + spread)
    ax.set_ylim(center[1] - spread, center[1] + spread)
    ax.set_zlim(center[2] - spread, center[2] + spread)

    fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(14, 5))
    ax2.plot(gt_xyz[:, 0], gt_xyz[:, 1], "b-", label="GT")
    ax2.plot(est_xyz[:, 0], est_xyz[:, 1], "r-", label=estimate_label)
    ax2.set_xlabel("X [m]"); ax2.set_ylabel("Y [m]")
    ax2.set_title("XY plane (top view)")
    ax2.axis("equal"); ax2.legend()

    ax3.plot(gt_xyz[:, 0], gt_xyz[:, 2], "b-", label="GT")
    ax3.plot(est_xyz[:, 0], est_xyz[:, 2], "r-", label=estimate_label)
    ax3.set_xlabel("X [m]"); ax3.set_ylabel("Z [m]")
    ax3.set_title("XZ plane (side view)")
    ax3.axis("equal"); ax3.legend()

    if output_prefix is None:
        out = str(traj_path).replace("stamped_traj_estimate.txt", "traj_3d.png")
    else:
        out = f"{output_prefix}.png"
    projections_out = out.replace(".png", "_projections.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig2.savefig(projections_out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    print(f"Saved: {projections_out}")

    print(f"\nTrajectory points: Reference={len(gt_xyz)}, Estimate={len(est_xyz)}")

    # Close explicitly so non-interactive runs do not retain figure resources.
    if "agg" not in matplotlib.get_backend().lower():
        plt.show()
    plt.close(fig)
    plt.close(fig2)
    return out, projections_out


def main():
    parser = argparse.ArgumentParser(description="Compare 3D trajectories")
    parser.add_argument("--traj", required=True, help="estimated trajectory file")
    parser.add_argument("--gt", required=True, help="ground-truth CSV")
    parser.add_argument("--output-prefix", default=None, help="output path without extension")
    parser.add_argument("--estimate-label", default="LepidIO", help="estimate label in the plot")
    args = parser.parse_args()
    render_trajectory(
        args.traj, args.gt, args.output_prefix, args.estimate_label
    )


if __name__ == "__main__":
    main()
