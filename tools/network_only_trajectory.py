#!/usr/bin/env python3
"""Build a dense world-frame trajectory from next-step network predictions.

The input window ends at frame k and the learned model predicts the body-frame
displacement from k to k+1.  Reconstruction therefore uses
``p_hat[k+1] = p_hat[k] + R_k @ delta_p_body``.

The first complete input window is copied from ground truth to provide the
initial position history.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
SRC_ROOT = REPO_ROOT / "src"
for import_path in (str(THIS_DIR), str(SRC_ROOT)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from common.columns import GT_P_COLS, GT_Q_COLS
from common.config_validation import validate_config
from learning.dataset import LepidDataset, load_csv, resample_for_network
from learning.network.model_factory import load_model_from_ckpt, resolve_device
from plot_trajectory_3d import render_trajectory


def build_network_trajectory(
    config_path: Path,
    checkpoint: Path,
    csv_path: Path,
    output_dir: Path,
    device_name: str = "auto",
) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config, require_training_splits=False)
    data_cfg = config["data"]
    sampling_frequency = int(data_cfg.get("imu_freq_net", 100))
    window_size = int(data_cfg["window_size"])
    window_intervals = window_size - 1
    inference_cfg = config.get("inference", {})
    output_frequency = int(
        inference_cfg.get("network_output_freq", sampling_frequency)
    )
    if output_frequency != sampling_frequency:
        raise ValueError(
            "next-step pure-network reconstruction requires "
            "inference.network_output_freq to equal "
            f"imu_freq_net={sampling_frequency}"
        )
    output_stride = 1

    frame = resample_for_network(load_csv(str(csv_path)), sampling_frequency)
    dataset = LepidDataset(
        str(csv_path),
        window_size=window_size,
        stride=output_stride,
        sampling_freq=sampling_frequency,
    )
    if len(dataset) == 0:
        raise ValueError(f"sequence is shorter than one network window: {csv_path}")

    device = resolve_device(device_name)
    model = load_model_from_ckpt(str(checkpoint), device)
    model.eval()

    timestamp_us = frame["timestamp_us"].to_numpy(dtype=np.int64)
    positions_gt = frame[GT_P_COLS].to_numpy(dtype=np.float64)
    quaternions = frame[GT_Q_COLS].to_numpy(dtype=np.float64)
    row_by_timestamp = {int(value): index for index, value in enumerate(timestamp_us)}

    # The first W samples (0 through W-1, inclusive) are GT warm-up anchors.
    # Subsequent positions are reconstructed at the requested output frequency.
    warmup_indices = list(range(0, window_size, output_stride))
    if warmup_indices[-1] != window_intervals:
        warmup_indices.append(window_intervals)
    reconstructed_by_index = {
        index: positions_gt[index].copy() for index in warmup_indices
    }
    trajectory_indices = list(warmup_indices)
    segment_rows = []

    inference_batch_size = max(
        1, int(inference_cfg.get("network_trajectory_batch_size", 256))
    )
    predictions = []
    with torch.no_grad():
        for batch_begin in range(0, len(dataset), inference_batch_size):
            batch_end = min(batch_begin + inference_batch_size, len(dataset))
            features = np.stack(
                [dataset[index][0] for index in range(batch_begin, batch_end)]
            )
            output = model(torch.from_numpy(features).to(device))["delta_p"]
            predictions.extend(
                output.detach().cpu().numpy().astype(np.float64)
            )

        for index, prediction in enumerate(predictions):
            _, target_body, begin_us, end_us = dataset[index]
            begin_us, end_us = int(begin_us), int(end_us)
            begin_index = row_by_timestamp[begin_us]
            end_index = row_by_timestamp[end_us]

            # The input window ends at begin_index; the prediction describes
            # exactly the next transition begin_index -> end_index.
            if end_index <= window_intervals:
                continue
            if begin_index not in reconstructed_by_index:
                raise RuntimeError(
                    f"missing reconstructed anchor at row {begin_index}; "
                    "check output frequency/window compatibility"
                )
            rotation = Rotation.from_quat(quaternions[begin_index])
            prediction_world = rotation.apply(prediction)
            reconstructed_position = (
                reconstructed_by_index[begin_index] + prediction_world
            )
            reconstructed_by_index[end_index] = reconstructed_position

            trajectory_indices.append(end_index)
            segment_rows.append({
                "ts_begin_us": begin_us,
                "ts_end_us": end_us,
                "dp_pred_body_x": prediction[0],
                "dp_pred_body_y": prediction[1],
                "dp_pred_body_z": prediction[2],
                "dp_gt_body_x": float(target_body[0]),
                "dp_gt_body_y": float(target_body[1]),
                "dp_gt_body_z": float(target_body[2]),
                "dp_pred_world_x": prediction_world[0],
                "dp_pred_world_y": prediction_world[1],
                "dp_pred_world_z": prediction_world[2],
                "position_x": reconstructed_position[0],
                "position_y": reconstructed_position[1],
                "position_z": reconstructed_position[2],
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_indices = np.asarray(trajectory_indices, dtype=np.int64)
    trajectory_timestamps = timestamp_us[trajectory_indices]
    trajectory_positions = np.asarray(
        [reconstructed_by_index[int(index)] for index in trajectory_indices],
        dtype=np.float64,
    )
    trajectory_quaternions = quaternions[trajectory_indices]
    tum = np.column_stack([
        trajectory_timestamps.astype(np.float64) * 1e-6,
        trajectory_positions,
        trajectory_quaternions,
    ])
    trajectory_path = output_dir / "stamped_traj_network.txt"
    np.savetxt(
        trajectory_path,
        tum,
        fmt="%.9f",
        header="ts x y z qx qy qz qw",
    )
    pd.DataFrame(segment_rows).to_csv(
        output_dir / "network_trajectory_segments.csv", index=False
    )

    gt_at_nodes = positions_gt[trajectory_indices]
    errors = trajectory_positions - gt_at_nodes
    error_norm = np.linalg.norm(errors, axis=1)
    prediction_mask = trajectory_indices > window_intervals
    prediction_errors = errors[prediction_mask]
    prediction_error_norm = error_norm[prediction_mask]
    if prediction_error_norm.size == 0:
        raise ValueError(
            "sequence must extend beyond the GT warm-up window to build a "
            "network-only trajectory"
        )
    metrics = {
        "method": "next_step_displacement_accumulation",
        "attitude_source": "recorded_csv_body_to_world_quaternion",
        "uses_gt_position_during_warmup": True,
        "uses_gt_position_after_warmup": False,
        "window_size": window_size,
        "window_time_s": float(data_cfg["window_time"]),
        "warmup_point_count": int(len(warmup_indices)),
        "network_output_frequency_hz": output_frequency,
        "network_output_stride_samples": output_stride,
        "segment_count": int(len(segment_rows)),
        "trajectory_point_count": int(len(trajectory_positions)),
        "position_rmse_m": float(np.sqrt(np.mean(prediction_error_norm ** 2))),
        "position_final_error_m": float(error_norm[-1]),
        "axis_rmse_m": {
            axis: float(np.sqrt(np.mean(prediction_errors[:, index] ** 2)))
            for index, axis in enumerate("xyz")
        },
    }
    (output_dir / "network_trajectory_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    render_trajectory(
        trajectory_path,
        csv_path,
        output_prefix=output_dir / "network_traj_3d",
        estimate_label="Network + recorded attitude",
    )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Accumulate next-step network Δp into a dense trajectory"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    metrics = build_network_trajectory(
        Path(args.config), Path(args.checkpoint), Path(args.csv),
        Path(args.out_dir), args.device,
    )
    print(json.dumps(metrics, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
