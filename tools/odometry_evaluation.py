#!/usr/bin/env python3
"""Paper-grade Lepid-IO network and trajectory evaluation.

The module deliberately keeps the trajectory metric core NumPy-only so the
protocol can be unit-tested without PyTorch or a display.  Network replay and
plot rendering import their heavier dependencies only when requested.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


METRIC_SPEC_VERSION = "lepid_odometry_v1"
RELATIVE_WINDOWS_S = (1.0, 5.0)
NETWORK_WINDOW_S = 0.6  # Legacy fallback for checkpoints without window metadata.
NETWORK_STEP_S = 0.05
MIN_COVERAGE = 0.95
MAX_GT_GAP_FACTOR = 2.5
SEQUENCE_METRICS_FILE = "evaluation_metrics.json"


def evaluation_protocol(network_window_s: float = NETWORK_WINDOW_S,
                        network_step_s: float = NETWORK_STEP_S) -> dict:
    return {
        "metric_spec_version": METRIC_SPEC_VERSION,
        "quaternion_order": "xyzw",
        "timestamp_association": "reference position interpolation and quaternion SLERP",
        "timestamp_offset_optimization": False,
        "minimum_coverage": MIN_COVERAGE,
        "maximum_gt_interpolation_gap": (
            f"{MAX_GT_GAP_FACTOR} times the median positive GT sampling interval"
        ),
        "ate_primary": "raw common-world-frame translation RMSE",
        "ate_alignment_diagnostic": "Horn SE(3) rigid alignment without scale",
        "relative_windows_s": list(RELATIVE_WINDOWS_S),
        "rte": "translation RMSE of SE(3) relative-pose error",
        "rre": "SO(3) geodesic RMSE of relative-pose error",
        "drift_percent": "final raw translation error / GT path length * 100",
        "attitude_primary": "SO(3) geodesic angle",
        "euler_error": "ZYX/321 decomposition of R_gt^-1 R_est, reported as roll/pitch/yaw",
        "network_window_s": float(network_window_s),
        "network_step_s": float(network_step_s),
        "aggregate": "unweighted macro statistics across sequences",
    }


def _validated_window_contract(source: str, sampling_frequency, window_time,
                               window_size) -> dict:
    sampling_frequency = int(sampling_frequency)
    window_time = float(window_time)
    window_size = int(window_size)
    if sampling_frequency <= 0 or window_time <= 0 or window_size <= 1:
        raise ValueError(f"invalid network window metadata from {source}")
    intervals = window_time * sampling_frequency
    if not math.isclose(intervals, round(intervals), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"window_time={window_time} from {source} cannot be represented at "
            f"{sampling_frequency} Hz"
        )
    expected_size = int(round(intervals)) + 1
    if window_size != expected_size:
        raise ValueError(
            f"inconsistent network window metadata from {source}: "
            f"window_time={window_time}, sampling_frequency={sampling_frequency}, "
            f"window_size={window_size}, expected={expected_size}"
        )
    return {
        "sampling_frequency": sampling_frequency,
        "window_time": window_time,
        "window_size": window_size,
        "source": source,
    }


def resolve_network_window(checkpoint: Path | None,
                           config_path: Path | None) -> dict:
    """Resolve the trained model's input window and reject stale metadata.

    The checkpoint-adjacent ``model_param.json`` is authoritative because it is
    written together with the trained weights.  The run config is used as a
    fallback and, when both exist, as a consistency check.
    """
    candidates = []
    model_param_path = checkpoint.parent / "model_param.json" if checkpoint else None
    if model_param_path and model_param_path.is_file():
        model_param = json.loads(model_param_path.read_text(encoding="utf-8"))
        candidates.append(_validated_window_contract(
            str(model_param_path),
            model_param["sampling_freq"],
            model_param["window_time"],
            model_param["window_size"],
        ))

    if config_path and config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        data = config.get("data", {})
        network_param = config.get("network_param", {})
        network_values = (
            network_param.get("sampling_freq"),
            network_param.get("window_time"),
            network_param.get("window_size"),
        )
        if network_values[0] is not None and network_values[1] is not None \
                and network_values[2] is None:
            network_values = (
                network_values[0], network_values[1],
                int(round(float(network_values[1]) * int(network_values[0]))) + 1,
            )
        if None not in network_values:
            candidates.append(_validated_window_contract(
                f"{config_path}:network_param", *network_values,
            ))
        data_values = (
            data.get("imu_freq_net"), data.get("window_time"),
            data.get("window_size"),
        )
        if data_values[0] is not None and data_values[1] is not None \
                and data_values[2] is None:
            data_values = (
                data_values[0], data_values[1],
                int(round(float(data_values[1]) * int(data_values[0]))) + 1,
            )
        if None not in data_values:
            candidates.append(_validated_window_contract(
                f"{config_path}:data", *data_values,
            ))

    if not candidates:
        sampling_frequency = 100
        return _validated_window_contract(
            "legacy_default", sampling_frequency, NETWORK_WINDOW_S,
            int(round(NETWORK_WINDOW_S * sampling_frequency)) + 1,
        )

    selected = candidates[0]
    for candidate in candidates[1:]:
        comparable = ("sampling_frequency", "window_time", "window_size")
        if any(candidate[key] != selected[key] for key in comparable):
            raise ValueError(
                "network window mismatch between saved model metadata and run "
                f"config: {selected} vs {candidate}"
            )
    return selected


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(value), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten_metrics(value: dict, prefix: str = "") -> dict:
    flat = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict):
            flat.update(flatten_metrics(item, name))
        elif isinstance(item, (str, int, float, bool)) or item is None:
            flat[name] = item
    return flat


def write_flat_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fields} for row in rows])


def _load_named_csv(path: Path) -> dict[str, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    if data.size == 0 or data.dtype.names is None:
        raise ValueError(f"empty or invalid CSV: {path}")
    data = np.atleast_1d(data)
    return {name: np.asarray(data[name], dtype=np.float64) for name in data.dtype.names}


def load_ground_truth(path: Path) -> dict[str, np.ndarray]:
    data = _load_named_csv(path)
    required = (
        "timestamp_us", "gt_px", "gt_py", "gt_pz",
        "gt_qx", "gt_qy", "gt_qz", "gt_qw",
    )
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"GT CSV missing columns {missing}: {path}")
    timestamps = data["timestamp_us"] * 1e-6
    positions = np.column_stack([data["gt_px"], data["gt_py"], data["gt_pz"]])
    quaternions = normalize_quaternions(np.column_stack([
        data["gt_qx"], data["gt_qy"], data["gt_qz"], data["gt_qw"],
    ]))
    if len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"GT timestamps must be strictly increasing: {path}")
    return {
        "timestamps": timestamps,
        "positions": positions,
        "quaternions": quaternions,
    }


def load_estimated_trajectory(path: Path) -> dict[str, np.ndarray]:
    data = np.loadtxt(path, comments="#", dtype=np.float64)
    data = np.atleast_2d(data)
    if data.shape[1] < 8:
        raise ValueError(f"trajectory requires ts xyz qx qy qz qw: {path}")
    if len(data) < 2 or np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError(f"trajectory timestamps must be strictly increasing: {path}")
    return {
        "timestamps": data[:, 0],
        "positions": data[:, 1:4],
        "quaternions": normalize_quaternions(data[:, 4:8]),
    }


def normalize_quaternions(quaternions: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternions, dtype=np.float64)
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("zero-norm quaternion")
    return q / norms


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    result = np.array(q, dtype=np.float64, copy=True)
    result[..., :3] *= -1.0
    return result


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    l = np.asarray(left, dtype=np.float64)
    r = np.asarray(right, dtype=np.float64)
    lv, lw = l[..., :3], l[..., 3:4]
    rv, rw = r[..., :3], r[..., 3:4]
    vector = lw * rv + rw * lv + np.cross(lv, rv)
    scalar = lw * rw - np.sum(lv * rv, axis=-1, keepdims=True)
    return normalize_quaternions(np.concatenate([vector, scalar], axis=-1))


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    q = normalize_quaternions(q)
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], axis=-1).reshape(q.shape[:-1] + (3, 3))


def matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    matrices = np.asarray(matrix, dtype=np.float64)
    single = matrices.ndim == 2
    matrices = matrices.reshape((-1, 3, 3))
    result = []
    for m in matrices:
        trace = np.trace(m)
        if trace > 0:
            s = math.sqrt(trace + 1.0) * 2
            q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
                 (m[1, 0] - m[0, 1]) / s, 0.25 * s]
        else:
            index = int(np.argmax(np.diag(m)))
            if index == 0:
                s = math.sqrt(max(0.0, 1 + m[0, 0] - m[1, 1] - m[2, 2])) * 2
                q = [0.25 * s, (m[0, 1] + m[1, 0]) / s,
                     (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
            elif index == 1:
                s = math.sqrt(max(0.0, 1 + m[1, 1] - m[0, 0] - m[2, 2])) * 2
                q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s,
                     (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
            else:
                s = math.sqrt(max(0.0, 1 + m[2, 2] - m[0, 0] - m[1, 1])) * 2
                q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s,
                     0.25 * s, (m[1, 0] - m[0, 1]) / s]
        result.append(q)
    array = normalize_quaternions(np.asarray(result))
    return array[0] if single else array.reshape(matrix.shape[:-2] + (4,))


def slerp(timestamps: np.ndarray, quaternions: np.ndarray,
          query: np.ndarray) -> np.ndarray:
    timestamps = np.asarray(timestamps)
    query = np.asarray(query)
    upper = np.searchsorted(timestamps, query, side="right")
    upper = np.clip(upper, 1, len(timestamps) - 1)
    lower = upper - 1
    span = timestamps[upper] - timestamps[lower]
    alpha = ((query - timestamps[lower]) / span)[:, None]
    q0 = quaternions[lower]
    q1 = quaternions[upper].copy()
    dots = np.sum(q0 * q1, axis=1, keepdims=True)
    q1[dots[:, 0] < 0] *= -1
    dots = np.clip(np.abs(dots), 0.0, 1.0)
    omega = np.arccos(dots)
    sin_omega = np.sin(omega)
    near = sin_omega[:, 0] < 1e-8
    result = np.empty_like(q0)
    result[near] = (1 - alpha[near]) * q0[near] + alpha[near] * q1[near]
    regular = ~near
    result[regular] = (
        np.sin((1 - alpha[regular]) * omega[regular]) / sin_omega[regular] * q0[regular]
        + np.sin(alpha[regular] * omega[regular]) / sin_omega[regular] * q1[regular]
    )
    return normalize_quaternions(result)


def interpolate_vectors(timestamps: np.ndarray, vectors: np.ndarray,
                        query: np.ndarray) -> np.ndarray:
    return np.column_stack([
        np.interp(query, timestamps, vectors[:, axis]) for axis in range(vectors.shape[1])
    ])


def interpolate_pose(trajectory: dict[str, np.ndarray], query: np.ndarray):
    return (
        interpolate_vectors(trajectory["timestamps"], trajectory["positions"], query),
        slerp(trajectory["timestamps"], trajectory["quaternions"], query),
    )


def association_mask(reference_timestamps: np.ndarray,
                     query_timestamps: np.ndarray) -> tuple[np.ndarray, float]:
    """Reject out-of-range samples and interpolation across GT outages."""
    reference_timestamps = np.asarray(reference_timestamps, dtype=np.float64)
    query_timestamps = np.asarray(query_timestamps, dtype=np.float64)
    positive_steps = np.diff(reference_timestamps)
    positive_steps = positive_steps[positive_steps > 0]
    if not len(positive_steps):
        raise ValueError("reference timestamps have no positive sampling interval")
    maximum_gap = float(np.median(positive_steps) * MAX_GT_GAP_FACTOR)
    in_range = ((query_timestamps >= reference_timestamps[0])
                & (query_timestamps <= reference_timestamps[-1]))
    upper = np.searchsorted(reference_timestamps, query_timestamps, side="right")
    upper = np.clip(upper, 1, len(reference_timestamps) - 1)
    lower = upper - 1
    bracket_gap = reference_timestamps[upper] - reference_timestamps[lower]
    exact = np.isclose(query_timestamps, reference_timestamps[lower], atol=1e-9)
    exact |= np.isclose(query_timestamps, reference_timestamps[upper], atol=1e-9)
    return in_range & (exact | (bracket_gap <= maximum_gap + 1e-12)), maximum_gap


def rotation_error_quaternion(gt_q: np.ndarray, est_q: np.ndarray) -> np.ndarray:
    return quaternion_multiply(quaternion_conjugate(gt_q), est_q)


def geodesic_degrees(error_q: np.ndarray) -> np.ndarray:
    q = normalize_quaternions(error_q)
    return np.degrees(2.0 * np.arccos(np.clip(np.abs(q[..., 3]), 0.0, 1.0)))


def error_euler_rpy_degrees(error_q: np.ndarray) -> np.ndarray:
    matrices = quaternion_to_matrix(error_q)
    pitch = np.arcsin(np.clip(-matrices[..., 2, 0], -1.0, 1.0))
    roll = np.arctan2(matrices[..., 2, 1], matrices[..., 2, 2])
    yaw = np.arctan2(matrices[..., 1, 0], matrices[..., 0, 0])
    return np.degrees(np.stack([roll, pitch, yaw], axis=-1))


def _norm_stats(errors: np.ndarray) -> dict:
    values = np.linalg.norm(errors, axis=1) if errors.ndim == 2 else np.abs(errors)
    return {
        "rmse": float(np.sqrt(np.mean(values ** 2))),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p50": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "final": float(values[-1]),
    }


def _axis_metrics(errors: np.ndarray, names=("x", "y", "z")) -> dict:
    return {
        name: {
            "rmse": float(np.sqrt(np.mean(errors[:, index] ** 2))),
            "mae": float(np.mean(np.abs(errors[:, index]))),
            "bias": float(np.mean(errors[:, index])),
            "p95_abs": float(np.percentile(np.abs(errors[:, index]), 95)),
            "p95": float(np.percentile(np.abs(errors[:, index]), 95)),
            "final": float(errors[-1, index]),
        }
        for index, name in enumerate(names)
    }


def horn_align(estimated: np.ndarray, ground_truth: np.ndarray):
    est_mean = estimated.mean(axis=0)
    gt_mean = ground_truth.mean(axis=0)
    covariance = (estimated - est_mean).T @ (ground_truth - gt_mean)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = gt_mean - rotation @ est_mean
    return rotation, translation


def relative_pose_metrics(timestamps: np.ndarray, gt_pos: np.ndarray,
                          gt_q: np.ndarray, est_pos: np.ndarray,
                          est_q: np.ndarray, delta_s: float) -> dict:
    target = timestamps + delta_s
    valid = target <= timestamps[-1] + 1e-9
    if np.count_nonzero(valid) < 2:
        return {"available": False, "sample_count": int(np.count_nonzero(valid))}
    starts = np.flatnonzero(valid)
    targets = target[valid]
    est_target_pos = interpolate_vectors(timestamps, est_pos, targets)
    gt_target_pos = interpolate_vectors(timestamps, gt_pos, targets)
    est_target_q = slerp(timestamps, est_q, targets)
    gt_target_q = slerp(timestamps, gt_q, targets)
    est_r = quaternion_to_matrix(est_q[starts])
    gt_r = quaternion_to_matrix(gt_q[starts])
    est_dp = np.einsum("nij,nj->ni", np.swapaxes(est_r, 1, 2),
                       est_target_pos - est_pos[starts])
    gt_dp = np.einsum("nij,nj->ni", np.swapaxes(gt_r, 1, 2),
                      gt_target_pos - gt_pos[starts])
    translation_error = est_dp - gt_dp
    est_rel_q = quaternion_multiply(quaternion_conjugate(est_q[starts]), est_target_q)
    gt_rel_q = quaternion_multiply(quaternion_conjugate(gt_q[starts]), gt_target_q)
    rotation_error = geodesic_degrees(
        quaternion_multiply(quaternion_conjugate(gt_rel_q), est_rel_q)
    )
    return {
        "available": True,
        "sample_count": int(len(starts)),
        "translation_rmse_m": float(np.sqrt(np.mean(np.sum(translation_error ** 2, axis=1)))),
        "translation_p95_m": float(np.percentile(np.linalg.norm(translation_error, axis=1), 95)),
        "rotation_rmse_deg": float(np.sqrt(np.mean(rotation_error ** 2))),
        "rotation_p95_deg": float(np.percentile(rotation_error, 95)),
    }


def evaluate_trajectory(
    gt: dict[str, np.ndarray], estimated: dict[str, np.ndarray]
):
    est_ts_all = estimated["timestamps"]
    valid, maximum_gt_gap = association_mask(gt["timestamps"], est_ts_all)
    coverage = float(np.mean(valid))
    if coverage < MIN_COVERAGE:
        raise ValueError(f"trajectory/GT timestamp coverage {coverage:.3f} is below {MIN_COVERAGE:.2f}")
    timestamps = est_ts_all[valid]
    est_pos = estimated["positions"][valid]
    est_q = estimated["quaternions"][valid]
    gt_pos = interpolate_vectors(gt["timestamps"], gt["positions"], timestamps)
    gt_q = slerp(gt["timestamps"], gt["quaternions"], timestamps)
    position_error = est_pos - gt_pos
    attitude_error_q = rotation_error_quaternion(gt_q, est_q)
    attitude_angle = geodesic_degrees(attitude_error_q)
    attitude_rpy = error_euler_rpy_degrees(attitude_error_q)
    path_length = float(np.linalg.norm(np.diff(gt_pos, axis=0), axis=1).sum())
    duration = float(timestamps[-1] - timestamps[0])

    align_r, align_t = horn_align(est_pos, gt_pos)
    aligned_pos = (align_r @ est_pos.T).T + align_t
    align_q = matrix_to_quaternion(align_r)
    aligned_q = quaternion_multiply(np.broadcast_to(align_q, est_q.shape), est_q)
    aligned_attitude = geodesic_degrees(rotation_error_quaternion(gt_q, aligned_q))

    relative = {
        f"{int(window)}s": relative_pose_metrics(
            timestamps, gt_pos, gt_q, est_pos, est_q, window
        ) for window in RELATIVE_WINDOWS_S
    }
    raw_stats = _norm_stats(position_error)
    aligned_error = aligned_pos - gt_pos
    attitude_stats = _norm_stats(attitude_angle)
    yaw_unwrapped = np.unwrap(np.radians(attitude_rpy[:, 2]))
    yaw_drift = (
        float(np.degrees(yaw_unwrapped[-1] - yaw_unwrapped[0]) / (duration / 60.0))
        if duration > 0 else None
    )
    metrics = {
        "coverage": coverage,
        "maximum_gt_interpolation_gap_s": maximum_gt_gap,
        "sample_count": int(len(timestamps)),
        "duration_s": duration,
        "gt_path_length_m": path_length,
        "position": {
            "ate_raw_m": raw_stats,
            "axis": _axis_metrics(position_error),
            "ate_se3_aligned_m": _norm_stats(aligned_error),
            "final_drift_percent": (
                float(raw_stats["final"] / path_length * 100.0) if path_length > 1e-12 else None
            ),
        },
        "relative_pose": relative,
        "attitude": {
            "so3_geodesic_deg": attitude_stats,
            "rpy_error_deg": _axis_metrics(attitude_rpy, names=("roll", "pitch", "yaw")),
            "se3_aligned_so3_rmse_deg": float(np.sqrt(np.mean(aligned_attitude ** 2))),
            "yaw_drift_deg_per_min": yaw_drift,
        },
    }

    timeseries = {
        "timestamp_s": timestamps,
        "gt_x": gt_pos[:, 0], "gt_y": gt_pos[:, 1], "gt_z": gt_pos[:, 2],
        "est_x": est_pos[:, 0], "est_y": est_pos[:, 1], "est_z": est_pos[:, 2],
        "error_x": position_error[:, 0], "error_y": position_error[:, 1],
        "error_z": position_error[:, 2],
        "position_error_norm_m": np.linalg.norm(position_error, axis=1),
        "attitude_error_deg": attitude_angle,
        "roll_error_deg": attitude_rpy[:, 0], "pitch_error_deg": attitude_rpy[:, 1],
        "yaw_error_deg": attitude_rpy[:, 2],
    }
    aligned = {"positions": aligned_pos, "quaternions": aligned_q}
    return metrics, timeseries, aligned


def evaluate_network(checkpoint: Path, gt_path: Path, config_path: Path | None,
                     source: str, device_name: str = "auto",
                     window_contract: dict | None = None):
    repo_root = Path(__file__).resolve().parent.parent
    src_root = repo_root / "src"
    for path in (str(src_root), str(repo_root / "tools")):
        if path not in sys.path:
            sys.path.insert(0, path)
    import torch
    from learning.dataset import LepidDataset
    from learning.network.model_factory import load_model_from_ckpt, resolve_device

    contract = window_contract or resolve_network_window(checkpoint, config_path)
    sampling_frequency = int(contract["sampling_frequency"])
    window_time = float(contract["window_time"])
    window_size = int(contract["window_size"])
    stride = max(1, int(round(NETWORK_STEP_S * sampling_frequency)))
    dataset = LepidDataset(
        str(gt_path), window_size=window_size, stride=stride,
        sampling_freq=sampling_frequency,
    )
    device = resolve_device(device_name)
    model = load_model_from_ckpt(str(checkpoint), device)
    predictions, targets, begins, ends = [], [], [], []
    batch_size = 256
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            items = [dataset[index] for index in range(start, min(len(dataset), start + batch_size))]
            features = torch.stack([
                torch.from_numpy(item[0]) for item in items
            ]).to(device)
            output = model(features)["delta_p"].detach().cpu().numpy()
            predictions.append(output)
            targets.append(np.stack([np.asarray(item[1]) for item in items]))
            begins.extend(item[2] for item in items)
            ends.extend(item[3] for item in items)
    pred = np.concatenate(predictions)
    target = np.concatenate(targets)
    errors = pred - target
    metrics = {
        "available": True,
        "source": source,
        "sample_count": int(len(errors)),
        "window_s": float(window_time),
        "step_s": float(stride / sampling_frequency),
        "window_parameter_source": contract["source"],
        "axis": _axis_metrics(errors),
        "norm_m": _norm_stats(errors),
    }
    timeseries = {
        "timestamp_begin_us": np.asarray(begins),
        "timestamp_end_us": np.asarray(ends),
        "pred_x": pred[:, 0], "pred_y": pred[:, 1], "pred_z": pred[:, 2],
        "gt_x": target[:, 0], "gt_y": target[:, 1], "gt_z": target[:, 2],
        "error_x": errors[:, 0], "error_y": errors[:, 1], "error_z": errors[:, 2],
        "error_norm_m": np.linalg.norm(errors, axis=1),
    }
    return metrics, timeseries


def write_timeseries(path: Path, columns: dict[str, np.ndarray]) -> None:
    names = list(columns)
    values = np.column_stack([columns[name] for name in names])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, values, delimiter=",", header=",".join(names), comments="", fmt="%.12g")


def render_plots(out_dir: Path, trajectory_series: dict, network_series: dict | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = trajectory_series["timestamp_s"]
    elapsed = t - t[0]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes[0, 0].plot(elapsed, trajectory_series["position_error_norm_m"])
    axes[0, 0].set(title="Position error", xlabel="Time [s]", ylabel="Error [m]")
    for axis, name in zip(("x", "y", "z"), ("X", "Y", "Z")):
        axes[0, 1].plot(elapsed, trajectory_series[f"error_{axis}"], label=name)
    axes[0, 1].set(title="Position error by axis", xlabel="Time [s]", ylabel="Error [m]")
    axes[0, 1].legend()
    axes[1, 0].plot(elapsed, trajectory_series["attitude_error_deg"])
    axes[1, 0].set(title="SO(3) attitude error", xlabel="Time [s]", ylabel="Error [deg]")
    for axis, name in zip(("roll", "pitch", "yaw"), ("Roll", "Pitch", "Yaw")):
        axes[1, 1].plot(elapsed, trajectory_series[f"{axis}_error_deg"], label=name)
    axes[1, 1].set(title="ZYX attitude error", xlabel="Time [s]", ylabel="Error [deg]")
    axes[1, 1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "trajectory_error_plots.png", dpi=160)
    plt.close(fig)

    if network_series is None:
        return
    nt = (network_series["timestamp_end_us"] - network_series["timestamp_end_us"][0]) * 1e-6
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(nt, network_series["error_norm_m"])
    axes[0].set(title="Network displacement error", xlabel="Time [s]", ylabel="Error [m]")
    for axis, name in zip(("x", "y", "z"), ("X", "Y", "Z")):
        axes[1].plot(nt, network_series[f"error_{axis}"], label=name)
    axes[1].set(title="Network error by axis", xlabel="Time [s]", ylabel="Error [m]")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "network_error_plots.png", dpi=160)
    plt.close(fig)


def evaluate_sequence(args) -> dict:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config) if args.config else None
    gt = load_ground_truth(Path(args.gt))
    estimated = load_estimated_trajectory(Path(args.traj))
    trajectory_metrics, trajectory_series, _ = evaluate_trajectory(gt, estimated)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    window_contract = resolve_network_window(checkpoint_path, config_path)
    network_metrics = {"available": False, "reason": "checkpoint not provided"}
    network_series = None
    if args.checkpoint:
        network_metrics, network_series = evaluate_network(
            checkpoint_path, Path(args.gt), config_path,
            args.network_source, args.device, window_contract,
        )
    model_param_path = (
        checkpoint_path.parent / "model_param.json" if checkpoint_path else None
    )
    metadata_sources = {
        "trajectory": str(Path(args.traj)),
        "ground_truth": str(Path(args.gt)),
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_sha256": (
            sha256_file(checkpoint_path) if checkpoint_path and checkpoint_path.is_file() else None
        ),
        "run_config": str(config_path) if config_path and config_path.is_file() else None,
        "model_param": (
            str(model_param_path) if model_param_path and model_param_path.is_file() else None
        ),
        "fallback": (
            "legacy 0.6 s network window; saved metadata was unavailable"
            if window_contract["source"] == "legacy_default" else None
        ),
        "network_window_source": window_contract["source"],
    }
    metrics = {
        "metric_spec_version": METRIC_SPEC_VERSION,
        "evaluation_protocol": evaluation_protocol(
            window_contract["window_time"], NETWORK_STEP_S
        ),
        "sequence": args.sequence,
        "inference_mode": {"name": "learned_displacement_accumulation"},
        "metadata_sources": metadata_sources,
        "trajectory": trajectory_metrics,
        "network": network_metrics,
    }
    write_json(out_dir / SEQUENCE_METRICS_FILE, metrics)
    write_flat_csv(out_dir / "evaluation_metrics.csv", [flatten_metrics(metrics)])
    write_timeseries(out_dir / "trajectory_error_timeseries.csv", trajectory_series)
    if network_series is not None:
        write_timeseries(out_dir / "network_error_timeseries.csv", network_series)
    else:
        write_timeseries(out_dir / "network_error_timeseries.csv", {
            "timestamp_begin_us": np.array([], dtype=float),
            "timestamp_end_us": np.array([], dtype=float),
        })
    render_plots(out_dir, trajectory_series, network_series)
    if network_series is None:
        # Keep the artifact contract complete for trajectory-only backfills.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "Network metrics unavailable", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_dir / "network_error_plots.png", dpi=160)
        plt.close(fig)
    return metrics


def _numeric_flat(metrics: dict) -> dict[str, float]:
    return {
        key: float(value) for key, value in flatten_metrics(metrics).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(float(value))
    }


def summarize_run(run_dir: Path, sequences: Iterable[str]) -> dict:
    sequence_metrics = []
    for sequence in sequences:
        path = run_dir / sequence / SEQUENCE_METRICS_FILE
        if not path.is_file():
            raise FileNotFoundError(f"missing sequence metrics: {path}")
        sequence_metrics.append(json.loads(path.read_text(encoding="utf-8")))
    protocols = [
        metrics["evaluation_protocol"] for metrics in sequence_metrics
        if isinstance(metrics.get("evaluation_protocol"), dict)
    ]
    protocol = protocols[0] if protocols else evaluation_protocol()
    if any(item != protocol for item in protocols[1:]):
        raise ValueError("cannot summarize sequences evaluated with different protocols")
    numeric = [_numeric_flat(metrics) for metrics in sequence_metrics]
    common = sorted(set.intersection(*(set(row) for row in numeric))) if numeric else []
    aggregate = {"macro_mean": {}, "macro_median": {}, "macro_std": {}}
    for key in common:
        values = np.asarray([row[key] for row in numeric], dtype=float)
        aggregate["macro_mean"][key] = float(np.mean(values))
        aggregate["macro_median"][key] = float(np.median(values))
        aggregate["macro_std"][key] = float(np.std(values))
    summary = {
        "metric_spec_version": METRIC_SPEC_VERSION,
        "sequence_count": len(sequence_metrics),
        "sequences": sequence_metrics,
        "aggregate": aggregate,
    }
    write_json(run_dir / "evaluation_summary.json", summary)
    rows = []
    for metrics in sequence_metrics:
        row = flatten_metrics(metrics)
        row["row_type"] = "sequence"
        rows.append(row)
    for kind, values in aggregate.items():
        rows.append({"row_type": kind, **values})
    write_flat_csv(run_dir / "evaluation_summary.csv", rows)
    write_json(run_dir / "evaluation_protocol.json", protocol)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser("evaluate", help="evaluate one sequence")
    evaluate.add_argument("--sequence", required=True)
    evaluate.add_argument("--gt", required=True)
    evaluate.add_argument("--traj", required=True)
    evaluate.add_argument("--checkpoint")
    evaluate.add_argument("--config")
    evaluate.add_argument("--network-source", default="normal_test")
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--out-dir", required=True)
    summarize = subparsers.add_parser("summarize", help="aggregate sequence metrics")
    summarize.add_argument("--run-dir", required=True)
    summarize.add_argument("--sequences", nargs="+", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "evaluate":
        metrics = evaluate_sequence(args)
        print(json.dumps(_json_ready(metrics), ensure_ascii=False))
    else:
        summary = summarize_run(Path(args.run_dir), args.sequences)
        print(json.dumps(_json_ready(summary), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
