"""CSV loading, resampling, windowing, and data-loader construction."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation, Slerp
from torch.utils.data import DataLoader, Dataset

from common.columns import (
    ACC_COLS, CSV_COLUMNS, GYR_COLS, GT_P_COLS, GT_Q_COLS,
    NETWORK_INPUT_COLUMNS, NET_INPUT_DIM,
    WING_DISPLACEMENT_COLS, WING_PHASE_COLS,
)


def load_csv(csv_path: str) -> pd.DataFrame:
    """Load one CSV file and validate its schema."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")
    df = pd.read_csv(csv_path)
    missing = [c for c in CSV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required columns {missing}: {csv_path}")
    df = df[CSV_COLUMNS]
    numeric = df.to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError(f"CSV contains NaN or infinite values: {csv_path}")
    ts = df["timestamp_us"].to_numpy(dtype=np.int64)
    if len(ts) < 2 or np.any(np.diff(ts) <= 0):
        raise ValueError(f"CSV timestamps must be strictly increasing: {csv_path}")
    q = df[GT_Q_COLS].to_numpy(dtype=np.float64)
    q_norm = np.linalg.norm(q, axis=1)
    if not np.allclose(q_norm, 1.0, rtol=1e-4, atol=1e-6):
        raise ValueError(f"CSV contains non-unit ground-truth quaternions: {csv_path}")
    return df


def resample_for_network(df: pd.DataFrame, sampling_freq: float | None) -> pd.DataFrame:
    """Resample a raw high-rate sequence onto the model's fixed time grid.

    This fixed-rate stream follows the sampling contract stored in the model
    checkpoint.
    """
    if sampling_freq is None:
        return df
    if sampling_freq <= 0:
        raise ValueError("sampling_freq must be positive")

    source_ts = df["timestamp_us"].to_numpy(dtype=np.int64)
    step_us = int(round(1e6 / float(sampling_freq)))
    if step_us <= 0:
        raise ValueError("sampling_freq is too high for integer microsecond timestamps")
    if np.all(np.diff(source_ts) == step_us):
        return df

    target_ts = np.arange(source_ts[0], source_ts[-1] + 1, step_us, dtype=np.int64)
    target_ts = target_ts[target_ts <= source_ts[-1]]
    if len(target_ts) < 2:
        raise ValueError("sequence is too short after network-grid resampling")

    out = {"timestamp_us": target_ts}
    linear_columns = GYR_COLS + ACC_COLS + WING_DISPLACEMENT_COLS + GT_P_COLS
    source_float = source_ts.astype(np.float64)
    target_float = target_ts.astype(np.float64)
    for column in linear_columns:
        out[column] = np.interp(target_float, source_float, df[column].to_numpy(dtype=np.float64))

    for column in WING_PHASE_COLS:
        phase = np.unwrap(df[column].to_numpy(dtype=np.float64))
        out[column] = np.mod(np.interp(target_float, source_float, phase), 2.0 * np.pi)

    rotations = Rotation.from_quat(df[GT_Q_COLS].to_numpy(dtype=np.float64))
    out_quat = Slerp(source_float, rotations)(target_float).as_quat()
    for index, column in enumerate(GT_Q_COLS):
        out[column] = out_quat[:, index]
    return pd.DataFrame(out, columns=CSV_COLUMNS)


def compute_delta_position_body(gt_p: np.ndarray, gt_q: np.ndarray,
                                begin: int, end: int) -> np.ndarray:
    """Compute adjacent-frame body displacement.

    Args:
        gt_p: Ground-truth positions with shape [N, 3].
        gt_q: Ground-truth xyzw quaternions with shape [N, 4].
        begin, end: Adjacent indices satisfying ``end == begin + 1``.

    Returns:
        delta_p_body: [3]
    """
    if end != begin + 1:
        raise ValueError("next-step displacement requires end == begin + 1")
    dp_world = gt_p[end] - gt_p[begin]
    R_begin = Rotation.from_quat(gt_q[begin]).as_matrix()  # R_wb_begin
    return R_begin.T @ dp_world


def build_feature(df: pd.DataFrame) -> np.ndarray:
    """Extract raw body-frame network features with shape [N, 10].

    Channel order: gyr(3), acc(3), displacement_L/R, phase_L/R.
    """
    feature = df[NETWORK_INPUT_COLUMNS].to_numpy(dtype=np.float64)
    if feature.shape[1] != NET_INPUT_DIM:
        raise RuntimeError(
            f"network feature contract has {feature.shape[1]} columns, "
            f"expected {NET_INPUT_DIM}"
        )
    return feature


class LepidDataset(Dataset):
    """Windowed IMU, measured wing displacement, and phase dataset.

    Each sample contains a channel-first input window and the next-step
    body-frame displacement target.
    """

    def __init__(self, csv_path_or_dir: str, window_size: int, stride: int,
                 sampling_freq: float | None = None):
        self.window_size = window_size
        self.stride = stride
        self.sampling_freq = sampling_freq
        self.samples = []
        self.sequence_records = []
        self._build_index(csv_path_or_dir)
        if len(self) == 0:
            raise RuntimeError(
                f"no samples were created (window_size={window_size}, "
                f"stride={stride}); check sequence lengths")

    def _build_index(self, path):
        """Create strided windows independently within each CSV sequence.

        ``path`` may be a CSV file, directory, or list of file paths.
        """
        csv_files = self._collect_csv(path)
        for f in csv_files:
            df = resample_for_network(load_csv(f), self.sampling_freq)
            if len(df) < self.window_size + 1:
                continue
            feat = build_feature(df)
            p = df[GT_P_COLS].to_numpy(dtype=np.float64)
            q = df[GT_Q_COLS].to_numpy(dtype=np.float64)
            ts = df["timestamp_us"].to_numpy(dtype=np.int64)
            n = len(df)
            sample_begin = len(self.samples)
            # Input contains W samples through input_end.  The target is the
            # single transition input_end -> target_end.
            for window_begin in range(0, n - self.window_size, self.stride):
                input_end = window_begin + self.window_size - 1
                target_end = input_end + 1
                dp = compute_delta_position_body(p, q, input_end, target_end)
                self.samples.append(
                    (feat, dp, ts, window_begin, input_end, target_end)
                )
            self.sequence_records.append({
                "name": Path(f).stem,
                "path": str(f),
                "sample_begin": sample_begin,
                "sample_end": len(self.samples),
                "positions": p,
                "quaternions": q,
                "timestamps": ts,
            })

    def _collect_csv(self, path) -> list:
        if isinstance(path, (list, tuple)):
            files = []
            for p in path:
                files.extend(self._collect_csv(p))
            return sorted(files)
        if os.path.isfile(path):
            return [path]
        return sorted(
            os.path.join(path, f)
            for f in os.listdir(path) if f.endswith(".csv")
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        feat, dp, ts, window_begin, input_end, target_end = self.samples[idx]
        x = feat[window_begin:input_end + 1]        # [window_size, 10]
        x = np.ascontiguousarray(x.T, dtype=np.float32)  # [10, window_size]
        label = dp.astype(np.float32)              # [3]
        return x, label, int(ts[input_end]), int(ts[target_end])


def collate_fn(batch: list) -> tuple:
    """Stack samples into a batch.

    Returns:
        x: Channel-first input with shape [B, 10, window_size].
        label: Body-frame displacement target with shape [B, 3].
        ts_begin: Start timestamps in microseconds.
        ts_end: End timestamps in microseconds.
    """
    xs = [torch.from_numpy(item[0]) for item in batch]
    labels = [torch.from_numpy(item[1]) for item in batch]
    ts_begin = torch.tensor([item[2] for item in batch])
    ts_end = torch.tensor([item[3] for item in batch])
    return torch.stack(xs, dim=0), torch.stack(labels, dim=0), ts_begin, ts_end


def resolve_sequence_files(csv_dir: str, sequences: list[str], split_name: str) -> list[str]:
    """Resolve an explicit sequence manifest to existing CSV files."""
    if not os.path.isdir(csv_dir):
        raise FileNotFoundError(f"data directory does not exist: {csv_dir}")
    if not sequences:
        raise ValueError(f"{split_name} sequence list must not be empty")

    files = []
    seen = set()
    for sequence in sequences:
        if not isinstance(sequence, str) or not sequence.strip():
            raise ValueError(f"invalid sequence name in {split_name}: {sequence!r}")
        name = sequence.strip()
        if os.path.basename(name) != name:
            raise ValueError(
                f"{split_name} sequence must be a file name, not a path: {name!r}"
            )
        stem = name[:-4] if name.lower().endswith(".csv") else name
        if stem in seen:
            raise ValueError(f"duplicate sequence in {split_name}: {stem}")
        seen.add(stem)
        path = os.path.join(csv_dir, f"{stem}.csv")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{split_name} sequence CSV does not exist: {path}"
            )
        files.append(path)
    return files


def create_dataloaders(csv_dir: str, window_size: int, stride: int,
                       batch_size: int, train_ratio: float = 0.7,
                       num_workers: int = 4, seed: int = 42,
                       prefetch_factor: int = 2,
                       pin_memory: bool | None = None,
                       train_sequences: list[str] | None = None,
                       val_sequences: list[str] | None = None,
                       sampling_freq: float | None = None):
    """Create training and validation data loaders.

    Automatic splitting operates on complete sequences to prevent leakage.
    """
    if (train_sequences is None) != (val_sequences is None):
        raise ValueError(
            "train_sequences and val_sequences must either both be configured or both omitted"
        )
    if train_sequences is not None:
        train_files = resolve_sequence_files(csv_dir, train_sequences, "train")
        val_files = resolve_sequence_files(csv_dir, val_sequences, "validation")
        overlap = {Path(path).stem for path in train_files} & {
            Path(path).stem for path in val_files
        }
        if overlap:
            raise ValueError(f"training and validation sequences overlap: {sorted(overlap)}")
    else:
        if not os.path.isdir(csv_dir):
            raise FileNotFoundError(f"data directory does not exist: {csv_dir}")
        csv_files = sorted(
            f for f in os.listdir(csv_dir) if f.endswith(".csv"))
        if not csv_files:
            raise RuntimeError(f"no CSV files found in {csv_dir}")
        rng = np.random.default_rng(seed)
        rng.shuffle(csv_files)
        n_train = max(1, int(len(csv_files) * train_ratio))
        train_names, val_names = csv_files[:n_train], csv_files[n_train:]
        if not val_names:
            raise RuntimeError("validation split is empty; provide at least two sequences")
        train_files = [os.path.join(csv_dir, name) for name in train_names]
        val_files = [os.path.join(csv_dir, name) for name in val_names]

    train_ds = LepidDataset(
        train_files,
        window_size=window_size, stride=stride, sampling_freq=sampling_freq)
    val_ds = LepidDataset(
        val_files,
        window_size=window_size, stride=stride, sampling_freq=sampling_freq)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=pin_memory)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn)
    return train_loader, val_loader
