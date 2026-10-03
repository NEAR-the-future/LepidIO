"""Training, validation, model selection, and checkpoint persistence."""

import csv
import json
import logging
import os
import shutil
import time

import numpy as np
import torch
import torch.optim as optim
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader

from common.columns import FEATURE_SCHEMA, MODEL_ARCHITECTURE_SCHEMA, TARGET_SCHEMA

from learning.dataset import create_dataloaders
from learning.loss import compute_loss
from learning.model import LepidNet
from learning.network.model_factory import resolve_device
from common.config_validation import validate_config


METRIC_FIELDS = (
    "epoch", "train_loss", "train_rmse", "val_loss", "val_rmse",
    "val_macro_loss", "val_macro_rmse", "val_worst_rmse",
    "val_trajectory_macro_ate_rmse", "val_trajectory_macro_final_error",
    "val_trajectory_macro_final_drift_percent", "val_trajectory_worst_ate_rmse",
    "val_trajectory_xy_macro_ate_rmse", "val_trajectory_xy_worst_ate_rmse",
    "val_trajectory_xy_macro_rte_1s", "val_trajectory_xy_macro_rte_5s",
    "val_trajectory_xy_macro_final_drift_percent",
    "learning_rate", "epoch_seconds", "best_step_epoch", "best_macro_val_rmse",
    "best_trajectory_epoch", "best_trajectory_ate_rmse",
    "best_xy_epoch", "best_xy_ate_rmse",
    "best_epoch", "best_val_loss",
)

SEQUENCE_METRIC_FIELDS = (
    "epoch", "sequence", "sample_count", "loss",
    "rmse", "trajectory_ate_rmse", "trajectory_final_error",
    "trajectory_final_drift_percent", "trajectory_axis_rmse_x",
    "trajectory_axis_rmse_y", "trajectory_axis_rmse_z",
    "trajectory_xy_ate_rmse", "trajectory_xy_final_error",
    "trajectory_xy_final_drift_percent", "trajectory_xy_rte_1s",
    "trajectory_xy_rte_5s",
)


def append_training_metric(path: str, row: dict, *, reset: bool = False) -> None:
    """Append one structured epoch record and flush it for live monitoring."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mode = "w" if reset else "a"
    with open(path, mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        if reset:
            writer.writeheader()
        if row:
            writer.writerow({key: row.get(key, "") for key in METRIC_FIELDS})


def append_sequence_metrics(path: str, rows: list[dict], *, reset: bool = False) -> None:
    """Persist one row per validation sequence for every evaluated epoch."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mode = "w" if reset else "a"
    with open(path, mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SEQUENCE_METRIC_FIELDS)
        if reset:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in SEQUENCE_METRIC_FIELDS})


def save_training_curves(metrics_path: str, output_path: str) -> None:
    """Render the persisted loss/RMSE history without requiring a display."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = []
        with open(metrics_path, "r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        evaluated = [row for row in rows if row.get("val_loss") not in (None, "")]
        if not rows or not evaluated:
            return

        epochs = [int(row["epoch"]) for row in rows]
        eval_epochs = [int(row["epoch"]) for row in evaluated]
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
        axes[0].plot(epochs, [float(row["train_loss"]) for row in rows], label="train")
        axes[0].plot(
            eval_epochs, [float(row["val_loss"]) for row in evaluated], label="validation"
        )
        axes[0].set(title="Loss", xlabel="Epoch", ylabel="Huber loss")
        axes[1].plot(epochs, [float(row["train_rmse"]) for row in rows], label="train")
        axes[1].plot(
            eval_epochs, [float(row["val_rmse"]) for row in evaluated], label="validation"
        )
        macro_rows = [
            row for row in evaluated if row.get("val_macro_rmse") not in (None, "")
        ]
        if macro_rows:
            axes[1].plot(
                [int(row["epoch"]) for row in macro_rows],
                [float(row["val_macro_rmse"]) for row in macro_rows],
                label="validation macro",
            )
        axes[1].set(title="RMSE", xlabel="Epoch", ylabel="m")
        trajectory_rows = [
            row for row in evaluated
            if row.get("val_trajectory_macro_ate_rmse") not in (None, "")
        ]
        if trajectory_rows:
            trajectory_epochs = [int(row["epoch"]) for row in trajectory_rows]
            xy_rows = [
                row for row in trajectory_rows
                if row.get("val_trajectory_xy_macro_ate_rmse") not in (None, "")
            ]
            if xy_rows:
                axes[2].plot(
                    [int(row["epoch"]) for row in xy_rows],
                    [float(row["val_trajectory_xy_macro_ate_rmse"]) for row in xy_rows],
                    label="macro XY trajectory ATE",
                )
                axes[2].plot(
                    [int(row["epoch"]) for row in xy_rows],
                    [float(row["val_trajectory_xy_worst_ate_rmse"]) for row in xy_rows],
                    label="worst XY trajectory ATE",
                )
            axes[2].plot(
                trajectory_epochs,
                [float(row["val_trajectory_macro_ate_rmse"]) for row in trajectory_rows],
                label="macro 3D trajectory ATE",
                linestyle="--",
            )
        axes[2].set(title="Validation trajectory", xlabel="Epoch", ylabel="m")
        for axis in axes:
            axis.grid(alpha=0.25)
            axis.legend()
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        logging.warning("Could not save training curves: %s", exc)


def load_config(config_path: str) -> dict:
    """Load a JSON configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_optimizer_and_scheduler(model: torch.nn.Module, train_cfg: dict):
    """Build the optimizer and cosine annealing scheduler."""
    optimizer = optim.AdamW(
        model.parameters(), lr=train_cfg.get("lr", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-5))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg.get("max_epochs", 200),
        eta_min=train_cfg.get("min_lr", 1e-6))
    return optimizer, scheduler


def train_one_epoch(model: torch.nn.Module, loader: DataLoader,
                    optimizer: optim.Optimizer, config: dict,
                    device: torch.device) -> dict:
    """Train one epoch and return loss and RMSE."""
    model.train()
    del config

    loss_sum = 0.0
    squared_error_sum, sample_count = 0.0, 0
    for x, label, _, _ in loader:
        x, label = x.to(device), label.to(device)
        pred = model(x)
        target = {"delta_p": label}
        loss_dict = compute_loss(pred, target)

        optimizer.zero_grad()
        loss_dict["loss"].backward()
        optimizer.step()

        batch_size = int(label.shape[0])
        loss_sum += loss_dict["loss"].item() * batch_size
        squared_error_sum += float(((pred["delta_p"] - label) ** 2).sum().item())
        sample_count += batch_size

    denominator = max(1, sample_count)
    return {
        "loss": loss_sum / denominator,
        "rmse": float(np.sqrt(squared_error_sum / max(1, sample_count * 3))),
    }


def _planar_relative_rmse(
    estimate: np.ndarray, ground_truth: np.ndarray, offset: int,
) -> float | None:
    """World-XY relative displacement RMSE at one fixed sample offset."""
    if offset <= 0 or len(estimate) <= offset:
        return None
    estimate_delta = estimate[offset:, :2] - estimate[:-offset, :2]
    ground_truth_delta = ground_truth[offset:, :2] - ground_truth[:-offset, :2]
    relative_error = estimate_delta - ground_truth_delta
    return float(np.sqrt(np.mean(np.sum(relative_error ** 2, axis=1))))


def _trajectory_metrics(dataset, predictions: np.ndarray) -> dict[str, dict]:
    """Reconstruct equal-rate next-step validation trajectories."""
    if dataset.stride != 1:
        return {}
    output = {}
    window_size = int(dataset.window_size)
    for record in dataset.sequence_records:
        begin, end = record["sample_begin"], record["sample_end"]
        sequence_predictions = predictions[begin:end]
        if len(sequence_predictions) == 0:
            continue
        positions = record["positions"]
        quaternions = record["quaternions"]
        anchor_index = window_size - 1
        reconstructed = np.empty((len(sequence_predictions) + 1, 3), dtype=np.float64)
        reconstructed[0] = positions[anchor_index]
        rotations = Rotation.from_quat(
            quaternions[anchor_index:anchor_index + len(sequence_predictions)]
        )
        world_displacements = rotations.apply(sequence_predictions)
        reconstructed[1:] = reconstructed[0] + np.cumsum(world_displacements, axis=0)
        gt_positions = positions[anchor_index:anchor_index + len(reconstructed)]
        errors = reconstructed[1:] - gt_positions[1:]
        error_norm = np.linalg.norm(errors, axis=1)
        xy_error_norm = np.linalg.norm(errors[:, :2], axis=1)
        path_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
        xy_path_length = float(
            np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1).sum()
        )
        final_error = float(error_norm[-1])
        xy_final_error = float(xy_error_norm[-1])
        rte_1s = _planar_relative_rmse(
            reconstructed, gt_positions, int(round(dataset.sampling_freq * 1.0))
        )
        rte_5s = _planar_relative_rmse(
            reconstructed, gt_positions, int(round(dataset.sampling_freq * 5.0))
        )
        output[record["name"]] = {
            "trajectory_ate_rmse": float(np.sqrt(np.mean(error_norm ** 2))),
            "trajectory_final_error": final_error,
            "trajectory_final_drift_percent": (
                final_error / path_length * 100.0 if path_length > 0.0 else float("inf")
            ),
            "trajectory_axis_rmse_x": float(np.sqrt(np.mean(errors[:, 0] ** 2))),
            "trajectory_axis_rmse_y": float(np.sqrt(np.mean(errors[:, 1] ** 2))),
            "trajectory_axis_rmse_z": float(np.sqrt(np.mean(errors[:, 2] ** 2))),
            "trajectory_xy_ate_rmse": float(
                np.sqrt(np.mean(xy_error_norm ** 2))
            ),
            "trajectory_xy_final_error": xy_final_error,
            "trajectory_xy_final_drift_percent": (
                xy_final_error / xy_path_length * 100.0
                if xy_path_length > 0.0 else float("inf")
            ),
            "trajectory_xy_rte_1s": rte_1s,
            "trajectory_xy_rte_5s": rte_5s,
        }
    return output


def _is_better_trajectory_candidate(candidate: dict, best: dict | None) -> bool:
    """Rank trajectory candidates with a 2% tie band and robust tie-breakers."""
    if best is None:
        return True
    tolerance = 0.02
    comparisons = (
        (candidate["trajectory_macro_ate_rmse"], best["trajectory_macro_ate_rmse"]),
        (candidate["trajectory_worst_ate_rmse"], best["trajectory_worst_ate_rmse"]),
        (
            candidate["trajectory_macro_final_drift_percent"],
            best["trajectory_macro_final_drift_percent"],
        ),
    )
    for current, previous in comparisons:
        if current < previous * (1.0 - tolerance):
            return True
        if current > previous * (1.0 + tolerance):
            return False
    # Keeping the earlier checkpoint when all criteria are effectively tied
    # avoids preferring extra training without validation evidence.
    return False


def _is_better_xy_candidate(candidate: dict, best: dict | None) -> bool:
    """Rank planar candidates with ATE first and planar RTE tie-breakers."""
    if best is None:
        return True
    tolerance = 0.02
    comparisons = (
        (candidate["trajectory_xy_macro_ate_rmse"],
         best["trajectory_xy_macro_ate_rmse"]),
        (candidate["trajectory_xy_worst_ate_rmse"],
         best["trajectory_xy_worst_ate_rmse"]),
        (candidate["trajectory_xy_macro_rte_5s"],
         best["trajectory_xy_macro_rte_5s"]),
        (candidate["trajectory_xy_macro_rte_1s"],
         best["trajectory_xy_macro_rte_1s"]),
    )
    for current, previous in comparisons:
        if current is None:
            continue
        if previous is None:
            return True
        if current < previous * (1.0 - tolerance):
            return True
        if current > previous * (1.0 + tolerance):
            return False
    return False


@torch.no_grad()
def validate(model: torch.nn.Module, loader: DataLoader,
             config: dict, device: torch.device) -> dict:
    """Compute exact pooled, equal-sequence macro, and trajectory metrics."""
    model.eval()
    del config

    records = loader.dataset.sequence_records
    sequence_totals = {
        record["name"]: {
            "sample_count": 0, "loss_sum": 0.0, "squared_error_sum": 0.0,
        }
        for record in records
    }
    boundaries = [record["sample_end"] for record in records]
    predictions = []
    sample_offset = 0
    for x, label, _, _ in loader:
        x, label = x.to(device), label.to(device)
        pred = model(x)
        predictions.append(pred["delta_p"].detach().cpu().numpy().astype(np.float64))
        batch_size = int(label.shape[0])
        local_begin = 0
        while local_begin < batch_size:
            global_index = sample_offset + local_begin
            sequence_index = int(np.searchsorted(boundaries, global_index, side="right"))
            record = records[sequence_index]
            local_end = min(batch_size, record["sample_end"] - sample_offset)
            sub_pred = {"delta_p": pred["delta_p"][local_begin:local_end]}
            sub_label = label[local_begin:local_end]
            sub_loss = compute_loss(sub_pred, {"delta_p": sub_label})
            count = int(sub_label.shape[0])
            totals = sequence_totals[record["name"]]
            totals["sample_count"] += count
            totals["loss_sum"] += sub_loss["loss"].item() * count
            totals["squared_error_sum"] += float(
                ((sub_pred["delta_p"] - sub_label) ** 2).sum().item()
            )
            local_begin = local_end
        sample_offset += batch_size

    per_sequence = {}
    for name, totals in sequence_totals.items():
        count = totals["sample_count"]
        if count == 0:
            continue
        per_sequence[name] = {
            "sample_count": count,
            "loss": totals["loss_sum"] / count,
            "rmse": float(np.sqrt(totals["squared_error_sum"] / (count * 3))),
        }

    if not per_sequence:
        raise RuntimeError("validation loader produced no sequence samples")
    trajectory = _trajectory_metrics(loader.dataset, np.concatenate(predictions, axis=0))
    for name, metrics in trajectory.items():
        per_sequence[name].update(metrics)

    total_count = sum(item["sample_count"] for item in per_sequence.values())
    pooled = {
        "loss": sum(
            item["loss"] * item["sample_count"] for item in per_sequence.values()
        ) / total_count
    }
    pooled_squared_error = sum(
        item["rmse"] ** 2 * item["sample_count"] * 3
        for item in per_sequence.values()
    )
    result = {
        **pooled,
        "rmse": float(np.sqrt(pooled_squared_error / (total_count * 3))),
        "macro_loss": float(np.mean([item["loss"] for item in per_sequence.values()])),
        "macro_rmse": float(np.mean([item["rmse"] for item in per_sequence.values()])),
        "worst_rmse": float(max(item["rmse"] for item in per_sequence.values())),
        "per_sequence": per_sequence,
        "trajectory_available": bool(trajectory),
    }
    if trajectory:
        rte_1s_values = [
            item["trajectory_xy_rte_1s"] for item in per_sequence.values()
            if item.get("trajectory_xy_rte_1s") is not None
        ]
        rte_5s_values = [
            item["trajectory_xy_rte_5s"] for item in per_sequence.values()
            if item.get("trajectory_xy_rte_5s") is not None
        ]
        result.update({
            "trajectory_macro_ate_rmse": float(np.mean([
                item["trajectory_ate_rmse"] for item in per_sequence.values()
            ])),
            "trajectory_macro_final_error": float(np.mean([
                item["trajectory_final_error"] for item in per_sequence.values()
            ])),
            "trajectory_macro_final_drift_percent": float(np.mean([
                item["trajectory_final_drift_percent"] for item in per_sequence.values()
            ])),
            "trajectory_worst_ate_rmse": float(max(
                item["trajectory_ate_rmse"] for item in per_sequence.values()
            )),
            "trajectory_xy_macro_ate_rmse": float(np.mean([
                item["trajectory_xy_ate_rmse"] for item in per_sequence.values()
            ])),
            "trajectory_xy_worst_ate_rmse": float(max(
                item["trajectory_xy_ate_rmse"] for item in per_sequence.values()
            )),
            "trajectory_xy_macro_rte_1s": (
                float(np.mean(rte_1s_values)) if rte_1s_values else None
            ),
            "trajectory_xy_macro_rte_5s": (
                float(np.mean(rte_5s_values)) if rte_5s_values else None
            ),
            "trajectory_xy_macro_final_drift_percent": float(np.mean([
                item["trajectory_xy_final_drift_percent"]
                for item in per_sequence.values()
            ])),
        })
    return result


def save_checkpoint(model: torch.nn.Module, optimizer: optim.Optimizer,
                    scheduler, epoch: int, best_loss: float,
                    config: dict, path: str):
    """Save a checkpoint and its deployment metadata."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "feature_schema": FEATURE_SCHEMA,
        "model_architecture": MODEL_ARCHITECTURE_SCHEMA,
        "target_schema": TARGET_SCHEMA,
        "model_config": model.model_config,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "best_loss": best_loss,
        "data_config": {
            key: config.get("data", {}).get(key)
            for key in (
                "window_time", "window_size", "imu_freq_net",
                "train_sequences", "val_sequences", "test_sequences",
            )
        },
    }, path)

    # Runtime sampling and feature contract.
    np_json = os.path.join(os.path.dirname(path), "model_param.json")
    data_cfg = config.get("data", {})
    with open(np_json, "w", encoding="utf-8") as f:
        json.dump({
            "sampling_freq": data_cfg.get("imu_freq_net", 200),
            "window_time": data_cfg.get("window_time", 1.0),
            "window_size": data_cfg.get("window_size"),
            "window_convention": "inclusive_endpoints",
            "quaternion_order": "xyzw",
            "feature_schema": FEATURE_SCHEMA,
            "model_architecture": MODEL_ARCHITECTURE_SCHEMA,
            "target_schema": TARGET_SCHEMA,
            "input_modalities": ["imu", "measured_flapping_angle", "commanded_phase"],
            "phase_encoding": "sin_cos",
            "fusion_encoder": True,
        }, f, indent=2)

    split_json = os.path.join(os.path.dirname(path), "data_split.json")
    with open(split_json, "w", encoding="utf-8") as f:
        json.dump({
            "train_sequences": data_cfg.get("train_sequences", []),
            "val_sequences": data_cfg.get("val_sequences", []),
            "test_sequences": data_cfg.get("test_sequences", []),
            "seed": config.get("train", {}).get("seed"),
        }, f, indent=2, ensure_ascii=False)


def train(config_path: str, device_str: str = None):
    """Run training from ``main_net.py``."""
    config = load_config(config_path)
    validate_config(config, require_training_splits=True)
    data_cfg, model_cfg, train_cfg = config["data"], config["model"], config["train"]
    paths_cfg = config["paths"]

    device = resolve_device(device_str or train_cfg.get("device", "auto"))
    torch.manual_seed(train_cfg.get("seed", 42))
    np.random.seed(train_cfg.get("seed", 42))

    # Data
    train_sequences = data_cfg.get("train_sequences")
    val_sequences = data_cfg.get("val_sequences")
    logging.info(f"Training sequences: {train_sequences or '[automatic directory split]'}")
    logging.info(f"Validation sequences: {val_sequences or '[automatic directory split]'}")
    train_loader, val_loader = create_dataloaders(
        csv_dir=data_cfg["csv_dir"],
        window_size=data_cfg["window_size"],
        stride=data_cfg.get("stride", 10),
        batch_size=data_cfg.get("batch_size", 32),
        train_ratio=data_cfg.get("train_ratio", 0.7),
        num_workers=data_cfg.get("num_workers", 4),
        prefetch_factor=data_cfg.get("prefetch_factor", 2),
        pin_memory=(device.type == "cuda"),
        train_sequences=train_sequences,
        val_sequences=val_sequences,
        sampling_freq=data_cfg.get("imu_freq_net", 100),
        seed=train_cfg.get("seed", 42),
    )

    # Model and optimizer
    model = LepidNet(model_cfg).to(device)
    logging.info(
        f"Model created with {model.get_num_params()/1e6:.2f}M params, "
        f"raw_input_dim={model.input_dim}, "
        "modalities=imu+measured_flapping_angle+commanded_phase, "
        "phase_encoding=sin_cos, fusion_encoder=true")
    optimizer, scheduler = build_optimizer_and_scheduler(model, train_cfg)

    # Training loop
    max_epochs = train_cfg.get("max_epochs", 200)
    eval_freq = train_cfg.get("eval_freq", 5)
    save_freq = train_cfg.get("save_freq", 10)
    ckpt_dir = paths_cfg.get("ckpt_dir", "results/lepid_io_baseline/ckpt")
    exp_dir = paths_cfg.get("exp_dir", os.path.dirname(ckpt_dir))
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)
    metrics_path = os.path.join(exp_dir, "training_metrics.csv")
    sequence_metrics_path = os.path.join(exp_dir, "validation_sequence_metrics.csv")
    selection_path = os.path.join(exp_dir, "model_selection.json")
    curves_path = os.path.join(exp_dir, "training_curves.png")
    append_training_metric(metrics_path, {}, reset=True)
    append_sequence_metrics(sequence_metrics_path, [], reset=True)

    best_macro_rmse = float("inf")
    best_step_epoch = None
    best_pooled_loss = float("inf")
    best_pooled_epoch = None
    best_trajectory = None
    best_trajectory_epoch = None
    best_xy = None
    best_xy_epoch = None
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, config, device)
        scheduler.step()

        if epoch % eval_freq == 0 or epoch == max_epochs:
            val_metrics = validate(model, val_loader, config, device)
            if val_metrics["loss"] < best_pooled_loss:
                best_pooled_loss = val_metrics["loss"]
                best_pooled_epoch = epoch
            if val_metrics["macro_rmse"] < best_macro_rmse:
                best_macro_rmse = val_metrics["macro_rmse"]
                best_step_epoch = epoch
                save_checkpoint(
                    model, optimizer, scheduler, epoch, best_macro_rmse,
                    config, os.path.join(ckpt_dir, "best_step_model.pt"))
                logging.info(
                    f"[Epoch {epoch}] best step model saved "
                    f"(macro_val_rmse={best_macro_rmse:.6f})"
                )

            trajectory_ate = val_metrics.get("trajectory_macro_ate_rmse")
            if trajectory_ate is not None and _is_better_trajectory_candidate(
                val_metrics, best_trajectory
            ):
                best_trajectory = {
                    key: val_metrics[key] for key in (
                        "trajectory_macro_ate_rmse",
                        "trajectory_worst_ate_rmse",
                        "trajectory_macro_final_drift_percent",
                    )
                }
                best_trajectory_epoch = epoch
                trajectory_path = os.path.join(ckpt_dir, "best_trajectory_model.pt")
                save_checkpoint(
                    model, optimizer, scheduler, epoch, trajectory_ate,
                    config, trajectory_path)
                logging.info(
                    f"[Epoch {epoch}] best trajectory model saved "
                    f"(macro_trajectory_ate={trajectory_ate:.6f} m)"
                )

            xy_ate = val_metrics.get("trajectory_xy_macro_ate_rmse")
            if xy_ate is not None and _is_better_xy_candidate(val_metrics, best_xy):
                best_xy = {
                    key: val_metrics[key] for key in (
                        "trajectory_xy_macro_ate_rmse",
                        "trajectory_xy_worst_ate_rmse",
                        "trajectory_xy_macro_rte_5s",
                        "trajectory_xy_macro_rte_1s",
                        "trajectory_xy_macro_final_drift_percent",
                    )
                }
                best_xy_epoch = epoch
                xy_path = os.path.join(ckpt_dir, "best_xy_model.pt")
                save_checkpoint(
                    model, optimizer, scheduler, epoch, xy_ate, config, xy_path
                )
                shutil.copy2(xy_path, os.path.join(ckpt_dir, "best_model.pt"))
                logging.info(
                    f"[Epoch {epoch}] best XY model saved "
                    f"(macro_XY_ATE={xy_ate:.6f} m, "
                    f"macro_XY_RTE_1s="
                    f"{best_xy['trajectory_xy_macro_rte_1s'] if best_xy['trajectory_xy_macro_rte_1s'] is not None else float('nan'):.6f} m, "
                    f"macro_XY_RTE_5s="
                    f"{best_xy['trajectory_xy_macro_rte_5s'] if best_xy['trajectory_xy_macro_rte_5s'] is not None else float('nan'):.6f} m)"
                )

            append_sequence_metrics(sequence_metrics_path, [
                {"epoch": epoch, "sequence": name, **values}
                for name, values in val_metrics["per_sequence"].items()
            ])
            with open(selection_path, "w", encoding="utf-8") as handle:
                json.dump({
                    "selection_policy": (
                        "equal-sequence world-XY validation trajectory ATE RMSE "
                        "with 2% tie band; tie-break by worst XY ATE, XY RTE-5s, "
                        "XY RTE-1s, then earlier epoch"
                    ),
                    "default_model": "ckpt/best_model.pt",
                    "default_model_alias_of": "ckpt/best_xy_model.pt",
                    "best_step_model": "ckpt/best_step_model.pt",
                    "best_step_epoch": best_step_epoch,
                    "best_macro_val_rmse": best_macro_rmse,
                    "best_trajectory_model": "ckpt/best_trajectory_model.pt",
                    "best_trajectory_epoch": best_trajectory_epoch,
                    "best_trajectory_ate_rmse": (
                        None if best_trajectory is None else
                        best_trajectory["trajectory_macro_ate_rmse"]
                    ),
                    "best_trajectory_worst_ate_rmse": (
                        None if best_trajectory is None else
                        best_trajectory["trajectory_worst_ate_rmse"]
                    ),
                    "best_trajectory_macro_final_drift_percent": (
                        None if best_trajectory is None else
                        best_trajectory["trajectory_macro_final_drift_percent"]
                    ),
                    "best_xy_model": "ckpt/best_xy_model.pt",
                    "best_xy_epoch": best_xy_epoch,
                    "best_xy_ate_rmse": (
                        None if best_xy is None else
                        best_xy["trajectory_xy_macro_ate_rmse"]
                    ),
                    "best_xy_worst_ate_rmse": (
                        None if best_xy is None else
                        best_xy["trajectory_xy_worst_ate_rmse"]
                    ),
                    "best_xy_rte_1s": (
                        None if best_xy is None else
                        best_xy["trajectory_xy_macro_rte_1s"]
                    ),
                    "best_xy_rte_5s": (
                        None if best_xy is None else
                        best_xy["trajectory_xy_macro_rte_5s"]
                    ),
                    "best_xy_macro_final_drift_percent": (
                        None if best_xy is None else
                        best_xy["trajectory_xy_macro_final_drift_percent"]
                    ),
                    "trajectory_tie_tolerance_percent": 2.0,
                    "trajectory_tie_breakers": [
                        "worst_sequence_xy_ate_rmse",
                        "macro_xy_rte_5s",
                        "macro_xy_rte_1s",
                        "earlier_epoch",
                    ],
                    "validation_sequences": list(val_metrics["per_sequence"]),
                    "sequence_weighting": "equal macro average",
                }, handle, indent=2, ensure_ascii=False)

            logging.info(
                f"[Epoch {epoch}/{max_epochs}] train_loss={train_metrics['loss']:.5f} "
                f"train_rmse={train_metrics['rmse']:.5f} "
                f"val_loss={val_metrics['loss']:.5f} val_rmse={val_metrics['rmse']:.5f} "
                f"lr={scheduler.get_last_lr()[0]:.2e} ({time.time()-t0:.1f}s)")
            logging.info(
                f"[Epoch {epoch}] macro_val_rmse={val_metrics['macro_rmse']:.5f} "
                f"worst_val_rmse={val_metrics['worst_rmse']:.5f} "
                f"macro_trajectory_ate="
                f"{val_metrics.get('trajectory_macro_ate_rmse', float('nan')):.5f} m "
                f"macro_XY_ATE="
                f"{val_metrics.get('trajectory_xy_macro_ate_rmse', float('nan')):.5f} m"
            )
        else:
            val_metrics = None
            logging.info(
                f"[Epoch {epoch}/{max_epochs}] train_loss={train_metrics['loss']:.5f} "
                f"train_rmse={train_metrics['rmse']:.5f} "
                f"lr={scheduler.get_last_lr()[0]:.2e} ({time.time()-t0:.1f}s)")

        append_training_metric(metrics_path, {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_rmse": train_metrics["rmse"],
            "val_loss": "" if val_metrics is None else val_metrics["loss"],
            "val_rmse": "" if val_metrics is None else val_metrics["rmse"],
            "val_macro_loss": "" if val_metrics is None else val_metrics["macro_loss"],
            "val_macro_rmse": "" if val_metrics is None else val_metrics["macro_rmse"],
            "val_worst_rmse": "" if val_metrics is None else val_metrics["worst_rmse"],
            "val_trajectory_macro_ate_rmse": (
                "" if val_metrics is None else
                val_metrics.get("trajectory_macro_ate_rmse", "")
            ),
            "val_trajectory_macro_final_error": (
                "" if val_metrics is None else
                val_metrics.get("trajectory_macro_final_error", "")
            ),
            "val_trajectory_macro_final_drift_percent": (
                "" if val_metrics is None else
                val_metrics.get("trajectory_macro_final_drift_percent", "")
            ),
            "val_trajectory_worst_ate_rmse": (
                "" if val_metrics is None else
                val_metrics.get("trajectory_worst_ate_rmse", "")
            ),
            "val_trajectory_xy_macro_ate_rmse": (
                "" if val_metrics is None else
                val_metrics.get("trajectory_xy_macro_ate_rmse", "")
            ),
            "val_trajectory_xy_worst_ate_rmse": (
                "" if val_metrics is None else
                val_metrics.get("trajectory_xy_worst_ate_rmse", "")
            ),
            "val_trajectory_xy_macro_rte_1s": (
                "" if val_metrics is None else
                val_metrics.get("trajectory_xy_macro_rte_1s", "")
            ),
            "val_trajectory_xy_macro_rte_5s": (
                "" if val_metrics is None else
                val_metrics.get("trajectory_xy_macro_rte_5s", "")
            ),
            "val_trajectory_xy_macro_final_drift_percent": (
                "" if val_metrics is None else
                val_metrics.get("trajectory_xy_macro_final_drift_percent", "")
            ),
            "learning_rate": scheduler.get_last_lr()[0],
            "epoch_seconds": time.time() - t0,
            "best_step_epoch": "" if best_step_epoch is None else best_step_epoch,
            "best_macro_val_rmse": "" if best_step_epoch is None else best_macro_rmse,
            "best_trajectory_epoch": (
                "" if best_trajectory_epoch is None else best_trajectory_epoch
            ),
            "best_trajectory_ate_rmse": (
                "" if best_trajectory is None else
                best_trajectory["trajectory_macro_ate_rmse"]
            ),
            "best_xy_epoch": "" if best_xy_epoch is None else best_xy_epoch,
            "best_xy_ate_rmse": (
                "" if best_xy is None else best_xy["trajectory_xy_macro_ate_rmse"]
            ),
            "best_epoch": "" if best_pooled_epoch is None else best_pooled_epoch,
            "best_val_loss": "" if best_pooled_epoch is None else best_pooled_loss,
        })

        if epoch % save_freq == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_pooled_loss, config,
                os.path.join(ckpt_dir, f"checkpoint_{epoch:04d}.pt"))
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_pooled_loss, config,
                os.path.join(ckpt_dir, "newest.pt"))

    if best_xy_epoch is None:
        fallback_path = os.path.join(ckpt_dir, "best_trajectory_model.pt")
        fallback_name = "best_trajectory_model.pt"
        if not os.path.isfile(fallback_path):
            fallback_path = os.path.join(ckpt_dir, "best_step_model.pt")
            fallback_name = "best_step_model.pt"
        if not os.path.isfile(fallback_path):
            raise RuntimeError("training produced no selectable checkpoint")
        shutil.copy2(fallback_path, os.path.join(ckpt_dir, "best_model.pt"))
        logging.warning(
            "Dense XY validation trajectory selection requires stride=1; "
            f"best_model.pt falls back to {fallback_name}"
        )

    save_training_curves(metrics_path, curves_path)
    logging.info(
        f"Training finished. Best XY epoch={best_xy_epoch}, macro XY ATE="
        f"{best_xy['trajectory_xy_macro_ate_rmse'] if best_xy else float('nan'):.6f} m; "
        f"best 3D trajectory epoch={best_trajectory_epoch}, "
        f"best step epoch={best_step_epoch}, macro val RMSE={best_macro_rmse:.6f} m"
    )
    return (
        best_xy["trajectory_xy_macro_ate_rmse"]
        if best_xy is not None else (
            best_trajectory["trajectory_macro_ate_rmse"]
            if best_trajectory is not None else best_macro_rmse
        )
    )


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lepid_io.json")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    train(args.config, args.device)
