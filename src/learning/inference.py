"""Offline batched displacement inference."""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from learning.dataset import LepidDataset, collate_fn, resolve_sequence_files
from learning.network.model_factory import load_model_from_ckpt
from learning.network.model_factory import resolve_device
from common.config_validation import validate_config


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


@torch.no_grad()
def run_inference(model: torch.nn.Module, loader: DataLoader,
                  device: torch.device) -> dict:
    """Run inference over a complete dataset.

    Returns:
        {'ts_begin', 'ts_end', 'delta_p', 'delta_p_gt'}
    """
    model.eval()
    ts_begin, ts_end, delta_p, delta_p_gt = [], [], [], []
    for x, label, tb, te in loader:
        x = x.to(device)
        pred = model(x)
        delta_p.append(pred["delta_p"].cpu().numpy())
        delta_p_gt.append(label.cpu().numpy())
        ts_begin.append(tb.numpy())
        ts_end.append(te.numpy())
    return {
        "ts_begin": np.concatenate(ts_begin) if ts_begin else np.array([]),
        "ts_end": np.concatenate(ts_end) if ts_end else np.array([]),
        "delta_p": np.concatenate(delta_p, axis=0) if delta_p else np.zeros((0, 3)),
        "delta_p_gt": np.concatenate(delta_p_gt, axis=0) if delta_p_gt else np.zeros((0, 3)),
    }


def save_results(results: dict, output_path: str):
    """Write displacement predictions and targets to CSV."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    dp = results["delta_p"]
    dp_gt = results["delta_p_gt"]
    df = pd.DataFrame({
        "ts_begin_us": results["ts_begin"].astype(np.int64),
        "ts_end_us": results["ts_end"].astype(np.int64),
        "dp_pred_x": dp[:, 0], "dp_pred_y": dp[:, 1], "dp_pred_z": dp[:, 2],
        "dp_gt_x": dp_gt[:, 0], "dp_gt_y": dp_gt[:, 1], "dp_gt_z": dp_gt[:, 2],
    })
    df.to_csv(output_path, index=False)


def infer(config_path: str, ckpt_path: str, csv_dir: str | None,
          output: str, device_str: str = None, batch_size: int = 1,
          split: str = "test"):
    """Run the inference command used by ``main_net.py``."""
    config = load_config(config_path)
    validate_config(config)
    data_cfg = config["data"]
    inference_cfg = config.get("inference", {})

    device = resolve_device(device_str or inference_cfg.get("device", "auto"))
    model = load_model_from_ckpt(ckpt_path, device)

    if csv_dir is None:
        split_key = f"{split}_sequences"
        sequences = data_cfg.get(split_key)
        csv_source = resolve_sequence_files(
            data_cfg["csv_dir"], sequences, split
        )
        print(f"Inference split: {split_key} = {sequences}")
    else:
        csv_source = csv_dir
        print(f"Inference source: {csv_dir}")

    ds = LepidDataset(
        csv_source, window_size=data_cfg["window_size"],
        stride=data_cfg.get("stride", 10),
        sampling_freq=data_cfg.get("imu_freq_net", 100))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)

    results = run_inference(model, loader, device)
    save_results(results, output)

    diff = results["delta_p"] - results["delta_p_gt"]
    rmse = np.sqrt((diff ** 2).mean(axis=0))
    print(
        f"Inference complete: {results['delta_p'].shape[0]} windows, "
        f"saved to {output}"
    )
    print(f"Δp RMSE (x/y/z) = {rmse[0]:.4f} / {rmse[1]:.4f} / {rmse[2]:.4f} [m]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lepid_io.json")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--csv_dir", default=None,
                        help="override the CSV directory selected by --split")
    parser.add_argument("--split", choices=("train", "val", "test"),
                        default="test")
    parser.add_argument("--output", default="results/net_output.csv")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    infer(args.config, args.ckpt, args.csv_dir, args.output,
          args.device, args.batch_size, args.split)
