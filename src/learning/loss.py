"""Huber objective and displacement metrics."""

import torch


def huber_loss(diff: torch.Tensor, delta: float = 0.05) -> torch.Tensor:
    """Huber loss with a quadratic core and linear tails."""
    abs_diff = diff.abs()
    quadratic = torch.clamp(abs_diff, max=delta) ** 2 / 2
    linear = delta * (abs_diff - delta / 2)
    return (torch.where(abs_diff <= delta, quadratic, linear)).mean()


def compute_loss(pred: dict, target: dict) -> dict:
    """Compute the Huber objective used by the final model.

    Args:
        pred:       {'delta_p': [B,3]}
        target:     {'delta_p': [B,3]}

    Returns:
        {'loss', 'rmse'}
    """
    delta_p = pred["delta_p"]
    target_dp = target["delta_p"]
    diff = delta_p - target_dp

    loss = huber_loss(diff)
    rmse = torch.sqrt((diff ** 2).mean())
    return {"loss": loss, "rmse": rmse}


def compute_metrics(pred: dict, target: dict) -> dict:
    """Compute displacement metrics for logging and visualization.

    Returns:
        {'rmse': [3], 'mae': [3], 'dist': [B]}
    """
    delta_p = pred["delta_p"]
    target_dp = target["delta_p"]
    diff = delta_p - target_dp                     # [B, 3]

    rmse = torch.sqrt((diff ** 2).mean(dim=0))     # [3]
    mae = diff.abs().mean(dim=0)                   # [3]
    dist = torch.linalg.norm(diff, dim=1)          # [B]

    return {
        "rmse": rmse,
        "mae": mae,
        "dist": dist,
    }
