"""Model construction, device resolution, and checkpoint loading."""

import torch

from common.columns import FEATURE_SCHEMA, MODEL_ARCHITECTURE_SCHEMA, TARGET_SCHEMA
from learning.model import LepidNet


def resolve_device(device_str=None) -> torch.device:
    """Resolve ``auto`` to CUDA when available, otherwise CPU."""
    requested = "auto" if device_str is None else str(device_str).strip().lower()
    if requested in ("", "auto"):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device '{device}' was requested, but CUDA is not available. "
            "Use --device cpu or set the config device to 'auto'."
        )
    return device


def get_model() -> LepidNet:
    """Build the fixed paper C5 network."""
    return LepidNet()


def load_model_from_ckpt(ckpt_path: str, device: torch.device) -> LepidNet:
    """Load fixed C5 weights and dropout metadata from a checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device)
    if ckpt.get("feature_schema") != FEATURE_SCHEMA:
        raise ValueError(
            "checkpoint feature_schema does not match the current input "
            f"contract {FEATURE_SCHEMA!r}; retrain the model"
        )
    if ckpt.get("model_architecture") != MODEL_ARCHITECTURE_SCHEMA:
        raise ValueError(
            "checkpoint model_architecture does not match the current "
            f"contract {MODEL_ARCHITECTURE_SCHEMA!r}; retrain the model"
        )
    if ckpt.get("target_schema") != TARGET_SCHEMA:
        raise ValueError(
            "checkpoint target_schema does not match the current next-step "
            f"prediction contract {TARGET_SCHEMA!r}; retrain the model"
        )
    model = LepidNet(ckpt.get("model_config", {}))
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval().to(device)
    return model
