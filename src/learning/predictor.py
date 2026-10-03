"""Checkpoint-backed displacement prediction for the main Lepid-IO pipeline."""

from __future__ import annotations

import logging

import numpy as np
import torch

from common.columns import NET_INPUT_DIM
from learning.network.model_factory import load_model_from_ckpt, resolve_device


def assemble_network_features(
    gyr_body,
    acc_body,
    wing_displacement,
    wing_phase,
):
    """Assemble the raw ten-channel C5 feature matrix in canonical order."""
    features = np.concatenate(
        [gyr_body, acc_body, wing_displacement, wing_phase], axis=1
    )
    if features.ndim != 2 or features.shape[1] != NET_INPUT_DIM:
        raise ValueError(
            f"network input must have shape [T, {NET_INPUT_DIM}], got {features.shape}"
        )
    if not np.isfinite(features).all():
        raise ValueError("network input contains NaN or infinite values")
    return features


class DisplacementPredictor:
    """Load the final C5 checkpoint and predict one body-frame displacement."""

    def __init__(self, model_path, force_cpu=False, device=None):
        self.device = torch.device("cpu") if force_cpu else resolve_device(device)
        self.net = load_model_from_ckpt(model_path, self.device)
        if self.net.input_dim != NET_INPUT_DIM:
            raise ValueError(
                f"checkpoint expects {self.net.input_dim} input channels, "
                f"but the Lepid-IO contract provides {NET_INPUT_DIM}"
            )
        logging.info(
            "Model %s loaded on %s (input_dim=%s)",
            model_path,
            self.device,
            self.net.input_dim,
        )

    def predict_displacement(
        self,
        net_t_s,
        net_gyr_body,
        net_acc_body,
        net_wing_displacement,
        net_wing_phase,
    ):
        """Return ``Δp_body`` as a ``[3, 1]`` NumPy array.

        ``net_t_s`` is accepted as part of the synchronized-window contract;
        the current network uses sample order and does not embed timestamps.
        """
        del net_t_s
        features = assemble_network_features(
            net_gyr_body,
            net_acc_body,
            net_wing_displacement,
            net_wing_phase,
        )
        features_t = torch.unsqueeze(
            torch.from_numpy(features.T).float().to(self.device), 0
        )
        with torch.no_grad():
            output = self.net(features_t)
        return output["delta_p"].cpu().detach().numpy().reshape((3, 1))
