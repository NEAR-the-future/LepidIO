"""Validation for the Lepid-IO training and inference configuration."""

from __future__ import annotations

import copy
import math

from common.columns import (
    FEATURE_SCHEMA,
    MODEL_ARCHITECTURE_SCHEMA,
    TARGET_SCHEMA,
)


def expected_window_size(window_time: float, sampling_freq: int) -> int:
    """Return inclusive endpoint sample count for a window of ``window_time`` seconds."""
    intervals = float(window_time) * int(sampling_freq)
    rounded = round(intervals)
    if not math.isclose(intervals, rounded, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"window_time={window_time} cannot be represented at {sampling_freq} Hz"
        )
    return int(rounded) + 1


def materialize_derived_config(config: dict) -> dict:
    """Populate timing fields derived from ``data.window_time``.

    ``data.window_time`` and ``data.imu_freq_net`` are the only editable
    sources of truth.  ``data.window_size`` and ``network_param`` remain in
    runtime configs and checkpoints for reproducibility, but callers never
    need to maintain them manually.
    """
    data = config.setdefault("data", {})
    sampling_freq = int(data["imu_freq_net"])
    window_time = float(data["window_time"])
    window_size = expected_window_size(window_time, sampling_freq)
    data["window_size"] = window_size
    network_param = config.setdefault("network_param", {})
    network_param.update({
        "sampling_freq": sampling_freq,
        "window_time": window_time,
        "window_size": window_size,
        "window_convention": "inclusive_endpoints",
        "quaternion_order": "xyzw",
        "feature_schema": FEATURE_SCHEMA,
        "model_architecture": MODEL_ARCHITECTURE_SCHEMA,
        "target_schema": TARGET_SCHEMA,
    })
    return config


def config_for_user_storage(config: dict) -> dict:
    """Return a base config containing only user-editable timing fields."""
    stored = copy.deepcopy(config)
    stored.setdefault("data", {}).pop("window_size", None)
    stored.pop("network_param", None)
    return stored


def validate_config(config: dict, require_training_splits: bool = False) -> None:
    """Materialize and validate the network-only pipeline contract."""
    materialization_error = None
    try:
        materialize_derived_config(config)
    except (KeyError, TypeError, ValueError) as exc:
        materialization_error = str(exc)
    data = config.get("data", {})
    model = config.get("model", {})
    network_param = config.get("network_param", {})
    errors = []
    if materialization_error:
        errors.append(f"cannot derive network window: {materialization_error}")

    inference = config.get("inference", {})

    sampling_freq = int(data.get("imu_freq_net", 0))
    network_output_freq = int(
        inference.get("network_output_freq", sampling_freq)
    )
    window_time = float(data.get("window_time", 0.0))
    window_size = int(data.get("window_size", 0))

    if sampling_freq <= 0:
        errors.append("data.imu_freq_net must be positive")
    if network_output_freq <= 0:
        errors.append("inference.network_output_freq must be positive")
    if window_time <= 0:
        errors.append("data.window_time must be positive")

    if sampling_freq > 0 and window_time > 0:
        try:
            expected = expected_window_size(window_time, sampling_freq)
            if window_size != expected:
                errors.append(
                    f"data.window_size must be {expected} for an inclusive "
                    f"{window_time}s window at {sampling_freq}Hz (got {window_size})"
                )
        except ValueError as exc:
            errors.append(str(exc))

    if sampling_freq > 0 and network_output_freq > 0:
        if network_output_freq != sampling_freq:
            errors.append(
                "next-step pure-network reconstruction requires "
                "inference.network_output_freq to equal "
                f"imu_freq_net={sampling_freq}"
            )

    allowed_model_keys = {"_comment", "_architecture", "dropout", "gru_dropout"}
    unsupported_model_keys = sorted(set(model) - allowed_model_keys)
    if unsupported_model_keys:
        errors.append(
            "model topology is fixed to the paper C5 architecture; remove "
            f"unsupported options {unsupported_model_keys}"
        )
    for key in ("dropout", "gru_dropout"):
        value = model.get(key, 0.1)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not 0.0 <= float(value) < 1.0
        ):
            errors.append(f"model.{key} must be in [0, 1)")
    train_cfg = config.get("train", {})
    if train_cfg.get("loss", "Huber") != "Huber":
        errors.append("train.loss must be 'Huber' for the final LepidIO objective")
    if "cov_weight" in train_cfg:
        errors.append("train.cov_weight is not used without a covariance head")

    if network_param:
        expected_network_values = {
            "sampling_freq": sampling_freq,
            "window_time": window_time,
            "window_size": window_size,
            "window_convention": "inclusive_endpoints",
            "quaternion_order": "xyzw",
            "feature_schema": FEATURE_SCHEMA,
            "model_architecture": MODEL_ARCHITECTURE_SCHEMA,
            "target_schema": TARGET_SCHEMA,
        }
        for key, expected_value in expected_network_values.items():
            if network_param.get(key) != expected_value:
                errors.append(
                    f"network_param.{key}={network_param.get(key)!r} must match "
                    f"the data contract value {expected_value!r}"
                )

    split_keys = ("train_sequences", "val_sequences", "test_sequences")
    evaluation_only = "evaluation_sequences" in data
    configured_splits = {key: data.get(key) for key in split_keys}
    if any(value is not None for value in configured_splits.values()):
        normalized_splits = {}
        for key, values in configured_splits.items():
            if not isinstance(values, list):
                errors.append(f"data.{key} must be a list")
                normalized_splits[key] = set()
                continue
            if not values and require_training_splits:
                errors.append(f"data.{key} must be a non-empty list")
                normalized_splits[key] = set()
                continue
            invalid = [value for value in values if not isinstance(value, str) or not value.strip()]
            if invalid:
                errors.append(f"data.{key} contains invalid sequence names: {invalid!r}")
            names = [value.strip() for value in values if isinstance(value, str) and value.strip()]
            if len(names) != len(set(names)):
                errors.append(f"data.{key} contains duplicate sequence names")
            normalized_splits[key] = set(names)

        # Training and validation must remain independent because validation
        # drives checkpoint selection. Test/evaluation sequences may
        # intentionally overlap either split to report seen-vs-unseen results.
        train_val_overlap = (
            normalized_splits.get("train_sequences", set())
            & normalized_splits.get("val_sequences", set())
        )
        if train_val_overlap:
            errors.append(
                "data.train_sequences and data.val_sequences overlap: "
                f"{sorted(train_val_overlap)}"
            )

    if evaluation_only:
        evaluation_sequences = data.get("evaluation_sequences")
        if not isinstance(evaluation_sequences, list) or not evaluation_sequences:
            errors.append("data.evaluation_sequences must be a non-empty list")
        else:
            invalid = [
                value for value in evaluation_sequences
                if not isinstance(value, str) or not value.strip()
            ]
            if invalid:
                errors.append(
                    "data.evaluation_sequences contains invalid sequence names: "
                    f"{invalid!r}"
                )
            names = [
                value.strip() for value in evaluation_sequences
                if isinstance(value, str) and value.strip()
            ]
            if len(names) != len(set(names)):
                errors.append("data.evaluation_sequences contains duplicate sequence names")

    if errors:
        raise ValueError("Invalid Lepid-IO config:\n- " + "\n- ".join(errors))
