"""Regression tests for the public Lepid-IO network pipeline."""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from common.config_validation import (
    config_for_user_storage,
    expected_window_size,
    validate_config,
)
from learning.input_buffer import NetInputBuffer, interpolate_wrapped_phase


def minimal_config(**data_overrides):
    data = {"imu_freq_net": 100, "window_time": 1.0}
    data.update(data_overrides)
    return {
        "data": data,
        "model": {},
    }


class TimingContractTests(unittest.TestCase):
    def test_inclusive_window_size(self):
        self.assertEqual(expected_window_size(1.0, 100), 101)

    def test_window_contract_is_materialized(self):
        config = minimal_config(window_time=1.5)
        validate_config(config)
        self.assertEqual(config["data"]["window_size"], 151)
        self.assertEqual(config["network_param"]["window_size"], 151)
        self.assertEqual(
            config["network_param"]["feature_schema"],
            "imu6_actual_wing_displacement2_actual_phase2_v1",
        )
        self.assertEqual(
            config["network_param"]["target_schema"],
            "next_step_body_delta_position_v1",
        )

    def test_user_config_omits_derived_fields(self):
        config = minimal_config(window_size=101)
        config["network_param"] = {"window_size": 101}
        stored = config_for_user_storage(config)
        self.assertNotIn("window_size", stored["data"])
        self.assertNotIn("network_param", stored)

    def test_network_output_frequency_must_match_input_frequency(self):
        config = minimal_config()
        config["inference"] = {"network_output_freq": 50}
        with self.assertRaisesRegex(ValueError, "network_output_freq"):
            validate_config(config)

    def test_repository_config_is_consistent(self):
        config = json.loads(
            (ROOT / "configs" / "lepid_io.json").read_text(encoding="utf-8")
        )
        validate_config(config)
        self.assertNotIn("ekf", config)

    def test_selectable_model_topology_is_rejected(self):
        config = minimal_config()
        config["model"]["channels"] = [8]
        with self.assertRaisesRegex(ValueError, "topology is fixed"):
            validate_config(config)

    def test_training_and_validation_must_not_overlap(self):
        config = minimal_config(
            train_sequences=["seq_a"],
            val_sequences=["seq_a"],
            test_sequences=["seq_b"],
        )
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_config(config)

    def test_sequence_manifest_resolves_exact_files(self):
        from learning.dataset import resolve_sequence_files

        with tempfile.TemporaryDirectory() as directory:
            expected = Path(directory) / "seq_a.csv"
            expected.write_text("", encoding="utf-8")
            self.assertEqual(
                resolve_sequence_files(directory, ["seq_a"], "train"),
                [str(expected)],
            )


class InputPipelineTests(unittest.TestCase):
    def test_phase_interpolation_crosses_wrap_boundary(self):
        start = np.deg2rad([[350.0, 10.0]])
        end = np.deg2rad([[10.0, 350.0]])
        midpoint = interpolate_wrapped_phase(start, end, 0.5)
        wrapped_error = (midpoint + np.pi) % (2.0 * np.pi) - np.pi
        np.testing.assert_allclose(wrapped_error, np.zeros((1, 2)), atol=1e-12)

    def test_uniform_buffer_and_window(self):
        from learning.online_pipeline import OnlineInputPipeline

        pipeline = OnlineInputPipeline(sampling_freq=100, window_size=3)
        pipeline.start_at(0)
        zeros3 = np.zeros((3, 1))
        zeros2 = np.zeros((2, 1))
        pipeline.append_available(
            last_t_us=-1,
            t_us=0,
            last_gyr=None,
            gyr=zeros3,
            last_displacement=None,
            displacement=zeros2,
            last_phase=None,
            phase=zeros2,
            last_acc=None,
            acc=zeros3,
        )
        pipeline.append_available(
            last_t_us=0,
            t_us=20_000,
            last_gyr=zeros3,
            gyr=np.full((3, 1), 2.0),
            last_displacement=zeros2,
            displacement=np.full((2, 1), 2.0),
            last_phase=zeros2,
            phase=np.full((2, 1), 0.2),
            last_acc=zeros3,
            acc=np.full((3, 1), 2.0),
        )
        window = pipeline.get_window(20_000)
        np.testing.assert_allclose(window.timestamps_s, [0.0, 0.01, 0.02])
        np.testing.assert_allclose(window.gyr_body[:, 0], [0.0, 1.0, 2.0])
        self.assertEqual(pipeline.input_bounds(20_000), (0, 20_000))
        self.assertIsInstance(pipeline.buffer, NetInputBuffer)


class DatasetContractTests(unittest.TestCase):
    def test_body_displacement_uses_xyzw_quaternion_order(self):
        from learning.dataset import compute_delta_position_body

        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        quaternions = np.array([
            [0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)],
            [0.0, 0.0, 0.0, 1.0],
        ])
        displacement = compute_delta_position_body(
            positions, quaternions, 0, 1
        )
        np.testing.assert_allclose(displacement, [0.0, -1.0, 0.0], atol=1e-12)

    def test_network_feature_order_is_displacement_before_phase(self):
        import pandas as pd

        from common.columns import CSV_COLUMNS
        from learning.dataset import build_feature

        data = {column: [0.0] for column in CSV_COLUMNS}
        data["flap_displacement_left_actual_rad"] = [1.0]
        data["flap_displacement_right_actual_rad"] = [2.0]
        data["flap_phase_left_actual_rad"] = [3.0]
        data["flap_phase_right_actual_rad"] = [4.0]
        feature = build_feature(pd.DataFrame(data))
        np.testing.assert_array_equal(feature[0, 6:], [1.0, 2.0, 3.0, 4.0])


class ModelArchitectureTests(unittest.TestCase):
    def test_final_c5_has_fixed_six_layer_paper_architecture(self):
        import torch

        from learning.model import LepidNet, TCN_CHANNELS

        model = LepidNet({"dropout": 0.0, "gru_dropout": 0.0})
        output = model(torch.zeros(2, 10, 21))
        self.assertEqual(set(output), {"delta_p"})
        self.assertEqual(tuple(output["delta_p"].shape), (2, 3))
        self.assertTrue(model.use_fusion_encoder)
        self.assertEqual(len(model.tcn.network), 6)
        self.assertEqual(
            tuple(block.conv1.out_channels for block in model.tcn.network),
            TCN_CHANNELS,
        )
        self.assertEqual(model.imu_encoder.projection.out_channels, 32)
        self.assertEqual(model.displacement_encoder.projection.out_channels, 8)
        self.assertEqual(model.phase_encoder.projection.out_channels, 8)
        self.assertEqual(model.feature_fusion.projection.out_channels, 64)
        self.assertEqual(model.gru.hidden_size, 128)
        self.assertEqual(model.gru.num_layers, 2)
        self.assertFalse(model.gru.bidirectional)
        self.assertEqual(model.decoder.dp_head[-1].out_features, 3)


if __name__ == "__main__":
    unittest.main()
