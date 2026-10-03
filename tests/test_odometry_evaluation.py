"""Regression tests for the paper-grade odometry metric protocol."""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from odometry_evaluation import (  # noqa: E402
    METRIC_SPEC_VERSION,
    evaluate_trajectory,
    geodesic_degrees,
    quaternion_multiply,
    resolve_network_window,
    rotation_error_quaternion,
    slerp,
    summarize_run,
)


IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])


def trajectory(timestamps, positions, quaternions=None):
    timestamps = np.asarray(timestamps, dtype=float)
    positions = np.asarray(positions, dtype=float)
    if quaternions is None:
        quaternions = np.tile(IDENTITY, (len(timestamps), 1))
    return {
        "timestamps": timestamps,
        "positions": positions,
        "quaternions": np.asarray(quaternions, dtype=float),
    }


def ground_truth(timestamps, positions, quaternions=None):
    return trajectory(timestamps, positions, quaternions)


class TrajectoryMetricTests(unittest.TestCase):
    def test_identical_trajectory_has_zero_error(self):
        timestamps = np.arange(0.0, 6.1, 0.1)
        positions = np.column_stack([timestamps, timestamps * 0, timestamps * 0])
        metrics, _, _ = evaluate_trajectory(
            ground_truth(timestamps, positions), trajectory(timestamps, positions)
        )
        self.assertAlmostEqual(metrics["position"]["ate_raw_m"]["rmse"], 0.0)
        self.assertAlmostEqual(metrics["position"]["ate_se3_aligned_m"]["rmse"], 0.0)
        self.assertAlmostEqual(metrics["attitude"]["so3_geodesic_deg"]["rmse"], 0.0)
        self.assertAlmostEqual(
            metrics["relative_pose"]["5s"]["translation_rmse_m"], 0.0
        )

    def test_rigid_offset_is_removed_only_by_se3_alignment(self):
        timestamps = np.arange(0.0, 6.1, 0.1)
        gt_positions = np.column_stack([timestamps, np.sin(timestamps), timestamps * 0])
        est_positions = gt_positions + np.array([3.0, -2.0, 1.0])
        metrics, _, _ = evaluate_trajectory(
            ground_truth(timestamps, gt_positions),
            trajectory(timestamps, est_positions),
        )
        self.assertGreater(metrics["position"]["ate_raw_m"]["rmse"], 3.0)
        self.assertLess(metrics["position"]["ate_se3_aligned_m"]["rmse"], 1e-10)

    def test_linear_drift_has_analytic_relative_error_and_drift_rate(self):
        timestamps = np.arange(0.0, 10.01, 0.05)
        gt_positions = np.column_stack([timestamps, timestamps * 0, timestamps * 0])
        est_positions = np.column_stack([1.1 * timestamps, timestamps * 0, timestamps * 0])
        metrics, _, _ = evaluate_trajectory(
            ground_truth(timestamps, gt_positions), trajectory(timestamps, est_positions)
        )
        self.assertAlmostEqual(metrics["position"]["final_drift_percent"], 10.0, places=8)
        self.assertAlmostEqual(
            metrics["relative_pose"]["1s"]["translation_rmse_m"], 0.1, places=8
        )
        self.assertAlmostEqual(
            metrics["relative_pose"]["5s"]["translation_rmse_m"], 0.5, places=8
        )

    def test_quaternion_sign_and_wrap_are_invariant(self):
        q = np.array([[0.0, 0.0, math.sin(math.radians(179.0) / 2),
                       math.cos(math.radians(179.0) / 2)]])
        minus_q = -q
        error = rotation_error_quaternion(q, minus_q)
        np.testing.assert_allclose(geodesic_degrees(error), [0.0], atol=1e-10)
        q_minus = np.array([[0.0, 0.0, math.sin(math.radians(-179.0) / 2),
                             math.cos(math.radians(-179.0) / 2)]])
        wrapped = rotation_error_quaternion(q, q_minus)
        np.testing.assert_allclose(geodesic_degrees(wrapped), [2.0], atol=1e-8)

    def test_slerp_supports_asynchronous_timestamps(self):
        end = np.array([0.0, 0.0, 1.0, 0.0])
        interpolated = slerp(
            np.array([0.0, 2.0]), np.stack([IDENTITY, end]), np.array([1.0])
        )
        relative = quaternion_multiply(
            np.tile(IDENTITY, (1, 1)), interpolated
        )
        np.testing.assert_allclose(geodesic_degrees(relative), [90.0], atol=1e-8)

    def test_asynchronous_trajectory_uses_linear_position_and_slerp(self):
        gt_timestamps = np.arange(0.0, 6.01, 0.1)
        est_timestamps = np.arange(0.05, 5.96, 0.1)

        def yaw_quaternions(timestamps):
            half_angle = np.radians(timestamps * 10.0) / 2.0
            return np.column_stack([
                timestamps * 0, timestamps * 0,
                np.sin(half_angle), np.cos(half_angle),
            ])

        gt_positions = np.column_stack([
            2.0 * gt_timestamps, -gt_timestamps, gt_timestamps * 0.5,
        ])
        est_positions = np.column_stack([
            2.0 * est_timestamps, -est_timestamps, est_timestamps * 0.5,
        ])
        metrics, _, _ = evaluate_trajectory(
            ground_truth(gt_timestamps, gt_positions, yaw_quaternions(gt_timestamps)),
            trajectory(est_timestamps, est_positions, yaw_quaternions(est_timestamps)),
        )
        self.assertAlmostEqual(metrics["coverage"], 1.0)
        self.assertLess(metrics["position"]["ate_raw_m"]["rmse"], 1e-12)
        self.assertLess(metrics["attitude"]["so3_geodesic_deg"]["rmse"], 1e-8)

    def test_gt_outage_counts_against_association_coverage(self):
        gt_timestamps = np.array([0.0, 1.0, 2.0, 10.0, 11.0])
        est_timestamps = np.arange(0.0, 11.1, 1.0)
        gt_positions = np.column_stack([
            gt_timestamps, gt_timestamps * 0, gt_timestamps * 0,
        ])
        est_positions = np.column_stack([
            est_timestamps, est_timestamps * 0, est_timestamps * 0,
        ])
        with self.assertRaisesRegex(ValueError, "coverage"):
            evaluate_trajectory(
                ground_truth(gt_timestamps, gt_positions),
                trajectory(est_timestamps, est_positions),
            )

    def test_short_sequence_marks_five_second_metric_unavailable(self):
        timestamps = np.arange(0.0, 3.01, 0.05)
        positions = np.column_stack([timestamps, timestamps * 0, timestamps * 0])
        metrics, _, _ = evaluate_trajectory(
            ground_truth(timestamps, positions), trajectory(timestamps, positions)
        )
        self.assertFalse(metrics["relative_pose"]["5s"]["available"])


class AggregateMetricTests(unittest.TestCase):
    def test_summary_uses_unweighted_macro_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = (("short", 1.0, 10), ("long", 9.0, 10000))
            for name, ate, samples in values:
                sequence_dir = root / name
                sequence_dir.mkdir()
                (sequence_dir / "evaluation_metrics.json").write_text(json.dumps({
                    "metric_spec_version": METRIC_SPEC_VERSION,
                    "sequence": name,
                    "trajectory": {"sample_count": samples, "position": {
                        "ate_raw_m": {"rmse": ate}
                    }},
                    "network": {"available": False},
                }), encoding="utf-8")
            summary = summarize_run(root, ["short", "long"])
            mean = summary["aggregate"]["macro_mean"][
                "trajectory.position.ate_raw_m.rmse"
            ]
            self.assertEqual(mean, 5.0)


class NetworkWindowContractTests(unittest.TestCase):
    def test_checkpoint_model_param_controls_evaluation_window(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "best_model.pt"
            checkpoint.touch()
            (root / "model_param.json").write_text(json.dumps({
                "sampling_freq": 100,
                "window_time": 1.5,
                "window_size": 151,
            }), encoding="utf-8")
            contract = resolve_network_window(checkpoint, None)
            self.assertEqual(contract["sampling_frequency"], 100)
            self.assertEqual(contract["window_time"], 1.5)
            self.assertEqual(contract["window_size"], 151)

    def test_saved_model_and_run_config_must_agree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "best_model.pt"
            checkpoint.touch()
            (root / "model_param.json").write_text(json.dumps({
                "sampling_freq": 100,
                "window_time": 1.5,
                "window_size": 151,
            }), encoding="utf-8")
            config = root / "config_used.json"
            config.write_text(json.dumps({
                "data": {
                    "imu_freq_net": 100,
                    "window_time": 0.6,
                    "window_size": 61,
                },
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "window mismatch"):
                resolve_network_window(checkpoint, config)


if __name__ == "__main__":
    unittest.main()
