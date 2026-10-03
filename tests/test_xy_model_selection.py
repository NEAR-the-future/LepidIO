from types import SimpleNamespace
import unittest

import numpy as np

from learning.train import (
    _is_better_xy_candidate,
    _planar_relative_rmse,
    _trajectory_metrics,
)


class XYModelSelectionTests(unittest.TestCase):
    def test_planar_relative_rmse_ignores_vertical_error(self):
        gt = np.zeros((7, 3), dtype=np.float64)
        estimate = gt.copy()
        estimate[:, 2] = np.arange(7, dtype=np.float64) * 100.0

        self.assertEqual(_planar_relative_rmse(estimate, gt, offset=2), 0.0)

    def test_xy_candidate_uses_worst_sequence_inside_ate_tie_band(self):
        best = {
            "trajectory_xy_macro_ate_rmse": 1.0,
            "trajectory_xy_worst_ate_rmse": 2.0,
            "trajectory_xy_macro_rte_5s": 1.0,
            "trajectory_xy_macro_rte_1s": 0.5,
        }
        candidate = {
            "trajectory_xy_macro_ate_rmse": 1.01,
            "trajectory_xy_worst_ate_rmse": 1.5,
            "trajectory_xy_macro_rte_5s": 1.2,
            "trajectory_xy_macro_rte_1s": 0.6,
        }

        self.assertTrue(_is_better_xy_candidate(candidate, best))

    def test_trajectory_metrics_reports_world_xy_ate_and_rte(self):
        positions = np.column_stack([
            np.arange(8, dtype=np.float64),
            np.zeros(8, dtype=np.float64),
            np.arange(8, dtype=np.float64) * 10.0,
        ])
        dataset = SimpleNamespace(
            stride=1,
            window_size=2,
            sampling_freq=1,
            sequence_records=[{
                "name": "sequence",
                "sample_begin": 0,
                "sample_end": 6,
                "positions": positions,
                "quaternions": np.tile([0.0, 0.0, 0.0, 1.0], (8, 1)),
            }],
        )
        # X is predicted exactly; Z is deliberately wrong and must not affect XY.
        predictions = np.tile([1.0, 0.0, 0.0], (6, 1))

        result = _trajectory_metrics(dataset, predictions)["sequence"]

        self.assertEqual(result["trajectory_xy_ate_rmse"], 0.0)
        self.assertEqual(result["trajectory_xy_rte_1s"], 0.0)
        self.assertEqual(result["trajectory_xy_rte_5s"], 0.0)
        self.assertGreater(result["trajectory_ate_rmse"], 0.0)


if __name__ == "__main__":
    unittest.main()
