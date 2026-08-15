from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np

from serd.paper.metrics import absolute_coverage_error, empirical_crps, ensemble_spread, rank_histogram
from serd.paper.reconstruction import reconstruct_corrdiff, reconstruct_total_error, reconstruct_two_stage
from serd.paper.spec import (FORECAST_CHANNELS, PAPER_SPEC, PRESSURE_LEVELS_HPA,
                             SURFACE_CHANNEL_INDICES, file_candidates, select_date_split,
                             split_forecast_channels)


class PaperContractTests(unittest.TestCase):
    def test_dates_are_explicit(self):
        self.assertEqual(PAPER_SPEC.split_for(date(2023, 9, 30)), "train")
        self.assertEqual(PAPER_SPEC.split_for(date(2023, 10, 1)), "valid")
        self.assertEqual(PAPER_SPEC.split_for(date(2024, 1, 1)), "test")
        self.assertIsNone(PAPER_SPEC.split_for(date(2025, 1, 1)))
        paths = ["/x/20240101", "/x/20230930", "/x/20231001"]
        self.assertEqual(select_date_split(paths, "valid"), ["/x/20231001"])

    def test_channel_contract(self):
        self.assertEqual(len(FORECAST_CHANNELS), 45)
        self.assertEqual(len(PRESSURE_LEVELS_HPA), 8)
        self.assertEqual(SURFACE_CHANNEL_INDICES, (0, 9, 18, 27, 36))
        forecast = np.arange(45 * 2 * 3).reshape(45, 2, 3)
        surface, upper = split_forecast_channels(forecast)
        self.assertEqual(surface.shape, (5, 2, 3))
        self.assertEqual(upper.shape, (5, 8, 2, 3))
        np.testing.assert_array_equal(surface, forecast[list(SURFACE_CHANNEL_INDICES)])

    def test_err_filename_with_single_separator(self):
        with tempfile.TemporaryDirectory() as folder:
            expected = Path(folder) / "2020_01_08_48_err.npy"
            expected.touch()
            candidates = file_candidates(folder, "2020_01_08", 48, "_err")
            self.assertIn(expected, candidates)

    def test_empirical_crps_uses_k_squared_pair_term(self):
        ensemble = np.array([0.0, 2.0])
        observation = np.array(1.0)
        # term1=1; mean over all four ordered pairs=1; CRPS=1-0.5=0.5
        self.assertAlmostEqual(float(empirical_crps(ensemble, observation)), 0.5)

    def test_spread_coverage_and_rank_contract(self):
        ensemble = np.array([[0.0], [2.0]])
        self.assertAlmostEqual(float(ensemble_spread(ensemble).item()), np.sqrt(2.0))
        self.assertAlmostEqual(absolute_coverage_error(0.76, 0.80), 0.04)
        np.testing.assert_array_equal(rank_histogram(ensemble, np.array([1.0])), [0, 1, 0])

    def test_physical_reconstruction_for_all_method_families(self):
        forecast = np.zeros((45, 1, 1), dtype=np.float32)
        forecast[list(SURFACE_CHANNEL_INDICES)] = np.arange(5)[:, None, None]
        members = np.ones((16, 5, 1, 1), dtype=np.float32)
        expected_direct = np.broadcast_to(np.arange(5)[None] + 1, (16, 5))
        expected_two_stage = np.broadcast_to(np.arange(5)[None] + 2, (16, 5))
        np.testing.assert_allclose(reconstruct_total_error(forecast, members)[:, :, 0, 0], expected_direct)
        np.testing.assert_allclose(reconstruct_two_stage(forecast, np.ones((5, 1, 1)), members)[:, :, 0, 0], expected_two_stage)
        np.testing.assert_allclose(reconstruct_corrdiff(np.full((5, 1, 1), 3), members), 4)


if __name__ == "__main__":
    unittest.main()
