"""Focused tests for the XGBoost feature extraction pipeline."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from python_xgboost import run_xgboost


class ECGFeatureExtractionTests(unittest.TestCase):
    def test_multiscale_features_are_complete_unique_and_causal(self):
        # One valid RR observation per second, with enough history for every window.
        rr = 0.75 + 0.04 * np.sin(np.arange(80) / 5)
        qrs_seconds = np.concatenate(([0.1], 0.1 + np.cumsum(rr)))
        qrs = qrs_seconds * 100 + 1
        features, names = run_xgboost.extract_ecg_features(qrs, 100, 80)

        self.assertEqual(features.shape, (80, 131))
        self.assertEqual(len(names), 131)
        self.assertEqual(len(set(names)), len(names))
        self.assertTrue(all(name.startswith("ecg_") for name in names))
        self.assertEqual(names[:10], [
            "ecg_rr_current", "ecg_hr_current", "ecg_rr_mean_41s",
            "ecg_rr_std_41s", "ecg_rmssd_41s", "ecg_pnn50_41s",
            "ecg_rr_min_41s", "ecg_rr_max_41s", "ecg_beat_count_41s",
            "ecg_rr_slope_41s",
        ])

        # Appending future beats cannot alter any new feature at an earlier time.
        extended_rr = np.concatenate((rr, np.full(20, 1.5)))
        extended_qrs = np.concatenate(([0.1], 0.1 + np.cumsum(extended_rr))) * 100 + 1
        extended, extended_names = run_xgboost.extract_ecg_features(
            extended_qrs, 100, 100)
        self.assertEqual(names, extended_names)
        np.testing.assert_allclose(features[:50, 10:], extended[:50, 10:], equal_nan=True)

    def test_patient_features_detect_qrs_from_raw_ecg(self):
        data = {
            "ECG": [np.arange(500, dtype=float)],
            "SpO2": [np.full(5, 97.0)],
            "Class": [np.array(list("NNANN"))],
            "SR_ECG": [np.array([100])],
            "SR_SpO2": [np.array([1])],
        }
        detections = np.array([10, 110, 210, 310, 410])

        with patch.object(run_xgboost, "detect_qrs", return_value=detections) as detector:
            features, labels, names = run_xgboost.extract_patient_features(data, 0)

        detector.assert_called_once()
        np.testing.assert_array_equal(detector.call_args.args[0], data["ECG"][0])
        self.assertEqual(detector.call_args.args[1], 100.0)
        self.assertEqual(features.shape[0], labels.size)
        self.assertTrue(names[0].startswith("ecg_"))

    def test_legacy_cache_is_rebuilt(self):
        args = SimpleNamespace(rebuild_cache=False, patients="dev20", data="train.mat")
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "features.npz"
            np.savez(cache, selected_patients=run_xgboost.DEV20)
            expected = (np.empty((0, 0)), np.empty(0), np.empty(0), np.empty(0))
            with patch.object(run_xgboost, "build_feature_cache", return_value=expected) as build:
                actual = run_xgboost.load_or_build_cache(
                    args, run_xgboost.DEV20, cache)
            self.assertIs(actual, expected)
            build.assert_called_once()


if __name__ == "__main__":
    unittest.main()
