"""Kye-MLP raw-signal feature pipeline at one row per annotated second."""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_NAMES = (
    "ecg_mean", "ecg_std", "ecg_min", "ecg_max", "ecg_median", "ecg_rms",
    "ecg_range", "ecg_abs_mean", "ecg_q25", "ecg_q75", "ecg_zero_crossings",
    "ecg_mean_abs_diff", "spo2_mean", "spo2_std", "spo2_min", "spo2_max",
    "spo2_median", "spo2_range", "spo2_q25", "spo2_q75", "spo2_delta_1s",
    "spo2_delta_5s", "spo2_mean_31s", "spo2_std_31s", "spo2_min_31s",
    "spo2_max_31s", "spo2_below_90_31s", "spo2_below_92_31s",
    "spo2_below_95_31s", "ecg_std_31s",
)


def _blocks(signal, rate, seconds):
    rate = int(round(rate)); value = np.asarray(signal, np.float32).ravel()
    needed = rate * seconds
    if len(value) < needed: value = np.pad(value, (0, needed - len(value)), mode="edge")
    return value[:needed].reshape(seconds, rate)


def extract_patient_features(ecg, spo2, ecg_rate, spo2_rate, seconds):
    e, s = _blocks(ecg, ecg_rate, seconds), _blocks(spo2, spo2_rate, seconds)
    s[(s <= 0) | (s > 100)] = np.nan
    current = pd.Series(np.nanmedian(s, 1)).interpolate(limit_direction="both")
    ecg_std = np.nanstd(e, 1); rolling = current.rolling(31, center=True, min_periods=1)
    columns = [
        np.nanmean(e, 1), ecg_std, np.nanmin(e, 1), np.nanmax(e, 1), np.nanmedian(e, 1),
        np.sqrt(np.nanmean(e * e, 1)), np.nanmax(e, 1) - np.nanmin(e, 1),
        np.nanmean(np.abs(e), 1), np.nanquantile(e, .25, axis=1), np.nanquantile(e, .75, axis=1),
        np.sum(np.diff(np.signbit(e), axis=1), 1), np.nanmean(np.abs(np.diff(e, axis=1)), 1),
        np.nanmean(s, 1), np.nanstd(s, 1), np.nanmin(s, 1), np.nanmax(s, 1), np.nanmedian(s, 1),
        np.nanmax(s, 1) - np.nanmin(s, 1), np.nanquantile(s, .25, axis=1), np.nanquantile(s, .75, axis=1),
        current.diff().to_numpy(), current.diff(5).to_numpy(), rolling.mean().to_numpy(),
        rolling.std().to_numpy(), rolling.min().to_numpy(), rolling.max().to_numpy(),
        *((current < threshold).rolling(31, center=True, min_periods=1).mean().to_numpy()
          for threshold in (90, 92, 95)),
        pd.Series(ecg_std).rolling(31, center=True, min_periods=1).mean().to_numpy(),
    ]
    result = np.column_stack(columns).astype(np.float32)
    assert result.shape == (seconds, len(FEATURE_NAMES))
    return result


def build_mlp_features(data, label_lengths):
    from python_xgboost.run_xgboost import scalar_rate
    result = []
    for index, seconds in enumerate(label_lengths):
        result.append(extract_patient_features(
            data["ECG"][index], data["SpO2"][index],
            scalar_rate(data["SR_ECG"], index, "SR_ECG"),
            scalar_rate(data["SR_SpO2"], index, "SR_SpO2"), seconds))
    return result
