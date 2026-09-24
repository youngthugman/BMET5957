"""Kye-1D_CNN per-second feature cache, adapted for explicit patient keys.

Only persistence and identifiers differ from the branch implementation.  The
35-channel representation and four context lengths are intentionally fixed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CONTEXT_WINDOWS = (31, 61, 91, 121)
CHANNEL_NAMES = (
    "ecg_mean", "ecg_std", "ecg_min", "ecg_max", "ecg_median", "ecg_rms",
    "ecg_range", "ecg_abs_mean", "ecg_q10", "ecg_q25", "ecg_q75", "ecg_q90",
    "ecg_zero_crossings", "ecg_mean_abs_diff", "ecg_slope",
    "spo2_mean", "spo2_std", "spo2_min", "spo2_max", "spo2_median", "spo2_range",
    "spo2_q10", "spo2_q25", "spo2_q75", "spo2_q90", "spo2_slope",
    "spo2_delta_1s", "spo2_delta_5s", "spo2_drop_from_31s_max",
    "spo2_drop_from_61s_max", "spo2_below_90_31s", "spo2_below_92_31s",
    "spo2_below_95_31s", "ecg_std_31s", "ecg_rms_31s",
)
assert len(CHANNEL_NAMES) == 35


def _second_blocks(signal, sample_rate, seconds):
    signal = np.asarray(signal, dtype=np.float32).ravel()
    rate = int(round(sample_rate))
    if rate <= 0 or not np.isclose(rate, sample_rate):
        raise ValueError("CNN source pipeline requires an integer sample rate")
    required = seconds * rate
    if signal.size < required:
        signal = np.pad(signal, (0, required - signal.size), mode="edge")
    return signal[:required].reshape(seconds, rate)


def _slope(blocks):
    t = np.arange(blocks.shape[1], dtype=np.float32)
    t -= t.mean(); denominator = np.sum(t * t)
    return ((blocks - blocks.mean(1, keepdims=True)) @ t) / max(denominator, 1.)


def build_patient_features(ecg, spo2, ecg_rate, spo2_rate, seconds):
    """Return the branch's 35 normalized-later channels at one row per second."""
    e, s = _second_blocks(ecg, ecg_rate, seconds), _second_blocks(spo2, spo2_rate, seconds)
    s[(s <= 0) | (s > 100)] = np.nan
    e_quantiles = np.nanquantile(e, (.10, .25, .75, .90), axis=1).T
    s_quantiles = np.nanquantile(s, (.10, .25, .75, .90), axis=1).T
    current_spo2 = np.nanmedian(s, axis=1)
    current_spo2 = pd.Series(current_spo2).interpolate(limit_direction="both").to_numpy()
    ecg_rms = np.sqrt(np.nanmean(e * e, axis=1))
    ecg_std = np.nanstd(e, axis=1)
    spo2_series = pd.Series(current_spo2)
    channels = [
        np.nanmean(e, 1), ecg_std, np.nanmin(e, 1), np.nanmax(e, 1), np.nanmedian(e, 1),
        ecg_rms, np.nanmax(e, 1) - np.nanmin(e, 1), np.nanmean(np.abs(e), 1),
        *e_quantiles.T, np.sum(np.diff(np.signbit(e), axis=1), axis=1),
        np.nanmean(np.abs(np.diff(e, axis=1)), 1), _slope(e),
        np.nanmean(s, 1), np.nanstd(s, 1), np.nanmin(s, 1), np.nanmax(s, 1),
        np.nanmedian(s, 1), np.nanmax(s, 1) - np.nanmin(s, 1), *s_quantiles.T, _slope(s),
        spo2_series.diff().to_numpy(), spo2_series.diff(5).to_numpy(),
        (spo2_series.rolling(31, center=True, min_periods=1).max() - spo2_series).to_numpy(),
        (spo2_series.rolling(61, center=True, min_periods=1).max() - spo2_series).to_numpy(),
        *((spo2_series < threshold).rolling(31, center=True, min_periods=1).mean().to_numpy()
          for threshold in (90, 92, 95)),
        pd.Series(ecg_std).rolling(31, center=True, min_periods=1).mean().to_numpy(),
        pd.Series(ecg_rms).rolling(31, center=True, min_periods=1).mean().to_numpy(),
    ]
    result = np.column_stack(channels).astype(np.float32)
    assert result.shape == (seconds, 35)
    return result


def build_cnn_features(data, label_lengths):
    """Build patient arrays without consulting the XGBoost feature matrix."""
    from python_xgboost.run_xgboost import scalar_rate
    features = []
    for index, seconds in enumerate(label_lengths):
        features.append(build_patient_features(
            data["ECG"][index], data["SpO2"][index],
            scalar_rate(data["SR_ECG"], index, "SR_ECG"),
            scalar_rate(data["SR_SpO2"], index, "SR_SpO2"), seconds))
    return features
