
"""ECG and SpO2 feature extraction."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

from data_loader import clean_labels, scalar_rate


def _load_qrs_detector():
    """Load the submitted Group 5 detector from its original source file."""

    detector_path = (
        Path(__file__).resolve().parents[1]
        / "Submission2_Group5_FINAL"
        / "Submission2_Group5_FINAL"
        / "reference-data"
        / "Anthony-V2"
        / "Code"
        / "qrs_detector_causal.py"
    )

    if not detector_path.is_file():
        raise ImportError(
            f"Group 5 QRS detector not found: {detector_path}"
        )

    spec = importlib.util.spec_from_file_location(
        "submission2_group5_qrs_detector",
        detector_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load Group 5 QRS detector: {detector_path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.detect_qrs_causal


# Keep the externally supplied QRS detector separate from our own feature
# extraction code.
detect_qrs_causal = _load_qrs_detector()


def centred_slope(series, window):
    """Calculate a centred rolling least-squares slope."""

    t = pd.Series(np.arange(len(series), dtype=float))
    y = pd.Series(series, dtype=float)

    roll = y.rolling(
        window,
        center=True,
        min_periods=max(3, window // 4),
    )

    mt = t.rolling(
        window,
        center=True,
        min_periods=max(3, window // 4),
    ).mean()

    numerator = (
        (t * y).rolling(
            window,
            center=True,
            min_periods=max(3, window // 4),
        ).mean()
        - mt * roll.mean()
    )

    denominator = (
        (t * t).rolling(
            window,
            center=True,
            min_periods=max(3, window // 4),
        ).mean()
        - mt * mt
    )

    return (numerator / denominator).to_numpy()


def causal_slope(series, window, min_periods):
    """Rolling least-squares slope using only current and preceding rows."""

    t = pd.Series(np.arange(len(series), dtype=float))
    y = pd.Series(series, dtype=float)

    roll = y.rolling(
        window,
        min_periods=min_periods,
    )

    mt = t.rolling(
        window,
        min_periods=min_periods,
    ).mean()

    numerator = (
        (t * y).rolling(
            window,
            min_periods=min_periods,
        ).mean()
        - mt * roll.mean()
    )

    denominator = (
        (t * t).rolling(
            window,
            min_periods=min_periods,
        ).mean()
        - mt * mt
    )

    return (numerator / denominator).to_numpy()


def extract_ecg_features(qrs, sample_rate, n_seconds):
    """Convert detected QRS positions into one feature row per second."""

    qrs = np.asarray(qrs, dtype=float).ravel()
    qrs = qrs[np.isfinite(qrs)]

    # MATLAB QRS indices are one-based.
    # The constant offset has no effect on RR intervals.
    qrs_seconds = (qrs - 1.0) / sample_rate

    rr = np.diff(qrs_seconds)
    rr[(rr < 0.30) | (rr > 2.00)] = np.nan

    rr_time = (qrs_seconds[:-1] + qrs_seconds[1:]) / 2.0
    second = np.floor(rr_time).astype(int)

    valid_second = (
        (second >= 0)
        & (second < n_seconds)
    )

    # Average valid RR intervals falling within each second.
    sums = np.bincount(
        second[valid_second & np.isfinite(rr)],
        weights=rr[valid_second & np.isfinite(rr)],
        minlength=n_seconds,
    )

    counts = np.bincount(
        second[valid_second & np.isfinite(rr)],
        minlength=n_seconds,
    )

    current_rr = np.divide(
        sums,
        counts,
        out=np.full(n_seconds, np.nan),
        where=counts > 0,
    )

    current_rr = (
        pd.Series(current_rr)
        .interpolate(limit=2, limit_direction="both")
        .to_numpy()
    )

    rr_series = pd.Series(current_rr)

    window = 41  # approximately +/-20 seconds

    rolling = rr_series.rolling(
        window,
        center=True,
        min_periods=5,
    )

    rr_diff = rr_series.diff()

    beat_seconds = np.floor(qrs_seconds).astype(int)
    beat_seconds = beat_seconds[
        (beat_seconds >= 0)
        & (beat_seconds < n_seconds)
    ]

    beat_count = pd.Series(
        np.bincount(beat_seconds, minlength=n_seconds)
    ).rolling(
        window,
        center=True,
        min_periods=1,
    ).sum().to_numpy()

    # These original ten columns are retained exactly as in the baseline.
    baseline_columns = [
        current_rr,
        60.0 / current_rr,
        rolling.mean(),
        rolling.std(),
        np.sqrt(
            rr_diff.pow(2).rolling(
                window,
                center=True,
                min_periods=4,
            ).mean()
        ),
        (
            (rr_diff.abs() > 0.05)
            .astype(float)
            .rolling(
                window,
                center=True,
                min_periods=4,
            )
            .mean()
        ),
        rolling.min(),
        rolling.max(),
        beat_count,
        centred_slope(current_rr, window),
    ]

    names = [
        "ecg_rr_current",
        "ecg_hr_current",
        "ecg_rr_mean_41s",
        "ecg_rr_std_41s",
        "ecg_rmssd_41s",
        "ecg_pnn50_41s",
        "ecg_rr_min_41s",
        "ecg_rr_max_41s",
        "ecg_beat_count_41s",
        "ecg_rr_slope_41s",
    ]

    columns = list(baseline_columns)

    # Everything below is causal:
    # pandas' default rolling alignment ends at the current time, while
    # positive shifts refer only to earlier seconds.
    hr_series = pd.Series(60.0 / current_rr)

    for signal_name, series in (
        ("rr", rr_series),
        ("hr", hr_series),
    ):
        for lag in (5, 10, 20):
            columns.append(
                (series - series.shift(lag)).to_numpy()
            )
            names.append(
                f"ecg_{signal_name}_change_{lag}s"
            )

    rr_means = {}
    hr_means = {}

    for short_window in (5, 10, 20):
        min_periods = max(2, short_window // 4)

        rr_roll = rr_series.rolling(
            short_window,
            min_periods=min_periods,
        )

        rr_stats = {
            "mean": rr_roll.mean(),
            "std": rr_roll.std(),
            "min": rr_roll.min(),
            "max": rr_roll.max(),
        }

        rr_stats["range"] = (
            rr_stats["max"] - rr_stats["min"]
        )

        rr_means[short_window] = rr_stats["mean"]

        for statistic in (
            "mean",
            "std",
            "min",
            "max",
            "range",
        ):
            columns.append(
                rr_stats[statistic].to_numpy()
            )
            names.append(
                f"ecg_rr_{statistic}_{short_window}s"
            )

        columns.append(
            causal_slope(
                current_rr,
                short_window,
                min_periods,
            )
        )

        names.append(
            f"ecg_rr_slope_{short_window}s"
        )

        hr_roll = hr_series.rolling(
            short_window,
            min_periods=min_periods,
        )

        hr_stats = {
            "mean": hr_roll.mean(),
            "std": hr_roll.std(),
            "min": hr_roll.min(),
            "max": hr_roll.max(),
        }

        hr_stats["range"] = (
            hr_stats["max"] - hr_stats["min"]
        )

        hr_means[short_window] = hr_stats["mean"]

        for statistic in (
            "mean",
            "std",
            "min",
            "max",
            "range",
        ):
            columns.append(
                hr_stats[statistic].to_numpy()
            )
            names.append(
                f"ecg_hr_{statistic}_{short_window}s"
            )

    for hrv_window in (10, 20):
        min_periods = max(2, hrv_window // 4)

        diff_roll = rr_diff.rolling(
            hrv_window,
            min_periods=min_periods,
        )

        columns.extend([
            np.sqrt(
                diff_roll.apply(
                    lambda values: np.mean(values ** 2),
                    raw=True,
                )
            ).to_numpy(),
            rr_diff.abs().gt(0.05).rolling(
                hrv_window,
                min_periods=min_periods,
            ).mean().to_numpy(),
        ])

        names.extend([
            f"ecg_rmssd_{hrv_window}s",
            f"ecg_pnn50_{hrv_window}s",
        ])

    # The new long baselines are deliberately causal, unlike the retained
    # centred 41-second baseline features above.
    rr_mean_41s_causal = rr_series.rolling(
        41,
        min_periods=5,
    ).mean()

    hr_mean_41s_causal = hr_series.rolling(
        41,
        min_periods=5,
    ).mean()

    columns.extend([
        (rr_series - rr_mean_41s_causal).to_numpy(),
        (hr_series - hr_mean_41s_causal).to_numpy(),
    ])

    names.extend([
        "ecg_rr_vs_41s_mean",
        "ecg_hr_vs_41s_mean",
    ])

    for short_window in (5, 10, 20):
        columns.extend([
            (
                rr_means[short_window]
                - rr_mean_41s_causal
            ).to_numpy(),
            (
                hr_means[short_window]
                - hr_mean_41s_causal
            ).to_numpy(),
        ])

        names.extend([
            f"ecg_rr_mean_{short_window}s_minus_41s",
            f"ecg_hr_mean_{short_window}s_minus_41s",
        ])

    rr_delta = rr_series.diff()
    hr_delta = hr_series.diff()

    columns.extend([
        hr_delta.to_numpy(),
        rr_delta.to_numpy(),
        hr_delta.rolling(
            5,
            min_periods=2,
        ).std().to_numpy(),
        rr_delta.rolling(
            5,
            min_periods=2,
        ).std().to_numpy(),
    ])

    names.extend([
        "ecg_hr_delta_1s",
        "ecg_rr_delta_1s",
        "ecg_hr_delta_std_5s",
        "ecg_rr_delta_std_5s",
    ])

    features = np.column_stack(columns)

    assert features.shape[0] == n_seconds
    assert features.shape[1] == len(names)
    assert (
        len(names) == len(set(names))
        and all(name.startswith("ecg_") for name in names)
    )

    return features, names


def detect_qrs(ecg, sample_rate):
    """Run the submitted QRS detector and return zero-based sample indices."""

    rate = int(round(sample_rate))

    if not np.isclose(sample_rate, rate):
        raise ValueError(
            "The Group 5 QRS detector requires an integer ECG sample rate"
        )

    signal = np.asarray(ecg, dtype=float).ravel()

    if signal.size < 3 * rate:
        raise ValueError(
            "ECG recording is too short for the Group 5 QRS detector"
        )

    if not np.isfinite(signal).all():
        raise ValueError(
            "ECG contains non-finite samples"
        )

    return detect_qrs_causal(
        signal,
        fs=rate,
    )


def prepare_spo2(spo2, sample_rate, n_seconds):
    """Resample and lightly clean the SpO2 signal."""

    values = np.asarray(spo2, dtype=float).ravel()

    values[(values <= 0) | (values > 100)] = np.nan

    source_t = (
        (np.arange(values.size) + 0.5)
        / sample_rate
    )

    target_t = np.arange(n_seconds) + 0.5

    valid = np.isfinite(values)

    if valid.sum() < 2:
        return np.full(n_seconds, np.nan)

    result = np.interp(
        target_t,
        source_t[valid],
        values[valid],
        left=np.nan,
        right=np.nan,
    )

    result = pd.Series(result).interpolate(
        limit=10,
        limit_direction="both",
    ).to_numpy(copy=True)

    finite = np.isfinite(result)

    if finite.any():
        filled = (
            pd.Series(result)
            .ffill()
            .bfill()
            .to_numpy()
        )

        filtered = median_filter(
            filled,
            size=3,
            mode="nearest",
        )

        result[finite] = filtered[finite]

    return result


def extract_spo2_features(spo2, sample_rate, n_seconds):
    """Convert the SpO2 signal into one feature row per second."""

    values = prepare_spo2(
        spo2,
        sample_rate,
        n_seconds,
    )

    s = pd.Series(values)

    columns = [values]
    names = ["spo2_current"]

    rolls = {}

    # ========================================================
    # Existing rolling statistics
    # ========================================================

    for window in (21, 61):

        roll = s.rolling(
            window,
            center=True,
            min_periods=max(3, window // 4),
        )

        stats = {
            "mean": roll.mean(),
            "median": roll.median(),
            "std": roll.std(),
            "min": roll.min(),
            "max": roll.max(),
        }

        rolls[window] = stats

        for name in (
            "mean",
            "median",
            "std",
            "min",
            "max",
        ):

            columns.append(
                stats[name].to_numpy()
            )

            names.append(
                f"spo2_{name}_{window}s"
            )

        columns.append(
            (
                stats["max"]
                - stats["min"]
            ).to_numpy()
        )

        names.append(
            f"spo2_range_{window}s"
        )

    # ========================================================
    # Existing point-to-point changes
    # ========================================================

    for lag in (5, 10, 20):

        columns.extend([
            (
                s
                - s.shift(lag)
            ).to_numpy(),

            (
                s.shift(-lag)
                - s
            ).to_numpy(),
        ])

        names.extend([
            f"spo2_change_from_{lag}s_ago",
            f"spo2_change_to_{lag}s_ahead",
        ])

    # ========================================================
    # Existing local / future features
    # ========================================================

    columns.extend([

        (
            rolls[21]["max"]
            - s
        ).to_numpy(),

        (
            rolls[61]["median"]
            - s
        ).to_numpy(),

        (
            s.shift(-1)
            .rolling(
                20,
                min_periods=3,
            )
            .min()
            .shift(-19)
            - s
        ).to_numpy(),

        centred_slope(
            values,
            21,
        ),

        centred_slope(
            values,
            61,
        ),
    ])

    names.extend([
        "spo2_drop_from_local_max",
        "spo2_drop_from_60s_median",
        "spo2_future_20s_min_minus_current",
        "spo2_slope_21s",
        "spo2_slope_61s",
    ])

    # ========================================================
    # Existing threshold features
    # ========================================================

    for threshold in (
        90,
        92,
        95,
    ):

        columns.append(
            (
                (s < threshold)
                .astype(float)
                .rolling(
                    61,
                    center=True,
                    min_periods=15,
                )
                .mean()
                .to_numpy()
            )
        )

        names.append(
            f"spo2_fraction_below_{threshold}_61s"
        )

    # ========================================================
    # NEW: Short-term desaturation magnitude
    #
    # Positive value = SpO2 has dropped relative to the
    # earlier value.
    #
    # Example:
    #     current = 92
    #     10 seconds ago = 96
    #
    #     drop = 96 - 92 = 4
    # ========================================================

    for window in (
        3,
        5,
        10,
        15,
        20,
    ):

        drop = (
            s.shift(window)
            - s
        )

        columns.append(
            drop.to_numpy()
        )

        names.append(
            f"spo2_drop_{window}s"
        )

    # ========================================================
    # NEW: Short-term recovery magnitude
    #
    # Positive value = SpO2 has risen relative to the
    # earlier value.
    # ========================================================

    for window in (
        3,
        5,
        10,
        15,
        20,
    ):

        recovery = (
            s
            - s.shift(window)
        )

        columns.append(
            recovery.to_numpy()
        )

        names.append(
            f"spo2_recovery_{window}s"
        )

    # ========================================================
    # NEW: Maximum recent desaturation
    #
    # Instead of asking only:
    #
    #     "How much did SpO2 change over 10 seconds?"
    #
    # ask:
    #
    #     "What was the largest drop from any earlier point
    #      in this window to the current point?"
    #
    # This is useful because an apnoea-related desaturation
    # may not begin at exactly t-10 or t-20.
    # ========================================================

    for window in (
        10,
        20,
        30,
        60,
    ):

        previous_max = (
            s.shift(1)
            .rolling(
                window,
                min_periods=max(3, window // 4),
            )
            .max()
        )

        max_drop = (
            previous_max
            - s
        )

        columns.append(
            max_drop.to_numpy()
        )

        names.append(
            f"spo2_max_drop_{window}s"
        )

    # ========================================================
    # NEW: Distance below local baseline
    #
    # These distinguish:
    #
    #     94% while baseline is 95%
    #
    # from:
    #
    #     94% while baseline is 99%
    #
    # which can have very different meanings.
    # ========================================================

    for window in (
        21,
        61,
    ):

        median = rolls[window]["median"]
        mean = rolls[window]["mean"]
        maximum = rolls[window]["max"]

        columns.extend([

            (
                median
                - s
            ).to_numpy(),

            (
                mean
                - s
            ).to_numpy(),

            (
                maximum
                - s
            ).to_numpy(),
        ])

        names.extend([
            f"spo2_drop_below_median_{window}s",
            f"spo2_drop_below_mean_{window}s",
            f"spo2_drop_below_max_{window}s",
        ])

    # ========================================================
    # NEW: Actual duration below thresholds
    #
    # Existing features use FRACTION below threshold.
    # Here we explicitly count seconds.
    # ========================================================

    for threshold in (
        90,
        92,
        95,
    ):

        below = (
            s < threshold
        ).astype(float)

        duration = (
            below
            .rolling(
                61,
                center=True,
                min_periods=15,
            )
            .sum()
        )

        columns.append(
            duration.to_numpy()
        )

        names.append(
            f"spo2_seconds_below_{threshold}_61s"
        )

    # ========================================================
    # NEW: Threshold crossings
    #
    # A downward crossing is:
    #
    #     previous >= threshold
    #     current  < threshold
    #
    # We count crossings in the surrounding 61-second
    # window.
    # ========================================================

    for threshold in (
        90,
        92,
        95,
    ):

        previous = s.shift(1)

        downward_crossing = (
            (previous >= threshold)
            & (s < threshold)
        ).astype(float)

        upward_crossing = (
            (previous < threshold)
            & (s >= threshold)
        ).astype(float)

        down_count = (
            downward_crossing
            .rolling(
                61,
                center=True,
                min_periods=15,
            )
            .sum()
        )

        up_count = (
            upward_crossing
            .rolling(
                61,
                center=True,
                min_periods=15,
            )
            .sum()
        )

        columns.extend([
            down_count.to_numpy(),
            up_count.to_numpy(),
        ])

        names.extend([
            f"spo2_downward_crossings_{threshold}_61s",
            f"spo2_upward_crossings_{threshold}_61s",
        ])

    # ========================================================
    # NEW: Recent minimum / maximum distances
    #
    # These provide a simple description of where the current
    # point sits relative to the local extrema.
    # ========================================================

    for window in (
        10,
        20,
        30,
        60,
    ):

        minimum = (
            s.rolling(
                window,
                center=True,
                min_periods=max(3, window // 4),
            )
            .min()
        )

        maximum = (
            s.rolling(
                window,
                center=True,
                min_periods=max(3, window // 4),
            )
            .max()
        )

        columns.extend([
            (
                s
                - minimum
            ).to_numpy(),

            (
                maximum
                - s
            ).to_numpy(),
        ])

        names.extend([
            f"spo2_above_local_min_{window}s",
            f"spo2_below_local_max_{window}s",
        ])

    # ========================================================
    # NEW: Desaturation followed by recovery
    #
    # A useful event pattern is:
    #
    #       baseline
    #          ↓
    #       desaturation
    #          ↓
    #       recovery
    #
    # This feature measures whether the current/following
    # signal contains evidence of that pattern.
    # ========================================================

    drop_10 = (
        s.shift(10)
        - s
    )

    recovery_10 = (
        s.shift(-10)
        - s
    )

    columns.extend([

        (
            drop_10
            .clip(lower=0)
        ).to_numpy(),

        (
            recovery_10
            .clip(lower=0)
        ).to_numpy(),

        (
            drop_10.clip(lower=0)
            * recovery_10.clip(lower=0)
        ).to_numpy(),
    ])

    names.extend([
        "spo2_positive_drop_10s",
        "spo2_positive_recovery_10s",
        "spo2_drop_recovery_product_10s",
    ])

    # ========================================================
    # NEW: Local variability changes
    #
    # Detect whether variability itself is increasing or
    # decreasing around the current point.
    # ========================================================

    std_21 = rolls[21]["std"]
    std_61 = rolls[61]["std"]

    columns.extend([

        (
            std_21
            - std_61
        ).to_numpy(),

        (
            std_21
            / (std_61 + 1e-6)
        ).to_numpy(),
    ])

    names.extend([
        "spo2_std_21s_minus_61s",
        "spo2_std_21s_to_61s_ratio",
    ])

    # ========================================================
    # Final feature matrix
    # ========================================================

    features = np.column_stack(
        columns
    )

    assert (
        features.shape[0]
        == n_seconds
    )

    assert (
        features.shape[1]
        == len(names)
    )

    assert (
        len(names)
        == len(set(names))
    )

    assert all(
        name.startswith("spo2_")
        for name in names
    )

    return features, names

def extract_patient_features(data, patient_index):
    """Extract ECG and SpO2 features for one patient."""

    y = clean_labels(
        data["Class"][patient_index]
    )

    n = y.size

    ecg_rate = scalar_rate(
        data["SR_ECG"],
        patient_index,
        "SR_ECG",
    )

    spo2_rate = scalar_rate(
        data["SR_SpO2"],
        patient_index,
        "SR_SpO2",
    )

    detected_qrs = detect_qrs(
        data["ECG"][patient_index],
        ecg_rate,
    )

    # extract_ecg_features accepts MATLAB-style one-based indices.
    # The detector returns zero-based indices, so preserve the original
    # +1 conversion here.
    ecg_x, ecg_names = extract_ecg_features(
        detected_qrs + 1,
        ecg_rate,
        n,
    )

    spo2_x, spo2_names = extract_spo2_features(
        data["SpO2"][patient_index],
        spo2_rate,
        n,
    )

    x = np.column_stack(
        (ecg_x, spo2_x)
    ).astype(np.float32)

    assert (
        x.shape[0] == y.size == n
    ), "Feature extraction changed annotation length"

    return x, y, ecg_names + spo2_names
