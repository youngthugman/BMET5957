#!/usr/bin/env python3
"""Small patient-wise XGBoost benchmark for ProjectTrainData.mat."""

import argparse
import importlib.util
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.ndimage import median_filter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from tqdm import tqdm
from xgboost import XGBClassifier

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # scikit-learn before 1.1
    StratifiedGroupKFold = None


DEV20 = np.array([1, 3, 9, 12, 15, 17, 18, 30, 33, 36, 37, 38,
                  42, 47, 50, 57, 61, 72, 77, 94])
FIELDS = ("ECG", "SpO2", "Class", "SR_ECG", "SR_SpO2")
ECG_EXTRACTOR_VERSION = "submission2-group5-causal-qrs-v1"


def _load_qrs_detector():
    """Load the submitted Group 5 detector from its original source file."""
    detector_path = (Path(__file__).resolve().parents[1]
                     / "Submission2_Group5_FINAL" / "Submission2_Group5_FINAL"
                     / "reference-data" / "Anthony-V2" / "Code"
                     / "qrs_detector_causal.py")
    if not detector_path.is_file():
        raise ImportError(f"Group 5 QRS detector not found: {detector_path}")
    spec = importlib.util.spec_from_file_location("submission2_group5_qrs_detector",
                                                  detector_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Group 5 QRS detector: {detector_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.detect_qrs_causal


detect_qrs_causal = _load_qrs_detector()


def _scipy_cells(value):
    """Turn a regular MAT cell array (or scalar) into a patient list."""
    value = np.asarray(value)
    if value.dtype == object:
        return [np.asarray(x).squeeze() for x in value.ravel(order="F")]
    if value.ndim <= 1:
        return [value.squeeze()]
    return [value[:, i].squeeze() for i in range(value.shape[1])]


def _hdf5_value(handle, item):
    array = np.asarray(handle[item]) if isinstance(item, h5py.Reference) else np.asarray(item)
    # MATLAB stores arrays transposed in v7.3 files; vectors are unaffected.
    return array.T.squeeze()


def _hdf5_cells(handle, name):
    dataset = handle[name]
    if h5py.check_dtype(ref=dataset.dtype) is not None:
        return [_hdf5_value(handle, ref) for ref in dataset[()].ravel(order="F")]
    array = np.asarray(dataset).T
    if array.size == 1:
        return [array.squeeze()]
    if array.ndim == 1 or 1 in array.shape:
        return [np.asarray(x) for x in array.ravel(order="F")]
    return [array[:, i].squeeze() for i in range(array.shape[1])]


def load_training_data(path):
    """Load both ordinary and MATLAB v7.3/HDF5 MAT files."""
    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as handle:
            missing = [name for name in FIELDS if name not in handle]
            if missing:
                raise ValueError(f"MAT file is missing: {', '.join(missing)}")
            return {name: _hdf5_cells(handle, name) for name in FIELDS}
    raw = loadmat(path, variable_names=FIELDS, squeeze_me=False)
    missing = [name for name in FIELDS if name not in raw]
    if missing:
        raise ValueError(f"MAT file is missing: {', '.join(missing)}")
    return {name: _scipy_cells(raw[name]) for name in FIELDS}


def scalar_rate(values, patient_index, name):
    item = values[patient_index] if len(values) > 1 else values[0]
    item = np.asarray(item, dtype=float).ravel()
    if item.size != 1 or not np.isfinite(item[0]) or item[0] <= 0:
        raise ValueError(f"Invalid {name} for patient {patient_index + 1}")
    return float(item[0])


def clean_labels(value):
    array = np.asarray(value)
    if array.dtype.kind in "ui" and array.size and np.nanmax(array) <= 65535:
        text = "".join(chr(int(x)) for x in array.ravel(order="F") if int(x))
    else:
        text = "".join(str(x) for x in array.ravel(order="F"))
    labels = np.array([c.upper() for c in text if c.upper() in ("N", "A")])
    if labels.size == 0:
        raise ValueError("No N/A annotations found")
    return (labels == "A").astype(np.uint8)


def centred_slope(series, window):
    t = pd.Series(np.arange(len(series), dtype=float))
    y = pd.Series(series, dtype=float)
    roll = y.rolling(window, center=True, min_periods=max(3, window // 4))
    mt = t.rolling(window, center=True, min_periods=max(3, window // 4)).mean()
    numerator = (t * y).rolling(window, center=True,
                                min_periods=max(3, window // 4)).mean() - mt * roll.mean()
    denominator = (t * t).rolling(window, center=True,
                                  min_periods=max(3, window // 4)).mean() - mt * mt
    return (numerator / denominator).to_numpy()


def extract_ecg_features(qrs, sample_rate, n_seconds):
    qrs = np.asarray(qrs, dtype=float).ravel()
    qrs = qrs[np.isfinite(qrs)]
    # MATLAB QRS indices are one-based. The constant offset has no effect on RR.
    qrs_seconds = (qrs - 1.0) / sample_rate
    rr = np.diff(qrs_seconds)
    rr[(rr < 0.30) | (rr > 2.00)] = np.nan
    rr_time = (qrs_seconds[:-1] + qrs_seconds[1:]) / 2.0
    second = np.floor(rr_time).astype(int)
    valid_second = (second >= 0) & (second < n_seconds)

    sums = np.bincount(second[valid_second & np.isfinite(rr)],
                       weights=rr[valid_second & np.isfinite(rr)], minlength=n_seconds)
    counts = np.bincount(second[valid_second & np.isfinite(rr)], minlength=n_seconds)
    current_rr = np.divide(sums, counts, out=np.full(n_seconds, np.nan), where=counts > 0)
    current_rr = pd.Series(current_rr).interpolate(limit=2, limit_direction="both").to_numpy()
    rr_series = pd.Series(current_rr)
    window = 41  # approximately +/-20 seconds
    rolling = rr_series.rolling(window, center=True, min_periods=5)
    rr_diff = rr_series.diff()
    beat_seconds = np.floor(qrs_seconds).astype(int)
    beat_seconds = beat_seconds[(beat_seconds >= 0) & (beat_seconds < n_seconds)]
    beat_count = pd.Series(np.bincount(beat_seconds, minlength=n_seconds)).rolling(
        window, center=True, min_periods=1).sum().to_numpy()

    features = np.column_stack([
        current_rr, 60.0 / current_rr, rolling.mean(), rolling.std(),
        np.sqrt(rr_diff.pow(2).rolling(window, center=True, min_periods=4).mean()),
        (rr_diff.abs() > 0.05).astype(float).rolling(window, center=True,
                                                     min_periods=4).mean(),
        rolling.min(), rolling.max(), beat_count, centred_slope(current_rr, window),
    ])
    names = ["ecg_rr_current", "ecg_hr_current", "ecg_rr_mean_41s",
             "ecg_rr_std_41s", "ecg_rmssd_41s", "ecg_pnn50_41s",
             "ecg_rr_min_41s", "ecg_rr_max_41s", "ecg_beat_count_41s",
             "ecg_rr_slope_41s"]
    return features, names


def detect_qrs(ecg, sample_rate):
    """Run the submitted QRS detector and return zero-based sample indices."""
    rate = int(round(sample_rate))
    if not np.isclose(sample_rate, rate):
        raise ValueError("The Group 5 QRS detector requires an integer ECG sample rate")
    signal = np.asarray(ecg, dtype=float).ravel()
    if signal.size < 3 * rate:
        raise ValueError("ECG recording is too short for the Group 5 QRS detector")
    if not np.isfinite(signal).all():
        raise ValueError("ECG contains non-finite samples")
    return detect_qrs_causal(signal, fs=rate)


def prepare_spo2(spo2, sample_rate, n_seconds):
    values = np.asarray(spo2, dtype=float).ravel()
    values[(values <= 0) | (values > 100)] = np.nan
    source_t = (np.arange(values.size) + 0.5) / sample_rate
    target_t = np.arange(n_seconds) + 0.5
    valid = np.isfinite(values)
    if valid.sum() < 2:
        return np.full(n_seconds, np.nan)
    result = np.interp(target_t, source_t[valid], values[valid], left=np.nan, right=np.nan)
    result = pd.Series(result).interpolate(
        limit=10, limit_direction="both").to_numpy(copy=True)
    finite = np.isfinite(result)
    if finite.any():
        filled = pd.Series(result).ffill().bfill().to_numpy()
        filtered = median_filter(filled, size=3, mode="nearest")
        result[finite] = filtered[finite]
    return result


def extract_spo2_features(spo2, sample_rate, n_seconds):
    values = prepare_spo2(spo2, sample_rate, n_seconds)
    s = pd.Series(values)
    columns, names = [values], ["spo2_current"]
    rolls = {}
    for window in (21, 61):
        roll = s.rolling(window, center=True, min_periods=max(3, window // 4))
        stats = {"mean": roll.mean(), "median": roll.median(), "std": roll.std(),
                 "min": roll.min(), "max": roll.max()}
        rolls[window] = stats
        for name in ("mean", "median", "std", "min", "max"):
            columns.append(stats[name].to_numpy())
            names.append(f"spo2_{name}_{window}s")
        columns.append((stats["max"] - stats["min"]).to_numpy())
        names.append(f"spo2_range_{window}s")
    for lag in (5, 10, 20):
        columns.extend([(s - s.shift(lag)).to_numpy(), (s.shift(-lag) - s).to_numpy()])
        names.extend([f"spo2_change_from_{lag}s_ago", f"spo2_change_to_{lag}s_ahead"])
    columns.extend([(rolls[21]["max"] - s).to_numpy(),
                    (rolls[61]["median"] - s).to_numpy(),
                    (s.shift(-1).rolling(20, min_periods=3).min().shift(-19) - s).to_numpy(),
                    centred_slope(values, 21), centred_slope(values, 61)])
    names.extend(["spo2_drop_from_local_max", "spo2_drop_from_60s_median",
                  "spo2_future_20s_min_minus_current", "spo2_slope_21s", "spo2_slope_61s"])
    for threshold in (90, 92, 95):
        columns.append((s < threshold).astype(float).rolling(61, center=True,
                                                             min_periods=15).mean().to_numpy())
        names.append(f"spo2_fraction_below_{threshold}_61s")
    return np.column_stack(columns), names


def extract_patient_features(data, patient_index):
    y = clean_labels(data["Class"][patient_index])
    n = y.size
    ecg_rate = scalar_rate(data["SR_ECG"], patient_index, "SR_ECG")
    spo2_rate = scalar_rate(data["SR_SpO2"], patient_index, "SR_SpO2")
    detected_qrs = detect_qrs(data["ECG"][patient_index], ecg_rate)
    # extract_ecg_features accepts MATLAB-style one-based indices. Preserve that
    # interface while supplying the detector's documented zero-based output.
    ecg_x, ecg_names = extract_ecg_features(detected_qrs + 1, ecg_rate, n)
    spo2_x, spo2_names = extract_spo2_features(data["SpO2"][patient_index], spo2_rate, n)
    x = np.column_stack((ecg_x, spo2_x)).astype(np.float32)
    assert x.shape[0] == y.size == n, "Feature extraction changed annotation length"
    return x, y, ecg_names + spo2_names


def build_feature_cache(data_path, selected_patients, cache_path):
    data = load_training_data(data_path)
    n_patients = len(data["Class"])
    if selected_patients is None:
        selected_patients = np.arange(1, n_patients + 1)
    if selected_patients.max() > n_patients:
        raise ValueError(f"Requested patient {selected_patients.max()}, but MAT has {n_patients}")
    all_x, all_y, all_ids, feature_names = [], [], [], None
    for patient_id in tqdm(selected_patients, desc="Extracting patient features"):
        x, y, names = extract_patient_features(data, patient_id - 1)
        if feature_names is not None:
            assert names == feature_names
        feature_names = names
        all_x.append(x)
        all_y.append(y)
        all_ids.append(np.full(y.size, patient_id, dtype=np.int16))
    x, y, patient_id = np.concatenate(all_x), np.concatenate(all_y), np.concatenate(all_ids)
    assert x.shape[0] == y.size == patient_id.size
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, X=x, y=y, patient_id=patient_id,
                        feature_names=np.asarray(feature_names), selected_patients=selected_patients,
                        ecg_extractor_version=np.asarray(ECG_EXTRACTOR_VERSION))
    return x, y, patient_id, np.asarray(feature_names)


def load_or_build_cache(args, selected_patients, cache_path):
    if cache_path.exists() and not args.rebuild_cache:
        cached = np.load(cache_path, allow_pickle=False)
        cached_version = (str(cached["ecg_extractor_version"])
                          if "ecg_extractor_version" in cached.files else None)
        if cached_version != ECG_EXTRACTOR_VERSION:
            print("Cache uses a different ECG extractor; rebuilding it.")
        else:
            cached_patients = cached["selected_patients"]
            is_all_cache = np.array_equal(cached_patients,
                                          np.arange(1, cached_patients.max() + 1))
            selection_matches = (args.patients == "all" and is_all_cache) or (
                args.patients == "dev20" and np.array_equal(cached_patients, DEV20))
            if selection_matches:
                print(f"Loading feature cache: {cache_path}")
                return cached["X"], cached["y"], cached["patient_id"], cached["feature_names"]
            print("Cache patient selection differs; rebuilding it.")
    if not args.data:
        raise SystemExit("--data is required when a matching feature cache does not exist")
    return build_feature_cache(Path(args.data), selected_patients, cache_path)


def metrics(y, probability, threshold=0.5):
    prediction = probability >= threshold
    return np.array([recall_score(y, prediction, zero_division=0),
                     precision_score(y, prediction, zero_division=0),
                     f1_score(y, prediction, zero_division=0),
                     accuracy_score(y, prediction)])


def make_model(device, weight):
    return XGBClassifier(objective="binary:logistic", n_estimators=600,
                         learning_rate=0.05, max_depth=6, min_child_weight=5,
                         subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                         tree_method="hist", device=device, random_state=42,
                         scale_pos_weight=weight, n_jobs=-1)


def choose_device(requested):
    if requested == "cpu":
        return "cpu"
    try:
        subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=True, timeout=5)
    except (FileNotFoundError, subprocess.SubprocessError):
        print("CUDA requested, but no working NVIDIA GPU was detected; falling back to CPU.")
        return "cpu"
    print("CUDA requested and an NVIDIA GPU was detected.")
    return "cuda"


def fit_model(x, y, device, weight):
    model = make_model(device, weight)
    try:
        model.fit(x, y)
        return model, device
    except Exception as error:
        if device != "cuda":
            raise
        print(f"CUDA training failed ({error}); retrying this and later models on CPU.")
        model = make_model("cpu", weight)
        model.fit(x, y)
        return model, "cpu"


def patient_performance(y, probability, patient_id, best_threshold):
    """Summarise OOF performance separately for every selected patient."""
    rows = []
    for patient in np.unique(patient_id):
        selected = patient_id == patient
        patient_y = y[selected]
        patient_probability = probability[selected]
        scores_050 = metrics(patient_y, patient_probability)
        scores_best = metrics(patient_y, patient_probability, best_threshold)
        a_seconds = int(patient_y.sum())
        rows.append({
            "patient_id": int(patient),
            "seconds": int(selected.sum()),
            "A_seconds": a_seconds,
            "N_seconds": int(selected.sum() - a_seconds),
            "A_prevalence": float(patient_y.mean()),
            "sensitivity_050": scores_050[0],
            "ppv_050": scores_050[1],
            "f1_050": scores_050[2],
            "accuracy_050": scores_050[3],
            "mean_probability_A": float(patient_probability.mean()),
            "sensitivity_best_threshold": scores_best[0],
            "ppv_best_threshold": scores_best[1],
            "f1_best_threshold": scores_best[2],
            "accuracy_best_threshold": scores_best[3],
        })
    return pd.DataFrame(rows)


def run_cross_validation(x, y, patient_id, feature_names, device, results_dir,
                         cv_mode="5fold", feature_set="all", train_final_model=True):
    assert x.shape[0] == y.size == patient_id.size
    assert set(np.unique(y)).issubset({0, 1}) and 1 in y, "A must be positive class 1"
    patients = np.unique(patient_id)
    if cv_mode == "5fold" and patients.size < 5:
        raise ValueError("Five-fold CV requires at least five patients")
    if cv_mode == "logo":
        if patients.size < 2:
            raise ValueError("LOPO CV requires at least two patients")
        splitter = LeaveOneGroupOut()
    else:
        splitter = (StratifiedGroupKFold(5, shuffle=True, random_state=42)
                    if StratifiedGroupKFold else GroupKFold(5))
    oof = np.full(y.size, np.nan, dtype=np.float32)
    oof_count = np.zeros(y.size, dtype=np.uint8)
    fold_metrics = []
    splits = splitter.split(x, y, patient_id)
    total_splits = patients.size if cv_mode == "logo" else 5
    no_patient_overlap = True
    for fold, (train, validation) in enumerate(splits, 1):
        train_patients, validation_patients = np.unique(patient_id[train]), np.unique(patient_id[validation])
        overlap = np.intersect1d(train_patients, validation_patients).size > 0
        no_patient_overlap = no_patient_overlap and not overlap
        assert not overlap, "Training and validation patients must not overlap"
        negative, positive = np.bincount(y[train], minlength=2)
        if positive == 0:
            raise ValueError(f"Fold {fold} training data has no A labels")
        weight = float(np.sqrt(negative / positive))
        if cv_mode == "logo":
            print(f"\nLOPO {fold}/{total_splits}"
                  f"\n  held-out patient: {validation_patients[0]}"
                  f"\n  training patients: {train_patients.size}"
                  f"\n  training seconds: {train.size:,}"
                  f"\n  validation seconds: {validation.size:,}"
                  f"\n  scale_pos_weight: {weight:.4f}"
                  f"\n  A seconds: {(y[validation] == 1).sum():,}"
                  f"\n  N seconds: {(y[validation] == 0).sum():,}")
        else:
            print(f"\nFold {fold}\n  train patients: {train_patients.tolist()}"
                  f"\n  validation patients: {validation_patients.tolist()}"
                  f"\n  scale_pos_weight: {weight:.4f}")
        model, device = fit_model(x[train], y[train], device, weight)
        oof[validation] = model.predict_proba(x[validation])[:, 1]
        oof_count[validation] += 1
        scores = metrics(y[validation], oof[validation])
        fold_metrics.append(scores)
        print("  Sensitivity={:.4f}  PPV={:.4f}  F1={:.4f}  Accuracy={:.4f}".format(*scores))
    assert no_patient_overlap, "Training and validation patients overlapped"
    assert np.all(oof_count == 1) and np.isfinite(oof).all(), \
        "Every annotated second must have exactly one OOF prediction"
    fold_metrics = np.asarray(fold_metrics)
    if cv_mode == "5fold":
        print("\nMean fold metrics at threshold 0.50")
        print("  Sensitivity={:.4f}  PPV={:.4f}  F1={:.4f}  Accuracy={:.4f}".format(*fold_metrics.mean(0)))
        print(f"  Std fold F1={fold_metrics[:, 2].std():.4f}")
    pooled = metrics(y, oof)
    evaluation_name = "LOPO" if cv_mode == "logo" else "OOF"
    print(f"Pooled {evaluation_name} at threshold 0.50")
    print("  Sensitivity={:.4f}  PPV={:.4f}  F1={:.4f}  Accuracy={:.4f}".format(*pooled))
    thresholds = np.arange(0.10, 0.901, 0.01)
    swept = np.asarray([metrics(y, oof, threshold) for threshold in thresholds])
    best = int(np.argmax(swept[:, 2]))
    print("OOF threshold-tuning diagnostic (NOT an independent test result)")
    print("  threshold={:.2f}  Sensitivity={:.4f}  PPV={:.4f}  F1={:.4f}  Accuracy={:.4f}"
          .format(thresholds[best], *swept[best]))
    results_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(results_dir / f"oof_predictions_{cv_mode}_{feature_set}.npz",
                        probability_a=oof,
                        y_true=y, patient_id=patient_id, threshold_0_50=np.float32(0.5),
                        best_oof_threshold=np.float32(thresholds[best]))
    patient_results = patient_performance(y, oof, patient_id, thresholds[best])
    patient_results.to_csv(
        results_dir / f"patient_performance_{cv_mode}_{feature_set}.csv", index=False)

    if cv_mode == "logo":
        patient_scores = patient_results[["sensitivity_050", "ppv_050", "f1_050",
                                           "accuracy_050"]]
        means = patient_scores.mean()
        print("\nPatient-level metrics at threshold 0.50")
        print(f"  Mean patient Sensitivity={means['sensitivity_050']:.4f}"
              f"\n  Mean patient PPV={means['ppv_050']:.4f}"
              f"\n  Mean patient F1={means['f1_050']:.4f}"
              f"\n  Mean patient Accuracy={means['accuracy_050']:.4f}"
              f"\n  Std patient F1={patient_results['f1_050'].std(ddof=0):.4f}"
              f"\n  Median patient F1={patient_results['f1_050'].median():.4f}"
              f"\n  25th percentile patient F1={patient_results['f1_050'].quantile(.25):.4f}"
              f"\n  75th percentile patient F1={patient_results['f1_050'].quantile(.75):.4f}")
        rank_columns = ["patient_id", "A_seconds", "A_prevalence", "sensitivity_050",
                        "ppv_050", "f1_050", "accuracy_050"]
        rankable = patient_results[patient_results["A_seconds"] > 0]
        print("\nBest 10 patients by F1 at threshold 0.50 (A_seconds > 0)")
        print(rankable.nlargest(10, "f1_050")[rank_columns].to_string(index=False))
        print("\nWorst 10 patients by F1 at threshold 0.50 (A_seconds > 0)")
        print(rankable.nsmallest(10, "f1_050")[rank_columns].to_string(index=False))

    if train_final_model:
        negative, positive = np.bincount(y, minlength=2)
        final_model, _ = fit_model(x, y, device, float(np.sqrt(negative / positive)))
        importance = pd.DataFrame({"feature": feature_names,
                                   "importance": final_model.feature_importances_})
        importance = importance.sort_values("importance", ascending=False)
        importance.to_csv(results_dir / f"feature_importance_{feature_set}.csv", index=False)
        final_model.save_model(str(results_dir / f"xgboost_model_{feature_set}.json"))
        print("\nTop 20 feature importances")
        print(importance.head(20).to_string(index=False))

    return {"feature_set": feature_set, "features": x.shape[1], "pooled": pooled,
            "patient_results": patient_results}


def print_ablation_comparison(results, results_dir):
    """Save and display pooled and patient-level feature-ablation comparisons."""
    labels = {"ecg": "ECG only", "spo2": "SpO2 only", "all": "All"}
    summary_rows = []
    for result in results:
        patient_f1 = result["patient_results"]["f1_050"]
        pooled = result["pooled"]
        summary_rows.append({
            "feature_set": labels[result["feature_set"]],
            "features": result["features"],
            "sensitivity": pooled[0], "ppv": pooled[1], "pooled_f1": pooled[2],
            "accuracy": pooled[3], "mean_patient_f1": patient_f1.mean(),
            "median_patient_f1": patient_f1.median(),
            "std_patient_f1": patient_f1.std(ddof=0),
            "patient_f1_25th_percentile": patient_f1.quantile(.25),
            "patient_f1_75th_percentile": patient_f1.quantile(.75),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(results_dir / "feature_ablation_summary.csv", index=False)
    print("\nFeature ablation comparison at threshold 0.50")
    display_columns = ["feature_set", "features", "sensitivity", "ppv", "pooled_f1",
                       "accuracy", "median_patient_f1"]
    print(summary[display_columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    f1 = summary.set_index("feature_set")["pooled_f1"]
    print(f"\nAll - SpO2 F1 = {f1['All'] - f1['SpO2 only']:+.4f}")
    print(f"All - ECG F1 = {f1['All'] - f1['ECG only']:+.4f}")

    by_set = {result["feature_set"]: result["patient_results"].set_index("patient_id")
              for result in results}
    comparison = by_set["all"][["A_seconds", "A_prevalence"]].copy()
    for metric, source_column in (("f1", "f1_050"),
                                  ("sensitivity", "sensitivity_050"),
                                  ("ppv", "ppv_050")):
        for feature_set in ("ecg", "spo2", "all"):
            patient = by_set[feature_set]
            comparison[f"{metric}_{feature_set}"] = patient[source_column]
    comparison["all_minus_spo2_f1"] = comparison["f1_all"] - comparison["f1_spo2"]
    comparison["all_minus_ecg_f1"] = comparison["f1_all"] - comparison["f1_ecg"]
    comparison = comparison.reset_index()
    comparison.to_csv(results_dir / "feature_ablation_patient_comparison.csv", index=False)
    print("\n10 patients where adding ECG to SpO2 improves F1 the most")
    print(comparison.nlargest(10, "all_minus_spo2_f1").to_string(index=False))
    print("\n10 patients where adding ECG to SpO2 hurts F1 the most")
    print(comparison.nsmallest(10, "all_minus_spo2_f1").to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="Path to ProjectTrainData.mat")
    parser.add_argument("--patients", choices=("dev20", "all"), default="dev20")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cv", choices=("5fold", "logo"), default="5fold",
                        help="Patient-wise cross-validation method")
    parser.add_argument("--features", choices=("all", "ecg", "spo2", "compare"),
                        default="all", help="Feature set to evaluate")
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()
    if args.features == "compare" and args.cv != "logo":
        parser.error("--features compare requires --cv logo for LOPO feature ablation")
    root = Path(__file__).resolve().parent
    selected = DEV20 if args.patients == "dev20" else None
    cache_path = root / "cache" / "train_features.npz"
    x, y, patient_id, feature_names = load_or_build_cache(args, selected, cache_path)
    feature_names = feature_names.astype(str)
    feature_masks = {
        "ecg": np.char.startswith(feature_names, "ecg_"),
        "spo2": np.char.startswith(feature_names, "spo2_"),
        "all": np.ones(feature_names.size, dtype=bool),
    }
    if not feature_masks["ecg"].any() or not feature_masks["spo2"].any():
        raise ValueError("Cached feature_names must contain both ecg_ and spo2_ features")
    print(f"\nPatients: {np.unique(patient_id).size}\nSeconds: {y.size:,}"
          f"\nA labels: {(y == 1).sum():,}\nN labels: {(y == 0).sum():,}"
          f"\nFeatures: {x.shape[1]}\nFeature matrix RAM: {x.nbytes / 2**30:.2f} GiB")
    labels = {"ecg": "ECG only", "spo2": "SpO2 only", "all": "All"}
    requested_sets = ("ecg", "spo2", "all") if args.features == "compare" else (args.features,)
    device = choose_device(args.device)
    results = []
    for feature_set in requested_sets:
        mask = feature_masks[feature_set]
        print(f"\n{'=' * 72}\n{labels[feature_set]}: {mask.sum()} features\n{'=' * 72}")
        results.append(run_cross_validation(
            x[:, mask], y, patient_id, feature_names[mask], device, root / "results",
            args.cv, feature_set, train_final_model=args.features != "compare"))
    if args.features == "compare":
        print_ablation_comparison(results, root / "results")


if __name__ == "__main__":
    main()
