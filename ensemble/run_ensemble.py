#!/usr/bin/env python3
"""Resume-safe patient-wise CNN/MLP/XGBoost ensemble experiment."""
from __future__ import annotations

import argparse, json, sys, time, warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ensemble.alignment import (align_predictions, load_prediction_cache, make_shared_folds,
                                save_prediction_cache, second_indices)
from ensemble.cnn_adapter import CNNAdapter, MODEL_VERSION as CNN_VERSION, build_cnn_features
from ensemble.ensemble_methods import (best_threshold, diversity, make_meta_model, meta_features,
                                       metrics, weighted_search)
from ensemble.logging_utils import configure_logging
from ensemble.mlp_adapter import MLPAdapter, MODEL_VERSION as MLP_VERSION, build_mlp_features, data_loader as mlp_data_loader
from ensemble.source_fidelity import audit as audit_source
from ensemble.submission import create_submission, load_template
from ensemble.xgb_adapter import XGBAdapter, MODEL_VERSION as XGB_VERSION
from python_xgboost.run_xgboost import (SPO2_WINDOWS, build_feature_cache, extract_patient_features,
                                       load_or_build_cache, load_training_data)

ROOT = Path(__file__).resolve().parent
VERSIONS = {"cnn": CNN_VERSION, "mlp": MLP_VERSION, "xgb": XGB_VERSION}
SOURCE = {"cnn": "Kye-1D_CNN", "mlp": "Kye-MLP", "xgb": "19926-XGBoost-Tuning"}
FEATURE_VERSION = {"cnn": "Kye-1D_CNN-exact-blobs-35-channel-v2", "mlp": "Kye-MLP-exact-blobs-native-v2",
                   "xgb": "262-extended-v3"}


def load_or_build_native_features(name, raw, lengths, cache_path, builder, logger):
    """Cache native model inputs without object arrays or unsafe pickle loading."""
    expected = FEATURE_VERSION[name]
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        if str(cached.get("feature_version", "")) == expected and np.array_equal(cached["lengths"], lengths):
            logger.debug("Loaded %s native feature cache: %s", name, cache_path)
            joined, offsets = cached["features"], np.cumsum(cached["lengths"])[:-1]
            if name == "cnn": return [part.T for part in np.split(joined, offsets)]
            return joined, cached["feature_names"].astype(str)
        logger.debug("Rejected %s native feature cache: version or lengths differ", name)
    if name == "cnn":
        values = builder(raw, lengths); joined = np.concatenate([value.T for value in values])
        feature_names = np.asarray([], dtype=str)
    else:
        joined, native_y, feature_names = builder(raw)
        if len(native_y) != sum(lengths): raise ValueError("Native MLP label coverage mismatch")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, features=joined, lengths=np.asarray(lengths), feature_names=feature_names,
                        feature_version=np.asarray(expected))
    temporary.replace(cache_path)
    return values if name == "cnn" else (joined, feature_names)


def adapter(name, args):
    if name == "cnn": return CNNAdapter(args.device, args.cnn_epochs, args.cnn_batch_size)
    if name == "mlp": return MLPAdapter(args.device, args.mlp_epochs, args.mlp_batch_size)
    return XGBAdapter(args.device)


def _patient_lists(values, y, patient_id, mask):
    selected = np.unique(patient_id[mask])
    return ([values[int(patient) - 1] for patient in selected],
            [y[patient_id == patient] for patient in selected])


def fit_predict(name, representations, y, patient_id, train, valid, args, logger):
    model = adapter(name, args)
    try:
        if name == "cnn":
            train_x, train_y = _patient_lists(representations["cnn"], y, patient_id, train)
            valid_x, _ = _patient_lists(representations["cnn"], y, patient_id, valid)
            valid_y = [y[patient_id == patient] for patient in np.unique(patient_id[valid])]
            model.fit(train_x, train_y, logger, validation_features=valid_x, validation_labels=valid_y)
            probability = model.predict_proba(valid_x)
        elif name == "mlp":
            model.fit(representations["mlp"][train], y[train], patient_id[train], logger)
            probability = model.predict_proba(representations["mlp"][valid])
        else:
            model.fit(representations["xgb"][train], y[train], logger)
            probability = model.predict_proba(representations["xgb"][valid])
        assert len(probability) == int(np.sum(valid)), f"{name} validation coverage mismatch"
        return probability
    finally:
        model.close()


def model_report(name, y, probability, patient_id, results):
    threshold, diagnostic = best_threshold(y, probability); primary = metrics(y, probability)
    row = {"method": name, "evaluation_type": "OOF", "threshold": .5, **primary, "notes": ""}
    pd.DataFrame([{**{f"pooled_{key}_050": value for key, value in primary.items()},
                   "best_diagnostic_threshold": threshold,
                   **{f"best_threshold_{key}": value for key, value in diagnostic.items()}}]).to_csv(
                       results / f"model_performance_5fold_{name}.csv", index=False)
    patient_rows = []
    for patient in np.unique(patient_id):
        selected = patient_id == patient
        patient_rows.append({"patient_id": patient, "seconds": selected.sum(),
                             **{f"{key}_050": value for key, value in metrics(y[selected], probability[selected]).items()},
                             **{f"{key}_best": value for key, value in metrics(y[selected], probability[selected], threshold).items()}})
    pd.DataFrame(patient_rows).to_csv(results / f"patient_performance_5fold_{name}.csv", index=False)
    np.savez_compressed(results / f"oof_predictions_5fold_{name}.npz", probability_a=probability,
                        y_true=y, patient_id=patient_id, second_index=second_indices(patient_id),
                        threshold_0_50=np.float32(.5), best_oof_threshold=np.float32(threshold))
    return row


def save_method(name, y, probability, frame, results, evaluation_type, notes=""):
    threshold, diagnostic = best_threshold(y, probability); primary = metrics(y, probability)
    np.savez_compressed(results / f"oof_predictions_5fold_{name}.npz", probability_a=probability,
                        y_true=y, patient_id=frame.patient_id, second_index=frame.second_index,
                        threshold_0_50=np.float32(.5), best_oof_threshold=np.float32(threshold))
    model_report(name, y, probability, frame.patient_id.to_numpy(), results)
    return {"method": name, "evaluation_type": evaluation_type, "threshold": .5, **primary,
            "notes": notes + f"; diagnostic best threshold={threshold:.2f}, F1={diagnostic['f1']:.6f}"}


def _load_native_raw(path, require_qrs=True):
    raw = mlp_data_loader.load_training_data(path)
    if not require_qrs: return raw
    import h5py
    from scipy.io import loadmat
    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as handle:
            if "QRS" not in handle: raise ValueError("Native CNN requires QRS in ProjectTrainData.mat")
            raw["QRS"] = mlp_data_loader._hdf5_cells(handle, "QRS")
    else:
        qrs_mat = loadmat(path, variable_names=("QRS",), squeeze_me=False)
        if "QRS" not in qrs_mat: raise ValueError("Native CNN requires QRS in ProjectTrainData.mat")
        raw["QRS"] = mlp_data_loader._scipy_cells(qrs_mat["QRS"])
    return raw


def run_reproduce(args, logger):
    """Reproduce each branch's own native patient-fold evaluation independently."""
    from sklearn.model_selection import KFold, GroupKFold
    try:
        from sklearn.model_selection import StratifiedGroupKFold
    except ImportError:
        StratifiedGroupKFold = None
    results, cache = ROOT / "results", ROOT / "cache"
    results.mkdir(parents=True, exist_ok=True); cache.mkdir(parents=True, exist_ok=True)
    raw = _load_native_raw(args.data, require_qrs="cnn" in args.models)
    lengths = [len(mlp_data_loader.clean_labels(value)) for value in raw["Class"]]
    y = np.concatenate([mlp_data_loader.clean_labels(value) for value in raw["Class"]])
    patient_id = np.concatenate([np.full(length, i + 1, np.int16) for i, length in enumerate(lengths)])
    mlp_x = feature_names = None
    if "mlp" in args.models:
        mlp_x, native_y, feature_names = build_mlp_features(raw)
        if not np.array_equal(y, native_y): raise ValueError("Native MLP labels are not aligned")
        feature_hash = __import__("hashlib").sha256("\n".join(feature_names).encode()).hexdigest()
        logger.info("Native MLP feature count=%d feature-name SHA-256=%s", len(feature_names), feature_hash)
    for name in args.models:
        oof = np.full(len(y), np.nan, np.float32)
        if name == "cnn":
            cnn = build_cnn_features(raw, lengths)
            splits = KFold(n_splits=5, shuffle=True, random_state=42).split(np.arange(len(lengths)))
            for fold, (train_patients, valid_patients) in enumerate(splits, 1):
                model = CNNAdapter(args.device, args.cnn_epochs, args.cnn_batch_size)
                try:
                    tx, ty = [cnn[i] for i in train_patients], [y[patient_id == i + 1] for i in train_patients]
                    vx, vy = [cnn[i] for i in valid_patients], [y[patient_id == i + 1] for i in valid_patients]
                    model.fit(tx, ty, logger, validation_features=vx, validation_labels=vy, fold=fold)
                    positions = np.concatenate([np.flatnonzero(patient_id == i + 1) for i in valid_patients])
                    oof[positions] = model.predict_proba(vx)
                finally: model.close()
        else:
            splitter = (StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
                        if StratifiedGroupKFold else GroupKFold(n_splits=5))
            for fold, (train, valid) in enumerate(splitter.split(mlp_x, y, groups=patient_id), 1):
                model = MLPAdapter(args.device, args.mlp_epochs, args.mlp_batch_size)
                try:
                    model.fit(mlp_x[train], y[train], patient_id[train], logger)
                    oof[valid] = model.predict_proba(mlp_x[valid])
                finally: model.close()
        if not np.isfinite(oof).all(): raise RuntimeError(f"Native {name} reproduction has incomplete OOF predictions")
        score = metrics(y, oof); score["predicted_positive_fraction"] = float(np.mean(oof >= .5))
        pd.DataFrame([{ "model": name, **score }]).to_csv(results / f"native_reproduction_{name}.csv", index=False)
        np.savez_compressed(cache / f"native_reproduction_{name}.npz", probability_a=oof, y_true=y,
                            patient_id=patient_id, second_index=second_indices(patient_id), model_version=VERSIONS[name])
        logger.info("Native %s: sensitivity=%.4f PPV=%.4f F1=%.4f accuracy=%.4f predicted-positive=%.4f",
                    name.upper(), score["sensitivity"], score["ppv"], score["f1"], score["accuracy"], score["predicted_positive_fraction"])


def nested_stacking(frame, representations, row_fold, args, cache, logger, kind):
    output = np.full(len(frame), np.nan, np.float32)
    ids, y = frame.patient_id.to_numpy(), frame.y_true.to_numpy()
    for outer in range(1, 6):
        outer_train, outer_valid = row_fold != outer, row_fold == outer
        inner_path = cache / "nested" / f"outer_{outer}" / f"{kind}.npz"
        if inner_path.exists():
            loaded = np.load(inner_path); output[outer_valid] = loaded["probability_a"]; continue
        inner_fold = make_shared_folds(y[outer_train], ids[outer_train],
                                       cache / "nested" / f"outer_{outer}" / "inner_folds.csv")
        inner_prob = np.full((outer_train.sum(), 3), np.nan, np.float32)
        outer_positions = np.flatnonzero(outer_train)
        for inner in range(1, 6):
            train_local, valid_local = inner_fold != inner, inner_fold == inner
            for column, name in enumerate(("cnn", "mlp", "xgb")):
                train_mask, valid_mask = np.zeros(len(y), bool), np.zeros(len(y), bool)
                train_mask[outer_positions[train_local]] = True; valid_mask[outer_positions[valid_local]] = True
                inner_prob[valid_local, column] = fit_predict(
                    name, representations, y, ids, train_mask, valid_mask, args, logger)
        assert np.isfinite(inner_prob).all()
        meta = make_meta_model(kind).fit(meta_features(inner_prob), y[outer_train])
        base_outer = frame.loc[outer_valid, ["cnn_probability", "mlp_probability", "xgb_probability"]].to_numpy()
        output[outer_valid] = meta.predict_proba(meta_features(base_outer))[:, 1]
        inner_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(inner_path, probability_a=output[outer_valid], patient_id=ids[outer_valid])
    assert np.isfinite(output).all()
    return output


def run_cv(args, logger):
    results, cache = ROOT / "results", ROOT / "cache"; results.mkdir(parents=True, exist_ok=True)
    selected = None
    feature_cache = cache / "train_features_extended.npz"
    args.patients, args.rebuild_cache, args.spo2_windows = "all", args.rebuild_cache, "extended"
    x, y, patient_id, names = load_or_build_cache(args, selected, feature_cache)
    if x.shape[1] != 262: raise ValueError(f"Canonical extended feature count must be 262, got {x.shape[1]}")
    raw = _load_native_raw(args.data)
    label_lengths = [int(np.sum(patient_id == patient)) for patient in np.unique(patient_id)]
    cnn_features = load_or_build_native_features(
        "cnn", raw, label_lengths, cache / "cnn_native_features.npz", build_cnn_features, logger)
    mlp_features, mlp_names = load_or_build_native_features(
        "mlp", raw, label_lengths, cache / "mlp_native_features.npz", build_mlp_features, logger)
    assert len(mlp_features) == len(y) and sum(value.shape[1] for value in cnn_features) == len(y)
    logger.info("Native MLP features: %d; name SHA-256: %s", len(mlp_names), __import__('hashlib').sha256("\n".join(mlp_names).encode()).hexdigest())
    representations = {"cnn": cnn_features, "mlp": mlp_features, "xgb": x}
    row_fold = make_shared_folds(y, patient_id, results / "shared_patient_folds.csv")
    index = second_indices(patient_id); outputs = {}; timings = []
    logger.info("=" * 60); logger.info("ENSEMBLE EXPERIMENT"); logger.info("=" * 60)
    logger.info("Patients: %d\nSeconds: %,d\nDevice: %s\nOuter CV: 5 folds", len(np.unique(patient_id)), len(y), args.device.upper())
    for fold in range(1, 6):
        logger.info("\nFold %d/5", fold); train, valid = row_fold != fold, row_fold == fold
        assert not np.intersect1d(patient_id[train], patient_id[valid]).size
        for name in ("cnn", "mlp", "xgb"):
            path = cache / f"fold_{fold}" / f"{name}_predictions.npz"
            metadata = {"random_seed": 42, "feature_version": FEATURE_VERSION[name], "model_version": VERSIONS[name],
                        "validation_patients": np.unique(patient_id[valid]).tolist(), "branch_source": SOURCE[name]}
            start = datetime.now(timezone.utc); tick = time.monotonic()
            loaded = load_prediction_cache(path, metadata, logger)
            if loaded is None:
                probability = fit_predict(name, representations, y, patient_id, train, valid, args, logger)
                save_prediction_cache(path, patient_id=patient_id[valid], second_index=index[valid], y_true=y[valid],
                                      probability=probability, fold=fold, model_name=name, metadata=metadata)
            else: probability = loaded["probability_a"]
            elapsed = time.monotonic() - tick
            timings.append({"stage": name, "fold": fold, "start_time": start.isoformat(),
                            "end_time": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": elapsed})
            outputs.setdefault(name, []).append({"patient_id": patient_id[valid], "second_index": index[valid],
                                                 "y_true": y[valid], "probability_a": probability})
            logger.info("  %-4s ...... complete  F1=%.4f", name.upper(), metrics(y[valid], probability)["f1"])
        pd.DataFrame(timings).to_csv(results / "runtime_summary.csv", index=False)
        mean_fold = pd.DataFrame(timings).groupby("fold").elapsed_seconds.sum().mean()
        logger.info("  saved fold cache; estimated remaining %.1f min", mean_fold * (5 - fold) / 60)
    outputs = {name: {key: np.concatenate([part[key] for part in parts]) for key in parts[0]} for name, parts in outputs.items()}
    frame = align_predictions(outputs, results, args.write_aligned_csv); y_aligned = frame.y_true.to_numpy()
    p = frame[["cnn_probability", "mlp_probability", "xgb_probability"]].to_numpy()
    summary = [model_report(name, y_aligned, frame[f"{name}_probability"].to_numpy(), frame.patient_id.to_numpy(), results)
               for name in ("cnn", "mlp", "xgb")]
    base_summary = pd.DataFrame({
        "model": ["CNN", "MLP", "XGBoost"],
        "original_input_implementation": [SOURCE[name] for name in ("cnn", "mlp", "xgb")],
        "shared_fold_oof_f1": [row["f1"] for row in summary],
    })
    base_summary.to_csv(results / "base_model_summary.csv", index=False)
    logger.info("\n%-12s %-35s %s", "MODEL", "ORIGINAL INPUT IMPLEMENTATION", "SHARED-FOLD OOF F1")
    for row in base_summary.itertuples(index=False):
        logger.info("%-12s %-35s %.4f", row.model, row.original_input_implementation, row.shared_fold_oof_f1)
    if args.base_models_only:
        print(base_summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
        return
    hard = (np.sum(p >= .5, axis=1) >= 2).astype(np.uint8)
    np.savez_compressed(results / "oof_predictions_5fold_hard_majority_vote.npz", prediction=hard,
                        y_true=y_aligned, patient_id=frame.patient_id, second_index=frame.second_index)
    summary.append({"method": "hard_majority_vote", "evaluation_type": "OOF", "threshold": .5,
                    **metrics(y_aligned, hard), "notes": "two of three votes at 0.50"})
    summary.append(save_method("equal_soft_vote", y_aligned, p.mean(1), frame, results, "OOF"))
    search, best = weighted_search(y_aligned, p); search.to_csv(results / "weighted_voting_search.csv", index=False)
    weights = best[["cnn_weight", "mlp_weight", "xgb_weight"]].to_numpy(float)
    summary.append(save_method("weighted_soft_vote", y_aligned, p @ weights, frame, results, "diagnostic",
                               "DIAGNOSTIC — NOT AN INDEPENDENT TEST RESULT"))
    diversity(y_aligned, frame).to_csv(results / "model_diversity.csv", index=False)
    for kind in ("logistic_stacking", "xgb_meta_ensemble"):
        tick = time.monotonic()
        if args.stacking_mode == "nested":
            probability = nested_stacking(frame, representations, row_fold, args, cache, logger, kind); evaluation = "nested OOF"; note = "leakage-safe nested patient-wise CV"
        else:
            model = make_meta_model(kind).fit(meta_features(p), y_aligned)
            probability = model.predict_proba(meta_features(p))[:, 1]; evaluation = "diagnostic"; note = "DIAGNOSTIC ONLY — NOT AN INDEPENDENT RESULT"
        summary.append(save_method(kind, y_aligned, probability, frame, results, evaluation, note))
        timings.append({"stage": kind, "fold": 0, "start_time": "", "end_time": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - tick})
        pd.DataFrame(timings).to_csv(results / "runtime_summary.csv", index=False)
    pd.DataFrame(summary).to_csv(results / "ensemble_summary.csv", index=False)
    configuration = {"method": "equal_soft_vote", "threshold": .5, "cnn_weight": 1/3, "mlp_weight": 1/3,
                     "xgb_weight": 1/3, "threshold_source": "fixed default", "random_seed": 42}
    (results / "final_ensemble_configuration.json").write_text(json.dumps(configuration, indent=2) + "\n")
    print(pd.DataFrame(summary)[["method", "sensitivity", "ppv", "f1", "accuracy"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def run_test(args, logger):
    # Training uses precisely the same canonical feature cache/extractor as CV.
    train = load_training_data(args.train_data)
    train_native = _load_native_raw(args.train_data)
    train_x, train_y, train_ids, names = build_feature_cache(args.train_data, None, ROOT / "cache" / "test_train_features.npz", "extended")
    template = load_template(args.annotation_template); lengths = [np.asarray(x).size for x in template.ravel(order="F")]
    # Reuse the canonical loader by supplying template annotations as Class.
    import h5py
    from scipy.io import loadmat
    from python_xgboost.run_xgboost import _hdf5_cells, _scipy_cells
    fields = ("ECG", "SpO2", "SR_ECG", "SR_SpO2", "QRS")
    if h5py.is_hdf5(args.test_data):
        with h5py.File(args.test_data, "r") as handle:
            missing = [key for key in fields if key not in handle]
            if missing: raise ValueError("Native test inference requires: " + ", ".join(missing))
            test = {key: _hdf5_cells(handle, key) for key in fields}
    else:
        raw = loadmat(args.test_data, variable_names=fields, squeeze_me=False)
        missing = [key for key in fields if key not in raw]
        if missing: raise ValueError("Native test inference requires: " + ", ".join(missing))
        test = {key: _scipy_cells(raw[key]) for key in fields}
    test["Class"] = [np.full(length, "N") for length in lengths]
    xs, ids = [], []
    for i in range(100):
        features, _, current_names = extract_patient_features(test, i, SPO2_WINDOWS["extended"])
        if list(current_names) != list(names.astype(str)): raise ValueError("Train/test feature schemas differ")
        xs.append(features); ids.append(np.full(len(features), i + 1))
    test_x, test_ids = np.concatenate(xs), np.concatenate(ids)
    train_lengths = [int(np.sum(train_ids == patient)) for patient in np.unique(train_ids)]
    train_cnn = build_cnn_features(train_native, train_lengths); test_cnn = build_cnn_features(test, lengths)
    train_mlp, native_train_y, train_feature_names = build_mlp_features(train_native)
    test_mlp, _, test_feature_names = build_mlp_features(test)
    if not np.array_equal(train_feature_names, test_feature_names): raise ValueError("Native MLP train/test schemas differ")
    if not np.array_equal(native_train_y, train_y): raise ValueError("Native MLP/XGBoost label alignment differs")
    probabilities = {}
    for name in ("cnn", "mlp", "xgb"):
        prediction_cache = ROOT / "cache" / "test" / f"{name}_predictions.npz"
        expected = {"random_seed": 42, "feature_version": FEATURE_VERSION[name],
                    "model_version": VERSIONS[name], "validation_patients": list(range(1, 101)),
                    "branch_source": SOURCE[name], "training_patients": int(len(np.unique(train_ids)))}
        loaded = load_prediction_cache(prediction_cache, expected, logger)
        if loaded is not None and len(loaded["probability_a"]) == len(test_x):
            probabilities[name] = loaded["probability_a"]
            logger.info("  %-4s ...... loaded test cache", name.upper())
            continue
        model = adapter(name, args)
        try:
            if name == "cnn":
                train_labels = [train_y[train_ids == patient] for patient in np.unique(train_ids)]
                model.fit(train_cnn, train_labels, logger); probabilities[name] = model.predict_proba(test_cnn)
            elif name == "mlp":
                model.fit(train_mlp, train_y, train_ids, logger); probabilities[name] = model.predict_proba(test_mlp)
            else:
                model.fit(train_x, train_y, logger); probabilities[name] = model.predict_proba(test_x)
        finally: model.close()
        # Persist each expensive result before starting the next model.
        save_prediction_cache(prediction_cache, patient_id=test_ids, second_index=second_indices(test_ids),
                              y_true=np.full(len(test_ids), -1, np.int8), probability=probabilities[name],
                              fold=0, model_name=name, metadata=expected)
    config_path = Path(args.config) if args.config else ROOT / "results" / "final_ensemble_configuration.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {"method": args.method, "threshold": .5}
    method = args.method or config["method"]; matrix = np.column_stack([probabilities[x] for x in ("cnn", "mlp", "xgb")])
    if method == "equal_soft_vote": ensemble = matrix.mean(1)
    elif method == "weighted_soft_vote": ensemble = matrix @ np.array([config[f"{x}_weight"] for x in ("cnn", "mlp", "xgb")])
    elif method == "hard_majority_vote": ensemble = (np.sum(matrix >= .5, 1) >= 2).astype(float)
    else: raise ValueError("Test mode currently requires a frozen voting configuration")
    np.savez_compressed(ROOT / "results" / f"test_probabilities_{method}.npz", patient_id=test_ids,
                        second_index=second_indices(test_ids), cnn_probability=matrix[:, 0], mlp_probability=matrix[:, 1],
                        xgb_probability=matrix[:, 2], ensemble_probability=ensemble)
    filename = f"ProjectTestAnnotationsGroup{args.group}Submission{args.submission}.mat"
    output = ROOT / "submissions" / filename; report = ROOT / "submissions" / f"submission_validation_{filename[:-4]}.txt"
    split = np.cumsum(lengths)[:-1]
    create_submission(args.annotation_template, np.split(ensemble, split), config.get("threshold", .5), output)
    from ensemble.submission import validate_submission
    validate_submission(args.annotation_template, output, report); logger.info("Submission format validation: PASS\n%s", output)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("reproduce", "cv", "test"), required=True); parser.add_argument("--data")
    parser.add_argument("--train-data"); parser.add_argument("--test-data"); parser.add_argument("--annotation-template")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu"); parser.add_argument("--cv", choices=("5fold",), default="5fold")
    parser.add_argument("--methods", default="all"); parser.add_argument("--stacking-mode", choices=("diagnostic", "nested"), default="diagnostic")
    parser.add_argument("--method", default="equal_soft_vote"); parser.add_argument("--config"); parser.add_argument("--group", type=int, default=14)
    parser.add_argument("--submission", type=int, default=1)
    parser.add_argument("--models", nargs="+", choices=("cnn", "mlp"), default=("cnn", "mlp"))
    parser.add_argument("--base-models-only", action="store_true")
    parser.add_argument("--cnn-epochs", type=int, default=12); parser.add_argument("--mlp-epochs", type=int, default=30)
    parser.add_argument("--cnn-batch-size", type=int, default=512); parser.add_argument("--mlp-batch-size", type=int, default=512)
    parser.add_argument("--rebuild-cache", action="store_true"); parser.add_argument("--write-aligned-csv", action="store_true"); parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    required = ("data",) if args.mode in ("cv", "reproduce") else ("train_data", "test_data", "annotation_template")
    missing = [key for key in required if not getattr(args, key)]
    if missing: parser.error("missing required arguments: " + ", ".join("--" + x.replace("_", "-") for x in missing))
    return args


def resolve_device(args, logger):
    if args.device != "cuda": return
    try:
        import torch
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable; using CPU for this run (details in log)")
            args.device = "cpu"
    except Exception as error:
        logger.debug("CUDA probe failed: %s", error); args.device = "cpu"


if __name__ == "__main__":
    arguments = parse_args(); log, log_path = configure_logging(ROOT / "logs", arguments.verbose)
    warnings.showwarning = lambda message, category, filename, lineno, file=None, line=None: log.debug(
        "%s:%d: %s: %s", filename, lineno, category.__name__, message)
    log.debug("Detailed log: %s; arguments=%s", log_path, vars(arguments))
    resolve_device(arguments, log)
    try:
        audit_source()
        if arguments.mode == "reproduce": run_reproduce(arguments, log)
        elif arguments.mode == "cv": run_cv(arguments, log)
        else: run_test(arguments, log)
    except KeyboardInterrupt:
        log.error("Interrupted; completed prediction caches remain available for resume"); raise
    except Exception:
        log.exception("Ensemble experiment failed; completed caches remain available"); raise
