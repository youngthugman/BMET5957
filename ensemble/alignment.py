"""Shared folds, cache validation, and explicit patient/second alignment."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
CACHE_FIELDS = {"patient_id", "second_index", "y_true", "probability_a", "fold", "model_name", "metadata"}


def make_shared_folds(y, patient_id, output: Path, n_splits=5):
    """Assign each patient once with the required stratified group splitter."""
    y, patient_id = np.asarray(y), np.asarray(patient_id)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    row_fold = np.full(y.size, -1, dtype=np.int8)
    for fold, (train, valid) in enumerate(splitter.split(np.zeros(y.size), y, patient_id), 1):
        assert not np.intersect1d(patient_id[train], patient_id[valid]).size
        row_fold[valid] = fold
    assert np.all(row_fold > 0)
    assignments = pd.DataFrame({"patient_id": patient_id, "fold": row_fold}).drop_duplicates()
    assert assignments.patient_id.is_unique, "A patient was assigned to multiple folds"
    output.parent.mkdir(parents=True, exist_ok=True)
    assignments.sort_values("patient_id").to_csv(output, index=False)
    return row_fold


def second_indices(patient_id):
    return pd.Series(np.asarray(patient_id)).groupby(np.asarray(patient_id), sort=False).cumcount().to_numpy(np.int32)


def save_prediction_cache(path: Path, *, patient_id, second_index, y_true, probability,
                          fold, model_name, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, patient_id=patient_id, second_index=second_index,
                        y_true=y_true, probability_a=probability,
                        fold=np.full(len(y_true), fold), model_name=np.asarray(model_name),
                        metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
    tmp.replace(path)


def load_prediction_cache(path: Path, expected_metadata, logger):
    if not path.exists():
        return None
    try:
        cached = np.load(path, allow_pickle=False)
        if not CACHE_FIELDS.issubset(cached.files):
            raise ValueError("missing required fields")
        metadata = json.loads(str(cached["metadata"]))
        mismatches = [key for key, value in expected_metadata.items() if metadata.get(key) != value]
        if mismatches:
            raise ValueError("metadata mismatch: " + ", ".join(mismatches))
        probability = cached["probability_a"]
        if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
            raise ValueError("invalid probabilities")
        logger.info("Loaded compatible cache %s", path)
        return {key: cached[key] for key in cached.files}
    except Exception as error:
        logger.info("Rejected cache %s: %s", path, error)
        return None


def align_predictions(outputs, results_dir: Path, write_csv=False):
    """Join by keys (never array position), asserting identical labels and coverage."""
    merged = None
    for name in ("cnn", "mlp", "xgb"):
        item = outputs[name]
        frame = pd.DataFrame({"patient_id": item["patient_id"],
                              "second_index": item["second_index"],
                              f"y_{name}": item["y_true"],
                              f"{name}_probability": item["probability_a"]})
        assert not frame.duplicated(["patient_id", "second_index"]).any()
        merged = frame if merged is None else merged.merge(
            frame, on=["patient_id", "second_index"], how="inner", validate="one_to_one")
    expected = len(next(iter(outputs.values()))["y_true"])
    assert len(merged) == expected and all(len(value["y_true"]) == expected for value in outputs.values())
    assert (merged.y_cnn == merged.y_mlp).all() and (merged.y_cnn == merged.y_xgb).all()
    merged = (merged.rename(columns={"y_cnn": "y_true"}).drop(columns=["y_mlp", "y_xgb"])
              .sort_values(["patient_id", "second_index"]).reset_index(drop=True))
    probabilities = merged[["cnn_probability", "mlp_probability", "xgb_probability"]].to_numpy()
    assert np.isfinite(probabilities).all() and np.all((probabilities >= 0) & (probabilities <= 1))
    results_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(results_dir / "aligned_oof_predictions.npz", **{
        column: merged[column].to_numpy() for column in merged.columns})
    if write_csv:
        merged.to_csv(results_dir / "aligned_oof_predictions.csv", index=False)
    return merged
