"""Patient-local temporal cleanup search for development OOF predictions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ensemble.ensemble_methods import metrics

RULES = (
    ("none", 0, 0),
    ("fill_n_gaps_le_1", 1, 0),
    ("fill_n_gaps_le_2", 2, 0),
    ("remove_a_runs_le_1", 0, 1),
    ("remove_a_runs_le_2", 0, 2),
    ("fill_n_gaps_le_1_remove_a_runs_le_1", 1, 1),
    ("fill_n_gaps_le_2_remove_a_runs_le_1", 2, 1),
    ("fill_n_gaps_le_1_remove_a_runs_le_2", 1, 2),
)


def _replace_bounded_runs(prediction, value, maximum_length):
    """Replace short runs of ``value`` only when bounded by its opposite."""
    output = np.asarray(prediction, dtype=np.uint8).copy()
    if maximum_length == 0:
        return output
    start = 0
    while start < len(output):
        end = start + 1
        while end < len(output) and output[end] == output[start]:
            end += 1
        if (output[start] == value and end - start <= maximum_length
                and start > 0 and end < len(output)
                and output[start - 1] != value and output[end] != value):
            output[start:end] = 1 - value
        start = end
    return output


def apply_rule(prediction, patient_id, *, fill_n_max=0, remove_a_max=0):
    """Apply filling then removal within each patient, preserving row order."""
    prediction, patient_id = np.asarray(prediction), np.asarray(patient_id)
    if prediction.ndim != 1 or patient_id.ndim != 1 or len(prediction) != len(patient_id):
        raise ValueError("prediction and patient_id must be equal-length 1-D arrays")
    if not np.isin(prediction, (0, 1)).all():
        raise ValueError("post-processing input must contain only binary 0/1 predictions")
    output = prediction.astype(np.uint8, copy=True)
    for patient in pd.unique(patient_id):
        positions = np.flatnonzero(patient_id == patient)
        local = _replace_bounded_runs(output[positions], 0, fill_n_max)
        local = _replace_bounded_runs(local, 1, remove_a_max)
        output[positions] = local
    assert len(output) == len(prediction), "post-processing changed prediction length"
    assert np.isin(output, (0, 1)).all(), "post-processing produced non-binary output"
    return output


def run_search(cache_path, results_dir):
    """Evaluate the fixed development rule list from a nested OOF cache only."""
    cache_path, results_dir = Path(cache_path), Path(results_dir)
    with np.load(cache_path, allow_pickle=False) as saved:
        required = {"probability_a", "y_true", "patient_id", "second_index"}
        missing = required.difference(saved.files)
        if missing:
            raise ValueError(f"OOF cache is missing arrays: {sorted(missing)}")
        probability = np.asarray(saved["probability_a"], dtype=float)
        y_true = np.asarray(saved["y_true"])
        patient_id = np.asarray(saved["patient_id"])
        second_index = np.asarray(saved["second_index"])
    if len({len(probability), len(y_true), len(patient_id), len(second_index)}) != 1 or probability.ndim != 1:
        raise ValueError("OOF cache arrays do not have matching 1-D coverage")
    if not np.isfinite(probability).all() or not np.isin(y_true, (0, 1)).all():
        raise ValueError("OOF cache contains invalid probabilities or labels")
    for patient in pd.unique(patient_id):
        local_seconds = second_index[patient_id == patient]
        if len(np.unique(local_seconds)) != len(local_seconds) or np.any(np.diff(local_seconds) <= 0):
            raise ValueError(f"patient {patient!r} seconds are not unique and increasing")

    baseline = (probability >= .50).astype(np.uint8)
    rows, predictions = [], {}
    for name, fill_n, remove_a in RULES:
        output = apply_rule(baseline, patient_id, fill_n_max=fill_n, remove_a_max=remove_a)
        changed = output != baseline
        score = metrics(y_true, output)
        rows.append({"rule": name, "fill_n_max_seconds": fill_n,
                     "remove_a_max_seconds": remove_a, **score,
                     "predicted_positive_fraction": float(np.mean(output)),
                     "labels_changed": int(changed.sum()),
                     "n_to_a_changes": int(np.sum((baseline == 0) & (output == 1))),
                     "a_to_n_changes": int(np.sum((baseline == 1) & (output == 0)))})
        predictions[name] = output
    table = pd.DataFrame(rows).sort_values("f1", ascending=False, kind="stable").reset_index(drop=True)
    best = table.iloc[0].to_dict()
    baseline_metrics = {**metrics(y_true, baseline),
                        "predicted_positive_fraction": float(np.mean(baseline))}
    result_keys = ("sensitivity", "ppv", "f1", "accuracy", "predicted_positive_fraction",
                   "labels_changed", "n_to_a_changes", "a_to_n_changes")
    summary = {
        "evaluation": "development-set post-processing search",
        "selection_data": "leakage-safe nested weighted-soft-vote OOF predictions and training labels only",
        "threshold": .50, "patient_independent_processing": True,
        "baseline_nested_weighted_vote_metrics": baseline_metrics,
        "best_rule": best["rule"],
        "best_rule_parameters": {"fill_n_max_seconds": int(best["fill_n_max_seconds"]),
                                 "remove_a_max_seconds": int(best["remove_a_max_seconds"])},
        "resulting_metrics": {key: best[key] for key in result_keys},
        "absolute_f1_change": float(best["f1"] - baseline_metrics["f1"]),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(results_dir / "postprocessing_search.csv", index=False)
    (results_dir / "postprocessing_best.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(results_dir / "oof_predictions_5fold_nested_weighted_soft_vote_postprocessed_best.npz",
                        prediction=predictions[best["rule"]], baseline_prediction=baseline,
                        probability_a=probability, y_true=y_true, patient_id=patient_id,
                        second_index=second_index, threshold=np.float32(.5), rule=np.asarray(best["rule"]),
                        evaluation=np.asarray("development-set post-processing search"))
    print(table[["rule", "sensitivity", "ppv", "f1", "accuracy", "predicted_positive_fraction",
                 "labels_changed", "n_to_a_changes", "a_to_n_changes"]].to_string(
                     index=False, float_format=lambda value: f"{value:.4f}"))
    return table, summary
