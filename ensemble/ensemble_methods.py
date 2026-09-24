"""Metrics, voting, diversity, and probability-only meta classifiers."""
from __future__ import annotations

import itertools
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

THRESHOLDS = np.round(np.arange(.10, .901, .01), 2)


def metrics(y, probability, threshold=.5):
    y, prediction = np.asarray(y, dtype=bool), np.asarray(probability) >= threshold
    tp, fp = np.sum(y & prediction), np.sum(~y & prediction)
    fn, tn = np.sum(y & ~prediction), np.sum(~y & ~prediction)
    sensitivity = float(tp / (tp + fn)) if tp + fn else 0.
    ppv = float(tp / (tp + fp)) if tp + fp else 0.
    f1 = 2 * sensitivity * ppv / (sensitivity + ppv) if sensitivity + ppv else 0.
    return {"sensitivity": sensitivity, "ppv": ppv, "f1": f1,
            "accuracy": float((tp + tn) / len(y))}


def best_threshold(y, probability):
    rows = [(threshold, metrics(y, probability, threshold)) for threshold in THRESHOLDS]
    return max(rows, key=lambda item: item[1]["f1"])


def meta_features(probabilities):
    p = np.asarray(probabilities)
    return np.column_stack((p, p[:, 0] * p[:, 1], p[:, 0] * p[:, 2],
                            p[:, 1] * p[:, 2], p.mean(1), p.max(1), p.min(1), p.std(1)))


def make_meta_model(kind):
    if kind == "logistic_stacking":
        return LogisticRegression(max_iter=1000, random_state=42)
    if kind == "xgb_meta_ensemble":
        return XGBClassifier(objective="binary:logistic", n_estimators=100, max_depth=2,
                             learning_rate=.05, subsample=.8, colsample_bytree=1.,
                             reg_lambda=1., tree_method="hist", random_state=42, n_jobs=-1)
    raise ValueError(kind)


def weighted_search(y, probabilities):
    rows = []
    for cnn in range(21):
        for mlp in range(21 - cnn):
            weights = np.array([cnn, mlp, 20 - cnn - mlp]) / 20
            probability = probabilities @ weights
            score = metrics(y, probability)
            threshold, diagnostic = best_threshold(y, probability)
            rows.append({"cnn_weight": weights[0], "mlp_weight": weights[1], "xgb_weight": weights[2],
                         **{f"{key}_050": value for key, value in score.items()},
                         "best_threshold": threshold, "best_threshold_f1": diagnostic["f1"],
                         "notes": "DIAGNOSTIC — NOT AN INDEPENDENT TEST RESULT"})
    result = pd.DataFrame(rows)
    return result, result.loc[result.f1_050.idxmax()]


def diversity(y, frame):
    rows = []
    for a, b in itertools.combinations(("cnn", "mlp", "xgb"), 2):
        pa, pb = frame[f"{a}_probability"].to_numpy(), frame[f"{b}_probability"].to_numpy()
        ca, cb = (pa >= .5) == y, (pb >= .5) == y
        rows.append({"model_a": a, "model_b": b, "pearson_probability_correlation": np.corrcoef(pa, pb)[0, 1],
                     "binary_disagreement_rate": np.mean((pa >= .5) != (pb >= .5)),
                     "double_fault_rate": np.mean(~ca & ~cb),
                     "model_a_correct_model_b_incorrect": int(np.sum(ca & ~cb)),
                     "model_a_incorrect_model_b_correct": int(np.sum(~ca & cb))})
    return pd.DataFrame(rows)
