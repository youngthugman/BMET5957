
import os
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold

    HAS_STRATIFIED_GROUP_KFOLD = True

except ImportError:

    HAS_STRATIFIED_GROUP_KFOLD = False


from model import (
    fit_model,
    choose_device,
)


# ============================================================
# Configuration
# ============================================================

RANDOM_SEED = 42

N_SPLITS = 5

# ECG feature count is currently fixed by the feature extractor.
# SpO2 feature count is intentionally NOT fixed because we are
# experimenting with additional SpO2 features.
ECG_FEATURES = 65


# ============================================================
# Metrics
# ============================================================

def metrics(
    y_true,
    y_pred,
):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    ppv = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    f1 = (
        2 * sensitivity * ppv /
        (sensitivity + ppv)
        if (sensitivity + ppv) > 0
        else 0
    )

    accuracy = (
        (tp + tn) /
        (tp + tn + fp + fn)
    )

    return {
        "Sensitivity": sensitivity,
        "PPV": ppv,
        "F1": f1,
        "Accuracy": accuracy,
    }


# ============================================================
# Patient-level metrics
# ============================================================

def patient_performance(
    y_true,
    y_pred,
    patient_id,
):

    rows = []

    for patient in np.unique(
        patient_id
    ):

        mask = (
            patient_id == patient
        )

        result = metrics(
            y_true[mask],
            y_pred[mask],
        )

        result["Patient"] = int(
            patient
        )

        rows.append(result)

    return pd.DataFrame(rows)


# ============================================================
# Feature selection
# ============================================================

def get_feature_masks(
    feature_names,
):

    feature_names = np.asarray(
        feature_names
    ).astype(str)

    ecg_mask = np.char.startswith(
        feature_names,
        "ecg_",
    )

    spo2_mask = np.char.startswith(
        feature_names,
        "spo2_",
    )

    if not ecg_mask.any():

        raise ValueError(
            "No ECG features were found. "
            "Feature names must begin with "
            "'ecg_'."
        )

    if not spo2_mask.any():

        raise ValueError(
            "No SpO2 features were found. "
            "Feature names must begin with "
            "'spo2_'."
        )

    if np.any(
        ecg_mask & spo2_mask
    ):

        raise ValueError(
            "A feature cannot belong to "
            "both ECG and SpO2 groups."
        )

    unknown_mask = ~(
        ecg_mask | spo2_mask
    )

    if unknown_mask.any():

        unknown = feature_names[
            unknown_mask
        ]

        raise ValueError(
            "Found feature names that are "
            "neither ECG nor SpO2:\n"
            + "\n".join(
                str(name)
                for name in unknown
            )
        )

    return ecg_mask, spo2_mask


def select_features(
    x,
    feature_names,
    feature_set,
):

    feature_names = np.asarray(
        feature_names
    ).astype(str)

    ecg_mask, spo2_mask = (
        get_feature_masks(
            feature_names
        )
    )

    if x.shape[1] != len(
        feature_names
    ):

        raise ValueError(
            "x and feature_names "
            "contain different numbers "
            "of features."
        )

    if feature_set == "ECG":

        return x[:, ecg_mask]

    elif feature_set == "SPO2":

        return x[:, spo2_mask]

    elif feature_set == "ALL":

        return x

    else:

        raise ValueError(
            f"Unknown feature set: "
            f"{feature_set}"
        )


# ============================================================
# Feature-name selection
# ============================================================

def select_feature_names(
    feature_names,
    feature_set,
):

    feature_names = np.asarray(
        feature_names
    ).astype(str)

    ecg_mask, spo2_mask = (
        get_feature_masks(
            feature_names
        )
    )

    if feature_set == "ECG":

        return feature_names[
            ecg_mask
        ]

    elif feature_set == "SPO2":

        return feature_names[
            spo2_mask
        ]

    elif feature_set == "ALL":

        return feature_names

    else:

        raise ValueError(
            f"Unknown feature set: "
            f"{feature_set}"
        )


# ============================================================
# Cross validation
# ============================================================

def run_cross_validation(
    x,
    y,
    patient_id,
    feature_names,
    feature_set,
    device,
    output_dir="results",
):

    print()
    print("=" * 70)
    print(
        f"FEATURE SET: {feature_set}"
    )
    print("=" * 70)

    feature_names = np.asarray(
        feature_names
    ).astype(str)

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    if x.ndim != 2:

        raise ValueError(
            f"Expected x to be 2D, "
            f"got shape {x.shape}"
        )

    if x.shape[1] != len(
        feature_names
    ):

        raise ValueError(
            f"x contains {x.shape[1]} "
            f"features but feature_names "
            f"contains {len(feature_names)}."
        )

    if np.unique(
        feature_names
    ).size != len(feature_names):

        raise ValueError(
            "Duplicate feature names "
            "were detected."
        )

    if (
        len(x)
        != len(y)
    ):

        raise ValueError(
            "x and y have different "
            "numbers of samples."
        )

    if (
        len(x)
        != len(patient_id)
    ):

        raise ValueError(
            "x and patient_id have "
            "different lengths."
        )

    # --------------------------------------------------------
    # Feature selection
    # --------------------------------------------------------

    x_selected = select_features(
        x,
        feature_names,
        feature_set,
    )

    selected_feature_names = (
        select_feature_names(
            feature_names,
            feature_set,
        )
    )

    print(
        f"Input shape: "
        f"{x_selected.shape}"
    )

    print(
        f"Number of features: "
        f"{x_selected.shape[1]}"
    )

    print(
        f"ECG features: "
        f"{np.sum(np.char.startswith(feature_names, 'ecg_'))}"
    )

    print(
        f"SpO2 features: "
        f"{np.sum(np.char.startswith(feature_names, 'spo2_'))}"
    )

    if (
        x_selected.shape[1]
        != len(selected_feature_names)
    ):

        raise ValueError(
            "Number of selected features "
            "does not match number of "
            "selected feature names."
        )

    # --------------------------------------------------------
    # CV splitter
    # --------------------------------------------------------

    if HAS_STRATIFIED_GROUP_KFOLD:

        splitter = StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_SEED,
        )

        split_iterator = splitter.split(
            x_selected,
            y,
            groups=patient_id,
        )

    else:

        splitter = GroupKFold(
            n_splits=N_SPLITS
        )

        split_iterator = splitter.split(
            x_selected,
            y,
            groups=patient_id,
        )

    # --------------------------------------------------------
    # OOF storage
    # --------------------------------------------------------

    oof_predictions = np.full(
        len(y),
        np.nan,
        dtype=np.float32,
    )

    oof_count = np.zeros(
        len(y),
        dtype=np.int32,
    )

    fold_results = []

    # --------------------------------------------------------
    # Folds
    # --------------------------------------------------------

    for fold, (
        train_idx,
        validation_idx,
    ) in enumerate(
        split_iterator,
        start=1,
    ):

        print()
        print(
            f"Fold {fold}/{N_SPLITS}"
        )

        train_patients = np.unique(
            patient_id[train_idx]
        )

        validation_patients = np.unique(
            patient_id[validation_idx]
        )

        print(
            "  train patients:",
            train_patients.tolist()
        )

        print(
            "  validation patients:",
            validation_patients.tolist()
        )

        # ----------------------------------------------------
        # Leakage check
        # ----------------------------------------------------

        overlap = np.intersect1d(
            train_patients,
            validation_patients,
        )

        if len(overlap) > 0:

            raise RuntimeError(
                "Patient leakage detected: "
                f"{overlap}"
            )

        # ----------------------------------------------------
        # Class weight
        # ----------------------------------------------------

        y_train = y[train_idx]

        positive = np.sum(
            y_train == 1
        )

        negative = np.sum(
            y_train == 0
        )

        if positive == 0:

            raise RuntimeError(
                "Training fold contains "
                "no positive samples."
            )

        scale_pos_weight = np.sqrt(
            negative / positive
        )

        print(
            f"  training samples: "
            f"{len(train_idx):,}"
        )

        print(
            f"  validation samples: "
            f"{len(validation_idx):,}"
        )

        print(
            f"  scale_pos_weight: "
            f"{scale_pos_weight:.4f}"
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model, _ = fit_model(
            x_selected[train_idx],
            y[train_idx],
            device,
            scale_pos_weight,
        )

        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        probabilities = (
            model.predict_proba(
                x_selected[validation_idx]
            )[:, 1]
        )

        oof_predictions[
            validation_idx
        ] = probabilities

        oof_count[
            validation_idx
        ] += 1

        # ----------------------------------------------------
        # Fold metrics
        # ----------------------------------------------------

        predictions = (
            probabilities >= 0.50
        ).astype(int)

        fold_metrics = metrics(
            y[validation_idx],
            predictions,
        )

        fold_metrics["Fold"] = fold

        fold_results.append(
            fold_metrics
        )

        print(
            f"  Sensitivity="
            f"{fold_metrics['Sensitivity']:.4f} "
            f"PPV="
            f"{fold_metrics['PPV']:.4f} "
            f"F1="
            f"{fold_metrics['F1']:.4f} "
            f"Accuracy="
            f"{fold_metrics['Accuracy']:.4f}"
        )

    # --------------------------------------------------------
    # OOF validation
    # --------------------------------------------------------

    if not np.all(
        oof_count == 1
    ):

        raise RuntimeError(
            "OOF prediction count is "
            "not exactly one for every "
            "sample."
        )

    # --------------------------------------------------------
    # Fold summary
    # --------------------------------------------------------

    fold_df = pd.DataFrame(
        fold_results
    )

    print()
    print(
        "Mean fold metrics at "
        "threshold 0.50"
    )

    print(
        f"  Sensitivity="
        f"{fold_df['Sensitivity'].mean():.4f} "
        f"PPV="
        f"{fold_df['PPV'].mean():.4f} "
        f"F1="
        f"{fold_df['F1'].mean():.4f} "
        f"Accuracy="
        f"{fold_df['Accuracy'].mean():.4f}"
    )

    print(
        f"  Std fold F1="
        f"{fold_df['F1'].std():.4f}"
    )

    # --------------------------------------------------------
    # Pooled OOF
    # --------------------------------------------------------

    oof_labels = (
        oof_predictions >= 0.50
    ).astype(int)

    pooled_metrics = metrics(
        y,
        oof_labels,
    )

    print()
    print(
        "Pooled OOF at threshold 0.50"
    )

    print(
        f"  Sensitivity="
        f"{pooled_metrics['Sensitivity']:.4f} "
        f"PPV="
        f"{pooled_metrics['PPV']:.4f} "
        f"F1="
        f"{pooled_metrics['F1']:.4f} "
        f"Accuracy="
        f"{pooled_metrics['Accuracy']:.4f}"
    )

    # --------------------------------------------------------
    # Threshold sweep
    # --------------------------------------------------------

    best_threshold = 0.50
    best_f1 = -1

    best_threshold_metrics = None

    for threshold in np.arange(
        0.10,
        0.91,
        0.01,
    ):

        predictions = (
            oof_predictions >= threshold
        ).astype(int)

        result = metrics(
            y,
            predictions,
        )

        if result["F1"] > best_f1:

            best_f1 = result["F1"]

            best_threshold = (
                threshold
            )

            best_threshold_metrics = (
                result
            )

    print()
    print(
        "OOF threshold-tuning "
        "diagnostic "
        "(NOT an independent test result)"
    )

    print(
        f"  threshold="
        f"{best_threshold:.2f} "
        f"Sensitivity="
        f"{best_threshold_metrics['Sensitivity']:.4f} "
        f"PPV="
        f"{best_threshold_metrics['PPV']:.4f} "
        f"F1="
        f"{best_threshold_metrics['F1']:.4f} "
        f"Accuracy="
        f"{best_threshold_metrics['Accuracy']:.4f}"
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    np.savez(
        os.path.join(
            output_dir,
            f"oof_{feature_set.lower()}.npz",
        ),
        y=y,
        probabilities=oof_predictions,
        patient_id=patient_id,
    )

    fold_df.to_csv(
        os.path.join(
            output_dir,
            f"fold_metrics_{feature_set.lower()}.csv",
        ),
        index=False,
    )

    patient_df = patient_performance(
        y,
        oof_labels,
        patient_id,
    )

    patient_df.to_csv(
        os.path.join(
            output_dir,
            f"patient_metrics_{feature_set.lower()}.csv",
        ),
        index=False,
    )

    # --------------------------------------------------------
    # Final model
    # --------------------------------------------------------

    print()
    print(
        "Training final MLP on all data..."
    )

    positive = np.sum(
        y == 1
    )

    negative = np.sum(
        y == 0
    )

    final_weight = np.sqrt(
        negative / positive
    )

    final_model, _ = fit_model(
        x_selected,
        y,
        device,
        final_weight,
    )

    model_path = os.path.join(
        output_dir,
        f"mlp_model_{feature_set.lower()}.pt",
    )

    final_model.save_model(
        model_path
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    feature_importance = (
        final_model.feature_importances_
    )

    if len(feature_importance) != len(
        selected_feature_names
    ):

        raise ValueError(
            "Model returned "
            f"{len(feature_importance)} "
            "feature importances, but "
            f"{len(selected_feature_names)} "
            "feature names were selected."
        )

    importance_df = pd.DataFrame({
        "feature":
            selected_feature_names,
        "importance":
            feature_importance,
    })

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print()
    print(
        "Top 20 feature importances"
    )

    print(
        importance_df.head(20).to_string(
            index=False
        )
    )

    importance_df.to_csv(
        os.path.join(
            output_dir,
            f"feature_importance_"
            f"{feature_set.lower()}.csv",
        ),
        index=False,
    )

    return {
        "feature_set":
            feature_set,

        "fold_metrics":
            fold_df,

        "pooled_metrics":
            pooled_metrics,

        "best_threshold":
            best_threshold,

        "best_threshold_metrics":
            best_threshold_metrics,

        "patient_metrics":
            patient_df,

        "feature_importance":
            importance_df,
    }


# ============================================================
# Ablation comparison
# ============================================================

def print_ablation_comparison(
    results
):

    print()
    print("=" * 70)
    print(
        "ABLATION COMPARISON"
    )
    print("=" * 70)

    rows = []

    for name, result in results.items():

        pooled = (
            result["pooled_metrics"]
        )

        rows.append({
            "Feature Set":
                name,

            "Sensitivity":
                pooled["Sensitivity"],

            "PPV":
                pooled["PPV"],

            "F1":
                pooled["F1"],

            "Accuracy":
                pooled["Accuracy"],

            "Best OOF F1":
                result[
                    "best_threshold_metrics"
                ]["F1"],

            "Best Threshold":
                result[
                    "best_threshold"
                ],
        })

    comparison = pd.DataFrame(
        rows
    )

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    return comparison


# ============================================================
# Run all three MLP experiments
# ============================================================

def run_all_ablation(
    x,
    y,
    patient_id,
    feature_names,
    device,
    output_dir="results",
):

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    if x.ndim != 2:

        raise ValueError(
            f"Expected x to be 2D, "
            f"got shape {x.shape}"
        )

    feature_names = np.asarray(
        feature_names
    ).astype(str)

    if x.shape[1] != len(
        feature_names
    ):

        raise ValueError(
            f"x contains {x.shape[1]} "
            "features but feature_names "
            f"contains {len(feature_names)}."
        )

    if len(x) != len(y):

        raise ValueError(
            "x and y have different "
            "numbers of samples."
        )

    if len(x) != len(patient_id):

        raise ValueError(
            "x and patient_id have "
            "different lengths."
        )

    if np.unique(
        feature_names
    ).size != len(feature_names):

        raise ValueError(
            "Duplicate feature names "
            "were detected."
        )

    # Validate the feature groups before
    # running any experiments.
    ecg_mask, spo2_mask = (
        get_feature_masks(
            feature_names
        )
    )

    print()
    print(
        f"Detected {np.sum(ecg_mask)} "
        f"ECG features."
    )

    print(
        f"Detected {np.sum(spo2_mask)} "
        f"SpO2 features."
    )

    print(
        f"Detected {len(feature_names)} "
        f"total features."
    )

    if np.sum(ecg_mask) != ECG_FEATURES:

        raise ValueError(
            f"Expected {ECG_FEATURES} "
            "ECG features, but detected "
            f"{np.sum(ecg_mask)}."
        )

    # --------------------------------------------------------
    # Run experiments
    # --------------------------------------------------------

    results = {}

    for feature_set in [
        "ECG",
        "SPO2",
        "ALL",
    ]:

        results[feature_set] = (
            run_cross_validation(
                x=x,
                y=y,
                patient_id=patient_id,
                feature_names=feature_names,
                feature_set=feature_set,
                device=device,
                output_dir=output_dir,
            )
        )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    comparison = (
        print_ablation_comparison(
            results
        )
    )

    comparison.to_csv(
        os.path.join(
            output_dir,
            "mlp_ablation_comparison.csv",
        ),
        index=False,
    )

    return results


# ============================================================
# Example usage
# ============================================================
#
# Your existing data preparation code should produce:
#
#     x
#     y
#     patient_id
#     feature_names
#
# Then call:
#
#     device = choose_device()
#
#     results = run_all_ablation(
#         x,
#         y,
#         patient_id,
#         feature_names,
#         device,
#     )
#
# ============================================================

