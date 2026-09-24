
#!/usr/bin/env python3
"""Patient-wise MLP feature ablation benchmark for ProjectTrainData.mat."""

import argparse
from pathlib import Path

import numpy as np

from cache import load_or_build_cache
from config import DEV20
from evaluation import (
    run_cross_validation,
    run_all_ablation,
)
from model import choose_device


def main():
    """Parse command-line arguments and run the selected experiment."""

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--data",
        help="Path to ProjectTrainData.mat",
    )

    parser.add_argument(
        "--patients",
        choices=("dev20", "all"),
        default="dev20",
    )

    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
    )

    parser.add_argument(
        "--cv",
        choices=("5fold", "logo"),
        default="5fold",
        help="Patient-wise cross-validation method",
    )

    parser.add_argument(
        "--features",
        choices=("all", "ecg", "spo2", "compare"),
        default="all",
        help="Feature set to evaluate",
    )

    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    root = Path(__file__).resolve().parent

    selected = (
        DEV20
        if args.patients == "dev20"
        else None
    )

    cache_path = (
        root
        / "cache"
        / "train_features.npz"
    )

    # --------------------------------------------------------
    # Load feature cache
    # --------------------------------------------------------

    x, y, patient_id, feature_names = (
        load_or_build_cache(
            args,
            selected,
            cache_path,
        )
    )

    feature_names = feature_names.astype(str)

    # --------------------------------------------------------
    # Validate feature structure
    # --------------------------------------------------------

    feature_masks = {
        "ecg": np.char.startswith(
            feature_names,
            "ecg_",
        ),
        "spo2": np.char.startswith(
            feature_names,
            "spo2_",
        ),
        "all": np.ones(
            feature_names.size,
            dtype=bool,
        ),
    }

    if (
        not feature_masks["ecg"].any()
        or not feature_masks["spo2"].any()
    ):
        raise ValueError(
            "Cached feature_names must contain "
            "both ecg_ and spo2_ features."
        )

    ecg_names = feature_names[
        feature_masks["ecg"]
    ]

    spo2_names = feature_names[
        feature_masks["spo2"]
    ]

    # --------------------------------------------------------
    # ECG feature count
    #
    # The ECG representation is still deliberately fixed at
    # 65 features. SpO2 is allowed to change as we experiment
    # with new feature engineering.
    # --------------------------------------------------------

    if ecg_names.size != 65:
        raise ValueError(
            f"Expected 65 ECG features, "
            f"found {ecg_names.size}."
        )

    # SpO2 feature count is intentionally NOT hard-coded.
    if spo2_names.size == 0:
        raise ValueError(
            "No SpO2 features were found."
        )

    # --------------------------------------------------------
    # Check uniqueness
    # --------------------------------------------------------

    if np.unique(feature_names).size != feature_names.size:
        raise ValueError(
            "Feature names are not unique."
        )

    # --------------------------------------------------------
    # Check dimensions
    # --------------------------------------------------------

    if not (
        x.shape[0]
        == y.size
        == patient_id.size
    ):
        raise ValueError(
            "x, y and patient_id have "
            "different numbers of samples."
        )

    if x.shape[1] != feature_names.size:
        raise ValueError(
            "Number of feature columns does not "
            "match number of feature names."
        )

    # --------------------------------------------------------
    # Dataset information
    # --------------------------------------------------------

    print(
        f"\nPatients: "
        f"{np.unique(patient_id).size}"
        f"\nSeconds: {y.size:,}"
        f"\nA labels: {(y == 1).sum():,}"
        f"\nN labels: {(y == 0).sum():,}"
        f"\nECG features: {ecg_names.size}"
        f"\nSpO2 features: {spo2_names.size}"
        f"\nCombined features: {x.shape[1]}"
        f"\nFeature matrix RAM: "
        f"{x.nbytes / 2**30:.2f} GiB"
    )

    print()
    print("ECG feature names:")
    print(
        "  "
        + "\n  ".join(ecg_names)
    )

    print()
    print("SpO2 feature names:")
    print(
        "  "
        + "\n  ".join(spo2_names)
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = choose_device(
        args.device
    )

    output_dir = root / "results"

    # ========================================================
    # Compare ECG / SpO2 / ALL
    # ========================================================

    if args.features == "compare":

        print()
        print("=" * 72)
        print(
            "RUNNING MLP ABLATION:"
        )
        print(
            "ECG ONLY vs SpO2 ONLY vs ALL FEATURES"
        )
        print("=" * 72)

        run_all_ablation(
            x,
            y,
            patient_id,
            feature_names,
            device,
            output_dir,
        )

        return

    # ========================================================
    # Run a single feature set
    # ========================================================

    feature_set_map = {
        "ecg": "ECG",
        "spo2": "SPO2",
        "all": "ALL",
    }

    feature_set = feature_set_map[
        args.features
    ]

    print()
    print("=" * 72)
    print(
        f"RUNNING MLP: {feature_set}"
    )
    print("=" * 72)

    result = run_cross_validation(
        x=x,
        y=y,
        patient_id=patient_id,
        feature_names=feature_names,
        feature_set=feature_set,
        device=device,
        output_dir=output_dir,
    )

    # --------------------------------------------------------
    # Compact final summary
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print(
        f"{feature_set} FINAL SUMMARY"
    )
    print("=" * 72)

    pooled = result["pooled_metrics"]

    print(
        f"Pooled OOF @ 0.50:"
        f"\n  Sensitivity = "
        f"{pooled['Sensitivity']:.4f}"
        f"\n  PPV         = "
        f"{pooled['PPV']:.4f}"
        f"\n  F1          = "
        f"{pooled['F1']:.4f}"
        f"\n  Accuracy    = "
        f"{pooled['Accuracy']:.4f}"
    )

    print(
        f"\nBest OOF threshold = "
        f"{result['best_threshold']:.2f}"
    )

    best = result[
        "best_threshold_metrics"
    ]

    print(
        f"Best OOF F1 = "
        f"{best['F1']:.4f}"
    )


if __name__ == "__main__":
    main()

