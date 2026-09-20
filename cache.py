
"""Feature-cache creation and loading."""

from pathlib import Path

import numpy as np
from tqdm import tqdm

from config import (
    DEV20,
    FEATURE_EXTRACTOR_VERSION,
)
from data_loader import load_training_data
from feature_extraction import extract_patient_features


def build_feature_cache(
    data_path,
    selected_patients,
    cache_path,
):
    """Extract features for selected patients and save them to disk."""

    data = load_training_data(
        data_path
    )

    n_patients = len(
        data["Class"]
    )

    # If no explicit patient selection was supplied,
    # use every patient.
    if selected_patients is None:

        selected_patients = np.arange(
            1,
            n_patients + 1,
        )

    selected_patients = np.asarray(
        selected_patients,
        dtype=np.int16,
    )

    if selected_patients.size == 0:
        raise ValueError(
            "No patients were selected."
        )

    if selected_patients.min() < 1:
        raise ValueError(
            "Patient IDs must start at 1."
        )

    if selected_patients.max() > n_patients:
        raise ValueError(
            f"Requested patient "
            f"{selected_patients.max()}, "
            f"but MAT has {n_patients}"
        )

    all_x = []
    all_y = []
    all_ids = []
    feature_names = None

    for patient_id in tqdm(
        selected_patients,
        desc="Extracting patient features",
    ):

        x, y, names = extract_patient_features(
            data,
            patient_id - 1,
        )

        # Every patient must produce the same feature columns.
        if feature_names is not None:

            if names != feature_names:
                raise ValueError(
                    "Feature names differ between patients. "
                    f"Patient {patient_id} produced a "
                    "different feature structure."
                )

        feature_names = names

        all_x.append(x)
        all_y.append(y)

        all_ids.append(
            np.full(
                y.size,
                patient_id,
                dtype=np.int16,
            )
        )

    x = np.concatenate(
        all_x
    )

    y = np.concatenate(
        all_y
    )

    patient_id = np.concatenate(
        all_ids
    )

    feature_names = np.asarray(
        feature_names,
        dtype=str,
    )

    # --------------------------------------------------------
    # Final consistency checks
    # --------------------------------------------------------

    if not (
        x.shape[0]
        == y.size
        == patient_id.size
    ):
        raise ValueError(
            "Cached arrays have inconsistent "
            "numbers of samples."
        )

    if x.shape[1] != feature_names.size:
        raise ValueError(
            "Feature matrix column count does not "
            "match feature-name count."
        )

    if np.unique(feature_names).size != feature_names.size:
        raise ValueError(
            "Feature names are not unique."
        )

    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save cache
    # --------------------------------------------------------

    np.savez_compressed(
        cache_path,

        X=x,

        y=y,

        patient_id=patient_id,

        feature_names=feature_names,

        selected_patients=selected_patients,

        feature_extractor_version=np.asarray(
            FEATURE_EXTRACTOR_VERSION
        ),
    )

    print()
    print(
        "Feature cache created:"
    )

    print(
        f"  Samples: {x.shape[0]:,}"
    )

    print(
        f"  Features: {x.shape[1]}"
    )

    print(
        f"  ECG features: "
        f"{sum(name.startswith('ecg_') for name in feature_names)}"
    )

    print(
        f"  SpO2 features: "
        f"{sum(name.startswith('spo2_') for name in feature_names)}"
    )

    print(
        f"  Version: "
        f"{FEATURE_EXTRACTOR_VERSION}"
    )

    return (
        x,
        y,
        patient_id,
        feature_names,
    )


def load_or_build_cache(
    args,
    selected_patients,
    cache_path,
):
    """Load a compatible feature cache or rebuild it."""

    # --------------------------------------------------------
    # Try existing cache
    # --------------------------------------------------------

    if cache_path.exists() and not args.rebuild_cache:

        cached = np.load(
            cache_path,
            allow_pickle=False,
        )

        # ----------------------------------------------------
        # Version check
        # ----------------------------------------------------

        if (
            "feature_extractor_version"
            not in cached.files
        ):

            print(
                "Cache does not contain a feature "
                "extractor version; rebuilding it."
            )

        else:

            cached_version = str(
                cached[
                    "feature_extractor_version"
                ]
            )

            if (
                cached_version
                != FEATURE_EXTRACTOR_VERSION
            ):

                print(
                    "Cache uses a different feature "
                    "extractor version; rebuilding it."
                )

            else:

                # ------------------------------------------------
                # Patient selection check
                # ------------------------------------------------

                cached_patients = cached[
                    "selected_patients"
                ]

                if cached_patients.size == 0:

                    print(
                        "Cache contains no patients; "
                        "rebuilding it."
                    )

                else:

                    is_all_cache = np.array_equal(
                        cached_patients,
                        np.arange(
                            1,
                            cached_patients.max() + 1,
                        ),
                    )

                    selection_matches = (
                        (
                            args.patients == "all"
                            and is_all_cache
                        )
                        or
                        (
                            args.patients == "dev20"
                            and np.array_equal(
                                cached_patients,
                                DEV20,
                            )
                        )
                    )

                    if selection_matches:

                        print(
                            f"Loading feature cache: "
                            f"{cache_path}"
                        )

                        return (
                            cached["X"],
                            cached["y"],
                            cached["patient_id"],
                            cached["feature_names"],
                        )

                    print(
                        "Cache patient selection differs; "
                        "rebuilding it."
                    )

    # --------------------------------------------------------
    # Build new cache
    # --------------------------------------------------------

    if not args.data:
        raise SystemExit(
            "--data is required when a matching "
            "feature cache does not exist"
        )

    return build_feature_cache(
        Path(args.data),
        selected_patients,
        cache_path,
    )
