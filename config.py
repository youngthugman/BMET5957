"""Project-wide configuration and constants."""

import numpy as np


# Development patient subset used for the default experiments.
DEV20 = np.array([
    1, 3, 9, 12, 15, 17, 18, 30, 33, 36,
    37, 38, 42, 47, 50, 57, 61, 72, 77, 94
])


# Variables expected inside ProjectTrainData.mat.
FIELDS = ("ECG", "SpO2", "Class", "SR_ECG", "SR_SpO2")


# Used to invalidate feature caches when the ECG extraction implementation
# changes.
ECG_EXTRACTOR_VERSION = "submission2-group5-qrs-multiscale-ecg-v2"

FEATURE_EXTRACTOR_VERSION = "spo2_morphology_v1"