"""Thin adapter around the byte-for-byte vendored Kye MLP."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

MODEL_VERSION = "Kye-MLP-native-blobs-v2"
NATIVE = Path(__file__).with_name("native_mlp")


def _load(name, filename, aliases=()):
    spec = importlib.util.spec_from_file_location(name, NATIVE / filename)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    for alias in aliases: sys.modules.setdefault(alias, module)
    return module


config = _load("ensemble_native_mlp_config", "config.py", ("config",))
data_loader = _load("ensemble_native_mlp_data_loader", "data_loader.py", ("data_loader",))
feature_extraction = _load("ensemble_native_mlp_features", "feature_extraction.py", ("feature_extraction",))
native_model = _load("ensemble_native_mlp_model", "model.py", ("model",))


def build_mlp_features(data):
    matrices, labels, names = [], [], None
    for patient in range(len(data["Class"])):
        x, y, current = feature_extraction.extract_patient_features(data, patient)
        if names is not None and current != names: raise ValueError("Native MLP feature order changed between patients")
        matrices.append(x); labels.append(y); names = current
    return np.concatenate(matrices), np.concatenate(labels), np.asarray(names, dtype=str)


class MLPAdapter:
    def __init__(self, device="cpu", epochs=30, batch_size=512, seed=42, patience=None):
        import torch
        self.device = torch.device(device); self.epochs, self.batch_size, self.model = epochs, batch_size, None

    def fit(self, x, y, patient_id=None, logger=None):
        positive, negative = np.sum(y == 1), np.sum(y == 0)
        if not positive or not negative: raise ValueError("Native MLP training requires both classes")
        native_model.set_seed()
        self.model = native_model.MLPClassifier(x.shape[1], self.device, epochs=self.epochs,
                                                batch_size=self.batch_size, learning_rate=.001)
        self.model.fit(x, y, pos_weight=np.sqrt(negative / positive))
        return self

    def predict_proba(self, x):
        return self.model.predict_proba(x)[:, 1].astype(np.float32)

    def close(self):
        import torch
        self.model = None
        if torch.cuda.is_available(): torch.cuda.empty_cache()
