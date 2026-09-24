"""Thin in-memory adapter for the byte-for-byte vendored Kye CNN."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np

MODEL_VERSION = "Kye-1D_CNN-native-blobs-v2"
NATIVE = Path(__file__).with_name("native_cnn")


def _module(name, filename):
    spec = importlib.util.spec_from_file_location(name, NATIVE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


native_cache = _module("ensemble_native_cnn_cache", "cnn_cache.py")
native_model = _module("ensemble_native_cnn_model", "cnn_model.py")


def build_cnn_features(data, lengths=None):
    """Build native channel-first features from supplied MAT-file QRS values."""
    if "QRS" not in data:
        raise ValueError("Native CNN requires the supplied QRS variable")
    values = []
    for index, (spo2, qrs) in enumerate(zip(data["SpO2"], data["QRS"])):
        value = native_cache.build_patient_features(spo2, qrs)
        if value.shape[0] != native_model.NUM_CHANNELS:
            raise ValueError(f"Native CNN patient {index + 1} produced {value.shape[0]} channels")
        if lengths is not None and value.shape[1] != lengths[index]:
            raise ValueError(f"Native CNN patient {index + 1} length mismatch")
        values.append(value)
    return values


class _WindowDataset:
    def __init__(self, patients, labels, indices):
        import torch
        self.torch, self.patients, self.labels, self.indices = torch, patients, labels, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        patient, second = self.indices[item]
        features = self.patients[patient]
        contexts = []
        for window in (31, 61, 91, 121):
            half = window // 2
            start, end = second - half, second + half + 1
            left, right = max(0, -start), max(0, end - features.shape[1])
            chunk = features[:, max(0, start):min(features.shape[1], end)]
            if left or right:
                chunk = np.pad(chunk, ((0, 0), (left, right)), mode="edge")
            contexts.append(self.torch.from_numpy(chunk))
        return contexts, self.torch.tensor(float(self.labels[patient][second]), dtype=self.torch.float32)


class CNNAdapter:
    """Native normalization, sampling, architecture, loss, and optimization."""
    def __init__(self, device="cpu", epochs=12, batch_size=512, seed=42, patience=4):
        self.device, self.epochs, self.batch_size = device, epochs, batch_size
        self.seed, self.patience, self.model = seed, patience, None

    @staticmethod
    def _normalise(features):
        # This is a direct transcription of the final normalise_features definition.
        output = []
        for features_i in features:
            x = features_i.astype(np.float32, copy=True)
            x[0] = (x[0] - 100) / 10; x[1:3] /= 2
            x[3:7] = (x[3:7] - 100) / 10; x[7:11] /= 2; x[11:15] /= 5
            x[15:19] = (x[15:19] - 100) / 10; x[19:26] /= 5; x[26:29] /= 10
            x[29] = (x[29] - 70) / 30; x[30] /= 10; x[31] = (x[31] - 1) / .5
            x[32] /= .25; x[33] /= 15; x[34] /= .25
            output.append(x)
        return output

    def _indices(self, labels, fold_seed):
        rng, indices = np.random.default_rng(fold_seed), []
        for patient, target in enumerate(labels):
            positive, negative = np.flatnonzero(target == 1), np.flatnonzero(target == 0)
            if not len(positive):
                chosen = rng.choice(negative, size=min(1500, len(negative)), replace=False)
                indices.extend((patient, int(i)) for i in chosen); continue
            n_pos, n_neg = min(450, len(positive)), min(1050, len(negative))
            indices.extend((patient, int(i)) for i in rng.choice(positive, n_pos, replace=len(positive) < n_pos))
            indices.extend((patient, int(i)) for i in rng.choice(negative, n_neg, replace=len(negative) < n_neg))
        rng.shuffle(indices)
        return indices

    def fit(self, features, labels, logger=None, validation_features=None, validation_labels=None, fold=1):
        import torch
        torch.manual_seed(self.seed + fold); np.random.seed(self.seed + fold)
        train_x = self._normalise(features)
        train_indices = self._indices(labels, self.seed + fold)
        loader = torch.utils.data.DataLoader(_WindowDataset(train_x, labels, train_indices), self.batch_size,
                                             shuffle=True, num_workers=0, pin_memory=self.device == "cuda")
        valid_loader = None
        if validation_features is not None:
            valid_x = self._normalise(validation_features)
            valid_indices = [(p, s) for p, target in enumerate(validation_labels) for s in range(len(target))]
            valid_loader = torch.utils.data.DataLoader(_WindowDataset(valid_x, validation_labels, valid_indices),
                                                       self.batch_size, shuffle=False, num_workers=0)
        self.model = native_model.MultiScaleAttentionModel(num_channels=35).to(self.device)
        criterion = torch.nn.BCEWithLogitsLoss()  # deliberately no pos_weight
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=.001, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=self.device == "cuda")
        best, state, stale = -1., None, 0
        for epoch in range(self.epochs):
            self.model.train()
            for contexts, target in loader:
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=str(self.device).split(":")[0], enabled=self.device == "cuda"):
                    loss = criterion(self.model([x.to(self.device) for x in contexts]), target.to(self.device))
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            if valid_loader is None:
                continue
            probability = self._predict_loader(valid_loader)
            truth = np.concatenate(validation_labels)
            predicted = probability >= .5
            tp, fp, fn = np.sum(predicted & (truth == 1)), np.sum(predicted & (truth == 0)), np.sum(~predicted & (truth == 1))
            score = 2 * tp / max(2 * tp + fp + fn, 1)
            if logger: logger.debug("CNN fold %d epoch %d/%d validation F1=%.6f", fold, epoch + 1, self.epochs, score)
            if score > best:
                best, state, stale = score, copy.deepcopy(self.model.state_dict()), 0
            else:
                stale += 1
                if stale >= self.patience: break
        if state is not None: self.model.load_state_dict(state)
        return self

    def _predict_loader(self, loader):
        import torch
        self.model.eval(); result = []
        with torch.no_grad():
            for contexts, _ in loader:
                with torch.autocast(device_type=str(self.device).split(":")[0], enabled=self.device == "cuda"):
                    probability = torch.sigmoid(self.model([x.to(self.device) for x in contexts]))
                result.append(probability.float().cpu().numpy())
        return np.concatenate(result).astype(np.float32)

    def predict_proba(self, features):
        dummy = [np.zeros(value.shape[1], np.uint8) for value in features]
        indices = [(p, s) for p, value in enumerate(features) for s in range(value.shape[1])]
        loader = __import__("torch").utils.data.DataLoader(_WindowDataset(self._normalise(features), dummy, indices),
                                                           self.batch_size, shuffle=False, num_workers=0)
        return self._predict_loader(loader)

    def close(self):
        import torch
        self.model = None
        if torch.cuda.is_available(): torch.cuda.empty_cache()
