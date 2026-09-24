"""Shared-fold adapter around the Kye-MLP Python implementation."""
from __future__ import annotations

import copy
import numpy as np

from .kye_mlp.evaluation import predict_all
from .kye_mlp.main import (DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, EARLY_STOPPING_PATIENCE,
                           LEARNING_RATE, WEIGHT_DECAY)

MODEL_VERSION = "Kye-MLP-Python-LayerNorm-GELU-v1"


class MLPAdapter:
    def __init__(self, device="cpu", epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE,
                 seed=42, patience=EARLY_STOPPING_PATIENCE):
        self.device, self.epochs, self.batch_size = device, epochs, batch_size
        self.seed, self.patience = seed, patience
        self.model = self.median = self.mean = self.std = None

    def _transform(self, x):
        clean = np.where(np.isfinite(x), x, self.median)
        return ((clean - self.mean) / self.std).astype(np.float32)

    def fit(self, x, y, patient_id, logger=None):
        import torch
        from .kye_mlp.model import MLPClassifier
        torch.manual_seed(self.seed)
        patients = np.unique(patient_id); rng = np.random.default_rng(self.seed); rng.shuffle(patients)
        monitor_patients = patients[:max(1, round(.1 * len(patients)))]
        monitor = np.isin(patient_id, monitor_patients); train = ~monitor
        if not train.any(): train, monitor = np.ones(len(y), bool), np.zeros(len(y), bool)
        self.median = np.nanmedian(x[train], 0); self.median[~np.isfinite(self.median)] = 0
        clean_train = np.where(np.isfinite(x[train]), x[train], self.median)
        self.mean, self.std = clean_train.mean(0), clean_train.std(0); self.std[self.std < 1e-6] = 1
        transformed = self._transform(x)
        self.model = MLPClassifier(x.shape[1]).to(self.device)
        negative, positive = np.bincount(y[train], minlength=2)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative / max(positive, 1), device=self.device))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(transformed[train]), torch.from_numpy(y[train].astype(np.float32)))
        loader = torch.utils.data.DataLoader(dataset, self.batch_size, shuffle=True)
        best, state, stale = float("inf"), None, 0
        for epoch in range(self.epochs):
            self.model.train(); total = 0.
            for features, target in loader:
                optimizer.zero_grad(); loss = criterion(self.model(features.to(self.device)), target.to(self.device))
                loss.backward(); optimizer.step(); total += float(loss.detach()) * len(target)
            score = total / len(dataset)
            if monitor.any():
                self.model.eval()
                with torch.no_grad():
                    score = float(criterion(self.model(torch.from_numpy(transformed[monitor]).to(self.device)),
                                            torch.from_numpy(y[monitor].astype(np.float32)).to(self.device)))
            if logger: logger.debug("MLP epoch %d/%d loss=%.6f monitor=%.6f", epoch + 1, self.epochs, total / len(dataset), score)
            if score < best - 1e-5: best, state, stale = score, copy.deepcopy(self.model.state_dict()), 0
            else:
                stale += 1
                if stale >= self.patience: break
        if state is not None: self.model.load_state_dict(state)
        return self

    def predict_proba(self, x):
        return predict_all(self.model, self._transform(x), self.device, self.batch_size)

    def close(self):
        import torch
        del self.model; self.model = None
        if torch.cuda.is_available(): torch.cuda.empty_cache()
