"""Shared-fold adapter around the actual Kye-1D_CNN Python model family."""
from __future__ import annotations

import copy
import numpy as np

from .kye_cnn.cnn_cache import CHANNEL_NAMES, CONTEXT_WINDOWS
from .kye_cnn.cnn_evaluation import ContextDataset, predict_all

MODEL_VERSION = "Kye-1D_CNN-Python-MultiScaleAttentionModel-v1"


class CNNAdapter:
    def __init__(self, device="cpu", epochs=30, batch_size=256, seed=42, patience=5):
        self.device, self.epochs, self.batch_size = device, epochs, batch_size
        self.seed, self.patience, self.model = seed, patience, None
        self.median = self.mean = self.std = None

    def _normalise(self, patients, fit=False):
        if fit:
            joined = np.concatenate(patients)
            self.median = np.nanmedian(joined, 0); self.median[~np.isfinite(self.median)] = 0
            clean = np.where(np.isfinite(joined), joined, self.median)
            self.mean, self.std = clean.mean(0), clean.std(0); self.std[self.std < 1e-6] = 1
        return [((np.where(np.isfinite(value), value, self.median) - self.mean) / self.std).astype(np.float32)
                for value in patients]

    def fit(self, features, labels, logger=None):
        """Retain weighted sampling, BCE loss, AdamW, and patient-held-out early stopping."""
        import torch
        from .kye_cnn.cnn_model import MultiScaleAttentionModel
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        if any(value.shape[1] != len(CHANNEL_NAMES) for value in features):
            raise ValueError("Kye CNN input must contain its original 35 channels")
        # The early-stop patients are selected solely from this outer-training set.
        order = np.random.default_rng(self.seed).permutation(len(features))
        n_monitor = max(1, round(.1 * len(order)))
        monitor_patients, fit_patients = set(order[:n_monitor]), set(order[n_monitor:])
        if not fit_patients: fit_patients, monitor_patients = set(order), set()
        normalised = self._normalise(features, fit=True)
        all_labels = np.concatenate(labels)
        mapping = [(p, s) for p, value in enumerate(features) for s in range(len(value))]
        train_indices = np.array([i for i, (p, _) in enumerate(mapping) if p in fit_patients])
        monitor_indices = np.array([i for i, (p, _) in enumerate(mapping) if p in monitor_patients])
        train_targets = all_labels[train_indices]
        counts = np.bincount(train_targets.astype(int), minlength=2)
        sample_weights = (1 / np.maximum(counts, 1))[train_targets.astype(int)]
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(train_indices), replacement=True)
        train_data = ContextDataset(normalised, labels, train_indices)
        train_loader = torch.utils.data.DataLoader(train_data, self.batch_size, sampler=sampler)
        monitor_loader = None
        if monitor_indices.size:
            monitor_loader = torch.utils.data.DataLoader(
                ContextDataset(normalised, labels, monitor_indices), self.batch_size, shuffle=False)
        self.model = MultiScaleAttentionModel(input_channels=35).to(self.device)
        positive_weight = counts[0] / max(counts[1], 1)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=self.device))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        best, best_state, stale = float("inf"), None, 0
        for epoch in range(self.epochs):
            self.model.train(); total = 0.
            for contexts, target in train_loader:
                optimizer.zero_grad()
                loss = criterion(self.model([x.to(self.device) for x in contexts]), target.to(self.device))
                loss.backward(); optimizer.step(); total += float(loss.detach()) * len(target)
            monitor_loss = total / len(train_data)
            if monitor_loader:
                self.model.eval(); summed = count = 0
                with torch.no_grad():
                    for contexts, target in monitor_loader:
                        loss = criterion(self.model([x.to(self.device) for x in contexts]), target.to(self.device))
                        summed += float(loss) * len(target); count += len(target)
                monitor_loss = summed / count
            if logger: logger.debug("CNN epoch %d/%d loss=%.6f monitor=%.6f", epoch + 1, self.epochs, total / len(train_data), monitor_loss)
            if monitor_loss < best - 1e-5:
                best, best_state, stale = monitor_loss, copy.deepcopy(self.model.state_dict()), 0
            else:
                stale += 1
                if stale >= self.patience: break
        if best_state is not None: self.model.load_state_dict(best_state)
        return self

    def predict_proba(self, features):
        return predict_all(self.model, self._normalise(features), self.device, self.batch_size)

    def close(self):
        import torch
        del self.model; self.model = None
        if torch.cuda.is_available(): torch.cuda.empty_cache()
