"""Thin adapter retaining the canonical 262-feature XGBoost configuration."""
import numpy as np
from python_xgboost.run_xgboost import fit_model

MODEL_VERSION = "19926-XGBoost-Tuning-262-v3"


class XGBAdapter:
    def __init__(self, device="cpu"):
        self.device, self.model = device, None

    def fit(self, x, y, logger=None):
        negative, positive = np.bincount(y, minlength=2)
        if not positive: raise ValueError("Training labels have no A examples")
        weight = float(np.sqrt(negative / positive))
        if logger: logger.debug("XGBoost scale_pos_weight=%.8f from training labels only", weight)
        self.model, self.device = fit_model(x, y, self.device, weight)
        return self

    def predict_proba(self, x):
        return self.model.predict_proba(x)[:, 1]

    def close(self):
        self.model = None
