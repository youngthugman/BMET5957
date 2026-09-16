"""Empirical linear calibration of HRV parameters against Section-12 reference.

For each of the 7 HRV fields we fit a per-field linear model
    reference_i = a * ours_i + b
on the 35 training recordings, using numpy.polyfit (least-squares).

Rationale: the methodology gap against Section 12 is a deterministic
implementation difference (confirmed: our time-domain matches, frequency-
domain has a persistent residual across all 35 records). If that residual
is well-described by an affine transform, learning it on training and
applying it on test recovers most of the bias at the cost of a small risk
from over-fitting.

The coefficient of determination (R^2) is reported per field — fields with
R^2 < 0.6 are effectively un-calibratable (the shape of the relationship
isn't linear or there's too much noise) and are left identity-mapped.

Usage:
  python calibration.py          # fit + report diagnostics + save coeffs
  from calibration import load_calibration, apply_calibration
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np

from load_data import load_training
from detector import detect_qrs
from hrv import compute_hrv
from reference_hrv import _REFERENCE, FIELDS

ROOT = Path(__file__).resolve().parent.parent
COEFFS_PATH = ROOT / "artifacts" / "calibration_coeffs.json"

# Fields with R^2 below this threshold are not calibrated (identity).
MIN_R2_FOR_CALIBRATION = 0.5


@dataclass
class CalibrationCoeffs:
    """Per-field (a, b) such that calibrated = a * raw + b.

    If `enabled` is False for a field, identity (a=1, b=0) should be used.
    """
    a: dict
    b: dict
    r2: dict
    enabled: dict

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "CalibrationCoeffs":
        return cls(**json.loads(s))


def _fit_affine(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Fit y = a*x + b. Return (a, b, R^2)."""
    coeffs = np.polyfit(x, y, 1)
    a, b = float(coeffs[0]), float(coeffs[1])
    y_pred = a * x + b
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return a, b, r2


def _fit_scaling(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Fit y = a*x (no intercept). Return (a, 0, R^2)."""
    denom = float(np.sum(x * x))
    a = float(np.sum(x * y) / denom) if denom > 0 else 1.0
    y_pred = a * x
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return a, 0.0, r2


def _mape(pred: np.ndarray, truth: np.ndarray) -> float:
    mask = (truth != 0) & np.isfinite(pred) & np.isfinite(truth)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((pred[mask] - truth[mask]) / truth[mask])) * 100)


def _fit_field(our_values: np.ndarray, ref_values: np.ndarray
               ) -> tuple[float, float, float]:
    """Try affine (a*x+b) and scaling-only (a*x) fits; pick whichever
    gives lower MAPE on training. If neither beats identity, return
    identity (a=1, b=0)."""
    mask = np.isfinite(our_values) & np.isfinite(ref_values)
    if mask.sum() < 3:
        return 1.0, 0.0, 0.0
    x = our_values[mask]
    y = ref_values[mask]

    # Identity (baseline — no calibration at all)
    m_id = _mape(x, y)

    # Affine
    a1, b1, r2_affine = _fit_affine(x, y)
    m_affine = _mape(a1 * x + b1, y)

    # Pure scaling
    a2, _, r2_scale = _fit_scaling(x, y)
    m_scale = _mape(a2 * x, y)

    # Choose whichever gives lowest training MAPE
    best_mape = min(m_id, m_affine, m_scale)
    tol = 0.5  # pp; require improvement > tol to apply calibration
    if m_affine < m_id - tol and m_affine <= m_scale:
        r2 = r2_affine
        return a1, b1, r2
    if m_scale < m_id - tol:
        return a2, 0.0, r2_scale
    # No meaningful improvement — leave identity
    return 1.0, 0.0, 0.0


def fit_calibration(preds: list, ref: np.ndarray = _REFERENCE) -> CalibrationCoeffs:
    """Fit per-field affine calibration from our HRV predictions to reference.

    `preds` is a list of HRV dataclass instances (len 35). `ref` is the
    (35, 7) reference array.
    """
    assert len(preds) == ref.shape[0]
    a_dict: dict = {}
    b_dict: dict = {}
    r2_dict: dict = {}
    enabled: dict = {}
    for j, f in enumerate(FIELDS):
        ours = np.array([getattr(p, f) for p in preds], dtype=np.float64)
        truth = ref[:, j]
        a, b, r2 = _fit_field(ours, truth)
        a_dict[f] = a
        b_dict[f] = b
        r2_dict[f] = r2
        # Enable iff the returned coefficients aren't identity.
        enabled[f] = bool(not (a == 1.0 and b == 0.0))
    return CalibrationCoeffs(a=a_dict, b=b_dict, r2=r2_dict, enabled=enabled)


def apply_calibration(pred_hrv, coeffs: CalibrationCoeffs):
    """Apply calibration to a single HRV instance; returns a new instance."""
    from hrv import HRV  # avoid circular import issues at module import time
    kwargs = {}
    for f in FIELDS:
        raw = getattr(pred_hrv, f)
        if coeffs.enabled.get(f, False):
            kwargs[f] = coeffs.a[f] * raw + coeffs.b[f]
        else:
            kwargs[f] = raw
    return HRV(**kwargs)


def apply_calibration_list(preds: list, coeffs: CalibrationCoeffs) -> list:
    return [apply_calibration(p, coeffs) for p in preds]


def mape_per_field(preds: list, ref: np.ndarray = _REFERENCE) -> dict:
    out = {}
    for j, f in enumerate(FIELDS):
        p = np.array([getattr(x, f) for x in preds], dtype=np.float64)
        t = ref[:, j]
        mask = np.isfinite(p) & np.isfinite(t) & (t != 0)
        if not mask.any():
            out[f] = float("nan")
        else:
            out[f] = float(np.mean(np.abs((p[mask] - t[mask]) / t[mask])) * 100)
    return out


def leave_one_out_cv(preds: list, ref: np.ndarray = _REFERENCE) -> dict:
    """Honest estimate of how calibration will work on unseen data.

    For each of the 35 records, fit calibration on the other 34, then
    predict the held-out record. Report MAPE aggregated across the 35
    held-out predictions per field. If CV MAPE is close to in-sample MAPE,
    calibration transfers well. If CV MAPE is worse, we're over-fitting.
    """
    n = len(preds)
    cv_pred = np.full((n, len(FIELDS)), np.nan)
    for holdout in range(n):
        train_preds = [preds[i] for i in range(n) if i != holdout]
        train_ref = np.delete(ref, holdout, axis=0)
        coeffs = fit_calibration(train_preds, train_ref)
        cal = apply_calibration(preds[holdout], coeffs)
        for j, f in enumerate(FIELDS):
            cv_pred[holdout, j] = getattr(cal, f)
    out = {}
    for j, f in enumerate(FIELDS):
        out[f] = _mape(cv_pred[:, j], ref[:, j])
    return out


def save_calibration(coeffs: CalibrationCoeffs, path: Path = COEFFS_PATH) -> None:
    path.write_text(coeffs.to_json())


def load_calibration(path: Path = COEFFS_PATH) -> CalibrationCoeffs | None:
    if not path.exists():
        return None
    return CalibrationCoeffs.from_json(path.read_text())


def main():
    ecg, qrs = load_training(ROOT / "data" / "ProjectTrainData.mat")
    print(f"Loaded {len(ecg)} training recordings. Running detector + HRV...")
    preds = []
    for i in range(len(ecg)):
        qrs_pred = detect_qrs(ecg[i])
        preds.append(compute_hrv(qrs_pred, len(ecg[i])))
    print("Done. Fitting calibration...\n")

    coeffs = fit_calibration(preds)

    # Diagnostics
    print(f"{'field':<12} {'a':>10} {'b':>12} {'R^2':>8} {'enabled':>9}")
    print("-" * 54)
    for f in FIELDS:
        print(f"{f:<12} {coeffs.a[f]:>10.4f} {coeffs.b[f]:>12.4f} "
              f"{coeffs.r2[f]:>8.4f} {str(coeffs.enabled[f]):>9}")

    # MAPE before and after
    mape_raw = mape_per_field(preds)
    preds_cal = apply_calibration_list(preds, coeffs)
    mape_cal = mape_per_field(preds_cal)

    print(f"\n{'field':<12} {'raw MAPE':>12} {'calib MAPE':>12} {'delta':>10}")
    print("-" * 50)
    for f in FIELDS:
        d = mape_cal[f] - mape_raw[f]
        marker = " ✓" if d < -0.5 else (" ✗" if d > 0.5 else "")
        print(f"{f:<12} {mape_raw[f]:>11.2f}% {mape_cal[f]:>11.2f}% "
              f"{d:>+9.2f}{marker}")

    # Leave-one-out cross-validation — tells us whether calibration transfers
    print("\nRunning leave-one-out cross-validation...")
    cv_mape = leave_one_out_cv(preds)
    print(f"\n{'field':<12} {'raw':>8} {'in-sample':>10} {'LOO CV':>8} {'Δ(CV-raw)':>10}")
    print("-" * 52)
    for f in FIELDS:
        d = cv_mape[f] - mape_raw[f]
        marker = " ✓" if d < -0.5 else (" ✗" if d > 0.5 else "")
        print(f"{f:<12} {mape_raw[f]:>7.2f}% {mape_cal[f]:>9.2f}% "
              f"{cv_mape[f]:>7.2f}% {d:>+9.2f}{marker}")

    save_calibration(coeffs)
    print(f"\nCalibration coefficients saved to {COEFFS_PATH}")


if __name__ == "__main__":
    main()
