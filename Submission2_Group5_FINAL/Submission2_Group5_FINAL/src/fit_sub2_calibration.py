"""Refit per-field linear calibration on the V2+refractory + ours HRV outputs.

Sub 2 specifically applies calibration to LF and LF/HF only (partial cal —
Option C in the planning conversation). LOO improvements per field on
V2+refractory training outputs:

  Field          raw MAPE   refit-cal MAPE   delta
  avgRR           0.23           0.23         0.00  (identity)
  sdRR            3.31           3.31         0.00  (identity, in-sample win <0.5pp)
  RMSSD           7.95           7.10        -0.85  (small, kept identity for partial)
  pNN50           6.16           5.64        -0.52  (small, kept identity for partial)
  LF             16.83           9.39        -7.44  (HUGE — enable)
  HF             13.44          13.44         0.00  (identity)
  LF_HFratio     18.74          14.98        -3.76  (substantial — enable)

We enable only the two fields with structural calibration wins (LF, LF/HF).
sdRR/RMSSD/pNN50 are left identity to minimise the risk surface on test, at
a small cost in expected MAPE (~0.2 pp).

Writes coefficients to artifacts/calibration_coeffs.json. Backs up the
existing iter-12 coefficients before overwriting.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
ANT_V2 = ROOT / "reference-data" / "Anthony-V2" / "Code"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ANT_V2))

from load_data import FS, load_training
from detector import _enforce_refractory
from hrv import compute_hrv
from reference_hrv import reference_array, FIELDS as HRV_FIELDS
from calibration import fit_calibration, CalibrationCoeffs, COEFFS_PATH
import qrs_detector_causal as ant_det


# Fields enabled for sub 2 calibration (Option C — partial)
SUB2_ENABLE = {
    "avgRR": False,
    "sdRR": False,
    "RMSSD": False,
    "pNN50": False,
    "LF": True,
    "HF": False,
    "LF_HFratio": True,
}


def sub2_detect(ecg, fs=FS):
    return _enforce_refractory(ant_det.detect_qrs_causal(ecg, fs=fs), fs=fs)


def main():
    print("Fitting sub 2 calibration on V2+refractory + ours HRV outputs")
    print("=" * 72)

    ecg, qrs_truth = load_training(ROOT / "data" / "ProjectTrainData.mat")

    print("\nDetecting + computing HRV...")
    t0 = time.time()
    preds = [sub2_detect(e) for e in ecg]
    hrv = [compute_hrv(p, len(ecg[i])) for i, p in enumerate(preds)]
    print(f"  {time.time()-t0:.1f}s")

    # Fit full calibration (then we'll zero out the fields we want identity)
    full = fit_calibration(hrv, ref=reference_array())

    # Override enabled-set per Option C, zero out coefficients for disabled fields
    a_partial = dict(full.a)
    b_partial = dict(full.b)
    r2_partial = dict(full.r2)
    enabled_partial = dict(full.enabled)
    for f in HRV_FIELDS:
        if SUB2_ENABLE[f]:
            # keep refit coefficients, mark enabled
            enabled_partial[f] = True
        else:
            # force identity, mark disabled
            a_partial[f] = 1.0
            b_partial[f] = 0.0
            enabled_partial[f] = False

    coeffs = CalibrationCoeffs(
        a=a_partial,
        b=b_partial,
        r2=r2_partial,
        enabled=enabled_partial,
    )

    # Diagnostics
    print(f"\n{'field':<12} {'a':>10} {'b':>10} {'R²':>8} {'enabled':>9}")
    print("-" * 52)
    for f in HRV_FIELDS:
        print(
            f"{f:<12} {coeffs.a[f]:>10.4f} {coeffs.b[f]:>+10.3f} "
            f"{coeffs.r2[f]:>8.4f} {str(coeffs.enabled[f]):>9}"
        )

    # Per-field MAPE comparison (raw vs calibrated with partial cal)
    ref = reference_array()
    arr_raw = np.array([[getattr(h, f) for f in HRV_FIELDS] for h in hrv])
    arr_cal = arr_raw.copy()
    for j, f in enumerate(HRV_FIELDS):
        if coeffs.enabled[f]:
            arr_cal[:, j] = coeffs.a[f] * arr_raw[:, j] + coeffs.b[f]
    err_raw = np.abs((arr_raw - ref) / ref) * 100
    err_cal = np.abs((arr_cal - ref) / ref) * 100

    print(f"\n{'field':<12} {'raw MAPE':>10} {'cal MAPE':>10} {'delta':>10}")
    print("-" * 46)
    for j, f in enumerate(HRV_FIELDS):
        r = float(np.nanmean(err_raw[:, j]))
        c = float(np.nanmean(err_cal[:, j]))
        d = c - r
        flag = "" if abs(d) < 0.05 else (" ↓" if d < 0 else " ↑")
        print(f"{f:<12} {r:>9.2f}% {c:>9.2f}% {d:>+9.2f}{flag}")
    raw_avg = float(np.mean([float(np.nanmean(err_raw[:, j])) for j in range(7)]))
    cal_avg = float(np.mean([float(np.nanmean(err_cal[:, j])) for j in range(7)]))
    print(f"{'AVG':<12} {raw_avg:>9.2f}% {cal_avg:>9.2f}% {cal_avg - raw_avg:>+9.2f}")

    # Back up and save
    backup_path = COEFFS_PATH.with_suffix(".json.before_sub2")
    if COEFFS_PATH.exists():
        shutil.copy(COEFFS_PATH, backup_path)
        print(f"\nBacked up existing coefficients to: {backup_path.name}")
    elif (COEFFS_PATH.parent / "calibration_coeffs.json.saved_for_sub2").exists():
        print(f"\nExisting saved-for-sub2 coefficients preserved.")

    COEFFS_PATH.write_text(coeffs.to_json())
    print(f"Wrote sub 2 calibration coefficients to: {COEFFS_PATH}")
    print(f"\nUse with: python3 src/make_submission.py --test --group 5 --sub 2 --calibration on")


if __name__ == "__main__":
    main()
