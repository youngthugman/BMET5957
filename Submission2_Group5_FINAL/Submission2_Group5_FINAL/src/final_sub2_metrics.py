"""Final pre-submission metrics for sub 2 (V2 detector + refractory + Ours HRV).

Pipeline:
  detect_qrs(ecg) = _enforce_refractory(ant_det.detect_qrs_causal(ecg), FS)
  compute_hrv(qrs, ecg_len) = our_hrv

Outputs a clean shareable report:
  - Per-record F1 / Sens / PPV
  - Per-record per-field absolute % error
  - Aggregate F1 and per-field MAPE
  - Integrity confirmation (no refractory violations)
"""

from __future__ import annotations

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
from hrv import compute_hrv as our_hrv
from evaluate import score_all
from reference_hrv import reference_array, FIELDS as HRV_FIELDS
import qrs_detector_causal as ant_det


def sub2_detect(ecg):
    """V2 detector + 200 ms refractory enforcement."""
    return _enforce_refractory(ant_det.detect_qrs_causal(ecg), FS)


def main():
    ecg, qrs_truth = load_training(ROOT / "data" / "ProjectTrainData.mat")
    ref = reference_array()

    print("Running V2 + refractory detector on 35 training records...")
    t0 = time.time()
    preds = [sub2_detect(e) for e in ecg]
    print(f"  detection: {time.time()-t0:.1f}s")

    t0 = time.time()
    hrv = [our_hrv(p, len(ecg[i])) for i, p in enumerate(preds)]
    print(f"  HRV: {time.time()-t0:.1f}s")

    # Integrity: no sub-200ms gaps
    violations = 0
    for p in preds:
        violations += int(np.sum(np.diff(p.astype(np.int64)) < 20))
    print(f"  refractory violations (<200ms): {violations} (target 0)")

    per, total = score_all(preds, qrs_truth)
    arr = np.array(
        [[getattr(h, f) for f in HRV_FIELDS] for h in hrv]
    )
    err = np.abs((arr - ref) / np.where(ref != 0, ref, np.nan)) * 100

    # --------------------------------------------------------------------
    # Shareable report
    # --------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("SUB 2 CANDIDATE — V2 DETECTOR + REFRACTORY ENFORCEMENT + OURS HRV")
    print("Pipeline: Anthony V2 detector → 200 ms refractory pass → "
          "our iter-12 HRV (hybrid Lomb + PCHIP-FFT, no calibration)")
    print("=" * 90)

    print("\nPER-RECORD F1 / Sens / PPV  (training, vs QRSexpert, ±50 ms)")
    print("-" * 90)
    print(f"{'rec':>4}  {'TP':>7}  {'FP':>5}  {'FN':>5}  "
          f"{'Sens':>7}  {'PPV':>7}  {'F1':>7}")
    for i, s in enumerate(per):
        print(f"{i+1:>4}  {s.tp:>7}  {s.fp:>5}  {s.fn:>5}  "
              f"{s.sensitivity:>7.4f}  {s.ppv:>7.4f}  {s.f1:>7.4f}")
    print("-" * 90)
    print(f"{'ALL':>4}  {total.tp:>7}  {total.fp:>5}  {total.fn:>5}  "
          f"{total.sensitivity:>7.4f}  {total.ppv:>7.4f}  {total.f1:>7.4f}")

    print("\nPER-RECORD ABSOLUTE % ERROR PER FIELD  (training, vs Section-12)")
    print("-" * 90)
    print(f"{'rec':>4}  " + "  ".join(f"{f:>10}" for f in HRV_FIELDS))
    for i in range(35):
        print(f"{i+1:>4}  " + "  ".join(f"{err[i,j]:>10.2f}" for j in range(len(HRV_FIELDS))))

    print("\nAGGREGATE MAPE PER FIELD")
    print("-" * 90)
    mape = {}
    for j, f in enumerate(HRV_FIELDS):
        x = err[:, j]
        x = x[np.isfinite(x)]
        mape[f] = float(np.mean(x))
        print(f"  {f:>12}  MAPE = {mape[f]:>7.2f}%")
    avg = float(np.mean(list(mape.values())))
    print(f"  {'Average':>12}  MAPE = {avg:>7.2f}%")

    print("\n" + "=" * 90)
    print("HEADLINE")
    print("=" * 90)
    print(f"  Training F1 (±50 ms):  {total.f1:.4f}  "
          f"(Sens {total.sensitivity:.4f} / PPV {total.ppv:.4f})")
    print(f"  Training avg MAPE:     {avg:.2f}%")
    print(f"  Refractory violations: {violations}")
    print(f"  Pipeline causality:    detector core causal; V2 post-filters "
          f"use whole-recording medians (lenient interpretation only)")
    print()
    print(f"  Comparison points:")
    print(f"    Our sub 1 ship (baseline):       F1 0.9931 / MAPE 15.04%")
    print(f"    Sub 1 actual grader return:      F1 0.9940 / MAPE 20.85%")
    print(f"    Cohort top-tier (Philip):        F1 ~0.997 / MAPE 10–12%")
    print(f"    Our HRV ceiling (expert QRS):    MAPE 7.59% (lower bound)")
    print("=" * 90)


if __name__ == "__main__":
    main()
