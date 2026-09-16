"""Post-build verification: load the freshly written sub 2 .mat and confirm
its QRS + HRV fields exactly match what our in-memory pipeline produces.

Per Codex review: 'verify its QRS counts/HRV fields come from the same
V2+refractory path.' This closes the loop between measurement and shipped
file.

Run from project root: python3 src/verify_mat.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
ANT_V2 = ROOT / "reference-data" / "Anthony-V2" / "Code"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ANT_V2))

from load_data import FS, load_test_ecg
from detector import _enforce_refractory
from hrv import compute_hrv
from calibration import load_calibration, apply_calibration
import qrs_detector_causal as ant_det

MAT_PATH = ROOT / "submissions" / "ProjectTestDataAnalysisGroup5Submission2.mat"

HRV_FIELDS = ["avgRR", "sdRR", "RMSSD", "pNN50", "LF", "HF", "LF_HFratio"]


def sub2_detect(ecg, fs=FS):
    return _enforce_refractory(ant_det.detect_qrs_causal(ecg, fs=fs), fs=fs)


def sub2_hrv(qrs, ecg_len, coeffs=None):
    """Match make_submission's full HRV path: compute then optionally calibrate."""
    h = compute_hrv(qrs, ecg_len)
    if coeffs is not None:
        h = apply_calibration(h, coeffs)
    return h


def main():
    print(f"Loading {MAT_PATH.name}...")
    mat = loadmat(str(MAT_PATH))

    qrs_cell = mat["QRS"]
    n_records = qrs_cell.size
    print(f"  {n_records} QRS cells in .mat")

    print(f"\nLoading test ECG and re-running pipeline (V2 + refractory + ours HRV)...")
    ecg_test = load_test_ecg(ROOT / "data" / "ProjectTestData.mat")

    qrs_match_count = 0
    hrv_match_count = 0
    issues = []
    qrs_loaded = []

    for i in range(n_records):
        # Load QRS from .mat
        q_loaded = np.asarray(qrs_cell.flat[i]).flatten().astype(np.int64)
        q_loaded = np.sort(q_loaded)
        qrs_loaded.append(q_loaded)

        # Re-run production path
        q_recomputed = sub2_detect(ecg_test[i])

        # Compare QRS arrays
        if len(q_loaded) == len(q_recomputed) and bool(np.all(q_loaded == q_recomputed)):
            qrs_match_count += 1
        else:
            issues.append((i + 1, "QRS mismatch", len(q_loaded), len(q_recomputed)))

    print(f"\n  QRS match (loaded vs recomputed): {qrs_match_count}/{n_records}")

    # HRV comparison — match production: compute then calibrate if file exists
    coeffs = load_calibration()
    if coeffs is None:
        print("\n  No calibration file found — verifying against RAW HRV.")
    else:
        n_enabled = sum(1 for v in coeffs.enabled.values() if v)
        print(f"\n  Calibration file present — verifying against CALIBRATED HRV "
              f"({n_enabled} fields enabled).")
    hrv_recomputed = []
    for i in range(n_records):
        hrv_recomputed.append(sub2_hrv(qrs_loaded[i], len(ecg_test[i]), coeffs))

    print(f"\n  Comparing per-field HRV values:")
    field_to_key = {f: f for f in HRV_FIELDS}
    all_match = True
    for f in HRV_FIELDS:
        loaded = np.asarray(mat[field_to_key[f]]).ravel()
        recomp = np.array([getattr(h, f) for h in hrv_recomputed])
        # Allow tiny float imprecision
        diff = np.abs(loaded - recomp)
        max_diff = float(np.nanmax(diff))
        n_match = int(np.sum(diff < 1e-6))
        ok = n_match == n_records
        all_match = all_match and ok
        flag = "OK" if ok else f"FAIL (max diff {max_diff:.2e})"
        print(f"    {f:>12}  {n_match}/{n_records} match  {flag}")

    # Integrity checks
    print(f"\n  Integrity checks on loaded .mat:")
    n_nan = 0
    n_refr = 0
    n_oob = 0
    n_unsorted = 0
    for i in range(n_records):
        q = qrs_loaded[i]
        n = len(ecg_test[i])
        gaps = np.diff(q)
        n_refr += int(np.sum(gaps < 20))  # 200ms = 20 samples at 100Hz
        if not np.all(gaps >= 0):
            n_unsorted += 1
        if q.min() < 0 or q.max() >= n:
            n_oob += 1
        for f in HRV_FIELDS:
            v = float(np.asarray(mat[f]).ravel()[i])
            if not np.isfinite(v):
                n_nan += 1
    print(f"    Refractory violations (<200ms): {n_refr}  (target 0)")
    print(f"    Unsorted records:               {n_unsorted}  (target 0)")
    print(f"    Out-of-bounds records:          {n_oob}  (target 0)")
    print(f"    NaN/Inf HRV cells:              {n_nan}  (target 0)")

    # Determinism re-check
    print(f"\n  Determinism (V2+refractory on first 3 test records):")
    for i in range(3):
        a = sub2_detect(ecg_test[i])
        b = sub2_detect(ecg_test[i])
        match = (len(a) == len(b)) and bool(np.all(a == b))
        print(f"    rec x{i+1:02d}: {'OK' if match else 'FAIL'}  ({len(a)} vs {len(b)})")

    print(f"\n{'=' * 70}")
    if (qrs_match_count == n_records and all_match
            and n_refr == 0 and n_unsorted == 0
            and n_oob == 0 and n_nan == 0):
        print("VERDICT: SHIP READY — production path matches measured metrics exactly")
    else:
        print("VERDICT: NOT READY — discrepancies above")
    print(f"{'=' * 70}")
    print(f"Sub 2 file: {MAT_PATH.name}")
    print(f"  Path: {MAT_PATH}")


if __name__ == "__main__":
    main()
