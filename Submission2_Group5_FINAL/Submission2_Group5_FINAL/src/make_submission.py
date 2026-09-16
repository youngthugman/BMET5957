"""End-to-end pipeline driver.

Two modes:
  --train  Run detector on training ECGs, score against QRSexpert, print metrics.
  --test   Run detector on test ECGs, write ProjectTestDataAnalysisGroup{G}Submission{S}.mat.

Usage examples:
  python make_submission.py --train
  python make_submission.py --train --limit 5        (quick smoke test)
  python make_submission.py --test --group 7 --sub 1

Sub 2 pipeline (active): Anthony V2 detector + our 200 ms refractory
enforcement + our iter-12 HRV. Calibration applied if coefficients file is
present in artifacts/. The old `detector.detect_qrs` is still importable from
detector.py if a future submission wants to revert.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import savemat


ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT / "data" / "ProjectTrainData.mat"
TEST_PATH = ROOT / "data" / "ProjectTestData.mat"
TEMPLATE_PATH = ROOT / "data" / "ProjectTestDataAnalysis.mat"
OUT_DIR = ROOT / "submissions"

# Anthony V2 detector lives in reference-data/Anthony-V2/Code/ — make it
# importable for the sub 2 pipeline.
ANT_V2 = ROOT / "reference-data" / "Anthony-V2" / "Code"
if str(ANT_V2) not in sys.path:
    sys.path.insert(0, str(ANT_V2))

from load_data import FS, load_training, load_test_ecg, load_submission_template
from detector import _enforce_refractory
from hrv import compute_hrv, hrv_mape_report
from evaluate import score_all, print_report
from reference_hrv import mape_against_reference, FIELDS as HRV_FIELDS
from calibration import load_calibration, apply_calibration
from qrs_detector_causal import detect_qrs_causal as _ant_detect_qrs_causal


def detect_qrs(ecg: np.ndarray, fs: int = FS) -> np.ndarray:
    """Sub 2 production detector: Anthony V2 detector + 200 ms refractory.

    Matches `src/final_sub2_metrics.py:sub2_detect()` exactly so the .mat
    written by this script is identical to what the verification suite
    measured (training F1 0.9950, avg MAPE 9.52% vs Section-12).
    """
    return _enforce_refractory(_ant_detect_qrs_causal(ecg, fs=fs), fs=fs)


def run_training(limit: int | None = None) -> None:
    ecg, qrs_truth = load_training(TRAIN_PATH)
    if limit:
        ecg, qrs_truth = ecg[:limit], qrs_truth[:limit]

    t0 = time.time()
    preds = [detect_qrs(e) for e in ecg]
    dt_det = time.time() - t0

    per, total = score_all(preds, qrs_truth)
    print(f"\nDetection: {dt_det:.1f}s for {len(ecg)} recordings "
          f"({dt_det/len(ecg):.2f}s/rec)")
    print_report(per, total)

    # HRV: compute on integer R-peak positions. Detector uses interval-based
    # periodogram with FFT=256 (W8 lecture default).
    hrv_pred = [compute_hrv(p, len(ecg[i])) for i, p in enumerate(preds)]
    if len(hrv_pred) == 35:
        ref_mape = mape_against_reference(hrv_pred)
        print("\nHRV MAPE (%) vs Section-12 reference (raw) — lower is better:")
        for k in HRV_FIELDS:
            print(f"  {k:>12}: {ref_mape[k]:.2f}")

        # Apply empirical calibration learned from Section-12 (if available)
        coeffs = load_calibration()
        if coeffs is not None:
            hrv_cal = [apply_calibration(h, coeffs) for h in hrv_pred]
            cal_mape = mape_against_reference(hrv_cal)
            print("\nHRV MAPE (%) vs Section-12 reference (calibrated) — "
                  "what we submit:")
            for k in HRV_FIELDS:
                flag = " (calibrated)" if coeffs.enabled.get(k, False) else ""
                print(f"  {k:>12}: {cal_mape[k]:.2f}{flag}")


def _resolve_calibration(mode: str):
    """Return (coeffs, label). `mode` in {'off','on','auto'}. Off is the sub 2 default."""
    if mode == "off":
        return None, "OFF (raw HRV — explicit --calibration off or default)"
    coeffs = load_calibration()
    if mode == "on":
        if coeffs is None:
            raise FileNotFoundError(
                "--calibration on but no artifacts/calibration_coeffs.json found"
            )
        n = sum(1 for v in coeffs.enabled.values() if v)
        return coeffs, f"ON ({n} fields enabled)"
    # auto
    if coeffs is None:
        return None, "AUTO → no file found, using raw HRV"
    n = sum(1 for v in coeffs.enabled.values() if v)
    return coeffs, f"AUTO → file found, applying ({n} fields enabled)"


def run_test_submission(group: int, sub: int, calibration_mode: str = "off") -> None:
    ecg = load_test_ecg(TEST_PATH)
    template = load_submission_template(TEMPLATE_PATH)

    coeffs, cal_label = _resolve_calibration(calibration_mode)
    print(f"Calibration: {cal_label}")

    t0 = time.time()
    preds = [detect_qrs(e) for e in ecg]
    print(f"Detection: {time.time()-t0:.1f}s for {len(ecg)} recordings")

    # Build QRS cell array matching the template's (1, 35) shape.
    qrs_cell = np.empty((1, len(preds)), dtype=object)
    for i, p in enumerate(preds):
        qrs_cell[0, i] = p.astype(np.int32).reshape(1, -1)

    out = {
        "QRS": qrs_cell,
        "avgRR": np.full((1, len(preds)), np.nan),
        "sdRR": np.full((1, len(preds)), np.nan),
        "RMSSD": np.full((1, len(preds)), np.nan),
        "pNN50": np.full((1, len(preds)), np.nan),
        "LF": np.full((1, len(preds)), np.nan),
        "HF": np.full((1, len(preds)), np.nan),
        "LF_HFratio": np.full((1, len(preds)), np.nan),
    }
    for i, p in enumerate(preds):
        h = compute_hrv(p, len(ecg[i]))
        if coeffs is not None:
            h = apply_calibration(h, coeffs)
        out["avgRR"][0, i] = h.avgRR
        out["sdRR"][0, i] = h.sdRR
        out["RMSSD"][0, i] = h.RMSSD
        out["pNN50"][0, i] = h.pNN50
        out["LF"][0, i] = h.LF
        out["HF"][0, i] = h.HF
        out["LF_HFratio"][0, i] = h.LF_HFratio

    OUT_DIR.mkdir(exist_ok=True)
    fname = OUT_DIR / f"ProjectTestDataAnalysisGroup{group}Submission{sub}.mat"
    savemat(fname, out)
    print(f"Wrote {fname}")
    print(f"Pipeline summary: V2 detector + 200ms refractory + ours HRV; calibration = {cal_label}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="Run on training data and score.")
    ap.add_argument("--test", action="store_true", help="Run on test data and write submission.")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of recordings (training).")
    ap.add_argument("--group", type=int, default=0)
    ap.add_argument("--sub", type=int, default=1)
    ap.add_argument(
        "--calibration",
        choices=["off", "on", "auto"],
        default="off",
        help="Calibration mode for test submission. 'off' (default — sub 2 raw HRV), "
        "'on' (require artifacts/calibration_coeffs.json), 'auto' (apply if file exists).",
    )
    args = ap.parse_args()

    if args.train:
        run_training(limit=args.limit)
    if args.test:
        run_test_submission(
            group=args.group, sub=args.sub, calibration_mode=args.calibration
        )
    if not (args.train or args.test):
        ap.print_help()


if __name__ == "__main__":
    main()
