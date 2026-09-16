"""Decompose Josh's 7.26% advantage: calibration vs Welch ratio path.

Cells tested on training (all use V2 + refractory + ours HRV):
  A. Raw (no cal) — current sub 2 baseline (9.52% measured)
  B. Apply Josh's literal calibration coefficients
  C. Apply our existing calibration.py coefficients (saved_for_sub2)
  D. Refit per-field linear calibration on V2+refractory outputs (in-sample)
  E. Refit per-field calibration, strict LOO (out-of-sample transfer estimate)

Also reports per-field results for transparency.
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
from hrv import compute_hrv
from reference_hrv import reference_array, FIELDS as HRV_FIELDS
import qrs_detector_causal as ant_det


# Josh's literal calibration coefficients (from /tmp/josh_qrs/src/hrv.py)
JOSH_CAL = {
    "avgRR": (1.0, 0.0),
    "sdRR": (0.9866953876221705, 0.0),
    "RMSSD": (1.0123328025886276, -1.2176481640411145),
    "pNN50": (0.9908907495829226, 0.0),
    "LF": (1.1610889821772639, 0.0),
    "HF": (1.0, 0.0),
    "LF_HFratio": (1.114682918877966, 0.0),
}

# Our existing calibration coefficients (per docs: artifacts/.saved_for_sub2)
# From iter 12 refit, per Calibration.md
OUR_CAL = {
    "avgRR": (1.0, 0.0),
    "sdRR": (0.970, 0.0),
    "RMSSD": (1.021, -3.50),
    "pNN50": (0.967, 0.0),
    "LF": (1.145, 0.0),
    "HF": (1.0, 0.0),
    "LF_HFratio": (1.0, 0.0),
}


def sub2_detect(ecg, fs=FS):
    return _enforce_refractory(ant_det.detect_qrs_causal(ecg, fs=fs), fs=fs)


def apply_cal(arr, cal_dict):
    out = arr.copy()
    for j, f in enumerate(HRV_FIELDS):
        a, b = cal_dict[f]
        out[:, j] = a * out[:, j] + b
    return out


def fit_calibration_per_field(pred, ref, min_improvement_pp=0.5):
    """Refit per-field linear calibration. Choose affine/scaling-only/identity
    to minimise training MAPE. Returns coefficient dict."""
    cal = {}
    for j, f in enumerate(HRV_FIELDS):
        p = pred[:, j]
        t = ref[:, j]
        m = np.isfinite(p) & np.isfinite(t) & (t != 0)
        if m.sum() < 5:
            cal[f] = (1.0, 0.0)
            continue
        x = p[m]
        y = t[m]

        # Identity
        mape_id = float(np.mean(np.abs((x - y) / y))) * 100

        # Scaling only
        a_s = float(np.sum(x * y) / np.sum(x * x))
        mape_s = float(np.mean(np.abs((a_s * x - y) / y))) * 100

        # Affine
        A = np.column_stack([x, np.ones_like(x)])
        coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        a_a, b_a = float(coef[0]), float(coef[1])
        mape_a = float(np.mean(np.abs((a_a * x + b_a - y) / y))) * 100

        # Pick best with improvement threshold
        choices = [
            ("affine", mape_a, (a_a, b_a)),
            ("scaling", mape_s, (a_s, 0.0)),
        ]
        best_label, best_mape, best_coef = min(choices, key=lambda c: c[1])
        if mape_id - best_mape < min_improvement_pp:
            cal[f] = (1.0, 0.0)
        else:
            cal[f] = best_coef
    return cal


def loo_cal_mape(pred, ref, min_improvement_pp=0.5):
    """Strict LOO: for each record, fit cal on the other 34, apply to held-out."""
    n = pred.shape[0]
    pred_cal = pred.copy()
    for i in range(n):
        train_mask = np.arange(n) != i
        cal = fit_calibration_per_field(pred[train_mask], ref[train_mask], min_improvement_pp)
        for j, f in enumerate(HRV_FIELDS):
            a, b = cal[f]
            pred_cal[i, j] = a * pred[i, j] + b
    return pred_cal


def report_mape(arr, ref, label):
    err = np.abs((arr - ref) / ref) * 100
    per_field = {}
    for j, f in enumerate(HRV_FIELDS):
        x = err[:, j]
        x = x[np.isfinite(x)]
        per_field[f] = float(np.mean(x))
    avg = float(np.mean(list(per_field.values())))
    return per_field, avg


def main():
    print("Sub 2 calibration variant comparison")
    print("=" * 80)

    ecg, qrs_truth = load_training(ROOT / "data" / "ProjectTrainData.mat")
    ref = reference_array()

    print("\nDetecting + computing HRV (V2 + refractory + ours HRV, raw)...")
    t0 = time.time()
    preds = [sub2_detect(e) for e in ecg]
    hrv = [compute_hrv(p, len(ecg[i])) for i, p in enumerate(preds)]
    arr_raw = np.array([[getattr(h, f) for f in HRV_FIELDS] for h in hrv])
    print(f"  {time.time()-t0:.1f}s")

    # --- Cells A through D ---
    cells = {
        "A_raw_no_cal": arr_raw,
        "B_josh_cal": apply_cal(arr_raw, JOSH_CAL),
        "C_our_cal_iter12": apply_cal(arr_raw, OUR_CAL),
        "D_refit_in_sample": apply_cal(arr_raw, fit_calibration_per_field(arr_raw, ref)),
    }

    # --- Cell E: strict LOO ---
    print("\nFitting LOO calibration (35 folds)...")
    arr_loo = loo_cal_mape(arr_raw, ref)
    cells["E_refit_strict_LOO"] = arr_loo

    # Refit calibration is what we'd actually save
    fit_full = fit_calibration_per_field(arr_raw, ref)

    # ---------- Report ----------
    print("\n" + "-" * 80)
    print("Per-field MAPE matrix (lower is better)")
    print("-" * 80)
    field_results = {}
    avg_results = {}
    for name, arr in cells.items():
        per_field, avg = report_mape(arr, ref, name)
        field_results[name] = per_field
        avg_results[name] = avg
    print(f"{'field':>12}  " + "  ".join(f"{n:>18}" for n in cells.keys()))
    for f in HRV_FIELDS:
        row = [field_results[n][f] for n in cells.keys()]
        print(f"{f:>12}  " + "  ".join(f"{v:>17.2f}*" if v == min(row) else f"{v:>18.2f}" for v in row))
    avgs = list(avg_results.values())
    print(f"{'AVG':>12}  " + "  ".join(f"{v:>17.2f}*" if v == min(avgs) else f"{v:>18.2f}" for v in avgs))

    # ---------- Refit coefficients ----------
    print("\n" + "-" * 80)
    print(f"Refit calibration coefficients on V2+refractory + ours HRV (in-sample):")
    print("-" * 80)
    for f in HRV_FIELDS:
        a, b = fit_full[f]
        josh_a, josh_b = JOSH_CAL[f]
        active = "ENABLED" if (a, b) != (1.0, 0.0) else "identity"
        print(f"  {f:>12}: a = {a:>7.4f}, b = {b:>+8.3f}  [{active}]  "
              f"(vs Josh: a={josh_a:.4f}, b={josh_b:+.3f})")

    # ---------- Headline summary ----------
    print("\n" + "=" * 80)
    print("HEADLINE")
    print("=" * 80)
    print(f"  A. Raw (no cal, current sub 2): avg MAPE = {avg_results['A_raw_no_cal']:.2f}%")
    print(f"  B. Josh's coefficients:        avg MAPE = {avg_results['B_josh_cal']:.2f}%  "
          f"(Δ = {avg_results['B_josh_cal'] - avg_results['A_raw_no_cal']:+.2f} pp)")
    print(f"  C. Our iter-12 calibration:    avg MAPE = {avg_results['C_our_cal_iter12']:.2f}%  "
          f"(Δ = {avg_results['C_our_cal_iter12'] - avg_results['A_raw_no_cal']:+.2f} pp)")
    print(f"  D. Refit on V2+ref (in-sample): avg MAPE = {avg_results['D_refit_in_sample']:.2f}%  "
          f"(Δ = {avg_results['D_refit_in_sample'] - avg_results['A_raw_no_cal']:+.2f} pp)")
    print(f"  E. Refit strict LOO (transfer): avg MAPE = {avg_results['E_refit_strict_LOO']:.2f}%  "
          f"(Δ = {avg_results['E_refit_strict_LOO'] - avg_results['A_raw_no_cal']:+.2f} pp)")
    print()
    print(f"  Josh's published number:       avg MAPE = 7.26%")
    print(f"  Note: Josh's number is his coefficients × his outputs; the cell that matches")
    print(f"  his published number identically would only happen if we run his exact pipeline.")
    print(f"  Cell D shows the best in-sample fit on OUR pipeline outputs.")
    print(f"  Cell E shows what would actually transfer to test (LOO is the honest estimate).")


if __name__ == "__main__":
    main()
