"""Pre-submission verification of sub 2 candidate (V2 detector + Ours HRV).

Runs five checks:
  A. Per-record training F1 + per-field MAPE — confirm 9.52% isn't an
     aggregate-only fluke; flag any record whose F1 or MAPE regresses vs
     baseline (Ours D + Ours H).
  B. Test-set integrity — NaN, refractory >= 200 ms, monotonic, in-bounds.
  C. Test detection vs sqrs125+5 weak labels (iter 10 method, independent
     test F1 estimate).
  D. Test HRV value distributions vs sub 1 — sanity check for plausibility.
  E. Determinism — run V2 detector twice on first 3 records, confirm identical.

Run from project root: python3 src/verify_sub2.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
ANT_V2 = ROOT / "reference-data" / "Anthony-V2" / "Code"

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ANT_V2))

from load_data import FS, load_training, load_test_ecg
from detector import detect_qrs as our_detect, _enforce_refractory
from hrv import compute_hrv as our_hrv
from evaluate import score_all, DEFAULT_TOL_MS
from reference_hrv import reference_array, FIELDS as HRV_FIELDS

import qrs_detector_causal as ant_det


def sub2_detect(ecg, fs=FS):
    """Shipped sub 2 detector: V2 + 200ms refractory. Matches
    make_submission.detect_qrs and final_sub2_metrics.sub2_detect."""
    return _enforce_refractory(ant_det.detect_qrs_causal(ecg, fs=fs), fs=fs)


# -----------------------------------------------------------------------------
# WFDB annotation file parser (no wfdb library dependency)
# -----------------------------------------------------------------------------
def parse_wfdb_qrs(path: Path) -> np.ndarray:
    """Parse a WFDB binary annotation file (.qrs) and return sample positions.

    Format per physionet.org/physiotools/wag/annot-5.htm:
      Each annotation is a 2-byte little-endian word; low 10 bits = time
      delta from previous annotation (samples), high 6 bits = annotation
      code A. Code 0 = end-of-file. Code 59 (SKIP) = next 4 bytes hold a
      32-bit signed time delta (high16 first, both little-endian).
      Codes 60-63 (NUM, SUB, CHN, AUX) carry extra bytes.
    """
    data = path.read_bytes()
    samples: list[int] = []
    t = 0
    i = 0
    n = len(data)
    while i + 2 <= n:
        word = data[i] | (data[i + 1] << 8)
        i += 2
        code = (word >> 10) & 0x3F
        delta = word & 0x3FF
        if code == 0:
            break
        if code == 59:  # SKIP
            if i + 4 > n:
                break
            high = data[i] | (data[i + 1] << 8)
            low = data[i + 2] | (data[i + 3] << 8)
            i += 4
            skip = (high << 16) | low
            if skip & 0x80000000:
                skip -= 0x100000000
            t += skip
            continue
        if code == 63:  # AUX (length byte then string, padded to even)
            if i >= n:
                break
            aux_len = data[i]
            i += 1 + aux_len
            if aux_len % 2 == 0:
                i += 1
            continue
        if code in (60, 61, 62):  # NUM, SUB, CHN — 2 extra bytes each
            i += 2
            continue
        t += delta
        samples.append(t)
    return np.asarray(samples, dtype=np.int64)


# -----------------------------------------------------------------------------
# Check helpers
# -----------------------------------------------------------------------------
def field_mape_per_record(arr, ref):
    """Return (35, 7) per-record absolute % error array."""
    return np.abs((arr - ref) / np.where(ref != 0, ref, np.nan)) * 100


def field_mape_avg(arr, ref):
    err = field_mape_per_record(arr, ref)
    out = {}
    for j, f in enumerate(HRV_FIELDS):
        x = err[:, j]
        x = x[np.isfinite(x)]
        out[f] = float(np.mean(x)) if x.size else np.nan
    return out


def to_arr_ours(hrv_list):
    return np.array([[getattr(h, f) for f in HRV_FIELDS] for h in hrv_list])


def integrity_check(qrs_pred_list, hrv_list, ecg_lengths):
    """Return dict of integrity violations."""
    issues = {
        "nan_hrv": [],
        "refractory_violations": [],
        "not_monotonic": [],
        "out_of_bounds": [],
        "empty_qrs": [],
    }
    for i, (qrs, h, n) in enumerate(zip(qrs_pred_list, hrv_list, ecg_lengths)):
        rec = i + 1
        if len(qrs) == 0:
            issues["empty_qrs"].append(rec)
            continue
        if not np.all(np.diff(qrs) >= 0):
            issues["not_monotonic"].append(rec)
        if qrs.min() < 0 or qrs.max() >= n:
            issues["out_of_bounds"].append((rec, int(qrs.min()), int(qrs.max()), int(n)))
        gaps = np.diff(qrs.astype(np.int64))
        bad = int(np.sum(gaps < 20))  # 200 ms at 100 Hz = 20 samples
        if bad:
            issues["refractory_violations"].append((rec, bad))
        for f in HRV_FIELDS:
            v = getattr(h, f)
            if not np.isfinite(v):
                issues["nan_hrv"].append((rec, f))
    return issues


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print("Sub 2 verification — V2 detector + Ours HRV")
    print("=" * 80)

    # -------------------- A. Per-record training metrics --------------------
    print("\n" + "-" * 80)
    print("A. Per-record training F1 + MAPE — V2 D + Ours H vs Ours D + Ours H")
    print("-" * 80)
    ecg_train, qrs_truth = load_training(ROOT / "data" / "ProjectTrainData.mat")
    ref = reference_array()

    print("\nDetecting (V2)...")
    t0 = time.time()
    pred_v2 = [sub2_detect(e) for e in ecg_train]
    print(f"  {time.time()-t0:.1f}s")
    print("Detecting (ours)...")
    t0 = time.time()
    pred_ours = [our_detect(e) for e in ecg_train]
    print(f"  {time.time()-t0:.1f}s")

    print("Computing HRV with our pipeline (both detectors)...")
    hrv_v2_ours = [our_hrv(p, len(ecg_train[i])) for i, p in enumerate(pred_v2)]
    hrv_ours_ours = [our_hrv(p, len(ecg_train[i])) for i, p in enumerate(pred_ours)]

    per_v2, tot_v2 = score_all(pred_v2, qrs_truth)
    per_ours, tot_ours = score_all(pred_ours, qrs_truth)
    arr_v2 = to_arr_ours(hrv_v2_ours)
    arr_ours = to_arr_ours(hrv_ours_ours)
    err_v2 = field_mape_per_record(arr_v2, ref)
    err_ours = field_mape_per_record(arr_ours, ref)
    sum_err_v2 = np.nansum(err_v2, axis=1)
    sum_err_ours = np.nansum(err_ours, axis=1)

    print(f"\n{'rec':>4} {'F1_ours':>9} {'F1_V2':>8} {'dF1':>8} | "
          f"{'MAPEsum_ours':>13} {'MAPEsum_V2':>11} {'dSum':>8} | flag")
    n_f1_regress = 0
    n_mape_regress = 0
    for i in range(len(per_ours)):
        df1 = per_v2[i].f1 - per_ours[i].f1
        dsum = sum_err_v2[i] - sum_err_ours[i]
        flag = ""
        if df1 < -0.005:
            flag += "F1↓"; n_f1_regress += 1
        if dsum > 5.0:
            flag += " MAPE↑"; n_mape_regress += 1
        print(
            f"{i+1:>4} {per_ours[i].f1:>9.4f} {per_v2[i].f1:>8.4f} "
            f"{df1:>+8.4f} | "
            f"{sum_err_ours[i]:>13.1f} {sum_err_v2[i]:>11.1f} "
            f"{dsum:>+8.1f} | {flag}"
        )
    print(
        f"{'ALL':>4} {tot_ours.f1:>9.4f} {tot_v2.f1:>8.4f} "
        f"{tot_v2.f1 - tot_ours.f1:>+8.4f} | "
        f"avg MAPE: ours={float(np.mean(list(field_mape_avg(arr_ours, ref).values()))):.2f}%  "
        f"V2={float(np.mean(list(field_mape_avg(arr_v2, ref).values()))):.2f}%"
    )
    print(f"\n  Records with F1 drop > 0.005:    {n_f1_regress}")
    print(f"  Records with MAPE-sum increase > 5 pp: {n_mape_regress}")

    # -------------------- B + D. Run V2 on test --------------------
    print("\n" + "-" * 80)
    print("Running V2 detector + ours HRV on TEST data (for B + C + D)...")
    print("-" * 80)
    ecg_test = load_test_ecg(ROOT / "data" / "ProjectTestData.mat")
    test_lengths = [len(e) for e in ecg_test]
    print(f"  Loaded {len(ecg_test)} test records")

    print("\nDetecting (V2 on test)...")
    t0 = time.time()
    pred_test = [sub2_detect(e) for e in ecg_test]
    print(f"  {time.time()-t0:.1f}s")

    print("Computing HRV (ours on V2 detector output)...")
    t0 = time.time()
    hrv_test = [our_hrv(p, test_lengths[i]) for i, p in enumerate(pred_test)]
    print(f"  {time.time()-t0:.1f}s")

    # B. Integrity
    print("\n" + "-" * 80)
    print("B. Test-set integrity")
    print("-" * 80)
    issues = integrity_check(pred_test, hrv_test, test_lengths)
    for k, v in issues.items():
        if v:
            print(f"  FAIL: {k}: {v[:5]}{'...' if len(v) > 5 else ''} (total {len(v)})")
        else:
            print(f"  OK:   {k}")

    # D. HRV value distributions
    print("\n" + "-" * 80)
    print("D. Test HRV value distributions vs sub 1")
    print("-" * 80)
    sub1_path = ROOT / "submissions" / "ProjectTestDataAnalysisGroup5Submission1.mat"
    sub1 = loadmat(sub1_path)
    print(f"  {'field':>12}  {'sub1_min':>9}  {'sub1_max':>9}  {'sub1_mean':>10}  "
          f"{'sub2_min':>9}  {'sub2_max':>9}  {'sub2_mean':>10}  {'corr':>6}")
    field_to_sub1key = {
        "avgRR": "avgRR", "sdRR": "sdRR", "RMSSD": "RMSSD", "pNN50": "pNN50",
        "LF": "LF", "HF": "HF", "LF_HFratio": "LF_HFratio",
    }
    for f in HRV_FIELDS:
        s1_vals = np.asarray(sub1[field_to_sub1key[f]]).ravel()
        s2_vals = np.array([getattr(h, f) for h in hrv_test])
        finite1 = np.isfinite(s1_vals)
        finite2 = np.isfinite(s2_vals)
        both = finite1 & finite2
        if both.sum() >= 2:
            corr = float(np.corrcoef(s1_vals[both], s2_vals[both])[0, 1])
        else:
            corr = float("nan")
        print(
            f"  {f:>12}  {np.nanmin(s1_vals):>9.2f}  {np.nanmax(s1_vals):>9.2f}  "
            f"{np.nanmean(s1_vals):>10.2f}  {np.nanmin(s2_vals):>9.2f}  "
            f"{np.nanmax(s2_vals):>9.2f}  {np.nanmean(s2_vals):>10.2f}  {corr:>6.3f}"
        )

    # C. sqrs125+5 cross-check on test
    print("\n" + "-" * 80)
    print("C. Test detection vs PhysioNet sqrs125+5 weak labels")
    print("-" * 80)
    apnea_dir = ROOT / "reference-data" / "apnea-ecg"
    tol = int(round(DEFAULT_TOL_MS / 1000 * FS))
    print(f"  Tolerance: ±{tol} samples (±{DEFAULT_TOL_MS} ms)")
    print(f"  {'rec':>4}  {'PhysioNet_x':>11}  {'sqrs+5':>7}  {'V2_pred':>8}  "
          f"{'F1':>7}  {'Sens':>7}  {'PPV':>7}")

    f1_sum = []
    parse_failed = 0
    for i, pred in enumerate(pred_test):
        rec_name = f"x{i+1:02d}"
        qrs_path = apnea_dir / f"{rec_name}.qrs"
        if not qrs_path.exists():
            print(f"  {i+1:>4}  {rec_name:>11}  (no .qrs file)")
            parse_failed += 1
            continue
        try:
            sqrs = parse_wfdb_qrs(qrs_path) + 5
        except Exception as e:
            print(f"  {i+1:>4}  {rec_name:>11}  (parse failed: {e})")
            parse_failed += 1
            continue

        # Score V2 prediction vs sqrs125+5
        pred_sorted = np.sort(pred.astype(np.int64))
        truth_sorted = np.sort(sqrs.astype(np.int64))
        matched_t = np.zeros(len(truth_sorted), dtype=bool)
        matched_p = np.zeros(len(pred_sorted), dtype=bool)
        j = 0
        for ti, t in enumerate(truth_sorted):
            while j < len(pred_sorted) and pred_sorted[j] < t - tol:
                j += 1
            best_k, best_d = -1, tol + 1
            k = j
            while k < len(pred_sorted) and pred_sorted[k] <= t + tol:
                if not matched_p[k]:
                    d = abs(int(pred_sorted[k]) - int(t))
                    if d < best_d:
                        best_d, best_k = d, k
                k += 1
            if best_k >= 0:
                matched_p[best_k] = True
                matched_t[ti] = True
        tp = int(matched_t.sum())
        fp = int((~matched_p).sum())
        fn = int((~matched_t).sum())
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        ppv = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * sens * ppv / (sens + ppv) if (sens + ppv) else 0.0
        f1_sum.append(f1)
        print(
            f"  {i+1:>4}  {rec_name:>11}  {len(sqrs):>7}  {len(pred):>8}  "
            f"{f1:>7.4f}  {sens:>7.4f}  {ppv:>7.4f}"
        )
    if f1_sum:
        print(
            f"  {'AVG':>4}                                "
            f"{np.mean(f1_sum):>7.4f}"
        )
    print(f"  Sub 1 cross-check (for comparison): F1 = 0.9872 (per [[Sub 1]])")

    # E. Determinism
    print("\n" + "-" * 80)
    print("E. Determinism — run V2 detector twice on first 3 records")
    print("-" * 80)
    ok = True
    for i in range(3):
        a = sub2_detect(ecg_test[i])
        b = sub2_detect(ecg_test[i])
        match = (len(a) == len(b)) and bool(np.all(a == b))
        print(f"  rec x{i+1:02d}: {'OK' if match else 'FAIL'}  ({len(a)} vs {len(b)} beats)")
        if not match:
            ok = False
    print(f"  Determinism: {'OK' if ok else 'FAIL'}")

    print("\n" + "=" * 80)
    print("DONE")


if __name__ == "__main__":
    main()
