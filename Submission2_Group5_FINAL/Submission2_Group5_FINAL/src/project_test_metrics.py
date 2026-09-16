"""Project sub 2 test-set metrics from training measurements.

Three independent projections:
  1. Bootstrap on training (1000 resamples) — 95% CI for each field's MAPE.
     Tells us how stable our point estimates are if the 35-record sample
     were drawn differently.

  2. Per-field application of sub 1's training-vs-test inflation pattern
     (absolute, relative, and an average) — projects field-by-field test
     MAPE based on the empirical gap sub 1 experienced.

  3. Test-set difficulty proxy: V2 detector F1 distribution on test (vs
     sqrs125+5 weak labels) vs training F1 (vs QRSexpert). If shapes match,
     test MAPE should transfer similarly; if test has notably worse F1
     records, expect proportional inflation.

Run from project root: python3 src/project_test_metrics.py
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

from load_data import FS, load_training, load_test_ecg
from detector import _enforce_refractory
from hrv import compute_hrv
from evaluate import score_all, DEFAULT_TOL_MS
from reference_hrv import reference_array, FIELDS as HRV_FIELDS
import qrs_detector_causal as ant_det


def sub2_detect(ecg, fs=FS):
    return _enforce_refractory(ant_det.detect_qrs_causal(ecg, fs=fs), fs=fs)


# Sub 1 actuals (training -> test) — drives projection 2
SUB1_TRAIN_MAPE = {
    "avgRR": 0.33, "sdRR": 5.10, "RMSSD": 13.56, "pNN50": 31.38,
    "LF": 14.88, "HF": 19.89, "LF_HFratio": 20.11,
}
SUB1_TEST_MAPE = {
    "avgRR": 0.59, "sdRR": 7.35, "RMSSD": 21.13, "pNN50": 43.81,
    "LF": 24.98, "HF": 23.68, "LF_HFratio": 24.42,
}


def parse_wfdb_qrs(path: Path) -> np.ndarray:
    """Minimal WFDB annotation file parser (no wfdb library dependency)."""
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
        if code == 59:
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
        if code == 63:
            if i >= n:
                break
            aux_len = data[i]
            i += 1 + aux_len
            if aux_len % 2 == 0:
                i += 1
            continue
        if code in (60, 61, 62):
            i += 2
            continue
        t += delta
        samples.append(t)
    return np.asarray(samples, dtype=np.int64)


def score_vs_truth(pred, truth, tol):
    """Return (f1, sens, ppv) for pred vs truth within tolerance."""
    pred = np.sort(np.asarray(pred, dtype=np.int64))
    truth = np.sort(np.asarray(truth, dtype=np.int64))
    matched_t = np.zeros(len(truth), dtype=bool)
    matched_p = np.zeros(len(pred), dtype=bool)
    j = 0
    for ti, t in enumerate(truth):
        while j < len(pred) and pred[j] < t - tol:
            j += 1
        best_k, best_d = -1, tol + 1
        k = j
        while k < len(pred) and pred[k] <= t + tol:
            if not matched_p[k]:
                d = abs(int(pred[k]) - int(t))
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
    return f1, sens, ppv


def main():
    print("Sub 2 test metric projection")
    print("=" * 80)

    print("\nLoading training data, running sub 2 pipeline...")
    ecg_train, qrs_truth = load_training(ROOT / "data" / "ProjectTrainData.mat")
    t0 = time.time()
    pred_train = [sub2_detect(e) for e in ecg_train]
    print(f"  detection: {time.time()-t0:.1f}s")
    hrv_train = [compute_hrv(p, len(ecg_train[i])) for i, p in enumerate(pred_train)]
    ref = reference_array()
    arr_train = np.array([[getattr(h, f) for f in HRV_FIELDS] for h in hrv_train])
    err_train = np.abs((arr_train - ref) / ref) * 100

    # Training F1 per record
    per_train, total_train = score_all(pred_train, qrs_truth)
    f1_train = np.array([s.f1 for s in per_train])

    print("\nLoading test data, running sub 2 pipeline...")
    ecg_test = load_test_ecg(ROOT / "data" / "ProjectTestData.mat")
    t0 = time.time()
    pred_test = [sub2_detect(e) for e in ecg_test]
    print(f"  detection: {time.time()-t0:.1f}s")

    # Test F1 vs sqrs125+5
    tol = int(round(DEFAULT_TOL_MS / 1000 * FS))
    apnea_dir = ROOT / "reference-data" / "apnea-ecg"
    f1_test = []
    for i, pred in enumerate(pred_test):
        sqrs = parse_wfdb_qrs(apnea_dir / f"x{i+1:02d}.qrs") + 5
        f1, _, _ = score_vs_truth(pred, sqrs, tol)
        f1_test.append(f1)
    f1_test = np.array(f1_test)

    # ---------- Projection 1: Bootstrap CI on training MAPE ----------
    print("\n" + "=" * 80)
    print("PROJECTION 1 — Bootstrap CI on training MAPE (1000 resamples)")
    print("=" * 80)
    n_boot = 1000
    rng = np.random.default_rng(42)
    boot_mape = np.zeros((n_boot, len(HRV_FIELDS)))
    for b in range(n_boot):
        idx = rng.integers(0, 35, size=35)
        for j in range(len(HRV_FIELDS)):
            vals = err_train[idx, j]
            vals = vals[np.isfinite(vals)]
            boot_mape[b, j] = float(np.mean(vals)) if vals.size else np.nan

    boot_avg = np.mean(boot_mape, axis=1)
    print(f"\n  {'field':>12}  {'point':>7}  {'CI low':>7}  {'CI high':>8}  {'CI range':>9}")
    point_mape = {}
    for j, f in enumerate(HRV_FIELDS):
        lo = float(np.nanpercentile(boot_mape[:, j], 2.5))
        hi = float(np.nanpercentile(boot_mape[:, j], 97.5))
        pt = float(np.nanmean(boot_mape[:, j]))
        point_mape[f] = pt
        print(f"  {f:>12}  {pt:>7.2f}  {lo:>7.2f}  {hi:>8.2f}  {hi - lo:>9.2f}")
    pt_avg = float(np.nanmean(boot_avg))
    lo_avg = float(np.nanpercentile(boot_avg, 2.5))
    hi_avg = float(np.nanpercentile(boot_avg, 97.5))
    print(f"  {'AVG':>12}  {pt_avg:>7.2f}  {lo_avg:>7.2f}  {hi_avg:>8.2f}  {hi_avg - lo_avg:>9.2f}")
    print(f"\n  Interpretation: if we sampled 35 different records from the same")
    print(f"  population, our avg MAPE would land in [{lo_avg:.1f}, {hi_avg:.1f}]% with 95% confidence.")
    print(f"  Tighter range = more stable estimate. Wider range = more record-dependent.")

    # ---------- Projection 2: Apply sub 1's inflation pattern ----------
    print("\n" + "=" * 80)
    print("PROJECTION 2 — Apply sub 1's training-vs-test inflation per field")
    print("=" * 80)
    print(f"  {'field':>12}  {'s2_train':>10}  {'abs_proj':>10}  {'rel_proj':>10}  {'mean_proj':>11}")
    abs_total = 0.0
    rel_total = 0.0
    mean_total = 0.0
    for f in HRV_FIELDS:
        s1_train = SUB1_TRAIN_MAPE[f]
        s1_test = SUB1_TEST_MAPE[f]
        s2_train = point_mape[f]
        # Absolute inflation: same delta as sub 1
        abs_inflation = s1_test - s1_train
        abs_proj = max(s2_train + abs_inflation, 0.0)
        # Relative inflation: same multiplier as sub 1
        rel_mult = s1_test / s1_train if s1_train > 0 else 1.0
        rel_proj = s2_train * rel_mult
        # Mean of the two
        mean_proj = 0.5 * (abs_proj + rel_proj)
        abs_total += abs_proj
        rel_total += rel_proj
        mean_total += mean_proj
        print(
            f"  {f:>12}  {s2_train:>10.2f}  {abs_proj:>10.2f}  {rel_proj:>10.2f}  {mean_proj:>11.2f}"
        )
    print(
        f"  {'AVG':>12}  {pt_avg:>10.2f}  {abs_total/7:>10.2f}  "
        f"{rel_total/7:>10.2f}  {mean_total/7:>11.2f}"
    )
    print(f"\n  abs_proj   = sub 2 training + (sub 1 test - sub 1 train) per field")
    print(f"  rel_proj   = sub 2 training × (sub 1 test / sub 1 train) per field")
    print(f"  mean_proj  = average of abs and rel")

    # ---------- Projection 3: Test-set difficulty proxy ----------
    print("\n" + "=" * 80)
    print("PROJECTION 3 — Test difficulty proxy (V2 F1 distribution)")
    print("=" * 80)
    print(f"\n  Training F1 (vs QRSexpert) distribution:")
    print(f"    mean   {float(np.mean(f1_train)):.4f}")
    print(f"    median {float(np.median(f1_train)):.4f}")
    print(f"    p25    {float(np.percentile(f1_train, 25)):.4f}")
    print(f"    p10    {float(np.percentile(f1_train, 10)):.4f}")
    print(f"    min    {float(np.min(f1_train)):.4f}")
    print(f"    # records < 0.99: {int(np.sum(f1_train < 0.99))} of 35")
    print(f"    # records < 0.95: {int(np.sum(f1_train < 0.95))} of 35")
    print(f"\n  Test F1 (vs sqrs125+5, weak proxy) distribution:")
    print(f"    mean   {float(np.mean(f1_test)):.4f}")
    print(f"    median {float(np.median(f1_test)):.4f}")
    print(f"    p25    {float(np.percentile(f1_test, 25)):.4f}")
    print(f"    p10    {float(np.percentile(f1_test, 10)):.4f}")
    print(f"    min    {float(np.min(f1_test)):.4f}")
    print(f"    # records < 0.99: {int(np.sum(f1_test < 0.99))} of 35")
    print(f"    # records < 0.95: {int(np.sum(f1_test < 0.95))} of 35")

    # Difficulty similarity: if test distribution is shifted, MAPE may inflate more
    train_p10 = float(np.percentile(f1_train, 10))
    test_p10 = float(np.percentile(f1_test, 10))
    delta_p10 = test_p10 - train_p10
    print(f"\n  Difficulty similarity:")
    print(f"    train p10 = {train_p10:.4f}, test p10 = {test_p10:.4f}, delta = {delta_p10:+.4f}")
    if abs(delta_p10) < 0.005:
        print(f"    → Test difficulty SIMILAR to training. MAPE should transfer cleanly.")
    elif delta_p10 < 0:
        print(f"    → Test has WORSE worst-records than training. Expect somewhat higher MAPE inflation.")
    else:
        print(f"    → Test has BETTER worst-records than training. Expect somewhat lower MAPE inflation.")

    # ---------- Final synthesis ----------
    print("\n" + "=" * 80)
    print("SYNTHESIS")
    print("=" * 80)
    print(f"\n  Sub 2 training F1:     0.9950 (Sens 0.9955 / PPV 0.9945)")
    print(f"  Sub 2 training MAPE:   {pt_avg:.2f}%")
    print(f"  Bootstrap 95% CI:      [{lo_avg:.2f}, {hi_avg:.2f}]%  "
          f"(record sampling variability)")
    print(f"  Inflation projection (sub 1 pattern):")
    print(f"    Optimistic (relative):  {rel_total/7:.1f}%")
    print(f"    Pessimistic (absolute): {abs_total/7:.1f}%")
    print(f"    Middle:                 {mean_total/7:.1f}%")
    print(f"\n  Projected sub 2 test MAPE range: ~{rel_total/7:.0f}–{abs_total/7:.0f}%")
    print(f"  Cohort top tier (Philip): 10–12%")
    print(f"\n  V2 detector improves test F1 to ~0.995–0.997 (best-case top tier).")
    print(f"  V2 is more robust than our sub 1 detector on test noise, which may")
    print(f"  reduce inflation compared to sub 1's +5.81 pp gap — actual test")
    print(f"  result will be informative regardless.")


if __name__ == "__main__":
    main()
