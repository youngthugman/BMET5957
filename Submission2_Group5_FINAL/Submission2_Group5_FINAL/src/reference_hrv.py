"""Reference HRV values for the 35 training recordings.

Source: BMET3997_9997_MajorProject_FurtherInfo.pdf, Section 12 (released
2026-04-19). These are the exact values the grader will compare against
(computed by the teaching staff from the expert QRS annotations using their
canonical methodology — 5-min non-overlapping windows, RR bounds
[250, 2000] ms, min 4 min valid per window).

Use these directly for MAPE computation on the training set rather than
re-deriving values from the expert QRS with our own cleaner.
"""

from __future__ import annotations

import numpy as np


# Column order: avgRR [ms], sdRR [ms], RMSSD [ms], pNN50 [%],
#               LF [ms^2], HF [ms^2], LF_HFratio [unitless]
# Row i = recording i+1 (records are 1-indexed in the PDF; here 0-indexed).
_REFERENCE = np.array([
    # avgRR   sdRR    RMSSD   pNN50   LF       HF       LF_HFratio
    [ 992.0,  143.8,   77.1,  36.4,   8947.0,   1910.0,  5.10],   # rec 1
    [ 750.0,   45.3,   22.4,   1.6,    807.0,    109.0, 15.08],   # rec 2
    [ 922.0,  116.0,   77.5,  15.7,   2446.0,    959.0,  2.88],   # rec 3
    [ 960.0,   84.6,   42.0,   9.0,   1913.0,    414.0,  7.55],   # rec 4
    [ 949.0,   71.4,   31.8,   8.4,   1268.0,    375.0,  4.02],   # rec 5
    [ 990.0,   63.4,   52.0,  18.7,   1200.0,    977.0,  1.59],   # rec 6
    [ 820.0,  102.9,   59.5,  16.8,   5207.0,   1275.0,  4.24],   # rec 7
    [ 731.0,   47.9,   25.9,   2.3,    716.0,    139.0,  6.04],   # rec 8
    [ 947.0,   52.1,   40.4,   1.5,    473.0,    239.0,  3.77],   # rec 9
    [ 963.0,   63.3,   34.3,   8.1,   1088.0,    509.0,  2.64],   # rec 10
    [ 850.0,   43.1,   31.4,   6.2,    612.0,    363.0,  2.33],   # rec 11
    [ 977.0,   94.5,   42.8,  11.6,   2754.0,    455.0,  9.10],   # rec 12
    [ 748.0,   56.9,   30.7,   3.0,   1089.0,    161.0,  7.62],   # rec 13
    [1086.0,  141.4,   90.4,  44.5,   8361.0,   2919.0,  3.15],   # rec 14
    [ 905.0,   78.6,   28.3,   3.5,   2267.0,    230.0, 16.58],   # rec 15
    [ 833.0,   78.3,   37.5,   9.2,   2427.0,    495.0,  5.72],   # rec 16
    [ 788.0,   77.1,   68.6,  16.1,   1758.0,   2097.0,  2.74],   # rec 17
    [ 978.0,   44.3,   34.5,   1.1,    511.0,    199.0,  4.96],   # rec 18
    [ 776.0,   33.7,   13.9,   0.2,    350.0,     64.0,  8.63],   # rec 19
    [ 889.0,   52.4,   24.3,   2.7,    847.0,    230.0,  4.76],   # rec 20
    [ 836.0,   64.3,   35.7,  10.4,   1525.0,    558.0,  3.53],   # rec 21
    [ 902.0,   50.9,   20.1,   1.1,    894.0,    127.0, 11.07],   # rec 22
    [ 913.0,   48.6,   20.9,   1.0,    748.0,    138.0,  7.88],   # rec 23
    [1032.0,   87.0,   53.8,  22.8,   3402.0,    911.0,  4.23],   # rec 24
    [ 943.0,  110.9,  152.2,  39.1,   1258.0,   4133.0,  0.39],   # rec 25
    [1031.0,   87.9,   66.8,  32.7,   3356.0,   1702.0,  2.37],   # rec 26
    [ 922.0,   39.5,   23.9,   2.9,    394.0,    184.0,  2.58],   # rec 27
    [1151.0,  145.0,  142.8,  56.6,   8379.0,   6441.0,  1.43],   # rec 28
    [ 990.0,  206.6,  328.2,  71.5,   1412.0,  17240.0,  0.20],   # rec 29
    [ 999.0,   69.6,   53.7,  22.0,   1753.0,   1257.0,  1.99],   # rec 30
    [1001.0,   70.7,   54.0,  22.6,   1793.0,   1174.0,  2.06],   # rec 31
    [ 809.0,   52.8,   29.8,   3.5,   1020.0,    423.0,  3.94],   # rec 32
    [1022.0,   82.6,   53.7,  24.7,   3233.0,    939.0,  4.27],   # rec 33
    [ 880.0,   39.3,   22.0,   2.6,    460.0,    172.0,  3.27],   # rec 34
    [1054.0,   74.1,   46.7,  19.7,   2173.0,    979.0,  2.16],   # rec 35
])

FIELDS = ["avgRR", "sdRR", "RMSSD", "pNN50", "LF", "HF", "LF_HFratio"]


def reference_array() -> np.ndarray:
    """(35, 7) array of reference HRV values, rows = records 0-34."""
    return _REFERENCE.copy()


def reference_dict(record_idx: int) -> dict:
    """Return {field: value} for a single recording (0-indexed)."""
    row = _REFERENCE[record_idx]
    return {f: float(v) for f, v in zip(FIELDS, row)}


def mape_against_reference(pred_hrv: list, fields: list[str] = FIELDS) -> dict:
    """Compute MAPE (%) for each HRV field against the Section-12 reference.

    pred_hrv: list of 35 HRV dataclass instances (must have the 7 attributes).
    Returns: {field: MAPE_pct}.
    """
    out = {}
    ref = _REFERENCE
    if len(pred_hrv) != ref.shape[0]:
        raise ValueError(f"Expected 35 predictions, got {len(pred_hrv)}")
    for j, f in enumerate(fields):
        p = np.array([getattr(h, f) for h in pred_hrv], dtype=np.float64)
        t = ref[:, j]
        mask = np.isfinite(p) & np.isfinite(t) & (t != 0)
        if not mask.any():
            out[f] = float("nan")
        else:
            out[f] = float(np.mean(np.abs((p[mask] - t[mask]) / t[mask])) * 100)
    return out


if __name__ == "__main__":
    print("Reference HRV table (from FurtherInfo Section 12):")
    print(f"{'Rec':>4} {'avgRR':>7} {'sdRR':>7} {'RMSSD':>7} {'pNN50':>7} "
          f"{'LF':>8} {'HF':>8} {'LF/HF':>7}")
    for i, row in enumerate(_REFERENCE, start=1):
        print(f"{i:>4} {row[0]:>7.1f} {row[1]:>7.1f} {row[2]:>7.1f} "
              f"{row[3]:>7.1f} {row[4]:>8.0f} {row[5]:>8.0f} {row[6]:>7.2f}")
