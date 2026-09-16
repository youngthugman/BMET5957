"""Local scorer: QRS-detection F1 and HRV MAPE against expert annotations."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from load_data import FS

# Tolerance window for a predicted QRS to count as a match to expert.
# Re-confirmed by Siying (2026-04-20): 50 ms, per Lecture 7 slide 21.
# The 120 ms value in FurtherInfo Section 5 / the preliminary `Tol=12` code
# snippet is an error in the document — ignore it. Authoritative: 50 ms.
DEFAULT_TOL_MS = 50


@dataclass
class Score:
    tp: int
    fp: int
    fn: int

    @property
    def sensitivity(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def ppv(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def f1(self) -> float:
        s, p = self.sensitivity, self.ppv
        return 2 * s * p / (s + p) if (s + p) else 0.0


def score_record(pred: np.ndarray, truth: np.ndarray, tol_samples: int) -> Score:
    """One-to-one greedy match of predicted vs expert QRS within ±tol_samples."""
    pred = np.sort(np.asarray(pred, dtype=np.int64))
    truth = np.sort(np.asarray(truth, dtype=np.int64))
    matched_pred = np.zeros(len(pred), dtype=bool)
    matched_truth = np.zeros(len(truth), dtype=bool)

    j = 0  # pointer into pred
    for i, t in enumerate(truth):
        # advance j past predictions that can no longer match this truth
        while j < len(pred) and pred[j] < t - tol_samples:
            j += 1
        # find the closest unmatched pred within tolerance
        best_k, best_d = -1, tol_samples + 1
        k = j
        while k < len(pred) and pred[k] <= t + tol_samples:
            if not matched_pred[k]:
                d = abs(int(pred[k]) - int(t))
                if d < best_d:
                    best_d, best_k = d, k
            k += 1
        if best_k >= 0:
            matched_pred[best_k] = True
            matched_truth[i] = True

    tp = int(matched_truth.sum())
    fn = int((~matched_truth).sum())
    fp = int((~matched_pred).sum())
    return Score(tp, fp, fn)


def score_all(
    preds: list[np.ndarray],
    truths: list[np.ndarray],
    tol_ms: float = DEFAULT_TOL_MS,
    fs: int = FS,
) -> tuple[list[Score], Score]:
    """Per-record scores plus a micro-averaged aggregate (sum of tp/fp/fn)."""
    tol = int(round(tol_ms / 1000 * fs))
    per = [score_record(p, t, tol) for p, t in zip(preds, truths)]
    total = Score(
        tp=sum(s.tp for s in per),
        fp=sum(s.fp for s in per),
        fn=sum(s.fn for s in per),
    )
    return per, total


def mape(pred: np.ndarray, truth: np.ndarray) -> float:
    """Mean absolute percentage error, ignoring entries where truth==0."""
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    mask = truth != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((pred[mask] - truth[mask]) / truth[mask])) * 100)


def print_report(per: list[Score], total: Score) -> None:
    print(f"{'rec':>4} {'TP':>7} {'FP':>6} {'FN':>6} {'Sens':>7} {'PPV':>7} {'F1':>7}")
    for i, s in enumerate(per):
        print(f"{i:>4} {s.tp:>7} {s.fp:>6} {s.fn:>6} {s.sensitivity:>7.4f} {s.ppv:>7.4f} {s.f1:>7.4f}")
    print("-" * 50)
    print(f"{'ALL':>4} {total.tp:>7} {total.fp:>6} {total.fn:>6} "
          f"{total.sensitivity:>7.4f} {total.ppv:>7.4f} {total.f1:>7.4f}")
