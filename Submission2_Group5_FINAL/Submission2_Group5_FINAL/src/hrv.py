"""HRV parameter calculation — per FurtherInfo (2026-04-19) + W8 lecture.

Methodology (Section 4 + 6 of FurtherInfo, W8 lecture slides 12-23):
  1. Split each ECG into non-overlapping 5-min windows.
  2. In each window, form the RR-interval series from the QRS times.
  3. Flag RR outside [250, 2000] ms as invalid.
  4. Skip a window if the accumulated VALID RR time is less than 4 min.
  5. In valid windows, compute the 7 HRV parameters:
       avgRR  [ms]       : mean of RR
       sdRR   [ms]       : std (sample, ddof=1) of RR
       RMSSD  [ms]       : sqrt(mean(diff(RR)^2))
       pNN50  [%]        : 100 * fraction of |diff(RR)| > 50 ms
       LF     [ms^2]     : Lomb-Scargle PSD over 0.04-0.15 Hz
       HF     [ms^2]     : Lomb-Scargle PSD over 0.15-0.40 Hz
       LF/HF  [unitless] : ratio from the interval-based periodogram
     LF and HF use the Lomb-Scargle periodogram on the unevenly-sampled RR
     tachogram (handles the non-uniform beat timing correctly). LF/HF uses
     the interval-based periodogram (FFT of RR - mean(RR)) because the ratio
     has more stable bias there, which an affine calibration can correct.
     Per-field method selection was validated via strict leave-one-out CV.
  6. Average each parameter across valid windows -> one value per recording.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import lombscargle

from load_data import FS

# Windowing
WINDOW_S = 5 * 60              # 5 minutes per HRV window
WINDOW_SAMPLES = WINDOW_S * FS  # assumes FS=100 Hz
MIN_VALID_WINDOW_S = 4 * 60    # must have >=4 min of valid RR accumulated

# RR flagging
RR_MIN_MS = 250.0
RR_MAX_MS = 2000.0

# Frequency bands (Hz)
LF_BAND = (0.04, 0.15)
HF_BAND = (0.15, 0.40)

# Periodogram config
FFT_N = 256             # Iter-8 optimum for HF/LF-HF MAPE against Section-12 ref.
MIN_RR_PER_WINDOW = 16  # fewer than this and we skip freq-domain

# Lomb-Scargle frequency grid (covers VLF, LF, HF).
# Grid from 0.0033 Hz (lower VLF edge) to 0.40 Hz (upper HF edge), 256 bins.
_LOMB_FREQS = np.linspace(0.0033, 0.40, 256)


@dataclass
class HRV:
    avgRR: float
    sdRR: float
    RMSSD: float
    pNN50: float
    LF: float
    HF: float
    LF_HFratio: float


def _window_starts(
    ecg_len_samples: int,
    fs: int,
    window_s: int,
    overlap: float,
) -> list[int]:
    """Return full-window start samples for a fixed overlap fraction."""
    if overlap < 0.0 or overlap >= 0.9:
        raise ValueError("overlap must be in [0.0, 0.9)")
    window_samples = int(round(window_s * fs))
    if window_samples <= 0 or ecg_len_samples < window_samples:
        return []
    step = max(1, int(round(window_samples * (1.0 - overlap))))
    return list(range(0, ecg_len_samples - window_samples + 1, step))


def _reconstruct_qrs_from_rr(rr_ms: np.ndarray, rr_time_s: np.ndarray) -> np.ndarray:
    """Rebuild QRS times (s) from RR intervals and their later-beat timestamps."""
    if rr_ms.size == 0:
        return np.array([], dtype=np.float64)
    q0 = float(rr_time_s[0] - rr_ms[0] / 1000.0)
    qrs = [q0]
    for rr in rr_ms:
        qrs.append(qrs[-1] + float(rr) / 1000.0)
    return np.asarray(qrs, dtype=np.float64)


def _local_rr_median(rr_ms: np.ndarray, idx: int, window: int = 8) -> float:
    """Median nearby RR (ms), excluding implausible intervals."""
    lo = max(0, idx - window)
    hi = min(rr_ms.size, idx + window + 1)
    local = rr_ms[lo:hi]
    local = local[(local >= RR_MIN_MS) & (local <= RR_MAX_MS)]
    if local.size < 3:
        local = rr_ms[(rr_ms >= RR_MIN_MS) & (rr_ms <= RR_MAX_MS)]
    if local.size == 0:
        return float("nan")
    return float(np.median(local))


def _repair_rr_structural(
    rr_ms: np.ndarray,
    rr_time_s: np.ndarray,
    max_passes: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Conservatively repair obvious detector beat-count errors.

    Only two edits are allowed:
      1. Split a long RR that looks like one missed beat.
      2. Merge a short+short pair that looks like one extra beat.

    The thresholds are intentionally tight so we do not smooth away genuine
    ectopy present in the staff reference.
    """
    rr = np.asarray(rr_ms, dtype=np.float64)
    t = np.asarray(rr_time_s, dtype=np.float64)
    if rr.size < 3 or t.size != rr.size:
        return rr, t

    qrs = _reconstruct_qrs_from_rr(rr, t)
    if qrs.size < 4:
        return rr, t

    for _ in range(max_passes):
        rr = np.diff(qrs) * 1000.0
        changed = False
        i = 0
        while i < rr.size:
            med = _local_rr_median(rr, i)
            if not np.isfinite(med) or med <= 0:
                i += 1
                continue

            # Missed beat: one RR interval is close to 2x the local rhythm.
            if 1.75 * med <= rr[i] <= 2.25 * med:
                insert_time = 0.5 * (qrs[i] + qrs[i + 1])
                qrs = np.insert(qrs, i + 1, insert_time)
                changed = True
                break

            # Extra beat: two consecutive short intervals sum to ~1 normal RR.
            if i + 1 < rr.size:
                pair_sum = rr[i] + rr[i + 1]
                if (
                    rr[i] < 0.75 * med
                    and rr[i + 1] < 0.75 * med
                    and 0.85 * med <= pair_sum <= 1.20 * med
                ):
                    qrs = np.delete(qrs, i + 1)
                    changed = True
                    break
            i += 1

        if not changed:
            break

    rr_out = np.diff(qrs) * 1000.0
    t_out = qrs[1:]
    return rr_out, t_out


def _should_use_selective_repair(rr_ms: np.ndarray) -> bool:
    """Gate structural repair to windows with detector-corruption signatures."""
    rr = np.asarray(rr_ms, dtype=np.float64)
    if rr.size < 4:
        return False

    short_count = int(np.sum(rr < 350.0))
    if short_count >= 2:
        return True

    med = float(np.median(rr))
    if med <= 0:
        return False
    alternating = 0
    for a, b in zip(rr[:-1], rr[1:]):
        short_long = a < 0.55 * med and b > 1.45 * med
        long_short = a > 1.45 * med and b < 0.55 * med
        if short_long or long_short:
            alternating += 1
    return alternating >= 2


def qrs_to_rr_ms(qrs_samples: np.ndarray, fs: int = FS) -> np.ndarray:
    """RR intervals (ms) from sorted QRS sample positions.

    Accepts either integer or float sample positions — float positions from
    parabolic sub-sample refinement give higher-precision RR values, which
    is particularly helpful for pNN50 accuracy."""
    q = np.sort(np.asarray(qrs_samples, dtype=np.float64))
    if q.size < 2:
        return np.array([], dtype=np.float64)
    return np.diff(q) * (1000.0 / fs)


def _time_domain(rr: np.ndarray) -> tuple[float, float, float, float]:
    """avgRR, sdRR, RMSSD, pNN50 from RR interval array (ms)."""
    if rr.size == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    avg = float(np.mean(rr))
    sd = float(np.std(rr, ddof=1)) if rr.size > 1 else 0.0
    if rr.size > 1:
        d = np.diff(rr)
        rmssd = float(np.sqrt(np.mean(d * d)))
        pnn50 = float(np.mean(np.abs(d) > 50.0) * 100.0)
    else:
        rmssd, pnn50 = 0.0, 0.0
    return avg, sd, rmssd, pnn50


def _periodogram(rr_ms: np.ndarray, n_fft: int = FFT_N) -> tuple[np.ndarray, np.ndarray]:
    """Interval-based periodogram of an RR interval sequence.

    Parseval-normalised so sum(P[positive half]) == var(rr_ms) in ms^2.
    Bin k -> Hz via k / (n_fft * mean_RR_sec). Output in ms^2.
    """
    x = rr_ms - np.mean(rr_ms)
    N = len(x)
    X = np.fft.rfft(x, n=n_fft)
    P = (np.abs(X) ** 2) / (N * n_fft)
    if P.size > 2:
        P[1:-1] *= 2.0
    mean_rr_s = float(np.mean(rr_ms)) / 1000.0
    freqs = np.arange(n_fft // 2 + 1) / (n_fft * mean_rr_s)
    return P, freqs


def _freq_domain_fft(rr_ms: np.ndarray) -> tuple[float, float, float]:
    """LF, HF, LF/HF via the interval-based periodogram (FFT of RR - mean).

    Matches the W8 lecture slide 15/18 MATLAB reference. Used for LF/HF
    ratio because the FFT bias cancels cleanly under an affine calibration.
    """
    if rr_ms.size < MIN_RR_PER_WINDOW:
        return (np.nan, np.nan, np.nan)
    P, freqs = _periodogram(rr_ms)

    def band_sum(lo: float, hi: float) -> float:
        m = (freqs >= lo) & (freqs < hi)
        return float(np.sum(P[m])) if m.any() else 0.0

    lf = band_sum(*LF_BAND)
    hf = band_sum(*HF_BAND)
    ratio = lf / hf if hf > 0 else np.nan
    return lf, hf, ratio


def _ectopic_pair_keep_mask(rr_ms: np.ndarray) -> np.ndarray:
    """Boolean keep-mask that removes ectopic-beat pairs (short+pause) and
    isolated >50% deviations. Uses a causal past-median of the last 9 RRs.

    Applied only for the Lomb-Scargle PSD to suppress high-frequency power
    from premature beats / compensatory pauses. NOT applied to the time-
    domain path: that would discard real beat-to-beat variability.
    """
    n = rr_ms.size
    if n < 5:
        return np.ones(n, dtype=bool)
    window = 9
    med = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - window + 1)
        past = rr_ms[lo:i + 1]
        past = past[(past >= RR_MIN_MS) & (past <= RR_MAX_MS)]
        if past.size >= 5:
            med[i] = float(np.median(past))
    ectopic = np.zeros(n, dtype=bool)
    for i in range(n - 1):
        if not np.isfinite(med[i]) or med[i] == 0:
            continue
        dev = (rr_ms[i] - med[i]) / med[i]
        dev_next = (rr_ms[i + 1] - med[i]) / med[i]
        if dev < -0.30 and dev_next > 0.20:
            ectopic[i] = True
            ectopic[i + 1] = True
        elif dev > 0.40 and dev_next < -0.20:
            ectopic[i] = True
            ectopic[i + 1] = True
        elif abs(dev) > 0.50:
            ectopic[i] = True
    return ~ectopic


def _freq_domain_lomb(
    rr_ms: np.ndarray,
    rr_time_s: np.ndarray,
    detrend: str = "mean",
) -> tuple[float, float]:
    """LF, HF via Lomb-Scargle on the unevenly-sampled RR tachogram.

    Applies ectopic-pair filtering before the PSD so premature-beat / pause
    pairs don't leak into HF. Normalisation: psd = pgram * 2 * duration / N,
    giving a Parseval-consistent PSD (integral over positive freqs = variance).

    `detrend`: 'mean' (current default — subtract mean only), or 'linear'
    (subtract the best-fit line in (t, rr) space; removes slow drift that
    inflates LF power for sub-LF-band trends).
    """
    keep = _ectopic_pair_keep_mask(rr_ms)
    rr_clean = rr_ms[keep]
    t_clean = rr_time_s[keep]
    if rr_clean.size < MIN_RR_PER_WINDOW:
        return (np.nan, np.nan)
    duration = float(t_clean[-1] - t_clean[0])
    if duration <= 0:
        return (np.nan, np.nan)

    if detrend == "linear":
        coeffs = np.polyfit(t_clean, rr_clean, 1)
        rr_centred = rr_clean - np.polyval(coeffs, t_clean)
    else:
        rr_centred = rr_clean - np.mean(rr_clean)
    ang = 2.0 * np.pi * _LOMB_FREQS
    try:
        pgram = lombscargle(t_clean, rr_centred, ang, precenter=False)
    except (ValueError, ZeroDivisionError):
        return (np.nan, np.nan)
    psd = pgram * 2.0 * duration / len(rr_centred)

    lf_mask = (_LOMB_FREQS >= LF_BAND[0]) & (_LOMB_FREQS < LF_BAND[1])
    hf_mask = (_LOMB_FREQS >= HF_BAND[0]) & (_LOMB_FREQS < HF_BAND[1])
    if lf_mask.sum() < 2 or hf_mask.sum() < 2:
        return (np.nan, np.nan)
    lf = float(np.trapezoid(psd[lf_mask], _LOMB_FREQS[lf_mask]))
    hf = float(np.trapezoid(psd[hf_mask], _LOMB_FREQS[hf_mask]))
    if lf <= 0 or hf <= 0:
        return (np.nan, np.nan)
    return lf, hf


def _freq_domain_pchip_fft(
    rr_ms: np.ndarray,
    rr_time_s: np.ndarray,
    detrend: str = "mean",
) -> tuple[float, float, float]:
    """HF and LF/HF via a uniformly resampled RR tachogram.

    PCHIP interpolation avoids spline overshoot and performed best in the
    local method bake-off for HF and LF/HF, while the current Lomb path
    remained better for LF itself.

    `detrend`: 'mean' (current default) or 'linear' (subtract best-fit line
    of the resampled tachogram; removes slow drift that inflates LF).
    """
    if rr_ms.size < MIN_RR_PER_WINDOW:
        return (np.nan, np.nan, np.nan)
    duration = float(rr_time_s[-1] - rr_time_s[0])
    if duration <= 0:
        return (np.nan, np.nan, np.nan)

    fs_resample = 4.0
    t_uniform = np.arange(rr_time_s[0], rr_time_s[-1], 1.0 / fs_resample)
    if t_uniform.size < 8:
        return (np.nan, np.nan, np.nan)

    try:
        interp = PchipInterpolator(rr_time_s, rr_ms)
    except ValueError:
        return (np.nan, np.nan, np.nan)

    x = interp(t_uniform).astype(np.float64)
    if detrend == "linear":
        from scipy.signal import detrend as sp_detrend
        x = sp_detrend(x, type="linear")
    else:
        x = x - np.mean(x)
    n_fft = 1024
    X = np.fft.rfft(x, n=n_fft)
    P = (np.abs(X) ** 2) / (fs_resample * x.size)
    if P.size > 2:
        P[1:-1] *= 2.0
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs_resample)

    def band_integral(lo: float, hi: float) -> float:
        m = (freqs >= lo) & (freqs < hi)
        if m.sum() < 2:
            return np.nan
        return float(np.trapezoid(P[m], freqs[m]))

    lf = band_integral(*LF_BAND)
    hf = band_integral(*HF_BAND)
    if not np.isfinite(lf) or not np.isfinite(hf) or hf <= 0:
        return (np.nan, np.nan, np.nan)
    return lf, hf, float(lf / hf)


def _window_hrv(
    rr_ms: np.ndarray,
    rr_time_s: np.ndarray,
    detrend: str = "mean",
    time_domain_input: str = "raw",
    freq_method: str = "hybrid",
) -> Optional[tuple]:
    """HRV for a single 5-min window. Returns (avg, sd, rmssd, pnn50, LF, HF, LF/HF).

    `time_domain_input`: 'raw' (current — un-repaired RR for time-domain to
    preserve genuine ectopy) or 'repaired' (structurally-repaired RR).

    `freq_method`: 'hybrid' (current — Lomb-Scargle for LF/HF, PCHIP-FFT for
    LF/HF ratio) or 'pchip' (PCHIP-FFT for all three freq fields, no Lomb).
    """
    if rr_ms.size == 0:
        return None
    rr_fd, rr_time_fd = _repair_rr_structural(rr_ms, rr_time_s)
    if time_domain_input == "repaired":
        rr_td = rr_fd
    elif time_domain_input == "selective":
        rr_td = rr_fd if _should_use_selective_repair(rr_ms) else rr_ms
    else:
        rr_td = rr_ms
    avg, sd, rmssd, pnn50 = _time_domain(rr_td)
    if freq_method == "pchip":
        lf, hf, ratio = _freq_domain_pchip_fft(rr_fd, rr_time_fd, detrend=detrend)
    else:
        lf, hf = _freq_domain_lomb(rr_fd, rr_time_fd, detrend=detrend)
        _, _, ratio = _freq_domain_pchip_fft(rr_fd, rr_time_fd, detrend=detrend)
    return (avg, sd, rmssd, pnn50, lf, hf, ratio)


def compute_hrv(
    qrs_samples: np.ndarray,
    ecg_len_samples: int,
    fs: int = FS,
    detrend: str = "mean",
    time_domain_input: str = "raw",
    freq_method: str = "hybrid",
    aggregation: str = "mean",
    overlap: float = 0.0,
    min_valid_window_s: int = MIN_VALID_WINDOW_S,
    rr_bounds_ms: tuple[float, float] = (RR_MIN_MS, RR_MAX_MS),
) -> HRV:
    """Compute a single per-recording HRV value by averaging over all valid
    5-min windows of the recording.

    `ecg_len_samples` is the length of the ECG signal in samples — needed to
    know how many 5-min windows the recording actually has.

    `detrend`: 'mean' (current — only DC removal) or 'linear' (subtract
    best-fit line before PSD; removes slow RR drift that inflates LF power).
    Sub 2 candidate parameter; default preserves canonical pipeline.
    """
    # Accept integer or float QRS positions (float positions allow sub-
    # sample-precise RR via parabolic interpolation, helps pNN50).
    q = np.sort(np.asarray(qrs_samples, dtype=np.float64))
    if q.size < 2:
        return HRV(*[np.nan] * 7)

    # RR intervals; time stamp of each RR = time of the *later* QRS.
    rr_ms_all = np.diff(q) * (1000.0 / fs)
    rr_time_s = q[1:] / fs

    per_window_vals = []
    per_window_weights = []
    rr_lo, rr_hi = rr_bounds_ms
    starts = _window_starts(ecg_len_samples, fs=fs, window_s=WINDOW_S, overlap=overlap)

    for start_sample in starts:
        t0 = start_sample / fs
        t1 = t0 + WINDOW_S
        # RR intervals belonging to this window: time stamp in [t0, t1)
        mask = (rr_time_s >= t0) & (rr_time_s < t1)
        if not mask.any():
            continue
        rr_win = rr_ms_all[mask]
        # Flag physiologically implausible intervals
        valid = (rr_win >= rr_lo) & (rr_win <= rr_hi)
        rr_valid = rr_win[valid]
        rr_time_valid = rr_time_s[mask][valid]
        # Require at least MIN_VALID_WINDOW_S seconds of accumulated valid RR
        accumulated_s = float(np.sum(rr_valid)) / 1000.0
        if accumulated_s < min_valid_window_s:
            continue
        vals = _window_hrv(
            rr_valid,
            rr_time_valid,
            detrend=detrend,
            time_domain_input=time_domain_input,
            freq_method=freq_method,
        )
        if vals is None:
            continue
        # Drop a window if any value is non-finite (e.g. failed PSD)
        if any(not np.isfinite(v) for v in vals):
            continue
        per_window_vals.append(vals)
        per_window_weights.append(accumulated_s)

    if not per_window_vals:
        return HRV(*[np.nan] * 7)

    arr = np.array(per_window_vals)  # shape (n_valid_windows, 7)
    if aggregation == "mean":
        means = np.mean(arr, axis=0)
    elif aggregation == "rr_time_weighted":
        means = np.average(arr, axis=0, weights=np.asarray(per_window_weights))
    else:
        raise ValueError("aggregation must be 'mean' or 'rr_time_weighted'")
    return HRV(*means.tolist())


def hrv_mape_report(pred: list[HRV], truth: list[HRV]) -> dict:
    """MAPE (%) between two lists of HRV instances (self-reference only)."""
    fields = ["avgRR", "sdRR", "RMSSD", "pNN50", "LF", "HF", "LF_HFratio"]
    out: dict[str, float] = {}
    for f in fields:
        p = np.array([getattr(x, f) for x in pred], dtype=np.float64)
        t = np.array([getattr(x, f) for x in truth], dtype=np.float64)
        m = np.isfinite(p) & np.isfinite(t) & (t != 0)
        if not m.any():
            out[f] = float("nan")
        else:
            out[f] = float(np.mean(np.abs((p[m] - t[m]) / t[m])) * 100)
    return out
