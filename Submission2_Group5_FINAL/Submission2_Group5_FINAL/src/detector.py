"""Causal QRS detector — Pan-Tompkins style, but strictly past-and-present only.

The original MATLAB reference (pan_tompkin.m) uses filtfilt (non-causal). Here we
substitute lfilter everywhere so the detector is valid under BMET9997 rules.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter, find_peaks
from scipy.interpolate import CubicSpline

from load_data import FS


def _bandpass(ecg: np.ndarray, fs: int, lo: float = 5.0, hi: float = 15.0,
              order: int = 3) -> np.ndarray:
    b, a = butter(order, [lo, hi], btype="bandpass", fs=fs)
    return lfilter(b, a, ecg)


def _derivative(x: np.ndarray) -> np.ndarray:
    # 5-tap derivative kernel, causal application.
    b = np.array([1, 2, 0, -2, -1], dtype=np.float64) / 8.0
    return lfilter(b, [1.0], x)


def _moving_window(x: np.ndarray, fs: int, window_ms: float = 150.0) -> np.ndarray:
    n = max(1, int(round(window_ms / 1000 * fs)))
    b = np.ones(n) / n
    return lfilter(b, [1.0], x)


def _tile_stats(mwi: np.ndarray, tile_n: int) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised per-tile (p99, median) over non-overlapping tiles of MWI."""
    n_tiles = len(mwi) // tile_n
    if n_tiles == 0:
        return np.array([]), np.array([])
    tiles = mwi[: n_tiles * tile_n].reshape(n_tiles, tile_n)
    p99 = np.percentile(tiles, 99, axis=1)
    med = np.percentile(tiles, 50, axis=1)
    return p99, med


def _noise_burst_mask(
    med: np.ndarray,
    lookback: int = 10,
    high_factor: float = 4.0,
    low_factor: float = 0.25,
) -> np.ndarray:
    """Per-tile suppression mask: True if this tile's median is anomalously
    high or low vs the rolling baseline of the past `lookback` tile medians.

    Catches two noise-burst failure modes diagnosed on training records 17,
    25, 33: (a) elevated-baseline noise (r25 tile median ~17× clean baseline),
    (b) collapsed-signal noise (r17 tile median ~1/5 clean baseline). Both
    produce many false positives that the existing flat-tile gate misses.

    Strictly causal: baseline at tile i uses only tiles strictly prior to i.
    """
    n = len(med)
    suppress = np.zeros(n, dtype=bool)
    if n < lookback + 1:
        return suppress
    for i in range(lookback, n):
        baseline = float(np.median(med[i - lookback:i]))
        if baseline <= 0:
            continue
        ratio = med[i] / baseline
        if ratio > high_factor or ratio < low_factor:
            suppress[i] = True
    return suppress


def _adaptive_threshold(
    mwi: np.ndarray,
    fs: int,
    tile_s: float = 10.0,
    accept_frac: float = 0.32,
    flat_ratio: float = 5.0,
    noise_burst_suppress: bool = False,
) -> np.ndarray:
    """Accept MWI peaks that exceed `accept_frac * p99(recent)`, but suppress
    detections whenever the recent window looks flat (p99/median below
    `flat_ratio`) — i.e. lead disconnected or silent.

    With `noise_burst_suppress=True`, additionally suppress detections whose
    reference tile has anomalously high or low median vs a rolling baseline
    (catches noise-burst tiles diagnosed on r17, r25, r33).

    Strictly causal: each peak's decision uses stats from an earlier tile only.
    """
    min_dist = int(round(0.2 * fs))
    peaks, _ = find_peaks(mwi, distance=min_dist)
    if len(peaks) == 0:
        return np.array([], dtype=np.int64)

    tile_n = int(round(tile_s * fs))
    p99, med = _tile_stats(mwi, tile_n)
    if p99.size == 0:
        return np.array([], dtype=np.int64)
    med_safe = np.maximum(med, 1.0)

    # Strict causal lookup: use tile strictly before the peak's tile.
    peak_tile = peaks // tile_n
    ref_tile = np.clip(peak_tile - 1, 0, len(p99) - 1)
    # Skip peaks before we have a completed prior tile.
    valid = peak_tile >= 1
    # Suppress where the reference tile looks flat/unplugged.
    ratio = p99[ref_tile] / med_safe[ref_tile]
    active = ratio >= flat_ratio
    # Threshold: fraction of the recent high-amplitude percentile.
    thr = accept_frac * p99[ref_tile]

    mask = valid & active & (mwi[peaks] >= thr)
    if noise_burst_suppress:
        nb_mask = _noise_burst_mask(med)
        in_burst = nb_mask[ref_tile]
        mask = mask & ~in_burst
    return peaks[mask].astype(np.int64)


def _search_back(
    mwi_peaks: np.ndarray,
    mwi: np.ndarray,
    fs: int,
    tile_s: float = 10.0,
    local_window: int = 8,
    gap_factor: float = 1.66,
    thr_factor: float = 0.5,
    accept_frac: float = 0.32,
    flat_ratio: float = 5.0,
    noise_burst_suppress: bool = False,
) -> np.ndarray:
    """Stage-2 refinement: look for missed beats in long RR gaps.

    For each consecutive pair of first-pass detections whose gap exceeds
    `gap_factor * median(last_local_window_RRs)`, re-scan the MWI signal in
    that gap at a lowered threshold. Skip gaps whose reference tile looks
    flat (lead-off) — these are genuine signal dropouts, not missed beats.
    With `noise_burst_suppress=True`, also skip gaps whose reference tile is
    flagged as a noise-burst anomaly.
    Causal: reference tile is strictly before the gap's start.
    """
    if len(mwi_peaks) < local_window + 1:
        return mwi_peaks

    tile_n = int(round(tile_s * fs))
    p99, med = _tile_stats(mwi, tile_n)
    if p99.size == 0:
        return mwi_peaks
    med_safe = np.maximum(med, 1.0)
    min_dist = int(round(0.2 * fs))

    nb_mask = _noise_burst_mask(med) if noise_burst_suppress else None

    additions: list[int] = []
    for i in range(local_window, len(mwi_peaks)):
        recent_rr = np.diff(mwi_peaks[i - local_window : i + 1])
        median_rr = float(np.median(recent_rr))
        gap = int(mwi_peaks[i]) - int(mwi_peaks[i - 1])
        if gap <= gap_factor * median_rr:
            continue

        lo = int(mwi_peaks[i - 1]) + min_dist
        hi = int(mwi_peaks[i]) - min_dist
        if hi - lo < min_dist:
            continue

        ref_tile = max(0, (lo // tile_n) - 1)
        ref_tile = min(ref_tile, len(p99) - 1)
        # Skip gaps in flat/unplugged regions.
        if p99[ref_tile] / med_safe[ref_tile] < flat_ratio:
            continue
        if nb_mask is not None and nb_mask[ref_tile]:
            continue

        thr = thr_factor * accept_frac * p99[ref_tile]
        segment = mwi[lo:hi]
        if segment.size < 3:
            continue
        cands, _ = find_peaks(segment, distance=min_dist, height=thr)
        for c in cands:
            additions.append(lo + int(c))

    if not additions:
        return mwi_peaks

    combined = np.sort(np.unique(np.concatenate([mwi_peaks, np.asarray(additions, dtype=np.int64)])))
    # Re-enforce 200 ms refractory after merging.
    kept = [int(combined[0])]
    for p in combined[1:]:
        if int(p) - kept[-1] >= min_dist:
            kept.append(int(p))
    return np.asarray(kept, dtype=np.int64)


def _local_slope(mwi: np.ndarray, loc: int, window_samples: int) -> float:
    """Mean of the first difference of MWI over the `window_samples` ending at loc."""
    lo = max(0, loc - window_samples)
    if loc - lo < 2:
        return 0.0
    return float(np.mean(np.diff(mwi[lo : loc + 1])))


def _reject_t_waves(
    mwi_peaks: np.ndarray,
    mwi: np.ndarray,
    fs: int,
    t_wave_window_ms: float = 360.0,
    slope_window_ms: float = 75.0,
    slope_ratio: float = 0.5,
) -> np.ndarray:
    """Reject T-wave false positives via slope comparison.

    A candidate peak i that arrives within `t_wave_window_ms` of the previously
    kept R-peak is rejected if its local MWI slope is less than
    `slope_ratio` times the previous peak's slope — the shallow-rise signature
    of a T-wave vs the steep QRS upstroke. Causal: slopes use only samples up
    to each peak.
    """
    if len(mwi_peaks) < 2:
        return mwi_peaks

    t_wave_samples = int(round(t_wave_window_ms / 1000 * fs))
    slope_samples = int(round(slope_window_ms / 1000 * fs))

    kept: list[int] = [int(mwi_peaks[0])]
    slope_prev = _local_slope(mwi, kept[-1], slope_samples)

    for p in mwi_peaks[1:]:
        gap = int(p) - kept[-1]
        if gap >= t_wave_samples:
            kept.append(int(p))
            slope_prev = _local_slope(mwi, int(p), slope_samples)
            continue
        slope_i = _local_slope(mwi, int(p), slope_samples)
        if abs(slope_i) < slope_ratio * abs(slope_prev):
            continue  # T-wave, drop
        kept.append(int(p))
        slope_prev = slope_i
    return np.asarray(kept, dtype=np.int64)


def _refine_to_rpeak(raw: np.ndarray, band: np.ndarray, mwi_peaks: np.ndarray,
                    fs: int) -> np.ndarray:
    """Map each MWI peak to the nearest R-peak in the RAW ECG.

    Two-step refinement so the output is both accurate (vs the expert's raw-
    signal R-peak annotations) and stable (low jitter for HRV):
      1. Coarse: argmax of |bandpass signal| in a window backwards from the
         MWI peak. The bandpass signal is smooth so this localises the peak
         neighbourhood without being perturbed by high-frequency noise, but
         carries filter group delay.
      2. Fine: argmax of |raw ECG mean-subtracted| in a narrow ±fine window
         around the coarse estimate. This removes the filter's group delay
         while keeping the search tight enough to avoid noise spikes.
    """
    coarse_win = int(round(0.15 * fs))
    fine_win = max(1, int(round(0.05 * fs)))  # ±50 ms fine window
    rpeaks = []
    for p in mwi_peaks:
        lo = max(0, p - coarse_win)
        hi = min(len(band), p + 1)
        if hi <= lo:
            continue
        coarse = lo + int(np.argmax(np.abs(band[lo:hi])))
        flo = max(0, coarse - fine_win)
        # Cap the fine-window upper bound at p+1 so we never use any raw
        # sample later than the MWI peak that triggered this detection.
        # The MWI peak already lags the true R-peak by ~75 ms so this rarely
        # binds in practice, but closes the last causality loophole.
        fhi = min(len(raw), p + 1, coarse + fine_win + 1)
        if fhi <= flo:
            continue
        local = raw[flo:fhi].astype(np.float64)
        local = local - np.mean(local)
        rpeaks.append(flo + int(np.argmax(np.abs(local))))
    return np.array(rpeaks, dtype=np.int64)


def precise_rpeak_positions(
    ecg: np.ndarray,
    qrs_int: np.ndarray,
    fs: int = FS,
    window_samples: int = 5,
    upsample_factor: int = 4,
) -> np.ndarray:
    """Refine integer R-peak positions to sub-sample precision via cubic-
    spline upsampling of a local window around each peak.

    For each integer R-peak at sample p, we cubic-spline interpolate the
    raw ECG in [p - window_samples, p + window_samples] (inclusive),
    evaluate on a dense grid (upsample_factor times finer), and take the
    argmax of |y| on that dense grid as the refined sub-sample position.
    The spline smooths through the sample noise that killed a direct
    parabolic fit on raw adjacent samples.

    Literature support: HRV Task Force guidelines recommend parabolic
    interpolation at fs=100-250 Hz; Ellis et al. 2015 and CinC 2020 #088
    specifically validate cubic-spline upsampling to ~250 Hz for R-peak
    refinement from 100 Hz ECG. upsample_factor=4 gives 400 Hz effective.

    The ±window_samples * 10 ms window (default ±50 ms) covers the QRS
    peak region without including the Q and S troughs.
    """
    if qrs_int.size == 0:
        return np.asarray(qrs_int, dtype=np.float64)
    ecg = np.asarray(ecg, dtype=np.float64)
    n = ecg.size
    out = np.asarray(qrs_int, dtype=np.float64).copy()

    for i, p in enumerate(qrs_int):
        p = int(p)
        lo = max(0, p - window_samples)
        hi = min(n - 1, p + window_samples)
        if hi - lo < 3:
            continue
        x = np.arange(lo, hi + 1, dtype=np.float64)
        y = ecg[lo : hi + 1]
        y = y - np.mean(y)
        try:
            spline = CubicSpline(x, y, bc_type="not-a-knot")
        except ValueError:
            continue
        x_fine = np.linspace(
            x[0], x[-1], upsample_factor * (hi - lo) + 1
        )
        y_fine = spline(x_fine)
        out[i] = float(x_fine[int(np.argmax(np.abs(y_fine)))])
    return out


def _enforce_refractory(rpeaks: np.ndarray, fs: int = FS,
                         min_ms: float = 200.0) -> np.ndarray:
    """Drop the later of any two R-peaks closer than min_ms.

    Needed because _refine_to_rpeak can collapse two MWI peaks to the same
    raw R-peak (or adjacent samples), creating sub-200 ms pairs that are
    physiologically impossible and violate the refractory-period invariant.
    """
    if rpeaks.size < 2:
        return rpeaks
    min_dist = int(round(min_ms / 1000.0 * fs))
    kept = [int(rpeaks[0])]
    for p in rpeaks[1:]:
        if int(p) - kept[-1] >= min_dist:
            kept.append(int(p))
    return np.asarray(kept, dtype=rpeaks.dtype)


def _beat_window(
    raw: np.ndarray,
    peak: int,
    fs: int,
    left_ms: float = 90.0,
    right_ms: float = 120.0,
) -> np.ndarray | None:
    """Return a fixed raw-ECG window around one accepted R-peak."""
    left = int(round(left_ms / 1000.0 * fs))
    right = int(round(right_ms / 1000.0 * fs))
    lo = int(peak) - left
    hi = int(peak) + right + 1
    if lo < 0 or hi > raw.size or hi - lo < 3:
        return None
    return np.asarray(raw[lo:hi], dtype=np.float64)


def _normalise_beat(beat: np.ndarray) -> np.ndarray | None:
    """Zero-centre and L2-normalise one beat morphology vector."""
    x = np.asarray(beat, dtype=np.float64)
    x = x - float(np.mean(x))
    norm = float(np.linalg.norm(x))
    if norm <= 1e-12:
        return None
    return x / norm


def _morphology_veto(
    raw: np.ndarray,
    rpeaks: np.ndarray,
    fs: int = FS,
    corr_threshold: float = 0.15,
    min_template_beats: int = 8,
    template_history: int = 12,
) -> np.ndarray:
    """Conservative morphology veto for opt-in false-positive experiments.

    A rolling template is formed from recent accepted beat windows. Once the
    warm-up is complete, candidates with unusually low shape correlation to
    the current template are rejected. Disabled by default at the public
    detector entry point.
    """
    peaks = np.asarray(rpeaks, dtype=np.int64)
    if peaks.size == 0:
        return peaks

    kept: list[int] = []
    template_beats: list[np.ndarray] = []

    for peak in peaks:
        beat = _beat_window(raw, int(peak), fs)
        beat_norm = _normalise_beat(beat) if beat is not None else None

        if beat_norm is None or len(template_beats) < min_template_beats:
            kept.append(int(peak))
            if beat_norm is not None:
                template_beats.append(beat_norm)
                if len(template_beats) > template_history:
                    template_beats.pop(0)
            continue

        template = np.median(np.stack(template_beats, axis=0), axis=0)
        template_norm = _normalise_beat(template)
        corr = 1.0 if template_norm is None else float(np.dot(beat_norm, template_norm))
        if corr < corr_threshold:
            continue

        kept.append(int(peak))
        template_beats.append(beat_norm)
        if len(template_beats) > template_history:
            template_beats.pop(0)

    return np.asarray(kept, dtype=np.int64)


def detect_qrs(
    ecg: np.ndarray,
    fs: int = FS,
    noise_burst_suppress: bool = False,
    morphology_veto: bool = False,
    morphology_corr_threshold: float = 0.15,
) -> np.ndarray:
    """Detect QRS complexes in a single-lead ECG. Returns integer sample
    indices. Sub-sample-precise positions are available via
    precise_rpeak_positions().

    `noise_burst_suppress`: opt-in noise-burst tile gate (sub 2 candidate).
    Suppresses detections whose reference tile has anomalously high or low
    median MWI vs the rolling baseline of the past 10 tiles. Targets the
    failure modes diagnosed on training records 17, 25, 33.
    """
    ecg = np.asarray(ecg, dtype=np.float64)
    band = _bandpass(ecg, fs)
    der = _derivative(band)
    sq = der ** 2
    mwi = _moving_window(sq, fs)
    mwi_peaks = _adaptive_threshold(
        mwi, fs, noise_burst_suppress=noise_burst_suppress
    )
    mwi_peaks = _search_back(
        mwi_peaks, mwi, fs, noise_burst_suppress=noise_burst_suppress
    )
    mwi_peaks = _reject_t_waves(mwi_peaks, mwi, fs)
    rpeaks = _refine_to_rpeak(ecg, band, mwi_peaks, fs)
    rpeaks = _enforce_refractory(rpeaks, fs)
    if morphology_veto:
        rpeaks = _morphology_veto(
            ecg,
            rpeaks,
            fs=fs,
            corr_threshold=morphology_corr_threshold,
        )
    return rpeaks
