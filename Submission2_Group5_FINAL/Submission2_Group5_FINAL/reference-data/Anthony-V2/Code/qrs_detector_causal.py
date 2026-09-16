"""
BMET9997 Major Project - Causal Pan-Tompkins QRS Detector
==========================================================
Python port of pan_tompkin.m (Sedghamiz 2018) adapted for the BMET9997
real-time constraint: all filtering uses scipy.signal.lfilter (causal
forward-only) instead of MATLAB's filtfilt (zero-phase, non-causal).

The adaptive thresholding, T-wave rejection via slope comparison, and
searchback logic follow the reference implementation faithfully -
these are the parts that drive accuracy. Only the filtering has been
made causal.

Wrapped with polarity correction, chunked processing, and a
post-detection refinement stage to handle records where the base
Pan-Tompkins fails (see project brief section 11).

Reference:
  Sedghamiz H., "Matlab Implementation of Pan Tompkins ECG QRS
  detector", 2014, BioSigKit Toolbox.
  Pan J., Tompkins W.J., "A Real-Time QRS Detection Algorithm",
  IEEE Trans. Biomed. Eng., 1985.
"""

import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi, find_peaks

FS = 100   # default sampling rate for this project


def _detect_qrs_core(ecg, fs=FS):
    """
    Causal Pan-Tompkins QRS detector (core algorithm, single chunk).

    ecg: 1D ECG signal (uV) sampled at fs Hz
    Returns: numpy int array of QRS sample indices (0-based).
    """
    ecg = np.asarray(ecg, dtype=float).ravel()

    if len(ecg) < 3 * fs:   # less than 3 seconds - can't initialise thresholds
        return np.array([], dtype=int)

    # ================== Bandpass filter 5-15 Hz (causal) ==================
    low, high = 5.0, 15.0
    nyq = fs / 2.0
    b, a = butter(3, [low / nyq, high / nyq], btype='band')
    zi = lfilter_zi(b, a) * ecg[0]
    ecg_h, _ = lfilter(b, a, ecg, zi=zi)

    max_h = np.max(np.abs(ecg_h))
    if max_h > 0:
        ecg_h = ecg_h / max_h

    # ================== Derivative filter ==================
    d_kernel = np.array([1, 2, 0, -2, -1]) * (1 / 8) * fs
    ecg_d = lfilter(d_kernel, 1, ecg_h)
    max_d = np.max(np.abs(ecg_d))
    if max_d > 0:
        ecg_d = ecg_d / max_d

    # ================== Squaring ==================
    ecg_s = ecg_d ** 2

    # ================== Moving-window integration (causal trailing) ==================
    win_size = int(round(0.150 * fs))
    kernel = np.ones(win_size) / win_size
    ecg_m = np.convolve(ecg_s, kernel, mode='full')[:len(ecg_s)]

    # ================== Find candidate peaks in integrated signal ==================
    min_dist = int(round(0.2 * fs))
    locs, props = find_peaks(ecg_m, distance=min_dist)
    pks = ecg_m[locs]
    LLp = len(pks)

    if LLp == 0:
        return np.array([], dtype=int)

    # ================== Initialise thresholds (2-second training) ==================
    train_end = min(2 * fs, len(ecg_m))
    THR_SIG = np.max(ecg_m[:train_end]) / 3.0
    THR_NOISE = np.mean(ecg_m[:train_end]) / 2.0
    SIG_LEV = THR_SIG
    NOISE_LEV = THR_NOISE

    THR_SIG1 = np.max(np.abs(ecg_h[:train_end])) / 3.0
    THR_NOISE1 = np.mean(np.abs(ecg_h[:train_end])) / 2.0
    SIG_LEV1 = THR_SIG1
    NOISE_LEV1 = THR_NOISE1

    # ================== Main detection loop ==================
    qrs_i = []
    qrs_i_raw = []
    m_selected_RR = 0
    mean_RR = 0
    skip = 0
    ser_back = 0

    for i in range(LLp):
        current_loc = locs[i]
        current_pk = pks[i]

        # ====== Locate corresponding peak in filtered signal ======
        search_win = int(round(0.150 * fs))
        lo = current_loc - search_win
        hi = current_loc + 1

        if lo >= 0 and current_loc < len(ecg_h):
            seg = ecg_h[lo:hi]
            x_i_rel = int(np.argmax(np.abs(seg)))
            y_i = np.abs(seg[x_i_rel])
            x_i = lo + x_i_rel
            ser_back = 0
        elif i == 0:
            seg = ecg_h[:current_loc + 1]
            if len(seg) == 0:
                continue
            x_i_rel = int(np.argmax(np.abs(seg)))
            y_i = np.abs(seg[x_i_rel])
            x_i = x_i_rel
            ser_back = 1
        elif current_loc >= len(ecg_h):
            seg = ecg_h[max(0, lo):]
            if len(seg) == 0:
                continue
            x_i_rel = int(np.argmax(np.abs(seg)))
            y_i = np.abs(seg[x_i_rel])
            x_i = max(0, lo) + x_i_rel
            ser_back = 0
        else:
            continue

        # ====== Update heart rate using last 8 beats ======
        if len(qrs_i) >= 9:
            diffRR = np.diff(qrs_i[-9:])
            mean_RR = np.mean(diffRR)
            comp = qrs_i[-1] - qrs_i[-2]

            if comp <= 0.92 * mean_RR or comp >= 1.16 * mean_RR:
                THR_SIG = 0.5 * THR_SIG
                THR_SIG1 = 0.5 * THR_SIG1
            else:
                m_selected_RR = mean_RR

        # ====== Searchback for missed beats ======
        if m_selected_RR:
            test_m = m_selected_RR
        elif mean_RR and m_selected_RR == 0:
            test_m = mean_RR
        else:
            test_m = 0

        if test_m and len(qrs_i) > 0:
            if (current_loc - qrs_i[-1]) >= round(1.66 * test_m):
                sb_lo = qrs_i[-1] + round(0.200 * fs)
                sb_hi = current_loc - round(0.200 * fs)
                if sb_hi > sb_lo and sb_hi <= len(ecg_m):
                    sb_seg = ecg_m[sb_lo:sb_hi]
                    if len(sb_seg) > 0:
                        pks_temp = np.max(sb_seg)
                        locs_temp_rel = int(np.argmax(sb_seg))
                        locs_temp = sb_lo + locs_temp_rel

                        if pks_temp > THR_NOISE:
                            qrs_i.append(locs_temp)

                            f_lo = locs_temp - round(0.150 * fs)
                            f_hi = min(locs_temp + 1, len(ecg_h))
                            if f_lo >= 0:
                                f_seg = ecg_h[f_lo:f_hi]
                                if len(f_seg) > 0:
                                    y_i_t = np.max(np.abs(f_seg))
                                    x_i_t_rel = int(np.argmax(np.abs(f_seg)))
                                    if y_i_t > THR_NOISE1:
                                        qrs_i_raw.append(
                                            f_lo + x_i_t_rel)
                                        SIG_LEV1 = (0.25 * y_i_t
                                                    + 0.75 * SIG_LEV1)
                            SIG_LEV = 0.25 * pks_temp + 0.75 * SIG_LEV

        # ====== Find noise and QRS peaks (main rule) ======
        if current_pk >= THR_SIG:
            skip = 0
            if len(qrs_i) >= 3:
                if (current_loc - qrs_i[-1]) <= round(0.4500 * fs):
                    slope_win = round(0.075 * fs)
                    if (current_loc - slope_win >= 0
                            and qrs_i[-1] - slope_win >= 0):
                        s1_seg = ecg_m[current_loc - slope_win:current_loc]
                        s2_seg = ecg_m[qrs_i[-1] - slope_win:qrs_i[-1]]
                        if len(s1_seg) > 1 and len(s2_seg) > 1:
                            Slope1 = np.mean(np.diff(s1_seg))
                            Slope2 = np.mean(np.diff(s2_seg))
                            if abs(Slope1) <= abs(0.5 * Slope2):
                                skip = 1
                                NOISE_LEV1 = (0.125 * y_i
                                              + 0.875 * NOISE_LEV1)
                                NOISE_LEV = (0.125 * current_pk
                                             + 0.875 * NOISE_LEV)

            if skip == 0:
                qrs_i.append(current_loc)
                if y_i >= THR_SIG1:
                    if ser_back:
                        qrs_i_raw.append(x_i)
                    else:
                        qrs_i_raw.append(x_i)
                    SIG_LEV1 = 0.125 * y_i + 0.875 * SIG_LEV1
                SIG_LEV = 0.125 * current_pk + 0.875 * SIG_LEV

        elif THR_NOISE <= current_pk < THR_SIG:
            NOISE_LEV1 = 0.125 * y_i + 0.875 * NOISE_LEV1
            NOISE_LEV = 0.125 * current_pk + 0.875 * NOISE_LEV

        elif current_pk < THR_NOISE:
            NOISE_LEV1 = 0.125 * y_i + 0.875 * NOISE_LEV1
            NOISE_LEV = 0.125 * current_pk + 0.875 * NOISE_LEV

        # ====== Adjust thresholds ======
        if NOISE_LEV != 0 or SIG_LEV != 0:
            THR_SIG = NOISE_LEV + 0.25 * abs(SIG_LEV - NOISE_LEV)
            THR_NOISE = 0.5 * THR_SIG

        if NOISE_LEV1 != 0 or SIG_LEV1 != 0:
            THR_SIG1 = NOISE_LEV1 + 0.25 * abs(SIG_LEV1 - NOISE_LEV1)
            THR_NOISE1 = 0.5 * THR_SIG1

        skip = 0
        ser_back = 0

    qrs_out = np.array(sorted(set(qrs_i_raw)), dtype=int)

    # ================== Compensate for causal filter group delay ==================
    GROUP_DELAY = int(round(0.058 * fs))   # ~58 ms empirically measured
    qrs_out = qrs_out - GROUP_DELAY
    qrs_out = qrs_out[qrs_out >= 0]

    return qrs_out


# ---------------------------------------------------------------------------
# Robust wrapper: chunking, polarity correction, post-detection refinement
# ---------------------------------------------------------------------------
def _polarity_score(ecg, fs=FS):
    """
    Return (best_signal, was_flipped). Decides whether to flip polarity by
    comparing the sharpness of upward vs downward deflections. R peaks are
    typically the sharpest feature in the ECG.
    """
    n = min(60 * fs, len(ecg))
    seg = ecg[:n] - np.mean(ecg[:n])
    top_val = np.percentile(seg, 99.5)
    bot_val = np.percentile(seg, 0.5)
    if abs(bot_val) > abs(top_val) * 1.3:
        return -ecg, True
    return ecg, False


def _post_detection_refinement(qrs, ecg, fs=FS):
    """
    Second-stage algorithm (required by the brief):
    - Merge duplicate detections within the refractory period
    - Search for missed beats in gaps > 1.5 * local median RR
    """
    if len(qrs) < 3:
        return qrs

    qrs = np.sort(np.unique(qrs))

    # Remove detections within refractory period of previous.
    # Refractory is adaptive: max(300 ms, min(45 % of running RR, 500 ms)).
    # This extends blanking for slow-heart-rate records where T-waves appear
    # at 350-450 ms post-R (outside a fixed 300 ms window).
    # Within the window, keep the higher-amplitude peak (original Pan-Tompkins
    # behaviour) — this corrects cases where the algorithm fires slightly early.
    rr_est = int(0.80 * fs)   # running RR estimate, initialised at 75 bpm (800 ms)
    cleaned = [qrs[0]]
    for q in qrs[1:]:
        refractory = max(int(0.30 * fs),
                         min(int(0.45 * rr_est), int(0.50 * fs)))
        gap = q - cleaned[-1]
        if gap >= refractory:
            rr_est = int(0.875 * rr_est + 0.125 * gap)
            cleaned.append(q)
        elif abs(ecg[q]) > abs(ecg[cleaned[-1]]):
            cleaned[-1] = q   # keep the higher-amplitude candidate
    qrs = np.array(cleaned, dtype=int)

    # Searchback for missed beats in large gaps
    rr = np.diff(qrs)
    new_qrs = [qrs[0]]
    for i in range(1, len(qrs)):
        gap = qrs[i] - new_qrs[-1]
        past = rr[max(0, i - 8):i]
        past = past[(past >= 0.25 * fs) & (past <= 2.0 * fs)]
        if len(past) >= 3:
            med_rr = np.median(past)
            if gap > 1.5 * med_rr and gap > 0.8 * fs:
                n_missed = int(round(gap / med_rr)) - 1
                lo = new_qrs[-1] + int(0.2 * fs)
                hi = qrs[i] - int(0.2 * fs)
                if hi > lo:
                    seg = np.abs(ecg[lo:hi]).astype(float)
                    if len(seg) > 0 and n_missed > 0:
                        thresh = np.median(np.abs(
                            ecg[max(0, new_qrs[-1] - int(5 * fs)):new_qrs[-1]]
                        )) * 3
                        for _ in range(n_missed):
                            if len(seg) == 0:
                                break
                            pk = int(np.argmax(seg))
                            if seg[pk] < thresh:
                                break
                            new_qrs.append(lo + pk)
                            b_lo = max(0, pk - int(0.2 * fs))
                            b_hi = min(len(seg), pk + int(0.2 * fs))
                            seg[b_lo:b_hi] = 0
        new_qrs.append(qrs[i])

    return np.array(sorted(set(new_qrs)), dtype=int)


def _secondary_pass(qrs, ecg, fs=FS, min_gap_sec=30):
    """
    For any gap > min_gap_sec with no primary detections, retry
    _detect_qrs_core on that segment — first with the supplied polarity,
    then with polarity flipped (catches axis-shift segments missed by the
    one-shot polarity correction at the recording start).

    Only keeps secondary detections if the implied heart rate is
    physiologically plausible (20–200 bpm), to avoid inserting beats
    into genuine artifact or pause regions.
    """
    if len(qrs) < 2:
        return qrs

    min_gap = int(min_gap_sec * fs)
    extra = []

    boundaries = np.concatenate([[0], np.sort(qrs), [len(ecg)]])
    for g in range(len(boundaries) - 1):
        gap_lo = int(boundaries[g])
        gap_hi = int(boundaries[g + 1])
        if gap_hi - gap_lo <= min_gap:
            continue

        seg_lo = gap_lo + int(0.5 * fs)
        seg_hi = gap_hi - int(0.5 * fs)
        if seg_hi - seg_lo < 3 * fs:
            continue

        seg = ecg[seg_lo:seg_hi]
        duration_sec = (seg_hi - seg_lo) / fs

        found = None
        for candidate in [seg, -seg]:
            new_q = _detect_qrs_core(candidate, fs)
            new_q = new_q[(new_q >= 0) & (new_q < len(seg))]
            if len(new_q) < 2:
                continue
            rate_bpm = len(new_q) / duration_sec * 60
            if not (20 <= rate_bpm <= 200):
                continue
            found = new_q + seg_lo
            break

        if found is not None:
            extra.extend(found.tolist())

    if not extra:
        return qrs
    return np.array(sorted(set(list(qrs) + extra)), dtype=int)


def _low_amplitude_filter(qrs, ecg, fs, window_sec=60, lo_factor=0.3,
                          max_rate_bpm=110):
    """
    Remove detections from 60-second windows where BOTH conditions hold:
      1. ECG amplitude is below lo_factor × recording median (lead-off / flat)
      2. Detection rate in that window exceeds max_rate_bpm

    Condition 2 is critical: genuine low-amplitude QRS (e.g. pericardial
    effusion, distant electrode) has a normal physiological rate (~40-100 bpm).
    Lead-off noise triggers dense, anomalously fast detections (114-130+ bpm).
    Requiring both conditions avoids suppressing valid but quiet QRS complexes.
    """
    if len(qrs) == 0:
        return qrs
    window = int(window_sec * fs)
    n_win  = int(np.ceil(len(ecg) / window))
    rms = np.empty(n_win)
    for w in range(n_win):
        seg    = ecg[w * window : min((w + 1) * window, len(ecg))]
        rms[w] = np.sqrt(np.mean(seg ** 2)) if len(seg) > 0 else 0.0
    valid_rms = rms[rms > 0]
    if len(valid_rms) == 0:
        return qrs
    med = np.median(valid_rms)
    if med == 0:
        return qrs
    max_beats = int(max_rate_bpm * window_sec / 60)
    keep = np.ones(len(qrs), dtype=bool)
    for w in range(n_win):
        if rms[w] >= lo_factor * med:
            continue   # normal amplitude — keep unconditionally
        lo = w * window
        hi = min((w + 1) * window, len(ecg))
        count = int(((qrs >= lo) & (qrs < hi)).sum())
        if count > max_beats:   # low amplitude AND anomalously high rate
            keep[(qrs >= lo) & (qrs < hi)] = False
    return qrs[keep]


def _high_amplitude_rate_filter(qrs, ecg, fs, window_sec=60,
                                 rms_factor=5.0, rate_factor=1.4):
    """
    Remove detections from windows where signal amplitude is very high
    (rms > 5× median) AND the detection rate is elevated (rate > 1.4× typical).
    Catches obvious artifact bursts that happen to fire faster than normal.
    """
    if len(qrs) < 2:
        return qrs
    win = int(window_sec * fs)
    n_win = int(np.ceil(len(ecg) / win))
    rms = np.zeros(n_win)
    det_count = np.zeros(n_win, dtype=int)
    for w in range(n_win):
        lo, hi = w * win, min((w + 1) * win, len(ecg))
        seg = ecg[lo:hi]
        rms[w] = np.sqrt(np.mean(seg**2)) if len(seg) > 0 else 0.0
        det_count[w] = int(((qrs >= lo) & (qrs < hi)).sum())
    valid_rms = rms[rms > 0]
    if len(valid_rms) == 0:
        return qrs
    med_rms = np.median(valid_rms)
    normal_mask = (rms >= 0.3 * med_rms) & (rms <= 2.0 * med_rms)
    normal_counts = det_count[normal_mask & (det_count > 0)]
    if len(normal_counts) < 5:
        return qrs
    med_count = float(np.median(normal_counts))
    if med_count == 0:
        return qrs
    keep = np.ones(len(qrs), dtype=bool)
    for w in range(n_win):
        if rms[w] < rms_factor * med_rms:
            continue
        lo, hi = w * win, min((w + 1) * win, len(ecg))
        rate_ratio = det_count[w] / med_count
        if rate_ratio > rate_factor:
            keep[(qrs >= lo) & (qrs < hi)] = False
    return qrs[keep]


def _twave_suppression(qrs, ecg, fs, twave_min_ms=250, twave_max_ms=450,
                       amp_ratio=0.75):
    """
    Remove T-wave detections that slipped through the core algorithm's slope
    rejection. A candidate is a T-wave if:
      1. Its gap from the preceding kept detection is in [twave_min_ms, twave_max_ms]
         (the physiological QT-interval zone at normal heart rates).
      2. Its amplitude in the polarity-corrected signal is < amp_ratio × the
         preceding kept detection's amplitude.
    Real rapid beats (PACs, AFib) retain near-normal amplitude and are not removed.
    """
    if len(qrs) < 2:
        return qrs
    min_gap = int(twave_min_ms * fs / 1000)
    max_gap = int(twave_max_ms * fs / 1000)
    keep = np.ones(len(qrs), dtype=bool)
    last_kept = 0
    for i in range(1, len(qrs)):
        gap = int(qrs[i]) - int(qrs[last_kept])
        if min_gap <= gap <= max_gap:
            ref_amp = abs(float(ecg[qrs[last_kept]]))
            cand_amp = abs(float(ecg[qrs[i]]))
            if ref_amp > 0 and cand_amp < amp_ratio * ref_amp:
                keep[i] = False
                continue
        last_kept = i
    return qrs[keep]


def _rate_burst_filter(qrs, ecg_len, fs, max_bpm=250, window_sec=10):
    """
    Remove detections from windows where the implied rate exceeds max_bpm.
    Catches brief noise bursts where the secondary pass or searchback fires
    densely (e.g. >250 bpm in any 10-second window = unambiguous artefact).
    """
    if len(qrs) < 2:
        return qrs
    window  = int(window_sec * fs)
    max_per = int(max_bpm / 60.0 * window_sec)   # max allowed beats per window
    step    = window // 2                          # 50 % overlap
    to_remove = np.zeros(len(qrs), dtype=bool)
    for start in range(0, ecg_len, step):
        end  = min(start + window, ecg_len)
        mask = (qrs >= start) & (qrs < end)
        if mask.sum() > max_per:
            to_remove |= mask
    return qrs[~to_remove]


def _low_density_retry(qrs, ecg, fs, window_sec=60, min_density_ratio=0.20):
    """
    After all primary detection stages, find 60-second windows that are at
    normal signal amplitude but have far fewer detections than the recording's
    typical rate (< min_density_ratio × median).  These windows indicate
    threshold miscalibration — usually caused by adapting to a preceding
    high-amplitude artifact period (classic Pan-Tompkins failure mode seen
    in Record 17 minutes 386-387).

    Re-runs _detect_qrs_core on each flagged window with FRESH threshold
    initialisation from that window's own first two seconds (which is normal
    ECG, not artifact), recovering beats that the miscalibrated primary pass
    missed.

    Only fires in strictly normal-amplitude windows (0.3-3 × median RMS) to
    avoid false recovery in genuine artifact or lead-off segments.
    """
    if len(qrs) < 10:
        return qrs

    win = int(window_sec * fs)
    n_win = int(np.ceil(len(ecg) / win))

    rms = np.zeros(n_win)
    det_count = np.zeros(n_win, dtype=int)
    for w in range(n_win):
        lo, hi = w * win, min((w + 1) * win, len(ecg))
        seg = ecg[lo:hi]
        rms[w] = np.sqrt(np.mean(seg ** 2)) if len(seg) > 0 else 0.0
        det_count[w] = int(((qrs >= lo) & (qrs < hi)).sum())

    valid_rms = rms[rms > 0]
    if len(valid_rms) == 0:
        return qrs
    med_rms = np.median(valid_rms)

    # Reference: median per-window detection count from normal-amplitude windows
    normal_mask = (rms >= 0.3 * med_rms) & (rms <= 3.0 * med_rms)
    normal_det = det_count[normal_mask & (det_count > 0)]
    if len(normal_det) < 5:
        return qrs
    med_det = float(np.median(normal_det))

    extra = []
    for w in range(n_win):
        if not normal_mask[w]:
            continue
        if det_count[w] >= min_density_ratio * med_det:
            continue   # detection density is acceptable

        lo, hi = w * win, min((w + 1) * win, len(ecg))
        seg = ecg[lo:hi]
        if len(seg) < 3 * fs:
            continue

        # Retry with the polarity-corrected segment; also try flipped in case
        # there is a localised axis change within the recording.
        for cand in [seg, -seg]:
            new_q = _detect_qrs_core(cand, fs)
            new_q = new_q[(new_q >= 0) & (new_q < len(seg))]
            if len(new_q) < 2:
                continue
            rate_bpm = len(new_q) / (len(seg) / fs) * 60
            if 20 <= rate_bpm <= 200:
                extra.extend((new_q + lo).tolist())
                break

    if not extra:
        return qrs
    return np.array(sorted(set(list(qrs) + extra)), dtype=int)


def _kurtosis_artifact_filter(qrs, ecg, fs, window_sec=60,
                               hi_amp_factor=3.0, min_kurtosis=3.0):
    """
    Suppress detections in high-amplitude windows that look like noise
    rather than real ECG. Real ECG bandpass signal has high Fisher excess
    kurtosis (sparse QRS spikes on a near-zero baseline). EMG/artifact
    bursts are Gaussian-like with kurtosis near 0.
    Only acts on windows where RMS > hi_amp_factor * median(RMS).
    """
    if len(qrs) == 0:
        return qrs

    # Bandpass 5-15 Hz (same as main detector)
    nyq = fs / 2.0
    b, a = butter(2, [5.0 / nyq, 15.0 / nyq], btype='band')
    try:
        bp = lfilter(b, a, ecg)
    except Exception:
        return qrs

    win = int(window_sec * fs)
    n_win = int(np.ceil(len(ecg) / win))
    rms = np.zeros(n_win)
    for w in range(n_win):
        lo, hi = w * win, min((w + 1) * win, len(ecg))
        seg = ecg[lo:hi]
        rms[w] = np.sqrt(np.mean(seg ** 2)) if len(seg) > 0 else 0.0

    valid_rms = rms[rms > 0]
    if len(valid_rms) == 0:
        return qrs
    med_rms = np.median(valid_rms)
    if med_rms == 0:
        return qrs

    keep = np.ones(len(qrs), dtype=bool)
    for w in range(n_win):
        if rms[w] < hi_amp_factor * med_rms:
            continue  # only inspect high-amplitude windows
        lo, hi = w * win, min((w + 1) * win, len(ecg))
        seg = bp[lo:hi]
        if len(seg) < int(2 * fs):
            continue
        m = np.mean(seg)
        s = np.std(seg)
        if s == 0:
            continue
        kurtosis = float(np.mean(((seg - m) / s) ** 4)) - 3.0  # Fisher excess
        if kurtosis < min_kurtosis:
            keep[(qrs >= lo) & (qrs < hi)] = False

    return qrs[keep]


def _rr_irregularity_filter(qrs, ecg, fs, window_sec=60,
                             hi_amp_factor=3.0, max_cv=0.70,
                             min_beats=6):
    """
    Suppress detections in elevated-amplitude windows where the detected
    inter-beat intervals are highly irregular (coefficient of variation > max_cv).
    Random artifact produces RR CV near 1.0; real ECG including AFib rarely
    exceeds 0.5. Only applied to windows where RMS > hi_amp_factor * median(RMS)
    to avoid suppressing legitimate irregular rhythms at normal amplitude.
    """
    if len(qrs) < min_beats:
        return qrs
    win = int(window_sec * fs)
    n_win = int(np.ceil(len(ecg) / win))
    rms = np.zeros(n_win)
    for w in range(n_win):
        lo, hi = w * win, min((w + 1) * win, len(ecg))
        seg = ecg[lo:hi]
        rms[w] = np.sqrt(np.mean(seg**2)) if len(seg) > 0 else 0.0
    valid_rms = rms[rms > 0]
    if len(valid_rms) == 0:
        return qrs
    med_rms = np.median(valid_rms)
    if med_rms == 0:
        return qrs
    keep = np.ones(len(qrs), dtype=bool)
    for w in range(n_win):
        if rms[w] < hi_amp_factor * med_rms:
            continue
        lo, hi = w * win, min((w + 1) * win, len(ecg))
        idx = np.where((qrs >= lo) & (qrs < hi))[0]
        if len(idx) < min_beats:
            continue
        rr = np.diff(qrs[idx]).astype(float)
        if len(rr) < 2 or np.mean(rr) == 0:
            continue
        cv = np.std(rr) / np.mean(rr)
        if cv > max_cv:
            keep[idx] = False
    return qrs[keep]


def detect_qrs_causal(ecg, fs=FS, chunk_minutes=15):
    """
    Robust causal QRS detector.

    Processes the ECG in chunks so threshold estimates are refreshed
    periodically - this prevents total failure when one segment has
    abnormal amplitude or noise (a known Pan-Tompkins failure mode,
    see project brief section 11 on record 31).

    Adds polarity correction (flips the signal if R peaks are inverted)
    and a post-detection refinement stage (merges duplicates, searches
    for missed beats in large gaps).

    ecg: 1D ECG signal (uV)
    fs: sampling rate (Hz)
    chunk_minutes: chunk length in minutes (default 15)
    Returns: numpy int array of QRS sample indices.
    """
    ecg = np.asarray(ecg, dtype=float).ravel()

    # Polarity correction - happens once at the start
    ecg_corr, flipped = _polarity_score(ecg, fs)

    # Chunked detection with fresh threshold initialisation per chunk
    chunk_samples = chunk_minutes * 60 * fs
    all_qrs = []

    for start in range(0, len(ecg_corr), chunk_samples):
        end = min(start + chunk_samples, len(ecg_corr))
        overlap = 10 * fs   # 10 s warm-up for threshold initialisation
        lo = max(0, start - overlap)
        chunk = ecg_corr[lo:end]
        qrs_chunk = _detect_qrs_core(chunk, fs)
        qrs_chunk = qrs_chunk + lo
        qrs_chunk = qrs_chunk[(qrs_chunk >= start) & (qrs_chunk < end)]
        all_qrs.append(qrs_chunk)

    if len(all_qrs) == 0:
        return np.array([], dtype=int)
    qrs = np.concatenate(all_qrs)

    # Post-detection refinement on the polarity-corrected signal
    qrs = _post_detection_refinement(qrs, ecg_corr, fs)

    # R-peak refinement: snap each detection to the true R-peak (positive
    # maximum in polarity-corrected signal) within ±80 ms. Wider than the
    # original ±60 ms to handle below-normal-amplitude windows where the
    # causal bandpass group delay produces timing errors up to ~70 ms.
    search_r = int(round(0.060 * fs))
    refined = np.empty(len(qrs), dtype=int)
    for i, q in enumerate(qrs):
        lo = max(0, q - search_r)
        hi = min(len(ecg_corr), q + search_r + 1)
        refined[i] = lo + int(np.argmax(ecg_corr[lo:hi]))
    qrs = np.unique(refined)

    # Hard refractory after R-peak snapping: two detections originally 360 ms
    # apart (the adaptive refractory minimum) can snap within ~240 ms of each
    # other. np.unique only catches zero-distance duplicates, so we enforce a
    # 200 ms hard floor and keep the higher-amplitude candidate.
    if len(qrs) > 1:
        refractory_hard = int(0.20 * fs)
        cleaned = [int(qrs[0])]
        for q in qrs[1:]:
            if q - cleaned[-1] >= refractory_hard:
                cleaned.append(int(q))
            elif abs(ecg_corr[q]) > abs(ecg_corr[cleaned[-1]]):
                cleaned[-1] = int(q)
        qrs = np.array(cleaned, dtype=int)

    # Secondary pass: retry detection in large silent gaps (handles recording
    # segments where primary detection produces nothing due to morphology shift
    # or polarity inversion — the known failure mode on records like 31).
    qrs = _secondary_pass(qrs, ecg_corr, fs, min_gap_sec=30)

    # Low-density retry: re-detect in normal-amplitude windows where the
    # primary pass produced far fewer beats than the recording's typical rate.
    # Corrects threshold miscalibration after high-amplitude artifact periods.
    qrs = _low_density_retry(qrs, ecg_corr, fs)

    # Suppress detections in high-amplitude noise bursts (low kurtosis).
    qrs = _kurtosis_artifact_filter(qrs, ecg_corr, fs)

    # Suppress detections in elevated-amplitude windows with highly irregular
    # inter-beat intervals — random artifact produces CV near 1.0 while real
    # ECG (including AFib) stays below 0.5.
    qrs = _rr_irregularity_filter(qrs, ecg_corr, fs)

    # Suppress detections in windows with very high amplitude (>5× median) AND
    # elevated detection rate (>1.4× typical) — catches artifact bursts that
    # mimic QRS morphology and slip past kurtosis/CV filters.
    qrs = _high_amplitude_rate_filter(qrs, ecg_corr, fs)

    # Safety net: remove any window still exceeding 250 bpm after all stages.
    qrs = _rate_burst_filter(qrs, len(ecg_corr), fs)

    # Suppress detections in windows where ECG amplitude is well below the
    # recording median — these indicate lead-off or disconnected electrode.
    qrs = _low_amplitude_filter(qrs, ecg_corr, fs)

    # Remove T-wave detections that follow a QRS within the QT-interval zone
    # (250-450 ms) with much lower amplitude than the preceding R-peak.
    qrs = _twave_suppression(qrs, ecg_corr, fs)

    return qrs


# ---------------------------------------------------------------------------
# Self-test on training data
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    from data_loader import load_project_data

    path = sys.argv[1] if len(sys.argv) > 1 else 'ProjectTrainData.mat'
    data = load_project_data(path)
    ecg_list = data['ECG']
    expert_list = data.get('QRSexpert', None)

    if expert_list is None:
        print('No QRSexpert in this file - can only detect, not score.')
        for i, ecg in enumerate(ecg_list):
            qrs = detect_qrs_causal(ecg)
            print(f'Record {i + 1}: {len(qrs)} QRS detected')
        sys.exit(0)

    print('Record | My QRS | Expert | Sens  | PPV   | F1')
    print('-' * 50)
    tot_tp = tot_fp = tot_fn = 0
    tol = 5   # 50 ms at 100 Hz

    for i in range(len(ecg_list)):
        ecg = ecg_list[i]
        expert = expert_list[i].astype(int)
        mine = detect_qrs_causal(ecg)

        tp = 0
        for e in expert:
            if len(mine) and np.min(np.abs(mine - e)) <= tol:
                tp += 1
        fn = len(expert) - tp
        fp = 0
        for m in mine:
            if len(expert) and np.min(np.abs(expert - m)) > tol:
                fp += 1

        sens = tp / (tp + fn) if tp + fn else 0
        ppv = tp / (tp + fp) if tp + fp else 0
        f1 = 2 * sens * ppv / (sens + ppv) if sens + ppv else 0
        tot_tp += tp
        tot_fp += fp
        tot_fn += fn

        print(f'{i + 1:6d} | {len(mine):6d} | {len(expert):6d} | '
              f'{sens:.3f} | {ppv:.3f} | {f1:.3f}')

    sens = tot_tp / (tot_tp + tot_fn)
    ppv = tot_tp / (tot_tp + tot_fp)
    f1 = 2 * sens * ppv / (sens + ppv)
    print('-' * 50)
    print(f'Overall: Sens={sens:.4f} PPV={ppv:.4f} F1={f1:.4f}')
    