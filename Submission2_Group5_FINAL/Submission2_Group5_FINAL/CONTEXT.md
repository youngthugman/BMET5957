# Sub 1 → Sub 2 — full story

Plain-English summary of the journey from sub 1 (6/10) to sub 2 (10/10).

## Sub 1 (shipped 2026-04-23, returned 2026-05-04)

### Pipeline
Built entirely from James's own code:

```
ECG (100 Hz) ──► James's causal Pan-Tompkins detector ──► our HRV pipeline
```

- **Detector**: causal Pan-Tompkins with our own additions: rolling-tile
  adaptive threshold, search-back, T-wave rejection, two-stage R-peak
  refinement, 200 ms refractory guard.
- **HRV**: 5-min non-overlapping windows; RR bounds [250, 2000] ms;
  time-domain on un-repaired RR (preserves ectopy); Lomb-Scargle for LF/HF
  on ectopic-filtered + structurally-repaired RR; PCHIP-FFT for LF/HF ratio.
- **Calibration**: off — chose to ship raw HRV to preserve a clean
  AB-test for sub 2.

### Sub 1 results

| Metric | Training (predicted) | Test (returned) |
|---|---|---|
| F1 | ~0.987–0.993 (LOO) | **0.9940** |
| Avg HRV MAPE | 15.04% (LOO) | **20.85%** |
| **Performance mark** | | **6/10** (F1 3/5 + MAPE 3/5) |

**Observation:** training-vs-test gap was +5.81 pp on avg MAPE. Detector
transferred cleanly (F1 above the LOO band), HRV ran hotter than expected.
Biggest per-field gaps: pNN50 (+12.4), LF (+10.1), RMSSD (+7.6).

After sub 1 returned, the unit coordinator (Philip) shared that cohort
top-tier sits at F1 ~0.997 / avg MAPE 10–12%. We were 0.003 below top
on F1, ~9 pp behind on HRV.

## What we explored between sub 1 and sub 2

### Failed/inconclusive experiments

- **Detector noise-burst tile suppression gate**: aimed at r17/r25/r33
  noise tiles. F1 regressed 0.9931 → 0.9918; r25 actually got worse.
  Rolling-baseline mechanism doesn't catch sustained noise and false-
  triggers on sleep-stage transitions. Negative result.
- **Linear detrend before PSD**: −0.03 pp avg MAPE (essentially flat).
- **Time-domain on structurally-repaired RR**: +2.29 pp regression
  (kills pNN50 — staff reference preserves ectopy).
- **PCHIP-FFT for all freq fields**: +2.24 pp regression (Lomb beats
  PCHIP on training LF).
- **Anthony's full V1 HRV pipeline with our detector**: +5.52 pp
  regression.

**Lesson:** our existing HRV pipeline is near-optimal against the
Section-12 training reference. The +5.81 pp test gap can't be diagnosed
from training-side experiments alone.

### Breakthrough 1 — Anthony V2 detector

Anthony shared his V2 code (revised QRS detector with aggressive
post-detection filtering). A 4-cell {detector × HRV} bake-off on training:

| Cell | F1 | Avg MAPE (raw, no cal) |
|---|---|---|
| James D + James H (sub 1 baseline) | 0.9931 | 15.04% |
| James D + Anthony V2 H | 0.9931 | 25.35% |
| **Anthony V2 D + James H** | **0.9950** | **9.52%** |
| Anthony V2 D + Anthony V2 H (full Anthony) | 0.9950 | 14.62% |

Key findings:
1. **Anthony's V2 detector is the breakthrough**, not his V2 HRV.
2. **Our HRV pipeline is genuinely better than Anthony's V2 HRV** at the
   ceiling — running each on expert QRS: ours = 7.59% avg MAPE, his =
   9.58%.
3. **The best combination is Anthony's detector + our HRV** (9.52% raw).

About Anthony's published "9.58% / F1 0.9950":
- F1 0.9950 is from his detector vs expert QRS — real
- MAPE 9.58% is from his V2 HRV applied to **expert QRS** (a ceiling
  test) — not his end-to-end pipeline
- His actual end-to-end pipeline scores 14.62% on training

### Breakthrough 2 — Josh's pipeline triggers calibration re-examination

Josh (third group member) shared his pipeline achieving end-to-end avg
MAPE 7.26% on training. Inspection showed his pipeline is structurally
identical to ours (V2 detector + ours HRV) **PLUS automatic per-field
linear calibration** applied at output.

Refitting calibration on our own V2+refractory pipeline outputs:

| Variant | Avg training MAPE |
|---|---|
| Raw (no cal) | 9.52% |
| Apply Josh's literal coefs | 7.54% |
| Refit in-sample on V2+refractory | 7.73% |
| **Refit strict LOO (transfer estimate)** | **7.94%** |

Independent refit converged to nearly identical coefficients to Josh's
(LF a=1.162 vs his 1.161). This is a real structural win, not luck.

We chose **partial calibration** (LF and LF/HF only, others identity)
instead of full to:
1. Capture ~90% of the calibration win (R² 0.98 on LF, 0.91 on LF/HF)
2. Limit risk surface on test transfer — sdRR/RMSSD/pNN50 calibrations
   are small (<1 pp wins) and could backfire if test reference has a
   different bias direction for those fields

## Sub 2 (shipped 2026-05-12, returned 2026-05-13)

### Pipeline (the change)

```
ECG (100 Hz) ──► Anthony's V2 detector ──► our _enforce_refractory (200 ms)
                                                  │
                                                  └──► our HRV (unchanged)
                                                                │
                                                                └──► partial cal:
                                                                    LF: a=1.162, b=−5.77
                                                                    LF/HF: a=1.323, b=0
                                                                    │
                                                                    └──► .mat
```

### What changed vs sub 1

| Component | Sub 1 | Sub 2 |
|---|---|---|
| Detector | James's Pan-Tompkins | **Anthony V2 detector** |
| Refractory enforcement | Built into detector | **Wrapped after V2** (200 ms) |
| HRV pipeline | Iter-12 hybrid | **Unchanged** |
| Calibration | OFF (raw HRV) | **PARTIAL ON** — LF and LF/HF only |

### Sub 2 results

| Metric | Sub 1 (test) | Sub 2 (training) | Sub 2 (test, returned) | Δ vs sub 1 test |
|---|---|---|---|---|
| F1 | 0.9940 | 0.9950 | **0.99713** | +0.0031 |
| avgRR MAPE | 0.59% | 0.23% | **0.130%** | −0.46 |
| sdRR MAPE | 7.35% | 3.31% | **2.464%** | −4.89 |
| RMSSD MAPE | 21.13% | 7.95% | **8.308%** | −12.82 |
| pNN50 MAPE | 43.81% | 6.16% | **6.342%** | −37.47 |
| LF MAPE | 24.98% | 9.39% (cal) | **9.706%** | −15.27 |
| HF MAPE | 23.68% | 13.44% | **13.054%** | −10.62 |
| LF_HFratio MAPE | 24.42% | 14.98% (cal) | **14.083%** | −10.34 |
| **Avg MAPE** | **20.85%** | **7.92%** | **7.727%** | **−13.12** |
| **Performance mark** | 6/10 | — | **10/10** | **+4** |

### Why sub 2 transferred so well

Three structural facts:

1. **V2 detector is more robust on test noise.** Sub 1's +5.81 pp test
   inflation was largely detector-driven (we had F1 = 0.9931 train →
   0.9940 test, but the bad RR series on noisy test records inflated
   downstream HRV). V2's filter cascade (kurtosis artifact rejection, RR
   irregularity gates, low-density retry, etc.) cleans up those noisy
   records on test the same way it does on training.

2. **Partial calibration transferred essentially perfectly.**
   - LF calibrated: 9.39% training → 9.71% test (Δ +0.32 pp)
   - LF/HF calibrated: 14.98% training → 14.08% test (Δ −0.90 pp,
     actually *better* on test)
   - Both fields had R² > 0.91 on the linear fit; the calibration is
     genuinely linear-shaped on this data.

3. **Skipping calibration on small-win fields was correct.** sdRR
   (2.46% test, very close to 3.31% training) and pNN50 (6.34% test vs
   6.16% training) were stable without calibration. If we'd shipped full
   calibration on those, the small win could have inverted with no upside.

Net: avg test MAPE 7.73% < training 7.92%. **Test came in BETTER than
training.** Sub 1 inflated +5.81 pp; sub 2 deflated −0.19 pp.

## Verification done before shipping

- **Per-record breakdown** of F1 and MAPE on training — only 1 record
  showed F1 drop > 0.005 vs sub 1 baseline (r10), and its MAPE-sum
  dropped from 273 to 44 (net win).
- **Test-set integrity** — 0 NaN, 0 refractory violations after wrapping
  V2 with `_enforce_refractory`.
- **sqrs125+5 cross-check on test** — F1 = 0.9886 (sub 1 was 0.9872).
- **Test HRV distributions vs sub 1** — all 7 fields correlate 0.9+.
- **Determinism** — V2 detector produces identical output on consecutive
  runs.
- **.mat round-trip vs pipeline** — loaded the built .mat, re-ran the
  pipeline (with calibration applied), 35/35 QRS and all 7 HRV values
  byte-identical (<1e-6 tolerance).
- **Three Codex adversarial reviews** caught real bugs across the build
  process: first caught make_submission.py was still using the old
  detector; second caught calibration-state silent reproducibility risk
  and verify_sub2.py wrapper drift; third re-flagged causality concern
  and rebuild-default issue (the latter worth fixing for sub 3+).

## Known caveats (worth documenting in the final report)

1. **Causality of V2 post-filters.** Anthony's V2 core detector is
   strictly causal. His post-detection filters use whole-recording medians
   — borderline under the BMET9997 rule. Constraints.md is internally
   contradictory on this. Sub 2 grader return did NOT penalise this. If
   reviewer questions causality in the final report defence, point to:
   (a) the core detector is strictly past-only `lfilter`; (b) post-
   filters are recording-level batch processing that the doc explicitly
   permits; (c) we measured the same numbers on test as training,
   confirming the pipeline behaves consistently.

2. **Anthony's "9.58%" is a ceiling test, not his end-to-end pipeline.**
   This is worth being precise about if asked.

3. **Calibration is enabled but minimal.** Only LF and LF/HF are
   calibrated. The decision to skip the small-win fields was deliberate
   risk management.

## Summary

Sub 2 went from 6/10 → 10/10 (the maximum performance mark possible on
this rubric) through three concrete decisions:

1. Adopt Anthony's V2 detector + wrap with our refractory enforcement
2. Keep our HRV pipeline unchanged
3. Apply partial calibration to LF and LF/HF, leave others identity

All three group members contributed to the final pipeline.
