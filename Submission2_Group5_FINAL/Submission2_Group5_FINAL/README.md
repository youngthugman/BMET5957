# BMET3997/9997 Group 5 — Submission 2 (FINAL)

**Performance mark: 10/10** (returned 2026-05-13).

Causal QRS detection + HRV estimation pipeline for the BMET3997/9997 major
project, single-lead overnight ECG at 100 Hz.

## Returned test metrics (Siying, 2026-05-13)

| Metric | Value | Mark |
|---|---|---|
| Sensitivity | 0.99811 | — |
| PPV | 0.99615 | — |
| **F1** | **0.99713** | **5/5** (≥99.70 band) |
| MAPE_avgRR | 0.130% | — |
| MAPE_sdRR | 2.464% | — |
| MAPE_RMSSD | 8.308% | — |
| MAPE_pNN50 | 6.342% | — |
| MAPE_LF | 9.706% | — |
| MAPE_HF | 13.054% | — |
| MAPE_LF_HFratio | 14.083% | — |
| **averageMAPE** | **7.727%** | **5/5** (<10 band) |
| **Total performance mark** | | **10/10** |

For comparison, sub 1 returned F1 = 0.994 / avg MAPE = 20.85% = 6/10
(F1 3/5 + MAPE 3/5). Sub 2 improvement: +4 marks.

## Pipeline (as shipped)

```
ECG (100 Hz) ──► Anthony V2 detector (qrs_detector_causal.detect_qrs_causal)
                      │
                      └──► our `_enforce_refractory` (200 ms drop-later-of-pair)
                                  │
                                  └──► our HRV (iter-12 hybrid: Lomb-Scargle
                                              for LF/HF on ectopic-filter +
                                              repaired RR, PCHIP-FFT for
                                              LF/HF ratio, time-domain on
                                              raw RR with hard bounds)
                                              │
                                              └──► partial linear calibration:
                                                  LF: a=1.162, b=−5.77
                                                  LF/HF: a=1.323, b=0
                                                  others identity (no cal)
                                                  │
                                                  └──► .mat output
```

Detector core (`qrs_detector_causal._detect_qrs_core`) is strictly causal
(`lfilter`, past-only adaptive thresholds). Post-detection filters use
recording-level statistics — see "Caveats" below.

## Group contributions

- **James (group 5 representative)**: full sub-1 pipeline (detector + HRV);
  HRV pipeline reused unchanged for sub 2; calibration refitting strategy;
  integration, verification, and submission.
- **Anthony**: V2 QRS detector (the production detector for sub 2 after
  being wrapped with our refractory pass).
- **Josh**: end-to-end pipeline that catalysed the calibration revisit —
  his benchmark forced us to re-examine the calibration AB-test we had
  previously dismissed.

All three group members contributed essential parts of the final pipeline.

## Submission file

- **Filename**: `ProjectTestDataAnalysisGroup5Submission2.mat`
- **MD5**: `94c7c750417bef4e009a780972f4ea92`
- **Size**: 4.5 MB
- Included in `submissions/` folder of this zip.

## How to reproduce

Requires Python 3.10+, numpy, scipy. Project data `.mat` files should sit
in a `data/` folder relative to project root:

```
data/ProjectTrainData.mat
data/ProjectTestData.mat
data/ProjectTestDataAnalysis.mat
```

### Build the submission

```bash
python3 src/make_submission.py --test --group 5 --sub 2 --calibration on
```

Produces `submissions/ProjectTestDataAnalysisGroup5Submission2.mat`.

**Important:** `--calibration on` is required to reproduce the shipped file.
Without it, raw HRV is submitted (avg training MAPE 9.52% instead of 7.92%).

### Score on training (sanity check)

```bash
python3 src/make_submission.py --train
```

Should reproduce F1 = 0.9950 and per-field raw MAPE.

### Verification scripts

```bash
# Canonical training metrics report:
python3 src/final_sub2_metrics.py

# Verify the freshly written .mat matches the in-memory pipeline:
python3 src/verify_mat.py

# Full 5-check verification suite:
python3 src/verify_sub2.py

# Projection analysis (bootstrap CI + inflation projection + difficulty proxy):
python3 src/project_test_metrics.py

# Calibration decomposition (raw / Josh's coefs / refit / strict LOO):
python3 src/test_calibration_variants.py

# Re-fit calibration coefficients on V2+refractory output:
python3 src/fit_sub2_calibration.py
```

## File map

```
src/
├── make_submission.py          driver — builds the submission .mat
├── load_data.py                .mat ↔ list[np.ndarray]
├── detector.py                 original detector + _enforce_refractory
│                               (sub 2 uses only _enforce_refractory)
├── hrv.py                      HRV pipeline (iter-12 hybrid, unchanged from sub 1)
├── evaluate.py                 F1 scorer + MAPE
├── reference_hrv.py            Section-12 reference HRV table
├── calibration.py              per-field linear calibration utilities
├── fit_sub2_calibration.py     refits cal on V2+refractory outputs (partial)
├── test_calibration_variants.py  5-cell calibration comparison
├── final_sub2_metrics.py       headline metrics report
├── verify_mat.py               .mat ↔ pipeline output sanity check
├── verify_sub2.py              full 5-check verification (A–E)
└── project_test_metrics.py     test MAPE projection analysis

reference-data/Anthony-V2/Code/
└── qrs_detector_causal.py      Anthony's V2 detector (the production
                                detector for sub 2; wrapped with our
                                refractory pass in make_submission.py)

artifacts/
└── calibration_coeffs.json     partial calibration coefficients
                                (LF and LF/HF enabled; others identity)

submissions/
└── ProjectTestDataAnalysisGroup5Submission2.mat
                                the actual file submitted to Siying

CONTEXT.md                      full sub 1 → sub 2 story (read first)
PRESENTATION_NOTES.md           bullet-point summary for the final presentation
README.md                       this file
```

## Caveats (worth documenting for the report)

1. **Causality of V2 post-filters.** Anthony's V2 detector core is strictly
   causal. Its *post-detection* filters
   (`_kurtosis_artifact_filter`, `_rr_irregularity_filter`,
   `_low_density_retry`, `_high_amplitude_rate_filter`,
   `_low_amplitude_filter`) compute their gating thresholds from the median
   RMS or beat-count across **all** windows of the recording. The project
   Constraints doc is internally contradictory on this — it forbids
   "global statistics over the whole signal" but also says "processing
   a whole recording as a batch is fine." Our shipping decision was the
   lenient reading. Sub 2 grader return did not penalise this, but worth
   documenting in the final report's methodology section.

2. **Refractory enforcement is required.** V2's `_low_density_retry` and
   `_secondary_pass` can add beats post-hard-refractory, producing
   sub-200ms gaps on a small fraction of records. Wrapping the V2 detector
   with our `_enforce_refractory` (200 ms drop-later-of-pair) is the
   essential fix and is what `make_submission.detect_qrs` does.

3. **Partial calibration shipped.** LF and LF/HF only; sdRR/RMSSD/pNN50
   refit calibration improvements were small (<1 pp) and were left
   identity to minimise risk surface on test transfer. The test return
   validated this — calibrated fields (LF 9.71% test vs 9.39% training,
   LF/HF 14.08% vs 14.98%) transferred essentially perfectly.
