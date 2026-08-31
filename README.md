# BMET5934 one-second sleep-apnoea baseline

This repository contains a memory-conscious MATLAB baseline that emits `N` or `A` for every annotated second. It is an **offline** classifier: both past and future context are deliberately available.

## Architecture

`run_pipeline.m` orchestrates one-patient-at-a-time signal loading, feature caching, patient-wise cross-validation, final training, test scoring, and submission writing. ECG stays at its native rate for detection; SpO2 stays at 1 Hz. Both adapters independently produce exactly one feature row per annotation second, after which their columns are concatenated.

The existing ECG implementation is reused rather than replaced:

1. `simpleFilters` performs whole-record quality marking and amplitude clipping.
2. `pan_tompkin` performs QRS detection (including its own filtering).
3. `alignQRS` refines fiducial positions and `removeCloseQRS` removes close detections.
4. The new adapter runs that chain **once per full recording**, derives RR intervals, and computes centred per-second RR/HR/HRV summaries. It does not rerun QRS detection for each window.
5. `config.ecg.qrsMode="supplied"` bypasses detection and supports the dataset QRS positions as an explicit benchmark. The default is `"existing"`.

The legacy assignment scripts assumed 100 Hz, but the reusable detector has a 200 Hz branch. The adapter passes the configured/dataset rate (default 200 Hz); it does not copy the legacy scripts' hard-coded rate.

SpO2 is treated as oxygen saturation—not raw PPG. Optional none/3-point median/5-point Gaussian preprocessing is followed by centred level, percentile, variability, desaturation, past/future delta, slope, derivative, area, timing, and drop-count features. Boundaries use available samples, so no endpoint labels are discarded. `[10 20 30 60]` multi-scale half-widths are supported; version 1 uses configurable `-20:+20` context.

## Data assumptions and validation

Training MAT files must contain cell arrays `ECG`, `SpO2`, and `Class`; `QRS` is required only in supplied-QRS mode. Test signal files contain `ECG` and `SpO2`, while the template contains cell array `Class`. `SR_ECG` and `SR_SpO2`, when present, must agree with the configured 200 Hz and 1 Hz. SpO2 must have exactly one value per annotation second. ECG duration may differ by less than one second due to an incomplete final sample block, but is never silently truncated to force alignment. Unexpected variable names produce an error listing the MAT variables.

Partial cell access uses `matfile`, so v7.3 input MAT files are strongly recommended for the approximately 1.3 GB data. Each patient is discarded after a compact cache is saved under `Results/FeatureCache`. A JSON extraction signature invalidates caches when feature settings change; `forceFeatureExtraction=true` always rebuilds them.

## Model and leakage controls

The baseline is a Statistics and Machine Learning Toolbox `fitcensemble` RUSBoost tree ensemble. Missing-value imputation and median/IQR robust normalisation parameters are learned on training data only and stored with the classifier. Although trees do not require scaling, this prevents large-range features from dominating when the classifier is swapped later. Each CV fold contains complete patients only; an assertion checks disjoint patient IDs. The `A` score column is selected by its class name, and its unbounded ensemble margin is monotonically mapped through a logistic function. The final threshold maximises pooled held-out F1 over `0.10:0.01:0.90`; test labels are never used.

Metrics are TP, TN, FP, FN, sensitivity, PPV, F1, specificity, and accuracy with `A` positive. The final model is then fit on all training patients.

## Requirements

- MATLAB (recent release with string arrays, `arguments`, `jsonencode`, and `matfile`)
- Statistics and Machine Learning Toolbox (`fitcensemble`, `templateTree`, percentiles)
- Signal Processing Toolbox (`butter`, `filtfilt`, `findpeaks`, and the existing detector dependencies)

## Run

Edit the first four paths in `run_pipeline.m`, for example:

```matlab
trainDataPath = "/data/BMET5934/ProjectTrainData.mat";
testDataPath = "/data/BMET5934/ProjectTestData.mat";
testAnnotationTemplatePath = "/data/BMET5934/ProjectTestAnnotations.mat";
outputDirectory = fullfile(pwd,"Results");
```

Then, from the repository root:

```matlab
run("run_pipeline.m")
```

Useful changes after `default_config` is called include:

```matlab
config.ecg.qrsMode = "supplied";       % optional benchmark
config.spo2.preprocessing = "none";   % avoid all smoothing
config.contextBeforeSeconds = 20;
config.contextAfterSeconds = 40;       % asymmetric offline context
config.forceFeatureExtraction = true;
```

## Outputs

- `Results/FeatureCache/{train,test}_patient_NNN.mat`: extraction-only patient caches
- `Results/cv_metrics.csv`: per-fold metrics at the CV-optimised threshold
- `Results/cv_results.mat`: held-out scores, patient-fold assignment, threshold curve, pooled metrics, and configuration
- `Results/trained_model.mat`: classifier, robust scaling/imputation, feature names, chosen threshold, configuration, and metadata
- `Results/test_scores.mat`: diagnostic scores (not part of submission)
- `Results/ProjectTestAnnotationsPredicted.mat`: submission containing **only** `Class`

The writer preserves the template cell-array shape and each cell's row/column orientation, asserts every prediction length, permits only `N`/`A`, rejects remaining `?`, and verifies that no extra MAT variables were saved.

No cross-validation figures or metrics are committed because the large project datasets are not present in this repository. Running the pipeline produces them from genuine patient-held-out predictions.

## Suggested next improvements

1. Compare asymmetric/multi-scale SpO2 contexts (especially extra future context for delayed desaturation) on the identical saved patient folds.
2. Add ECG-derived respiration and robust rolling frequency-domain HRV features without rerunning QRS detection.
3. Tune ensemble/feature selection and add validation-only temporal score smoothing, retaining the same patient folds and threshold protocol.
