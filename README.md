# BMET5957 sleep-apnoea classification

This repository is being rebuilt as a lean MATLAB experimentation pipeline with a minimal number of files and an obvious data flow:

```text
raw ECG + SpO2
    ->
feature extraction
    ->
cached per-second feature matrices
    ->
patient-wise classifier training / cross-validation
    ->
test prediction
    ->
submission output
```

Caching will be retained because extracting features from raw ECG is expensive. Feature extractors, classifiers, and hyperparameters will remain easy to swap without unnecessary wrappers, helper layers, or deeply nested configuration.

## Planned files

- `run_model.m`
- `build_features.m`
- `extract_ecg_features.m`
- `extract_spo2_features.m`
- `train_classifier.m`
- `output_results.m`

These files are planned and have not yet been implemented.

## Future data contract

For a patient with `N` annotated seconds:

| Data | Shape |
| --- | --- |
| Raw ECG | approximately `1 x (N * 200)` |
| Raw SpO2 | `1 x N` |
| Raw Class | `1 x N` |
| ECG extractor output | `N x number_of_ECG_features` |
| SpO2 extractor output | `N x number_of_SpO2_features` |
| Combined feature matrix | `N x total_features` |

Each row of every extracted or combined feature matrix always represents exactly one second.

`ECG_Feature_Extraction/` is an existing external library and is kept unchanged.
