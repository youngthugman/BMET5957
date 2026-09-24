# Patient-wise ensemble experiment

This directory integrates the Python `Kye-1D_CNN` and `Kye-MLP` model families
while retaining `19926-XGBoost-Tuning` as the canonical label, patient/second,
metric, output, and submission pipeline. Only XGBoost receives its 262-feature
matrix. CNN receives its native 35 channels and 31/61/91/121-second contexts;
MLP receives its own raw-signal feature representation.

## Development run

```powershell
python .\ensemble\run_ensemble.py `
  --mode cv `
  --data "E:\Desktop\Downloads\ProjectTrainData.mat" `
  --device cuda `
  --cv 5fold `
  --methods all `
  --stacking-mode diagnostic
```

The large aligned CSV is opt-in with `--write-aligned-csv`; the aligned NPZ is
always written. Completed model/fold caches are loaded on a restart only when
their seed, feature/model version, source, and validation patients match.

## Leakage-safe overnight run

```powershell
python .\ensemble\run_ensemble.py `
  --mode cv `
  --data "E:\Desktop\Downloads\ProjectTrainData.mat" `
  --device cuda `
  --cv 5fold `
  --methods all `
  --stacking-mode nested
```

For every outer fold, nested mode creates five inner patient folds using only
outer-training patients. It obtains inner OOF probabilities for all three base
models, fits each probability-only meta-classifier, and applies it to the three
cached base predictions for the untouched outer-validation patients. Outer
validation labels are not used for fitting, preprocessing, or threshold choice.

## Hidden-test submission

```powershell
python .\ensemble\run_ensemble.py `
  --mode test `
  --train-data "E:\Desktop\Downloads\ProjectTrainData.mat" `
  --test-data "E:\Desktop\Downloads\ProjectTestData.mat" `
  --annotation-template "E:\Desktop\Downloads\ProjectTestAnnotations.mat" `
  --device cuda `
  --method equal_soft_vote `
  --group 14 `
  --submission 1
```

This writes `submissions/ProjectTestAnnotationsGroup14Submission1.mat`. The MAT
file has only `Class`, with the template's 100-cell shape and every patient
cell's exact shape/orientation preserved. Only `A`/`N` values are permitted.
Base and ensemble probabilities are kept separately under `results/`.

Detailed timestamped logs are in `logs/`; the console deliberately reports only
fold progress and the final compact table. The adapters train sequentially and
release CUDA memory between models.

## Source preservation

The adapted CNN sources live under `kye_cnn/` using the original source names:
`cnn_cache.py`, `cnn_model.py`, and `cnn_evaluation.py`. Its fixed representation
is 35 channels, and `MultiScaleAttentionModel` receives centred contexts of 31,
61, 91, and 121 seconds. The shared-fold adapter retains normalization, weighted
training sampling, weighted binary loss, AdamW, and patient-only early stopping.

The adapted MLP sources live under `kye_mlp/` using the original source names:
`model.py`, `feature_extraction.py`, `evaluation.py`, and `main.py`. Its network
is `Linear(input,128) → LayerNorm → GELU → Dropout(0.25) → Linear(128,64) →
LayerNorm → GELU → Dropout(0.25) → Linear(64,1)`. Median imputation and
standardisation are fitted exclusively on outer-training patients.
