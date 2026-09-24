# Patient-wise ensemble experiment

This directory integrates the Python `Kye-1D_CNN` and `Kye-MLP` model families
while retaining `19926-XGBoost-Tuning` as the canonical label, patient/second,
metric, output, and submission pipeline. Only XGBoost receives its 262-feature
matrix. CNN receives its native 35 channels and 31/61/91/121-second contexts;
MLP receives its own raw-signal feature representation.

## Development run

First reproduce the two branch-native evaluations:

```powershell
python .\ensemble\run_ensemble.py `
  --mode reproduce `
  --data "E:\Desktop\Downloads\ProjectTrainData.mat" `
  --device cuda `
  --models cnn mlp
```

This writes `results/native_reproduction_cnn.csv` and
`results/native_reproduction_mlp.csv`. Every run first performs the fail-closed
blob and scientific-setting audit in `results/source_fidelity_audit.txt`.

Then run only the three base models on shared folds:

```powershell
python .\ensemble\run_ensemble.py `
  --mode cv `
  --data "E:\Desktop\Downloads\ProjectTrainData.mat" `
  --device cuda `
  --cv 5fold `
  --base-models-only
```

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

The authoritative byte-for-byte CNN sources live under `native_cnn/`:
`cnn_cache.py`, `cnn_model.py`, and `cnn_evaluation.py`. Its fixed representation
is 35 channels, and `MultiScaleAttentionModel` receives centred contexts of 31,
61, 91, and 121 seconds. The adapter retains native normalization, 1,500 samples
per patient with a 30% positive target, unweighted `BCEWithLogitsLoss`, AdamW,
and validation-F1 early stopping. It uses the MAT file's supplied `QRS` and does
not run the Group 5 detector.

The authoritative byte-for-byte MLP sources live under `native_mlp/`. Its network
is `Linear(input,128) → LayerNorm → GELU → Dropout(0.25) → Linear(128,64) →
LayerNorm → GELU → Dropout(0.25) → Linear(64,1)`. Mean `SimpleImputer`
preprocessing and `StandardScaler` are fitted exclusively on training patients.
The MLP independently runs its native extractor and never consumes CNN channels
or XGBoost's 262-feature matrix.
