# BMET5957 sleep-apnoea XGBoost benchmark

This repository contains a deliberately small Python experiment: supplied QRS
detections and SpO2 are converted into compact per-second features, then evaluated
with strict five-fold patient-wise XGBoost cross-validation. `A` is positive class
1, and every annotated second receives one out-of-fold prediction.

```bash
python -m pip install -r python_xgboost/requirements.txt
python python_xgboost/run_xgboost.py --data "/path/to/ProjectTrainData.mat" --patients dev20 --device cuda
```

Use `--patients all` for the complete training set and `--rebuild-cache` to force
feature extraction. Generated caches are in `python_xgboost/cache/`; OOF
predictions, feature importance, and the final model are in
`python_xgboost/results/`.
