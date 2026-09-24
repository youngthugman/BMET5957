import numpy as np
import pandas as pd
from scipy.io import savemat, loadmat

from ensemble.alignment import align_predictions, make_shared_folds
from ensemble.ensemble_methods import diversity, weighted_search
from ensemble.submission import create_submission, validate_submission
from ensemble.kye_cnn.cnn_cache import CHANNEL_NAMES, CONTEXT_WINDOWS, build_patient_features
from ensemble.kye_mlp.feature_extraction import FEATURE_NAMES, extract_patient_features


def test_shared_folds_and_alignment(tmp_path):
    patient = np.repeat(np.arange(1, 11), 4)
    y = np.tile([0, 0, 1, 1], 10)
    folds = make_shared_folds(y, patient, tmp_path / "folds.csv")
    assert set(folds) == set(range(1, 6))
    base = {"patient_id": patient, "second_index": np.tile(np.arange(4), 10), "y_true": y}
    outputs = {name: {**base, "probability_a": y * .8 + .1} for name in ("cnn", "mlp", "xgb")}
    frame = align_predictions(outputs, tmp_path, write_csv=True)
    assert list(frame.columns) == ["patient_id", "second_index", "y_true", "cnn_probability", "mlp_probability", "xgb_probability"]


def test_ensemble_search_and_diversity():
    y = np.array([0, 1, 0, 1])
    p = np.array([[.1, .2, .3], [.9, .8, .7], [.2, .3, .4], [.8, .7, .6]])
    search, best = weighted_search(y, p)
    assert len(search) == 231 and np.isclose(best[["cnn_weight", "mlp_weight", "xgb_weight"]].sum(), 1)
    frame = pd.DataFrame(p, columns=["cnn_probability", "mlp_probability", "xgb_probability"])
    assert len(diversity(y, frame)) == 3


def test_submission_preserves_orientation_and_only_class(tmp_path):
    template = np.empty((1, 100), object)
    probabilities = []
    for i in range(100):
        shape = (1, 3) if i % 2 else (3, 1)
        template[0, i] = np.full(shape, "?")
        probabilities.append(np.array([.1, .5, .9]))
    source, output = tmp_path / "template.mat", tmp_path / "submission.mat"
    savemat(source, {"Class": template})
    create_submission(source, probabilities, .5, output)
    validate_submission(source, output)
    saved = loadmat(output)
    assert {key for key in saved if not key.startswith("__")} == {"Class"}
    for expected, actual in zip(template.ravel(order="F"), saved["Class"].ravel(order="F")):
        assert expected.shape == actual.shape
        assert set(actual.ravel()) <= {"A", "N"}


def test_source_model_representations_and_architectures():
    seconds = 130
    ecg = np.sin(np.arange(seconds * 50) / 10)
    spo2 = np.full(seconds, 96.)
    cnn = build_patient_features(ecg, spo2, 50, 1, seconds)
    mlp = extract_patient_features(ecg, spo2, 50, 1, seconds)
    assert cnn.shape == (seconds, 35) and len(CHANNEL_NAMES) == 35
    assert mlp.shape == (seconds, len(FEATURE_NAMES)) and mlp.shape[1] != 262
    assert CONTEXT_WINDOWS == (31, 61, 91, 121)


def test_source_torch_architectures():
    import pytest
    torch = pytest.importorskip("torch")
    from ensemble.kye_cnn.cnn_model import MultiScaleAttentionModel
    from ensemble.kye_mlp.model import MLPClassifier
    cnn_model = MultiScaleAttentionModel()
    contexts = [torch.zeros(2, window, 35) for window in CONTEXT_WINDOWS]
    assert cnn_model(contexts).shape == (2,)
    linear = [layer for layer in MLPClassifier(30).network if isinstance(layer, torch.nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in linear] == [(30, 128), (128, 64), (64, 1)]
