import numpy as np
import pandas as pd
from scipy.io import savemat, loadmat

from ensemble.alignment import align_predictions, make_shared_folds
from ensemble.ensemble_methods import diversity, weighted_search
from ensemble.submission import create_submission, validate_submission
from ensemble.cnn_adapter import native_cache, native_model
from ensemble.mlp_adapter import native_model as native_mlp_model
from ensemble.source_fidelity import audit


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


def test_source_model_representations_and_architectures(tmp_path):
    seconds = 130
    spo2 = np.full(seconds, 96.)
    qrs = np.arange(0, seconds * 200, 200)
    cnn = native_cache.build_patient_features(spo2, qrs)
    assert cnn.shape == (35, seconds) and native_model.NUM_CHANNELS == 35
    assert native_model.MultiScaleAttentionModel().windows == [31, 61, 91, 121]
    assert "OVERALL: MATCH" in audit(tmp_path / "audit.txt")


def test_source_torch_architectures():
    import pytest
    torch = pytest.importorskip("torch")
    cnn_model = native_model.MultiScaleAttentionModel()
    contexts = [torch.zeros(2, 35, window) for window in (31, 61, 91, 121)]
    assert cnn_model(contexts).shape == (2,)
    linear = [layer for layer in native_mlp_model.MLP(30).network if isinstance(layer, torch.nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in linear] == [(30, 128), (128, 64), (64, 1)]
