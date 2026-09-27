import json

import numpy as np
import pandas as pd

from ensemble.postprocessing import apply_rule, run_search


def test_fill_bounded_non_apnea_gaps():
    patients = np.ones(8)
    np.testing.assert_array_equal(
        apply_rule([1, 0, 1], patients[:3], fill_n_max=1), [1, 1, 1])
    np.testing.assert_array_equal(
        apply_rule([1, 0, 0, 1], patients[:4], fill_n_max=2), [1, 1, 1, 1])
    np.testing.assert_array_equal(
        apply_rule([1, 0, 0, 1], patients[:4], fill_n_max=1), [1, 0, 0, 1])


def test_remove_bounded_apnea_runs():
    patients = np.ones(8)
    np.testing.assert_array_equal(
        apply_rule([0, 1, 0], patients[:3], remove_a_max=1), [0, 0, 0])
    np.testing.assert_array_equal(
        apply_rule([0, 1, 1, 0], patients[:4], remove_a_max=2), [0, 0, 0, 0])
    np.testing.assert_array_equal(
        apply_rule([0, 1, 1, 0], patients[:4], remove_a_max=1), [0, 1, 1, 0])


def test_record_edges_are_never_replaced():
    np.testing.assert_array_equal(
        apply_rule([0, 1, 0, 0], [1] * 4, fill_n_max=2), [0, 1, 0, 0])
    np.testing.assert_array_equal(
        apply_rule([1, 0, 1, 1], [1] * 4, remove_a_max=2), [1, 0, 1, 1])


def test_rules_never_cross_patient_boundaries():
    # Concatenating these records resembles A N A, but neither local N has two bounds.
    prediction = np.array([1, 0, 1])
    output = apply_rule(prediction, [1, 1, 2], fill_n_max=1)
    np.testing.assert_array_equal(output, prediction)


def test_search_writes_all_development_artifacts(tmp_path, capsys):
    cache = tmp_path / "nested.npz"
    probability = np.array([.9, .1, .9, .1, .8, .1])
    labels = np.array([1, 1, 1, 0, 1, 0])
    np.savez_compressed(cache, probability_a=probability, y_true=labels,
                        patient_id=np.array([1, 1, 1, 2, 2, 2]),
                        second_index=np.array([0, 1, 2, 0, 1, 2]))
    table, summary = run_search(cache, tmp_path / "results")

    assert len(table) == 8
    assert table.f1.is_monotonic_decreasing
    assert summary["evaluation"] == "development-set post-processing search"
    assert summary["patient_independent_processing"] is True
    assert (tmp_path / "results" / "postprocessing_search.csv").is_file()
    assert json.loads((tmp_path / "results" / "postprocessing_best.json").read_text()) == summary
    with np.load(tmp_path / "results" /
                 "oof_predictions_5fold_nested_weighted_soft_vote_postprocessed_best.npz",
                 allow_pickle=False) as saved:
        assert len(saved["prediction"]) == len(labels)
        assert set(saved["prediction"]) <= {0, 1}
    assert "predicted_positive_fraction" in pd.read_csv(
        tmp_path / "results" / "postprocessing_search.csv").columns
    assert "fill_n_gaps_le_1" in capsys.readouterr().out
