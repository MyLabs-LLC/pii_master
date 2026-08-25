from __future__ import annotations

import numpy as np

from training.tune_priority_hash import (
    _is_validation,
    _objective,
    fast_metrics,
    threshold_bank,
    trial_configs,
)


def test_trial_configs_are_unique_and_bounded() -> None:
    configs = trial_configs(300)
    assert len(configs) == 300
    assert len({tuple(config.values()) for config in configs}) == 300


def test_validation_split_has_stable_missing_hash_fallback() -> None:
    row = {"dataset": "source", "uid": "missing", "text_sha256": ""}
    assert _is_validation(row) == _is_validation(dict(row))


def test_feasible_objective_prefers_macro_f2_over_excess_recall() -> None:
    recall_max = {
        "priority_point_passes": 10,
        "measurable_priority_gates": 10,
        "worst_priority_recall": 0.99,
        "equal_corpus_macro_f2": 0.4,
        "equal_corpus_micro_f1": 0.5,
    }
    f2_max = {**recall_max, "worst_priority_recall": 0.90, "equal_corpus_macro_f2": 0.6}
    assert _objective(f2_max) > _objective(recall_max)


def test_threshold_bank_uses_worst_source_for_priority_tag() -> None:
    label = "sensitive_pii_social_security_number"
    labels = (label,)
    datasets = np.asarray(["a"] * 30 + ["b"] * 30)
    y_true = np.ones((60, 1), dtype=np.bool_)
    scores = {
        mode: np.asarray([[0.9]] * 30 + [[0.2]] * 30)
        for mode in ("top1", "top3", "top6")
    }
    bank = threshold_bank(scores, y_true, datasets, labels)
    assert bank["top1"][0, 5] == np.float32(0.2)


def test_fast_metrics_equal_weights_complete_corpora() -> None:
    labels = ("sensitive_pii_social_security_number",)
    y_true = np.asarray([[True], [True]], dtype=np.bool_)
    predicted = np.asarray([[True], [False]], dtype=np.bool_)
    datasets = np.asarray(["a", "b"])
    complete = np.asarray([True, True])
    quality = {
        "a": {"tag_counts": {labels[0]: 1}},
        "b": {"tag_counts": {labels[0]: 1}},
    }
    result = fast_metrics(predicted, y_true, datasets, complete, labels, quality)
    assert result["equal_corpus_macro_f2"] == 0.5
    assert result["equal_corpus_micro_f1"] == 0.5
