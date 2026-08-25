from __future__ import annotations

from pathlib import Path

import numpy as np

from training.priority_hash import (
    HashCounts,
    HashCueModel,
    build_weights,
    document_features,
    score_modes,
)


def test_features_redact_values_and_are_deterministic() -> None:
    first = document_features("SSN: 123-45-6789")
    second = document_features("SSN: 987-65-4321")
    assert np.array_equal(first, second)
    assert len(first) > 0


def test_partial_rows_never_create_negative_counts() -> None:
    counts = HashCounts.empty(("tag-a", "tag-b"), n_features=32)
    features = np.asarray([1, 3, 5], dtype=np.int32)
    counts.update(features, {"tag-a"}, label_complete=False)
    assert counts.n_complete == 0
    assert counts.positive_partial_df[0, 1] == 1
    assert counts.positive_complete_df.sum() == 0
    assert counts.complete_df.sum() == 0


def test_complete_rows_create_positive_and_background_evidence() -> None:
    counts = HashCounts.empty(("tag-a", "tag-b"), n_features=32)
    features = np.asarray([2, 4], dtype=np.int32)
    counts.update(features, {"tag-a"}, label_complete=True)
    weights = build_weights(counts, min_document_frequency=1)
    assert weights[0, 2] > weights[1, 2]


def test_model_round_trip(tmp_path: Path) -> None:
    labels = ("tag-a", "tag-b")
    weights = np.zeros((2, 1 << 17), dtype=np.float32)
    features = document_features("passport number AB123456")
    weights[0, features] = 2.0
    scores = score_modes(weights, features)["top1"]
    model = HashCueModel(labels, weights, np.asarray([1.0, 1.0]), "top1", 20_000)
    assert scores[0] == 2.0
    assert model.predict("passport number XY999999") == ["tag-a"]
    model.save(tmp_path)
    loaded = HashCueModel.load(tmp_path)
    assert loaded.predict("passport number XY999999") == ["tag-a"]
