from __future__ import annotations

from pathlib import Path

import numpy as np

from training.priority_embeddingbag import (
    LowRankEmbeddingBagModel,
    embeddingbag_scores,
    factorize_positive_weights,
    identity_calibration,
)
from training.priority_hash import document_features


def test_full_rank_factorization_reconstructs_positive_weights() -> None:
    weights = np.asarray([[1.0, -2.0, 3.0], [0.0, 2.0, 1.0]], dtype=np.float32)
    embeddings, head = factorize_positive_weights(weights, rank=2)
    reconstructed = head @ embeddings.T
    assert np.allclose(reconstructed, np.maximum(weights, 0.0), atol=1e-5)


def test_embeddingbag_model_round_trip(tmp_path: Path) -> None:
    labels = ("tag",)
    features = document_features("passport number AB123456")
    embeddings = np.zeros((1 << 17, 2), dtype=np.float32)
    embeddings[features, 0] = 2.0
    head = np.asarray([[1.0, 0.0]], dtype=np.float32)
    scale, bias = identity_calibration(1)
    raw = embeddingbag_scores(embeddings, head, features)
    assert raw[0] == 2.0
    model = LowRankEmbeddingBagModel(
        labels,
        embeddings,
        head,
        scale,
        bias,
        np.asarray([1.0]),
        20_000,
    )
    assert model.predict("passport number XY999999") == ["tag"]
    model.save(tmp_path)
    assert LowRankEmbeddingBagModel.load(tmp_path).predict(
        "passport number XY999999"
    ) == ["tag"]
