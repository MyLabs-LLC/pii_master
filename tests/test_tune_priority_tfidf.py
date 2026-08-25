from __future__ import annotations

import numpy as np

from training.priority_hash import HashCounts
from training.tune_priority_tfidf import build_tfidf_weights


def test_tfidf_boosts_rare_discriminative_feature() -> None:
    counts = HashCounts.empty(("tag",), n_features=8)
    counts.n_all = 100
    counts.n_complete = 100
    counts.all_df[:] = 50
    counts.complete_df[:] = 50
    counts.all_df[1] = 2
    counts.complete_df[1] = 2
    counts.n_positive_complete[0] = 10
    counts.positive_complete_df[0, 0] = 5
    counts.positive_complete_df[0, 1] = 2
    weights = build_tfidf_weights(counts, min_document_frequency=1)
    assert np.isfinite(weights).all()
    assert weights[0, 1] > weights[0, 0]
