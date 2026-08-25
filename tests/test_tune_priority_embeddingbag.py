from __future__ import annotations

import numpy as np

from training.tune_priority_embeddingbag import (
    _is_calibration,
    factorize_ranks,
    trial_configs,
)


def test_embeddingbag_trial_allocation_is_unique() -> None:
    configs = trial_configs(300)
    assert len(configs) == 300
    assert len({tuple(config.values()) for config in configs}) == 300


def test_calibration_split_is_deterministic() -> None:
    assert _is_calibration("source", "one") == _is_calibration("source", "one")


def test_nested_factorizations_have_expected_shapes() -> None:
    rng = np.random.default_rng(7)
    weights = rng.normal(size=(61, 64)).astype(np.float32)
    factors = factorize_ranks(weights)
    assert factors[8][0].shape == (64, 8)
    assert factors[32][1].shape == (61, 32)
