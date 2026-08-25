from __future__ import annotations

from training.priority_data import PRIORITY_TAGS
from training.tune_priority_fusion import OPTIONS, strategy_configs


def test_fusion_trial_allocation_and_priority_lock() -> None:
    labels = (PRIORITY_TAGS[0], *(f"generic-{index}" for index in range(5)))
    ranked = {label: list(OPTIONS) for label in labels}
    configs = strategy_configs(ranked, labels, n_trials=100)
    assert len(configs) == 100
    assert all(config[PRIORITY_TAGS[0]] == "source:recall" for config in configs)
