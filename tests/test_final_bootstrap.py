from __future__ import annotations

from training.final_bootstrap import cluster_bootstrap_ci


def test_cluster_bootstrap_is_deterministic_and_bounded() -> None:
    first = cluster_bootstrap_ci([0.4, 0.5, 0.6], n_resamples=100, seed=7)
    second = cluster_bootstrap_ci([0.4, 0.5, 0.6], n_resamples=100, seed=7)
    assert first == second
    assert first["ci_low"] <= first["value"] <= first["ci_high"]
