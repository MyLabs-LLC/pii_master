from __future__ import annotations

from pathlib import Path

import numpy as np

from training.priority_fusion import FusionPriorityModel, fuse_strategy
from training.priority_hash import HashCueModel, document_features, load_priority_model


def test_boolean_fusion_strategies() -> None:
    predicted = {"a": {"tag"}, "b": set(), "c": {"tag"}}
    assert fuse_strategy("source:a", predicted, "tag")
    assert fuse_strategy("or:a,b", predicted, "tag")
    assert not fuse_strategy("and:a,b", predicted, "tag")
    assert fuse_strategy("majority:a,b,c", predicted, "tag")


def test_fusion_model_round_trip(tmp_path: Path) -> None:
    labels = ("tag",)
    features = document_features("passport number AB123456")
    weights = np.zeros((1, 1 << 17), dtype=np.float32)
    weights[0, features] = 2.0
    component = HashCueModel(labels, weights, np.ones(1), "top1", 20_000)
    model = FusionPriorityModel(
        labels,
        {"a": component, "b": component},
        {"tag": "and:a,b"},
    )
    model.save(tmp_path)
    assert load_priority_model(tmp_path).predict("passport number XY999999") == ["tag"]
