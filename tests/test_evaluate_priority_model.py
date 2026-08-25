from __future__ import annotations

from pathlib import Path

import numpy as np

from training.evaluate_priority_model import _load_worker, _predict_one
from training.priority_hash import HashCueModel, document_features


def test_worker_loads_and_predicts_without_leaking_text(tmp_path: Path) -> None:
    text = "passport number AB123456"
    features = document_features(text)
    weights = np.zeros((1, 1 << 17), dtype=np.float32)
    weights[0, features] = 2.0
    model = HashCueModel(
        ("sensitive_pii_passport_number",),
        weights,
        np.asarray([1.0]),
        "top1",
        20_000,
    )
    model_dir = tmp_path / "model"
    model.save(model_dir)
    document = tmp_path / "document.txt"
    document.write_text(text, encoding="utf-8")
    _load_worker(str(model_dir))
    dataset, uid, labels, error = _predict_one(("heldout", "one", str(document)))
    assert (dataset, uid, error) == ("heldout", "one", "")
    assert labels == ["sensitive_pii_passport_number"]
