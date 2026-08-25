from __future__ import annotations

from training.benchmark_read_depth import stratified_sample


def test_stratified_sample_caps_each_dataset() -> None:
    rows = [
        {"dataset": dataset, "uid": f"{dataset}-{index}", "read_error": ""}
        for dataset in ("a", "b")
        for index in range(20)
    ]
    sample = stratified_sample(rows, per_dataset=5)
    assert len(sample) == 10
    assert {row["dataset"] for row in sample} == {"a", "b"}
