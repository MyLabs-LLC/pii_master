from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.priority_eval import (
    EvaluationRow,
    aggregate_arms,
    evaluate_corpus,
    freeze_catalogue,
)


def _row(
    uid: str, gold: set[str], predicted: set[str], *, complete: bool = True
) -> EvaluationRow:
    return EvaluationRow(
        "heldout", uid, frozenset(gold), frozenset(predicted), complete
    )


def test_complete_metrics_use_fixed_catalogue_denominator() -> None:
    rows = [
        _row("1", {"a"}, {"a", "b"}),
        _row("2", {"b"}, {"a"}),
    ]
    result = evaluate_corpus(rows, catalogue=["a", "b"])
    assert result["per_tag"]["a"]["recall"] == 1.0
    assert result["per_tag"]["a"]["precision"] == 0.5
    assert result["per_tag"]["b"]["recall"] == 0.0
    assert result["macro_f2"] == pytest.approx(5 / 12)
    assert result["micro_f1"] == pytest.approx(0.4)


def test_partial_labels_only_score_known_positive_recall() -> None:
    tag = "sensitive_pii_full_name"
    rows = [
        _row(str(i), {tag}, {tag} if i < 27 else set(), complete=False)
        for i in range(30)
    ]
    result = evaluate_corpus(rows, catalogue=[tag])
    assert result["macro_f2"] is None
    assert result["per_tag"][tag]["precision"] is None
    assert result["priority"][tag]["recall"] == 0.9
    assert result["priority"][tag]["status"] == "POINT_PASS_UNVERIFIED"
    assert result["engine"]["assessable"] is False


def test_support_gate_and_bootstrap_are_deterministic() -> None:
    tag = "sensitive_pii_social_security_number"
    rows = [_row(str(i), {tag}, {tag}) for i in range(40)]
    first = evaluate_corpus(rows, catalogue=[tag], bootstrap=True, n_resamples=50)
    second = evaluate_corpus(rows, catalogue=[tag], bootstrap=True, n_resamples=50)
    assert first["priority"][tag] == second["priority"][tag]
    assert first["priority"][tag]["status"] == "PASS"
    assert first["priority"][tag]["ci_low"] == 1.0


def test_aggregate_is_equal_corpus_not_row_weighted() -> None:
    arm_a = evaluate_corpus([_row("1", {"a"}, {"a"})], catalogue=["a"])
    arm_b = evaluate_corpus([_row("2", {"a"}, set())], catalogue=["a"])
    summary = aggregate_arms([arm_a, arm_b])
    assert summary["equal_corpus_macro_f2"] == 0.5
    assert summary["equal_corpus_micro_f1"] == 0.5


def test_freeze_catalogue_records_hash_and_read_depths(tmp_path: Path) -> None:
    quality = tmp_path / "quality.json"
    quality.write_text(
        json.dumps(
            {
                "eval": {
                    "eval-a": {
                        "tag_counts": {"sensitive_pii_full_name": 3},
                        "complete_label_rows": 4,
                        "n_rows": 4,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    frozen = freeze_catalogue(quality, tmp_path / "catalogue.json")
    assert frozen["corpora"]["eval-a"]["label_complete"] is True
    assert frozen["metric_contract"]["read_depths"] == [1000, 2500, 10000, 20000]
