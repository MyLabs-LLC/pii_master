"""The absence contract and the frozen collapse are load-bearing; test them."""

from __future__ import annotations

from pathlib import Path

import pytest

from training.quiet_data import (
    COLLAPSE,
    EVAL_ROOT,
    JUDGE_ASSERTED,
    TRAIN_ROOT,
    assert_absence_contract,
    collapse_tags,
    iter_quiet_corpus,
    resolve_dataset,
)


def _dir(name: str) -> Path:
    """Resolve by stem, so a dataset rename skips no test silently.

    A literal lookup here skipped `test_complete_corpus_reports_real_negatives`
    the moment `pii2_eval_30k` was renamed -- and that is the test guarding the
    4,851 negatives, so the suite would have gone green while the property it
    exists to protect was the one that had just broken.
    """
    try:
        return resolve_dataset(name)
    except FileNotFoundError:
        pytest.skip(f"{name} not present")


@pytest.mark.parametrize("name", JUDGE_ASSERTED)
def test_absence_contract_holds(name: str) -> None:
    """Empty entities+classes <-> sensitivity 'none', on every row.

    This is the licence for reading a dual-judge row as a confirmed negative.
    If it ever stops holding, training on those rows would teach the model to
    stay silent on documents nobody actually examined.
    """
    counts = assert_absence_contract(_dir(name))
    assert counts["rows"] > 0
    assert counts["negative"] > 0
    # Ambiguous rows are expected and small; they must stay excluded, not be
    # quietly folded into either class.
    assert counts["ambiguous"] / counts["rows"] <= 0.02
    assert counts["positive"] + counts["negative"] + counts["ambiguous"] == counts["rows"]


def test_judge_negatives_are_admitted() -> None:
    """The correction's whole point: these rows used to be invisible."""
    rows = list(iter_quiet_corpus(_dir("6589_govdocs2-dualjudge-eval20-3.53k")))
    negatives = [r for r in rows if r.doc_has_pii is False]
    assert len(negatives) >= 3_000
    # The 40 entities-rated-none rows are dropped, not counted either way.
    dropped = [r for r in rows if r.doc_has_pii is None]
    assert any(r.doc_gold_source == "ambiguous_entities_rated_none" for r in dropped)
    assert all(r.doc_gold_source == "judge_asserted_absence" for r in negatives)
    # ...and they remain positive-only for the TAG question.
    assert not any(r.tag_labels_complete for r in rows)


def test_manifest_label_field_is_not_used() -> None:
    """`label` is a document-type verdict, orthogonal to PII presence.

    Guards the exact defect this lineage exists to correct: on govdocs2 eval
    there are `label=positive` documents with no PII entity and
    `label=negative` documents with one.
    """
    import json

    manifest = json.loads(
        (_dir("6589_govdocs2-dualjudge-eval20-3.53k") / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    pos_without = sum(
        1
        for r in manifest
        if r.get("label") == "positive" and not (r.get("pii_entities") or r.get("pii_classes"))
    )
    neg_with = sum(
        1
        for r in manifest
        if r.get("label") == "negative" and (r.get("pii_entities") or r.get("pii_classes"))
    )
    assert pos_without > 1_000
    assert neg_with > 1_000


def test_collapse_is_idempotent_and_total() -> None:
    once = collapse_tags(["sensitive_pii_given_name", "sensitive_pii_family_name"])
    assert once == ("sensitive_pii_full_name",)
    assert collapse_tags(once) == once
    # No collapse target is itself a collapse source, or the map would not be
    # confluent and gold/prediction collapse could disagree by ordering.
    assert not (set(COLLAPSE.values()) & set(COLLAPSE))


def test_complete_corpus_reports_real_negatives() -> None:
    rows = list(iter_quiet_corpus(_dir("pii2_eval_30k")))
    assert sum(1 for r in rows if r.doc_has_pii is False) > 4_000
    assert all(r.tag_labels_complete for r in rows)
