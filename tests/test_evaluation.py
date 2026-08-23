import pytest

from pii_master.evaluation import (
    CorpusDoc,
    GoldEntity,
    evaluate,
    load_corpus,
    parse_markup,
)


def test_parse_markup_offsets():
    text, entities = parse_markup("a [[X:bc]] d [[Y:e]]")
    assert text == "a bc d e"
    assert entities == [
        GoldEntity("X", 2, 4, "bc"),
        GoldEntity("Y", 7, 8, "e"),
    ]
    for e in entities:
        assert text[e.start : e.end] == e.text


def test_parse_markup_plain_text_passthrough():
    text, entities = parse_markup("no markers here")
    assert text == "no markers here"
    assert entities == []


def test_evaluate_exact_hit():
    docs = [
        CorpusDoc(
            "d1", "PII", "The form lists SSN 123-45-6789 today.",
            [GoldEntity("SSN", 19, 30, "123-45-6789")],
        )
    ]
    report = evaluate(docs)
    assert report.exact["SSN"].tp == 1
    assert report.exact["SSN"].fp == 0
    assert report.exact["SSN"].fn == 0
    assert report.doc_accuracy == 1.0


def test_evaluate_person_name_is_fn_under_rules():
    """Fast mode has no name detector; PERSON_NAME gold is still a miss."""
    docs = [
        CorpusDoc(
            "d1", "PII", "Applicant Jane Doe, SSN 123-45-6789.",
            [
                GoldEntity("PERSON_NAME", 10, 18, "Jane Doe"),
                GoldEntity("SSN", 24, 35, "123-45-6789"),
            ],
        )
    ]
    report = evaluate(docs)
    assert report.exact["PERSON_NAME"].fn == 1
    assert report.exact["PERSON_NAME"].recall == 0.0
    assert report.exact["SSN"].tp == 1


def test_partial_credits_overlap_but_exact_does_not():
    docs = [
        CorpusDoc(
            "d1", "PII", "The form lists SSN 123-45-6789 today.",
            # Gold span deliberately one char wider than the true span.
            [GoldEntity("SSN", 18, 30, " 123-45-6789")],
        )
    ]
    report = evaluate(docs)
    assert report.exact["SSN"].tp == 0
    assert report.exact["SSN"].fn == 1
    assert report.partial["SSN"].tp == 1
    assert report.partial["SSN"].fn == 0


def test_doc_confusion_and_phi_recall():
    docs = [
        CorpusDoc("d1", "PHI", "Patient MRN: 4829471 admitted.",
                  [GoldEntity("MRN", 13, 20, "4829471")]),
        CorpusDoc("d2", "NONE", "Nothing to see in this memo.", []),
        # Gold PII, but the known bare-10-digit FP class makes it interesting:
        CorpusDoc("d3", "PII", "Call me at 4155552671 soon.",
                  [GoldEntity("PHONE_US", 11, 21, "4155552671")]),
    ]
    report = evaluate(docs)
    assert report.confusion["PHI"]["PHI"] == 1
    assert report.confusion["NONE"]["NONE"] == 1
    assert report.phi_recall == 1.0
    assert report.doc_count == 3
    payload = report.to_dict()
    assert payload["documents"] == 3
    assert "SSN" not in payload["span_exact"] or payload["span_exact"]["SSN"]["fp"] == 0


def test_render_is_text(tmp_path):
    docs = [CorpusDoc("d1", "NONE", "clean memo", [])]
    out = evaluate(docs).render()
    assert "Document-level" in out
    assert "PHI recall" in out


def test_error_taxonomy_classes():
    docs = [
        # context_miss: PERSON_NAME is first-class but rules emit nothing
        CorpusDoc("d1", "PII", "Applicant Jane Doe applied.",
                  [GoldEntity("PERSON_NAME", 10, 18, "Jane Doe")]),
        # boundary: right type, gold span one char wider
        CorpusDoc("d2", "PII", "The form lists SSN 123-45-6789 today.",
                  [GoldEntity("SSN", 18, 30, " 123-45-6789")]),
        # context_miss: gold span of a detectable type we emit nothing for
        CorpusDoc("d3", "PII", "Reference 999 on file.",
                  [GoldEntity("SSN", 10, 13, "999")]),
        # spurious: we emit an email the gold does not have
        CorpusDoc("d4", "NONE", "write to a@b.com please", []),
    ]
    hist = evaluate(docs).error_histogram
    assert hist["undetectable"] == 0
    assert hist["boundary"] >= 1
    assert hist["context_miss"] == 2
    assert hist["spurious"] == 1


def test_compare_scores_flags_only_drops():
    from pii_master.evaluation import compare_scores

    base = {"doc_accuracy": 1.0, "phi_recall": 1.0,
            "span_exact": {"SSN": {"precision": 1.0, "recall": 1.0, "f1": 1.0}}}
    assert compare_scores(base, base) == []

    better = {"doc_accuracy": 1.0, "phi_recall": 1.0,
              "span_exact": {"SSN": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
                             "EMAIL": {"precision": 1.0, "recall": 1.0, "f1": 1.0}}}
    assert compare_scores(better, base) == []  # improvements are never failures

    worse = {"doc_accuracy": 0.9, "phi_recall": 1.0,
             "span_exact": {"SSN": {"precision": 1.0, "recall": 0.5, "f1": 0.66}}}
    drops = compare_scores(worse, base)
    assert any("doc_accuracy" in d for d in drops)
    assert any("SSN.recall" in d for d in drops)

    missing = {"doc_accuracy": 1.0, "phi_recall": 1.0, "span_exact": {}}
    assert any("missing" in d for d in compare_scores(missing, base))


def test_committed_scores_baseline_matches_frozen_corpus():
    """The committed gate file must reflect the current pipeline."""
    import json
    from pathlib import Path

    from pii_master.evaluation import compare_scores

    root = Path(__file__).resolve().parent.parent
    scores_path = root / "eval" / "corpus" / "frozen_v1.scores.json"
    assert scores_path.exists(), "run: pii-master eval ... --save-scores"
    baseline = json.loads(scores_path.read_text(encoding="utf-8"))
    current = evaluate(load_corpus(sorted((root / "eval" / "corpus").glob("*.jsonl")))).scores()
    assert compare_scores(current, baseline) == []


def test_crosswalk_partitions_the_nemotron_label_space():
    """Mapped and unmodelled must be disjoint, and cover every known label."""
    from pii_master.crosswalk import (
        ALL_UNMODELLED,
        NEMOTRON_TO_ENTITY,
        to_entity_type,
    )
    from pii_master.entities import EntityType

    assert not (set(NEMOTRON_TO_ENTITY) & ALL_UNMODELLED)
    # The dataset had exactly 55 labels when surveyed (docs/NEMOTRON_PII_TAGS.md).
    assert len(NEMOTRON_TO_ENTITY) + len(ALL_UNMODELLED) == 55
    assert all(isinstance(v, EntityType) for v in NEMOTRON_TO_ENTITY.values())
    assert to_entity_type("email") is EntityType.EMAIL
    assert to_entity_type("first_name") is EntityType.PERSON_NAME
    assert to_entity_type("last_name") is EntityType.PERSON_NAME
    assert to_entity_type("street_address") is EntityType.ADDRESS
    assert to_entity_type("user_name") is EntityType.USERNAME
    assert to_entity_type("race_ethnicity") is None
    # A new dataset label must fail loudly, not become silent background.
    with pytest.raises(KeyError):
        to_entity_type("some_new_label_v2")
