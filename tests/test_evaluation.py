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


def test_evaluate_undetectable_type_counts_as_fn():
    docs = [
        CorpusDoc(
            "d1", "PII", "Biometric id AB12 on file, SSN 123-45-6789.",
            [
                GoldEntity("BIOMETRIC_ID", 13, 17, "AB12"),
                GoldEntity("SSN", 31, 42, "123-45-6789"),
            ],
        )
    ]
    report = evaluate(docs)
    assert report.exact["BIOMETRIC_ID"].fn == 1
    assert report.exact["BIOMETRIC_ID"].recall == 0.0
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
    assert "leakage" in out


def test_error_taxonomy_classes():
    docs = [
        # undetectable: no rule can emit BIOMETRIC_ID
        CorpusDoc("d1", "PII", "Biometric id AB12 on file.",
                  [GoldEntity("BIOMETRIC_ID", 13, 17, "AB12")]),
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
    assert hist["undetectable"] == 1
    assert hist["boundary"] >= 1
    assert hist["context_miss"] == 1
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
    assert to_entity_type("race_ethnicity") is None
    # A new dataset label must fail loudly, not become silent background.
    with pytest.raises(KeyError):
        to_entity_type("some_new_label_v2")


def test_f2_weights_recall_four_times_as_heavily_as_precision():
    """F1 is the wrong headline for a scanner whose cost matrix is lopsided.

    docs/DESIGN.md section 1: a missed identifier is a reportable incident, a
    false alarm costs a reviewer minutes. F1 prices those the same; F2 does
    not. Both are reported, because the gap between them is the tradeoff a
    confidence threshold is choosing.
    """
    from pii_master.evaluation import TypeScore

    # 10 gold, 8 found, 2 of those wrong -> P 0.80, R 0.80. Balanced, so the
    # two agree exactly; this pins the shared formula.
    balanced = TypeScore(gold=10, tp=8, fp=2, fn=2)
    assert balanced.f1 == pytest.approx(0.8)
    assert balanced.f2 == pytest.approx(0.8)

    # High recall, low precision: F2 must be the kinder of the two.
    recall_heavy = TypeScore(gold=10, tp=10, fp=10, fn=0)
    assert recall_heavy.precision == pytest.approx(0.5)
    assert recall_heavy.recall == pytest.approx(1.0)
    assert recall_heavy.f1 == pytest.approx(2 / 3)
    assert recall_heavy.f2 == pytest.approx(5 / 6)
    assert recall_heavy.f2 > recall_heavy.f1

    # High precision, low recall: F2 must be the harsher of the two, because
    # missing five of ten identifiers is the expensive error here.
    precision_heavy = TypeScore(gold=10, tp=5, fp=0, fn=5)
    assert precision_heavy.f1 == pytest.approx(2 / 3)
    assert precision_heavy.f2 == pytest.approx(5 / 9)
    assert precision_heavy.f2 < precision_heavy.f1


def test_fbeta_is_the_one_formula_behind_both():
    from pii_master.evaluation import TypeScore

    score = TypeScore(gold=10, tp=7, fp=3, fn=3)
    assert score.fbeta(1.0) == pytest.approx(score.f1)
    assert score.fbeta(2.0) == pytest.approx(score.f2)

    # beta only bites when precision and recall differ -- at P == R every
    # beta returns the same number, which is why the case above cannot test
    # the weighting and this one has to.
    assert score.precision == score.recall
    assert score.fbeta(0.5) == pytest.approx(score.fbeta(2.0))

    lopsided = TypeScore(gold=10, tp=5, fp=0, fn=5)   # P 1.0, R 0.5
    assert lopsided.fbeta(0.5) == pytest.approx(5 / 6)   # favours precision
    assert lopsided.fbeta(2.0) == pytest.approx(5 / 9)   # favours recall
    assert lopsided.fbeta(0.5) > lopsided.fbeta(2.0)

    assert TypeScore().fbeta(2.0) == 0.0          # no gold, no predictions


def test_f2_is_gated_by_fail_under_like_every_other_metric():
    from pii_master.evaluation import compare_scores

    baseline = {"doc_accuracy": 1.0, "phi_recall": 1.0,
                "span_exact": {"SSN": {"precision": 1.0, "recall": 1.0,
                                       "f1": 1.0, "f2": 1.0}}}
    worse = {"doc_accuracy": 1.0, "phi_recall": 1.0,
             "span_exact": {"SSN": {"precision": 1.0, "recall": 0.8,
                                    "f1": 0.89, "f2": 0.83}}}
    drops = compare_scores(worse, baseline)
    assert any("SSN.f2" in line for line in drops), drops


def test_document_leakage_counts_a_missed_detectable_span():
    """TAB / PRIOR_ART: one missed identifier makes the document unsafe."""
    leaked = [
        CorpusDoc("d1", "PII", "The form lists SSN 123-45-6789 today.",
                  [GoldEntity("SSN", 19, 30, "123-45-6789"),
                   GoldEntity("EMAIL", 0, 3, "The")]),
    ]
    report = evaluate(leaked)
    assert report.docs_with_gold == 1
    assert report.docs_leaked == 1
    assert report.doc_leakage_rate == 1.0

    clean = [
        CorpusDoc("d1", "PII", "The form lists SSN 123-45-6789 today.",
                  [GoldEntity("SSN", 19, 30, "123-45-6789")]),
    ]
    assert evaluate(clean).doc_leakage_rate == 0.0

    # A model-only type does not count as leakage on the rules path.
    excused = [
        CorpusDoc("d1", "PII", "The form lists SSN 123-45-6789 today.",
                  [GoldEntity("SSN", 19, 30, "123-45-6789"),
                   GoldEntity("BIOMETRIC_ID", 4, 8, "form")]),
    ]
    assert evaluate(excused).doc_leakage_rate == 0.0


def test_leakage_rise_is_a_regression_drop_is_not():
    from pii_master.evaluation import compare_scores

    base = {"doc_accuracy": 1.0, "phi_recall": 1.0, "doc_leakage_rate": 0.1,
            "span_exact": {}}
    worse = {**base, "doc_leakage_rate": 0.4}
    better = {**base, "doc_leakage_rate": 0.0}
    assert any("doc_leakage_rate" in line for line in compare_scores(worse, base))
    assert compare_scores(better, base) == []


def test_a_baseline_written_before_f2_existed_still_works():
    # The committed scores file is regenerated deliberately, not implicitly.
    # Until someone does, --fail-under must keep gating what the old file has.
    from pii_master.evaluation import compare_scores

    old_style = {"doc_accuracy": 1.0, "phi_recall": 1.0,
                 "span_exact": {"SSN": {"precision": 1.0, "recall": 1.0,
                                        "f1": 1.0}}}
    current = {"doc_accuracy": 1.0, "phi_recall": 1.0,
               "span_exact": {"SSN": {"precision": 1.0, "recall": 1.0,
                                      "f1": 1.0, "f2": 1.0}}}
    assert compare_scores(current, old_style) == []
