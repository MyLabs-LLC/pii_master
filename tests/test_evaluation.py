from pii_master.evaluation import (
    CorpusDoc,
    GoldEntity,
    evaluate,
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
