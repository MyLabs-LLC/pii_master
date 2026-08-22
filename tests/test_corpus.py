"""Validity checks for the frozen evaluation corpus.

The corpus is append-only: these checks guard structure, not scores, so
adding hard cases never requires touching this file.
"""

from pathlib import Path

from pii_master.evaluation import DOC_LABELS, KNOWN_TYPES, load_corpus

CORPUS_DIR = Path(__file__).resolve().parent.parent / "eval" / "corpus"


def corpus_docs():
    paths = sorted(CORPUS_DIR.glob("*.jsonl"))
    assert paths, f"no corpus files found in {CORPUS_DIR}"
    return load_corpus(paths)


def test_corpus_loads_and_is_nonempty():
    docs = corpus_docs()
    assert len(docs) >= 30


def test_ids_unique_and_labels_valid():
    docs = corpus_docs()
    ids = [d.id for d in docs]
    assert len(ids) == len(set(ids))
    assert all(d.label in DOC_LABELS for d in docs)
    # Every label class is represented.
    assert {d.label for d in docs} == set(DOC_LABELS)


def test_gold_spans_align_with_text():
    for doc in corpus_docs():
        assert doc.text.strip(), doc.id
        assert "[[" not in doc.text and "]]" not in doc.text, doc.id
        for e in doc.entities:
            assert e.type in KNOWN_TYPES, (doc.id, e.type)
            assert e.text, (doc.id, e)
            assert doc.text[e.start : e.end] == e.text, (doc.id, e)
