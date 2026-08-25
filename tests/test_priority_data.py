from __future__ import annotations

import json
from pathlib import Path

from training.priority_data import PRIORITY_TAGS, iter_corpus, normalized_text_digest


def test_priority_catalog_has_requested_tags() -> None:
    assert len(PRIORITY_TAGS) == 16
    assert "sensitive_pii_full_name" in PRIORITY_TAGS
    assert "sensitive_pii_address" in PRIORITY_TAGS


def test_catalog_rows_are_complete_and_paths_resolve(tmp_path: Path) -> None:
    corpus = tmp_path / "catalog_train"
    (corpus / "documents").mkdir(parents=True)
    (corpus / "documents" / "one.txt").write_text("SSN 123-45-6789", encoding="utf-8")
    (corpus / "labels.jsonl").write_text(
        json.dumps(
            {
                "uid": "one",
                "file": "documents/one.txt",
                "gold": ["sensitive_pii_social_security_number"],
                "split": "train",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (corpus / "manifest.json").write_text("{}\n", encoding="utf-8")
    rows = list(iter_corpus(corpus))
    assert rows[0].label_complete is True
    assert Path(rows[0].path).is_file()


def test_coarse_rows_are_positive_unlabelled(tmp_path: Path) -> None:
    corpus = tmp_path / "coarse_train"
    corpus.mkdir()
    (corpus / "one.txt").write_text("Name: Ada Lovelace", encoding="utf-8")
    (corpus / "manifest.json").write_text(
        json.dumps([{"doc_id": "one", "path": "one.txt", "pii_entities": ["Full Name"]}]),
        encoding="utf-8",
    )
    row = next(iter_corpus(corpus))
    assert row.labels == ("sensitive_pii_full_name",)
    assert row.label_complete is False


def test_digest_normalizes_space_and_case() -> None:
    assert normalized_text_digest("SSN  123") == normalized_text_digest("ssn\n123")
