"""Corrected, negative-bearing loaders for the `pii-quiet-alarm` lineage.

Two corrections over :mod:`training.priority_data`, both of which the
precision-first run depends on.

**1. A judge assertion of absence is a negative.**  The dual-judge corpora carry
``pii_entities`` / ``pii_classes`` / ``pii_sensitivity``.  A row whose entity
and class lists are both empty *and* whose sensitivity is the explicit string
``"none"`` is the judges saying they looked and found nothing -- as opposed to
nobody having annotated it.  Those rows, and only those, become negatives.

The reverse implication does **not** hold, and assuming it did would have been a
silent defect: 217 rows across the two govdocs2 directories carry
``sensitivity == "none"`` alongside a non-empty entity list (typically a bare
``State`` or ``ZIP Code``), and 4 carry an empty list with a null sensitivity.
Every such row is **ambiguous** -- reading it as positive would penalise correct
silence, reading it as negative would teach silence over real entities -- so it
is excluded from document-level gold entirely and counted by name.
:func:`assert_absence_contract` re-verifies the classification and caps the
ambiguous share; it is a hard failure, not a warning, because the whole run
rests on it.

``priority_data.normalize_row`` reads only ``gold`` and ``pii_entities``, marks
these corpora ``label_complete=False``, and therefore discards all 20,714
real-world clean documents they contain.  Six of the eight training corpora
hold no labelled-clean document at all, so the model that lineage fitted had
essentially never seen one.

This does **not** turn coarse-labelled corpora into complete ones.  Document
*absence* is now known where a judge asserted it; which of the 61 tags a
positive document carries is still positive-only there, and
``tag_labels_complete`` stays false so no per-tag precision is ever computed
against gold that cannot support it.

**2. The document-level gold field.**  The prior lineage read the manifest's
``label``.  That is the judges' *document-type* verdict and is orthogonal to
PII presence -- on the govdocs2 evaluation set 1,501 ``label=positive``
documents carry no PII entity and 1,220 ``label=negative`` documents do.  It is
not read here at all.

The frozen taxonomy collapse also lives here so that gold and predictions can
never be collapsed by two different maps.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from training.priority_data import (
    PRIORITY_TAGS,
    SENSITIVE_PREFIXES,
    CorpusRow,
    iter_raw_rows,
    normalize_row,
    read_document,
    resolve_document_path,
)

TRAIN_ROOT = Path("/home/lence/workspace/data/1-train")
EVAL_ROOT = Path("/home/lence/workspace/data/2-eval")

#: Corpora whose ``gold`` field is an exhaustive sensitive-tag catalogue, so a
#: row listing no tag is a genuine negative rather than an unannotated one.
#:
#: This is declared, not inferred. ``priority_data.normalize_row`` infers it
#: from ``dataset_dir.name.startswith("pii")`` -- and on 2026-08-25 an external
#: pass renamed every dataset directory to a ``<rows>_<name>_<positives>``
#: convention, so ``pii2_eval_30k`` became ``30000_pii2_eval_25.15k``. The
#: prefix stopped matching, every unlabelled row silently became "unknown"
#: instead of "negative", and 54,812 of the run's 70,600 negatives vanished
#: without raising anything. A property of a corpus must not live in its
#: folder name.
COMPLETE_CATALOGUE_STEMS = frozenset({
    # Added 2026-08-27. Its own `metadata/taxonomy.json` IS the GAIA scorecard's
    # 60 tags, so an absent tag is an asserted absence rather than an unannotated
    # one. Declared here rather than inferred, for the reason this constant exists.
    # NOT added to `h2h_eval.DOC_MEASURABLE_STEMS`: only 20 of its 1,612 documents
    # carry no tag, which is far too few negatives to judge document precision.
    "Synthetic_PDF_Corpus_v2",
    # The 80/20 split of that corpus, now first-class train/eval members.
    # Same taxonomy, same asserted-absence semantics.
    "synthetic_pdf_train", "synthetic_pdf_eval",
    "pii2_train", "pii2_eval",
    "pii_trainset", "pii_holdout",
    "ai4privacy_pii_masking_train", "ai4privacy_pii_masking_eval",
    "betterdataai_ner_silver_train", "betterdataai_ner_silver_eval",
    "openpii_pii_train", "openpii_pii_eval",
})

#: Directories whose rows carry judge entity/class/sensitivity fields.
#: Matched by stem, not by literal name, for the reason in
#: COMPLETE_CATALOGUE_STEMS -- these folder names carry row counts that an
#: external pass rewrites, and a literal match silently stops matching.
JUDGE_ASSERTED = (
    "15986_datax-dualjudge-trainset-5.36k",
    "23693_govdocs2-dualjudge-train80-12.86k",
    "4000_datax-dualjudge-evalset-1.32k",
    "6589_govdocs2-dualjudge-eval20-3.53k",
)

#: The frozen collapse.  Two independent judges agree at F1 0.99 on whether a
#: name is present and at 0.02-0.40 on which name tag it is; scoring that
#: distinction charges a model twice for a disagreement the gold cannot settle.
#: Frozen here, applied identically to gold and to predictions, audited before
#: any model is fitted.
COLLAPSE: dict[str, str] = {
    "sensitive_pii_given_name": "sensitive_pii_full_name",
    "sensitive_pii_family_name": "sensitive_pii_full_name",
    "sensitive_pii_middle_name": "sensitive_pii_full_name",
    "sensitive_pii_street_number_and_name": "sensitive_pii_address",
}


def _stems(names: tuple[str, ...]) -> frozenset[str]:
    return frozenset(canonical_stem(n) for n in names)


def canonical_stem(name: str) -> str:
    """Corpus identity, independent of the size counters in its folder name.

    ``30000_pii2_eval_25.15k`` and ``pii2_eval_30k`` are the same corpus; the
    numbers are bookkeeping that an external pass rewrites. Strips a leading
    row-count token and any trailing size token joined by ``_`` or ``-``.
    """
    def _size(tok: str) -> bool:
        return bool(tok) and tok.rstrip("k").replace(".", "").isdigit()

    parts = [p for p in name.split("_") if p]
    if parts and _size(parts[0]):
        parts = parts[1:]
    if parts and _size(parts[-1]):
        parts = parts[:-1]
    stem = "_".join(parts)
    head, sep, tail = stem.rpartition("-")
    if sep and _size(tail):
        stem = head
    return stem


def resolve_dataset(name: str, root: Path | None = None) -> Path:
    """Find the directory currently holding a corpus, whatever it is called now.

    Canonical names are the run's identity -- the cache, the suite and the
    frozen snapshot are all keyed by them -- so a directory rename must not
    break a run in flight. Matching is on the stem, and an ambiguous match is an
    error rather than a guess.
    """
    stem = canonical_stem(name)
    roots = [root] if root is not None else [TRAIN_ROOT, EVAL_ROOT]
    hits = [d for r in roots if r.is_dir() for d in r.iterdir()
            if d.is_dir() and canonical_stem(d.name) == stem]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise FileNotFoundError(
            f"no directory matches corpus stem {stem!r} under {[str(r) for r in roots]}")
    raise ValueError(f"corpus stem {stem!r} matches {len(hits)} directories: "
                     f"{[d.name for d in hits]}")


#: The same four corpora, keyed by stem so a rename cannot detach them.
JUDGE_ASSERTED_STEMS = _stems(JUDGE_ASSERTED)


def collapse_tags(tags: object) -> tuple[str, ...]:
    """Apply the frozen collapse to any iterable of tag names."""
    return tuple(sorted({COLLAPSE.get(t, t) for t in tags}))


@dataclass(frozen=True)
class QuietRow:
    """A corpus row that knows what its gold can and cannot say.

    ``doc_has_pii`` is the document-level target: ``True`` / ``False`` where the
    corpus can say, ``None`` where silence means "not annotated".
    ``tag_labels_complete`` is the separate, stricter claim that the row's tag
    list is exhaustive over the sensitive catalogue.
    """

    row: CorpusRow
    doc_has_pii: bool | None
    tag_labels_complete: bool
    doc_gold_source: str
    sensitivity: str | None = None

    @property
    def dataset(self) -> str:
        return self.row.dataset

    @property
    def uid(self) -> str:
        return self.row.uid

    @property
    def path(self) -> str:
        return self.row.path

    @property
    def labels(self) -> tuple[str, ...]:
        return self.row.labels

    @property
    def collapsed_labels(self) -> tuple[str, ...]:
        return collapse_tags(self.row.labels)

    @property
    def provenance(self) -> str:
        return self.row.provenance


def _judge_fields(raw: dict[str, Any]) -> tuple[list, list, Any]:
    return (
        raw.get("pii_entities") or [],
        raw.get("pii_classes") or [],
        raw.get("pii_sensitivity"),
    )


def classify_judge_row(raw: dict[str, Any]) -> tuple[bool | None, str]:
    """Document-level verdict for one dual-judge row, and why.

    Three outcomes, never two.  ``None`` is not "probably clean"; it is a row
    whose gold cannot answer the question, and it is dropped rather than
    guessed at.
    """
    entities, classes, sensitivity = _judge_fields(raw)
    empty = not entities and not classes
    if empty and sensitivity == "none":
        return False, "judge_asserted_absence"
    if not empty and sensitivity in ("low", "medium", "high"):
        return True, "judge_entities"
    if not empty and sensitivity == "none":
        # The judge listed something and then rated it non-sensitive -- usually a
        # bare State or ZIP Code. Neither reading is safe; drop it.
        return None, "ambiguous_entities_rated_none"
    if empty:
        return None, "unannotated"
    return None, "ambiguous_other"


#: An ambiguous share above this is a data change, not a rounding detail.
MAX_AMBIGUOUS_SHARE = 0.02


def assert_absence_contract(dataset_dir: Path) -> dict[str, int]:
    """Classify every row and refuse a corpus that has drifted.

    A silent tolerance here would let a document nobody examined be trained on
    as a confirmed negative, which is the one way this correction could make
    the model *worse* at recall.
    """
    counts: Counter[str] = Counter({"positive": 0, "negative": 0, "ambiguous": 0})
    for raw in iter_raw_rows(dataset_dir):
        verdict, reason = classify_judge_row(raw)
        counts["rows"] += 1
        counts[reason] += 1
        counts[{True: "positive", False: "negative", None: "ambiguous"}[verdict]] += 1
    if not counts["negative"]:
        raise ValueError(f"{dataset_dir.name}: no judge-asserted negatives found")
    share = counts["ambiguous"] / counts["rows"]
    if share > MAX_AMBIGUOUS_SHARE:
        raise ValueError(
            f"{dataset_dir.name}: {counts['ambiguous']} of {counts['rows']} rows "
            f"({share:.2%}) are ambiguous, above the {MAX_AMBIGUOUS_SHARE:.0%} cap; "
            "the judge fields have changed shape and the contract needs re-deriving"
        )
    return dict(counts)


def _doc_gold(dataset_dir: Path, raw: dict[str, Any], base: CorpusRow) -> tuple[bool | None, str, str | None]:
    """The document-level target and where it came from."""
    if canonical_stem(dataset_dir.name) in JUDGE_ASSERTED_STEMS:
        verdict, reason = classify_judge_row(raw)
        return verdict, reason, raw.get("pii_sensitivity")
    if base.label_complete:
        # A complete catalogue can say absence directly.
        return bool(base.labels), "complete_catalogue", None
    if base.labels:
        return True, "positive_only", None
    return None, "positive_only", None


def iter_quiet_corpus(dataset_dir: Path) -> Iterator[QuietRow]:
    complete_catalogue = canonical_stem(dataset_dir.name) in COMPLETE_CATALOGUE_STEMS
    for index, raw in enumerate(iter_raw_rows(dataset_dir)):
        base = normalize_row(dataset_dir, raw, index)
        if complete_catalogue and "gold" in raw:
            # Declared, not inferred from the folder name -- see
            # COMPLETE_CATALOGUE_STEMS for what this replaces and why.
            base = replace(base, label_complete=True)
        has, source, sensitivity = _doc_gold(dataset_dir, raw, base)
        yield QuietRow(
            row=base,
            doc_has_pii=has,
            tag_labels_complete=base.label_complete,
            doc_gold_source=source,
            sensitivity=sensitivity,
        )


def list_dataset_dirs(root: Path) -> list[Path]:
    return sorted(d for d in root.iterdir() if d.is_dir())


def summarize(dataset_dir: Path) -> dict[str, Any]:
    """Per-corpus counts used by the data-quality record and the suite."""
    n = 0
    doc_pos = doc_neg = doc_unknown = 0
    tag_complete = 0
    tags = Counter()
    for qr in iter_quiet_corpus(dataset_dir):
        n += 1
        if qr.doc_has_pii is True:
            doc_pos += 1
        elif qr.doc_has_pii is False:
            doc_neg += 1
        else:
            doc_unknown += 1
        tag_complete += qr.tag_labels_complete
        tags.update(qr.collapsed_labels)
    return {
        "dataset": dataset_dir.name,
        "n_rows": n,
        "doc_positive": doc_pos,
        "doc_negative": doc_neg,
        "doc_unknown": doc_unknown,
        "doc_prevalence": doc_pos / (doc_pos + doc_neg) if doc_pos + doc_neg else None,
        "tag_complete_rows": tag_complete,
        "priority_support": {
            t: tags.get(c, 0)
            for t in PRIORITY_TAGS
            for c in (COLLAPSE.get(t, t),)
        },
        "distinct_collapsed_tags": len(tags),
    }


__all__ = [
    "COLLAPSE",
    "EVAL_ROOT",
    "JUDGE_ASSERTED",
    "PRIORITY_TAGS",
    "SENSITIVE_PREFIXES",
    "TRAIN_ROOT",
    "QuietRow",
    "COMPLETE_CATALOGUE_STEMS",
    "assert_absence_contract",
    "canonical_stem",
    "classify_judge_row",
    "resolve_dataset",
    "collapse_tags",
    "iter_quiet_corpus",
    "list_dataset_dirs",
    "read_document",
    "resolve_document_path",
    "summarize",
]
