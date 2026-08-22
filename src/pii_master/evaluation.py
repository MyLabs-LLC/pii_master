"""Frozen-corpus evaluation: span-level P/R/F1 per type + document metrics.

Corpus format (JSONL, one document per line):

    {"id": "phi-001", "label": "PHI", "text": "Patient [[MRN:4829471]] ..."}

Gold spans are authored as inline ``[[TYPE:content]]`` markup; the loader
strips the markup and computes character offsets in the stripped text, so
gold offsets can never drift out of sync with the document. Gold types may
include entity types the system cannot detect yet (e.g. PERSON_NAME) —
those show up as measured recall 0, which is the point.

Matching modes:
  exact   — a predicted span is a true positive iff type, start, and end all
            match a gold span.
  partial — type must match and the spans must overlap (each gold and each
            prediction is matched at most once, greedily by position).
Both are reported: boundary sloppiness and type confusion are different bugs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from .classify import scan_text
from .entities import EntityType
from .models import DocumentReport

MARKUP = re.compile(r"\[\[([A-Z0-9_]+):(.*?)\]\]", re.DOTALL)

DOC_LABELS = ("NONE", "PII", "PHI")

# Gold-only types the current system cannot emit; kept in the corpus so
# Stage 2's job is a measured number, not a footnote.
FUTURE_TYPES = frozenset({"PERSON_NAME", "ADDRESS"})

KNOWN_TYPES = frozenset(t.value for t in EntityType) | FUTURE_TYPES


@dataclass(frozen=True)
class GoldEntity:
    type: str
    start: int
    end: int
    text: str


@dataclass
class CorpusDoc:
    id: str
    label: str
    text: str
    entities: list[GoldEntity]


def parse_markup(marked: str) -> tuple[str, list[GoldEntity]]:
    """Strip [[TYPE:content]] markers, returning plain text and gold spans."""
    parts: list[str] = []
    entities: list[GoldEntity] = []
    pos = 0
    plain_len = 0
    for match in MARKUP.finditer(marked):
        head = marked[pos : match.start()]
        parts.append(head)
        plain_len += len(head)
        content = match.group(2)
        entities.append(
            GoldEntity(match.group(1), plain_len, plain_len + len(content), content)
        )
        parts.append(content)
        plain_len += len(content)
        pos = match.end()
    parts.append(marked[pos:])
    return "".join(parts), entities


def load_corpus(paths: Iterable[str | Path]) -> list[CorpusDoc]:
    docs: list[CorpusDoc] = []
    for path in paths:
        for line_no, line in enumerate(
            Path(path).read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            text, entities = parse_markup(record["text"])
            docs.append(CorpusDoc(record["id"], record["label"], text, entities))
    return docs


@dataclass
class TypeScore:
    gold: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def to_dict(self) -> dict:
        return {
            "gold": self.gold,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


# Error classes, in the DESIGN.md section 10 taxonomy. Each names a
# different fix, which is the point: a regression becomes a ticket.
ERROR_CLASSES = (
    "undetectable",     # gold type no detector can emit yet (Stage 2 work)
    "boundary",         # right type, wrong span edges
    "type_confusion",   # overlapping span, wrong type
    "context_miss",     # nothing emitted where gold has a span
    "spurious",         # emitted a span where gold has nothing
)


@dataclass
class EvalReport:
    exact: dict[str, TypeScore] = field(default_factory=dict)
    partial: dict[str, TypeScore] = field(default_factory=dict)
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)
    doc_count: int = 0
    doc_correct: int = 0
    mislabeled: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)

    @property
    def doc_accuracy(self) -> float:
        return self.doc_correct / self.doc_count if self.doc_count else 0.0

    @property
    def phi_recall(self) -> float:
        gold_phi = sum(self.confusion.get("PHI", {}).values())
        return self.confusion.get("PHI", {}).get("PHI", 0) / gold_phi if gold_phi else 0.0

    @property
    def error_histogram(self) -> dict[str, int]:
        counts = {k: 0 for k in ERROR_CLASSES}
        for err in self.errors:
            counts[err["class"]] = counts.get(err["class"], 0) + 1
        return counts

    def to_dict(self) -> dict:
        return {
            "documents": self.doc_count,
            "doc_accuracy": round(self.doc_accuracy, 4),
            "phi_recall": round(self.phi_recall, 4),
            "confusion": self.confusion,
            "mislabeled": self.mislabeled,
            "error_histogram": self.error_histogram,
            "errors": self.errors,
            "span_exact": {t: s.to_dict() for t, s in sorted(self.exact.items())},
            "span_partial": {t: s.to_dict() for t, s in sorted(self.partial.items())},
        }

    def scores(self) -> dict:
        """The gate-relevant subset, for --save-scores / --fail-under.

        Deliberately excludes error lists and mislabel details: the gate
        compares quality numbers, not diagnostics.
        """
        return {
            "doc_accuracy": round(self.doc_accuracy, 4),
            "phi_recall": round(self.phi_recall, 4),
            "span_exact": {
                t: {
                    "precision": round(sc.precision, 4),
                    "recall": round(sc.recall, 4),
                    "f1": round(sc.f1, 4),
                }
                for t, sc in sorted(self.exact.items())
            },
        }

    def render(self) -> str:
        lines = []
        for title, scores in (("exact match", self.exact), ("partial match", self.partial)):
            lines.append(f"Span-level ({title})")
            lines.append(
                f"  {'TYPE':<18} {'gold':>4} {'TP':>4} {'FP':>4} {'FN':>4}"
                f" {'P':>6} {'R':>6} {'F1':>6}"
            )
            for t, s in sorted(scores.items()):
                lines.append(
                    f"  {t:<18} {s.gold:>4} {s.tp:>4} {s.fp:>4} {s.fn:>4}"
                    f" {s.precision:>6.2f} {s.recall:>6.2f} {s.f1:>6.2f}"
                )
            lines.append("")
        lines.append("Document-level (rows = gold, columns = predicted)")
        lines.append(f"  {'':<6}" + "".join(f"{p:>6}" for p in DOC_LABELS))
        for gold in DOC_LABELS:
            row = self.confusion.get(gold, {})
            lines.append(
                f"  {gold:<6}" + "".join(f"{row.get(p, 0):>6}" for p in DOC_LABELS)
            )
        lines.append("")
        lines.append(
            f"documents: {self.doc_count}   accuracy: {self.doc_accuracy:.2f}"
            f"   PHI recall: {self.phi_recall:.2f}"
        )
        for m in self.mislabeled:
            lines.append(f"  mislabeled: {m['id']} gold={m['gold']} predicted={m['predicted']}")
        lines.append("")
        lines.append("Error taxonomy (each class names a different fix)")
        histogram = self.error_histogram
        widest = max(histogram.values()) or 1
        for name in ERROR_CLASSES:
            count = histogram.get(name, 0)
            bar = "#" * round(24 * count / widest) if count else ""
            lines.append(f"  {name:<16} {count:>4} {bar}")
        return "\n".join(lines)


def _classify_miss(gold_span: GoldEntity, predicted: list[tuple[str, int, int]]) -> str:
    """Why did we miss this gold span?"""
    if gold_span.type in FUTURE_TYPES:
        return "undetectable"
    overlapping = [
        p for p in predicted if gold_span.start < p[2] and p[1] < gold_span.end
    ]
    if any(p[0] == gold_span.type for p in overlapping):
        return "boundary"
    if overlapping:
        return "type_confusion"
    return "context_miss"


def _classify_spurious(
    pred: tuple[str, int, int], gold: list[GoldEntity]
) -> str:
    """Why did we emit this span?"""
    overlapping = [g for g in gold if g.start < pred[2] and pred[1] < g.end]
    if any(g.type == pred[0] for g in overlapping):
        return "boundary"
    if overlapping:
        return "type_confusion"
    return "spurious"


def _collect_errors(
    doc: "CorpusDoc", predicted: list[tuple[str, int, int]]
) -> list[dict]:
    gold_keys = {(g.type, g.start, g.end) for g in doc.entities}
    pred_keys = set(predicted)
    errors: list[dict] = []
    for g in doc.entities:
        if (g.type, g.start, g.end) in pred_keys:
            continue
        errors.append({
            "doc": doc.id, "kind": "fn", "class": _classify_miss(g, predicted),
            "type": g.type, "text": g.text,
        })
    for p in predicted:
        if p in gold_keys:
            continue
        errors.append({
            "doc": doc.id, "kind": "fp", "class": _classify_spurious(p, doc.entities),
            "type": p[0], "text": doc.text[p[1]:p[2]],
        })
    return errors


def _score_spans(
    gold: list[GoldEntity],
    predicted: list[tuple[str, int, int]],
    exact: dict[str, TypeScore],
    partial: dict[str, TypeScore],
) -> None:
    types = {g.type for g in gold} | {p[0] for p in predicted}
    for t in types:
        g_spans = [(g.start, g.end) for g in gold if g.type == t]
        p_spans = [(p[1], p[2]) for p in predicted if p[0] == t]
        ex = exact.setdefault(t, TypeScore())
        pa = partial.setdefault(t, TypeScore())
        ex.gold += len(g_spans)
        pa.gold += len(g_spans)

        exact_tp = len(set(g_spans) & set(p_spans))
        ex.tp += exact_tp
        ex.fp += len(p_spans) - exact_tp
        ex.fn += len(g_spans) - exact_tp

        # Greedy one-to-one overlap matching, by position.
        unmatched = sorted(p_spans)
        partial_tp = 0
        for gs, ge in sorted(g_spans):
            for i, (ps, pe) in enumerate(unmatched):
                if gs < pe and ps < ge:
                    partial_tp += 1
                    del unmatched[i]
                    break
        pa.tp += partial_tp
        pa.fp += len(p_spans) - partial_tp
        pa.fn += len(g_spans) - partial_tp


def evaluate(
    docs: list[CorpusDoc],
    scan: Callable[[str], DocumentReport] = scan_text,
) -> EvalReport:
    report = EvalReport()
    report.confusion = {g: {p: 0 for p in DOC_LABELS} for g in DOC_LABELS}
    for doc in docs:
        result = scan(doc.text)
        predicted = [(e.type.value, e.start, e.end) for e in result.entities]
        _score_spans(doc.entities, predicted, report.exact, report.partial)
        report.errors.extend(_collect_errors(doc, predicted))
        report.doc_count += 1
        report.confusion[doc.label][result.label.name] += 1
        if result.label.name == doc.label:
            report.doc_correct += 1
        else:
            report.mislabeled.append(
                {"id": doc.id, "gold": doc.label, "predicted": result.label.name}
            )
    return report


def compare_scores(current: dict, baseline: dict, tolerance: float = 1e-6) -> list[str]:
    """Regressions in `current` relative to `baseline`, as readable lines.

    Only drops are reported: an improvement is never a failure, but it does
    not silently become the new floor either -- raising the bar is a
    deliberate --save-scores edit, the same rule as the append-only corpus.
    """
    drops: list[str] = []
    for key in ("doc_accuracy", "phi_recall"):
        was, now = baseline.get(key), current.get(key)
        if was is not None and now is not None and now < was - tolerance:
            drops.append(f"{key}: {was:.4f} -> {now:.4f}")
    for entity_type, base in sorted(baseline.get("span_exact", {}).items()):
        cur = current.get("span_exact", {}).get(entity_type)
        if cur is None:
            drops.append(f"{entity_type}: missing from current run")
            continue
        for metric in ("precision", "recall", "f1"):
            was, now = base.get(metric), cur.get(metric)
            if was is not None and now is not None and now < was - tolerance:
                drops.append(f"{entity_type}.{metric}: {was:.4f} -> {now:.4f}")
    return drops
