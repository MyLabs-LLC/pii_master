"""Document-level question: does this file contain PII/PHI, and what kind?

Everything else in eval/ measures SPANS. This measures the decision a user
actually makes with the tool -- quarantine this document or not -- which is a
different metric with a different failure cost, and it had only ever been
measured on the 39-document frozen corpus. That corpus is a regression test,
not a quality claim.

Gold here is derived from Nemotron's own span annotations, not from our
classifier: a document is *sensitive* if it carries at least one gold span that
crosswalks to a type we model. That is honest -- it comes from annotations
nobody in this repo authored.

What this deliberately does NOT claim to measure is the PII/PHI *split*.
Nemotron has no document labels and no medical-context annotation, so any gold
for that boundary would have to be derived using the same `has_medical_context`
heuristic the classifier uses, and scoring a rule against itself measures
nothing. The split is reported as a distribution, not as an accuracy.

Because Nemotron documents are synthetic PII documents, nearly all of them are
sensitive -- so recall is the number this corpus can speak to and precision is
not. Precision against adversarial negatives is what the frozen corpus is for,
and `--negatives` scores it there in the same run.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pii_master.classify import scan_text  # noqa: E402
from pii_master.crosswalk import to_entity_type  # noqa: E402
from pii_master.entities import DocLabel  # noqa: E402
from pii_master.evaluation import load_corpus  # noqa: E402


def parse_spans(raw):
    if raw is None:
        return []
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return ast.literal_eval(raw)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model-dir")
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--min-confidence", type=float, default=0.5)
    ap.add_argument("--rules-only", action="store_true")
    ap.add_argument("--negatives", default="eval/corpus/frozen_v1.jsonl")
    args = ap.parse_args(argv)

    import pyarrow.parquet as pq

    if args.rules_only:
        scan = scan_text
        mode = "fast (rules only)"
    else:
        from pii_master.pipeline import deep_pipeline
        pipeline = deep_pipeline(args.model_dir,
                                 min_confidence=args.min_confidence)
        scan = lambda text: scan_text(text, pipeline)   # noqa: E731
        mode = f"deep @{args.min_confidence:.2f}"

    table = pq.read_table(
        sorted(Path(args.data_dir).glob("test-*.parquet"))[0],
        columns=["text", "spans"])
    texts = table.column("text").to_pylist()[:args.limit]
    raws = table.column("spans").to_pylist()[:args.limit]

    tp = fn = sensitive = 0
    predicted = Counter()
    missed_examples = []
    for text, raw in zip(texts, raws):
        gold_sensitive = any(to_entity_type(s["label"]) is not None
                             for s in parse_spans(raw))
        report = scan(text)
        flagged = report.label is not DocLabel.NONE
        predicted[report.label.name] += 1
        if not gold_sensitive:
            continue
        sensitive += 1
        if flagged:
            tp += 1
        else:
            fn += 1
            if len(missed_examples) < 3:
                missed_examples.append(text[:110].replace("\n", " "))

    print(f"Document-level detection — {mode}")
    print(f"  {len(texts):,} Nemotron test documents, {sensitive:,} carry at "
          f"least one gold identifier we model\n")
    print(f"  RECALL on sensitive documents: {tp / sensitive:.4f}"
          f"   ({tp:,} flagged, {fn:,} missed)")
    if missed_examples:
        print("  missed, for example:")
        for example in missed_examples:
            print(f"    {example!r}")
    print("\n  label distribution: "
          + ", ".join(f"{k} {v:,}" for k, v in sorted(predicted.items())))
    print("  (the PII/PHI split is a distribution, not an accuracy -- see the "
          "module docstring)")

    negatives = [d for d in load_corpus([args.negatives]) if d.label == "NONE"]
    if negatives:
        false_alarms = [d for d in negatives
                        if scan(d.text).label is not DocLabel.NONE]
        print(f"\n  Adversarial negatives (frozen corpus, near-miss "
              f"identifiers): {len(negatives)} documents")
        print(f"  FALSE-ALARM RATE: {len(false_alarms) / len(negatives):.4f}"
              f"   ({len(false_alarms)} flagged)")
        for doc in false_alarms[:5]:
            print(f"    {doc.id}: {doc.text[:90]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
