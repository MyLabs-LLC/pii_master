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
    ap.add_argument("--sweep", help="comma-separated thresholds to try instead "
                                    "of a single --min-confidence")
    ap.add_argument("--json-out", help="write the results here (for model cards)")
    args = ap.parse_args(argv)

    import pyarrow.parquet as pq

    table = pq.read_table(
        sorted(Path(args.data_dir).glob("test-*.parquet"))[0],
        columns=["text", "spans"])
    texts = table.column("text").to_pylist()[:args.limit]
    raws = table.column("spans").to_pylist()[:args.limit]

    gold_flags = [any(to_entity_type(s["label"]) is not None
                      for s in parse_spans(raw)) for raw in raws]
    sensitive = sum(gold_flags)
    negatives = [d for d in load_corpus([args.negatives]) if d.label == "NONE"]

    thresholds = ([float(v) for v in args.sweep.split(",")] if args.sweep
                  else [args.min_confidence])
    results = []
    for threshold in thresholds:
        if args.rules_only:
            scan, mode = scan_text, "fast (rules only)"
        else:
            from pii_master.pipeline import deep_pipeline
            pipeline = deep_pipeline(args.model_dir, min_confidence=threshold)
            scan = lambda text, p=pipeline: scan_text(text, p)   # noqa: E731
            mode = f"deep @{threshold:.2f}"

        tp = fn = 0
        predicted = Counter()
        missed = []
        for text, gold_sensitive in zip(texts, gold_flags):
            report = scan(text)
            predicted[report.label.name] += 1
            if not gold_sensitive:
                continue
            if report.label is not DocLabel.NONE:
                tp += 1
            else:
                fn += 1
                if len(missed) < 3:
                    missed.append(text[:110].replace("\n", " "))
        alarms = sum(1 for d in negatives
                     if scan(d.text).label is not DocLabel.NONE)
        results.append({
            "mode": mode, "threshold": None if args.rules_only else threshold,
            "documents": len(texts), "sensitive": sensitive,
            "recall": tp / sensitive, "flagged": tp, "missed": fn,
            "false_alarm_rate": alarms / len(negatives) if negatives else None,
            "false_alarms": alarms, "negatives": len(negatives),
            "labels": dict(predicted), "missed_examples": missed,
        })
        if args.rules_only:
            break

    print(f"Document-level detection — {len(texts):,} Nemotron test documents, "
          f"{sensitive:,} carry at least one gold identifier we model")
    print(f"  adversarial negatives: {len(negatives)} (frozen corpus, "
          f"near-miss identifiers)\n")
    print(f"  {'configuration':>16} {'recall':>8} {'missed':>7} "
          f"{'false alarms':>13}")
    for r in results:
        alarm = ("n/a" if r["false_alarm_rate"] is None
                 else f"{r['false_alarm_rate']:.3f} ({r['false_alarms']})")
        print(f"  {r['mode']:>16} {r['recall']:>8.4f} {r['missed']:>7,} "
              f"{alarm:>13}")

    last = results[-1]
    if last["missed_examples"]:
        print(f"\n  missed by {last['mode']}, for example:")
        for example in last["missed_examples"]:
            print(f"    {example!r}")
    print("\n  label distribution (" + last["mode"] + "): "
          + ", ".join(f"{k} {v:,}" for k, v in sorted(last["labels"].items())))
    print("  The PII/PHI split is a distribution, not an accuracy -- Nemotron "
          "has no document\n  labels, so there is no external gold for that "
          "boundary. See the module docstring.")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2))
        print(f"\n  wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
