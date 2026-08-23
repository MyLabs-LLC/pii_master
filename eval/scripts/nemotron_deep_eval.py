"""Score the SHIPPED deep cascade against the Nemotron-PII holdout.

`training/eval_student.py` scores the student in PyTorch, on GPU, through
training-side decoding, against a pipeline assembled in that script. This
scores `pii_master.pipeline.deep_pipeline()` -- the ONNX bundle, the shipped
decoder, the shipped guards and the shipped fusion tiers -- on one CPU core, on
data nobody in this repo authored. When the two disagree, this one is the
number that describes the product.

**Span merging.** Nemotron tags "Jane Doe" as `first_name` + `last_name` and
"44 Elm Street, Springfield" as `street_address` + `city`. Both collapse to one
of our types, and the detector rejoins them (`ner.merge_adjacent`) so a report
shows one entity per real-world identifier. Scoring merged predictions against
unmerged gold would count every correct full name as one miss and one false
positive, which measures the label convention rather than the model. So the
SAME merge function is applied to gold. `--no-merge` turns it off on both
sides; the difference between the two runs is exactly the size of the
convention gap, and both are reported in docs/STAGE2_INTEGRATION.md.

Usage:

    python eval/scripts/nemotron_deep_eval.py \\
        --data-dir ~/nemotron --model-dir training/artifacts/bundle \\
        --limit 10000 --sweep 0.0,0.3,0.5,0.7,0.9
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pii_master.crosswalk import to_entity_type  # noqa: E402
from pii_master.entities import MODEL_ONLY_TYPES, EntityType  # noqa: E402
from pii_master.evaluation import TypeScore  # noqa: E402
from pii_master.ner import OnnxNerDetector, merge_adjacent  # noqa: E402
from pii_master.pipeline import Pipeline  # noqa: E402
from pii_master.detectors import default_detectors  # noqa: E402

MODEL_TIER = {t.value for t in MODEL_ONLY_TYPES}


def parse_spans(raw):
    if raw is None:
        return []
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return ast.literal_eval(raw)


def read_split(data_dir, split, limit=None):
    import pyarrow.parquet as pq

    files = sorted(Path(data_dir).glob(f"{split}-*.parquet"))
    if not files:
        raise SystemExit(f"no {split} parquet under {data_dir}")
    table = pq.read_table(files[0], columns=["text", "spans"])
    texts = table.column("text").to_pylist()
    spans = table.column("spans").to_pylist()
    if limit:
        texts, spans = texts[:limit], spans[:limit]
    return texts, spans


def merge_like_the_detector(spans, text):
    """Apply ner.merge_adjacent to (type, start, end) triples.

    merge_adjacent works on the detector's internal (kind_index, start, end,
    confidence) tuples, so gold is translated into that shape and back. Using
    the shipped function rather than a reimplementation is the point: if the
    merge rule changes, gold and predictions change together.
    """
    kinds = sorted({t for t, _, _ in spans})
    index = {t: i for i, t in enumerate(kinds)}
    packed = sorted(((index[t], s, e, 1.0) for t, s, e in spans),
                    key=lambda x: (x[1], x[2]))
    typed = tuple(EntityType(k) for k in kinds)
    return [(kinds[k], s, e) for k, s, e, _ in merge_adjacent(packed, text, typed)]


def tally(scores, gold_spans, pred_spans, partial=None):
    gold, pred = defaultdict(set), defaultdict(set)
    for t, a, b in gold_spans:
        gold[t].add((a, b))
    for t, a, b in pred_spans:
        pred[t].add((a, b))
    for t in set(gold) | set(pred):
        g, p = gold.get(t, set()), pred.get(t, set())
        sc = scores[t]
        sc.gold += len(g)
        sc.tp += len(g & p)
        sc.fp += len(p - g)
        sc.fn += len(g - p)
        if partial is None:
            continue
        remaining, hits = sorted(p), 0
        for gs, ge in sorted(g):
            for i, (ps, pe) in enumerate(remaining):
                if gs < pe and ps < ge:
                    hits += 1
                    del remaining[i]
                    break
        sp = partial[t]
        sp.gold += len(g)
        sp.tp += hits
        sp.fp += len(p) - hits
        sp.fn += len(g) - hits


def micro(scores):
    tp = sum(s.tp for s in scores.values())
    fp = sum(s.fp for s in scores.values())
    fn = sum(s.fn for s in scores.values())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def split_tiers(spans):
    return ([s for s in spans if s[0] not in MODEL_TIER],
            [s for s in spans if s[0] in MODEL_TIER])


def score(texts, gold_by_doc, pipeline, label, merge):
    rule_exact, model_exact = defaultdict(TypeScore), defaultdict(TypeScore)
    rule_loose, model_loose = defaultdict(TypeScore), defaultdict(TypeScore)
    latency = []
    for text, gold in zip(texts, gold_by_doc):
        t0 = time.perf_counter()
        entities = pipeline.run(text)
        latency.append((time.perf_counter() - t0) * 1000)
        pred = [(e.type.value, e.start, e.end) for e in entities]
        if merge:
            pred = merge_like_the_detector(pred, text)
        g_rule, g_model = split_tiers(gold)
        p_rule, p_model = split_tiers(pred)
        tally(rule_exact, g_rule, p_rule, rule_loose)
        tally(model_exact, g_model, p_model, model_loose)
    latency.sort()
    return {
        "label": label,
        "rule_tier": micro(rule_exact),
        "rule_tier_partial": micro(rule_loose),
        "model_tier": micro(model_exact),
        "model_tier_partial": micro(model_loose),
        "per_type": {t: s.to_dict() for t, s in
                     {**rule_exact, **model_exact}.items()},
        "p95_ms": latency[max(0, round(0.95 * len(latency)) - 1)],
        "mean_ms": sum(latency) / len(latency),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=10_000)
    ap.add_argument("--sweep", default="0.0,0.3,0.5,0.7,0.9",
                    help="comma-separated min_confidence values to try")
    ap.add_argument("--no-merge", action="store_true")
    ap.add_argument("--json-out")
    args = ap.parse_args(argv)

    merge = not args.no_merge
    texts, raw_spans = read_split(args.data_dir, args.split, args.limit)
    gold_by_doc, dropped = [], Counter()
    for text, raw in zip(texts, raw_spans):
        gold = []
        for span in parse_spans(raw):
            entity = to_entity_type(span["label"])
            if entity is None:
                dropped[span["label"]] += 1
                continue
            gold.append((entity.value, span["start"], span["end"]))
        gold_by_doc.append(merge_like_the_detector(gold, text) if merge else gold)

    total_gold = sum(len(g) for g in gold_by_doc)
    print(f"{len(texts):,} {args.split} documents, {total_gold:,} mapped gold "
          f"spans ({sum(dropped.values()):,} dropped as unmodelled), "
          f"merge={'on' if merge else 'off'}", flush=True)

    runs = [score(texts, gold_by_doc, Pipeline(), "rules only", merge)]
    for threshold in [float(x) for x in args.sweep.split(",")]:
        detector = OnnxNerDetector(args.model_dir, min_confidence=threshold,
                                   merge_adjacent_spans=False)
        pipeline = Pipeline([*default_detectors(), detector])
        runs.append(score(texts, gold_by_doc, pipeline, f"deep @{threshold:.2f}",
                          merge))
        print(f"  scored {runs[-1]['label']}", flush=True)

    print(f"\n{'configuration':>16} | {'rule-tier types':^25} | "
          f"{'model-tier types':^25} | latency")
    print(f"{'':>16} | {'P':>7} {'R':>7} {'F1':>7}  | "
          f"{'P':>7} {'R':>7} {'F1':>7}  | {'p95':>8}")
    for run in runs:
        rp, rr, rf = run["rule_tier"]
        mp, mr, mf = run["model_tier"]
        print(f"{run['label']:>16} | {rp:>7.3f} {rr:>7.3f} {rf:>7.3f}  | "
              f"{mp:>7.3f} {mr:>7.3f} {mf:>7.3f}  | {run['p95_ms']:>6.2f}ms")

    best = max(runs[1:], key=lambda r: r["model_tier"][2] + r["rule_tier"][2])
    print(f"\nbest combined F1: {best['label']}")
    print(f"\nPer-type for {best['label']} (gold / P / R / F1):")
    print(f"{'type':>20} {'gold':>9} {'P':>7} {'R':>7} {'F1':>7}  tier")
    for name in sorted(best["per_type"]):
        row = best["per_type"][name]
        tier = "model" if name in MODEL_TIER else "rule"
        print(f"{name:>20} {row['gold']:>9,} {row['precision']:>7.3f} "
              f"{row['recall']:>7.3f} {row['f1']:>7.3f}  {tier}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"documents": len(texts), "merge": merge, "runs": runs}, indent=2))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
