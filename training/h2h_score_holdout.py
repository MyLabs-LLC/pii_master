"""Score a model on the OUT-OF-DISTRIBUTION holdout, after the gate has been decided.

`data/3-holdout/Synthetic_PDF_Corpus_v2_1612` is deliberately not one of the eight
scored corpora. It is a different kind of document — synthetic PDFs with controlled
fields, read from extracted text — and nothing in `data/1-train` resembles it. That
makes it the closest thing this repo has to a genuine out-of-distribution test, and
it is worth more as one corpus reported separately than as a ninth vote diluted
into an equal-corpus average.

**It is scored after selection, never during it.** A corpus that influences which
model is chosen stops being out-of-distribution the moment it does so. The order is:
select on the training carve, gate on the eight sealed corpora, then come here and
find out what the decision was actually worth.

## What it measured the first time, and why that was worth knowing

Run against four already-selected models it reversed the ranking. The baseline
`cascade_scorecard61` scored micro F1 0.5836 here; the three precision-tuned models
that beat it comfortably on the eight scored 0.4992-0.5167. Precision held up
(0.66-0.72) and recall collapsed (0.39-0.52), and five PHI clinical tags -
`medical_condition`, `medical_treatment`, `patient_id_number`, `medication`,
`icd_10` - fired **zero** times across 561-611 documents each.

So the precision gained by raising thresholds was in part a fit to the character of
the eight, and this corpus is where that shows up. Any future claim about a model
generalising should be checked here before it is made.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.h2h_scorecard_rebuild import retarget_cache  # noqa: E402

HOLDOUT = "Synthetic_PDF_Corpus_v2_1612"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--cache", type=Path,
                    default=Path("projects/pii-scorecard-60/cache"))
    ap.add_argument("--labels", type=int, default=61)
    ap.add_argument("--corpus", default=HOLDOUT)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cat = retarget_cache(args.cache, args.labels)
    labels = tuple(cat["labels"])

    from training.h2h_eval import evaluate_corpus  # noqa: E402
    from training.h2h_score import _load_cached, predict_cascade  # noqa: E402
    from training.quiet_model import QuietCascade  # noqa: E402

    model = QuietCascade.load(args.model)
    t0 = time.perf_counter()
    cached = _load_cached(args.corpus, "deep")
    fired, fired_doc, tag_scores = predict_cascade(model, cached["X"])
    body = evaluate_corpus(args.corpus, fired, fired_doc, cached["Y"],
                           cached["tag_complete"], cached["doc_target"], labels,
                           seed=99_991, tag_scores=tag_scores)
    n = cached["X"].shape[0]
    print(f"{args.name} on {args.corpus}: {n:,} documents, "
          f"{time.perf_counter() - t0:.1f}s", flush=True)

    # `evaluate_corpus` nests the aggregates under "summary"; `assemble_arm`
    # flattens them only when it builds a multi-corpus arm, which this is not.
    summary = body.get("summary", {})
    keys = ("f1_micro", "precision_micro", "recall_micro", "f2_macro_catalogue",
            "f05_macro_catalogue", "recall_macro_catalogue",
            "precision_macro_catalogue", "prediction_rate")
    out = {"model": str(args.model), "name": args.name, "corpus": args.corpus,
           "n_documents": n, "split": "out-of-distribution holdout",
           "metrics": {k: summary[k] for k in keys if summary.get(k) is not None},
           "per_tag": body.get("per_tag", {})}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")

    for k in keys:
        v = summary.get(k)
        if v is not None:
            print(f"  {k:<28}{v:>10.4f}")

    # The tags that never fire are the headline finding here, not a footnote.
    dead = [t for t, r in body.get("per_tag", {}).items()
            if r.get("support", 0) >= 30 and r.get("predicted", 0) == 0]
    if dead:
        print(f"\n  {len(dead)} tag(s) with >=30 gold instances that NEVER fire:")
        for t in dead:
            r = body["per_tag"][t]
            print(f"    {t.replace('sensitive_', ''):<46} gold {r['support']:>5}")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
