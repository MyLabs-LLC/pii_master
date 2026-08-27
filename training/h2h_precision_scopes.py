"""Expose per-tag PRECISION as a gateable scope, without touching the evaluator.

`h2h_eval` exports one metric per tag x corpus scope -- `recall` -- because on most
of this suite's corpora the tag gold is positive-only, where a precision figure is
meaningless. That is the right default, and it is why every policy in this lineage
gates recall and merely reports precision.

It also makes "80% precision across the board" ungateable: a hard constraint on a
metric no scope carries is NOT_MEASURABLE everywhere, which blocks every arm and
says nothing about any of them.

The numbers exist. `evaluate_corpus` already writes `tp`/`fp`/`fn`/`precision` per
tag per corpus into the arm's `per_tag` block, and `per_corpus[...]
["can_measure_precision"]` already says which corpora may answer the question.
This lifts those into `scopes` so a policy can read them. It does not recompute
anything, does not touch `h2h_eval.py`, and does not alter a single existing
number -- an arm augmented here has the same recall scopes and the same headline
metrics it had before.

## The interval is Wilson, and that is a real difference worth naming

Recall scopes carry a *document bootstrap* lower bound: resample documents, recompute,
take the 2.5th percentile. Per-tag precision has no bootstrap, so a `ci_lower` gate
on it would have nothing to read.

What is available is the binomial structure of precision itself -- tp of (tp+fp)
predictions were right -- so this attaches a **Wilson score lower bound** at the
same 95%. It is a legitimate interval and it is *not* the same estimator as the
recall bound beside it: Wilson conditions on the predictions made, the bootstrap
resamples the documents. Both answer "conclusively above the bar?", by different
routes. A policy that gates both is mixing two interval types and its report
should say so, which is why this note exists rather than a silent `ci_low` key.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

Z = 1.959963984540054      # two-sided 95%


def wilson_lower(tp: int, n: int, z: float = Z) -> float:
    """Lower bound of the Wilson score interval for tp successes in n trials."""
    if n <= 0:
        return 0.0
    p = tp / n
    d = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - margin) / d)


def augment(arm: dict, min_support: int = 30) -> tuple[dict, dict]:
    added = skipped_corpora = 0
    detail: dict[str, int] = {}
    for corpus, tags in arm.get("per_tag", {}).items():
        body = arm.get("per_corpus", {}).get(corpus, {})
        if not body.get("can_measure_precision"):
            skipped_corpora += 1
            continue
        for tag, t in tags.items():
            tp, fp = int(t.get("tp", 0)), int(t.get("fp", 0))
            n = tp + fp
            # No predictions at all: precision is undefined, not zero. A tag that
            # never fires has not been imprecise, and scoring it 0.0 would fail a
            # precision gate for silence.
            if n < min_support:
                continue
            scope = arm["scopes"].setdefault(f"{tag}@{corpus}", {})
            scope["precision"] = {
                "value": tp / n, "ci_low": wilson_lower(tp, n),
                "ci_high": None, "support": n,
                "basis": "wilson_95_on_predictions",
            }
            added += 1
            detail[corpus] = detail.get(corpus, 0) + 1
    return arm, {"scopes_added": added, "corpora_without_precision_gold": skipped_corpora,
                 "per_corpus": detail, "min_support": min_support}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--min-support", type=int, default=30)
    args = ap.parse_args()

    arm = json.loads(args.arm.read_text())
    before = len(arm["scopes"])
    arm, info = augment(arm, args.min_support)
    out = args.out or args.arm
    out.write_text(json.dumps(arm, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{args.arm.name}: {before} scopes -> {len(arm['scopes'])} "
          f"(+{info['scopes_added']} precision)")
    print(f"  corpora whose gold cannot measure precision, skipped: "
          f"{info['corpora_without_precision_gold']}")
    for c, n in sorted(info["per_corpus"].items()):
        print(f"    {c:<48} {n:>4} tags")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
