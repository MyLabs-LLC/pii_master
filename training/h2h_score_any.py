"""Score any 61-label cascade on the eight sealed corpora, through the fixed evaluator.

`h2h_score_bundle` is bound to `pii-head-to-head-v1`'s project directory and its
58-label cache. This takes a model directory and a catalogue and scores whatever
it is handed, so the target-box arms join the existing results on the same terms:
same `h2h_eval.evaluate_corpus`, same corpora, same aggregation.

Latency is CARRIED, not re-measured, and that is a claim rather than a shortcut:
these arms differ from their source only in the 61 float comparison points. The
gate, the weights, the feature hashing and the read window are byte-identical, and
`predict_cascade` scores every head unconditionally, so the per-document work is
the same instruction sequence. A re-measurement would be measuring the same code.
The source's measured p95 is recorded along with this note.
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--cache", type=Path,
                    default=Path("projects/pii-scorecard-60/cache"))
    ap.add_argument("--labels", type=int, default=61)
    ap.add_argument("--name", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--latency", type=Path, default=None,
                    help="a benchmark json to carry; see the module docstring")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cat = retarget_cache(args.cache, args.labels)
    labels = tuple(cat["labels"])

    from training.h2h_eval import assemble_arm, evaluate_corpus  # noqa: E402
    from training.h2h_priority import eval_corpora  # noqa: E402
    from training.h2h_score import _load_cached, predict_cascade  # noqa: E402
    from training.quiet_model import QuietCascade  # noqa: E402

    model = QuietCascade.load(args.model)
    lat = json.loads(args.latency.read_text()) if args.latency else {}
    print(f"scoring {args.model} ({len(labels)} labels)", flush=True)

    per_corpus = {}
    for seed, corpus in enumerate(eval_corpora()):
        t0 = time.perf_counter()
        cached = _load_cached(corpus, "deep")
        fired, fired_doc, tag_scores = predict_cascade(model, cached["X"])
        per_corpus[corpus] = evaluate_corpus(
            corpus, fired, fired_doc, cached["Y"], cached["tag_complete"],
            cached["doc_target"], labels, seed=1000 * (seed + 1),
            tag_scores=tag_scores)
        print(f"  {corpus:<48s} {cached['X'].shape[0]:>7,} docs  "
              f"{time.perf_counter() - t0:5.1f}s", flush=True)

    arm = assemble_arm(
        name=args.name, label=args.label or str(args.model), per_corpus=per_corpus,
        p95_latency_ms=lat.get("p95_ms"), docs_per_s=lat.get("docs_per_s"),
        extra={"model": str(args.model), "n_labels": len(labels),
               "latency_source": str(args.latency) if args.latency else None,
               "latency_note": ("carried from the source model: this arm differs only "
                                "in threshold values, so the per-document work is "
                                "identical")})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(arm, indent=1, sort_keys=True) + "\n",
                        encoding="utf-8")

    v = lambda a, k: (a["metrics"][k]["value"] if isinstance(a["metrics"][k], dict)  # noqa: E731
                      else a["metrics"][k])
    print(f"\n{'metric':<34}{args.name:>20}")
    for k in ("micro_f1", "precision_micro", "macro_f05", "macro_f2",
              "recall_macro_catalogue", "precision_macro_catalogue",
              "severity_recall_min", "prediction_rate",
              "equal_corpus_doc_recall", "equal_corpus_doc_precision"):
        val = v(arm, k)
        if val is not None:
            print(f"{k:<34}{val:>20.4f}")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
