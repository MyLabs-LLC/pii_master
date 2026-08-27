"""Score every model against every evaluation corpus, in one matrix.

One row per **model x corpus**, which is the only shape the Experiment Log's
metric columns can hold and the only shape a reader can compare. Nine models over
nine corpora is 81 measurements; collapsing them into nine model-level averages
would hide exactly the thing this matrix exists to show, which is that the ranking
depends on which corpus you look at.

## The three things that vary, and why each is an argument not a flag

**Label space.** Five models emit the 61-label scorecard catalogue; two emit the
older 58-label collapsed one (`given/family/middle_name` folded into `full_name`,
`street_number_and_name` into `address`). Gold must be read under the space the
model emits or every subtype is a guaranteed miss, so each model is scored against
its own catalogue and its own cache. **A 58-label micro F1 is not comparable with a
61-label one** and is marked as such in the output rather than placed silently in
the same column.

**Architecture.** Two models are fusions — a cascade plus a static-token-embedding
content tagger combined per tag — and need both artifacts plus their fusion rules.
They are scored through the same evaluator all the same.

**Corpus role.** Eight corpora are the sealed suite that decides gates. The ninth,
`Synthetic_PDF_Corpus_v2_1612`, is an out-of-distribution holdout that is
deliberately **not** part of any headline: it lives outside `data/2-eval` and is
reported in its own column. Including it in an equal-corpus average would let one
corpus nobody selected against quietly rewrite every published number.
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

SC = Path("projects/pii-scorecard-60")
TG = Path("projects/pii-target-8070")
CV = Path("projects/pii-content-v5")
HH = Path("projects/pii-head-to-head-v1")
CACHE61 = SC / "cache"
CACHE58 = Path("projects/pii-quiet-alarm/cache")
HOLDOUT = "Synthetic_PDF_Corpus_v2_1612"

#: name, kind, weights/bundle, cache, n_labels, note
MODELS = [
    ("cascade_scorecard61", "cascade", SC / "models/cascade_scorecard61", CACHE61, 61,
     "61-label baseline; thresholds from the group-recall-cap rule"),
    ("cascade_p80r70", "cascade", TG / "models/cascade_p80r70", CACHE61, 61,
     "box P>=0.80 R>=0.70, F0.5 inside"),
    ("cascade_p80r90", "cascade", TG / "models/cascade_p80r90", CACHE61, 61,
     "box P>=0.80 R>=0.90, F0.5 inside"),
    ("cascade_p88r90", "cascade", TG / "models/cascade_p88r90", CACHE61, 61,
     "box P>=0.88 R>=0.90, F0.5 inside — meets the 90/80/80 target"),
    ("cascade_p90r90", "cascade", TG / "models/cascade_p90r90", CACHE61, 61,
     "box P>=0.90 R>=0.90, F0.5 inside"),
    ("cascade_p90r85b1", "cascade", TG / "models/cascade_p90r85b1", CACHE61, 61,
     "box P>=0.90 R>=0.85, F1 inside — the generalisation-first model"),
    ("cascade_p88r90b1", "cascade", TG / "models/cascade_p88r90b1", CACHE61, 61,
     "box P>=0.88 R>=0.90, F1 inside"),
    ("v5d_fused", "fused", CV / "package/stage_pii-content-v5d-fused-v1", CACHE61, 61,
     "cascade_scorecard61 + content tagger, per-tag fusion"),
    ("v5e_fused", "fused", CV / "package/stage_pii-content-v5e-fused-v1", CACHE61, 61,
     "cascade_p80r70 + content tagger, per-tag fusion"),
    ("cascade_balanced_v4", "cascade", HH / "models/cascade_v4", CACHE58, 58,
     "58-label collapsed taxonomy — NOT comparable with the 61-label rows"),
    ("cascade_balanced_v3", "cascade",
     HH / "dist/pii-cascade-balanced-v3/models/model", CACHE58, 58,
     "58-label; two thresholds tuned on the sealed set — see its model card"),
]

KEYS = ("f1_micro", "precision_micro", "recall_micro", "f1_macro_catalogue",
        "f2_macro_catalogue", "f05_macro_catalogue", "recall_macro_catalogue",
        "precision_macro_catalogue", "prediction_rate", "n_rows")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=TG / "evaluations/matrix.json")
    args = ap.parse_args()

    from training.quiet_data import EVAL_ROOT, list_dataset_dirs  # noqa: E402
    sealed = [d.name for d in list_dataset_dirs(EVAL_ROOT)]
    corpora = sealed + [HOLDOUT]
    print(f"{len(MODELS)} models x {len(corpora)} corpora "
          f"({len(sealed)} sealed + 1 holdout) = {len(MODELS) * len(corpora)} "
          f"measurements\n", flush=True)

    results = []
    for name, kind, path, cache, n_labels, note in MODELS:
        cat = retarget_cache(cache, n_labels)
        labels = tuple(cat["labels"])
        from training.h2h_eval import evaluate_corpus  # noqa: E402
        from training.h2h_score import _load_cached, predict_cascade  # noqa: E402
        from training.quiet_model import QuietCascade  # noqa: E402

        t0 = time.perf_counter()
        # A fused bundle keeps its cascade one level down. The content arm is
        # reproduced from cached features and the bundle's own fusion rules, so
        # the static token table is never loaded here — the matrix has no
        # tokenizer dependency and cannot drift from what was packaged.
        cascade = QuietCascade.load(
            path / "models" / "cascade" if kind == "fused" else path)

        for seed, corpus in enumerate(corpora):
            cached = _load_cached(corpus, "deep")
            fired, fired_doc, tag_scores = predict_cascade(cascade, cached["X"])
            if kind == "fused":
                # Reproduce the packaged fusion from its own rules, on cached
                # features, rather than re-deriving anything here.
                fusion = json.loads(
                    (path / "models" / "fusion.json").read_text(encoding="utf-8"))
                picks = fusion["per_tag"]
                head = np.load(path / "models" / "content" / "head.npz")
                W, b, thr = head["W"], head["b"], head["thresholds"]
                mu, sd = head["mu"], head["sd"]
                fp = CV / "cache/features/eval" / f"{corpus}.npz"
                if not fp.exists():
                    raise SystemExit(
                        f"no content features for {corpus}: {fp}. A fused model "
                        f"cannot be scored on a corpus it has no features for, and "
                        f"skipping it would leave a hole the matrix reads as a gap.")
                feats = np.load(fp)
                cont = (((feats["X"] - mu) / sd) @ W.T + b) >= thr
                fused = np.zeros_like(fired)
                for j, tag in enumerate(labels):
                    r = picks.get(tag, "cascade")
                    a, c = fired[:, j], cont[:, j]
                    fused[:, j] = (a if r == "cascade" else c if r == "content"
                                   else (a | c) if r == "or" else (a & c))
                fired = fused
            body = evaluate_corpus(corpus, fired, fired_doc, cached["Y"],
                                   cached["tag_complete"], cached["doc_target"],
                                   labels, seed=1000 * (seed + 1),
                                   tag_scores=tag_scores)
            s = body["summary"]
            results.append({
                "model": name, "kind": kind, "n_labels": n_labels, "note": note,
                "corpus": corpus,
                "role": "holdout (out-of-distribution)" if corpus == HOLDOUT
                        else "sealed",
                "metrics": {k: s[k] for k in KEYS if s.get(k) is not None},
                "n_rows": body["n_rows"],
                "per_tag": body["per_tag"],
            })
        got = sum(1 for r in results if r["model"] == name)
        print(f"  {name:<22} {n_labels} labels  {got}/{len(corpora)} corpora  "
              f"{time.perf_counter() - t0:5.1f}s", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"models": len(MODELS), "corpora": corpora, "sealed": sealed,
         "holdout": HOLDOUT, "results": results}, indent=1), encoding="utf-8")
    print(f"\n{len(results)} measurements -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
