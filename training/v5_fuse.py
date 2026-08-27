"""v5d -- fuse the content tagger with `cascade_scorecard61` and score it sealed.

The two arms answer the same 61-label question from different evidence: the
cascade from hashed character/word n-grams, the content tagger from static token
embeddings distilled out of a fine-tuned transformer. Fusing them is only
worthwhile where they disagree, so the fusion is chosen **per tag** rather than
globally, and the choice is made on the calibration carve — never on the sealed
corpora.

## The strategies, and why the menu is this short

Per tag, one of four:

| | fires when | good when |
| --- | --- | --- |
| `cascade` | the cascade fires | the content model adds nothing (most format-anchored identifiers) |
| `content` | the content tagger fires | the cascade has no signal (contextual tags: names, addresses, geography) |
| `or` | either fires | both are precise and each finds cases the other misses |
| `and` | both fire | both are noisy in different ways — the intersection is the precision play |

`or` trades precision for recall and `and` does the reverse, so with a
precision-led ranker the menu covers the useful corners without a search. The
choice is scored by **F0.5 on the calibration carve**, matching the declared
policy's ranker, and ties break toward `cascade` — the incumbent — so that a
strategy is only adopted where it demonstrably beats doing nothing.

## The document question is fused too, and separately

The same four strategies are applied once more to the **document** decision —
"does this contain sensitive PII at all" — with its own choice made on the
calibration carve.

Leaving it out was the first version of this file and it was a design error worth
naming: the fused arm's document metrics would have been the cascade's by
construction, and `doc_precision`, `doc_specificity` and `doc_recall` are three of
the five hard constraints in the contra-view policy. The content model would have
been structurally unable to move most of what that policy gates on, and the run
would have concluded "no document-level improvement" having never allowed one.

The content tagger has no document head, so its document opinion is **"any enabled
tag fired"** — the same rule the priority-fusion lineage used. That is a stated
convention, not a derived one, and it is the weaker half of this fusion.

## What is held fixed

`cascade_scorecard61` is frozen: it is fused with, not refit. The evaluator, the
catalogue, the suite and both policies are the ones already declared. The sealed
corpora are opened here and only here, once.
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

PROJECT = Path("projects/pii-content-v5")
SCORECARD = Path("projects/pii-scorecard-60")
STRATEGIES = ("cascade", "content", "or", "and")


def fbeta(tp: int, fp: int, fn: int, beta: float = 0.5) -> float:
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    if p + r == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def bench_fused(cascade, featurise, content_fire, picks, doc_best, docs, repeats: int,
                cpus: str = "1") -> dict:
    """The REAL end-to-end fused latency, on one core.

    The hard constraint is on the fused arm, and until now the only figure for it
    was arithmetic: the cascade's measured 4.029 ms plus the content path's
    measured 2.533 ms. Adding two separately-measured numbers is an estimate. It
    omits whatever the two paths cost together -- reading the document once for
    both, the fusion itself -- and a gate constraint that is checked against an
    estimate is not checked.

    So this times the whole serving call: featurise, score both arms, fuse, on the
    same >=10 KB real documents `h2h_bench` uses and with the same one-core budget,
    so the number is comparable with every other latency in this project.
    """
    from model_pipeline import set_cpus
    set_cpus(cpus if cpus == "all" else int(cpus))
    timings = []
    for _ in range(repeats):
        for text in docs:
            t0 = time.perf_counter()
            casc_tags = np.asarray(cascade.predict(text))
            casc_doc = cascade.has_pii(text)
            feat = featurise(text)
            cont = content_fire(feat[None, :])[0]
            casc = np.zeros(len(picks), dtype=bool)
            for j, tag in enumerate(cascade.labels):
                casc[j] = tag in casc_tags
            fused = np.array([apply(s, casc[j], cont[j])
                              for j, s in enumerate(picks)], dtype=bool)
            _ = apply(doc_best, casc_doc, bool(cont.any())), fused
            timings.append((time.perf_counter() - t0) * 1000.0)
    a = np.asarray(timings)
    return {"n": int(a.size), "repeats": repeats, "cpu_budget": cpus,
            "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)),
            "p99_ms": float(np.percentile(a, 99)),
            "mean_ms": float(a.mean()),
            "docs_per_s": float(1000.0 / a.mean())}


def apply(strategy: str, casc: np.ndarray, cont: np.ndarray) -> np.ndarray:
    if strategy == "cascade":
        return casc
    if strategy == "content":
        return cont
    if strategy == "or":
        return casc | cont
    return casc & cont


def choose(casc: np.ndarray, cont: np.ndarray, Y: np.ndarray, labels) -> tuple:
    """Per-tag strategy, by F0.5 on the calibration carve. Ties keep the cascade."""
    picks, table = [], {}
    for j, tag in enumerate(labels):
        best, best_f = "cascade", -1.0
        row = {}
        for s in STRATEGIES:
            f = apply(s, casc[:, j], cont[:, j])
            tp = int((f & Y[:, j]).sum())
            fp = int((f & ~Y[:, j]).sum())
            fn = int((~f & Y[:, j]).sum())
            score = fbeta(tp, fp, fn)
            row[s] = {"f05": score, "tp": tp, "fp": fp, "fn": fn}
            if score > best_f + 1e-9:
                best, best_f = s, score
        picks.append(best)
        table[tag] = {"chosen": best, "by_strategy": row}
    return picks, table


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tagger", type=Path, default=PROJECT / "models/v5c-tagger")
    ap.add_argument("--features", type=Path, default=PROJECT / "cache/features")
    ap.add_argument("--cascade", default="cascade_scorecard61")
    ap.add_argument("--cascade-dir", type=Path, default=None,
                    help="full path to the cascade; overrides --cascade, "
                         "so an arm from another project can be fused into")
    ap.add_argument("--m2v", type=Path, default=PROJECT / "models/v5b-m2v")
    ap.add_argument("--name", default="v5d-fused")
    ap.add_argument("--bench-docs", type=int, default=200)
    ap.add_argument("--bench-repeats", type=int, default=3)
    ap.add_argument("--cpus", default="1")
    args = ap.parse_args()

    cat = retarget_cache(SCORECARD / "cache", 61)
    labels = tuple(cat["labels"])

    from training.h2h_eval import assemble_arm, evaluate_corpus  # noqa: E402
    from training.h2h_priority import eval_corpora  # noqa: E402
    from training.h2h_score import _load_cached, predict_cascade  # noqa: E402
    from training.quiet_model import QuietCascade  # noqa: E402

    cascade = QuietCascade.load(args.cascade_dir
                                or SCORECARD / "models" / args.cascade)
    head = np.load(args.tagger / "head.npz")
    W, b, thr = head["W"], head["b"], head["thresholds"]
    mu, sd = head["mu"], head["sd"]

    def content_fire(X: np.ndarray) -> np.ndarray:
        return (((X - mu) / sd) @ W.T + b) >= thr

    # ------------------------------------------- choose the fusion on calibration
    # Both sides are concatenated in `train_corpora()` order EXPLICITLY, by name.
    # Globbing the feature directory and trusting that its sort matches is the same
    # class of assumption that cost this repo 54,812 negatives when an external
    # rename broke a folder-name prefix (commit 3ce3c8c). It happens to hold today;
    # it is not worth depending on, because the failure is silent and pairs
    # unrelated documents.
    from training.quiet_fit import load, train_corpora  # noqa: E402
    order = train_corpora()
    Xs, Ys, fits = [], [], []
    for name in order:
        path = args.features / "train" / f"{name}.npz"
        if not path.exists():
            raise SystemExit(f"missing content features for {name}: {path}. "
                             f"Run v5_tagger.py --build-only first.")
        with np.load(path) as z:
            Xs.append(z["X"]); Ys.append(z["Y"]); fits.append(z["fit"])
    X = np.concatenate(Xs); Y = np.concatenate(Ys); fit = np.concatenate(fits)
    del Xs, Ys, fits
    calib = ~fit
    print(f"choosing fusion on {int(calib.sum()):,} calibration rows "
          f"({len(order)} corpora, joined by name)", flush=True)

    # The cascade's own decision on the same rows, from its own cached features,
    # loaded in that same explicit order.
    ds = load(order, profile="deep")
    c_fired, _, _ = predict_cascade(cascade, ds.X)
    if len(c_fired) != len(X):
        raise SystemExit(
            f"row-count mismatch: cascade features have {len(c_fired):,} rows and "
            f"content features {len(X):,}. Both were built from {order}; a mismatch "
            f"means one side skipped rows, and fusing them would pair unrelated "
            f"documents.")

    m_calib = content_fire(X[calib])
    picks, table = choose(c_fired[calib], m_calib, Y[calib], labels)
    from collections import Counter
    print("fusion chosen per tag:", dict(Counter(picks)), flush=True)

    # ------------------------------------------------- the document question too
    # Without this the fused arm's document metrics are the cascade's by
    # construction, and doc precision / specificity / recall are three of the five
    # hard constraints in the contra-view policy -- so the content model would be
    # structurally unable to move most of what that policy gates on.
    #
    # The content model has no document head; its document opinion is "any enabled
    # tag fired", which is the same rule the priority-fusion arm used and is stated
    # rather than derived. `doc_target` is 1 / 0 / -1 and only >= 0 can answer.
    _, c_doc_calib, _ = predict_cascade(cascade, ds.X)
    known = ds.doc_target >= 0
    dc = known[calib]
    doc_gold = ds.doc_target[calib][dc].astype(bool)
    doc_pair = {"cascade": c_doc_calib[calib][dc],
                "content": m_calib.any(axis=1)[dc]}
    doc_best, doc_best_f, doc_table = "cascade", -1.0, {}
    for s in STRATEGIES:
        f = apply(s, doc_pair["cascade"], doc_pair["content"])
        tp = int((f & doc_gold).sum()); fp = int((f & ~doc_gold).sum())
        fn = int((~f & doc_gold).sum())
        score = fbeta(tp, fp, fn)
        doc_table[s] = {"f05": score, "tp": tp, "fp": fp, "fn": fn}
        if score > doc_best_f + 1e-9:
            doc_best, doc_best_f = s, score
    print(f"document gate fusion: {doc_best} "
          f"(F0.5 {doc_best_f:.4f} on {int(dc.sum()):,} answerable calibration rows)",
          flush=True)

    (PROJECT / "probe").mkdir(parents=True, exist_ok=True)
    (PROJECT / "probe" / f"{args.name}_fusion.json").write_text(
        json.dumps({"per_tag": table,
                    "document": {"chosen": doc_best, "by_strategy": doc_table,
                                 "n_answerable": int(dc.sum())}},
                   indent=1), encoding="utf-8")

    # --------------------------------------------------------- score the sealed
    per_corpus = {}
    for seed, corpus in enumerate(eval_corpora()):
        t0 = time.perf_counter()
        cached = _load_cached(corpus, "deep")
        c_f, c_doc, c_scores = predict_cascade(cascade, cached["X"])
        with np.load(args.features / "eval" / f"{corpus}.npz") as z:
            Xe = z["X"]
        if len(Xe) != len(c_f):
            raise SystemExit(f"{corpus}: {len(Xe):,} content rows vs "
                             f"{len(c_f):,} cascade rows")
        m_f = content_fire(Xe)
        fused = np.zeros_like(c_f)
        for j, s in enumerate(picks):
            fused[:, j] = apply(s, c_f[:, j], m_f[:, j])
        fused_doc = apply(doc_best, c_doc, m_f.any(axis=1))
        per_corpus[corpus] = evaluate_corpus(
            corpus, fused, fused_doc, cached["Y"], cached["tag_complete"],
            cached["doc_target"], labels, seed=1000 * (seed + 1),
            tag_scores=c_scores)
        print(f"  {corpus:<48s} {len(Xe):>7,} docs  "
              f"{time.perf_counter() - t0:5.1f}s", flush=True)

    # ------------------------------------------- the fused latency, measured
    from training.h2h_bench import sample_documents  # noqa: E402
    from training.v5_tagger import READ_CHARS, TokenFeaturiser  # noqa: E402
    feat = TokenFeaturiser(args.m2v)
    bench_docs = [d[:READ_CHARS] for d in sample_documents(args.bench_docs, 10_000)]
    lat = bench_fused(cascade, feat.one, content_fire, picks, doc_best,
                      bench_docs, args.bench_repeats, args.cpus)
    (PROJECT / "evaluations").mkdir(parents=True, exist_ok=True)
    (PROJECT / "evaluations" / "latency_v5d.json").write_text(
        json.dumps(lat | {"read_chars": READ_CHARS, "n_documents": len(bench_docs)},
                   indent=1), encoding="utf-8")
    print(f"\nfused one-core latency: p50 {lat['p50_ms']:.3f}  p95 {lat['p95_ms']:.3f}  "
          f"p99 {lat['p99_ms']:.3f} ms", flush=True)

    arm = assemble_arm(
        name=args.name, label="cascade_scorecard61 + v5c content tagger, per-tag fusion",
        per_corpus=per_corpus, p95_latency_ms=lat["p95_ms"],
        docs_per_s=lat["docs_per_s"],
        extra={"cascade": str(args.cascade_dir or args.cascade),
               "tagger": str(args.tagger),
               "fusion_tags": dict(Counter(picks)), "fusion_document": doc_best,
               "n_labels": len(labels)})
    out = PROJECT / "evaluations" / f"arm_{args.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(arm, indent=1, sort_keys=True) + "\n", encoding="utf-8")

    v = lambda a, k: (a["metrics"][k]["value"] if isinstance(a["metrics"][k], dict)  # noqa: E731
                      else a["metrics"][k])
    print(f"\n{'metric':<34}{args.name:>18}")
    for k in ("micro_f1", "precision_micro", "macro_f05", "macro_f2",
              "priority_macro_f05", "recall_macro_catalogue",
              "precision_macro_catalogue", "severity_recall_min", "prediction_rate",
              "equal_corpus_doc_recall", "equal_corpus_doc_precision",
              "equal_corpus_doc_specificity"):
        val = v(arm, k)
        if val is not None:
            print(f"{k:<34}{val:>18.4f}")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
