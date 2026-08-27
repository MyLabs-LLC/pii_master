"""Score one saved cascade -- a `models/<name>` dir or a packaged bundle -- sealed.

`h2h_score.py` carries a fixed three-arm registry (A/B/C) and is deliberately
not touched: those arms' published numbers must stay exactly reproducible. This
scores an arbitrary saved cascade through the **same** fixed evaluator,
`h2h_eval.evaluate_corpus` / `assemble_arm`, so a new arm joins the existing 128
results on the same terms.

Two things it insists on, both borrowed from `h2h_score.verify`:

* **the cached feature path must equal the model's own `predict`**, checked on a
  sample per corpus. The cache is an optimisation; if it and the serving path
  disagree, the number is not the model's.
* **a packaged bundle is scored through its own `runtime/`**, not through the
  training tree, which is what `champion-package.md` means by re-scoring a bundle
  through its own code. `--bundle` puts the bundle's `runtime/` on the path ahead
  of `training/` and loads its `tagger.Tagger`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_pipeline import set_cpus  # noqa: E402
from training.h2h_bench import sample_documents  # noqa: E402
from training.h2h_eval import assemble_arm, evaluate_corpus  # noqa: E402
from training.h2h_priority import PROJECT, eval_corpora  # noqa: E402
from training.h2h_score import _load_cached, predict_cascade  # noqa: E402
from training.quiet_cache import load_catalogue  # noqa: E402
from training.quiet_data import iter_quiet_corpus, read_document, resolve_dataset  # noqa: E402
from training.quiet_model import QuietCascade  # noqa: E402

#: Arm B's lineage, and therefore every cascade derived from it, was fitted on
#: the 12,000-character extraction. See the note at the head of `h2h_score.ARMS`:
#: for `.docx` the extraction itself depends on the limit, so a cascade must be
#: read at the limit it was trained against or it is being scored on text it
#: never saw.
READ_LIMIT = 12_000
PROFILE = "deep"


def load_bundle_tagger(bundle: Path):
    """The bundle's own entry point, loaded through the bundle's own runtime."""
    sys.path.insert(0, str(bundle / "runtime"))
    spec = importlib.util.spec_from_file_location(
        f"_bundle_tagger_{bundle.name}".replace("-", "_"), bundle / "tagger.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Tagger()


def bench(predictor, has_pii, docs: list[str], repeats: int, cpus: str) -> dict:
    """One-core latency through the same serving call `h2h_bench` times.

    `h2h_bench.py` carries the fixed A/B/C registry and is left alone for the same
    reason `h2h_score.py` is. The measurement is identical: `set_cpus`, real >=10 KB
    documents from the two real-world corpora, `predict` on each, percentiles over
    the pooled timings, and the split by gate decision -- the cascade short-circuits
    on a document its gate rejects, so the mix matters and is reported rather than
    averaged away.
    """
    set_cpus(cpus if cpus == "all" else int(cpus))
    timings: list[float] = []
    gate_open: list[bool] = []
    for _ in range(repeats):
        for text in docs:
            t0 = time.perf_counter()
            predictor(text)
            timings.append((time.perf_counter() - t0) * 1000.0)
            gate_open.append(bool(has_pii(text)))
    a = np.asarray(timings)
    payload = {
        "n": int(a.size), "repeats": repeats,
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "mean_ms": float(a.mean()),
        "docs_per_s": float(1000.0 / a.mean()),
        "cpu_budget": cpus,
    }
    g = np.asarray(gate_open)
    if g.any() and (~g).any():
        payload |= {"p95_ms_firing": float(np.percentile(a[g], 95)),
                    "p95_ms_silent": float(np.percentile(a[~g], 95)),
                    "fire_rate": float(g.mean())}
    return payload


def verify(model, predictor, labels, n: int, seed: int = 5) -> dict:
    """Cached-feature predictions must equal the model's own `predict`, exactly."""
    rng = np.random.default_rng(seed)
    checked = mismatched = 0
    examples: list[str] = []
    for corpus in eval_corpora():
        cached = _load_cached(corpus, PROFILE)
        rows = list(iter_quiet_corpus(resolve_dataset(corpus)))
        take = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
        fired, _, _ = predict_cascade(model, cached["X"][take])
        for k, i in enumerate(take):
            text = read_document(Path(rows[i].path), limit=READ_LIMIT)
            direct = set(predictor(text))
            from_cache = {labels[j] for j in np.flatnonzero(fired[k])}
            checked += 1
            if direct != from_cache:
                mismatched += 1
                if len(examples) < 5:
                    examples.append(
                        f"{corpus}#{rows[i].uid}: only-direct={sorted(direct - from_cache)} "
                        f"only-cache={sorted(from_cache - direct)}")
    return {"checked": checked, "mismatched": mismatched, "examples": examples}


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", help="a directory under projects/<p>/models/")
    src.add_argument("--bundle", type=Path,
                     help="a packaged bundle dir; scored through its own runtime/")
    ap.add_argument("--name", required=True, help="arm name, used for the output file")
    ap.add_argument("--label", default="", help="human-readable arm label")
    ap.add_argument("--verify", type=int, default=0,
                    help="documents per corpus to re-score through the model itself")
    ap.add_argument("--latency", type=Path, default=None,
                    help="an existing benchmark json to carry (mutually exclusive "
                         "with --bench-out, which measures this arm's own)")
    ap.add_argument("--bench-out", type=Path, default=None,
                    help="measure this arm's one-core latency and write it here")
    ap.add_argument("--bench-docs", type=int, default=200)
    ap.add_argument("--bench-chars", type=int, default=10_000)
    ap.add_argument("--bench-repeats", type=int, default=5)
    ap.add_argument("--cpus", default="1")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.latency and args.bench_out:
        raise SystemExit("--latency carries a number measured elsewhere and --bench-out "
                         "measures a new one; pick one so the report can say which.")

    labels = tuple(load_catalogue()["labels"])
    if args.bundle:
        tagger = load_bundle_tagger(args.bundle)
        model = QuietCascade.load(args.bundle / "models" / "model")
        predictor = tagger.predict
        source = str(args.bundle)
        if int(tagger.read_window_chars) != READ_LIMIT:
            raise SystemExit(
                f"bundle reads {tagger.read_window_chars} chars, this scorer extracts at "
                f"{READ_LIMIT}; the extraction depends on the limit, so they must agree.")
    else:
        model = QuietCascade.load(PROJECT / "models" / args.model)
        predictor = model.predict
        source = f"models/{args.model}"
    print(f"scoring {source}", flush=True)

    if args.verify:
        report = verify(model, predictor, labels, args.verify)
        print(json.dumps({"phase": "verify", **report}), flush=True)
        if report["mismatched"]:
            raise SystemExit(
                f"{report['mismatched']} of {report['checked']} sampled documents "
                f"disagree between the cached features and the model's own predict(). "
                f"The cached path is not the serving path.\n  "
                + "\n  ".join(report["examples"]))

    if args.bench_out:
        docs = sample_documents(args.bench_docs, args.bench_chars)
        print(f"benchmarking on {len(docs)} documents of >= {args.bench_chars:,} "
              f"characters, cpu_budget={args.cpus}", flush=True)
        latency = bench(predictor, model.has_pii, docs, args.bench_repeats, args.cpus)
        latency |= {"source": source, "doc_chars_min": args.bench_chars,
                    "n_documents": len(docs)}
        args.bench_out.parent.mkdir(parents=True, exist_ok=True)
        args.bench_out.write_text(json.dumps(latency, indent=1) + "\n", encoding="utf-8")
        print(f"  p50={latency['p50_ms']:.4f} ms  p95={latency['p95_ms']:.4f} ms  "
              f"{'PASS' if latency['p95_ms'] <= 5.0 else 'FAIL'} against the 5 ms budget",
              flush=True)
    else:
        latency = (json.loads(args.latency.read_text(encoding="utf-8"))
                   if args.latency else {})
    per_corpus: dict[str, dict] = {}
    for seed, corpus in enumerate(eval_corpora()):
        t0 = time.perf_counter()
        cached = _load_cached(corpus, PROFILE)
        fired, fired_doc, tag_scores = predict_cascade(model, cached["X"])
        per_corpus[corpus] = evaluate_corpus(
            corpus, fired, fired_doc, cached["Y"], cached["tag_complete"],
            cached["doc_target"], labels, seed=1000 * (seed + 1),
            tag_scores=tag_scores)
        print(f"  {corpus:<48s} {cached['X'].shape[0]:>7,} docs  "
              f"{time.perf_counter() - t0:5.1f}s", flush=True)

    arm = assemble_arm(
        name=args.name, label=args.label or source, per_corpus=per_corpus,
        p95_latency_ms=latency.get("p95_ms"), docs_per_s=latency.get("docs_per_s"),
        extra={"source": source, "read_limit": READ_LIMIT, "profile": PROFILE,
               "verified_documents": (args.verify * len(eval_corpora())
                                      if args.verify else 0)})
    out = args.out or PROJECT / "evaluations" / f"arm_{args.name}.json"
    out.write_text(json.dumps(arm, indent=1, sort_keys=True) + "\n", encoding="utf-8")

    v = lambda a, k: (a["metrics"][k]["value"] if isinstance(a["metrics"][k], dict)  # noqa: E731
                      else a["metrics"][k])
    print(f"\n{'metric':<34}{args.name:>16}")
    for k in ("macro_f2", "micro_f1", "priority_macro_f05",
              "equal_corpus_doc_recall", "equal_corpus_doc_precision",
              "equal_corpus_doc_specificity", "recall_macro_catalogue",
              "precision_macro_catalogue", "severity_recall_min", "prediction_rate"):
        val = v(arm, k)
        if val is not None:
            print(f"{k:<34}{val:>16.4f}")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
