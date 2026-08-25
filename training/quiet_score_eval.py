"""Score finalists against the eight sealed corpora and write `arms.json`.

This is the run's single pass over `2-eval`. It scores each named artifact one
corpus at a time, measures one-core latency separately from the batch scoring,
and emits arms in the shape `mp decide` reads against `policy.yaml`.

Batch scoring uses the cached features; the latency figure does **not**. A
number quoted as "milliseconds per document on one core" has to include reading
and tokenising the document, so it is measured on the real serving path over
real files, with the CPU budget pinned. Mixing the two would quote a cached
matrix multiply as if it were inference.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_pipeline import set_cpus  # noqa: E402
from training.quiet_cache import CACHE_ROOT, load_catalogue  # noqa: E402
from training.quiet_data import EVAL_ROOT, iter_quiet_corpus, read_document  # noqa: E402
from training.quiet_eval import assemble_arm, evaluate_corpus  # noqa: E402
from training.quiet_fit import load as load_split  # noqa: E402
from training.quiet_model import QuietCascade  # noqa: E402
from training.quiet_score_baseline import baseline_predictions  # noqa: E402

PROJECT = Path("/home/lence/workspace/pii_master/projects/pii-quiet-alarm")


def eval_corpora() -> list[str]:
    return sorted(d.name for d in EVAL_ROOT.iterdir() if d.is_dir())


def cascade_predictions(model: QuietCascade, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Batch predictions from the cached features for the model's own profile."""
    profile = {v[0]: k for k, v in
               {k: tuple(v) for k, v in load_catalogue()["profiles"].items()}.items()}[model.window]
    ds = load_split([name], profile=profile)
    X = ds.X
    gate = (X @ model.gate_weights + model.gate_intercept).astype(np.float32)
    open_doc = gate >= model.gate_threshold
    if model.score_mode == "sum":
        S = (X @ model.tag_weights.T).astype(np.float32)
    else:
        from training.quiet_fit import score as score_fn
        S = score_fn(X, model.tag_weights, mode=model.score_mode)
    fired_tags = (S >= model.tag_thresholds[None, :]) & open_doc[:, None]
    fired_doc = open_doc & fired_tags.any(axis=1)
    return fired_tags, fired_doc


def measure_latency(model: QuietCascade, n_docs: int, min_chars: int) -> tuple[float, float]:
    """One-core p95 milliseconds and documents/second on the real serving path."""
    texts: list[str] = []
    for corpus in ("6589_govdocs2-dualjudge-eval20-3.53k", "4000_datax-dualjudge-evalset-1.32k"):
        for qr in iter_quiet_corpus(EVAL_ROOT / corpus):
            try:
                text = read_document(Path(qr.path), limit=min_chars * 2)
            except (FileNotFoundError, OSError):
                continue
            if len(text) >= min_chars:
                texts.append(text)
            if len(texts) >= n_docs:
                break
        if len(texts) >= n_docs:
            break
    timings = []
    for _ in range(3):
        for text in texts:
            t0 = time.perf_counter()
            model.predict(text)
            timings.append((time.perf_counter() - t0) * 1000.0)
    a = np.asarray(timings)
    return float(np.percentile(a, 95)), float(1000.0 / a.mean())


def score_model(name: str, label: str, predict, p95: float, dps: float,
                extra: dict[str, Any]) -> dict[str, Any]:
    per_corpus: dict[str, dict[str, Any]] = {}
    catalogue = tuple(load_catalogue()["labels"])
    for i, corpus in enumerate(eval_corpora()):
        with np.load(CACHE_ROOT / f"{corpus}.npz", allow_pickle=False) as z:
            doc_target = z["doc_target"]
            tag_complete = z["tag_complete"]
            indptr, cols = z["label_indptr"], z["label_cols"]
        Y = np.zeros((len(doc_target), len(catalogue)), dtype=bool)
        for r in range(len(doc_target)):
            Y[r, cols[indptr[r]:indptr[r + 1]]] = True
        fired_tags, fired_doc = predict(corpus)
        per_corpus[corpus] = evaluate_corpus(
            corpus, fired_tags, fired_doc, Y, tag_complete, doc_target,
            catalogue, seed=90_000 + i * 97)
        print(f"    {corpus:<45} scored", file=sys.stderr, flush=True)
    return assemble_arm(name, label, per_corpus, p95, dps, extra)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=[],
                    metavar="NAME=DIR", help="a materialised cascade to score")
    ap.add_argument("--baseline", action="store_true",
                    help="also score the frozen pii-priority-fusion-1k-v1 champion")
    ap.add_argument("--cpus", default="1")
    ap.add_argument("--latency-docs", type=int, default=150)
    ap.add_argument("--latency-chars", type=int, default=10_000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    set_cpus(args.cpus if args.cpus == "all" else int(args.cpus))

    arms: list[dict[str, Any]] = []
    for spec in args.model:
        name, _, directory = spec.partition("=")
        model = QuietCascade.load(Path(directory))
        print(f"  scoring {name} ({model.config})", file=sys.stderr)
        p95, dps = measure_latency(model, args.latency_docs, args.latency_chars)
        arms.append(score_model(
            name, name, lambda c, m=model: cascade_predictions(m, c), p95, dps,
            {"config": model.config, "artifact": str(directory)}))
        print(f"    p95 {p95:.3f} ms/doc, {dps:.0f} docs/s on {args.cpus} core(s)",
              file=sys.stderr)

    if args.baseline:
        print("  scoring champion_1k (frozen prior lineage)", file=sys.stderr)
        preds, p95, dps = baseline_predictions()
        arms.append(score_model(
            "champion_1k", "pii-priority-fusion-1k-v1 (prior champion)",
            lambda c: preds[c], p95, dps,
            {"lineage": "pii-priority-recall-v1", "note": "recall-first baseline"}))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"arms": arms}, indent=1), encoding="utf-8")
    print(f"wrote {len(arms)} arm(s) -> {args.out}")
    for a in arms:
        m = a["metrics"]
        get = lambda k: (m[k]["value"] if m[k]["value"] is not None else float("nan"))  # noqa: E731
        print(f"  {a['name']:<22} priorityF0.5={get('priority_macro_f05'):.4f} "
              f"docP={get('equal_corpus_doc_precision'):.4f} "
              f"docSp={get('equal_corpus_doc_specificity'):.4f} "
              f"docR={get('equal_corpus_doc_recall'):.4f} "
              f"p95={get('p95_latency_ms'):.2f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
