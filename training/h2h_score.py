"""Score the three arms on the eight sealed corpora, and prove the shortcut.

Predictions come from the cached feature matrices, because scoring three arms
over 126,129 documents by re-reading every file is minutes of arithmetic behind
an hour of disk. The shortcut is only legitimate if it is *identical* to running
the model on the document, so `--verify` does exactly that on a sample: reads the
real file, calls the real model object's own `predict`, and requires an exact
match on every tag of every sampled document. A single mismatch fails the run
rather than being reported as a rounding difference.

This is the only module in the run that opens `2-eval`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.h2h_cache import CACHE_ROOT as H2H_CACHE  # noqa: E402
from training.h2h_cache import N_FEATURES as H2H_FEATURES  # noqa: E402
from training.h2h_eval import assemble_arm, evaluate_corpus  # noqa: E402
from training.h2h_priority import PROJECT, eval_corpora, score_embeddingbag, score_top_modes  # noqa: E402
from training.priority_fusion import FusionPriorityModel, fuse_strategy  # noqa: E402
from training.priority_hash import HashCueModel  # noqa: E402
from training.quiet_cache import CACHE_ROOT as QUIET_CACHE  # noqa: E402
from training.quiet_cache import load_catalogue  # noqa: E402
from training.quiet_data import iter_quiet_corpus, read_document, resolve_dataset  # noqa: E402
from training.quiet_model import QuietCascade  # noqa: E402

#: `read_limit` is the `limit` its lineage's cache passed to `read_document`, and
#: it is NOT cosmetic. `_xml_archive_text` concatenates zip-XML parts until it has
#: `limit` characters, so for `.docx` the extraction itself depends on the limit:
#: one datax file yields 10,788 characters at limit=12,000 and 11,995 at
#: limit=20,000, diverging in the tail. 564 of 4,000 datax documents (14.1%) are
#: affected; govdocs2 is unaffected (0 of 6,578).
#:
#: Each arm is therefore verified -- and was trained -- at its own lineage's
#: limit, which is the only internally consistent choice: arm B's gate and heads
#: were fitted on the 12,000-limit extraction, so scoring it on the 20,000-limit
#: one would measure a model on text it was never fitted against.
ARMS = {
    "A": {"label": "priority-fusion @ 1,000 chars (as shipped)",
          "kind": "fusion", "window": 1_000, "profile": "serve1k",
          "read_limit": 20_000},
    "B": {"label": "steady-aim cascade @ 12,000 chars (as shipped)",
          "kind": "cascade", "window": 12_000, "profile": "deep",
          "read_limit": 12_000},
    "C": {"label": "priority-fusion @ 12,000 chars (read-window control)",
          "kind": "fusion", "window": 12_000, "profile": "serve12k",
          "read_limit": 20_000},
}


def _load_cached(corpus: str, profile: str) -> dict[str, Any]:
    root = QUIET_CACHE if profile in ("fast", "std", "deep") else H2H_CACHE
    n_features = int(load_catalogue()["n_features"]) if root is QUIET_CACHE else H2H_FEATURES
    with np.load(root / f"{corpus}.npz", allow_pickle=False) as z:
        indptr, indices = z[f"indptr_{profile}"], z[f"indices_{profile}"]
        X = sp.csr_matrix((np.ones(len(indices), dtype=np.float32), indices, indptr),
                          shape=(len(indptr) - 1, n_features))
        n_labels = len(load_catalogue()["labels"])
        lp, lc = z["label_indptr"], z["label_cols"]
        Y = sp.csr_matrix((np.ones(len(lc), dtype=np.float32), lc, lp),
                          shape=(len(lp) - 1, n_labels))
        return {"X": X, "Y": np.asarray(Y.todense()).astype(bool),
                "doc_target": z["doc_target"], "tag_complete": z["tag_complete"]}


# --------------------------------------------------------------------- arms
def predict_cascade(model: QuietCascade, X: sp.csr_matrix
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gate first, then tags on what it lets through -- the serving order."""
    gate = (X @ model.gate_weights + model.gate_intercept).astype(np.float32)
    open_doc = (gate >= model.gate_threshold) & (np.diff(X.indptr) > 0)
    if model.score_mode != "sum":
        raise SystemExit(f"cascade score_mode {model.score_mode!r} not supported here")
    scores = (X @ model.tag_weights.T).astype(np.float32)
    fired = (scores >= model.tag_thresholds) & open_doc[:, None]
    return fired, open_doc, scores


def predict_fusion(model: FusionPriorityModel, X: sp.csr_matrix, labels: tuple[str, ...],
                   tag: str = "") -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    per_component: dict[str, np.ndarray] = {}
    cached_top: dict[int, dict[str, np.ndarray]] = {}
    for name, comp in model.components.items():
        if isinstance(comp, HashCueModel):
            key = id(comp.weights)
            if key not in cached_top:
                cached_top[key] = score_top_modes(X, comp.weights, label=f"{tag}/{name}")
            per_component[name] = cached_top[key][comp.score_mode] >= comp.thresholds
        else:
            rank = comp.embeddings.shape[1]
            raw = score_embeddingbag(X, {rank: (comp.embeddings, comp.head)},
                                     label=f"{tag}/{name}")[f"rank{rank}"]
            per_component[name] = (raw @ comp.calibration.T + comp.bias) >= comp.thresholds
    fired = np.zeros((X.shape[0], len(labels)), dtype=bool)
    for j, label in enumerate(labels):
        op, raw_names = model.strategies[label].split(":", 1)
        votes = np.stack([per_component[n][:, j] for n in raw_names.split(",")], axis=1)
        if op == "source":
            fired[:, j] = votes[:, 0]
        elif op == "or":
            fired[:, j] = votes.any(axis=1)
        elif op == "and":
            fired[:, j] = votes.all(axis=1)
        elif op == "majority":
            fired[:, j] = votes.sum(axis=1) >= (votes.shape[1] // 2 + 1)
        else:
            raise ValueError(op)
    # The fusion has no separate document gate: it says "this document has PII"
    # exactly when it emits a tag. Recording that explicitly matters, because the
    # cascade's document answer is a different mechanism, not the same one.
    #
    # It also returns no ranking. Per-label Boolean fusion produces a SET: there
    # is no k-th most likely tag, because the four components vote independently
    # per label and the votes are not commensurable across labels. So the top-k
    # ladder is genuinely undefined for this architecture rather than zero, and
    # `None` is what the evaluator is handed.
    return fired, fired.any(axis=1), None


def load_arm(arm: str):
    spec = ARMS[arm]
    if spec["kind"] == "cascade":
        return QuietCascade.load(PROJECT / "models" / "cascade")
    return FusionPriorityModel.load(PROJECT / "models" / f"fusion_{spec['window']}")


# ------------------------------------------------------------------- verify
def verify(arm: str, model, labels: tuple[str, ...], n: int, seed: int = 5) -> dict[str, Any]:
    """Cached-feature predictions must equal the model's own `predict`, exactly."""
    rng = np.random.default_rng(seed)
    checked = mismatched = 0
    examples: list[str] = []
    for corpus in eval_corpora():
        cached = _load_cached(corpus, ARMS[arm]["profile"])
        rows = list(iter_quiet_corpus(resolve_dataset(corpus)))
        take = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
        X = cached["X"]
        if ARMS[arm]["kind"] == "cascade":
            fired, _, _ = predict_cascade(model, X[take])
        else:
            fired, _, _ = predict_fusion(model, X[take], labels)
        for k, i in enumerate(take):
            text = read_document(Path(rows[i].path), limit=ARMS[arm]["read_limit"])
            direct = set(model.predict(text))
            from_cache = {labels[j] for j in np.flatnonzero(fired[k])}
            checked += 1
            if direct != from_cache:
                mismatched += 1
                if len(examples) < 5:
                    examples.append(
                        f"{corpus}#{rows[i].uid}: only-direct={sorted(direct - from_cache)} "
                        f"only-cache={sorted(from_cache - direct)}")
    return {"checked": checked, "mismatched": mismatched, "examples": examples}


# --------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--verify", type=int, default=0,
                    help="documents per corpus to re-score through the model itself")
    ap.add_argument("--latency", type=Path, default=None,
                    help="benchmark json produced by h2h_bench.py")
    args = ap.parse_args()

    spec = ARMS[args.arm]
    labels = tuple(load_catalogue()["labels"])
    model = load_arm(args.arm)

    if args.verify:
        report = verify(args.arm, model, labels, args.verify)
        print(json.dumps({"phase": "verify", **report}), flush=True)
        if report["mismatched"]:
            raise SystemExit(
                f"arm {args.arm}: {report['mismatched']} of {report['checked']} sampled "
                f"documents disagree between the cached features and the model's own "
                f"predict(). The cached path is not the serving path.\n  "
                + "\n  ".join(report["examples"]))

    latency = json.loads(args.latency.read_text(encoding="utf-8")) if args.latency else {}
    per_corpus: dict[str, dict[str, Any]] = {}
    for seed, corpus in enumerate(eval_corpora()):
        t0 = time.perf_counter()
        cached = _load_cached(corpus, spec["profile"])
        if spec["kind"] == "cascade":
            fired, fired_doc, tag_scores = predict_cascade(model, cached["X"])
        else:
            fired, fired_doc, tag_scores = predict_fusion(
                model, cached["X"], labels, tag=corpus[:18])
        per_corpus[corpus] = evaluate_corpus(
            corpus, fired, fired_doc, cached["Y"], cached["tag_complete"],
            cached["doc_target"], labels, seed=1000 * (seed + 1),
            tag_scores=tag_scores)
        s = per_corpus[corpus]["summary"]
        print(f"  {corpus:44s} n={per_corpus[corpus]['n_rows']:>6,} "
              f"macroF2={s['f2_macro_catalogue'] if s['f2_macro_catalogue'] is None else round(s['f2_macro_catalogue'],4)} "
              f"microF1={s['f1_micro'] if s['f1_micro'] is None else round(s['f1_micro'],4)} "
              f"predrate={s['prediction_rate']:.3f}  ({time.perf_counter()-t0:.0f}s)",
              flush=True)

    arm = assemble_arm(
        name=f"arm-{args.arm}", label=spec["label"], per_corpus=per_corpus,
        p95_latency_ms=latency.get("p95_ms"), docs_per_s=latency.get("docs_per_s"),
        extra={"arm": args.arm, "kind": spec["kind"],
               "read_window_chars": spec["window"], "profile": spec["profile"],
               "verify": report if args.verify else None,
               "latency_evidence": latency or None})
    out = PROJECT / "evaluations" / f"arm_{args.arm}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(arm, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    m = arm["metrics"]
    print(f"arm {args.arm}: macro_f2={m['macro_f2']['value']} "
          f"micro_f1={m['micro_f1']['value']} "
          f"priority_macro_f05={m['priority_macro_f05']['value']}")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
