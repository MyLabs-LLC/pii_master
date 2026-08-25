"""Per-label gate-boundary threshold selection for the priority head.

``tune_priority_hash._thresholds_for_trial`` picks **one** ``priority_target_index``
for every priority tag, so the recall target the hardest tag needs (bank account,
worst-corpus recall 0.60) is imposed on tags already saturated at 1.0000 (IBAN,
ITIN, MRN, PIN, credit card). Those tags pay precision for recall the gate never
asked for: the shipped champion runs at ~0.99 recall / ~0.16 implied precision.

This selects the target **per label** instead: for each priority tag, the highest
threshold (lowest recall target) whose bootstrap ``ci_lower`` still clears the
0.90 gate on every held-in corpus with support >= 30, plus a transfer margin.

Only ``HashCueModel.thresholds`` changes. Weights, features, read window, the
scorer and the split are untouched, so latency is unchanged by construction.
"""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from training.priority_data import PRIORITY_TAGS, read_document
from training.priority_hash import document_features, score_modes
from training.priority_hash import load_priority_model as load_fusion_model

# The same ladder tune_priority_hash searches, so a per-label choice stays
# comparable with the shared-index arms it replaces.
TARGETS = np.asarray(
    [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.92, 0.94, 0.96, 0.98, 0.99, 0.995, 0.999]
)
MIN_SUPPORT = 30
GATE = 0.90
N_RESAMPLES = 2000
CONFIDENCE = 0.95

#: Held-in corpus -> the holdout corpus drawn from the same source. The splits
#: are 4:1 (5:1 for the two largest), so a holdout scope carries roughly a
#: quarter of the positives its held-in twin does -- and a quarter of the
#: statistical power. Selecting on held-in point recall alone therefore picks
#: thresholds whose ``ci_lower`` clears 0.90 on 3,000 documents and misses it on
#: 150. The gate below scales each held-in scope down to the support its holdout
#: twin will actually have before taking the bootstrap bound, so a threshold is
#: only chosen when the recall would still be *provable* at holdout sample size.
#: Only split sizes are used -- never holdout labels, predictions or scores.
CORPUS_PAIRS = {
    "15986_datax-dualjudge-trainset-5.36k": "4000_datax-dualjudge-evalset-1.32k",
    "23693_govdocs2-dualjudge-train80-12.86k": "6589_govdocs2-dualjudge-eval20-3.53k",
    "42504_ai4privacy_pii_masking_train_42.50k": "10626_ai4privacy_pii_masking_eval_10.63k",
    "41429_betterdataai_ner_silver_train_41.43k": "10360_betterdataai_ner_silver_eval_10.36k",
    "21743_nemotron_train_20.80k": "5617_nemotron_eval_5.36k",
    "151708_openpii_pii_train_151.71k": "38937_openpii_pii_eval_38.94k",
    "148775_pii2_train_98.81k": "30000_pii2_eval_25.15k",
    "85593_pii_trainset_85.59k": "20000_pii_holdout_20.00k",
}


def holdout_power_scale(project: Path, kept) -> dict[str, float]:
    """Per-corpus ``eval_rows / sampled_train_rows`` -- structural sizes only."""
    eval_rows: dict[str, int] = {}
    for line in (project / "data" / "eval_index.jsonl").read_text(
        encoding="utf-8"
    ).splitlines():
        ds = json.loads(line)["dataset"]
        eval_rows[ds] = eval_rows.get(ds, 0) + 1
    sampled: dict[str, int] = {}
    for row in kept:
        sampled[row["dataset"]] = sampled.get(row["dataset"], 0) + 1
    scale: dict[str, float] = {}
    for train_ds, n in sampled.items():
        twin = CORPUS_PAIRS.get(train_ds)
        scale[train_ds] = (eval_rows.get(twin, n) / n) if twin and n else 1.0
    return scale

_STATE: dict[str, Any] = {}


def _init_worker(model_dir: str) -> None:
    model = load_fusion_model(Path(model_dir))
    recall = model.components["recall"]
    _STATE["recall"] = recall
    _STATE["window"] = model.read_window_chars
    _STATE["n_features"] = recall.n_features
    _STATE["max_tokens"] = recall.max_tokens
    _STATE["max_features"] = recall.max_document_features


def _score_one(payload: tuple[str, str]) -> tuple[str, list[float] | None]:
    uid, path = payload
    try:
        text = read_document(Path(path), limit=_STATE["window"])
    except Exception:
        return uid, None
    features = document_features(
        text[: _STATE["window"]],
        n_features=_STATE["n_features"],
        max_tokens=_STATE["max_tokens"],
        max_features=_STATE["max_features"],
    )
    recall = _STATE["recall"]
    scores = score_modes(recall.weights, features)[recall.score_mode]
    return uid, scores.astype(np.float32).tolist()


def _bootstrap_recall_lower(tp: int, fn: int, n_rows: int, seed: int) -> float | None:
    """Document bootstrap collapsed to TP/FN/other -- identical to priority_eval."""
    if tp + fn == 0:
        return None
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(
        n_rows,
        np.asarray([tp, fn, n_rows - tp - fn], dtype=np.float64) / n_rows,
        size=N_RESAMPLES,
    )
    support = draws[:, 0] + draws[:, 1]
    samples = draws[support > 0, 0] / support[support > 0]
    return float(np.quantile(samples, (1.0 - CONFIDENCE) / 2.0))


def build_matrix(project: Path, model_dir: Path, sample: int, workers: int, seed: int):
    rows = [
        json.loads(line)
        for line in (project / "data" / "train_index.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    # Every corpus, not just the label-complete ones. The gate is a *recall*
    # gate, and a positive-only corpus measures recall perfectly well -- the
    # frozen evaluator scores those corpora exactly that way. Filtering them out
    # is what let the first candidate raise full_name/address past the gate on
    # govdocs2, the one holdout corpus selection could not see.
    by_ds: dict[str, list[dict]] = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], []).append(r)

    rng = random.Random(seed)
    picked: list[dict] = []
    per_ds = max(1, sample // len(by_ds))
    for ds in sorted(by_ds):
        corpus = by_ds[ds]
        rng.shuffle(corpus)
        picked.extend(corpus[:per_ds])   # uniform -> class balance preserved
    print(f"[select] scoring {len(picked)} held-in docs from {len(by_ds)} corpora")

    tasks = [(r["uid"], r["path"]) for r in picked]
    scores: dict[str, list[float]] = {}
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(str(model_dir),)
    ) as pool:
        for i, (uid, vec) in enumerate(pool.map(_score_one, tasks, chunksize=64), 1):
            if vec is not None:
                scores[uid] = vec
            if i % 20000 == 0:
                print(f"[select]   {i}/{len(tasks)}", flush=True)
    kept = [r for r in picked if r["uid"] in scores]
    print(f"[select] scored {len(kept)} docs ({len(picked) - len(kept)} read errors)")
    return kept, scores


def select(project: Path, model_dir: Path, kept, scores, margin: float) -> dict:
    scale = holdout_power_scale(project, kept)
    model = load_fusion_model(model_dir)
    labels = list(model.components["recall"].labels)
    index = {lab: i for i, lab in enumerate(labels)}
    datasets = sorted({r["dataset"] for r in kept})
    matrix = np.asarray([scores[r["uid"]] for r in kept], dtype=np.float32)
    ds_of = np.asarray([r["dataset"] for r in kept])
    gold = [set(r.get("labels") or []) for r in kept]
    # Precision is only defined where a document's absent tag really is a
    # negative. On positive-only corpora it is unknown, so the diagnostic is
    # computed over the label-complete rows alone. The gate below is recall and
    # uses every corpus.
    complete = np.asarray([bool(r.get("label_complete")) for r in kept])

    chosen: dict[str, Any] = {}
    for tag in PRIORITY_TAGS:
        if tag not in index:
            continue
        col = index[tag]
        positive = np.asarray([tag in g for g in gold])
        usable = [d for d in datasets if int((positive & (ds_of == d)).sum()) >= MIN_SUPPORT]
        if not usable:
            chosen[tag] = {"target": None, "threshold": None, "reason": "no corpus >= 30 positives"}
            continue

        best = None
        for target in TARGETS:                      # low target -> high threshold -> precision
            per_source = [
                float(np.quantile(matrix[positive & (ds_of == d), col], 1.0 - target, method="lower"))
                for d in usable
            ]
            threshold = min(per_source)
            ok, worst_lower = True, 1.0
            for d in usable:
                mask = ds_of == d
                pred = matrix[mask, col] >= threshold
                pos_d = positive[mask]
                tp = int((pred & pos_d).sum())
                fn = int((~pred & pos_d).sum())
                # Shrink to the support the holdout twin will actually have, so
                # the bound is the one the gate will apply, not a tighter one
                # bought with 4x the documents.
                k = scale.get(d, 1.0)
                lower = _bootstrap_recall_lower(
                    max(0, round(tp * k)),
                    max(0, round(fn * k)),
                    max(MIN_SUPPORT, round(int(mask.sum()) * k)),
                    seed=abs(hash((tag, d))) % (2**32),
                )
                if lower is None or lower < GATE + margin:
                    ok = False
                    break
                worst_lower = min(worst_lower, lower)
            if not ok:
                continue
            pred_all = matrix[:, col] >= threshold
            tp_c = int((pred_all & positive & complete).sum())
            fp_c = int((pred_all & ~positive & complete).sum())
            precision = tp_c / (tp_c + fp_c) if tp_c + fp_c else 0.0
            tp = int((pred_all & positive).sum())
            recall = tp / int(positive.sum()) if positive.sum() else 0.0
            best = {
                "target": float(target),
                "threshold": threshold,
                "held_in_precision": precision,
                "held_in_recall": recall,
                "worst_corpus_ci_lower": worst_lower,
                "corpora_used": usable,
            }
            break                                    # first (lowest) target that holds the gate
        if best is None:
            per_source = [
                float(np.quantile(matrix[positive & (ds_of == d), col], 1.0 - 0.999, method="lower"))
                for d in usable
            ]
            best = {"target": 0.999, "threshold": min(per_source), "reason": "fell back to max recall"}
        chosen[tag] = best
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=Path, default=Path("projects/pii-priority-recall-v1"))
    ap.add_argument("--source", default="champion_1k")
    ap.add_argument("--out", default="perlabel_1k")
    ap.add_argument("--sample", type=int, default=120_000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--margin", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--matrix-cache", type=Path, default=None)
    args = ap.parse_args()

    model_dir = args.project / "models" / args.source
    cache = args.matrix_cache or (args.project / "cache" / f"{args.source}_recall_heldin.npz")
    if cache.exists():
        print(f"[select] reusing cached matrix {cache}")
        stored = np.load(cache, allow_pickle=True)
        kept = json.loads(str(stored["rows"]))
        scores = {r["uid"]: vec for r, vec in zip(kept, stored["matrix"].tolist())}
    else:
        kept, scores = build_matrix(args.project, model_dir, args.sample, args.workers, args.seed)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache,
            rows=json.dumps(kept),
            matrix=np.asarray([scores[r["uid"]] for r in kept], dtype=np.float32),
        )
        print(f"[select] cached matrix -> {cache}")

    chosen = select(args.project, model_dir, kept, scores, args.margin)
    out_json = args.project / "tuning" / args.out
    out_json.mkdir(parents=True, exist_ok=True)
    (out_json / "selected_thresholds.json").write_text(
        json.dumps(chosen, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(chosen, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
