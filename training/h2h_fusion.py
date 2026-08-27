"""The fusion stage, and the two arms it materialises.

The priority lineage's last step: four component heads have been tuned
independently, and this chooses, per label, which Boolean combination of them to
serve. `tune_priority_fusion`'s option set, ranking rule and strategy sampler are
imported unchanged; only the feature source differs, as in `h2h_priority`.

**Why arms A and C are produced here together.** In the shipped lineage the
components predict at their own 20,000-character window during fusion selection,
and the serving window is applied afterwards as a `read_window_override` chosen
by a separate read-depth benchmark -- that is how `pii-priority-fusion-1k-v1`
came to be calibrated at 20,000 and served at 1,000. Reproducing that faithfully
means the strategy selection happens once, at 20,000, and the two arms differ by
exactly one field:

    arm A  read_window_override =  1,000   (as shipped)
    arm C  read_window_override = 12,000   (matched to arm B)

Nothing else about them differs -- same counts, same component weights, same
per-label thresholds, same fusion strategies. That is what makes C a control on
the read window rather than a third model.

The train/serve mismatch this preserves is a real property of the recipe, not an
artefact of the harness: thresholds chosen against 20,000-character scores are
applied to 1,000-character scores. It is reproduced rather than fixed, and
measuring what it costs is part of what arm C is for.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.h2h_priority import (  # noqa: E402
    PROJECT, _dense_labels, _save, _prepare, score_embeddingbag, score_top_modes,
)
from training.priority_embeddingbag import LowRankEmbeddingBagModel  # noqa: E402
from training.priority_fusion import FusionPriorityModel  # noqa: E402
from training.priority_hash import HashCueModel  # noqa: E402
from training.tune_priority_embeddingbag import RANKS  # noqa: E402
from training.tune_priority_fusion import (  # noqa: E402
    fuse_predictions, rank_options, strategy_configs,
)
from training.tune_priority_hash import _objective, fast_metrics  # noqa: E402

#: fusion component name -> the tuning family that produced it. The shipped
#: bundle's names, kept so its `strategies` map stays readable against this one.
COMPONENTS = {"recall": "hash", "hash": "hash", "tfidf": "tfidf", "embedding": "embed"}
#: The lineage's own window for component scoring during selection.
SELECTION_PROFILE = "train20k"


def _best(family: str) -> dict[str, Any]:
    return json.loads((PROJECT / "tuning" / family / "best.json").read_text(encoding="utf-8"))


def build_components(labels: tuple[str, ...]) -> dict[str, Any]:
    """The four tuned heads, as servable model objects at the 20,000 window.

    `recall` and `hash` share one weight matrix and differ only in the operating
    point the ladder chose for them -- which is exactly the shipped lineage's
    arrangement (`hash_sgd` and `hash_sgd_f2` are two threshold selections over
    one count matrix), and why the fusion has a recall-shaped head and an
    F2-shaped head to combine.
    """
    W_hash = np.load(PROJECT / "cache" / "W_hash.npy")
    W_tfidf = np.load(PROJECT / "cache" / "W_tfidf.npy")
    best_hash, best_tfidf, best_embed = _best("hash"), _best("tfidf"), _best("embed")

    with np.load(PROJECT / "cache" / "embed_factors.npz", allow_pickle=False) as z:
        factors = {r: (z[f"emb{r}"], z[f"head{r}"]) for r in RANKS}
    with np.load(PROJECT / "cache" / "embed_calibration.npz", allow_pickle=False) as z:
        cals = {r: (z[f"cal{r}"], z[f"bias{r}"]) for r in RANKS}
    rank = int(best_embed["config"]["score_mode"].removeprefix("rank"))

    # `recall` is the ladder's most recall-shaped feasible point, `hash` its best
    # overall point, taken from the same family's trial record.
    trials = json.loads((PROJECT / "tuning" / "hash" / "trials.json").read_text(encoding="utf-8"))
    feasible = [t for t in trials if t["objective"][0] == 1.0] or trials
    recall_trial = max(feasible, key=lambda t: t["metrics"]["worst_priority_recall"])
    if not recall_trial.get("thresholds"):
        # Falling back to the F0.5 winner's cuts here would make `recall` a copy
        # of `hash`, and the fusion would quietly lose the recall-shaped head
        # that half its per-label strategies vote with. Fail instead.
        raise SystemExit(
            "hash trials.json carries no per-trial thresholds, so the recall-shaped "
            "component cannot be rebuilt. Re-run: h2h_priority.py family --name hash")
    recall_thr = np.asarray(recall_trial["thresholds"], dtype=np.float32)

    def cue(weights: np.ndarray, best: dict[str, Any], thresholds=None) -> HashCueModel:
        return HashCueModel(
            labels=labels, weights=weights.astype(np.float32),
            thresholds=(np.asarray(best["thresholds"], dtype=np.float32)
                        if thresholds is None else thresholds),
            score_mode=best["config"]["score_mode"],
            read_window_chars=20_000, n_features=weights.shape[1],
            max_tokens=768, max_document_features=512)

    return {
        "recall": cue(W_hash, recall_trial, recall_thr),
        "hash": cue(W_hash, best_hash),
        "tfidf": cue(W_tfidf, best_tfidf),
        "embedding": LowRankEmbeddingBagModel(
            labels=labels, embeddings=factors[rank][0], head=factors[rank][1],
            calibration=cals[rank][0], bias=cals[rank][1],
            thresholds=np.asarray(best_embed["thresholds"], dtype=np.float32),
            read_window_chars=20_000, max_tokens=768, max_document_features=512),
    }, {"recall_trial": recall_trial["trial"], "hash_trial": best_hash["trial"],
        "tfidf_trial": best_tfidf["trial"], "embed_trial": best_embed["trial"],
        "embed_rank": rank}


def component_predictions(X, comps: dict[str, Any], labels: tuple[str, ...],
                          ) -> dict[str, np.ndarray]:
    """Boolean per-component predictions over a cached feature matrix."""
    out: dict[str, np.ndarray] = {}
    top = {}
    for name, model in comps.items():
        if isinstance(model, HashCueModel):
            key = id(model.weights)
            if key not in top:
                top[key] = score_top_modes(X, model.weights, label=f"fusion/{name}")
            out[name] = top[key][model.score_mode] >= model.thresholds
        else:
            raw = score_embeddingbag(X, {model.embeddings.shape[1]: (model.embeddings, model.head)},
                                     label=f"fusion/{name}")
            mode = f"rank{model.embeddings.shape[1]}"
            out[name] = (raw[mode] @ model.calibration.T + model.bias) >= model.thresholds
    return out


def main() -> int:
    import mlflow

    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    args = ap.parse_args()

    mlflow.set_tracking_uri(f"sqlite:///{PROJECT / 'mlflow.db'}")
    mlflow.set_experiment("pii-head-to-head-v1")

    p = _prepare(profile=SELECTION_PROFILE)
    ds, labels = p["ds"], p["labels"]
    calA, calB = p["calA"], p["calB"]
    comps, provenance = build_components(labels)

    both = calA | calB
    preds_all = component_predictions(ds.X[both], comps, labels)
    Y_all = _dense_labels(ds.Y[both])
    complete_all = ds.tag_complete[both]
    datasets_all = p["datasets_all"][both]
    is_a = calA[both]

    # Rank each label's options on calA, then score whole strategies on calB --
    # the sampler draws from a label's top three, so ranking and scoring on the
    # same rows would let a noisy option be chosen and then confirmed by its own
    # noise.
    ranked = rank_options(
        {"calibration": is_a, "complete": complete_all, "y_true": Y_all,
         "predictions": preds_all}, labels)
    configs = strategy_configs(ranked, labels, n_trials=args.trials)

    sel = ~is_a
    sel_components = {k: v[sel] for k, v in preds_all.items()}
    Y, datasets, complete = Y_all[sel], datasets_all[sel], complete_all[sel]

    out_dir = PROJECT / "tuning" / "fusion"
    out_dir.mkdir(parents=True, exist_ok=True)
    best: dict[str, Any] | None = None
    records: list[dict[str, Any]] = []
    with mlflow.start_run(run_name="tune-fusion"):
        mlflow.log_params({"family": "fusion", "trials": len(configs), "cpu_budget": "all"})
        for number, config in enumerate(configs):
            predicted = fuse_predictions(config, labels, sel_components)
            metrics = fast_metrics(predicted, Y, datasets, complete, labels, p["quality"])
            objective = _objective(metrics)
            record = {"trial": number, "family": "fusion",
                      "metrics": {k: v for k, v in metrics.items() if k != "per_corpus"},
                      "per_corpus": metrics["per_corpus"], "objective": list(objective)}
            records.append(record)
            with mlflow.start_run(nested=True, run_name=f"fusion-{number:03d}"):
                mlflow.log_metrics({
                    "equal_corpus_macro_f2": metrics["equal_corpus_macro_f2"],
                    "equal_corpus_micro_f1": metrics["equal_corpus_micro_f1"],
                    "worst_priority_recall": metrics["worst_priority_recall"],
                    "priority_point_passes": float(metrics["priority_point_passes"]),
                    "feasible": float(objective[0]),
                })
            if best is None or objective > tuple(best["objective"]):
                best = dict(record, strategies=config)
            if (number + 1) % 20 == 0 or number + 1 == len(configs):
                _save(out_dir / "trials.json", records)
                _save(out_dir / "best.json", best)
    _save(out_dir / "trials.json", records)
    _save(out_dir / "best.json", best)

    # Two arms, one field apart.
    for arm, window in (("A", 1_000), ("C", 12_000)):
        model = FusionPriorityModel(
            labels=labels, components=comps, strategies=best["strategies"],
            read_window_override=window)
        model.save(PROJECT / "models" / f"fusion_{window}",
                   metadata={"arm": arm, "read_window_override": window,
                             "fusion_trial": best["trial"],
                             "selection_profile": SELECTION_PROFILE,
                             **provenance})
    m = best["metrics"]
    print(f"fusion: {len(configs)} trials, best trial {best['trial']}")
    print(f"  macro_f2={m['equal_corpus_macro_f2']:.4f} micro_f1={m['equal_corpus_micro_f1']:.4f} "
          f"gates {m['priority_point_passes']}/{m['measurable_priority_gates']} "
          f"worst_recall={m['worst_priority_recall']:.4f}")
    print(f"  materialised models/fusion_1000 (arm A) and models/fusion_12000 (arm C)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
