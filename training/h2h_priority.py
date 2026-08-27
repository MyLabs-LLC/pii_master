"""The priority-fusion lineage, re-tuned on the full corpus from cached features.

Same recipe as `tune_priority_hash` / `tune_priority_tfidf` /
`tune_priority_embeddingbag` / `tune_priority_fusion`, and the shared pieces are
imported from them rather than restated -- the threshold bank, the trial
ladders, the objective, `fast_metrics`, the fusion options and the strategy
sampler all come from those modules unchanged. What changes is where the
features come from and which rows are held in.

**Features come from `h2h_cache`.** The original tuners re-read every document
on every family run. The extraction call is identical, so the counts and scores
are identical; only the disk traffic goes away.

**The split is `quiet_fit.carve_holdin`.** The priority lineage used a 10% carve
keyed on `text_sha256`; the steady-aim lineage used a 15% carve keyed on a hash
of corpus name and row ordinal. Two models fitted on different rows are not a
head-to-head, so both lineages use the steady-aim carve here and therefore fit
on exactly the same 451,548 rows and select operating points on exactly the same
79,883.

Inside the calibration half, the priority recipe's own two-way sub-carve is
preserved at its original proportions: 20% (`calA`) fits the ASL calibration and
ranks the fusion options, 80% (`calB`) carries the threshold bank and scores the
fusion trials. Fitting a calibration and then selecting a threshold against it
on the same rows is the one shortcut that would quietly inflate every number
downstream.
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

from training.h2h_cache import CACHE_ROOT, N_FEATURES, PROFILES  # noqa: E402
from training.priority_data import PRIORITY_TAGS  # noqa: E402
from training.priority_hash import build_weights  # noqa: E402
from training.quiet_cache import load_catalogue  # noqa: E402
from training.quiet_fit import Dataset, accumulate, carve_holdin  # noqa: E402
from training.quiet_data import EVAL_ROOT, TRAIN_ROOT, list_dataset_dirs  # noqa: E402
from training.tune_priority_embeddingbag import (  # noqa: E402
    RANKS, factorize_ranks, fit_asl_calibration,
)
from training.tune_priority_embeddingbag import _thresholds as _emb_thresholds  # noqa: E402
from training.tune_priority_embeddingbag import trial_configs as emb_trial_configs  # noqa: E402
from training.tune_priority_fusion import (  # noqa: E402
    OPTIONS, fuse_predictions, rank_options, strategy_configs,
)
from training.tune_priority_hash import (  # noqa: E402
    _objective, _thresholds_for_trial, fast_metrics, threshold_bank, trial_configs,
)
from training.tune_priority_tfidf import build_tfidf_weights  # noqa: E402

PROJECT = Path("/home/lence/workspace/pii_master/projects/pii-head-to-head-v1")
#: The proportion of the calibration half used to fit calibrations and rank
#: fusion options, leaving the rest to select thresholds and score trials.
CAL_A_FRACTION = 0.20
SCORE_MODES = ("top1", "top3", "top6")


# --------------------------------------------------------------------- loading
def load_h2h(names: list[str], profile: str) -> Dataset:
    """The `h2h_cache` equivalent of `quiet_fit.load`, same `Dataset` shape.

    `uid_hash` is reproduced exactly as `quiet_fit.load` builds it -- corpus name
    blake2b seed, row ordinal, same multiplier -- because `carve_holdin` reads
    it, and the whole point is that both lineages get the same partition.
    """
    import hashlib

    cat = load_catalogue()
    labels = tuple(cat["labels"])
    n_labels = len(labels)
    Xs, Ys, tgt, comp, corp, uids = [], [], [], [], [], []
    for ci, name in enumerate(names):
        with np.load(CACHE_ROOT / f"{name}.npz", allow_pickle=False) as z:
            indptr, indices = z[f"indptr_{profile}"], z[f"indices_{profile}"]
            Xs.append(sp.csr_matrix(
                (np.ones(len(indices), dtype=np.float32), indices, indptr),
                shape=(len(indptr) - 1, N_FEATURES)))
            lab_indptr, lab_cols = z["label_indptr"], z["label_cols"]
            Ys.append(sp.csr_matrix(
                (np.ones(len(lab_cols), dtype=np.float32), lab_cols, lab_indptr),
                shape=(len(lab_indptr) - 1, n_labels)))
            tgt.append(z["doc_target"])
            comp.append(z["tag_complete"])
            n = len(z["doc_target"])
            corp.append(np.full(n, ci, dtype=np.int16))
            seed = hashlib.blake2b(name.encode(), digest_size=8).digest()
            base = int.from_bytes(seed, "little")
            uids.append(((np.arange(n, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15))
                         ^ np.uint64(base)))
    return Dataset(
        X=sp.vstack(Xs, format="csr"), Y=sp.vstack(Ys, format="csr"),
        doc_target=np.concatenate(tgt), tag_complete=np.concatenate(comp),
        corpus=np.concatenate(corp), corpus_names=tuple(names),
        uid_hash=np.concatenate(uids), labels=labels,
    )


def train_corpora() -> list[str]:
    return [d.name for d in list_dataset_dirs(TRAIN_ROOT)]


def eval_corpora() -> list[str]:
    return [d.name for d in list_dataset_dirs(EVAL_ROOT)]


def carve_calibration(ds: Dataset, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split the calibration half into calA (fit) and calB (select).

    Re-mixed before taking the modulus. `carve_holdin` already selected this
    half with `uid_hash % 10_000 < 1_500`, so a second plain `% 1_000` is
    correlated with the first cut and would hand calA 27% of the rows rather
    than 20% -- deterministic and disjoint either way, but not the split it
    claims to be.
    """
    mixed = ds.uid_hash * np.uint64(0xBF58476D1CE4E5B9)
    bucket = ((mixed >> np.uint64(17)) % np.uint64(1_000)).astype(np.int64)
    a = mask & (bucket < int(CAL_A_FRACTION * 1_000))
    return a, mask & ~a


def per_corpus_catalogue() -> dict[str, dict[str, Any]]:
    """`fast_metrics`'s `data_quality` argument, from the frozen catalogue."""
    cat = load_catalogue()
    return {name: {"tag_counts": counts}
            for name, counts in cat["per_corpus_counts"].items()}


# --------------------------------------------------------------------- scoring
def score_top_modes(X: sp.csr_matrix, W: np.ndarray, *,
                    chunk: int = 4_096, label: str = "") -> dict[str, np.ndarray]:
    """`priority_hash.score_modes` over a whole CSR matrix, batched.

    Identical arithmetic to the per-document version the tuners call: clamp the
    weights of the document's own features at zero, then take the mean of the
    top k. Batched only so 500,000 documents do not cost 500,000 Python frames.
    """
    n_docs, n_labels = X.shape[0], W.shape[0]
    out = {m: np.zeros((n_docs, n_labels), dtype=np.float32) for m in SCORE_MODES}
    Wt = np.ascontiguousarray(W.T)
    started = time.perf_counter()
    for lo in range(0, n_docs, chunk):
        hi = min(lo + chunk, n_docs)
        for i in range(lo, hi):
            cols = X.indices[X.indptr[i]:X.indptr[i + 1]]
            if not len(cols):
                continue
            vals = np.maximum(Wt[cols], 0.0)          # (n_cols, n_labels)
            out["top1"][i] = vals.max(axis=0)
            for k in (3, 6):
                kk = min(k, vals.shape[0])
                out[f"top{k}"][i] = np.partition(vals, -kk, axis=0)[-kk:].mean(axis=0)
        if label and (hi % (chunk * 10) == 0 or hi == n_docs):
            rate = hi / max(time.perf_counter() - started, 1e-9)
            print(f"    {label}: {hi:,}/{n_docs:,} ({rate:,.0f} docs/s)",
                  file=sys.stderr, flush=True)
    return out


def score_embeddingbag(X: sp.csr_matrix, factors: dict[int, tuple[np.ndarray, np.ndarray]],
                       *, label: str = "") -> dict[str, np.ndarray]:
    """Raw (uncalibrated) low-rank bag scores for every rank, batched."""
    # Iterate the ranks actually handed in, not the module's full ladder: the
    # search wants all four, serving wants only the one the winner chose.
    ranks = sorted(factors)
    max_rank = max(ranks)
    max_embeddings = factors[max_rank][0]
    n_docs = X.shape[0]
    bags = np.zeros((n_docs, max_rank), dtype=np.float32)
    started = time.perf_counter()
    for i in range(n_docs):
        cols = X.indices[X.indptr[i]:X.indptr[i + 1]]
        if len(cols):
            bags[i] = max_embeddings[cols].mean(axis=0)
        if label and (i + 1) % 100_000 == 0:
            rate = (i + 1) / max(time.perf_counter() - started, 1e-9)
            print(f"    {label}: {i+1:,}/{n_docs:,} ({rate:,.0f} docs/s)",
                  file=sys.stderr, flush=True)
    return {f"rank{r}": (bags[:, :r] @ factors[r][1].T).astype(np.float32) for r in ranks}


# ------------------------------------------------------------------- artifacts
def _dense_labels(Y: sp.csr_matrix) -> np.ndarray:
    return np.asarray(Y.todense()).astype(bool)


def _save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def _run_ladder(family: str, scores: dict[str, np.ndarray], bank: dict[str, np.ndarray],
                y_true: np.ndarray, datasets: np.ndarray, complete: np.ndarray,
                labels: tuple[str, ...], quality: dict[str, Any],
                configs: list[dict[str, Any]], thresholds_fn) -> dict[str, Any]:
    """One family's trial ladder: threshold lookup, predict, score, rank."""
    import mlflow

    out_dir = PROJECT / "tuning" / family
    out_dir.mkdir(parents=True, exist_ok=True)
    best: dict[str, Any] | None = None
    records: list[dict[str, Any]] = []
    with mlflow.start_run(run_name=f"tune-{family}"):
        mlflow.log_params({"family": family, "trials": len(configs), "cpu_budget": "all"})
        for number, config in enumerate(configs):
            thr = thresholds_fn(config, bank, labels)
            predicted = scores[config["score_mode"]] >= thr
            metrics = fast_metrics(predicted, y_true, datasets, complete, labels, quality)
            objective = _objective(metrics)
            record = {"trial": number, "family": family, "config": config,
                      "metrics": {k: v for k, v in metrics.items() if k != "per_corpus"},
                      "per_corpus": metrics["per_corpus"],
                      "objective": list(objective),
                      # Every trial keeps its own operating point. The fusion
                      # takes two different points off this one ladder -- an
                      # F0.5-shaped head and a recall-shaped one, exactly as the
                      # shipped lineage's `hash_sgd` / `hash_sgd_f2` pair did --
                      # and a trial that recorded only its config could not be
                      # rebuilt without re-deriving the whole bank.
                      "thresholds": thr.tolist()}
            records.append(record)
            with mlflow.start_run(nested=True, run_name=f"{family}-{number:03d}"):
                mlflow.log_params({k: str(v) for k, v in config.items()})
                mlflow.log_metrics({
                    "equal_corpus_macro_f2": metrics["equal_corpus_macro_f2"],
                    "equal_corpus_micro_f1": metrics["equal_corpus_micro_f1"],
                    "worst_priority_recall": metrics["worst_priority_recall"],
                    "priority_point_passes": float(metrics["priority_point_passes"]),
                    "measurable_priority_gates": float(metrics["measurable_priority_gates"]),
                    "feasible": float(objective[0]),
                })
            if best is None or objective > tuple(best["objective"]):
                best = record
                best["thresholds"] = thr.tolist()
            if (number + 1) % 25 == 0 or number + 1 == len(configs):
                # Checkpoint every 25 trials: a search that dies late must not
                # take its results with it.
                _save(out_dir / "trials.json", records)
                _save(out_dir / "best.json", best)
    _save(out_dir / "trials.json", records)
    _save(out_dir / "best.json", best)
    return best


# ------------------------------------------------------------------------ main
def cmd_counts(args: argparse.Namespace) -> int:
    ds = load_h2h(train_corpora(), profile="train20k")
    fit_mask, calib_mask = carve_holdin(ds)
    print(f"rows {len(ds):,}  fit {int(fit_mask.sum()):,}  calib {int(calib_mask.sum()):,}",
          flush=True)
    counts = accumulate(ds, fit_mask)
    out = PROJECT / "cache" / "counts.npz"
    counts.save(out)
    _save(PROJECT / "cache" / "counts_stats.json", {
        "n_all": counts.n_all, "n_complete": counts.n_complete,
        "fit_rows": int(fit_mask.sum()), "calibration_rows": int(calib_mask.sum()),
        "n_features": counts.n_features, "n_labels": len(counts.labels),
        "profile": "train20k",
    })
    print(f"counts -> {out}  n_all={counts.n_all:,} n_complete={counts.n_complete:,}")
    return 0


def _prepare(profile: str = "train20k"):
    """Dataset, masks and the calibration score substrate every family shares."""
    from training.quiet_fit import Counts

    ds = load_h2h(train_corpora(), profile=profile)
    fit_mask, calib_mask = carve_holdin(ds)
    calA_mask, calB_mask = carve_calibration(ds, calib_mask)
    counts = Counts.load(PROJECT / "cache" / "counts.npz")
    names = np.asarray(ds.corpus_names)
    return {
        "ds": ds, "counts": counts,
        "fit": fit_mask, "calib": calib_mask, "calA": calA_mask, "calB": calB_mask,
        "labels": ds.labels,
        "datasets_all": names[ds.corpus],
        "quality": per_corpus_catalogue(),
    }


def cmd_family(args: argparse.Namespace) -> int:
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{PROJECT / 'mlflow.db'}")
    mlflow.set_experiment("pii-head-to-head-v1")
    p = _prepare()
    ds, counts, labels = p["ds"], p["counts"], p["labels"]
    calA, calB = p["calA"], p["calB"]

    if args.name == "hash":
        W = build_weights(counts, alpha=1.0, partial_weight=0.75,
                          min_document_frequency=3)
    elif args.name == "tfidf":
        W = build_tfidf_weights(counts, idf_power=0.65, alpha=1.0,
                                partial_weight=0.75, min_document_frequency=3)
    else:
        raise SystemExit(f"unknown counting family: {args.name}")
    np.save(PROJECT / "cache" / f"W_{args.name}.npy", W)

    print(f"scoring calB ({int(calB.sum()):,} rows) ...", file=sys.stderr, flush=True)
    S = score_top_modes(ds.X[calB], W, label=f"{args.name}/calB")
    Y = _dense_labels(ds.Y[calB])
    datasets = p["datasets_all"][calB]
    complete = ds.tag_complete[calB]
    bank = threshold_bank(S, Y, datasets, labels)

    best = _run_ladder(args.name, S, bank, Y, datasets, complete, labels,
                       p["quality"], trial_configs(n_trials=args.trials),
                       _thresholds_for_trial)
    m = best["metrics"]
    print(f"{args.name}: {args.trials} trials, best trial {best['trial']}")
    print(f"  macro_f2={m['equal_corpus_macro_f2']:.4f} micro_f1={m['equal_corpus_micro_f1']:.4f} "
          f"gates {m['priority_point_passes']}/{m['measurable_priority_gates']} "
          f"worst_recall={m['worst_priority_recall']:.4f}")
    return 0


def cmd_embed(args: argparse.Namespace) -> int:
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{PROJECT / 'mlflow.db'}")
    mlflow.set_experiment("pii-head-to-head-v1")
    p = _prepare()
    ds, counts, labels = p["ds"], p["counts"], p["labels"]
    calA, calB = p["calA"], p["calB"]

    # The source of the factorisation is the lineage's own choice: tf-idf
    # weights at idf_power 0.35, not the 0.65 the tfidf component ships with.
    source = build_tfidf_weights(counts, idf_power=0.35, alpha=1.0,
                                 partial_weight=0.75, min_document_frequency=3)
    factors = factorize_ranks(source)
    np.savez_compressed(PROJECT / "cache" / "embed_factors.npz",
                        **{f"emb{r}": factors[r][0] for r in RANKS},
                        **{f"head{r}": factors[r][1] for r in RANKS})

    both = calA | calB
    print(f"scoring calA+calB ({int(both.sum()):,} rows) ...", file=sys.stderr, flush=True)
    raw_all = score_embeddingbag(ds.X[both], factors, label="embed")
    Y_all = _dense_labels(ds.Y[both])
    complete_all = ds.tag_complete[both]
    datasets_all = p["datasets_all"][both]
    # Which entries of the label matrix gold can actually speak to.
    observed = complete_all[:, None] | Y_all
    is_a = calA[both]

    calibrated: dict[str, np.ndarray] = {}
    cals: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for r in RANKS:
        mode = f"rank{r}"
        cal, bias, loss = fit_asl_calibration(
            raw_all[mode][is_a], Y_all[is_a], observed[is_a])
        cals[r] = (cal, bias)
        calibrated[mode] = raw_all[mode] @ cal.T + bias
        print(json.dumps({"phase": "asl", "rank": r, "loss": loss}), flush=True)
    np.savez_compressed(PROJECT / "cache" / "embed_calibration.npz",
                        **{f"cal{r}": cals[r][0] for r in RANKS},
                        **{f"bias{r}": cals[r][1] for r in RANKS})

    sel = ~is_a
    S = {m: v[sel] for m, v in calibrated.items()}
    Y, datasets, complete = Y_all[sel], datasets_all[sel], complete_all[sel]
    bank = threshold_bank(S, Y, datasets, labels)
    best = _run_ladder("embed", S, bank, Y, datasets, complete, labels,
                       p["quality"], emb_trial_configs(n_trials=args.trials),
                       _emb_thresholds)
    m = best["metrics"]
    print(f"embed: {args.trials} trials, best trial {best['trial']}")
    print(f"  macro_f2={m['equal_corpus_macro_f2']:.4f} micro_f1={m['equal_corpus_micro_f1']:.4f} "
          f"gates {m['priority_point_passes']}/{m['measurable_priority_gates']}")
    return 0


COMMANDS = {"counts": cmd_counts, "family": cmd_family, "embed": cmd_embed}


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("counts")
    f = sub.add_parser("family")
    f.add_argument("--name", required=True, choices=["hash", "tfidf"])
    f.add_argument("--trials", type=int, default=300)
    e = sub.add_parser("embed")
    e.add_argument("--trials", type=int, default=300)
    args = ap.parse_args()
    return COMMANDS[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
