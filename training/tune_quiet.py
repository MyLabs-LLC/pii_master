"""The search. Four families, one budget, one objective per family.

Every trial is an MLflow child run under one parent per family, and every trial
is scored on **held-in calibration** data carved from the training corpora. The
eight sealed evaluation directories are not reachable from this module; they are
scored once, later, by the evaluator, on the finalists only.

The families exist because they are different *mechanisms*, not different
hyperparameters of one:

``docgate``      a discriminative binary "does this contain sensitive PII" head.
                 The mechanism the prior lineage never had, and the one the
                 document-level gates depend on.
``tagcount``     per-tag Bernoulli log-odds from counts accumulated once, with
                 per-label F0.5 operating points. Cheap, so it can afford to
                 explore operating points densely.
``tagdisc``      per-tag discriminative heads. More expensive per fit, so fits
                 are cached and many operating points are explored per fit --
                 which is the only way this family fits in the budget at all.
``cascade``      the promotion candidate: a gate from the first family in front
                 of heads from one of the others, with the joint operating point
                 selected together rather than bolted on.

Fit caching is what makes the accounting honest. A trial is one *evaluated
configuration*; several configurations that share an expensive fit reuse it
rather than refitting, and the budget is spent on configurations rather than on
recomputing the same weights.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow  # noqa: E402
import optuna  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.linear_model import SGDClassifier  # noqa: E402

from training.quiet_fit import (  # noqa: E402
    accumulate, build_weights, carve_holdin, load, priority_indices, score, train_corpora,
)
from training.quiet_objective import Score, evaluate, evaluate_gate, group_masks  # noqa: E402
from training.quiet_select import select_doc_threshold, select_per_label  # noqa: E402

PROJECT = Path("/home/lence/workspace/pii_master/projects/pii-quiet-alarm")
TUNING = PROJECT / "tuning"
PROFILES = ("fast", "std", "deep")
#: Labels are independent; fitting them in parallel is what keeps the
#: discriminative family inside the budget rather than dominating it.
HEAD_JOBS = 8

optuna.logging.set_verbosity(optuna.logging.WARNING)


# --------------------------------------------------------------------- state
class Workspace:
    """Cached datasets, counts and fits, shared across trials of one family."""

    def __init__(self) -> None:
        self._ds: dict[str, Any] = {}
        self._counts: dict[tuple, Any] = {}
        self._gates: dict[tuple, np.ndarray] = {}
        self._heads: dict[tuple, np.ndarray] = {}
        self.fits = 0

    def data(self, profile: str):
        if profile not in self._ds:
            ds = load(train_corpora(), profile=profile)
            fit_mask, calib_mask = carve_holdin(ds)
            calib = ds.subset(calib_mask)
            self._ds[profile] = {
                "ds": ds, "fit": fit_mask, "calib_mask": calib_mask, "calib": calib,
                "Ycal": np.asarray(calib.Y.todense()).astype(bool),
                "groups": group_masks(calib.corpus, calib.corpus_names),
                "priority": priority_indices(calib.labels),
            }
        return self._ds[profile]

    def counts(self, profile: str):
        if profile not in self._counts:
            d = self.data(profile)
            self._counts[profile] = accumulate(d["ds"], d["fit"])
        return self._counts[profile]

    def gate(self, profile: str, alpha: float, loss: str, max_iter: int,
             neg_weight: float) -> np.ndarray:
        key = (profile, round(alpha, 12), loss, max_iter, round(neg_weight, 4))
        if key not in self._gates:
            d = self.data(profile)
            ds, fit = d["ds"], d["fit"]
            known = ds.doc_target >= 0
            rows = fit & known
            y = ds.doc_target[rows].astype(np.int8)
            w = np.where(y == 0, neg_weight, 1.0)
            clf = SGDClassifier(loss=loss, alpha=alpha, max_iter=max_iter, tol=None,
                                random_state=7, n_jobs=-1)
            clf.fit(ds.X[rows], y, sample_weight=w)
            self._gates[key] = clf.decision_function(d["calib"].X).astype(np.float32)
            self._coef = clf.coef_.astype(np.float32)
            self._intercept = float(clf.intercept_[0])
            self._gate_meta = {"profile": profile, "alpha": alpha, "loss": loss,
                               "max_iter": max_iter, "neg_weight": neg_weight}
            self.fits += 1
        return self._gates[key]

    def heads(self, profile: str, alpha: float, max_iter: int, pos_weight: float) -> np.ndarray:
        """One-vs-rest discriminative per-tag scores on the calibration split.

        Two things make 58 fits affordable rather than the family's whole
        budget. Ineligible rows -- those whose positive-only gold cannot supply
        a negative -- are given **zero sample weight** instead of being sliced
        out, which avoids materialising 58 copies of a 450,000-row sparse
        matrix. And the labels are independent, so they are fitted across
        processes; the serial loop was the only thing keeping 30 cores idle.
        """
        key = (profile, round(alpha, 12), max_iter, round(pos_weight, 4))
        if key not in self._heads:
            d = self.data(profile)
            ds, fit = d["ds"], d["fit"]
            X, Y = ds.X[fit], ds.Y[fit]
            complete = ds.tag_complete[fit]
            Xc = d["calib"].X
            Yd = np.asarray(Y.todense()).astype(bool)

            def one(j: int) -> np.ndarray:
                pos = Yd[:, j]
                eligible = pos | complete
                if int(pos.sum()) < 20 or not (~pos & eligible).any():
                    return np.full(Xc.shape[0], -1e9, dtype=np.float32)
                w = np.where(eligible, np.where(pos, pos_weight, 1.0), 0.0)
                clf = SGDClassifier(loss="log_loss", alpha=alpha, max_iter=max_iter,
                                    tol=None, random_state=11)
                clf.fit(X, pos.astype(np.int8), sample_weight=w)
                return clf.decision_function(Xc).ravel().astype(np.float32)

            # `threading`, not `loky`: scikit-learn's SGD releases the GIL in its
            # inner loop, and a thread shares the 450,000-row sparse matrix
            # instead of pickling a copy of it into every worker. The copies
            # were costing more than the parallelism bought.
            cols = Parallel(n_jobs=HEAD_JOBS, backend="threading")(
                delayed(one)(j) for j in range(Yd.shape[1]))
            self._heads[key] = np.stack(cols, axis=1)
            self.fits += 1
        return self._heads[key]


# ------------------------------------------------------------------ families
def trial_docgate(t: optuna.Trial, ws: Workspace) -> tuple[Score, dict]:
    profile = t.suggest_categorical("profile", PROFILES)
    alpha = t.suggest_float("alpha", 1e-8, 1e-3, log=True)
    loss = t.suggest_categorical("loss", ["log_loss", "modified_huber"])
    max_iter = t.suggest_int("max_iter", 6, 30)
    neg_weight = t.suggest_float("neg_weight", 0.5, 12.0, log=True)
    select_on = t.suggest_categorical("select_on", ["real", "all"])
    rec_target = t.suggest_float("recall_target", 0.85, 0.97)
    spec_target = t.suggest_float("spec_target", 0.85, 0.99)

    d = ws.data(profile)
    g = ws.gate(profile, alpha, loss, max_iter, neg_weight)
    calib = d["calib"]
    sel = d["groups"]["real"] if select_on == "real" else np.ones(len(g), dtype=bool)
    cut, _ = select_doc_threshold(g[sel], calib.doc_target[sel],
                                  recall_floor=rec_target, specificity_floor=spec_target)
    s = evaluate_gate(g, cut, calib.doc_target, d["groups"])
    return s, {"gate_threshold": float(cut), "profile": profile}


def trial_tagcount(t: optuna.Trial, ws: Workspace) -> tuple[Score, dict]:
    profile = t.suggest_categorical("profile", PROFILES)
    alpha = t.suggest_float("alpha", 0.05, 8.0, log=True)
    partial_weight = t.suggest_float("partial_weight", 0.0, 1.0)
    min_df = t.suggest_int("min_df", 1, 60, log=True)
    clip = t.suggest_float("clip", 2.0, 16.0)
    idf_power = t.suggest_float("idf_power", 0.0, 2.5)
    mode = t.suggest_categorical("score_mode", ["sum", "mean", "top3", "top6"])
    recall_floor = t.suggest_float("recall_floor", 0.75, 0.95)
    min_support = t.suggest_int("min_support_fit", 5, 200, log=True)

    d = ws.data(profile)
    W = build_weights(ws.counts(profile), alpha=alpha, partial_weight=partial_weight,
                      min_df=min_df, clip=clip, idf_power=idf_power)
    S = score(d["calib"].X, W, mode=mode)
    thr, _ = select_per_label(S, d["Ycal"], d["calib"].tag_complete,
                              beta=0.5, recall_floor=recall_floor, min_support=min_support)
    s = evaluate(S, thr, d["Ycal"], d["calib"].tag_complete, d["calib"].doc_target,
                 d["groups"], d["priority"], doc_constraints=False)
    return s, {"profile": profile, "score_mode": mode}


def trial_tagdisc(t: optuna.Trial, ws: Workspace) -> tuple[Score, dict]:
    profile = t.suggest_categorical("profile", PROFILES)
    alpha = t.suggest_categorical("alpha", [1e-7, 1e-6, 1e-5, 1e-4])
    max_iter = t.suggest_categorical("max_iter", [8, 15])
    pos_weight = t.suggest_categorical("pos_weight", [1.0, 3.0, 10.0])
    recall_floor = t.suggest_float("recall_floor", 0.75, 0.95)
    min_support = t.suggest_int("min_support_fit", 5, 200, log=True)

    d = ws.data(profile)
    S = ws.heads(profile, alpha, max_iter, pos_weight)
    thr, _ = select_per_label(S, d["Ycal"], d["calib"].tag_complete,
                              beta=0.5, recall_floor=recall_floor, min_support=min_support)
    s = evaluate(S, thr, d["Ycal"], d["calib"].tag_complete, d["calib"].doc_target,
                 d["groups"], d["priority"], doc_constraints=False)
    return s, {"profile": profile}


def _load_best(family: str, k: int = 8) -> list[dict]:
    path = TUNING / family / "best.json"
    if not path.is_file():
        raise SystemExit(f"{family} has not been tuned yet: {path} missing")
    return json.loads(path.read_text(encoding="utf-8"))[:k]


def trial_cascade(t: optuna.Trial, ws: Workspace) -> tuple[Score, dict]:
    gates = _load_best("docgate")
    heads = _load_best("tagcount") + _load_best("tagdisc")
    gi = t.suggest_int("gate_choice", 0, len(gates) - 1)
    hi = t.suggest_int("head_choice", 0, len(heads) - 1)
    gate_cfg, head_cfg = gates[gi], heads[hi]
    profile = gate_cfg["params"]["profile"]
    if head_cfg["params"]["profile"] != profile:
        # A cascade reads the document once; two stages cannot want two
        # different read profiles. Mismatched pairs are pruned, not silently
        # rescored under one of them.
        raise optuna.TrialPruned()

    d = ws.data(profile)
    g = ws.gate(profile, gate_cfg["params"]["alpha"], gate_cfg["params"]["loss"],
                gate_cfg["params"]["max_iter"], gate_cfg["params"]["neg_weight"])
    if head_cfg["family"] == "tagcount":
        p = head_cfg["params"]
        W = build_weights(ws.counts(profile), alpha=p["alpha"],
                          partial_weight=p["partial_weight"], min_df=p["min_df"],
                          clip=p["clip"], idf_power=p["idf_power"])
        S = score(d["calib"].X, W, mode=p["score_mode"])
    else:
        p = head_cfg["params"]
        S = ws.heads(profile, p["alpha"], p["max_iter"], p["pos_weight"])

    # The joint operating point: shifting the gate and re-cutting the tags
    # together, rather than freezing one and tuning the other.
    gate_shift = t.suggest_float("gate_shift", -3.0, 3.0)
    recall_floor = t.suggest_float("recall_floor", 0.75, 0.95)
    min_support = t.suggest_int("min_support_fit", 5, 200, log=True)
    cut = gate_cfg["extra"]["gate_threshold"] + gate_shift

    open_doc = g >= cut
    # Tag thresholds are chosen on the documents the gate lets through, because
    # that is the only population they will ever see at serving time.
    thr, _ = select_per_label(S[open_doc], d["Ycal"][open_doc],
                              d["calib"].tag_complete[open_doc],
                              beta=0.5, recall_floor=recall_floor, min_support=min_support)
    s = evaluate(S, thr, d["Ycal"], d["calib"].tag_complete, d["calib"].doc_target,
                 d["groups"], d["priority"], gate=g, gate_threshold=cut)
    return s, {"profile": profile, "gate_threshold": float(cut),
               "gate_source": gate_cfg["number"], "head_source": head_cfg["number"],
               "head_family": head_cfg["family"], "thresholds": thr.tolist()}


FAMILIES = {"docgate": trial_docgate, "tagcount": trial_tagcount,
            "tagdisc": trial_tagdisc, "cascade": trial_cascade}


# ---------------------------------------------------------------------- main
def _rank(records: list[dict]) -> list[dict]:
    return sorted(records, key=lambda r: (r["feasible"], r["metrics"]["objective"]),
                  reverse=True)


def _checkpoint(out: Path, records: list[dict]) -> None:
    (out / "trials.json").write_text(json.dumps(records, indent=1), encoding="utf-8")
    (out / "best.json").write_text(json.dumps(_rank(records)[:16], indent=1), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=sorted(FAMILIES))
    ap.add_argument("--trials", type=int, required=True)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--timeout-s", type=float, default=None)
    args = ap.parse_args()

    out = TUNING / args.family
    out.mkdir(parents=True, exist_ok=True)
    # The filesystem store is in maintenance mode in this MLflow; SQLite is the
    # backend it points at, and it keeps run history queryable for the report.
    PROJECT.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{PROJECT / 'mlflow.db'}")
    mlflow.set_experiment("pii-quiet-alarm")

    ws = Workspace()
    fn = FAMILIES[args.family]
    records: list[dict] = []
    started = time.time()

    with mlflow.start_run(run_name=f"tune-{args.family}") as parent:
        mlflow.log_params({"family": args.family, "trials": args.trials,
                           "seed": args.seed, "cpu_budget": "all"})

        def objective(t: optuna.Trial) -> float:
            with mlflow.start_run(nested=True, run_name=f"{args.family}-{t.number}"):
                t0 = time.time()
                s, extra = fn(t, ws)
                mlflow.log_params({k: str(v) for k, v in t.params.items()})
                mlflow.log_metrics({**s.as_metrics(), "seconds": time.time() - t0})
                records.append({
                    "number": t.number, "family": args.family, "params": dict(t.params),
                    "extra": {k: v for k, v in extra.items() if k != "thresholds"},
                    "metrics": s.as_metrics(), "feasible": s.feasible,
                    "deficits": s.deficits, "doc": s.doc,
                })
                if extra.get("thresholds") is not None:
                    (out / f"thresholds-{t.number}.json").write_text(
                        json.dumps(extra["thresholds"]), encoding="utf-8")
                # Checkpoint every trial. A four-hour search that dies at three
                # fifty must not take its results with it, and an interrupted
                # family still has to hand the cascade a usable `best.json`.
                _checkpoint(out, records)
                return s.objective

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=25),
        )
        study.optimize(objective, n_trials=args.trials, timeout=args.timeout_s,
                       show_progress_bar=False)

        feasible = [r for r in records if r["feasible"]]
        ranked = _rank(records)
        mlflow.log_metrics({
            "n_trials": len(records), "n_feasible": len(feasible),
            "n_fits": ws.fits, "wallclock_s": time.time() - started,
            "best_objective": ranked[0]["metrics"]["objective"] if ranked else 0.0,
        })
        _checkpoint(out, records)

    print(f"{args.family}: {len(records)} trials, {len(feasible)} feasible, "
          f"{ws.fits} underlying fits, {time.time()-started:.0f}s")
    if ranked:
        b = ranked[0]
        m = b["metrics"]
        print(f"  best objective {m['objective']:.4f} feasible={b['feasible']}")
        print(f"    priority F0.5={m.get('priority_macro_f05', 0):.4f} "
              f"P={m.get('priority_macro_precision', 0):.4f} "
              f"R={m.get('priority_macro_recall', 0):.4f} "
              f"minR={m.get('priority_min_recall', 0):.4f}")
        for g in ("real", "synth"):
            if f"doc_{g}_precision" in m:
                print(f"    doc[{g}] P={m[f'doc_{g}_precision']:.4f} "
                      f"R={m[f'doc_{g}_recall']:.4f} sp={m[f'doc_{g}_specificity']:.4f}")
        if b["deficits"]:
            print(f"    deficits: {b['deficits']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
