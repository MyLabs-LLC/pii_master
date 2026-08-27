"""Why does the document gate lose 20 points of recall on real-world documents?

Measured: arm B's gate scores 0.8564 document recall on held-in real-world
calibration and 0.6495-0.6790 on the sealed real-world corpora, while on
synthetic data the same gate moves 0.9348 -> 0.9311 -- no gap at all. Something
about real documents does not transfer, and every tag metric on real documents is
capped by it.

Three hypotheses produce that signature, and they need different fixes:

**(a) The weights overfit.** `alpha = 7.2e-7` over 262,144 features against
~40,000 real training documents is close to unregularised. If this is it, the
*ranking* of sealed real documents is poor, and it shows up threshold-free as a
drop in ROC-AUC between held-in and sealed.

**(b) The threshold does not transfer.** The cut is chosen to land on a recall
target; where the score distribution is steep, a small shift moves recall a lot.
If this is it, sealed AUC is fine and only the recall *at the calibration-chosen
cut* collapses -- so re-cutting on sealed scores would recover it.

**(c) The gold shifted.** `govdocs2` train and eval carry different document-level
prevalence (0.484 against 0.532), and the dual-judge labels were produced by
different judge pairs. If this is it, no amount of regularisation helps, because
the two halves are not asking the same question.

These are separated by measuring, per alpha and per source-balance setting:

* **AUC held-in vs sealed** -- threshold-free, so it isolates the weights (a);
* **recall at the held-in cut vs at the sealed oracle cut** -- the gap between
  them is the threshold's fault, not the model's (b);
* **prevalence and score-distribution shift** between the halves (c).

Nothing here touches the tag heads: this is one binary head, and the whole point
is that it caps everything downstream.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.linear_model import SGDClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from training.h2h_priority import PROJECT, eval_corpora  # noqa: E402
from training.h2h_score import _load_cached  # noqa: E402
from training.quiet_data import canonical_stem  # noqa: E402
from training.quiet_fit import carve_holdin, load, train_corpora  # noqa: E402
from training.quiet_model import QuietCascade  # noqa: E402

ARM_B = PROJECT / "models" / "cascade"
#: Corpora made of real files rather than generated text, by stem so a rename
#: cannot detach them.
REAL_STEMS = frozenset({"datax-dualjudge-trainset", "govdocs2-dualjudge-train80",
                        "datax-dualjudge-evalset", "govdocs2-dualjudge-eval20"})


def is_real(name: str) -> bool:
    return canonical_stem(name) in REAL_STEMS


def _recall_at(scores: np.ndarray, y: np.ndarray, cut: float) -> float:
    pos = scores[y]
    return float((pos >= cut).mean()) if pos.size else float("nan")


def _spec_at(scores: np.ndarray, y: np.ndarray, cut: float) -> float:
    neg = scores[~y]
    return float((neg < cut).mean()) if neg.size else float("nan")


def _cut_for_specificity(scores: np.ndarray, y: np.ndarray, target: float) -> float:
    """The cut that delivers exactly `target` specificity on these scores.

    Holding specificity fixed is what makes two recalls comparable: a recall
    quoted at a different false-alarm rate is a different measurement.
    """
    neg = scores[~y]
    return float(np.quantile(neg, target)) if neg.size else -np.inf


def fit_gate(X, y, w, *, alpha: float, loss: str, max_iter: int) -> tuple[np.ndarray, float]:
    clf = SGDClassifier(loss=loss, alpha=alpha, max_iter=max_iter, tol=None,
                        random_state=7)
    clf.fit(X, y, sample_weight=w)
    return clf.coef_.ravel().astype(np.float32), float(clf.intercept_[0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[7.209510351431974e-07, 1e-5, 1e-4, 1e-3, 1e-2])
    ap.add_argument("--balance", nargs="+", default=["none", "equal", "real2x"],
                    help="how much of the loss real-world documents get")
    ap.add_argument("--spec-target", type=float, default=0.95,
                    help="specificity all recalls are quoted at, so they compare")
    args = ap.parse_args()

    meta = json.loads((ARM_B / "model.json").read_text(encoding="utf-8"))["metadata"]
    gp = meta["gate_params"]
    print(f"arm B gate: alpha={gp['alpha']:.3g} loss={gp['loss']} "
          f"max_iter={gp['max_iter']} neg_weight={gp['neg_weight']:.4f}", flush=True)

    ds = load(train_corpora(), profile=meta["profile"])
    fit_mask, calib_mask = carve_holdin(ds)
    known = ds.doc_target >= 0
    names = np.asarray(ds.corpus_names)
    real_row = np.asarray([is_real(n) for n in names])[ds.corpus]

    rows = fit_mask & known
    y_fit = ds.doc_target[rows].astype(bool)
    Xfit = ds.X[rows]
    real_fit = real_row[rows]
    n_real, n_synth = int(real_fit.sum()), int((~real_fit).sum())
    print(f"fit rows with document gold: {len(y_fit):,}  "
          f"real {n_real:,} ({n_real/len(y_fit):.1%})  synthetic {n_synth:,}", flush=True)

    # ------------------------------------------------------- hypothesis (c)
    print("\n(c) has the gold shifted between the halves?", flush=True)
    calib_real = calib_mask & known & real_row
    print(f"    held-in real   prevalence {ds.doc_target[calib_real].astype(bool).mean():.4f} "
          f"(n={int(calib_real.sum()):,})")
    sealed: dict[str, dict[str, Any]] = {}
    for corpus in eval_corpora():
        cached = _load_cached(corpus, meta["profile"])
        t = cached["doc_target"]
        if (t >= 0).sum() == 0:
            continue
        sealed[corpus] = {"X": cached["X"], "y": t[t >= 0].astype(bool),
                          "mask": t >= 0, "real": is_real(corpus)}
        print(f"    sealed {'real ' if is_real(corpus) else 'synth'} "
              f"{corpus[:38]:<38} prevalence {sealed[corpus]['y'].mean():.4f} "
              f"(n={len(sealed[corpus]['y']):,})")

    # --------------------------------------------------- hypotheses (a)+(b)
    print(f"\n(a)+(b) all recalls quoted at specificity {args.spec_target:.2f}, "
          f"so they are comparable:", flush=True)
    hdr = (f"{'alpha':>10} {'balance':>8} | {'AUC held-in':>11} {'AUC sealed':>10} "
           f"{'dAUC':>7} | {'R held-in':>9} {'R sealed@own':>12} {'R sealed@held-in':>16}")
    print(hdr); print("-" * len(hdr))
    out: list[dict[str, Any]] = []
    for balance in args.balance:
        base = np.where(y_fit, 1.0, gp["neg_weight"])
        if balance == "equal":
            scale = np.where(real_fit, n_synth / max(n_real, 1), 1.0)
        elif balance == "real2x":
            scale = np.where(real_fit, 2.0 * n_synth / max(n_real, 1), 1.0)
        else:
            scale = np.ones_like(base)
        w = base * scale
        for alpha in args.alphas:
            coef, b = fit_gate(Xfit, y_fit.astype(np.int8), w,
                               alpha=alpha, loss=gp["loss"], max_iter=gp["max_iter"])

            def score(X):
                return (X @ coef + b).astype(np.float32)

            s_cal = score(ds.X[calib_real])
            y_cal = ds.doc_target[calib_real].astype(bool)
            auc_in = roc_auc_score(y_cal, s_cal) if y_cal.any() and not y_cal.all() else float("nan")
            cut_in = _cut_for_specificity(s_cal, y_cal, args.spec_target)
            r_in = _recall_at(s_cal, y_cal, cut_in)

            # Pool the sealed real corpora: that is the population the claim is about.
            ss, ys = [], []
            for corpus, d in sealed.items():
                if not d["real"]:
                    continue
                ss.append(score(d["X"][d["mask"]])); ys.append(d["y"])
            s_seal, y_seal = np.concatenate(ss), np.concatenate(ys)
            auc_out = roc_auc_score(y_seal, s_seal)
            # (b): recall if the cut were re-derived on sealed at the same specificity
            r_out_own = _recall_at(s_seal, y_seal,
                                   _cut_for_specificity(s_seal, y_seal, args.spec_target))
            # (a)+(b) combined: recall at the cut the held-in split chose
            r_out_in = _recall_at(s_seal, y_seal, cut_in)

            print(f"{alpha:>10.3g} {balance:>8} | {auc_in:>11.4f} {auc_out:>10.4f} "
                  f"{auc_out - auc_in:>7.4f} | {r_in:>9.4f} {r_out_own:>12.4f} "
                  f"{r_out_in:>16.4f}", flush=True)
            out.append({"alpha": alpha, "balance": balance, "auc_held_in": auc_in,
                        "auc_sealed_real": auc_out, "recall_held_in": r_in,
                        "recall_sealed_own_cut": r_out_own,
                        "recall_sealed_held_in_cut": r_out_in,
                        "spec_target": args.spec_target})

    path = PROJECT / "probe" / "gate_diagnosis.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"\n-> {path}")
    print("\nreading it: dAUC near 0 exonerates the weights (a). "
          "'R sealed@own' near 'R held-in' with 'R sealed@held-in' far below "
          "convicts the threshold (b). Both low convicts the data (c).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
