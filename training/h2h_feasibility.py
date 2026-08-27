"""Are the document-level bars reachable at all, before more budget is spent on them?

The precision-view policy asks for three things at once on real-world documents:
document precision >= 0.90, specificity >= 0.85 and recall >= 0.85, each
conclusively. Nothing has come close, and every dial tried so far -- per-tag
floor, selection beta, gate regularisation, source balancing, synthetic
near-miss negatives -- has moved the number by hundredths.

That pattern is worth taking seriously rather than tuning through. A scorer with
a given ranking quality has a **frontier**: for each specificity there is exactly
one best achievable recall, and no choice of threshold beats it. If the three
bars do not intersect that frontier, they are not a hard target, they are an
impossible one, and the right output is the number the evidence *does* support.

This walks the measured score distribution on the sealed real-world corpora and
reports, without fitting anything:

* the best recall available at each required specificity and precision;
* whether any single threshold satisfies all three bars simultaneously;
* the same for the improved (balanced, regularised) gate, so the verdict is not
  pinned to the weakest scorer measured.

`mp feasibility` records the verdict formally; this produces the evidence it
takes. A `target_infeasible` verdict is a real result and belongs in the report
with the reachable number beside it.
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

from training.h2h_gate_diag import is_real  # noqa: E402
from training.h2h_priority import PROJECT  # noqa: E402
from training.h2h_score import _load_cached  # noqa: E402
from training.quiet_fit import carve_holdin, load, train_corpora  # noqa: E402

SEALED_REAL = ("4000_datax-dualjudge-evalset-1.32k",
               "6589_govdocs2-dualjudge-eval20-3.53k")
PREC_BAR, SPEC_BAR, REC_BAR = 0.90, 0.85, 0.85


def frontier(scores: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """Every realisable operating point, descending by threshold."""
    order = np.argsort(-scores, kind="stable")
    s, g = scores[order], y[order]
    tp = np.cumsum(g)
    fp = np.cumsum(~g)
    last = np.r_[s[1:] != s[:-1], True]
    tp, fp, thr = tp[last], fp[last], s[last]
    n_pos, n_neg = int(g.sum()), int((~g).sum())
    return {"thr": thr,
            "recall": tp / max(n_pos, 1),
            "precision": tp / np.maximum(tp + fp, 1),
            "specificity": (n_neg - fp) / max(n_neg, 1)}


def probe(name: str, scores: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    f = frontier(scores, y)
    auc = roc_auc_score(y, scores)
    rec, prec, spec = f["recall"], f["precision"], f["specificity"]

    both = (prec >= PREC_BAR) & (spec >= SPEC_BAR)
    all_three = both & (rec >= REC_BAR)
    best_rec_at_bars = float(rec[both].max()) if both.any() else None
    best_prec_at_rec = float(prec[rec >= REC_BAR].max()) if (rec >= REC_BAR).any() else None
    best_spec_at_rec = float(spec[rec >= REC_BAR].max()) if (rec >= REC_BAR).any() else None

    print(f"\n=== {name}   (AUC {auc:.4f}, {int(y.sum()):,} positives / "
          f"{int((~y).sum()):,} negatives)")
    print(f"  all three bars simultaneously (P>={PREC_BAR}, sp>={SPEC_BAR}, "
          f"R>={REC_BAR}): {'REACHABLE' if all_three.any() else 'NOT REACHABLE'}")
    if best_rec_at_bars is not None:
        print(f"  best recall while holding P>={PREC_BAR} and sp>={SPEC_BAR}: "
              f"{best_rec_at_bars:.4f}   (bar is {REC_BAR}, "
              f"shortfall {REC_BAR - best_rec_at_bars:+.4f})")
    else:
        print(f"  no threshold reaches P>={PREC_BAR} with sp>={SPEC_BAR} at all")
    if best_prec_at_rec is not None:
        print(f"  best precision while holding R>={REC_BAR}: {best_prec_at_rec:.4f}"
              f"   (bar is {PREC_BAR}, shortfall {PREC_BAR - best_prec_at_rec:+.4f})")
        print(f"  best specificity while holding R>={REC_BAR}: {best_spec_at_rec:.4f}"
              f"   (bar is {SPEC_BAR}, shortfall {SPEC_BAR - best_spec_at_rec:+.4f})")
    # What the bars would have to be relaxed to.
    if not all_three.any() and both.any():
        print(f"  -> the reachable point closest to the policy holds P and sp and "
              f"delivers recall {best_rec_at_bars:.4f}")
    return {"name": name, "auc": float(auc), "reachable": bool(all_three.any()),
            "best_recall_at_precision_and_specificity_bars": best_rec_at_bars,
            "best_precision_at_recall_bar": best_prec_at_rec,
            "best_specificity_at_recall_bar": best_spec_at_rec,
            "n_pos": int(y.sum()), "n_neg": int((~y).sum())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=PROJECT / "probe" / "doc_feasibility.json")
    args = ap.parse_args()

    meta = json.loads((PROJECT / "models" / "cascade" / "model.json").read_text(
        encoding="utf-8"))["metadata"]
    gp = meta["gate_params"]

    ds = load(train_corpora(), profile="deep")
    fit_mask, _ = carve_holdin(ds)
    known = ds.doc_target >= 0
    names = np.asarray(ds.corpus_names)
    real_row = np.asarray([is_real(n) for n in names])[ds.corpus]
    rows = fit_mask & known
    y_fit = ds.doc_target[rows].astype(bool)
    rf = real_row[rows]
    nr, nsy = int(rf.sum()), int((~rf).sum())

    Xs, ys = [], []
    for corpus in SEALED_REAL:
        d = _load_cached(corpus, "deep")
        m = d["doc_target"] >= 0
        Xs.append(d["X"][m])
        ys.append(d["doc_target"][m].astype(bool))
    X_seal = sp.vstack(Xs, format="csr")
    y_seal = np.concatenate(ys)

    out = []
    for label, alpha, balance in (("arm B gate (as shipped)", gp["alpha"], "none"),
                                  ("balanced + regularised gate", 1e-2, "equal")):
        w = np.where(y_fit, 1.0, gp["neg_weight"])
        if balance == "equal":
            w = w * np.where(rf, nsy / max(nr, 1), 1.0)
        clf = SGDClassifier(loss=gp["loss"], alpha=alpha, max_iter=gp["max_iter"],
                            tol=None, random_state=7)
        clf.fit(ds.X[rows], y_fit.astype(np.int8), sample_weight=w)
        s = (X_seal @ clf.coef_.ravel().astype(np.float32)
             + float(clf.intercept_[0])).astype(np.float32)
        out.append(probe(label, s, y_seal))

    # An oracle bound: the best any scorer on these exact features could do is
    # bounded above by a model fitted ON the sealed set itself. If even that
    # cannot clear the bars, no amount of training data will.
    clf = SGDClassifier(loss="log_loss", alpha=1e-6, max_iter=30, tol=None,
                        random_state=1)
    clf.fit(X_seal, y_seal.astype(np.int8))
    s_oracle = (X_seal @ clf.coef_.ravel().astype(np.float32)
                + float(clf.intercept_[0])).astype(np.float32)
    out.append(probe("ORACLE — fitted on the sealed set itself (upper bound)",
                     s_oracle, y_seal))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"bars": {"precision": PREC_BAR,
                                             "specificity": SPEC_BAR,
                                             "recall": REC_BAR},
                                    "probes": out}, indent=1) + "\n", encoding="utf-8")
    print(f"\n-> {args.out}")
    print("\nreading it: the ORACLE row is the ceiling for this feature space. A bar "
          "the oracle misses is not a tuning problem.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
