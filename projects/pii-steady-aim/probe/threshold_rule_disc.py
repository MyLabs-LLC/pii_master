"""The same comparison on the family that actually wins.

The count-based heads pay heavily for the worst-group rule, but a threshold move
costs precision in proportion to how badly the score ranks positives against
negatives. The discriminative heads rank far better (held-in priority macro F0.5
0.93 against 0.71), so the same robustness should be much cheaper there. If it
is not, the fix trades away the precision the run exists to buy, and that has to
be known before 1,000 trials are spent on it.
"""

from __future__ import annotations

import sys

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import SGDClassifier

sys.path.insert(0, "/home/lence/workspace/pii_master")

from training.quiet_fit import carve_holdin, load, priority_indices, train_corpora  # noqa: E402
from training.quiet_select import (  # noqa: E402
    fbeta, select_per_label, select_per_label_robust,
)

FLOOR = 0.75


def fit_heads(ds, fit_mask, alpha=1e-6, max_iter=15, pos_weight=3.0, jobs=10):
    X, Y = ds.X[fit_mask], ds.Y[fit_mask]
    complete = ds.tag_complete[fit_mask]
    Yd = np.asarray(Y.todense()).astype(bool)
    Xc = ds.X[~fit_mask]

    def one(j):
        pos = Yd[:, j]
        eligible = pos | complete
        if int(pos.sum()) < 20 or not (~pos & eligible).any():
            return np.full(Xc.shape[0], -1e9, dtype=np.float32)
        w = np.where(eligible, np.where(pos, pos_weight, 1.0), 0.0)
        clf = SGDClassifier(loss="log_loss", alpha=alpha, max_iter=max_iter,
                            tol=None, random_state=11)
        clf.fit(X, pos.astype(np.int8), sample_weight=w)
        return clf.decision_function(Xc).ravel().astype(np.float32)

    return np.stack(Parallel(n_jobs=jobs, backend="threading")(
        delayed(one)(j) for j in range(Yd.shape[1])), axis=1)


def worst_group(S, thr, Y, source, priority):
    fired = S >= thr[None, :]
    worst, n_below = 1.0, 0
    for j in priority:
        pos = Y[:, j]
        for g in np.unique(source):
            m = pos & (source == g)
            if m.sum() < 30:
                continue
            r = float(fired[m, j].mean())
            worst = min(worst, r)
            n_below += r < FLOOR
    return worst, n_below


def main() -> int:
    ds = load(train_corpora(), profile="deep")
    fit_mask, calib_mask = carve_holdin(ds)
    calib = ds.subset(calib_mask)
    print("fitting 58 discriminative heads ...", file=sys.stderr, flush=True)
    S = fit_heads(ds, fit_mask)
    Y = np.asarray(calib.Y.todense()).astype(bool)
    pri = priority_indices(calib.labels)
    src = calib.corpus

    print(f"{'rule':<34}{'priF0.5':>9}{'priP':>8}{'priR':>8}"
          f"{'worst grp R':>13}{'pairs<0.75':>12}")
    for label, kw in (("pooled (pii-quiet-alarm)", None),
                      ("worst-group, margin 0.00", {"margin": 0.0}),
                      ("worst-group, margin 0.03", {"margin": 0.03}),
                      ("worst-group, margin 0.05", {"margin": 0.05}),
                      ("worst-group, margin 0.10", {"margin": 0.10})):
        if kw is None:
            thr, rep = select_per_label(S, Y, calib.tag_complete, beta=0.5,
                                        recall_floor=FLOOR)
        else:
            thr, rep = select_per_label_robust(S, Y, calib.tag_complete, src, beta=0.5,
                                               recall_floor=FLOOR, **kw)
        ok = [rep[j] for j in pri if not rep[j]["disabled"]]
        p = float(np.mean([r["precision"] for r in ok]))
        r_ = float(np.mean([r["recall"] for r in ok]))
        f = float(np.mean([fbeta(np.asarray([r["precision"]]), np.asarray([r["recall"]]),
                                 0.5)[0] for r in ok]))
        worst, n_below = worst_group(S, thr, Y, src, pri)
        print(f"{label:<34}{f:>9.4f}{p:>8.4f}{r_:>8.4f}{worst:>13.4f}{n_below:>12}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
