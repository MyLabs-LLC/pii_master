"""Does the worst-group threshold rule actually move the worst group?

The claim behind this run is narrow and testable: `pii-quiet-alarm` failed on
tag-corpus pairs sitting just under the floor because its thresholds were chosen
where *pooled* recall met the floor exactly, leaving nothing for source shift.
If that is right, requiring the floor per source should raise the worst source's
recall a lot and cost precision only a little.

This compares the two rules on the same scores, same labels, same split. It is
held-in only -- it says whether the mechanism works, not whether it ships.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/lence/workspace/pii_master")

from training.quiet_fit import (  # noqa: E402
    accumulate, build_weights, carve_holdin, load, priority_indices, score, train_corpora,
)
from training.quiet_select import (  # noqa: E402
    fbeta, select_per_label, select_per_label_robust,
)

FLOOR = 0.75


def worst_group_recall(S, thr, Y, source, priority) -> tuple[float, int, float]:
    """Min per-source recall over measurable priority tag x source pairs."""
    fired = S >= thr[None, :]
    worst, n_below, recalls = 1.0, 0, []
    for j in priority:
        pos = Y[:, j]
        for g in np.unique(source):
            m = pos & (source == g)
            if m.sum() < 30:
                continue
            r = float(fired[m, j].mean())
            recalls.append(r)
            worst = min(worst, r)
            n_below += r < FLOOR
    return worst, n_below, float(np.mean(recalls))


def main() -> int:
    ds = load(train_corpora(), profile="deep")
    fit_mask, calib_mask = carve_holdin(ds)
    calib = ds.subset(calib_mask)
    W = build_weights(accumulate(ds, fit_mask), alpha=1.0, partial_weight=0.5,
                      min_df=3, clip=8.0)
    S = score(calib.X, W, mode="sum")
    Y = np.asarray(calib.Y.todense()).astype(bool)
    pri = priority_indices(calib.labels)
    src = calib.corpus

    print(f"{'rule':<34}{'priF0.5':>9}{'priP':>8}{'priR':>8}"
          f"{'worst grp R':>13}{'pairs<0.75':>12}")
    rows = [("pooled (pii-quiet-alarm)", dict(fn=select_per_label, kw={})),
            ("worst-group, margin 0.00", dict(fn=select_per_label_robust, kw={"margin": 0.0})),
            ("worst-group, margin 0.05", dict(fn=select_per_label_robust, kw={"margin": 0.05})),
            ("worst-group, margin 0.10", dict(fn=select_per_label_robust, kw={"margin": 0.10})),
            ("worst-group, margin 0.15", dict(fn=select_per_label_robust, kw={"margin": 0.15}))]
    for label, spec in rows:
        if spec["fn"] is select_per_label:
            thr, rep = select_per_label(S, Y, calib.tag_complete, beta=0.5,
                                        recall_floor=FLOOR)
        else:
            thr, rep = select_per_label_robust(S, Y, calib.tag_complete, src, beta=0.5,
                                               recall_floor=FLOOR, **spec["kw"])
        ok = [rep[j] for j in pri if not rep[j]["disabled"]]
        p = np.mean([r["precision"] for r in ok])
        r_ = np.mean([r["recall"] for r in ok])
        f = np.mean([fbeta(np.asarray([r["precision"]]), np.asarray([r["recall"]]), 0.5)[0]
                     for r in ok])
        worst, n_below, _ = worst_group_recall(S, thr, Y, src, pri)
        print(f"{label:<34}{f:>9.4f}{p:>8.4f}{r_:>8.4f}{worst:>13.4f}{n_below:>12}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
