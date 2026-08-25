"""Operating points: where to threshold, given that precision is now the point.

Two selections, and they answer different questions.

**Per tag.**  For each label independently, the threshold that maximises F0.5 --
precision weighted twice recall -- subject to a hard recall floor.  The floor is
what stops a precision-first objective from choosing "predict nothing", which is
its degenerate optimum on a rare tag.

**Per document.**  One threshold on the gate score, chosen against the
document-level question on the corpora that hold genuine negatives.

Both respect the masked/positive-unlabelled discipline: a row whose gold is
positive-only cannot supply a negative, so it is excluded from the denominator
of precision rather than counted as a correct silence.  Getting this wrong would
make every partial-label corpus look like a precision triumph.
"""

from __future__ import annotations

import numpy as np


def fbeta(precision: np.ndarray, recall: np.ndarray, beta: float) -> np.ndarray:
    b2 = beta * beta
    denom = b2 * precision + recall
    return np.where(denom > 0, (1 + b2) * precision * recall / np.maximum(denom, 1e-12), 0.0)


def sweep(scores: np.ndarray, positive: np.ndarray, eligible: np.ndarray
          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precision / recall / threshold along the full curve for one label.

    ``eligible`` marks rows that may act as a negative.  Positives are always
    eligible; unlabelled rows are not.
    """
    keep = positive | eligible
    s, p = scores[keep], positive[keep]
    if not len(s) or not p.any():
        return np.zeros(0), np.zeros(0), np.zeros(0)
    order = np.argsort(-s, kind="stable")
    s, p = s[order], p[order]
    tp = np.cumsum(p)
    fp = np.cumsum(~p)
    # Only cut between distinct scores; a threshold inside a tie is not realisable.
    last = np.r_[s[1:] != s[:-1], True]
    tp, fp, thr = tp[last], fp[last], s[last]
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / p.sum()
    return precision, recall, thr


def best_threshold(scores: np.ndarray, positive: np.ndarray, eligible: np.ndarray,
                   *, beta: float = 0.5, recall_floor: float = 0.0,
                   ) -> tuple[float, dict[str, float]]:
    """The F-beta-optimal cut that still clears the recall floor.

    When no cut clears the floor the least-bad one is returned and reported as
    ``floor_met: False`` -- silently returning the unconstrained optimum would
    hide a tag the model cannot hold up.
    """
    precision, recall, thr = sweep(scores, positive, eligible)
    if not len(thr):
        return float("inf"), {"precision": 0.0, "recall": 0.0, "f": 0.0,
                              "support": 0.0, "floor_met": False}
    f = fbeta(precision, recall, beta)
    ok = recall >= recall_floor
    if ok.any():
        idx = int(np.flatnonzero(ok)[np.argmax(f[ok])])
        floor_met = True
    else:
        idx = int(np.argmax(recall))
        floor_met = False
    return float(thr[idx]), {
        "precision": float(precision[idx]), "recall": float(recall[idx]),
        "f": float(f[idx]), "support": float(positive.sum()), "floor_met": floor_met,
    }


def select_per_label(S: np.ndarray, Y: np.ndarray, tag_complete: np.ndarray,
                     *, beta: float = 0.5, recall_floor: float = 0.0,
                     min_support: int = 5,
                     ) -> tuple[np.ndarray, list[dict[str, float]]]:
    """One threshold per label; +inf disables a label with too little evidence.

    Disabling is deliberate. A tag with four training positives cannot have a
    trustworthy operating point, and shipping a guess for it costs precision on
    every document it fires on.
    """
    n_labels = S.shape[1]
    thresholds = np.full(n_labels, np.inf, dtype=np.float32)
    report: list[dict[str, float]] = []
    for j in range(n_labels):
        positive = Y[:, j].astype(bool)
        if positive.sum() < min_support:
            report.append({"label": j, "precision": 0.0, "recall": 0.0, "f": 0.0,
                           "support": float(positive.sum()), "floor_met": False,
                           "disabled": True})
            continue
        thr, info = best_threshold(S[:, j], positive, tag_complete & ~positive,
                                   beta=beta, recall_floor=recall_floor)
        thresholds[j] = thr
        report.append({"label": j, **info, "disabled": False})
    return thresholds, report


def doc_metrics(fired: np.ndarray, target: np.ndarray) -> dict[str, float | None]:
    """Document-level confusion on rows whose gold can answer the question."""
    known = target >= 0
    f, g = fired[known], target[known].astype(bool)
    tp = int((f & g).sum()); fp = int((f & ~g).sum())
    fn = int((~f & g).sum()); tn = int((~f & ~g).sum())
    ratio = lambda a, b: (a / b) if b else None  # noqa: E731
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": tp + fp + fn + tn,
        "recall": ratio(tp, tp + fn), "precision": ratio(tp, tp + fp),
        "specificity": ratio(tn, tn + fp),
        "prevalence": ratio(tp + fn, tp + fp + fn + tn),
    }


def select_doc_threshold(gate: np.ndarray, target: np.ndarray, *,
                         recall_floor: float, specificity_floor: float,
                         ) -> tuple[float, dict[str, float | None]]:
    """The document cut: maximise precision subject to both floors."""
    known = target >= 0
    s, g = gate[known], target[known].astype(bool)
    if not len(s) or not g.any() or g.all():
        return -np.inf, doc_metrics(np.ones(len(gate), dtype=bool), target)
    order = np.argsort(-s, kind="stable")
    s, g = s[order], g[order]
    tp = np.cumsum(g); fp = np.cumsum(~g)
    last = np.r_[s[1:] != s[:-1], True]
    tp, fp, thr = tp[last], fp[last], s[last]
    n_pos, n_neg = int(g.sum()), int((~g).sum())
    recall = tp / n_pos
    specificity = (n_neg - fp) / n_neg
    precision = tp / np.maximum(tp + fp, 1)
    ok = (recall >= recall_floor) & (specificity >= specificity_floor)
    if ok.any():
        idx = int(np.flatnonzero(ok)[np.argmax(precision[ok])])
    else:
        # Nothing satisfies both; take the point closest to satisfying them, and
        # let the caller see that the floors were missed.
        deficit = np.maximum(recall_floor - recall, 0) + np.maximum(specificity_floor - specificity, 0)
        idx = int(np.argmin(deficit))
    cut = float(thr[idx])
    return cut, doc_metrics(gate >= cut, target)


__all__ = ["best_threshold", "doc_metrics", "fbeta", "select_doc_threshold",
           "select_per_label", "sweep"]
