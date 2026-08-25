"""The search objective, written so it cannot disagree with the shipping policy.

`policy.yaml` decides what ships. This decides what the search climbs. If the
two differ, a thousand trials optimise something the gate will then reject, so
this module mirrors the policy exactly: the same five hard floors, the same
ranker, and the same refusal to let an unmeasurable scope count as a pass.

One difference is deliberate and is not a relaxation. During search everything
is measured on **held-in calibration** data carved from the training corpora,
never on the eight sealed evaluation directories. Held-in numbers run optimistic
-- that is what the sealed set exists to catch -- so the floors here are set at
the policy's own thresholds rather than below them.

Document-level floors are judged separately on **real** and **synthetic**
negatives. Pooling them lets 49,961 synthetic negatives outvote 20,639
real-world ones, and it is the real ones that predict the customer's corpus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from training.quiet_data import canonical_stem
from training.quiet_select import doc_metrics, fbeta

#: Training corpora whose documents are real files rather than generated text.
REAL_CORPORA = (
    "15986_datax-dualjudge-trainset-5.36k",
    "23693_govdocs2-dualjudge-train80-12.86k",
)

DOC_PRECISION_FLOOR = 0.90
DOC_SPECIFICITY_FLOOR = 0.85
DOC_RECALL_FLOOR = 0.85
PRIORITY_RECALL_FLOOR = 0.75


@dataclass
class Score:
    """What a trial achieved, and whether it would survive the gate."""

    objective: float
    feasible: bool
    priority_macro_f05: float = 0.0
    priority_macro_precision: float = 0.0
    priority_macro_recall: float = 0.0
    priority_min_recall: float = 0.0
    macro_f05: float = 0.0
    doc: dict[str, dict[str, float | None]] = field(default_factory=dict)
    deficits: dict[str, float] = field(default_factory=dict)
    n_priority_measurable: int = 0

    def as_metrics(self) -> dict[str, float]:
        out = {
            "objective": self.objective,
            "feasible": float(self.feasible),
            "priority_macro_f05": self.priority_macro_f05,
            "priority_macro_precision": self.priority_macro_precision,
            "priority_macro_recall": self.priority_macro_recall,
            "priority_min_recall": self.priority_min_recall,
            "macro_f05": self.macro_f05,
            "n_priority_measurable": float(self.n_priority_measurable),
            "total_deficit": float(sum(self.deficits.values())),
        }
        for group, m in self.doc.items():
            for k in ("precision", "recall", "specificity"):
                if m.get(k) is not None:
                    out[f"doc_{group}_{k}"] = float(m[k])
        return out


def group_masks(corpus: np.ndarray, corpus_names: tuple[str, ...]) -> dict[str, np.ndarray]:
    # Matched by stem: the folder names carry row counts an external pass
    # rewrites, and a literal match would quietly reclassify every real-world
    # document as synthetic without raising anything.
    wanted = {canonical_stem(n) for n in REAL_CORPORA}
    real = np.zeros(len(corpus), dtype=bool)
    for i, name in enumerate(corpus_names):
        if canonical_stem(name) in wanted:
            real |= corpus == i
    return {"real": real, "synth": ~real}


def evaluate(
    tag_scores: np.ndarray,
    thresholds: np.ndarray,
    Y: np.ndarray,
    tag_complete: np.ndarray,
    doc_target: np.ndarray,
    groups: dict[str, np.ndarray],
    priority: np.ndarray,
    *,
    gate: np.ndarray | None = None,
    gate_threshold: float = -np.inf,
    min_support: int = 30,
    doc_constraints: bool = True,
) -> Score:
    """Score one configuration exactly as the policy would read it.

    ``doc_constraints`` is off for the per-tag families, and that is a
    correction rather than a relaxation. A bare set of tag heads decides "this
    document has PII" by firing at least one tag, which is not the mechanism
    the document gates are meant to test; leaving the gates on made every
    head-only trial infeasible and ranked the family by a deficit no
    hyperparameter of it could close -- preferring heads that simply fire less
    over heads with better tag quality, which is the opposite of what the
    cascade needs from this family. The document floors are the gate's job and
    are enforced, with the gate present, in the cascade.
    """
    fired_tags = tag_scores >= thresholds[None, :]
    if gate is not None:
        # The cascade: per-tag output is suppressed on a document the gate
        # judged clean. This is what makes document precision a property of the
        # artifact rather than of a downstream filter nobody wrote.
        open_doc = gate >= gate_threshold
        fired_tags = fired_tags & open_doc[:, None]
        fired_doc = open_doc & fired_tags.any(axis=1)
    else:
        fired_doc = fired_tags.any(axis=1)

    # ---- per-tag, honouring the positive-unlabelled mask
    n_labels = tag_scores.shape[1]
    precision = np.zeros(n_labels)
    recall = np.zeros(n_labels)
    measurable = np.zeros(n_labels, dtype=bool)
    for j in range(n_labels):
        pos = Y[:, j]
        support = int(pos.sum())
        if support < min_support:
            continue
        measurable[j] = True
        eligible = pos | (tag_complete & ~pos)
        pred = fired_tags[:, j] & eligible
        tp = int((pred & pos).sum())
        fp = int((pred & ~pos).sum())
        precision[j] = tp / (tp + fp) if tp + fp else 0.0
        recall[j] = tp / support

    pri = np.asarray([j for j in priority if measurable[j]], dtype=np.int64)
    if not len(pri):
        return Score(objective=-1e9, feasible=False, deficits={"no_measurable_priority": 1.0})
    pri_p, pri_r = precision[pri], recall[pri]
    pri_f05 = fbeta(pri_p, pri_r, 0.5)
    all_idx = np.flatnonzero(measurable)
    macro_f05 = float(fbeta(precision[all_idx], recall[all_idx], 0.5).mean())

    # ---- document level, real and synthetic judged apart
    doc: dict[str, dict[str, float | None]] = {}
    deficits: dict[str, float] = {}
    for group, mask in groups.items():
        m = doc_metrics(fired_doc[mask], doc_target[mask])
        doc[group] = m
        if m["n"] == 0 or m["precision"] is None:
            # Unmeasurable is not a pass: a group with no negatives cannot
            # clear a precision floor, so it contributes no evidence either way.
            continue
        for key, floor in (("precision", DOC_PRECISION_FLOOR),
                           ("recall", DOC_RECALL_FLOOR),
                           ("specificity", DOC_SPECIFICITY_FLOOR)):
            value = m.get(key)
            if value is None:
                continue
            if doc_constraints and value < floor:
                deficits[f"doc_{group}_{key}"] = float(floor - value)

    worst_priority_recall = float(pri_r.min())
    if worst_priority_recall < PRIORITY_RECALL_FLOOR:
        deficits["priority_recall"] = PRIORITY_RECALL_FLOOR - worst_priority_recall

    feasible = not deficits
    priority_macro_f05 = float(pri_f05.mean())
    # Infeasible trials are ranked below every feasible one and among themselves
    # by how far they still are, so the search has a gradient to climb rather
    # than a flat plateau of rejection.
    objective = priority_macro_f05 if feasible else -float(sum(deficits.values()))

    return Score(
        objective=objective, feasible=feasible,
        priority_macro_f05=priority_macro_f05,
        priority_macro_precision=float(pri_p.mean()),
        priority_macro_recall=float(pri_r.mean()),
        priority_min_recall=worst_priority_recall,
        macro_f05=macro_f05, doc=doc, deficits=deficits,
        n_priority_measurable=int(len(pri)),
    )


def evaluate_gate(
    gate: np.ndarray,
    gate_threshold: float,
    doc_target: np.ndarray,
    groups: dict[str, np.ndarray],
) -> Score:
    """The document question alone, for the family whose only job it is.

    Ranked by the *worst* group's precision rather than the mean. The gate has
    to hold up on real business documents specifically; a mean lets a strong
    synthetic score paper over the group that matters.
    """
    fired = gate >= gate_threshold
    doc: dict[str, dict[str, float | None]] = {}
    deficits: dict[str, float] = {}
    precisions: list[float] = []
    for group, mask in groups.items():
        m = doc_metrics(fired[mask], doc_target[mask])
        doc[group] = m
        if m["n"] == 0 or m["precision"] is None:
            continue
        precisions.append(float(m["precision"]))
        for key, floor in (("precision", DOC_PRECISION_FLOOR),
                           ("recall", DOC_RECALL_FLOOR),
                           ("specificity", DOC_SPECIFICITY_FLOOR)):
            value = m.get(key)
            if value is not None and value < floor:
                deficits[f"doc_{group}_{key}"] = float(floor - value)
    if not precisions:
        return Score(objective=-1e9, feasible=False, deficits={"no_measurable_group": 1.0})
    feasible = not deficits
    worst = min(precisions)
    return Score(
        objective=worst if feasible else -float(sum(deficits.values())),
        feasible=feasible, doc=doc, deficits=deficits,
    )


__all__ = ["REAL_CORPORA", "Score", "evaluate", "evaluate_gate", "group_masks"]
