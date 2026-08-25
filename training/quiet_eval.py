"""Score a finalist against the eight sealed corpora, in the shape the policy reads.

This is the only module that touches `2-eval`, and it is deliberately separate
from everything the search can reach. It emits one `Arm` per model with:

* arm-level ``priority_macro_f05`` / ``macro_f05`` / ``p95_latency_ms``, the
  ranker and the serving constraint;
* a ``doc@<corpus>`` scope carrying document precision, recall and specificity
  with bootstrap intervals -- present **only** on the three corpora that hold
  genuine negatives, so the other five come back NOT_MEASURABLE rather than
  passing by absence;
* a ``<tag>@<corpus>`` scope per priority tag carrying recall with an interval
  and its support, so ``min_support: 30`` can exclude and name the thin pairs.

Every headline is the suite's equal-corpus mean, and a corpus contributes to a
precision-bearing mean only if its gold can measure one.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.quiet_data import PRIORITY_TAGS, canonical_stem  # noqa: E402
from training.quiet_select import fbeta  # noqa: E402

N_RESAMPLES = 1_000
CONFIDENCE = 0.95
MIN_SUPPORT = 30
#: Corpora whose gold can answer the document question in both directions.
DOC_MEASURABLE = (
    "30000_pii2_eval_25.15k",
    "4000_datax-dualjudge-evalset-1.32k",
    "6589_govdocs2-dualjudge-eval20-3.53k",
)
#: Matched by stem, so a directory rename cannot silently turn a corpus that
#: carries negatives into one the document gates report as NOT_MEASURABLE.
DOC_MEASURABLE_STEMS = frozenset(canonical_stem(n) for n in DOC_MEASURABLE)


def _ci(flags: np.ndarray, seed: int) -> tuple[float | None, float | None]:
    """Bootstrap interval over a per-document 0/1 outcome."""
    if flags.size == 0:
        return None, None
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, flags.size, size=(N_RESAMPLES, flags.size))
    means = flags[draws].mean(axis=1)
    a = (1.0 - CONFIDENCE) / 2.0
    return float(np.quantile(means, a)), float(np.quantile(means, 1.0 - a))


def _m(value: float | None, lo: float | None = None, hi: float | None = None,
       support: int | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"value": value, "ci_low": lo, "ci_high": hi}
    if support is not None:
        out["support"] = support
    return out


def evaluate_corpus(
    name: str,
    fired_tags: np.ndarray,
    fired_doc: np.ndarray,
    Y: np.ndarray,
    tag_complete: np.ndarray,
    doc_target: np.ndarray,
    labels: tuple[str, ...],
    seed: int,
) -> dict[str, Any]:
    """Per-corpus scopes and the per-corpus contributions to the headlines."""
    scopes: dict[str, dict[str, Any]] = {}

    # ---- document level, only where negatives exist
    if canonical_stem(name) in DOC_MEASURABLE_STEMS:
        known = doc_target >= 0
        f, g = fired_doc[known], doc_target[known].astype(bool)
        tp = int((f & g).sum())
        fp = int((~g & f).sum())
        fn = int((g & ~f).sum())
        tn = int((~g & ~f).sum())
        rec_lo, rec_hi = _ci(f[g].astype(float), seed)
        spec_lo, spec_hi = _ci((~f[~g]).astype(float), seed + 1)
        # Precision is a ratio over predicted-positives, so its interval comes
        # from resampling documents rather than from a per-document indicator.
        rng = np.random.default_rng(seed + 2)
        draws = rng.integers(0, f.size, size=(N_RESAMPLES, f.size))
        fd, gd = f[draws], g[draws]
        prec_boot = (fd & gd).sum(axis=1) / np.maximum(fd.sum(axis=1), 1)
        a = (1.0 - CONFIDENCE) / 2.0
        scopes[f"doc@{name}"] = {
            "doc_precision": _m(tp / (tp + fp) if tp + fp else None,
                                float(np.quantile(prec_boot, a)),
                                float(np.quantile(prec_boot, 1 - a)),
                                support=int(tp + fp)),
            "doc_recall": _m(tp / (tp + fn) if tp + fn else None, rec_lo, rec_hi,
                             support=int(tp + fn)),
            "doc_specificity": _m(tn / (tn + fp) if tn + fp else None, spec_lo, spec_hi,
                                  support=int(tn + fp)),
        }

    # ---- per-tag
    can_measure_precision = bool(tag_complete.any())
    per_tag: dict[str, dict[str, Any]] = {}
    for j, label in enumerate(labels):
        pos = Y[:, j]
        support = int(pos.sum())
        eligible = pos | (tag_complete & ~pos)
        pred = fired_tags[:, j] & eligible
        tp = int((pred & pos).sum())
        fp = int((pred & ~pos).sum())
        precision = (tp / (tp + fp)) if (can_measure_precision and tp + fp) else None
        recall = (tp / support) if support else None
        f05 = (float(fbeta(np.asarray([precision]), np.asarray([recall]), 0.5)[0])
               if precision is not None and recall is not None else None)
        per_tag[label] = {
            "support": support, "tp": tp, "fp": fp,
            "precision": precision, "recall": recall, "f05": f05,
            "predicted": int(fired_tags[:, j].sum()),
        }
        if label in PRIORITY_TAGS:
            if support >= MIN_SUPPORT:
                lo, hi = _ci(fired_tags[pos, j].astype(float), seed + 3 + j)
                scopes[f"{label}@{name}"] = {"recall": _m(recall, lo, hi, support=support)}
            else:
                # Named and excluded: "could not measure" is not "passed".
                scopes[f"{label}@{name}"] = {"recall": _m(None, None, None, support=support)}

    measurable = [v for v in per_tag.values()
                  if v["support"] >= MIN_SUPPORT and v["f05"] is not None]
    pri_measurable = [per_tag[t] for t in PRIORITY_TAGS
                      if t in per_tag and per_tag[t]["support"] >= MIN_SUPPORT
                      and per_tag[t]["f05"] is not None]
    mean = lambda rows, key: (float(np.mean([r[key] for r in rows])) if rows else None)  # noqa: E731
    summary = {
        "macro_f05": mean(measurable, "f05"),
        "priority_macro_f05": mean(pri_measurable, "f05"),
        "priority_macro_precision": mean(pri_measurable, "precision"),
        "priority_macro_recall": mean(pri_measurable, "recall"),
        "n_measurable_tags": len(measurable),
        "n_priority_measurable": len(pri_measurable),
        "prediction_rate": float(fired_tags.any(axis=1).mean()),
        "tags_predicted_zero_times": int((fired_tags.sum(axis=0) == 0).sum()),
    }
    return {"scopes": scopes, "per_tag": per_tag, "summary": summary,
            "n_rows": int(len(doc_target))}


def assemble_arm(name: str, label: str, per_corpus: dict[str, dict[str, Any]],
                 p95_latency_ms: float, docs_per_s: float,
                 extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Equal-corpus headlines plus every scope, as one Arm."""
    scopes: dict[str, Any] = {}
    for body in per_corpus.values():
        scopes.update(body["scopes"])

    def headline(key: str) -> float | None:
        vals = [b["summary"][key] for b in per_corpus.values()
                if b["summary"].get(key) is not None]
        return float(np.mean(vals)) if vals else None

    def doc_mean(metric: str) -> float | None:
        vals = [s[metric]["value"] for k, s in scopes.items()
                if k.startswith("doc@") and s[metric]["value"] is not None]
        return float(np.mean(vals)) if vals else None

    return {
        "name": name,
        "label": label,
        "metrics": {
            "priority_macro_f05": _m(headline("priority_macro_f05")),
            "macro_f05": _m(headline("macro_f05")),
            "priority_macro_precision": _m(headline("priority_macro_precision")),
            "priority_macro_recall": _m(headline("priority_macro_recall")),
            "equal_corpus_doc_precision": _m(doc_mean("doc_precision")),
            "equal_corpus_doc_specificity": _m(doc_mean("doc_specificity")),
            "equal_corpus_doc_recall": _m(doc_mean("doc_recall")),
            "p95_latency_ms": _m(p95_latency_ms),
            "docs_per_s": _m(docs_per_s),
            "prediction_rate": _m(headline("prediction_rate")),
        },
        "scopes": scopes,
        "extra": extra or {},
        "per_corpus": {k: dict(v["summary"], n_rows=v["n_rows"]) for k, v in per_corpus.items()},
    }


__all__ = ["DOC_MEASURABLE", "assemble_arm", "evaluate_corpus"]
