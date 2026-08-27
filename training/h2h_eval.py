"""The one fixed evaluator. Every arm is scored by this and nothing else.

It extends `quiet_eval`'s semantics rather than replacing them -- the masked
positive-unlabelled discipline is carried over exactly, because it is the part
that is easy to get wrong in a way that flatters everybody:

* a row whose tag gold is positive-only **cannot act as a negative**, so it is
  excluded from the denominator of precision rather than counted as a correct
  silence. Get this wrong and every partial-label corpus reports a precision
  triumph.
* a precision-bearing metric on a corpus with no complete tag gold is
  **NOT_MEASURABLE (`None`)**, never `0.0`. The two lead to opposite decisions
  and a run that averages the second as the first has published a number that
  means nothing.

What it adds is the full F-beta family the run was asked for -- F0.5, F1, F2,
F3, each in macro and micro form -- plus the per-tag table behind them.

## Two macro averages, both reported, neither hidden

`macro_*_catalogue` averages over **every tag the corpus's gold contains**, so a
tag the model has no head for, or never emits, counts as a real 0. This is the
domain profile's definition and it is the one that ranks arms: averaging over
only the tags a model happens to predict rewards it for being able to predict
less, which is backwards for a recall-first objective.

`macro_*_support30` averages over tags with at least 30 instances. Recall on a
tag with 19 instances moves 0.05 when one document changes, so this is the
number to read when asking what the model actually does; it is a diagnostic and
never a gate.

Reporting one without the other is how a dead tail hides.

## Confidence intervals

A multinomial bootstrap over documents (1,000 resamples, 95%). Per-document
tp/fp/fn contributions are accumulated once per tag, so a resample is a matrix
product rather than 1,000 rescorings -- which is what makes CIs affordable on
every headline instead of only on the cheap ones.
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

from training.quiet_data import PRIORITY_TAGS, canonical_stem  # noqa: E402

N_RESAMPLES = 1_000
CONFIDENCE = 0.95
MIN_SUPPORT = 30
BETAS = (0.5, 1.0, 2.0, 3.0)
BETA_NAMES = {0.5: "f05", 1.0: "f1", 2.0: "f2", 3.0: "f3"}

#: The three corpora whose gold answers the document question in both
#: directions. Matched by stem so a directory rename cannot silently turn a
#: corpus that carries negatives into one reported as NOT_MEASURABLE.
DOC_MEASURABLE = (
    "30000_pii2_eval_25.15k",
    "4000_datax-dualjudge-evalset-1.32k",
    "6589_govdocs2-dualjudge-eval20-3.53k",
)
DOC_MEASURABLE_STEMS = frozenset(canonical_stem(n) for n in DOC_MEASURABLE)


# ---------------------------------------------------------------------- fbeta
def fbeta(precision: np.ndarray, recall: np.ndarray, beta: float) -> np.ndarray:
    b2 = beta * beta
    denom = b2 * precision + recall
    return np.where(denom > 0, (1 + b2) * precision * recall / np.maximum(denom, 1e-12), 0.0)


def _m(value: float | None, lo: float | None = None, hi: float | None = None,
       support: int | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"value": value, "ci_low": lo, "ci_high": hi}
    if support is not None:
        out["support"] = support
    return out


def _quantiles(samples: np.ndarray) -> tuple[float, float]:
    a = (1.0 - CONFIDENCE) / 2.0
    return float(np.quantile(samples, a)), float(np.quantile(samples, 1.0 - a))


def _draws(n: int, seed: int, resamples: int = N_RESAMPLES,
           block: int = 250) -> list[np.ndarray]:
    """Multinomial resample weights, in blocks so (R, n) never lands at once."""
    rng = np.random.default_rng(seed)
    p = np.full(n, 1.0 / n)
    out = []
    remaining = resamples
    while remaining > 0:
        take = min(block, remaining)
        out.append(rng.multinomial(n, p, size=take).astype(np.float32))
        remaining -= take
    return out


# ------------------------------------------------------------------ per corpus
def _topk_ladder(scores: np.ndarray, Y: np.ndarray, fired_doc: np.ndarray,
                 ks: tuple[int, ...] = (1, 3, 5)) -> dict[str, float]:
    """Precision@k / recall@k / F1@k, the tagging ladder the log leads with.

    A file is a hit@k when at least one of its gold tags is in the top k the
    model would show. Both rates share that numerator: precision@k divides by
    the files the tagger predicted anything for, recall@k by the files that have
    a gold tag. The cascade predicts nothing when its gate stays shut, so a
    gate-shut document is "no prediction" here rather than a wrong one -- which
    is why `prediction_rate` has to be read beside precision@k.
    """
    out: dict[str, float | None] = {}
    has_gold = Y.any(axis=1)
    predicted = fired_doc
    order = np.argsort(-scores, axis=1)
    p_den, r_den = int(predicted.sum()), int(has_gold.sum())
    for k in ks:
        topk = order[:, :k]
        hit = np.take_along_axis(Y, topk, axis=1).any(axis=1) & predicted
        # An empty denominator is NOT MEASURABLE, not zero and never `nan`: on
        # nemotron the loader recovers no tag positives at all, so recall@k has
        # nothing to divide by. Returning `nan` here silently poisoned the
        # equal-corpus mean, which is the same class of mistake as writing 0.0
        # for a precision nobody can compute.
        p = float(hit.sum() / p_den) if p_den else None
        r = float((hit & has_gold).sum() / r_den) if r_den else None
        out[f"precision@{k}"] = p
        out[f"recall@{k}"] = r
        out[f"f1@{k}"] = (2 * p * r / (p + r)) if (p and r) else (
            None if (p is None or r is None) else 0.0)
    return out


def evaluate_corpus(name: str, fired_tags: np.ndarray, fired_doc: np.ndarray,
                    Y: np.ndarray, tag_complete: np.ndarray, doc_target: np.ndarray,
                    labels: tuple[str, ...], seed: int,
                    tag_scores: np.ndarray | None = None) -> dict[str, Any]:
    n_docs, n_labels = Y.shape
    scopes: dict[str, dict[str, Any]] = {}
    blocks = _draws(n_docs, seed)

    # ---------------------------------------------------------- document level
    if canonical_stem(name) in DOC_MEASURABLE_STEMS:
        known = doc_target >= 0
        f, g = fired_doc[known], doc_target[known].astype(bool)
        tp = int((f & g).sum()); fp = int((~g & f).sum())
        fn = int((g & ~f).sum()); tn = int((~g & ~f).sum())
        d_blocks = _draws(int(known.sum()), seed + 11)
        tpv, fpv = (f & g).astype(np.float32), (~g & f).astype(np.float32)
        fnv, tnv = (g & ~f).astype(np.float32), (~g & ~f).astype(np.float32)
        prec_s, rec_s, spec_s, f1_s = [], [], [], []
        for w in d_blocks:
            TP, FP, FN, TN = w @ tpv, w @ fpv, w @ fnv, w @ tnv
            prec_s.append(np.divide(TP, TP + FP, out=np.zeros_like(TP), where=(TP + FP) > 0))
            rec_s.append(np.divide(TP, TP + FN, out=np.zeros_like(TP), where=(TP + FN) > 0))
            spec_s.append(np.divide(TN, TN + FP, out=np.zeros_like(TN), where=(TN + FP) > 0))
            f1_s.append(np.divide(2 * TP, 2 * TP + FP + FN, out=np.zeros_like(TP),
                                  where=(2 * TP + FP + FN) > 0))
        cat = np.concatenate
        scopes[f"doc@{name}"] = {
            "doc_precision": _m(tp / (tp + fp) if tp + fp else None,
                                *_quantiles(cat(prec_s)), support=int(tp + fp)),
            "doc_recall": _m(tp / (tp + fn) if tp + fn else None,
                             *_quantiles(cat(rec_s)), support=int(tp + fn)),
            "doc_specificity": _m(tn / (tn + fp) if tn + fp else None,
                                  *_quantiles(cat(spec_s)), support=int(tn + fp)),
            "doc_f1": _m(2 * tp / (2 * tp + fp + fn) if tp else None,
                         *_quantiles(cat(f1_s)), support=int(tp + fn)),
        }

    # --------------------------------------------------------------- per tag
    can_measure_precision = bool(tag_complete.any())
    # A row may act as a negative for tag j only when its gold is exhaustive.
    eligible = Y | tag_complete[:, None]
    pred = fired_tags & eligible
    tp_doc = (pred & Y).astype(np.float32)
    fp_doc = (pred & ~Y).astype(np.float32)
    fn_doc = (~pred & Y).astype(np.float32)

    support = Y.sum(axis=0).astype(np.int64)
    tp = tp_doc.sum(axis=0); fp = fp_doc.sum(axis=0); fn = fn_doc.sum(axis=0)
    predicted = fired_tags.sum(axis=0).astype(np.int64)

    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        rec = np.divide(tp, support.astype(np.float32),
                        out=np.zeros_like(tp), where=support > 0)
    fvals = {b: fbeta(prec, rec, b) for b in BETAS}

    in_catalogue = support > 0
    per_tag: dict[str, dict[str, Any]] = {}
    for j, label in enumerate(labels):
        row: dict[str, Any] = {
            "support": int(support[j]), "tp": int(tp[j]), "fp": int(fp[j]),
            "fn": int(fn[j]), "predicted": int(predicted[j]),
            "recall": float(rec[j]) if support[j] else None,
            "precision": (float(prec[j]) if can_measure_precision and (tp[j] + fp[j]) > 0
                          else None),
        }
        for b in BETAS:
            row[BETA_NAMES[b]] = (float(fvals[b][j])
                                  if can_measure_precision and support[j] else None)
        per_tag[label] = row
        if label in PRIORITY_TAGS:
            if support[j] >= MIN_SUPPORT:
                pos = Y[:, j]
                hits = fired_tags[pos, j].astype(np.float32)
                p_blocks = _draws(int(pos.sum()), seed + 3 + j)
                s = np.concatenate([w @ hits / w.sum(axis=1) for w in p_blocks])
                scopes[f"{label}@{name}"] = {
                    "recall": _m(float(rec[j]), *_quantiles(s), support=int(support[j]))}
            else:
                # Named and excluded: "could not measure" is not "passed".
                scopes[f"{label}@{name}"] = {
                    "recall": _m(None, None, None, support=int(support[j]))}

    # -------------------------------------------------------- macro and micro
    summary: dict[str, Any] = {}
    cat_idx = np.flatnonzero(in_catalogue)
    sup30_idx = np.flatnonzero(support >= MIN_SUPPORT)
    pri_idx = np.asarray([j for j, t in enumerate(labels)
                          if t in PRIORITY_TAGS and support[j] >= MIN_SUPPORT], dtype=np.int64)

    def mean_or_none(v: np.ndarray) -> float | None:
        return float(v.mean()) if v.size else None

    for tag, idx in (("catalogue", cat_idx), ("support30", sup30_idx)):
        if can_measure_precision:
            summary[f"precision_macro_{tag}"] = mean_or_none(prec[idx])
            for b in BETAS:
                summary[f"{BETA_NAMES[b]}_macro_{tag}"] = mean_or_none(fvals[b][idx])
        else:
            summary[f"precision_macro_{tag}"] = None
            for b in BETAS:
                summary[f"{BETA_NAMES[b]}_macro_{tag}"] = None
        summary[f"recall_macro_{tag}"] = mean_or_none(rec[idx])
        summary[f"n_tags_{tag}"] = int(idx.size)

    if can_measure_precision and cat_idx.size:
        TP, FP, FN = tp[cat_idx].sum(), fp[cat_idx].sum(), fn[cat_idx].sum()
        mp = TP / (TP + FP) if TP + FP else 0.0
        mr = TP / (TP + FN) if TP + FN else 0.0
        summary["precision_micro"] = float(mp)
        summary["recall_micro"] = float(mr)
        for b in BETAS:
            summary[f"{BETA_NAMES[b]}_micro"] = float(fbeta(np.asarray([mp]),
                                                            np.asarray([mr]), b)[0])
    else:
        summary["precision_micro"] = None
        summary["recall_micro"] = None
        for b in BETAS:
            summary[f"{BETA_NAMES[b]}_micro"] = None

    # Priority-tag macro, the profile's severity view.
    if pri_idx.size:
        summary["priority_macro_recall"] = float(rec[pri_idx].mean())
        summary["severity_recall_mean"] = float(rec[pri_idx].mean())
        summary["severity_recall_min"] = float(rec[pri_idx].min())
        if can_measure_precision:
            summary["priority_macro_precision"] = float(prec[pri_idx].mean())
            for b in BETAS:
                summary[f"priority_macro_{BETA_NAMES[b]}"] = float(fvals[b][pri_idx].mean())
        else:
            summary["priority_macro_precision"] = None
            for b in BETAS:
                summary[f"priority_macro_{BETA_NAMES[b]}"] = None
    else:
        summary["priority_macro_recall"] = summary["severity_recall_min"] = None
        summary["severity_recall_mean"] = None
        summary["priority_macro_precision"] = None
        for b in BETAS:
            summary[f"priority_macro_{BETA_NAMES[b]}"] = None
    summary["n_priority_measurable"] = int(pri_idx.size)

    # The tail diagnostics the profile asks for on every arm.
    if can_measure_precision and cat_idx.size:
        f2c = fvals[2.0][cat_idx]
        summary["f2_min"] = float(f2c.min())
        summary["f2_median"] = float(np.median(f2c))
        summary["n_tags_f2_zero"] = int((f2c == 0).sum())
        summary["n_tags_f2_below_10pct"] = int((f2c < 0.10).sum())
    else:
        summary["f2_min"] = summary["f2_median"] = None
        summary["n_tags_f2_zero"] = summary["n_tags_f2_below_10pct"] = None
    # The top-k ladder needs a RANKING. A model that emits an unranked set --
    # the fusion's per-label Boolean vote does exactly that -- has no k-th tag to
    # show, so the ladder is genuinely not defined for it rather than zero.
    if tag_scores is not None:
        summary |= _topk_ladder(tag_scores, Y, fired_doc)
    summary["prediction_rate"] = float(fired_tags.any(axis=1).mean())
    summary["tags_predicted_zero_times"] = int((predicted == 0).sum())
    summary["doc_fire_rate"] = float(fired_doc.mean())
    summary["can_measure_precision"] = can_measure_precision

    # ------------------------------------------------------- bootstrapped CIs
    if can_measure_precision and cat_idx.size:
        tpc = np.ascontiguousarray(tp_doc[:, cat_idx])
        fpc = np.ascontiguousarray(fp_doc[:, cat_idx])
        fnc = np.ascontiguousarray(fn_doc[:, cat_idx])
        sup_c = support[cat_idx].astype(np.float32)
        macro_s: dict[str, list[np.ndarray]] = {BETA_NAMES[b]: [] for b in BETAS}
        micro_s: dict[str, list[np.ndarray]] = {BETA_NAMES[b]: [] for b in BETAS}
        for w in blocks:
            TPk, FPk, FNk = w @ tpc, w @ fpc, w @ fnc      # (block, n_cat)
            Pk = np.divide(TPk, TPk + FPk, out=np.zeros_like(TPk), where=(TPk + FPk) > 0)
            Rk = np.divide(TPk, TPk + FNk, out=np.zeros_like(TPk), where=(TPk + FNk) > 0)
            sTP, sFP, sFN = TPk.sum(axis=1), FPk.sum(axis=1), FNk.sum(axis=1)
            mP = np.divide(sTP, sTP + sFP, out=np.zeros_like(sTP), where=(sTP + sFP) > 0)
            mR = np.divide(sTP, sTP + sFN, out=np.zeros_like(sTP), where=(sTP + sFN) > 0)
            for b in BETAS:
                macro_s[BETA_NAMES[b]].append(fbeta(Pk, Rk, b).mean(axis=1))
                micro_s[BETA_NAMES[b]].append(fbeta(mP, mR, b))
        for b in BETAS:
            nm = BETA_NAMES[b]
            summary[f"{nm}_macro_catalogue_ci"] = list(_quantiles(np.concatenate(macro_s[nm])))
            summary[f"{nm}_micro_ci"] = list(_quantiles(np.concatenate(micro_s[nm])))
        del tpc, fpc, fnc, sup_c

    return {"scopes": scopes, "per_tag": per_tag, "summary": summary,
            "n_rows": int(n_docs)}


# ----------------------------------------------------------------------- arm
def assemble_arm(name: str, label: str, per_corpus: dict[str, dict[str, Any]],
                 p95_latency_ms: float | None, docs_per_s: float | None,
                 extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Equal-corpus headlines plus every scope, in the shape `mp decide` reads."""
    scopes: dict[str, Any] = {}
    for body in per_corpus.values():
        scopes.update(body["scopes"])

    def headline(key: str) -> float | None:
        vals = [b["summary"].get(key) for b in per_corpus.values()
                if b["summary"].get(key) is not None]
        return float(np.mean(vals)) if vals else None

    def doc_mean(metric: str) -> float | None:
        vals = [s[metric]["value"] for k, s in scopes.items()
                if k.startswith("doc@") and s.get(metric, {}).get("value") is not None]
        return float(np.mean(vals)) if vals else None

    metrics: dict[str, Any] = {
        # The declared ranker, and the contra-view's ranker beside it.
        "macro_f2": _m(headline("f2_macro_catalogue")),
        "micro_f1": _m(headline("f1_micro")),
        "priority_macro_f05": _m(headline("priority_macro_f05")),
        "macro_f05": _m(headline("f05_macro_catalogue")),
        "p95_latency_ms": _m(p95_latency_ms),
        "docs_per_s": _m(docs_per_s),
        "prediction_rate": _m(headline("prediction_rate")),
        "severity_recall_min": _m(headline("severity_recall_min")),
        "severity_recall_mean": _m(headline("severity_recall_mean")),
        "f1@1": _m(headline("f1@1")), "f1@3": _m(headline("f1@3")),
        "f1@5": _m(headline("f1@5")),
        "precision@1": _m(headline("precision@1")),
        "precision@3": _m(headline("precision@3")),
        "precision@5": _m(headline("precision@5")),
        "recall@1": _m(headline("recall@1")), "recall@3": _m(headline("recall@3")),
        "recall@5": _m(headline("recall@5")),
        "equal_corpus_doc_precision": _m(doc_mean("doc_precision")),
        "equal_corpus_doc_specificity": _m(doc_mean("doc_specificity")),
        "equal_corpus_doc_recall": _m(doc_mean("doc_recall")),
        "equal_corpus_doc_f1": _m(doc_mean("doc_f1")),
    }
    for scope in ("catalogue", "support30"):
        for b in BETAS:
            metrics[f"{BETA_NAMES[b]}_macro_{scope}"] = _m(headline(f"{BETA_NAMES[b]}_macro_{scope}"))
        metrics[f"precision_macro_{scope}"] = _m(headline(f"precision_macro_{scope}"))
        metrics[f"recall_macro_{scope}"] = _m(headline(f"recall_macro_{scope}"))
    for b in BETAS:
        metrics[f"{BETA_NAMES[b]}_micro"] = _m(headline(f"{BETA_NAMES[b]}_micro"))
    metrics["precision_micro"] = _m(headline("precision_micro"))
    metrics["recall_micro"] = _m(headline("recall_micro"))
    for b in BETAS:
        metrics[f"priority_macro_{BETA_NAMES[b]}"] = _m(headline(f"priority_macro_{BETA_NAMES[b]}"))
    metrics["priority_macro_precision"] = _m(headline("priority_macro_precision"))
    metrics["priority_macro_recall"] = _m(headline("priority_macro_recall"))
    metrics["f2_min"] = _m(headline("f2_min"))
    metrics["f2_median"] = _m(headline("f2_median"))

    return {
        "name": name, "label": label, "metrics": metrics, "scopes": scopes,
        "extra": extra or {},
        "per_corpus": {k: dict(v["summary"], n_rows=v["n_rows"])
                       for k, v in per_corpus.items()},
        "per_tag": {k: v["per_tag"] for k, v in per_corpus.items()},
    }


__all__ = ["BETAS", "BETA_NAMES", "DOC_MEASURABLE", "MIN_SUPPORT",
           "assemble_arm", "evaluate_corpus", "fbeta"]
