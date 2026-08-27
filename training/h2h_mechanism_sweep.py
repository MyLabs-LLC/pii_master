"""Sweep genuinely distinct threshold/decision mechanisms on the calibration carve.

## Why this is not "250 experiments"

Every mechanism here costs a sealed measurement if it is taken seriously, and the
sealed suite survives being touched once per candidate — not once per idea. So the
sweep runs entirely on the **training calibration carve**, ranks mechanisms there,
and hands only the finalists to the sealed set. That is the difference between a
search and a slow leak.

It is also why the list is ~20 and not 250: these are the mechanisms that are
actually distinct *given what has already been measured in this repo*. A longer
list would be padding, and padding still costs measurements.

## What every mechanism must produce

A boolean fire matrix over the calibration rows, derived from the frozen cascade's
scores. Mechanisms fall into two classes and the distinction decides whether a
winner is deployable:

* **THRESHOLD-EXPRESSIBLE** — the mechanism reduces to a per-tag threshold vector.
  These ship by swapping 61 floats; latency is unchanged; `h2h_target_box`-style
  packaging applies directly.
* **RUNTIME** — the mechanism needs logic at predict time (a per-document cap, a
  consistency rule, a margin). These are still worth measuring, but a winner
  requires a change to the serving path and a re-benchmark, and is marked so.

## The scoring

Equal-corpus micro precision / recall / F1 — the same aggregation the suite
declares, so a calibration number here is comparable in kind with the sealed
headline. The positive-unlabelled discipline is carried over: a row whose gold
cannot act as a negative is excluded from the precision denominator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.h2h_scorecard_rebuild import retarget_cache  # noqa: E402
from training.quiet_select import sweep  # noqa: E402

SCORECARD = Path("projects/pii-scorecard-60")
TARGET = {"precision": 0.90, "recall": 0.80, "micro_f1": 0.80}


# --------------------------------------------------------------------- scoring
def equal_corpus_micro(fired, Y, TC, corpus):
    """Micro P/R/F1 per corpus, then averaged over corpora."""
    Ps, Rs, Fs = [], [], []
    for c in np.unique(corpus):
        m = corpus == c
        f, y, tc = fired[m], Y[m], TC[m]
        tp = int((f & y).sum())
        fp = int((f & ~y & tc[:, None]).sum())
        fn = int((~f & y).sum())
        if tp + fn == 0:
            continue
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        Ps.append(p); Rs.append(r)
        Fs.append(0.0 if p + r == 0 else 2 * p * r / (p + r))
    return (float(np.mean(Ps)), float(np.mean(Rs)), float(np.mean(Fs)))


def box_thresholds(S, Y, TC, p_target, r_target, min_support=30, beta=0.5):
    """The current champion mechanism: F-beta optimum inside a (P,R) box."""
    thr = np.full(S.shape[1], np.inf, dtype=np.float32)
    for j in range(S.shape[1]):
        pos = Y[:, j]
        if pos.sum() < min_support:
            continue
        p, r, t = sweep(S[:, j], pos, TC & ~pos)
        if not len(t):
            continue
        f = (1 + beta**2) * p * r / np.maximum(beta**2 * p + r, 1e-12)
        ok = (p >= p_target) & (r >= r_target)
        thr[j] = t[int(np.flatnonzero(ok)[np.argmax(f[ok])])] if ok.any() \
            else t[int(np.argmax(f))]
    return thr


def fire_from_thr(S, thr):
    return S >= thr


# ------------------------------------------------------------------ mechanisms
def mechanisms(S, Y, TC, corpus, labels):
    """Yield (name, kind, note, fired). Each is ONE change from the baseline."""
    n_tags = S.shape[1]
    base_thr = box_thresholds(S, Y, TC, 0.88, 0.90)
    yield ("00_baseline_box_p88r90", "threshold",
           "the shipped mechanism: F0.5 optimum inside P>=0.88, R>=0.90",
           fire_from_thr(S, base_thr))

    # --- 1. beta: what the choice INSIDE the box optimises -------------------
    for beta, tag in ((0.25, "01"), (1.0, "02"), (2.0, "03")):
        yield (f"{tag}_box_beta{beta}", "threshold",
               f"same box, but the point inside it maximises F{beta} instead of F0.5",
               fire_from_thr(S, box_thresholds(S, Y, TC, 0.88, 0.90, beta=beta)))

    # --- 2. per-tag prevalence prior ----------------------------------------
    prev = Y.mean(axis=0)
    for scale, tag in ((0.5, "04"), (1.0, "05")):
        shift = scale * np.log(np.maximum(prev, 1e-6) / (1 - np.minimum(prev, 1 - 1e-6)))
        yield (f"{tag}_prevalence_prior_x{scale}", "threshold",
               "shift each threshold by the tag's prior log-odds: rare tags are "
               "asked for more evidence, common tags less",
               fire_from_thr(S, base_thr - shift.astype(np.float32)))

    # --- 3. per-tag score standardisation -----------------------------------
    mu, sd = S.mean(axis=0), np.maximum(S.std(axis=0), 1e-6)
    Z = (S - mu) / sd
    yield ("06_zscore_then_box", "threshold",
           "standardise each tag's score distribution before selecting, so one "
           "threshold rule sees comparable scales across tags",
           fire_from_thr(Z, box_thresholds(Z, Y, TC, 0.88, 0.90)))

    # --- 4. document-length normalisation -----------------------------------
    # A long document accumulates score simply by having more tokens.
    nfeat = np.maximum((S != 0).sum(axis=1, keepdims=True), 1)
    Sl = S / np.sqrt(nfeat)
    yield ("07_length_normalised", "threshold",
           "divide scores by sqrt(active features): removes the advantage a long "
           "document gets from sheer length",
           fire_from_thr(Sl, box_thresholds(Sl, Y, TC, 0.88, 0.90)))

    # --- 5. per-document top-k cap ------------------------------------------
    for k, tag in ((3, "08"), (5, "09"), (8, "10")):
        fired = fire_from_thr(S, base_thr)
        order = np.argsort(-S, axis=1)
        keep = np.zeros_like(fired)
        rows = np.arange(S.shape[0])[:, None]
        keep[rows, order[:, :k]] = True
        yield (f"{tag}_topk_cap_{k}", "runtime",
               f"emit at most {k} tags per document, keeping the highest-scoring "
               f"— a precision play against documents that light up everything",
               fired & keep)

    # --- 6. margin above threshold ------------------------------------------
    for m, tag in ((0.5, "11"), (1.0, "12")):
        yield (f"{tag}_margin_{m}", "threshold",
               f"require the score to clear its threshold by {m}, not merely reach it",
               fire_from_thr(S, base_thr + m))

    # --- 7. parent/child consistency ----------------------------------------
    idx = {t: i for i, t in enumerate(labels)}
    pairs = [("sensitive_pii_given_name", "sensitive_pii_full_name"),
             ("sensitive_pii_family_name", "sensitive_pii_full_name"),
             ("sensitive_pii_middle_name", "sensitive_pii_full_name"),
             ("sensitive_pii_street_number_and_name", "sensitive_pii_address")]
    fired = fire_from_thr(S, base_thr).copy()
    for child, parent in pairs:
        if child in idx and parent in idx:
            fired[:, idx[parent]] |= fired[:, idx[child]]
    yield ("13_parent_implied_by_child", "runtime",
           "if a name subtype fires, the parent tag fires too — the taxonomy says "
           "a given_name IS a full_name, so silence on the parent is incoherent",
           fired)

    fired = fire_from_thr(S, base_thr).copy()
    for child, parent in pairs:
        if child in idx and parent in idx:
            fired[:, idx[child]] &= fired[:, idx[parent]]
    yield ("14_child_requires_parent", "runtime",
           "the converse: suppress a subtype unless the parent also fires, on the "
           "grounds that a lone subtype is usually a false positive",
           fired)

    # --- 8. gate-margin-conditional thresholds ------------------------------
    # Documents the gate barely admitted are riskier; ask them for more.
    total = S.max(axis=1)
    marginal = total < np.percentile(total, 25)
    fired = fire_from_thr(S, base_thr)
    strict = fire_from_thr(S, base_thr + 1.0)
    yield ("15_strict_on_marginal_docs", "runtime",
           "raise every threshold by 1.0 on the weakest-scoring quartile of "
           "documents, leaving confident documents alone",
           np.where(marginal[:, None], strict, fired))

    # --- 9. disable tags that cannot reach the box --------------------------
    thr2 = base_thr.copy()
    for j in range(n_tags):
        pos = Y[:, j]
        if pos.sum() < 30:
            continue
        p, r, t = sweep(S[:, j], pos, TC & ~pos)
        if len(t) and not ((p >= 0.88) & (r >= 0.90)).any():
            thr2[j] = np.inf
    yield ("16_disable_unreachable", "threshold",
           "silence every tag whose curve never enters the box, rather than "
           "parking it at its best-F0.5 point",
           fire_from_thr(S, thr2))

    # --- 10. per-vertical boxes ---------------------------------------------
    thr3 = base_thr.copy()
    for pref, (pt, rt) in (("sensitive_phi_", (0.85, 0.92)),
                           ("sensitive_pci_", (0.92, 0.88)),
                           ("sensitive_pii_", (0.88, 0.90))):
        sel = [i for i, t in enumerate(labels) if t.startswith(pref)]
        if not sel:
            continue
        sub = box_thresholds(S[:, sel], Y[:, sel], TC, pt, rt)
        thr3[sel] = sub
    yield ("17_per_vertical_box", "threshold",
           "PHI leans to recall, PCI to precision, PII stays put — the three "
           "verticals have different costs of error",
           fire_from_thr(S, thr3))

    # --- 11. co-occurrence prune --------------------------------------------
    co = (Y.T.astype(np.float32) @ Y.astype(np.float32))
    np.fill_diagonal(co, 0)
    never = co == 0
    fired = fire_from_thr(S, base_thr).copy()
    strongest = np.argmax(np.where(fired, S, -np.inf), axis=1)
    for i in range(0, S.shape[0], 20000):          # chunked, memory
        sl = slice(i, min(i + 20000, S.shape[0]))
        blk = fired[sl]
        anchor = strongest[sl]
        blk &= ~never[anchor]
        fired[sl] = blk
    yield ("18_cooccurrence_prune", "runtime",
           "drop a tag when it never co-occurs in training gold with the "
           "document's highest-scoring tag",
           fired)

    # --- 12. ensemble of boxes ----------------------------------------------
    a = fire_from_thr(S, box_thresholds(S, Y, TC, 0.80, 0.90))
    b = fire_from_thr(S, box_thresholds(S, Y, TC, 0.88, 0.90))
    c = fire_from_thr(S, box_thresholds(S, Y, TC, 0.92, 0.85))
    yield ("19_majority_of_three_boxes", "runtime",
           "fire when at least two of three differently-parameterised boxes agree",
           (a.astype(np.int8) + b + c) >= 2)
    yield ("20_intersection_of_three_boxes", "runtime",
           "fire only when all three boxes agree — maximum precision",
           a & b & c)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path("projects/pii-target-8070/probe/mechanism_sweep.json"))
    args = ap.parse_args()

    cat = retarget_cache(SCORECARD / "cache", 61)
    labels = tuple(cat["labels"])
    from training.quiet_fit import carve_holdin, load, score, train_corpora  # noqa
    from training.quiet_model import QuietCascade  # noqa

    model = QuietCascade.load(SCORECARD / "models" / "cascade_scorecard61")
    ds = load(train_corpora(), profile="deep")
    _, calib_mask = carve_holdin(ds)
    calib = ds.subset(calib_mask)
    g = (calib.X @ model.gate_weights + model.gate_intercept).astype(np.float32)
    open_doc = g >= model.gate_threshold
    S = score(calib.X[open_doc], model.tag_weights, mode="sum")
    Y = np.asarray(calib.Y[open_doc].todense()).astype(bool)
    TC = calib.tag_complete[open_doc]
    corpus = calib.corpus[open_doc]
    print(f"{S.shape[0]:,} gate-admitted calibration rows, {S.shape[1]} tags, "
          f"{len(np.unique(corpus))} corpora\n", flush=True)

    rows = []
    print(f"{'mechanism':<34}{'kind':<11}{'calib P':>9}{'calib R':>9}{'calib F1':>10}"
          f"   vs baseline")
    base = None
    for name, kind, note, fired in mechanisms(S, Y, TC, corpus, labels):
        P, R, F = equal_corpus_micro(fired, Y, TC, corpus)
        if base is None:
            base = (P, R, F)
        d = F - base[2]
        rows.append({"mechanism": name, "kind": kind, "note": note,
                     "calib_precision": P, "calib_recall": R, "calib_f1": F,
                     "delta_f1_vs_baseline": d})
        print(f"{name:<34}{kind:<11}{P:>9.4f}{R:>9.4f}{F:>10.4f}   {d:+.4f}",
              flush=True)

    rows.sort(key=lambda r: -r["calib_f1"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"split": "training calibration carve", "aggregation": "equal_corpus",
         "baseline": rows[0]["mechanism"] if rows else None,
         "target": TARGET, "results": rows}, indent=1), encoding="utf-8")
    print(f"\ntop 5 by calibration micro F1:")
    for r in rows[:5]:
        print(f"  {r['mechanism']:<34}{r['kind']:<11}F1 {r['calib_f1']:.4f}  "
              f"P {r['calib_precision']:.4f}  R {r['calib_recall']:.4f}")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
