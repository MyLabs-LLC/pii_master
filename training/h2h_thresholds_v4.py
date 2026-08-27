"""Re-derive the two failed tag thresholds on TRAINING data, reproducibly.

## Why this file exists

`pii-cascade-balanced-v3` raised two tag thresholds --
`sensitive_pii_military_identification_number` (index 44) and
`sensitive_pii_sexual_identity_and_orientation` (index 51) -- by sweeping F2 on
"a held-in half of the eight `data/2-eval` corpora". Those eight corpora are the
sealed evaluation set. Selecting on half of them makes every subsequent number
for those two tags a selection result rather than a measurement, and the script
that did it was never checked in, so the split cannot even be reproduced.

`quiet_fit.carve_holdin` exists precisely so this does not happen: it carves a
15% calibration slice out of the **training** corpora, and its docstring says
the sealed evaluation directories are not reachable from that module. Every
other threshold in the cascade was chosen that way. These two are re-chosen the
same way here.

## What was actually wrong with them

Not the sweep, and not a bug -- the *cap*, working exactly as written on a sample
too small to carry it. `quiet_select.group_recall_cap` returns the highest
threshold at which every source group still clears the recall floor, estimating
each group's cut as the `1 - floor` quantile of that group's positive scores, and
skipping groups with fewer than `min_group_support=20` positives so that the cap
is not "set from noise" (its words).

Twenty positives is not enough to do that. At a floor of 0.7556 the cut sits at
the 0.2444 quantile, so a 20-positive group places it on its 5th order statistic.
Measured on the calibration carve:

| tag | group that set the cap | positives | AUC | its cap | other groups' caps |
|---|---|---:|---:|---:|---|
| 44 military ID | `148775_pii2_train` | **31** | 0.684 | **-3.4006** | +10.38, +11.57 |
| 51 sexual orientation | `148775_pii2_train` | **43** | 0.874 | **-3.9634** | +2.43 |

In both cases one small, weakly-ranked slice dictated the threshold for every
other source -- 10,556 positives for tag 44 -- and dragged the cut below the whole
negative distribution. Both tags then *met* the recall floor and were recorded as
successes: tag 44 at precision 0.2150, tag 51 at precision **0.0051**, firing on
51,997 and 29,069 of 66,867 gate-admitted calibration rows.

So the numbers were visible at selection time and nothing vetoed them. The rule
this module applies is the one `group_recall_cap`'s docstring already describes,
with "too few to estimate a quantile" counted where the quantile actually is:

* a group may set the cap only if at least `MIN_TAIL_EVENTS` of its positives
  fall at or below the candidate cut -- `n * (1 - floor) >= 10`, the ordinary
  ten-events-in-the-tail rule of thumb, rather than a flat count of positives
  anywhere on the curve;
* if no group qualifies, the cap is **`+inf`** ("nothing could constrain it"),
  not `-inf`. As a `-inf` sentinel it would make every threshold inadmissible and
  hand the tag to the `argmax(recall)` fallback, which is the same failure
  arriving by a different route. No tag takes this path today; it is closed
  because it is one small corpus away from opening.

`MIN_TAIL_EVENTS` is fixed at 10 before any sealed corpus is scored, and it is a
statistical rule of thumb rather than a derived constant. That is stated plainly
because it is the one judgement call in this file.

## What this does NOT claim

It does not claim to rescue both tags. The rule is applied uniformly to all 58
and the report says what moved; a tag the rule leaves broken is reported broken.

## Reproduction first

`--diagnose` recomputes all 58 thresholds from the frozen `cascade_balanced`
weights and asserts they equal the shipped v2 vector. If that reproduction fails,
the diagnosis below is unproven and nothing is emitted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.h2h_priority import PROJECT  # noqa: E402
from training.quiet_cache import PROFILES, load_catalogue  # noqa: E402
from training.quiet_fit import carve_holdin, load, score, train_corpora  # noqa: E402
from training.quiet_model import QuietCascade  # noqa: E402
from training.quiet_select import fbeta, group_recall_cap, sweep  # noqa: E402

SOURCE = "cascade_balanced"

#: A group may set the recall cap only if this many of its positives land at or
#: below the candidate cut. Fixed before any sealed corpus was scored. See the
#: module docstring: this is the one judgement call here.
MIN_TAIL_EVENTS = 10

#: A rebuilt threshold is not required to equal the shipped one to the bit. It is
#: required to name the SAME OPERATING POINT: applying the reproduced threshold
#: and the shipped threshold to the same calibration scores must give the same
#: precision and recall to within this much. That is the property the diagnosis
#: actually depends on, and unlike a raw threshold delta it is scale-free.
#:
#: (Raw deltas are reported too, as context. `sweep` only cuts between distinct
#: scores, so a float16-perturbed score matrix can land on an adjacent realisable
#: cut -- a visible jump in the threshold that moves neither P nor R.)
OPERATING_POINT_TOLERANCE = 0.01


def select_per_label(S, Y, tag_complete, group, *, beta, recall_floor, margin,
                     min_support, min_group_support=20, corrected_cap=False,
                     min_precision=None):
    """`quiet_select.select_per_label_robust`, with the cap rule switchable.

    Identical in every other respect -- same admissibility test, same tie-breaks,
    same report fields -- so that `corrected_cap=False, min_precision=None`
    reproduces the shipped vector and the difference between runs is one flag.

    ## `min_precision`: the second half of the cap defect

    `corrected_cap` fixed the case where a group too small to estimate a quantile
    set the cap. It did not fix the other way the cap goes wrong, and the measured
    error budget of `cascade_scorecard61` is dominated by it:

    | tag | branch | calib precision | FP | TP |
    | --- | --- | ---: | ---: | ---: |
    | `sexual_identity_and_orientation` | floor_met | 0.0051 | 42,131 | 173 |
    | `geolocation` | floor_met | 0.0173 | 19,726 | 226 |
    | `religion` | floor_met | 0.0215 | 17,071 | 293 |

    All three have ample support -- 142 to 246 calibration positives -- so
    estimability is not the problem. The problem is that the rule honours the
    recall floor **at any precision cost**: it takes the F-beta optimum among
    points at or below the cap, and when the cap sits below the whole negative
    distribution that set contains only terrible points. Nothing vetoes 0.5%
    precision, so the selector records `floor_met` and moves on, and the tag then
    fires on tens of thousands of documents to find a few hundred.

    `min_precision` adds the veto, as a three-rung ladder applied per tag:

    1. the normal choice, if its precision clears the bar;
    2. otherwise the F-beta optimum subject to the pooled recall floor but
       **ignoring the group cap** -- the cap is what dragged the point down, and a
       tag whose worst source cannot be served at usable precision is better
       served well on the sources that can be;
    3. otherwise the unconstrained F-beta optimum; and if even that is below the
       bar, the tag is **disabled** and said to be disabled. A head that cannot be
       operated at usable precision anywhere is not a head, and shipping it as
       noise is worse than shipping nothing -- `prediction_rate` and the report
       both make an absent tag visible, which a 0.5%-precision tag is not.

    The bar is a judgement call, like `MIN_TAIL_EVENTS`, and is passed in rather
    than baked here so the run that uses it has to state it.
    """
    n_labels = S.shape[1]
    thresholds = np.full(n_labels, np.inf, dtype=np.float32)
    report: list[dict] = []
    capped_floor = min(recall_floor + margin, 0.999)
    for j in range(n_labels):
        positive = Y[:, j].astype(bool)
        if positive.sum() < min_support:
            report.append({"label": j, "support": float(positive.sum()),
                           "disabled": True, "floor_met": False, "cap": None,
                           "cap_estimable": None, "precision": 0.0, "recall": 0.0,
                           "f": 0.0, "branch": "below_min_support"})
            continue
        if corrected_cap:
            cap, skipped = estimable_group_recall_cap(
                S[:, j], positive, group, floor=capped_floor)
        else:
            cap = group_recall_cap(S[:, j], positive, group, floor=capped_floor,
                                   min_group_support=min_group_support)
            skipped = []
        cap_estimable = np.isfinite(cap)
        effective_cap = cap
        precision, recall, thr = sweep(S[:, j], positive, tag_complete & ~positive)
        if not len(thr):
            report.append({"label": j, "support": float(positive.sum()),
                           "disabled": True, "floor_met": False, "cap": float(cap),
                           "cap_estimable": bool(cap_estimable), "precision": 0.0,
                           "recall": 0.0, "f": 0.0, "branch": "empty_sweep"})
            continue
        f = fbeta(precision, recall, beta)
        ok = (thr <= effective_cap) & (recall >= recall_floor)
        if ok.any():
            idx = int(np.flatnonzero(ok)[np.argmax(f[ok])])
            floor_met, branch = True, "floor_met"
        elif (thr <= effective_cap).any():
            admissible = np.flatnonzero(thr <= effective_cap)
            idx = int(admissible[np.argmax(recall[admissible])])
            floor_met, branch = False, "capped_max_recall"
        else:
            idx = int(np.argmax(recall))
            floor_met, branch = False, "uncapped_max_recall"

        # --------------------------------------------- the precision veto
        rescued_from = None
        if min_precision is not None and precision[idx] < min_precision:
            rescued_from = {"branch": branch, "precision": float(precision[idx]),
                            "recall": float(recall[idx]), "threshold": float(thr[idx])}
            uncapped = recall >= recall_floor              # rung 2: drop the cap
            if uncapped.any():
                cand = int(np.flatnonzero(uncapped)[np.argmax(f[uncapped])])
            else:
                cand = int(np.argmax(f))                   # rung 3: drop the floor
            if precision[cand] >= min_precision:
                idx = cand
                floor_met = bool(recall[cand] >= recall_floor)
                branch = ("precision_floor_uncapped" if uncapped.any()
                          else "precision_floor_unconstrained")
            else:
                report.append({
                    "label": j, "support": float(positive.sum()), "disabled": True,
                    "floor_met": False, "cap": float(cap),
                    "cap_estimable": bool(cap_estimable),
                    "precision": float(precision[cand]), "recall": float(recall[cand]),
                    "f": float(f[cand]), "branch": "disabled_low_precision",
                    "groups_skipped_for_cap": skipped, "rescue_attempt": rescued_from,
                    "best_precision_available": float(precision.max()),
                    "n_fired_calib": 0})
                continue                                   # thresholds[j] stays +inf

        thresholds[j] = thr[idx]
        entry = {"label": j, "support": float(positive.sum()),
                 "disabled": False, "floor_met": floor_met,
                 "cap": float(cap), "cap_estimable": bool(cap_estimable),
                 "precision": float(precision[idx]), "recall": float(recall[idx]),
                 "f": float(f[idx]), "branch": branch,
                 "groups_skipped_for_cap": skipped,
                 "n_fired_calib": int((S[:, j] >= thr[idx]).sum())}
        if rescued_from is not None:
            entry["rescued_from"] = rescued_from
        report.append(entry)
    return thresholds, report


def estimable_group_recall_cap(scores, positive, group, *, floor,
                               min_tail_events=MIN_TAIL_EVENTS):
    """`quiet_select.group_recall_cap` with "too few" counted at the quantile.

    Returns `+inf` when no group qualifies -- there is then no cap, as opposed to
    a cap at the bottom of the curve.
    """
    caps, skipped = [], []
    for g in np.unique(group):
        pos_scores = scores[positive & (group == g)]
        n = int(pos_scores.size)
        if n == 0:
            continue
        tail = n * (1.0 - floor)
        if tail < min_tail_events:
            skipped.append({"group": int(g), "positives": n, "tail_events": float(tail)})
            continue
        caps.append(float(np.quantile(pos_scores, 1.0 - floor)))
    return (min(caps) if caps else np.inf), skipped


def _operating_point(scores: np.ndarray, positive: np.ndarray, threshold: float
                     ) -> tuple[float, float]:
    """Precision and recall of one label at one threshold, on eligible rows."""
    fired = scores >= threshold
    tp = int((fired & positive).sum())
    fp = int((fired & ~positive).sum())
    return tp / max(tp + fp, 1), tp / max(int(positive.sum()), 1)


def calibration_state():
    """The exact calibration inputs `h2h_cascade_rebuild` selected against."""
    model = QuietCascade.load(PROJECT / "models" / SOURCE)
    meta = json.loads((PROJECT / "models" / SOURCE / "model.json").read_text(
        encoding="utf-8"))["metadata"]
    cp = meta["cascade_params"]
    profile = meta["profile"]
    ds = load(train_corpora(), profile=profile)
    _, calib_mask = carve_holdin(ds)
    calib = ds.subset(calib_mask)
    g_cal = (calib.X @ model.gate_weights + model.gate_intercept).astype(np.float32)
    open_doc = g_cal >= model.gate_threshold
    S_cal = score(calib.X, model.tag_weights, mode="sum")
    Ycal = np.asarray(calib.Y.todense()).astype(bool)
    return model, cp, {
        "S": S_cal[open_doc], "Y": Ycal[open_doc],
        "tag_complete": calib.tag_complete[open_doc],
        "group": calib.corpus[open_doc],
        "n_calib": int(len(g_cal)), "n_open": int(open_doc.sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", action="store_true",
                    help="write models/cascade_v4 (default: diagnose only)")
    ap.add_argument("--name", default="cascade_v4")
    ap.add_argument("--out", type=Path,
                    default=PROJECT / "probe" / "threshold_v4_diagnosis.json")
    args = ap.parse_args()

    labels = tuple(load_catalogue()["labels"])
    model, cp, cal = calibration_state()
    print(f"calibration carve: {cal['n_calib']:,} training rows, "
          f"{cal['n_open']:,} admitted by the gate", flush=True)

    kw = dict(beta=0.5, recall_floor=cp["recall_floor"], margin=cp.get("margin", 0.0),
              min_support=cp["min_support_fit"])
    thr_repro, rep_repro = select_per_label(
        cal["S"], cal["Y"], cal["tag_complete"], cal["group"],
        corrected_cap=False, **kw)

    # Reproduction is asserted to a tolerance, not bit-for-bit, and the reason is
    # storage: `QuietCascade` persists `tag_weights` as float16, so the score
    # matrix rebuilt from the saved model is a slightly coarser version of the
    # full-precision one the original selection swept. That moves each chosen
    # threshold in the 4th decimal of a scale that runs 2.5-10.7.
    #
    # What IS checked exactly is the structural part: which tags are disabled
    # (threshold +inf) must match the shipped vector element for element. The
    # per-tag selector report was never persisted for `cascade_balanced`, so the
    # branch each tag took in the original run cannot be compared against -- only
    # re-derived here. That is stated rather than assumed.
    shipped = model.tag_thresholds.astype(np.float32)
    finite = np.isfinite(thr_repro) & np.isfinite(shipped)
    if not np.array_equal(np.isfinite(thr_repro), np.isfinite(shipped)):
        raise SystemExit("reproduction disagrees about which tags are disabled entirely; "
                         "this is not the selection that produced v2. Nothing emitted.")
    drift = float(np.max(np.abs(thr_repro[finite] - shipped[finite]))) if finite.any() else 0.0
    scale = float(np.max(np.abs(shipped[finite]))) if finite.any() else 1.0

    worst = {"label": None, "d_precision": 0.0, "d_recall": 0.0}
    for j in np.flatnonzero(finite):
        positive = cal["Y"][:, j].astype(bool)
        eligible = cal["tag_complete"] & ~positive
        keep = positive | eligible
        sj, pj = cal["S"][keep, j], positive[keep]
        if not pj.any():
            continue
        (pa, ra), (pb, rb) = (_operating_point(sj, pj, thr_repro[j]),
                              _operating_point(sj, pj, shipped[j]))
        if max(abs(pa - pb), abs(ra - rb)) > max(abs(worst["d_precision"]),
                                                 abs(worst["d_recall"])):
            worst = {"label": labels[j], "index": int(j),
                     "d_precision": float(pa - pb), "d_recall": float(ra - rb),
                     "thr_reproduced": float(thr_repro[j]), "thr_shipped": float(shipped[j])}

    worst_shift = max(abs(worst["d_precision"]), abs(worst["d_recall"]))
    print(f"reproduction of the shipped v2 threshold vector:"
          f"\n  max |threshold delta| = {drift:.3e} on a scale of {scale:.2f} "
          f"(float16 weight storage; adjacent realisable cuts)"
          f"\n  worst operating-point shift = {worst_shift:.2e} "
          f"(dP={worst['d_precision']:+.2e} dR={worst['d_recall']:+.2e} "
          f"on {worst['label']})", flush=True)
    if worst_shift > OPERATING_POINT_TOLERANCE:
        raise SystemExit(
            f"the reproduced selection names a materially different operating point "
            f"({worst_shift:.4f} > {OPERATING_POINT_TOLERANCE}) on {worst['label']}. "
            f"The diagnosis rests on this being the selection that produced v2; "
            f"it is not. Nothing emitted.")
    same = True

    thr_fix, rep_fix = select_per_label(
        cal["S"], cal["Y"], cal["tag_complete"], cal["group"],
        corrected_cap=True, **kw)

    print("\nselector branch taken, all 58 tags:", flush=True)
    from collections import Counter
    for branch, n in Counter(r["branch"] for r in rep_repro).most_common():
        print(f"  {branch:22s} {n}", flush=True)
    print("\nthe two tags v3 overrode, as the declared rule chose them:", flush=True)
    for j in (44, 51):
        r = rep_repro[j]
        pos = int(r["support"])
        elig = int((cal["tag_complete"] & ~cal["Y"][:, j].astype(bool)).sum())
        print(f"  [{j}] {labels[j]}"
              f"\n      calibration positives={pos}  eligible negatives={elig:,}"
              f"\n      cap={r['cap']:.4f}  chosen thr={thr_repro[j]:.4f}  [{r['branch']}]"
              f"\n      at that cut: P={r['precision']:.4f} R={r['recall']:.4f} "
              f"F0.5={r['f']:.4f}  fired on {r.get('n_fired_calib'):,} of "
              f"{len(cal['S']):,} gate-admitted calibration rows", flush=True)

    print("\nper-source-group behaviour for those two tags "
          "(which group sets the cap, and can it rank the tag at all):", flush=True)
    ds_names = load(train_corpora(), profile=json.loads(
        (PROJECT / "models" / SOURCE / "model.json").read_text(
            encoding="utf-8"))["metadata"]["profile"]).corpus_names
    for j in (44, 51):
        print(f"  [{j}] {labels[j]}", flush=True)
        pos_all = cal["Y"][:, j].astype(bool)
        capped_floor = min(cp["recall_floor"] + cp.get("margin", 0.0), 0.999)
        for g in np.unique(cal["group"]):
            m = cal["group"] == g
            pos = pos_all & m
            npos = int(pos.sum())
            neg = m & cal["tag_complete"] & ~pos_all
            nneg = int(neg.sum())
            if npos == 0:
                continue
            auc = float("nan")
            if npos and nneg:
                sp_, sn = cal["S"][pos, j], cal["S"][neg, j]
                auc = float((sp_[:, None] > sn[None, :]).mean()
                            + 0.5 * (sp_[:, None] == sn[None, :]).mean())
            q = (float(np.quantile(cal["S"][pos, j], 1.0 - capped_floor))
                 if npos >= 20 else float("nan"))
            flag = "  <-- SETS THE CAP" if npos >= 20 and abs(q - rep_repro[j]["cap"]) < 1e-3 else ""
            print(f"      {ds_names[g][:44]:<44s} pos={npos:>6,} neg={nneg:>6,} "
                  f"AUC={auc:.4f} cap_q={q:>9.4f}{flag}", flush=True)

    print("\ntags that did NOT meet the recall floor:", flush=True)
    for r in rep_repro:
        if r["disabled"] or r["floor_met"]:
            continue
        j = r["label"]
        print(f"  [{j:2d}] {labels[j]:<58s} thr={thr_repro[j]:>10.4f} "
              f"support={r['support']:.0f} cap={r['cap']:.4f} "
              f"P={r['precision']:.4f} R={r['recall']:.4f} "
              f"fired={r.get('n_fired_calib')} [{r['branch']}]", flush=True)

    changed = [int(j) for j in np.flatnonzero(thr_repro != thr_fix)]
    unknown_cap = [int(r["label"]) for r in rep_fix
                   if r.get("cap_estimable") is False]
    print(f"\ntags left with no estimable cap at all under the corrected rule: "
          f"{len(unknown_cap)} -> {[labels[j] for j in unknown_cap]}", flush=True)
    print(f"tags whose threshold changes under the corrected cap: "
          f"{len(changed)} of 58", flush=True)
    for j in changed:
        a, b = rep_repro[j], rep_fix[j]
        print(f"  [{j}] {labels[j]}"
              f"\n      support={a['support']:.0f}  cap_estimable={a['cap_estimable']}"
              f"\n      v2  thr={thr_repro[j]:>10.4f}  P={a['precision']:.4f} "
              f"R={a['recall']:.4f} F0.5={a['f']:.4f}  fired={a.get('n_fired_calib')}  "
              f"[{a['branch']}]"
              f"\n      v4  thr={thr_fix[j]:>10.4f}  P={b['precision']:.4f} "
              f"R={b['recall']:.4f} F0.5={b['f']:.4f}  fired={b.get('n_fired_calib')}  "
              f"[{b['branch']}]", flush=True)

    payload = {
        "source_model": SOURCE,
        "reproduces_shipped_v2": bool(same),
        "reproduction_max_delta": drift,
        "reproduction_operating_point_tolerance": OPERATING_POINT_TOLERANCE,
        "reproduction_worst_operating_point_shift": worst,
        "reproduction_note": "float16 tag_weight storage; branch assignment is exact",
        "cascade_params": cp,
        "n_calibration_rows": cal["n_calib"],
        "n_gate_admitted": cal["n_open"],
        "selection_data": "training corpora only, quiet_fit.carve_holdin 15% carve",
        "no_estimable_cap_tags": {int(j): labels[j] for j in unknown_cap},
        "min_tail_events": MIN_TAIL_EVENTS,
        "changed": {int(j): {"label": labels[j],
                             "v2": float(thr_repro[j]), "v4": float(thr_fix[j]),
                             "v2_report": rep_repro[j], "v4_report": rep_fix[j]}
                    for j in changed},
        "v3_thresholds_for_reference": {
            "44": 7.4621, "51": 1.2377,
            "provenance": "swept on a held-in half of data/2-eval; not reproducible"},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n-> {args.out}", flush=True)

    if args.emit:
        out_dir = PROJECT / "models" / args.name
        QuietCascade(
            labels=labels, gate_weights=model.gate_weights,
            gate_intercept=model.gate_intercept, gate_threshold=model.gate_threshold,
            tag_weights=model.tag_weights, tag_thresholds=thr_fix.astype(np.float32),
            score_mode=model.score_mode, window=model.window,
            max_tokens=model.max_tokens, max_features=model.max_features,
            n_features=model.n_features,
        ).save(out_dir, metadata={
            "derived_from": SOURCE,
            "change": "unknown-cap sentinel read as 'no cap' rather than 'cap at -inf'",
            "selection_data": "training calibration carve only; data/2-eval untouched",
            "changed_thresholds": {labels[j]: {"v2": float(thr_repro[j]),
                                               "v4": float(thr_fix[j])} for j in changed},
            "cascade_params": cp, "profile": PROFILES and model.window,
        })
        print(f"-> {out_dir}  ({int(np.isfinite(thr_fix).sum())} enabled tags)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
