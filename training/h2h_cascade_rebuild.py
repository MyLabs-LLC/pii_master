"""Carry the balanced, regularised gate through a full cascade and score it sealed.

The gate diagnosis measured a real improvement -- sealed real AUC 0.8453 ->
0.8804 -- but only on ranking. A ranking improvement is not a product
improvement: the cascade's document answer depends on where the gate is cut, and
its tag answers depend on which documents the gate admits, so an AUC gain can
evaporate once the operating points are re-derived.

This closes that. The gate is refit balanced and regularised, the heads are refit
unchanged (they do not depend on the gate), both operating points are re-derived
exactly as `quiet_materialize` derives them, and the result is scored on the
eight sealed corpora by the same fixed evaluator every other arm went through.

Only the gate changes. Head hyperparameters, the tag-threshold rule, the read
profile and the evaluator are arm B's.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.linear_model import SGDClassifier  # noqa: E402

from training.h2h_eval import assemble_arm, evaluate_corpus  # noqa: E402
from training.h2h_gate_diag import is_real  # noqa: E402
from training.h2h_priority import PROJECT, eval_corpora  # noqa: E402
from training.h2h_score import _load_cached, predict_cascade  # noqa: E402
from training.quiet_cache import PROFILES, load_catalogue  # noqa: E402
from training.quiet_fit import carve_holdin, load, score, train_corpora  # noqa: E402
from training.quiet_materialize import fit_disc_heads  # noqa: E402
from training.quiet_model import QuietCascade  # noqa: E402
from training.quiet_objective import group_masks  # noqa: E402
from training.quiet_select import (  # noqa: E402
    select_doc_threshold_robust, select_per_label_robust,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=1e-2)
    ap.add_argument("--balance", default="equal", choices=["none", "equal"])
    ap.add_argument("--name", default="cascade_balanced")
    args = ap.parse_args()

    meta = json.loads((PROJECT / "models" / "cascade" / "model.json").read_text(
        encoding="utf-8"))["metadata"]
    gp, hp = meta["gate_params"], meta["head_params"]
    trials = json.loads((PROJECT / "tuning" / "cascade" / "trials.json").read_text(
        encoding="utf-8"))
    cp = next(t for t in trials if t["number"] == meta["cascade_trial"])["params"]
    profile = meta["profile"]
    window, max_tokens, max_features = PROFILES[profile]
    labels = tuple(load_catalogue()["labels"])

    ds = load(train_corpora(), profile=profile)
    fit_mask, calib_mask = carve_holdin(ds)
    calib = ds.subset(calib_mask)
    known = ds.doc_target >= 0
    names = np.asarray(ds.corpus_names)
    real_row = np.asarray([is_real(n) for n in names])[ds.corpus]

    rows = fit_mask & known
    y = ds.doc_target[rows].astype(bool)
    rf = real_row[rows]
    nr, nsy = int(rf.sum()), int((~rf).sum())
    w = np.where(y, 1.0, gp["neg_weight"])
    if args.balance == "equal":
        w = w * np.where(rf, nsy / max(nr, 1), 1.0)
    print(f"gate: alpha={args.alpha:.3g} balance={args.balance} "
          f"(real {nr:,} of {len(y):,} fit rows)", flush=True)
    clf = SGDClassifier(loss=gp["loss"], alpha=args.alpha, max_iter=gp["max_iter"],
                        tol=None, random_state=7)
    clf.fit(ds.X[rows], y.astype(np.int8), sample_weight=w)
    gate_w = clf.coef_.ravel().astype(np.float32)
    gate_b = float(clf.intercept_[0])

    print("refitting 58 heads (arm B hyperparameters, unchanged) ...", flush=True)
    W = fit_disc_heads(ds, fit_mask, hp)

    g_cal = (calib.X @ gate_w + gate_b).astype(np.float32)
    groups = group_masks(calib.corpus, calib.corpus_names)
    sel = groups["real"] if gp["select_on"] == "real" else np.ones(len(g_cal), bool)
    cut, doc = select_doc_threshold_robust(
        g_cal[sel], calib.doc_target[sel], calib.corpus[sel],
        recall_floor=gp["recall_target"], specificity_floor=gp["spec_target"],
        margin=gp["margin"])
    # Arm B's cascade trial carried a `gate_shift` of -2.196. It is an ABSOLUTE
    # offset, searched against arm B's score scale where the cut sits near 162.
    # Regularising at alpha=1e-2 shrinks the weights by orders of magnitude, so
    # the same -2.196 is enormous relative to the new spread: applied naively it
    # moved the cut to -2.13 and dropped document specificity from 0.883 to
    # 0.011. A tuned constant does not survive a change of scale, so it is not
    # carried over -- the selector's own cut is used, and the joint shift would
    # have to be re-searched on this gate to mean anything.
    print(f"gate threshold {float(cut):.4f}  (calibration real: "
          f"P={doc['precision']:.4f} R={doc['recall']:.4f} sp={doc['specificity']:.4f})",
          flush=True)
    cut = float(cut)

    S_cal = score(calib.X, W, mode="sum")
    Ycal = np.asarray(calib.Y.todense()).astype(bool)
    open_doc = g_cal >= cut
    thr, _ = select_per_label_robust(
        S_cal[open_doc], Ycal[open_doc], calib.tag_complete[open_doc],
        calib.corpus[open_doc], beta=0.5, recall_floor=cp["recall_floor"],
        margin=cp.get("margin", 0.0), min_support=cp["min_support_fit"])

    model = QuietCascade(
        labels=labels, gate_weights=gate_w, gate_intercept=gate_b,
        gate_threshold=cut, tag_weights=W.astype(np.float32), tag_thresholds=thr,
        score_mode="sum", window=window, max_tokens=max_tokens,
        max_features=max_features, n_features=int(load_catalogue()["n_features"]))
    out_dir = PROJECT / "models" / args.name
    model.save(out_dir, metadata={
        "derived_from": "arm B", "change": "gate refit balanced + regularised",
        "gate_alpha": args.alpha, "gate_balance": args.balance,
        "gate_params": gp, "head_params": hp, "cascade_params": cp,
        "profile": profile, "calibration_doc_real": doc})
    print(f"-> {out_dir}  ({int(np.isfinite(thr).sum())} enabled tags)", flush=True)

    per_corpus = {}
    for seed, corpus in enumerate(eval_corpora()):
        cached = _load_cached(corpus, profile)
        fired, fired_doc, tag_scores = predict_cascade(model, cached["X"])
        per_corpus[corpus] = evaluate_corpus(
            corpus, fired, fired_doc, cached["Y"], cached["tag_complete"],
            cached["doc_target"], labels, seed=1000 * (seed + 1),
            tag_scores=tag_scores)
    lat = json.loads((PROJECT / "evaluations" / "latency_B.json").read_text(
        encoding="utf-8"))
    arm = assemble_arm(name=f"arm-B-{args.name}",
                       label="steady-aim cascade, balanced + regularised gate",
                       per_corpus=per_corpus, p95_latency_ms=lat.get("p95_ms"),
                       docs_per_s=lat.get("docs_per_s"),
                       extra={"derived_from": "arm B", "gate_alpha": args.alpha,
                              "gate_balance": args.balance,
                              "latency_note": "carried from arm B; gate cost unchanged"})
    path = PROJECT / "evaluations" / f"arm_{args.name}.json"
    path.write_text(json.dumps(arm, indent=1, sort_keys=True) + "\n", encoding="utf-8")

    base = json.loads((PROJECT / "evaluations" / "arm_B.json").read_text(encoding="utf-8"))
    v = lambda a, k: (a["metrics"][k]["value"] if isinstance(a["metrics"][k], dict)  # noqa: E731
                      else a["metrics"][k])
    print(f"\n{'metric':<34}{'arm B':>10}{'balanced':>11}{'delta':>10}")
    for k in ("equal_corpus_doc_recall", "equal_corpus_doc_precision",
              "equal_corpus_doc_specificity", "macro_f2", "micro_f1",
              "priority_macro_f05", "recall_macro_catalogue",
              "precision_macro_catalogue", "severity_recall_min", "prediction_rate"):
        a, b = v(base, k), v(arm, k)
        if a is None or b is None:
            continue
        print(f"{k:<34}{a:>10.4f}{b:>11.4f}{b - a:>+10.4f}")
    print(f"\n-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
