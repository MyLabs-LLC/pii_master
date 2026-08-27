"""Do the target-box thresholds survive transfer to an unseen source?

`h2h_loco` measured the architecture under the baseline threshold policy and
found a mean transfer gap of 0.1604, carried almost entirely by recall
(-0.2556 against precision's -0.1004). The shipped variants — `p88r90` and
`p90r85b1` — buy precision by spending exactly that recall, so the obvious
worry is that they transfer worse than the baseline they were derived from.

That worry is inference. This module measures it.

## What is held constant

The folds, the fit, and the evaluator are `h2h_loco`'s. Within each fold the
gate and the 61 heads are fitted on the other eight sources, and then **three
threshold policies are derived from that fold's own calibration carve**:

* `baseline`   — `select_per_label`, the policy `h2h_loco` measured;
* `p88r90`     — `select_box(P>=0.88, R>=0.90, beta=0.5)`;
* `p90r85b1`   — `select_box(P>=0.90, R>=0.85, beta=1.0)`.

Same weights, same gate, same calibration rows — the only difference between
the three numbers on any fold is where the per-tag thresholds sit. That is the
comparison the shipped models represent, so it is the one worth transferring.

**The thresholds are re-derived inside every fold, never reused from the
packaged models.** A packaged threshold was selected on a carve that included
the held-out source; importing it would leak the source back in and report a
transfer gap smaller than the truth.

## Reading the output

Per fold, three micro F1 values on a source none of the three has seen. If the
variants' advantage on the sealed suite (0.8470 and 0.8564 against the
baseline's 0.7299) narrows, vanishes or inverts here, then that advantage is a
property of the eight corpora rather than of the models.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.linear_model import SGDClassifier  # noqa: E402

from training.h2h_loco import PAIRS  # noqa: E402
from training.h2h_scorecard_rebuild import SOURCE_MODEL, retarget_cache  # noqa: E402

SC = Path("projects/pii-scorecard-60")
OUT = Path("projects/pii-target-8070")

#: name -> (p_target, r_target, beta), matching the packaged models' metadata.
BOXES = {
    "p88r90": (0.88, 0.90, 0.5),
    "p90r85b1": (0.90, 0.85, 1.0),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=1e-2)
    ap.add_argument("--out", type=Path,
                    default=OUT / "evaluations/loco_variants.json")
    ap.add_argument("--only", default=None, help="run a single fold, by train stem")
    args = ap.parse_args()

    cat = retarget_cache(SC / "cache", 61)
    labels = tuple(cat["labels"])

    from training.h2h_eval import evaluate_corpus  # noqa: E402
    from training.h2h_gate_diag import is_real  # noqa: E402
    from training.h2h_score import _load_cached, predict_cascade  # noqa: E402
    from training.h2h_target_box import select_box  # noqa: E402
    from training.h2h_thresholds_v4 import select_per_label  # noqa: E402
    from training.quiet_cache import PROFILES  # noqa: E402
    from training.quiet_fit import carve_holdin, load, score, train_corpora  # noqa: E402
    from training.quiet_materialize import fit_disc_heads  # noqa: E402
    from training.quiet_model import QuietCascade  # noqa: E402
    from training.quiet_objective import group_masks  # noqa: E402
    from training.quiet_select import select_doc_threshold_robust  # noqa: E402

    meta = json.loads((SOURCE_MODEL / "model.json").read_text(
        encoding="utf-8"))["metadata"]
    gp, hp, cp = meta["gate_params"], meta["head_params"], meta["cascade_params"]
    profile = meta["profile"]
    window, max_tokens, max_features = PROFILES[profile]

    all_train = train_corpora()
    folds = [f for f in PAIRS if f in all_train]
    if args.only:
        folds = [f for f in folds if f == args.only]
    print(f"{len(folds)} folds x {1 + len(BOXES)} threshold policies\n", flush=True)

    results = []
    for held in folds:
        t0 = time.perf_counter()
        keep = [c for c in all_train if c != held]
        ds = load(keep, profile=profile)
        fit_mask, calib_mask = carve_holdin(ds)
        calib = ds.subset(calib_mask)
        known = ds.doc_target >= 0
        names = np.asarray(ds.corpus_names)
        real_row = np.asarray([is_real(n) for n in names])[ds.corpus]

        rows = fit_mask & known
        y = ds.doc_target[rows].astype(bool)
        rf = real_row[rows]
        w = np.where(y, 1.0, gp["neg_weight"])
        nr, nsy = int(rf.sum()), int((~rf).sum())
        w = w * np.where(rf, nsy / max(nr, 1), 1.0)
        clf = SGDClassifier(loss=gp["loss"], alpha=args.alpha,
                            max_iter=gp["max_iter"], tol=None, random_state=7)
        clf.fit(ds.X[rows], y.astype(np.int8), sample_weight=w)
        gate_w = clf.coef_.ravel().astype(np.float32)
        gate_b = float(clf.intercept_[0])
        W = fit_disc_heads(ds, fit_mask, hp)

        g_cal = (calib.X @ gate_w + gate_b).astype(np.float32)
        groups = group_masks(calib.corpus, calib.corpus_names)
        sel = groups["real"] if gp["select_on"] == "real" else np.ones(len(g_cal), bool)
        cut, _ = select_doc_threshold_robust(
            g_cal[sel], calib.doc_target[sel], calib.corpus[sel],
            recall_floor=gp["recall_target"], specificity_floor=gp["spec_target"],
            margin=gp["margin"])
        cut = float(cut)
        S_cal = score(calib.X, W, mode="sum")
        Ycal = np.asarray(calib.Y.todense()).astype(bool)
        open_doc = g_cal >= cut

        # Every policy sees exactly the same gate-admitted calibration rows.
        S_o, Y_o, TC_o = S_cal[open_doc], Ycal[open_doc], calib.tag_complete[open_doc]
        policies = {}
        thr_base, _ = select_per_label(
            S_o, Y_o, TC_o, calib.corpus[open_doc], beta=0.5,
            recall_floor=cp["recall_floor"], margin=cp.get("margin", 0.0),
            min_support=cp["min_support_fit"], corrected_cap=True)
        policies["baseline"] = (thr_base, {})
        for name, (p_t, r_t, beta) in BOXES.items():
            thr, report = select_box(S_o, Y_o, TC_o, labels, p_target=p_t,
                                     r_target=r_t, beta=beta, min_support=30)
            policies[name] = (thr, dict(Counter(
                r["verdict"] for r in report.values())))

        target = PAIRS[held]
        cached = _load_cached(target, profile)
        row = {"held_out_train": held, "evaluated_on": target,
               "n_train_rows": int(len(ds)), "policies": {}}
        for name, (thr, verdicts) in policies.items():
            model = QuietCascade(
                labels=labels, gate_weights=gate_w, gate_intercept=gate_b,
                gate_threshold=cut, tag_weights=W.astype(np.float32),
                tag_thresholds=thr.astype(np.float32), score_mode="sum",
                window=window, max_tokens=max_tokens, max_features=max_features,
                n_features=int(cat["n_features"]))
            fired, fired_doc, tag_scores = predict_cascade(model, cached["X"])
            body = evaluate_corpus(target, fired, fired_doc, cached["Y"],
                                   cached["tag_complete"], cached["doc_target"],
                                   labels, seed=7919, tag_scores=tag_scores)
            s = body["summary"]
            row["policies"][name] = {
                "metrics": {k: s[k] for k in
                            ("f1_micro", "precision_micro", "recall_micro",
                             "f2_macro_catalogue", "f05_macro_catalogue",
                             "recall_macro_catalogue", "precision_macro_catalogue")
                            if s.get(k) is not None},
                "verdicts": verdicts,
                "enabled_tags": int(np.isfinite(thr).sum())}
        row["n_eval_rows"] = body["n_rows"]
        row["seconds"] = round(time.perf_counter() - t0, 1)
        results.append(row)

        cells = "  ".join(
            f"{n}={row['policies'][n]['metrics'].get('f1_micro'):.4f}"
            if row["policies"][n]["metrics"].get("f1_micro") is not None
            else f"{n}=n/a" for n in ("baseline", *BOXES))
        print(f"  {held[:40]:<40} -> {target[:30]:<30} {cells}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"folds": len(results), "boxes": BOXES, "pairs": PAIRS,
         "results": results}, indent=1), encoding="utf-8")
    print(f"\n{len(results)} folds -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
