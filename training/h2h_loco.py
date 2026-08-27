"""Leave-one-corpus-out: how well does this architecture transfer to an unseen source?

Every headline in this repository is measured on corpora whose training halves the
model was fitted on. That answers "how good is it here" and says nothing about
"how good is it somewhere new" — and the one out-of-distribution corpus that could
have answered the second question is now 80% training data.

This answers it from existing data. For each of the nine source families: **train
on the other eight, evaluate on this one's sealed split.** The model has then never
seen a single document from that source, nor anything drawn from its generator, so
the number is a genuine transfer measurement rather than an interpolation.

Nine folds, each a full refit — gate, 61 heads, and threshold selection on that
fold's own calibration carve. Nothing is shared between folds except the code.

## Reading the output

`loco` is the score on the held-out source. `in_dist` is what the all-corpora model
scores on the same corpus. The gap between them is what this repository has been
unable to see:

* a small gap means the source is learnable from the others — its documents look
  like something already in the training mix;
* a large gap means the source carries something unique, and every published
  number for it is an interpolation rather than a prediction.

The mean gap across the nine is the honest estimate of what to expect on a source
that is genuinely new.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.linear_model import SGDClassifier  # noqa: E402

from training.h2h_scorecard_rebuild import SOURCE_MODEL, retarget_cache  # noqa: E402

SC = Path("projects/pii-scorecard-60")
OUT = Path("projects/pii-target-8070")

#: train corpus stem -> the sealed corpus drawn from the same source. Written out
#: rather than inferred: the names do not share a prefix (`pii_trainset` pairs with
#: `pii_holdout`), and a wrong pairing would silently score a fold on data it did
#: train on, which is the exact failure this module exists to avoid.
PAIRS = {
    "148775_pii2_train_98.81k": "30000_pii2_eval_25.15k",
    "151708_openpii_pii_train_151.71k": "38937_openpii_pii_eval_38.94k",
    "15986_datax-dualjudge-trainset-5.36k": "4000_datax-dualjudge-evalset-1.32k",
    "21743_nemotron_train_20.80k": "5617_nemotron_eval_5.36k",
    "23693_govdocs2-dualjudge-train80-12.86k": "6589_govdocs2-dualjudge-eval20-3.53k",
    "41429_betterdataai_ner_silver_train_41.43k": "10360_betterdataai_ner_silver_eval_10.36k",
    "42504_ai4privacy_pii_masking_train_42.50k": "10626_ai4privacy_pii_masking_eval_10.63k",
    "85593_pii_trainset_85.59k": "20000_pii_holdout_20.00k",
    "1290_synthetic_pdf_train_1.27k": "322_synthetic_pdf_eval_318",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=1e-2)
    ap.add_argument("--out", type=Path, default=OUT / "evaluations/loco.json")
    ap.add_argument("--only", default=None, help="run a single fold, by train stem")
    args = ap.parse_args()

    cat = retarget_cache(SC / "cache", 61)
    labels = tuple(cat["labels"])

    from training.h2h_eval import evaluate_corpus  # noqa: E402
    from training.h2h_gate_diag import is_real  # noqa: E402
    from training.h2h_score import _load_cached, predict_cascade  # noqa: E402
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
    print(f"{len(folds)} folds; each trains on {len(all_train) - 1} of "
          f"{len(all_train)} sources\n", flush=True)

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
        thr, _ = select_per_label(
            S_cal[open_doc], Ycal[open_doc], calib.tag_complete[open_doc],
            calib.corpus[open_doc], beta=0.5, recall_floor=cp["recall_floor"],
            margin=cp.get("margin", 0.0), min_support=cp["min_support_fit"],
            corrected_cap=True)

        model = QuietCascade(
            labels=labels, gate_weights=gate_w, gate_intercept=gate_b,
            gate_threshold=cut, tag_weights=W.astype(np.float32),
            tag_thresholds=thr.astype(np.float32), score_mode="sum", window=window,
            max_tokens=max_tokens, max_features=max_features,
            n_features=int(cat["n_features"]))

        target = PAIRS[held]
        cached = _load_cached(target, profile)
        fired, fired_doc, tag_scores = predict_cascade(model, cached["X"])
        body = evaluate_corpus(target, fired, fired_doc, cached["Y"],
                               cached["tag_complete"], cached["doc_target"],
                               labels, seed=7919, tag_scores=tag_scores)
        s = body["summary"]
        row = {"held_out_train": held, "evaluated_on": target,
               "n_train_rows": int(len(ds)), "n_eval_rows": body["n_rows"],
               "metrics": {k: s[k] for k in
                           ("f1_micro", "precision_micro", "recall_micro",
                            "f2_macro_catalogue", "f05_macro_catalogue",
                            "recall_macro_catalogue", "precision_macro_catalogue")
                           if s.get(k) is not None},
               "seconds": round(time.perf_counter() - t0, 1)}
        results.append(row)
        f1 = row["metrics"].get("f1_micro")
        print(f"  held out {held[:42]:<42} -> {target[:34]:<34} "
              f"micro F1 {f1:.4f}" if f1 is not None else
              f"  held out {held[:42]:<42} -> {target[:34]:<34} (no micro F1)",
              flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"folds": len(results), "pairs": PAIRS,
                                    "results": results}, indent=1), encoding="utf-8")
    print(f"\n{len(results)} folds -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
