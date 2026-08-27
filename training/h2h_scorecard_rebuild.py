"""Refit the cascade on the scorecard's 61-label taxonomy and score it sealed.

Structurally this is `h2h_cascade_rebuild.py` with two substitutions and nothing
else, so that a difference in the result is attributable to those two:

1. **the catalogue** — 61 scorecard labels from
   `h2h_scorecard_catalogue.py`, with `given`/`family`/`middle_name` and
   `street_number_and_name` restored as tags in their own right rather than
   folded into `full_name` and `address`;
2. **the threshold rule** — `h2h_thresholds_v4.select_per_label(corrected_cap=True)`,
   which requires a source group to hold at least `MIN_TAIL_EVENTS` positives in
   the tail before it may set the group-recall cap.

Gate hyperparameters, head hyperparameters, the read profile, the calibration
carve and the evaluator are arm B's, unchanged.

## Pointing the loaders at the new cache

`quiet_fit` and `h2h_score` bind `CACHE_ROOT` at import time (`from ... import
CACHE_ROOT`), so rebinding `quiet_cache.CACHE_ROOT` alone would move
`load_catalogue()` to the new catalogue while leaving those two modules reading
features and labels from the old one. That mismatch would not raise — the arrays
are the same shape until the label count changes — it would silently score a
61-label model against 58-label gold.

So all three names are rebound together, in one place, and the catalogue is
checked afterwards. Nothing else in the tree is modified: `projects/pii-quiet-alarm/cache`
is read for nothing here, and the 128 published results that depend on it are
untouched.
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

from training import h2h_score, quiet_cache, quiet_fit  # noqa: E402

PROJECT = Path("projects/pii-scorecard-60")
SOURCE_MODEL = Path("projects/pii-head-to-head-v1/models/cascade_balanced")


def retarget_cache(root: Path, expect_labels: int) -> dict:
    """Rebind every cache root that was bound at import time. See the header."""
    root = root.resolve()
    quiet_cache.CACHE_ROOT = root      # load_catalogue() reads this at call time
    quiet_fit.CACHE_ROOT = root        # bound by `from quiet_cache import CACHE_ROOT`
    h2h_score.QUIET_CACHE = root       # same, under a different alias
    cat = quiet_cache.load_catalogue()
    if len(cat["labels"]) != expect_labels:
        raise SystemExit(
            f"catalogue at {root} has {len(cat['labels'])} labels, expected "
            f"{expect_labels}. Refusing to fit: the label space is the point of "
            f"this run and getting it wrong is not visible in any later number.")
    return cat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=PROJECT / "cache")
    ap.add_argument("--alpha", type=float, default=1e-2)
    ap.add_argument("--balance", default="equal", choices=["none", "equal"])
    ap.add_argument("--name", default="cascade_scorecard61")
    ap.add_argument("--labels", type=int, default=61)
    args = ap.parse_args()

    cat = retarget_cache(args.cache, args.labels)
    labels = tuple(cat["labels"])
    print(f"catalogue: {len(labels)} labels from {args.cache}", flush=True)

    # Imported only after the rebind, so their module-level lookups see the new
    # root. (They resolve CACHE_ROOT through the modules patched above.)
    from training.h2h_eval import assemble_arm, evaluate_corpus  # noqa: E402
    from training.h2h_gate_diag import is_real  # noqa: E402
    from training.h2h_priority import eval_corpora  # noqa: E402
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

    t0 = time.perf_counter()
    ds = load(train_corpora(), profile=profile)
    fit_mask, calib_mask = carve_holdin(ds)
    calib = ds.subset(calib_mask)
    known = ds.doc_target >= 0
    names = np.asarray(ds.corpus_names)
    real_row = np.asarray([is_real(n) for n in names])[ds.corpus]
    print(f"loaded {len(ds):,} training rows in {time.perf_counter() - t0:.0f}s; "
          f"fit {int(fit_mask.sum()):,} / calib {int(calib_mask.sum()):,}", flush=True)

    # ------------------------------------------------------------------- gate
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

    # ------------------------------------------------------------------ heads
    t0 = time.perf_counter()
    print(f"refitting {len(labels)} heads (arm B hyperparameters, unchanged) ...",
          flush=True)
    W = fit_disc_heads(ds, fit_mask, hp)
    print(f"  {time.perf_counter() - t0:.0f}s", flush=True)

    # ------------------------------------------------------- operating points
    g_cal = (calib.X @ gate_w + gate_b).astype(np.float32)
    groups = group_masks(calib.corpus, calib.corpus_names)
    sel = groups["real"] if gp["select_on"] == "real" else np.ones(len(g_cal), bool)
    cut, doc = select_doc_threshold_robust(
        g_cal[sel], calib.doc_target[sel], calib.corpus[sel],
        recall_floor=gp["recall_target"], specificity_floor=gp["spec_target"],
        margin=gp["margin"])
    cut = float(cut)
    print(f"gate threshold {cut:.4f}  (calibration real: P={doc['precision']:.4f} "
          f"R={doc['recall']:.4f} sp={doc['specificity']:.4f})", flush=True)

    S_cal = score(calib.X, W, mode="sum")
    Ycal = np.asarray(calib.Y.todense()).astype(bool)
    open_doc = g_cal >= cut
    thr, report = select_per_label(
        S_cal[open_doc], Ycal[open_doc], calib.tag_complete[open_doc],
        calib.corpus[open_doc], beta=0.5, recall_floor=cp["recall_floor"],
        margin=cp.get("margin", 0.0), min_support=cp["min_support_fit"],
        corrected_cap=True)
    n_enabled = int(np.isfinite(thr).sum())
    print(f"selected {n_enabled} of {len(labels)} thresholds "
          f"({len(labels) - n_enabled} below min support)", flush=True)

    restored = ("sensitive_pii_given_name", "sensitive_pii_family_name",
                "sensitive_pii_middle_name", "sensitive_pii_street_number_and_name")
    print("\nthe four restored tags, as selected on the training carve:", flush=True)
    for tag in restored:
        j = labels.index(tag)
        r = report[j]
        print(f"  {tag:<46} thr={thr[j]:>9.4f}  support={r['support']:>7.0f}  "
              f"P={r['precision']:.4f} R={r['recall']:.4f} F0.5={r['f']:.4f} "
              f"[{r['branch']}]", flush=True)

    model = QuietCascade(
        labels=labels, gate_weights=gate_w, gate_intercept=gate_b,
        gate_threshold=cut, tag_weights=W.astype(np.float32),
        tag_thresholds=thr.astype(np.float32), score_mode="sum", window=window,
        max_tokens=max_tokens, max_features=max_features,
        n_features=int(cat["n_features"]))
    out_dir = PROJECT / "models" / args.name
    model.save(out_dir, metadata={
        "derived_from": str(SOURCE_MODEL), "catalogue": str(args.cache / "catalogue.json"),
        "change": "scorecard 61-label taxonomy; name/address collapse reversed",
        "gate_alpha": args.alpha, "gate_balance": args.balance,
        "gate_params": gp, "head_params": hp, "cascade_params": cp,
        "profile": profile, "calibration_doc_real": doc,
        "threshold_rule": "h2h_thresholds_v4.select_per_label(corrected_cap=True)"})
    (PROJECT / "probe").mkdir(parents=True, exist_ok=True)
    (PROJECT / "probe" / "threshold_report.json").write_text(
        json.dumps({labels[r["label"]]: r for r in report}, indent=1), encoding="utf-8")
    print(f"\n-> {out_dir}  ({n_enabled} enabled tags)", flush=True)

    # ------------------------------------------------------------------ score
    per_corpus = {}
    for seed, corpus in enumerate(eval_corpora()):
        t0 = time.perf_counter()
        cached = _load_cached(corpus, profile)
        fired, fired_doc, tag_scores = predict_cascade(model, cached["X"])
        per_corpus[corpus] = evaluate_corpus(
            corpus, fired, fired_doc, cached["Y"], cached["tag_complete"],
            cached["doc_target"], labels, seed=1000 * (seed + 1),
            tag_scores=tag_scores)
        print(f"  {corpus:<48s} {cached['X'].shape[0]:>7,} docs  "
              f"{time.perf_counter() - t0:5.1f}s", flush=True)

    arm = assemble_arm(
        name=args.name, label=f"scorecard 61-label cascade ({len(labels)} tags)",
        per_corpus=per_corpus, p95_latency_ms=None, docs_per_s=None,
        extra={"catalogue": str(args.cache / "catalogue.json"),
               "n_labels": len(labels), "n_enabled": n_enabled,
               "read_limit": 12_000, "profile": profile})
    path = PROJECT / "evaluations" / f"arm_{args.name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(arm, indent=1, sort_keys=True) + "\n", encoding="utf-8")

    v = lambda a, k: (a["metrics"][k]["value"] if isinstance(a["metrics"][k], dict)  # noqa: E731
                      else a["metrics"][k])
    print(f"\n{'metric':<34}{args.name:>22}")
    for k in ("macro_f05", "macro_f2", "micro_f1", "priority_macro_f05",
              "equal_corpus_doc_recall", "equal_corpus_doc_precision",
              "equal_corpus_doc_specificity", "recall_macro_catalogue",
              "precision_macro_catalogue", "severity_recall_min", "prediction_rate"):
        val = v(arm, k)
        if val is not None:
            print(f"{k:<34}{val:>22.4f}")
    print(f"\n-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
