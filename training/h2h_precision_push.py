"""Push arm B's precision as hard as a stated recall floor allows.

Arm B is already the precision-max point of its own 350-trial cascade search, so
there is no better *operating point* to find at its current settings. What limits
its precision is the rule those settings were chosen under: per-label thresholds
are the **F0.5-optimum subject to a per-source recall floor**. That is two dials,
not one, and they do different jobs:

``recall_floor``  the constraint -- how much recall the tag is required to keep.
``beta``          how greedily precision is bought *inside* the admissible set.

Arm B uses beta = 0.5, which still trades some precision away for recall even
after the floor is satisfied. Driving beta toward 0 makes the selector take the
highest-precision threshold that still clears the floor: at beta = 0, F-beta
reduces to precision itself, so the rule becomes exactly "maximise precision
subject to recall >= floor". That is the dial to turn when the recall budget is
already decided and the only remaining question is how much precision it buys.

Both are swept here, because the floor alone does not answer "as much precision
as possible at this recall" -- it answers "what happens if I relax the recall
requirement", which is a different question.

Two things are held fixed so the trade is isolated:

* **the weights** -- gate and heads are refit once, deterministically, from arm
  B's recorded hyperparameters and the same fit split. Nothing is retrained per
  step; only the thresholds move.
* **the document gate** -- same threshold as arm B. The gate drives document
  precision and specificity; the tag floor drives tag precision. Moving both at
  once would make it impossible to say which produced the change.

## On selecting from this table

The thresholds are selected on **held-in calibration**, exactly as arm B's were.
The sealed corpora are then scored once per ladder point, which is a measurement
rather than a selection -- and it stops being that the moment somebody picks a
row by reading the sealed column. Choose the floor on the calibration curve; read
the sealed column to see what it actually delivered. The two are reported side by
side for precisely this reason.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.h2h_eval import assemble_arm, evaluate_corpus  # noqa: E402
from training.h2h_priority import PROJECT, eval_corpora  # noqa: E402
from training.h2h_score import _load_cached, predict_cascade  # noqa: E402
from training.quiet_cache import PROFILES, load_catalogue  # noqa: E402
from training.quiet_data import PRIORITY_TAGS  # noqa: E402
from training.quiet_fit import carve_holdin, load, priority_indices, score, train_corpora  # noqa: E402
from training.quiet_materialize import fit_disc_heads, fit_gate  # noqa: E402
from training.quiet_model import QuietCascade  # noqa: E402
from training.quiet_select import fbeta, select_per_label_robust  # noqa: E402

#: Arm B's own operating point, read from the artifact rather than retyped.
ARM_B = PROJECT / "models" / "cascade"


def _calibration_summary(S: np.ndarray, thr: np.ndarray, Y: np.ndarray,
                         tag_complete: np.ndarray, open_doc: np.ndarray,
                         labels: tuple[str, ...]) -> dict[str, float]:
    """Macro/micro precision, recall and F-betas on the calibration split.

    Same masked positive-unlabelled discipline as the sealed evaluator: a row
    whose gold is positive-only cannot supply a negative.
    """
    fired = (S >= thr) & open_doc[:, None]
    eligible = Y | tag_complete[:, None]
    pred = fired & eligible
    tp = (pred & Y).sum(axis=0).astype(np.float64)
    fp = (pred & ~Y).sum(axis=0).astype(np.float64)
    support = Y.sum(axis=0).astype(np.float64)
    keep = support > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        rec = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    pri = np.asarray([j for j, t in enumerate(labels)
                      if t in PRIORITY_TAGS and support[j] >= 30], dtype=np.int64)
    TP, FP, FN = tp[keep].sum(), fp[keep].sum(), (support[keep] - tp[keep]).sum()
    mP = TP / (TP + FP) if TP + FP else 0.0
    mR = TP / (TP + FN) if TP + FN else 0.0
    return {
        "macro_precision": float(prec[keep].mean()),
        "macro_recall": float(rec[keep].mean()),
        "macro_f05": float(fbeta(prec[keep], rec[keep], 0.5).mean()),
        "macro_f1": float(fbeta(prec[keep], rec[keep], 1.0).mean()),
        "macro_f2": float(fbeta(prec[keep], rec[keep], 2.0).mean()),
        "micro_precision": float(mP), "micro_recall": float(mR),
        "micro_f1": float(fbeta(np.asarray([mP]), np.asarray([mR]), 1.0)[0]),
        "priority_macro_precision": float(prec[pri].mean()) if pri.size else 0.0,
        "priority_macro_recall": float(rec[pri].mean()) if pri.size else 0.0,
        "priority_macro_f05": float(fbeta(prec[pri], rec[pri], 0.5).mean()) if pri.size else 0.0,
        "priority_min_recall": float(rec[pri].min()) if pri.size else 0.0,
        "n_tags_disabled": int((~np.isfinite(thr)).sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--floors", type=float, nargs="+", default=None,
                    help="defaults to arm B's own floor followed by the ladder below it")
    ap.add_argument("--betas", type=float, nargs="+", default=None,
                    help="F-beta used to pick inside the admissible set. 0.5 is arm "
                         "B's. Smaller buys precision more greedily; 0.0 is pure "
                         "precision subject to the recall floor.")
    ap.add_argument("--score-sealed", action="store_true")
    args = ap.parse_args()

    base = QuietCascade.load(ARM_B)
    meta = json.loads((ARM_B / "model.json").read_text(encoding="utf-8"))["metadata"]
    gate_p, head_p = meta["gate_params"], meta["head_params"]
    profile = meta["profile"]
    window, max_tokens, max_features = PROFILES[profile]
    labels = tuple(load_catalogue()["labels"])
    # The threshold-selection parameters are the CASCADE trial's, not the head
    # trial's -- `quiet_materialize` re-derives the tag cuts from
    # `cascade_trial["params"]`, and the head trial's own floor/margin belong to
    # the earlier, gate-less selection that only chose which weights to use.
    # Taking the head trial's by mistake makes the ladder's first rung *nearly*
    # arm B, which is worse than useless: every delta below it would be measured
    # against a baseline that is not the model being compared.
    trials = json.loads(
        (PROJECT / "tuning" / "cascade" / "trials.json").read_text(encoding="utf-8"))
    cascade_trial = next(t for t in trials if t["number"] == meta["cascade_trial"])
    cp = cascade_trial["params"]
    margin = cp.get("margin", 0.0)
    min_support = cp["min_support_fit"]
    base_floor = cp["recall_floor"]

    print(f"arm B operating point (cascade trial {meta['cascade_trial']}): "
          f"recall_floor={base_floor:.6f} margin={margin:.6f} "
          f"min_support_fit={min_support} gate_threshold={base.gate_threshold:.4f}",
          flush=True)

    ds = load(train_corpora(), profile=profile)
    fit_mask, calib_mask = carve_holdin(ds)
    calib = ds.subset(calib_mask)
    print(f"fit {int(fit_mask.sum()):,}  calibration {int(calib_mask.sum()):,}", flush=True)

    print("refitting gate and 58 heads once (arm B's recorded hyperparameters) ...",
          flush=True)
    gate_w, gate_b = fit_gate(ds, fit_mask, gate_p)
    W = fit_disc_heads(ds, fit_mask, head_p)

    g_cal = (calib.X @ gate_w + gate_b).astype(np.float32)
    S_cal = score(calib.X, W, mode="sum")
    Ycal = np.asarray(calib.Y.todense()).astype(bool)
    open_doc = g_cal >= base.gate_threshold
    print(f"gate opens on {open_doc.mean():.4f} of calibration rows", flush=True)

    # The first rung is arm B's exact floor, so it reproduces arm B and every
    # delta below it is measured against the real baseline.
    floors = args.floors or [base_floor, 0.72, 0.70, 0.68, 0.65]
    betas = args.betas or [0.5]
    rows: list[dict[str, Any]] = []
    for floor in floors:
      for beta in betas:
        thr, _ = select_per_label_robust(
            S_cal[open_doc], Ycal[open_doc], calib.tag_complete[open_doc],
            calib.corpus[open_doc], beta=beta, recall_floor=floor,
            margin=margin, min_support=min_support)
        summary = _calibration_summary(S_cal, thr, Ycal, calib.tag_complete,
                                       open_doc, labels)
        model = QuietCascade(
            labels=labels, gate_weights=gate_w, gate_intercept=gate_b,
            gate_threshold=base.gate_threshold, tag_weights=W.astype(np.float32),
            tag_thresholds=thr, score_mode="sum", window=window,
            max_tokens=max_tokens, max_features=max_features,
            n_features=int(load_catalogue()["n_features"]))
        tag = f"f{int(round(floor * 100)):02d}b{int(round(beta * 100)):02d}"
        out = PROJECT / "models" / f"cascade_{tag}"
        model.save(out, metadata={
            "derived_from": "arm B (cascade trial %s)" % meta.get("cascade_trial"),
            "recall_floor": floor, "selection_beta": beta,
            "margin": margin, "min_support_fit": min_support,
            "gate_threshold": base.gate_threshold,
            "gate_params": gate_p, "head_params": head_p, "profile": profile,
            "calibration": summary})
        rows.append({"floor": floor, "beta": beta, "tag": tag,
                     "model_dir": str(out), "calibration": summary})
        print(f"  floor {floor:.2f} beta {beta:.2f} -> calib priP={summary['priority_macro_precision']:.4f} "
              f"priR={summary['priority_macro_recall']:.4f} "
              f"macroP={summary['macro_precision']:.4f} "
              f"macroR={summary['macro_recall']:.4f} "
              f"microF1={summary['micro_f1']:.4f} "
              f"disabled={summary['n_tags_disabled']}", flush=True)

    if args.score_sealed:
        print("\nscoring each ladder point on the eight sealed corpora ...", flush=True)
        for row in rows:
            model = QuietCascade.load(Path(row["model_dir"]))
            per_corpus = {}
            for seed, corpus in enumerate(eval_corpora()):
                cached = _load_cached(corpus, "deep")
                fired, fired_doc, tag_scores = predict_cascade(model, cached["X"])
                per_corpus[corpus] = evaluate_corpus(
                    corpus, fired, fired_doc, cached["Y"], cached["tag_complete"],
                    cached["doc_target"], labels, seed=1000 * (seed + 1),
                    tag_scores=tag_scores)
            arm = assemble_arm(
                name=f"cascade-{row['tag']}",
                label=(f"steady-aim cascade, recall floor {row['floor']:.2f}, "
                       f"selection beta {row['beta']:.2f}"),
                per_corpus=per_corpus, p95_latency_ms=None, docs_per_s=None,
                extra={"recall_floor": row["floor"], "selection_beta": row["beta"],
                       "derived_from": "arm B",
                       "gate_threshold": base.gate_threshold})
            path = PROJECT / "evaluations" / f"cascade_{row['tag']}.json"
            path.write_text(json.dumps(arm, indent=1, sort_keys=True) + "\n",
                            encoding="utf-8")
            row["sealed"] = {k: (v["value"] if isinstance(v, dict) else v)
                             for k, v in arm["metrics"].items()}
            m = row["sealed"]
            print(f"  floor {row['floor']:.2f} beta {row['beta']:.2f} -> sealed macroP="
                  f"{m['precision_macro_catalogue']:.4f} macroR={m['recall_macro_catalogue']:.4f} "
                  f"microF1={m['micro_f1']:.4f} macroF1={m['f1_macro_catalogue']:.4f} "
                  f"priP={m['priority_macro_precision']:.4f} "
                  f"priR={m['priority_macro_recall']:.4f}", flush=True)

    (PROJECT / "tuning" / "precision_push.json").write_text(
        json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    print(f"\nladder -> {PROJECT / 'tuning' / 'precision_push.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
