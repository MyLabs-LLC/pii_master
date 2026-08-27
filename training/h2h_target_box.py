"""Re-select every tag threshold to land inside a (precision, recall) box.

The lineage's selector optimises F-beta subject to a recall floor and a
group-recall cap. It has no notion of a precision requirement, which is why
`cascade_scorecard61` ships tags at 0.5% precision that it records as successes.
This asks a different and simpler question, the one the declared target asks:

    is there a threshold at which this tag has P >= p and R >= r, and if so,
    which one?

Nothing is retrained. The gate, the weights and the feature space are
`cascade_scorecard61`'s, untouched; only the 61 comparison points move. That is
why the arms this produces are directly comparable with it and why their latency
is identical by construction.

## The three outcomes, and what each means

  IN BOX          the curve enters the box. Among in-box points take the F0.5
                  optimum -- precision-led, matching the declared ranker, so the
                  choice inside the box is not arbitrary.
  UNREACHABLE     the curve exists and never enters the box. No threshold fixes
                  this; the head or the gold has to improve. The tag keeps its
                  best F0.5 point and is REPORTED as missing the target rather
                  than silently parked somewhere flattering.
  NOT MEASURABLE  under `min_support` positives on the calibration carve. Not a
                  failure -- an absence of evidence, and named as one.

## Selection is on the calibration carve, never the sealed set

`quiet_fit.carve_holdin`'s calib side: training rows the cascade's weights were
not fitted to. The sealed corpora are scored once, afterwards, by the caller. A
threshold chosen on the data it is later judged on is not a threshold, it is a
memory -- which is the defect this project already found in `v3`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.h2h_scorecard_rebuild import retarget_cache  # noqa: E402
from training.quiet_select import fbeta, sweep  # noqa: E402

SCORECARD = Path("projects/pii-scorecard-60")
PROJECT = Path("projects/pii-target-8070")


def select_box(S, Y, tag_complete, labels, *, p_target, r_target, beta=0.5,
               min_support=30):
    thresholds = np.full(len(labels), np.inf, dtype=np.float32)
    report = {}
    for j, tag in enumerate(labels):
        pos = Y[:, j].astype(bool)
        n = int(pos.sum())
        if n < min_support:
            report[tag] = {"support": n, "verdict": "not_measurable",
                           "precision": None, "recall": None, "threshold": None}
            continue
        precision, recall, thr = sweep(S[:, j], pos, tag_complete & ~pos)
        if not len(thr):
            report[tag] = {"support": n, "verdict": "empty_sweep",
                           "precision": None, "recall": None, "threshold": None}
            continue
        f = fbeta(precision, recall, beta)
        in_box = (precision >= p_target) & (recall >= r_target)
        if in_box.any():
            idx = int(np.flatnonzero(in_box)[np.argmax(f[in_box])])
            verdict = "in_box"
        else:
            idx = int(np.argmax(f))
            verdict = "unreachable"
        thresholds[j] = thr[idx]
        report[tag] = {
            "support": n, "verdict": verdict, "threshold": float(thr[idx]),
            "precision": float(precision[idx]), "recall": float(recall[idx]),
            "f05": float(f[idx]),
            "max_precision_at_recall_target": (
                float(precision[recall >= r_target].max())
                if (recall >= r_target).any() else 0.0),
            "max_recall_at_precision_target": (
                float(recall[precision >= p_target].max())
                if (precision >= p_target).any() else 0.0),
        }
    return thresholds, report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", type=float, required=True)
    ap.add_argument("--recall", type=float, required=True)
    ap.add_argument("--source", default="cascade_scorecard61")
    ap.add_argument("--out-project", type=Path, default=None,
                    help="where to write the model (default: pii-target-8070)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--min-support", type=int, default=30)
    ap.add_argument("--beta", type=float, default=0.5,
                    help="which F-beta picks the point INSIDE the box")
    args = ap.parse_args()

    cat = retarget_cache(SCORECARD / "cache", 61)
    labels = tuple(cat["labels"])

    from training.quiet_fit import carve_holdin, load, score, train_corpora  # noqa: E402
    from training.quiet_model import QuietCascade  # noqa: E402

    model = QuietCascade.load(SCORECARD / "models" / args.source)
    ds = load(train_corpora(), profile="deep")
    _, calib_mask = carve_holdin(ds)
    calib = ds.subset(calib_mask)
    g = (calib.X @ model.gate_weights + model.gate_intercept).astype(np.float32)
    open_doc = g >= model.gate_threshold
    S = score(calib.X[open_doc], model.tag_weights, mode="sum")
    Y = np.asarray(calib.Y[open_doc].todense()).astype(bool)
    TC = calib.tag_complete[open_doc]
    print(f"{S.shape[0]:,} gate-admitted calibration rows; "
          f"target P>={args.precision} R>={args.recall}", flush=True)

    thr, report = select_box(S, Y, TC, labels, p_target=args.precision,
                             r_target=args.recall, beta=args.beta,
                             min_support=args.min_support)
    from collections import Counter
    counts = Counter(r["verdict"] for r in report.values())
    print("selection:", dict(counts), flush=True)
    miss = [t for t, r in report.items() if r["verdict"] == "unreachable"]
    if miss:
        print(f"\ncannot reach the box at any threshold ({len(miss)}):", flush=True)
        for t in miss:
            r = report[t]
            print(f"  {t.replace('sensitive_', ''):<50} sup={r['support']:<6} "
                  f"bestP@R={r['max_precision_at_recall_target']:.4f}  "
                  f"chosen P={r['precision']:.4f} R={r['recall']:.4f}", flush=True)

    out = (args.out_project or PROJECT) / "models" / args.name
    QuietCascade(
        labels=labels, gate_weights=model.gate_weights,
        gate_intercept=model.gate_intercept, gate_threshold=model.gate_threshold,
        tag_weights=model.tag_weights, tag_thresholds=thr.astype(np.float32),
        score_mode=model.score_mode, window=model.window,
        max_tokens=model.max_tokens, max_features=model.max_features,
        n_features=model.n_features,
    ).save(out, metadata={
        "derived_from": f"{SCORECARD}/models/{args.source}",
        "change": f"thresholds re-selected into the box P>={args.precision}, "
                  f"R>={args.recall}; weights and gate untouched",
        "p_target": args.precision, "r_target": args.recall, "beta": args.beta,
        "selection_data": "training calibration carve only",
        "verdicts": dict(counts)})
    probe = (args.out_project or PROJECT) / "probe"
    probe.mkdir(parents=True, exist_ok=True)
    (probe / f"{args.name}_selection.json").write_text(
        json.dumps({"p_target": args.precision, "r_target": args.recall, "beta": args.beta,
                    "summary": dict(counts), "per_tag": report}, indent=1),
        encoding="utf-8")
    print(f"\n-> {out}  ({int(np.isfinite(thr).sum())} enabled tags)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
