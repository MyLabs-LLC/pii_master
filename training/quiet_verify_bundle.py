"""Re-score a sealed corpus **through the bundle's own entry point**.

This is the step being verified, not the model. The bundle ships its own
extractor, its own thresholds and its own loader, and each of those is a place
the delivered artifact can quietly diverge from the one that was measured. So
this imports `tagger.py` from the bundle -- with the training tree deliberately
*not* on the path first -- reads real documents off disk, and recomputes the
metric from the bundle's predictions.

Off by more than the tolerance and the bundle is not sealed. Find the
difference; do not widen the tolerance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/lence/workspace/pii_master")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--corpus", required=True, help="canonical corpus name")
    ap.add_argument("--expected", type=float, required=True)
    ap.add_argument("--metric", default="priority_macro_f05")
    ap.add_argument("--tolerance", type=float, default=0.01)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    # Gold and document paths come from the project; predictions come from the
    # bundle. Import the project pieces first, then put the bundle ahead of them
    # so `tagger` resolves to the shipped one.
    sys.path.insert(0, str(REPO))
    from training.quiet_data import (
        PRIORITY_TAGS, collapse_tags, iter_quiet_corpus, read_document, resolve_dataset,
    )
    from training.quiet_select import fbeta

    sys.path.insert(0, str(args.bundle.resolve()))
    import tagger as bundle_tagger

    t = bundle_tagger.Tagger()
    print(f"bundle loaded: {t.model.config}", file=sys.stderr)

    rows = list(iter_quiet_corpus(resolve_dataset(args.corpus)))
    labels = list(t.labels)
    index = {lab: i for i, lab in enumerate(labels)}
    n = len(rows)
    pred = np.zeros((n, len(labels)), dtype=bool)
    gold = np.zeros((n, len(labels)), dtype=bool)
    complete = np.zeros(n, dtype=bool)
    read_errors = 0
    for i, qr in enumerate(rows):
        try:
            text = read_document(Path(qr.path), limit=t.read_window_chars * 2)
        except (FileNotFoundError, OSError):
            read_errors += 1
            text = ""
        for lab in t.predict(text):
            pred[i, index[lab]] = True
        for lab in collapse_tags(qr.labels):
            if lab in index:
                gold[i, index[lab]] = True
        complete[i] = qr.tag_labels_complete
        if (i + 1) % 5_000 == 0:
            print(f"  {i + 1}/{n}", file=sys.stderr, flush=True)

    f05s = []
    per_tag = {}
    for tag in PRIORITY_TAGS:
        j = index.get(tag)
        if j is None:
            continue
        pos = gold[:, j]
        if int(pos.sum()) < 30:
            continue
        eligible = pos | (complete & ~pos)
        p = pred[:, j] & eligible
        tp = int((p & pos).sum())
        fp = int((p & ~pos).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / int(pos.sum())
        f = float(fbeta(np.asarray([precision]), np.asarray([recall]), 0.5)[0])
        per_tag[tag] = {"precision": precision, "recall": recall, "f05": f,
                        "support": int(pos.sum())}
        f05s.append(f)

    measured = float(np.mean(f05s)) if f05s else 0.0
    delta = abs(measured - args.expected)
    ok = delta <= args.tolerance
    payload = {
        "checked": f"{args.corpus} through the packaged tagger.py",
        "n": n, "read_errors": read_errors,
        "metric": args.metric, "expected": args.expected, "measured": measured,
        "delta": delta, "tolerance": args.tolerance, "passed": ok,
        "n_priority_tags_measured": len(f05s), "per_tag": per_tag,
        "bundle_config": t.model.config,
    }
    args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"{args.metric}: expected {args.expected:.4f}, measured {measured:.4f}, "
          f"delta {delta:.5f} (tolerance {args.tolerance}) -> "
          f"{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
