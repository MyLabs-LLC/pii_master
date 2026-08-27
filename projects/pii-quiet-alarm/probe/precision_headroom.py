"""Feasibility probe: how much precision is reachable, and what does it cost?

Cheap, training-free. It reuses the frozen `pii-priority-fusion-1k-v1` champion's
recall-max component -- the one every priority tag is currently locked to -- and
sweeps its per-label decision thresholds. That traces the operating curve the
current artifact already sits on, at its far recall end.

Two questions the approval gate needs answered before a 1,000-trial budget:

1. **Document level** -- "does this document contain PII at all". At what
   specificity (share of genuinely clean documents we correctly stay silent on)
   does priority recall start to break? Measured only where true negatives
   exist.
2. **Tag level** -- what does per-tag precision on the 16 priority tags reach as
   thresholds rise, and what recall survives?

Nothing here is a ship claim; it bounds the bet.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/lence/workspace/pii_master")
sys.path.insert(0, str(REPO))

from training.priority_data import PRIORITY_TAGS, iter_corpus, read_document  # noqa: E402
from training.priority_hash import HashCueModel, document_features  # noqa: E402

BUNDLE = REPO / "projects/pii-priority-recall-v1/dist/pii-priority-fusion-1k-v1"
COMPONENT = BUNDLE / "models/model/component_recall"
EVAL_ROOT = Path("/home/lence/workspace/data/2-eval")
OUT = REPO / "projects/pii-quiet-alarm/probe"

# Corpora carrying genuine document-level negatives. Everywhere else in 2-eval is
# either prevalence 1.0 (no negatives to be wrong about) or positive-only gold.
DOC_NEG = {
    "pii2_eval_30k": "complete",          # 4,851 real negatives, complete gold
    "4000_datax-dualjudge-evalset-1.32k": "dual_judge",
    "6589_govdocs2-dualjudge-eval20-3.53k": "dual_judge",
}
# Label-complete corpora used for the tag-level precision curve.
TAG_CORPORA = ("pii2_eval_30k", "pii_holdout_20k")

MIN_SUPPORT = 30
SCALES = np.concatenate([np.linspace(1.0, 4.0, 13), np.geomspace(4.5, 400.0, 28)])


def judge_gold(dataset: str) -> dict[str, str]:
    """positive / negative / disputed, from the dual-judge manifest."""
    manifest = json.loads((EVAL_ROOT / dataset / "manifest.json").read_text(encoding="utf-8"))
    return {row["doc_id"]: row.get("label") for row in manifest}


def score_corpus(model: HashCueModel, dataset: str) -> tuple[np.ndarray, list[tuple[str, ...]], list[str]]:
    scores, gold, uids = [], [], []
    read_errors = 0
    for row in iter_corpus(EVAL_ROOT / dataset):
        try:
            text = read_document(Path(row.path), limit=model.read_window_chars)
        except (FileNotFoundError, OSError):
            # The prior run's evaluator reports these as `read_errors`; a document
            # that cannot be read is excluded, never scored as an empty string.
            read_errors += 1
            continue
        feats = document_features(
            text[: model.read_window_chars],
            n_features=model.n_features,
            max_tokens=model.max_tokens,
            max_features=model.max_document_features,
        )
        scores.append(model.predict_scores_from_features(feats))
        gold.append(row.labels)
        uids.append(row.uid)
    if read_errors:
        print(f'  {dataset}: {read_errors} unreadable document(s) excluded', file=sys.stderr)
    return np.asarray(scores, dtype=np.float32), gold, uids


def main() -> int:
    model = HashCueModel.load(COMPONENT)
    labels = list(model.labels)
    pri_idx = [labels.index(t) for t in PRIORITY_TAGS if t in labels]
    pri_names = [t for t in PRIORITY_TAGS if t in labels]
    base_thresholds = np.asarray(model.thresholds, dtype=np.float32)

    cache: dict[str, tuple[np.ndarray, list[tuple[str, ...]], list[str]]] = {}
    for dataset in sorted(set(DOC_NEG) | set(TAG_CORPORA)):
        print(f"scoring {dataset} ...", file=sys.stderr, flush=True)
        cache[dataset] = score_corpus(model, dataset)

    report: dict = {
        "component": "pii-priority-fusion-1k-v1 / component_recall (all 16 priority tags)",
        "read_window_chars": int(model.read_window_chars),
        "note": "threshold scale 1.0 == the shipped champion's own operating point",
        "doc_level": {},
        "tag_level": {},
    }

    # ---- document level -------------------------------------------------
    for dataset, mode in DOC_NEG.items():
        scores, gold, uids = cache[dataset]
        if mode == "complete":
            keep = np.ones(len(gold), dtype=bool)
            pos = np.array([bool(g) for g in gold])
        else:
            jg = judge_gold(dataset)
            lab = [jg.get(u) for u in uids]
            keep = np.array([x in ("positive", "negative") for x in lab])
            pos = np.array([x == "positive" for x in lab])
        curve = []
        for scale in SCALES:
            fired = (scores >= base_thresholds * scale).any(axis=1)
            f, p = fired[keep], pos[keep]
            tp = int((f & p).sum()); fp = int((f & ~p).sum())
            fn = int((~f & p).sum()); tn = int((~f & ~p).sum())
            curve.append({
                "scale": round(float(scale), 4),
                "recall": tp / (tp + fn) if tp + fn else None,
                "precision": tp / (tp + fp) if tp + fp else None,
                "specificity": tn / (tn + fp) if tn + fp else None,
                "flag_rate": float(f.mean()),
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            })
        report["doc_level"][dataset] = {
            "gold_mode": mode, "n_scored": int(keep.sum()),
            "prevalence": float(pos[keep].mean()), "curve": curve,
        }

    # ---- tag level (priority tags, label-complete corpora) ---------------
    for dataset in TAG_CORPORA:
        scores, gold, _ = cache[dataset]
        truth = np.zeros((len(gold), len(pri_idx)), dtype=bool)
        for i, tags in enumerate(gold):
            s = set(tags)
            for j, name in enumerate(pri_names):
                truth[i, j] = name in s
        support = truth.sum(axis=0)
        measurable = support >= MIN_SUPPORT
        curve = []
        for scale in SCALES:
            pred = scores[:, pri_idx] >= (base_thresholds[pri_idx] * scale)
            tp = (pred & truth).sum(axis=0)
            fp = (pred & ~truth).sum(axis=0)
            fn = (~pred & truth).sum(axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
                rec = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
                f05 = np.where(0.25 * prec + rec > 0,
                               1.25 * prec * rec / np.maximum(0.25 * prec + rec, 1e-12), 0.0)
            m = measurable
            curve.append({
                "scale": round(float(scale), 4),
                "macro_precision": float(prec[m].mean()),
                "macro_recall": float(rec[m].mean()),
                "macro_f05": float(f05[m].mean()),
                "min_recall": float(rec[m].min()),
                "n_measurable_tags": int(m.sum()),
            })
        report["tag_level"][dataset] = {
            "n_rows": len(gold),
            "measurable_tags": [n for n, ok in zip(pri_names, measurable) if ok],
            "curve": curve,
        }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "precision_headroom.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: len(v) for k, v in report["doc_level"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
