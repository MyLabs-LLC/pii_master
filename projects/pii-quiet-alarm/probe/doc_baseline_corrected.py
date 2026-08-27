"""The document-level baseline, scored against gold that actually answers the question.

The prior run took its dual-judge document-level gold from the manifest's
``label`` field. That field is the judges' *document-type variant* verdict, not
a PII-presence verdict: on the govdocs2 evaluation set 1,501 ``label=positive``
documents carry no PII entity at all and 1,220 ``label=negative`` documents do.
Cross-tabulated, ``label`` and PII presence are orthogonal.

The fields that do answer it are ``pii_entities`` / ``pii_classes`` /
``pii_sensitivity``, which are mutually consistent on every row of all four
dual-judge directories: empty entities and empty classes occur if and only if
``pii_sensitivity == "none"``. That is a judge assertion of absence, so those
documents are genuine negatives.

This re-scores the frozen champion's own recorded predictions -- no inference,
no training -- against that corrected gold, so the new run starts from a
baseline that measures the thing the run is about.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/lence/workspace/pii_master")
sys.path.insert(0, str(REPO))
from training.priority_data import iter_corpus  # noqa: E402

EVAL_ROOT = Path("/home/lence/workspace/data/2-eval")
PREDS = REPO / "projects/pii-priority-recall-v1/evaluations/champion_1k/predictions.jsonl"
JUDGE_ASSERTED = ("4000_datax-dualjudge-evalset-1.32k", "6589_govdocs2-dualjudge-eval20-3.53k")
COMPLETE = ("pii2_eval_30k",)
N_RESAMPLES = 1_000


def bootstrap_ci(flags: np.ndarray, seed: int) -> tuple[float | None, float | None]:
    if flags.size == 0:
        return None, None
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, flags.size, size=(N_RESAMPLES, flags.size))
    means = flags[draws].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def judge_gold(dataset: str) -> dict[str, bool]:
    """doc_id -> contains PII, from the judges' entity/class/sensitivity fields."""
    rows = json.loads((EVAL_ROOT / dataset / "manifest.json").read_text(encoding="utf-8"))
    gold = {}
    for r in rows:
        ents, cls = r.get("pii_entities") or [], r.get("pii_classes") or []
        sens = r.get("pii_sensitivity")
        empty = not ents and not cls
        if empty and sens not in (None, "none"):
            continue                       # inconsistent row: not usable as gold
        if not empty and sens == "none":
            continue
        gold[str(r.get("doc_id"))] = not empty
    return gold


def main() -> int:
    preds: dict[tuple[str, str], bool] = {}
    with PREDS.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            preds[(r["dataset"], r["uid"])] = bool(r["labels"])

    out = {"task": "document-level: does this document contain sensitive PII",
           "gold_correction": "pii_entities/pii_classes/pii_sensitivity, not manifest 'label'",
           "arm": "champion_1k (pii-priority-fusion-1k-v1)", "per_corpus": {}}

    for dataset in (*JUDGE_ASSERTED, *COMPLETE):
        if dataset in JUDGE_ASSERTED:
            gold = judge_gold(dataset)
            pairs = [(gold[u], preds[(dataset, u)])
                     for u in gold if (dataset, u) in preds]
        else:
            pairs = [(bool(row.labels), preds[(dataset, row.uid)])
                     for row in iter_corpus(EVAL_ROOT / dataset)
                     if (dataset, row.uid) in preds]
        g = np.array([p[0] for p in pairs]); f = np.array([p[1] for p in pairs])
        tp = int((g & f).sum()); fp = int((~g & f).sum())
        fn = int((g & ~f).sum()); tn = int((~g & ~f).sum())
        spec_flags = (~f[~g]).astype(float)
        lo, hi = bootstrap_ci(spec_flags, 4242)
        out["per_corpus"][dataset] = {
            "n_scored": len(pairs), "prevalence": float(g.mean()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "recall": tp / (tp + fn) if tp + fn else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "specificity": tn / (tn + fp) if tn + fp else None,
            "specificity_ci_low": lo, "specificity_ci_high": hi,
            "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        }

    vals = out["per_corpus"].values()
    out["aggregate"] = {
        "equal_corpus_specificity": float(np.mean([v["specificity"] for v in vals])),
        "equal_corpus_precision": float(np.mean([v["precision"] for v in vals])),
        "equal_corpus_recall": float(np.mean([v["recall"] for v in vals])),
        "total_negatives_available": sum(v["tn"] + v["fp"] for v in vals),
    }
    p = REPO / "projects/pii-quiet-alarm/probe/doc_baseline_corrected.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")

    print(f"{'corpus':<42}{'n':>7}{'prev':>7}{'recall':>8}{'precis':>8}{'specif':>8}{'FPR':>8}")
    for k, v in out["per_corpus"].items():
        print(f"{k:<42}{v['n_scored']:>7}{v['prevalence']:>7.3f}{v['recall']:>8.4f}"
              f"{v['precision']:>8.4f}{v['specificity']:>8.4f}{v['false_positive_rate']:>8.4f}")
    a = out["aggregate"]
    print(f"\nequal-corpus specificity {a['equal_corpus_specificity']:.4f} | "
          f"precision {a['equal_corpus_precision']:.4f} | recall {a['equal_corpus_recall']:.4f}")
    print(f"negatives available for measurement: {a['total_negatives_available']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
