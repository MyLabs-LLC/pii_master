"""Document-level PII detection: does this document contain any PII at all?

The project's headline metric is a 61-way multi-label tagging score, which the
label-agreement probe showed the gold can only measure to about F1 0.51 -- and
part of even that is reproducing one corpus's naming convention. The *document*
question is a different task and a far better-measured one: two independent
judges over the same 4,708 documents agree on "does this document contain any
PII" far more than they agree on which tags it carries.

This scores that task under the same discipline the tagging evaluator uses:

* **Gold** is "the corpus lists at least one sensitive tag for this document".
* **Predicted** is "the model emits at least one tag **within the corpus's
  frozen catalogue**". Restricting to the catalogue matters -- a corpus's gold
  only covers its own catalogue, so crediting or penalising a prediction
  outside it compares the model against labels that were never collected.
* **A corpus reports only what its gold can measure.** On ``label_complete``
  corpora every metric is available. On positive-only corpora a document
  without a listed tag is *unknown*, not negative, so precision, F1, accuracy
  and specificity are **not measurable** -- they are reported as ``null``,
  never as zero, and recall is reported alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

N_RESAMPLES = 1_000
CONFIDENCE = 0.95


def _fbeta(precision: float | None, recall: float | None, beta: float) -> float | None:
    if precision is None or recall is None:
        return None
    b2 = beta * beta
    denom = (b2 * precision) + recall
    return ((1 + b2) * precision * recall / denom) if denom else 0.0


def _ratio(n: int, d: int) -> float | None:
    return n / d if d else None


def _bootstrap_ci(flags: np.ndarray, seed: int) -> tuple[float | None, float | None]:
    """Document bootstrap over a per-document 0/1 outcome."""
    if flags.size == 0:
        return None, None
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, flags.size, size=(N_RESAMPLES, flags.size))
    means = flags[draws].mean(axis=1)
    alpha = (1.0 - CONFIDENCE) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def evaluate_corpus(rows: list[dict], *, catalogue: set[str], complete: bool,
                    seed: int) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    hit_flags: list[int] = []          # per positive document: did we flag it
    for r in rows:
        gold = bool(r["gold"] & catalogue) if catalogue else bool(r["gold"])
        pred = bool(r["pred"] & catalogue) if catalogue else bool(r["pred"])
        if gold:
            hit_flags.append(1 if pred else 0)
            tp += pred
            fn += not pred
        else:
            fp += pred
            tn += not pred

    recall = _ratio(tp, tp + fn)
    lo, hi = _bootstrap_ci(np.asarray(hit_flags, dtype=float), seed)
    out: dict[str, Any] = {
        "n_rows": len(rows),
        "gold_positive": tp + fn,
        "predicted_positive": tp + fp,
        "tp": tp, "fn": fn,
        "recall": recall,
        "recall_ci_low": lo, "recall_ci_high": hi,
        "label_complete": complete,
    }
    if complete:
        precision = _ratio(tp, tp + fp)
        out.update({
            "fp": fp, "tn": tn,
            "precision": precision,
            "f1": _fbeta(precision, recall, 1.0),
            "f2": _fbeta(precision, recall, 2.0),
            "accuracy": _ratio(tp + tn, len(rows)),
            "specificity": _ratio(tn, tn + fp),
            "prevalence": _ratio(tp + fn, len(rows)),
        })
    else:
        # Not measurable is not zero, and it is not a pass either.
        out.update({
            "fp": None, "tn": None, "precision": None, "f1": None, "f2": None,
            "accuracy": None, "specificity": None,
            "prevalence": None,
            "not_measurable": ["precision", "f1", "f2", "accuracy", "specificity"],
            "reason": "positive-only gold: an unlisted tag is unknown, not absent",
        })
    return out


#: The two corpora that carry a real document-level verdict *with negatives*.
#: Their manifests record ``label`` as positive / negative / disputed from the
#: dual-judge pass. Everywhere else in ``2-eval`` is either positive-only gold
#: or ~100% PII prevalence, where "does this document contain PII" is not a
#: question the corpus can answer: a constant "yes" scores ~0.98.
DUAL_JUDGE = {
    "4000_datax-dualjudge-evalset-1.32k": "/home/lence/workspace/data/2-eval/4000_datax-dualjudge-evalset-1.32k",
    "6589_govdocs2-dualjudge-eval20-3.53k": "/home/lence/workspace/data/2-eval/6589_govdocs2-dualjudge-eval20-3.53k",
}


def run_dual_judge(project: Path, families: list[str]) -> dict[str, Any]:
    """Document-level detection where genuine negatives exist.

    Predictions are **not** restricted to the tagging catalogue here. The judge
    was asked whether the document contains any of ~60 sensitive types, so the
    matching question of the model is whether it emits any tag at all.
    ``disputed`` documents -- the ones the two judges could not agree on -- are
    excluded from the headline and reported as their own slice, because gold two
    annotators disagreed about is not gold.
    """
    index_uid = {}
    for line in (project / "data" / "eval_index.jsonl").read_text(
        encoding="utf-8"
    ).splitlines():
        r = json.loads(line)
        if r["dataset"] in DUAL_JUDGE:
            index_uid[(r["dataset"], r["uid"])] = True

    gold: dict[tuple[str, str], str] = {}
    for ds, root in DUAL_JUDGE.items():
        for row in json.loads((Path(root) / "manifest.json").read_text(encoding="utf-8")):
            gold[(ds, row["doc_id"])] = row.get("label")

    out: dict[str, Any] = {}
    for family in families:
        preds: dict[tuple[str, str], bool] = {}
        with (project / "evaluations" / family / "predictions.jsonl").open(
            encoding="utf-8"
        ) as stream:
            for line in stream:
                r = json.loads(line)
                if r["dataset"] in DUAL_JUDGE:
                    preds[(r["dataset"], r["uid"])] = bool(r["labels"])

        per_corpus = {}
        for offset, ds in enumerate(sorted(DUAL_JUDGE)):
            buckets: dict[str, dict[str, int]] = {
                k: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for k in ("headline", "disputed")
            }
            hits: list[int] = []
            for (d, uid), label in gold.items():
                if d != ds or (d, uid) not in index_uid:
                    continue
                pred = preds.get((d, uid), False)
                if label == "disputed":
                    # judged positive by one judge only -- treat as positive for
                    # a recall-style read, and never mix it into the headline
                    b = buckets["disputed"]
                    b["tp" if pred else "fn"] += 1
                    continue
                positive = label == "positive"
                b = buckets["headline"]
                if positive:
                    hits.append(1 if pred else 0)
                    b["tp" if pred else "fn"] += 1
                else:
                    b["fp" if pred else "tn"] += 1

            h = buckets["headline"]
            precision = _ratio(h["tp"], h["tp"] + h["fp"])
            recall = _ratio(h["tp"], h["tp"] + h["fn"])
            lo, hi = _bootstrap_ci(np.asarray(hits, dtype=float), 61_803 + offset)
            d_ = buckets["disputed"]
            per_corpus[ds] = {
                **h,
                "n_scored": sum(h.values()),
                "prevalence": _ratio(h["tp"] + h["fn"], sum(h.values())),
                "precision": precision,
                "recall": recall,
                "recall_ci_low": lo, "recall_ci_high": hi,
                "f1": _fbeta(precision, recall, 1.0),
                "f2": _fbeta(precision, recall, 2.0),
                "accuracy": _ratio(h["tp"] + h["tn"], sum(h.values())),
                "specificity": _ratio(h["tn"], h["tn"] + h["fp"]),
                "disputed_n": d_["tp"] + d_["fn"],
                "disputed_flag_rate": _ratio(d_["tp"], d_["tp"] + d_["fn"]),
            }
        measurable = [v for v in per_corpus.values() if v["f1"] is not None]
        out[family] = {
            "per_corpus": per_corpus,
            "aggregate": {
                "equal_corpus_f1": float(np.mean([v["f1"] for v in measurable])),
                "equal_corpus_f2": float(np.mean([v["f2"] for v in measurable])),
                "equal_corpus_precision": float(np.mean([v["precision"] for v in measurable])),
                "equal_corpus_recall": float(np.mean([v["recall"] for v in measurable])),
                "equal_corpus_accuracy": float(np.mean([v["accuracy"] for v in measurable])),
                "equal_corpus_specificity": float(np.mean([v["specificity"] for v in measurable])),
            },
        }
    return out


def judge_ceiling() -> dict[str, Any]:
    """The same question asked of two independent judges over the same documents."""
    from training.label_agreement_probe import (
        DATAX, JUDGE_A, JUDGE_B, cohen_kappa, load_judge,
    )

    a = load_judge(DATAX / JUDGE_A)
    b = load_judge(DATAX / JUDGE_B)
    shared = sorted(set(a) & set(b))
    both = sum(1 for k in shared if a[k] and b[k])
    only_a = sum(1 for k in shared if a[k] and not b[k])
    only_b = sum(1 for k in shared if not a[k] and b[k])
    neither = len(shared) - both - only_a - only_b
    f1 = 2 * both / (2 * both + only_a + only_b) if both else 0.0
    return {
        "documents": len(shared),
        "judge_a_positive": both + only_a,
        "judge_b_positive": both + only_b,
        "both": both,
        "raw_agreement": (both + neither) / len(shared),
        "f1": f1,
        "cohen_kappa": cohen_kappa(both, only_a, only_b, neither),
        "source": "datax manifest_gemini_5k_v4 vs manifest_sonnet_5k_v4",
    }


def run(project: Path, families: list[str]) -> dict[str, Any]:
    index = [
        json.loads(line)
        for line in (project / "data" / "eval_index.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    frozen = json.loads(
        (project / "data" / "evaluation_catalogue.json").read_text(encoding="utf-8")
    )
    catalogues = {
        ds: set(v["catalogue"]) for ds, v in frozen["corpora"].items()
    }

    report: dict[str, Any] = {
        "task": "document-level PII detection (any sensitive tag present)",
        "judge_ceiling": judge_ceiling(),
        "families": {},
    }
    for family in families:
        preds: dict[tuple[str, str], set[str]] = {}
        pred_path = project / "evaluations" / family / "predictions.jsonl"
        with pred_path.open(encoding="utf-8") as stream:
            for line in stream:
                r = json.loads(line)
                preds[(r["dataset"], r["uid"])] = set(r["labels"])

        by_ds: dict[str, list[dict]] = {}
        for r in index:
            ds = r["dataset"]
            by_ds.setdefault(ds, []).append({
                "gold": set(r.get("labels") or []),
                "pred": preds.get((ds, r["uid"]), set()),
                "complete": bool(r.get("label_complete")),
            })

        per_corpus = {}
        for offset, ds in enumerate(sorted(by_ds)):
            rows = by_ds[ds]
            cat = catalogues.get(ds, set())
            if not cat:
                continue                       # nemotron declares no catalogue
            complete = all(r["complete"] for r in rows)
            per_corpus[ds] = evaluate_corpus(
                rows, catalogue=cat, complete=complete, seed=31_337 + offset
            )

        measurable = [v for v in per_corpus.values() if v["label_complete"]]
        recall_only = [v for v in per_corpus.values() if not v["label_complete"]]
        agg = {
            "complete_corpora": len(measurable),
            "recall_only_corpora": len(recall_only),
            "equal_corpus_f1": float(np.mean([v["f1"] for v in measurable])) if measurable else None,
            "equal_corpus_f2": float(np.mean([v["f2"] for v in measurable])) if measurable else None,
            "equal_corpus_precision": float(np.mean([v["precision"] for v in measurable])) if measurable else None,
            "equal_corpus_recall": float(np.mean([v["recall"] for v in measurable])) if measurable else None,
            "equal_corpus_accuracy": float(np.mean([v["accuracy"] for v in measurable])) if measurable else None,
            "recall_only_mean": float(np.mean([v["recall"] for v in recall_only])) if recall_only else None,
            "worst_corpus_f1": min((v["f1"] for v in measurable), default=None),
        }
        report["families"][family] = {"aggregate": agg, "per_corpus": per_corpus}
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=Path, default=Path("projects/pii-priority-recall-v1"))
    ap.add_argument("--families", nargs="+", default=[
        "current_rules", "hash_sgd", "hash_sgd_f2", "tfidf_linear",
        "embeddingbag_asl", "hybrid_priority_001", "hybrid_priority",
        "champion_1k", "perlabel_v4",
    ])
    ap.add_argument("--out", type=Path,
                    default=Path("projects/pii-priority-recall-v1/doc_level.json"))
    args = ap.parse_args()

    report = run(args.project, args.families)
    report["dual_judge"] = run_dual_judge(args.project, args.families)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("=== DUAL-JUDGE corpora: the only document-level gold with real negatives ===")
    print(f"{'family':20} {'F1':>7} {'F2':>7} {'P':>7} {'R':>7} {'spec':>7} {'acc':>7}")
    for family, body in report["dual_judge"].items():
        a = body["aggregate"]
        print(f"{family:20} {a['equal_corpus_f1']:>7.4f} {a['equal_corpus_f2']:>7.4f} "
              f"{a['equal_corpus_precision']:>7.4f} {a['equal_corpus_recall']:>7.4f} "
              f"{a['equal_corpus_specificity']:>7.4f} {a['equal_corpus_accuracy']:>7.4f}")
    print()
    for ds in sorted(DUAL_JUDGE):
        print(f"  {ds}")
        print(f"    {'family':20} {'n':>6} {'prev':>6} {'P':>7} {'R':>7} {'F1':>7} {'spec':>7} {'disputed flagged':>17}")
        for family, body in report["dual_judge"].items():
            v = body["per_corpus"][ds]
            print(f"    {family:20} {v['n_scored']:>6} {v['prevalence']:>6.3f} "
                  f"{v['precision']:>7.4f} {v['recall']:>7.4f} {v['f1']:>7.4f} "
                  f"{v['specificity']:>7.4f} {v['disputed_flag_rate']:>16.3f} ")
        print()

    c = report["judge_ceiling"]
    print(f"JUDGE CEILING on the same question ({c['documents']} documents, two independent judges)")
    print(f"  raw agreement {c['raw_agreement']:.4f}   F1 {c['f1']:.4f}   kappa {c['cohen_kappa']:.4f}\n")

    print(f"{'family':20} {'F1':>7} {'F2':>7} {'P':>7} {'R':>7} {'acc':>7} {'worstF1':>8} {'recall-only':>12}")
    for family, body in report["families"].items():
        a = body["aggregate"]
        ro = a["recall_only_mean"]
        print(f"{family:20} {a['equal_corpus_f1']:>7.4f} {a['equal_corpus_f2']:>7.4f} "
              f"{a['equal_corpus_precision']:>7.4f} {a['equal_corpus_recall']:>7.4f} "
              f"{a['equal_corpus_accuracy']:>7.4f} {a['worst_corpus_f1']:>8.4f} "
              f"{(f'{ro:.4f}' if ro is not None else 'n/a'):>12}")

    best = max(report["families"], key=lambda f: report["families"][f]["aggregate"]["equal_corpus_f1"])
    print(f"\nper-corpus detail for {best}:")
    print(f"  {'corpus':40} {'gold+':>7} {'P':>8} {'R':>8} {'F1':>8} {'prev':>7}")
    for ds, v in report["families"][best]["per_corpus"].items():
        f = lambda x: f"{x:.4f}" if isinstance(x, float) else "not-meas"
        print(f"  {ds[:40]:40} {v['gold_positive']:>7} {f(v['precision']):>8} "
              f"{f(v['recall']):>8} {f(v['f1']):>8} {f(v['prevalence']):>7}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
