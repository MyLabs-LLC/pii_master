"""Scoping probe: what would the taxonomy collapse actually be worth?

**This does not change the evaluator.** It re-scores predictions that already
exist under a collapsed *label mapping*, so the payoff can be estimated before
anyone commits to re-running every arm under a changed metric contract. Nothing
it writes is a gate result, and no model is selected from it.

The collapse folds three name tags into ``full_name`` and one street tag into
``address``, in **both** gold and predictions, then drops the four folded tags
from each corpus catalogue. Two effects are separated, because they pull in
opposite directions and only one of them is a real improvement:

* **Denominator** -- macro F2 averages over the frozen catalogue, so removing
  four tags changes the mean even if nothing else moves. Measured on
  ``perlabel_v4`` this is **-0.0093**: those tags score 0.70-0.93 on three of
  the five complete corpora, above the corpus mean, so the collapse starts at a
  deficit rather than a discount.
* **Merge** -- a document labelled ``given_name`` but not ``full_name`` today
  makes a ``full_name`` prediction a false positive. After the fold it is a true
  positive. This is the genuine gain the collapse is for.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from training.priority_eval import EvaluationRow, aggregate_arms, evaluate_corpus

#: folded tag -> the tag that absorbs it
COLLAPSE = {
    "sensitive_pii_given_name": "sensitive_pii_full_name",
    "sensitive_pii_family_name": "sensitive_pii_full_name",
    "sensitive_pii_middle_name": "sensitive_pii_full_name",
    "sensitive_pii_street_number_and_name": "sensitive_pii_address",
}


def fold(tags: set[str]) -> set[str]:
    return {COLLAPSE.get(t, t) for t in tags}


def fold_catalogue(catalogue: list[str]) -> list[str]:
    out: list[str] = []
    for tag in catalogue:
        target = COLLAPSE.get(tag, tag)
        if target not in out:
            out.append(target)
    return out


def load_predictions(path: Path) -> dict[tuple[str, str], set[str]]:
    out: dict[tuple[str, str], set[str]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            r = json.loads(line)
            out[(r["dataset"], r["uid"])] = set(r["labels"])
    return out


def score(family: str, project: Path, *, collapsed: bool, bootstrap: bool) -> dict[str, Any]:
    index = [
        json.loads(line)
        for line in (project / "data" / "eval_index.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    preds = load_predictions(project / "evaluations" / family / "predictions.jsonl")
    frozen = json.loads(
        (project / "data" / "evaluation_catalogue.json").read_text(encoding="utf-8")
    )

    grouped: dict[str, list[EvaluationRow]] = {}
    for row in index:
        ds, uid = row["dataset"], row["uid"]
        gold = set(row.get("labels") or [])
        predicted = preds.get((ds, uid), set())
        if collapsed:
            gold, predicted = fold(gold), fold(predicted)
        grouped.setdefault(ds, []).append(
            EvaluationRow(
                dataset=ds,
                uid=uid,
                gold=frozenset(gold),
                predicted=frozenset(predicted),
                label_complete=bool(row.get("label_complete")),
            )
        )

    arms = []
    for ds in sorted(grouped):
        catalogue = frozen["corpora"][ds]["catalogue"]
        if collapsed:
            catalogue = fold_catalogue(list(catalogue))
        arms.append(
            evaluate_corpus(grouped[ds], catalogue=catalogue, bootstrap=bootstrap)
        )
    agg = aggregate_arms(arms)
    measurable = [e for a in arms for e in a["priority"].values() if e["support"] >= 30]
    agg["priority_conclusive_passes"] = sum(e["status"] == "PASS" for e in measurable)
    agg["priority_inconclusive"] = sum(e["status"] == "INCONCLUSIVE" for e in measurable)
    agg["priority_failures_measurable"] = sum(e["status"] == "FAIL" for e in measurable)
    agg["measurable"] = len(measurable)
    return {"aggregate": agg, "arms": arms}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=Path, default=Path("projects/pii-priority-recall-v1"))
    ap.add_argument("--families", nargs="+", default=["champion_1k", "perlabel_v4"])
    ap.add_argument("--bootstrap", action="store_true")
    ap.add_argument("--out", type=Path,
                    default=Path("projects/pii-priority-recall-v1/taxonomy_collapse_scope.json"))
    args = ap.parse_args()

    report: dict[str, Any] = {"collapse": COLLAPSE, "families": {}}
    print(f"{'family':14} {'taxonomy':10} {'macroF2':>8} {'microF1':>8} "
          f"{'pass':>5} {'incon':>6} {'fail':>5} {'measurable':>11}")
    for family in args.families:
        entry = {}
        for label, collapsed in (("current", False), ("collapsed", True)):
            res = score(family, args.project, collapsed=collapsed, bootstrap=args.bootstrap)
            a = res["aggregate"]
            entry[label] = a
            print(f"{family:14} {label:10} {a['equal_corpus_macro_f2']:>8.4f} "
                  f"{a['equal_corpus_micro_f1']:>8.4f} {a['priority_conclusive_passes']:>5} "
                  f"{a['priority_inconclusive']:>6} {a['priority_failures_measurable']:>5} "
                  f"{a['measurable']:>11}")
        d_f2 = entry["collapsed"]["equal_corpus_macro_f2"] - entry["current"]["equal_corpus_macro_f2"]
        d_f1 = entry["collapsed"]["equal_corpus_micro_f1"] - entry["current"]["equal_corpus_micro_f1"]
        entry["delta"] = {"macro_f2": d_f2, "micro_f1": d_f1}
        print(f"{family:14} {'DELTA':10} {d_f2:>+8.4f} {d_f1:>+8.4f}")
        report["families"][family] = entry

    # Per-tag detail for the two tags that absorb the folds.
    watch = ["sensitive_pii_full_name", "sensitive_pii_address"]
    print(f"\n{'corpus':34} {'tag':12} {'sup':>6}->{'sup':>6} {'P':>6}->{'P':>6} {'R':>6}->{'R':>6}")
    cur = score(args.families[-1], args.project, collapsed=False, bootstrap=False)["arms"]
    col = score(args.families[-1], args.project, collapsed=True, bootstrap=False)["arms"]
    for a, b in zip(cur, col):
        for tag in watch:
            pa, pb = a["per_tag"].get(tag), b["per_tag"].get(tag)
            if not pa or not pb or not pa["support"]:
                continue
            f = lambda v: f"{v:.3f}" if isinstance(v, float) else "  n/a"
            print(f"{a['dataset'][:34]:34} {tag.replace('sensitive_pii_',''):12} "
                  f"{pa['support']:>6}->{pb['support']:>6} "
                  f"{f(pa['precision']):>6}->{f(pb['precision']):>6} "
                  f"{f(pa['recall']):>6}->{f(pb['recall']):>6}")

    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
