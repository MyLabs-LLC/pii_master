"""Two independent measurements of how much the corpora agree about labels.

Part A -- **inter-judge agreement**, the hard ceiling. The datax corpus was
labelled twice, independently, by two different judge models over the same
4,708 documents (`manifest_gemini_5k_v4.jsonl` / `manifest_sonnet_5k_v4.jsonl`).
Where two gold sources cover the same items, their agreement bounds any score
measured against either one: a model cannot be *measured* above the reliability
of the thing measuring it. Per tag this reports raw agreement, Cohen's kappa,
Jaccard, and the F1 one judge scores when the other is treated as gold -- that
last number is the ceiling in the units the project actually reports.

Part B -- **cross-corpus definitional agreement**. The eight corpora share no
documents, so inter-annotator agreement is undefined across them. What *is*
measurable: hold the model fixed and ask what each corpus demands of it. For
one tag, the score threshold needed to reach a fixed recall should be similar
across corpora if they mean the same thing by that tag. A wide spread means
they do not, and the fixed evaluator will then read the difference as model
error. Prevalence per tag per corpus is reported beside it.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

DATAX = Path("/home/lence/workspace/data/datax/data")
JUDGE_A = "manifest_gemini_5k_v4.jsonl"
JUDGE_B = "manifest_sonnet_5k_v4.jsonl"

#: datax judge vocabulary -> the pipeline's priority tag names. Only the tags
#: that actually collide with the 16 gated ones matter here; the rest are
#: reported under their own names.
TO_PRIORITY = {
    "full_name": "sensitive_pii_full_name",
    "address": "sensitive_pii_address",
    "street_number_and_name": "sensitive_pii_address",
    "password": "sensitive_pii_password",
    "vehicle_identification_number_vin": "sensitive_pii_vehicle_id",
    "social_security_number": "sensitive_pii_social_security_number",
    "passport_number": "sensitive_pii_passport_number",
    "driver_s_license_number": "sensitive_pii_driver_s_license_number",
    "bank_account_number": "sensitive_pci_bank_account_number",
    "credit_card_number": "sensitive_pci_credit_card_number",
    "iban": "sensitive_pci_iban",
    "medical_record_number_mrn": "sensitive_phi_medical_record_number_mrn",
}


def _key(row: dict) -> str | None:
    return row.get("file", {}).get("sha256") or row.get("source", {}).get("reference")


def load_judge(path: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = _key(row)
            if key:
                out[key] = set(row.get("pii", {}).get("labels") or [])
    return out


def cohen_kappa(both: int, only_a: int, only_b: int, neither: int) -> float | None:
    n = both + only_a + only_b + neither
    if not n:
        return None
    observed = (both + neither) / n
    pa = ((both + only_a) / n) * ((both + only_b) / n)
    pn = ((only_b + neither) / n) * ((only_a + neither) / n)
    expected = pa + pn
    if expected >= 1.0:
        return None
    return (observed - expected) / (1.0 - expected)


def part_a(min_support: int) -> dict[str, Any]:
    a = load_judge(DATAX / JUDGE_A)
    b = load_judge(DATAX / JUDGE_B)
    shared = sorted(set(a) & set(b))
    labels = sorted({t for k in shared for t in (a[k] | b[k])})

    rows = []
    for tag in labels:
        both = sum(1 for k in shared if tag in a[k] and tag in b[k])
        only_a = sum(1 for k in shared if tag in a[k] and tag not in b[k])
        only_b = sum(1 for k in shared if tag not in a[k] and tag in b[k])
        neither = len(shared) - both - only_a - only_b
        either = both + only_a + only_b
        if either < min_support:
            continue
        rows.append({
            "tag": tag,
            "priority_tag": TO_PRIORITY.get(tag),
            "judge_a_positive": both + only_a,
            "judge_b_positive": both + only_b,
            "both": both,
            "either": either,
            # F1 of one judge against the other == the ceiling in report units.
            "f1_ceiling": (2 * both / (2 * both + only_a + only_b)) if both else 0.0,
            "jaccard": both / either if either else 0.0,
            "raw_agreement": (both + neither) / len(shared),
            "cohen_kappa": cohen_kappa(both, only_a, only_b, neither),
        })
    rows.sort(key=lambda r: r["f1_ceiling"])

    weighted = (
        sum(r["f1_ceiling"] * r["either"] for r in rows) / sum(r["either"] for r in rows)
        if rows else None
    )
    # Document-level: does a document carry any PII at all?
    doc_a = [bool(a[k]) for k in shared]
    doc_b = [bool(b[k]) for k in shared]
    doc_raw = sum(x == y for x, y in zip(doc_a, doc_b)) / len(shared)
    return {
        "documents_compared": len(shared),
        "tags_reported": len(rows),
        "min_support": min_support,
        "macro_f1_ceiling": float(np.mean([r["f1_ceiling"] for r in rows])) if rows else None,
        "support_weighted_f1_ceiling": weighted,
        "document_level_has_pii_agreement": doc_raw,
        "per_tag": rows,
    }


def part_b(project: Path, cache: Path, recall_target: float) -> dict[str, Any]:
    from training.priority_data import PRIORITY_TAGS

    stored = np.load(cache, allow_pickle=True)
    rows = json.loads(str(stored["rows"]))
    matrix = stored["matrix"]
    ds_of = np.asarray([r["dataset"] for r in rows])
    gold = [set(r.get("labels") or []) for r in rows]

    model_json = json.loads(
        (project / "models" / "champion_1k" / "component_recall" / "model.json").read_text()
    )
    labels = list(model_json["labels"])
    index = {lab: i for i, lab in enumerate(labels)}
    corpora = sorted(set(ds_of))

    out = []
    for tag in PRIORITY_TAGS:
        if tag not in index:
            continue
        col = index[tag]
        positive = np.asarray([tag in g for g in gold])
        per_corpus = {}
        for d in corpora:
            mask = positive & (ds_of == d)
            n = int(mask.sum())
            if n < 30:
                continue
            per_corpus[d] = {
                "support": n,
                "prevalence": n / int((ds_of == d).sum()),
                # Threshold this corpus demands to reach the target recall.
                "threshold_at_recall": float(
                    np.quantile(matrix[mask, col], 1.0 - recall_target, method="lower")
                ),
            }
        if len(per_corpus) < 2:
            continue
        thresholds = [v["threshold_at_recall"] for v in per_corpus.values()]
        prevalences = [v["prevalence"] for v in per_corpus.values()]
        lo, hi = min(thresholds), max(thresholds)
        out.append({
            "tag": tag,
            "corpora": len(per_corpus),
            "threshold_min": lo,
            "threshold_max": hi,
            "threshold_spread_ratio": (hi / lo) if lo > 0 else None,
            "prevalence_min": min(prevalences),
            "prevalence_max": max(prevalences),
            "prevalence_spread_ratio": (max(prevalences) / min(prevalences)) if min(prevalences) > 0 else None,
            "per_corpus": per_corpus,
        })
    out.sort(key=lambda r: -(r["threshold_spread_ratio"] or 0))
    return {"recall_target": recall_target, "per_tag": out}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=Path, default=Path("projects/pii-priority-recall-v1"))
    ap.add_argument("--cache", type=Path,
                    default=Path("projects/pii-priority-recall-v1/cache/champion_1k_recall_allcorpora.npz"))
    ap.add_argument("--min-support", type=int, default=30)
    ap.add_argument("--recall-target", type=float, default=0.90)
    ap.add_argument("--out", type=Path,
                    default=Path("projects/pii-priority-recall-v1/label_agreement.json"))
    args = ap.parse_args()

    a = part_a(args.min_support)
    b = part_b(args.project, args.cache, args.recall_target)
    payload = {"inter_judge": a, "cross_corpus": b}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"\n=== A. inter-judge agreement (datax, {a['documents_compared']} shared documents) ===")
    print(f"document-level 'has any PII' agreement : {a['document_level_has_pii_agreement']:.4f}")
    print(f"macro F1 ceiling over {a['tags_reported']} tags          : {a['macro_f1_ceiling']:.4f}")
    print(f"support-weighted F1 ceiling            : {a['support_weighted_f1_ceiling']:.4f}")
    print(f"\n{'tag':40} {'A+':>6} {'B+':>6} {'both':>6} {'F1ceil':>7} {'kappa':>7}")
    for r in a["per_tag"]:
        star = " *" if r["priority_tag"] else ""
        k = r["cohen_kappa"]
        print(f"{r['tag'][:40]:40} {r['judge_a_positive']:>6} {r['judge_b_positive']:>6} "
              f"{r['both']:>6} {r['f1_ceiling']:>7.4f} {(f'{k:.4f}' if k is not None else '   n/a'):>7}{star}")

    print(f"\n=== B. cross-corpus threshold demand at recall {b['recall_target']} ===")
    print(f"{'tag':52} {'corpora':>7} {'thr_min':>9} {'thr_max':>9} {'spread':>8} {'prev_spread':>12}")
    for r in b["per_tag"]:
        sp = r["threshold_spread_ratio"]
        pv = r["prevalence_spread_ratio"]
        print(f"{r['tag']:52} {r['corpora']:>7} {r['threshold_min']:>9.4f} {r['threshold_max']:>9.4f} "
              f"{(f'{sp:.1f}x' if sp else 'n/a'):>8} {(f'{pv:.1f}x' if pv else 'n/a'):>12}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
