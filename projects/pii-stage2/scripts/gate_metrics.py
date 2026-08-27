"""Span-level gate metrics the engine's tagging family does not compute.

``evaluate("tagging")`` scores document tag *sets* at top-k. This product is a
span tagger, and the ship gate is per-tag recall on high-severity identifiers
plus macro F2 over the mapped catalogue (skill convention for PII/PHI). Both
are derived from the same exact-match ``TypeScore`` tables
``nemotron_eval.py`` / ``eval_student.py`` already produce — the match rule is
not reopened here.
"""

from __future__ import annotations

from statistics import median
from typing import Any, Mapping

# Ship-gate tags. CREDIT_CARD participates only on the Luhn-valid subset
# (see spec); the full Nemotron card gold is 88% unreachable by design.
HIGH_SEVERITY = (
    "SSN",
    "MRN",
    "HEALTH_PLAN_ID",
    "ACCOUNT_NUMBER",
    "US_DRIVER_LICENSE",
)
MIN_SUPPORT = 30


def f2(precision: float, recall: float) -> float:
    """F-beta at β=2: recall weighted 4× precision."""
    denom = (4.0 * precision) + recall
    return (5.0 * precision * recall / denom) if denom else 0.0


def _pr(row: Mapping[str, Any]) -> tuple[float, float, int]:
    tp = int(row.get("tp", 0))
    fp = int(row.get("fp", 0))
    fn = int(row.get("fn", 0))
    gold = int(row.get("gold", tp + fn))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return precision, recall, gold


def per_tag_table(exact: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for name, row in exact.items():
        precision, recall, gold = _pr(row)
        rows.append(
            {
                "tag": name,
                "gold": gold,
                "tp": int(row.get("tp", 0)),
                "fp": int(row.get("fp", 0)),
                "fn": int(row.get("fn", 0)),
                "precision": precision,
                "recall": recall,
                "f1": (2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
                "f2": f2(precision, recall),
            }
        )
    rows.sort(key=lambda r: (r["f2"], r["recall"], r["tag"]))
    return rows


def span_gate(
    exact: Mapping[str, Mapping[str, Any]],
    *,
    card_luhn: int | None = None,
    card_luhn_hit: int | None = None,
    catalogue: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """macro_f2, severity_recall_*, tail watches. Worst-first per-tag table."""
    table = per_tag_table(exact)
    by_tag = {r["tag"]: r for r in table}
    names = catalogue or tuple(sorted(by_tag))
    f2s = []
    for name in names:
        if name in by_tag:
            f2s.append(by_tag[name]["f2"])
        else:
            f2s.append(0.0)  # unlearned / never predicted: 0, not skipped
    severity = []
    excluded = []
    for name in HIGH_SEVERITY:
        row = by_tag.get(name)
        if row is None or row["gold"] < MIN_SUPPORT:
            excluded.append({"tag": name, "gold": 0 if row is None else row["gold"],
                             "reason": "thin_support"})
            continue
        severity.append({"tag": name, "recall": row["recall"], "gold": row["gold"]})
    credit_luhn = None
    if card_luhn is not None and card_luhn_hit is not None:
        credit_luhn = {
            "tag": "CREDIT_CARD_luhn_valid",
            "gold": card_luhn,
            "recall": (card_luhn_hit / card_luhn) if card_luhn else 0.0,
        }
        if card_luhn >= MIN_SUPPORT:
            severity.append(
                {"tag": "CREDIT_CARD_luhn_valid", "recall": credit_luhn["recall"],
                 "gold": card_luhn}
            )
        else:
            excluded.append({"tag": "CREDIT_CARD_luhn_valid", "gold": card_luhn,
                             "reason": "thin_support"})
        excluded.append({
            "tag": "CREDIT_CARD",
            "gold": by_tag.get("CREDIT_CARD", {}).get("gold", 0),
            "reason": "luhn_invalid_gold_excluded_from_gate",
        })

    recs = [s["recall"] for s in severity]
    return {
        "macro_f2": (sum(f2s) / len(f2s)) if f2s else 0.0,
        "f2_min": min(f2s) if f2s else 0.0,
        "f2_median": float(median(f2s)) if f2s else 0.0,
        "n_tags_f2_zero": sum(1 for x in f2s if x == 0.0),
        "n_tags_f2_below_10pct": sum(1 for x in f2s if x < 0.10),
        "n_catalogue": len(names),
        "severity_recall_min": min(recs) if recs else None,
        "severity_recall_mean": (sum(recs) / len(recs)) if recs else None,
        "severity_tags_measured": severity,
        "severity_tags_excluded": excluded,
        "credit_card_luhn": credit_luhn,
        "per_tag": table,
        "micro_tp": sum(r["tp"] for r in table),
        "micro_fp": sum(r["fp"] for r in table),
        "micro_fn": sum(r["fn"] for r in table),
    }


def micro_prf(gate: Mapping[str, Any]) -> tuple[float, float, float]:
    tp, fp, fn = gate["micro_tp"], gate["micro_fp"], gate["micro_fn"]
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f
