"""Re-rank a completed family's trials under the corrected criterion.

The first ``tagcount`` sweep was scored with the document-level floors switched
on. A bare set of per-tag heads decides "this document has PII" by firing at
least one tag, so it cannot clear those floors by construction: all 250 trials
came back infeasible, and ``best.json`` then ordered them by smallest deficit --
which rewards heads that fire *less*, the opposite of what the cascade wants
from this family.

The search direction was wrong; the measurements were not. Every trial recorded
its own ``priority_macro_f05``, ``priority_min_recall`` and the rest, so the
family can be re-ranked from what it already measured instead of re-run. This
writes a corrected ``best.json`` and leaves ``trials.json`` untouched, so the
original ordering stays auditable.

Nothing here computes a new number. It reorders recorded ones, which is why it
is a re-rank and not a re-scoring.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TUNING = Path("/home/lence/workspace/pii_master/projects/pii-quiet-alarm/tuning")
PRIORITY_RECALL_FLOOR = 0.75


def rerank(family: str, floor: float = PRIORITY_RECALL_FLOOR, keep: int = 16) -> list[dict]:
    out = TUNING / family
    trials = json.loads((out / "trials.json").read_text(encoding="utf-8"))
    for r in trials:
        m = r["metrics"]
        # Feasibility for a head-only family is the priority recall floor alone.
        meets = m.get("priority_min_recall", 0.0) >= floor
        r["feasible"] = bool(meets)
        r["metrics"]["objective"] = m.get("priority_macro_f05", 0.0) if meets else -(
            floor - m.get("priority_min_recall", 0.0))
        r["rerank"] = {"criterion": "priority_macro_f05 subject to priority_min_recall >= "
                                   f"{floor}", "doc_floors_applied": False}
    ranked = sorted(trials, key=lambda r: (r["feasible"], r["metrics"]["objective"]),
                    reverse=True)
    (out / "best.json").write_text(json.dumps(ranked[:keep], indent=1), encoding="utf-8")
    (out / "trials_reranked.json").write_text(json.dumps(ranked, indent=1), encoding="utf-8")
    return ranked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--floor", type=float, default=PRIORITY_RECALL_FLOOR)
    args = ap.parse_args()
    ranked = rerank(args.family, args.floor)
    n_ok = sum(r["feasible"] for r in ranked)
    print(f"{args.family}: {len(ranked)} trials re-ranked, {n_ok} meet the "
          f"priority recall floor {args.floor}")
    print(f"{'trial':>7}{'F0.5':>9}{'P':>9}{'R':>9}{'minR':>9}  profile/mode")
    for r in ranked[:8]:
        m = r["metrics"]
        print(f"{r['number']:>7}{m['priority_macro_f05']:>9.4f}"
              f"{m['priority_macro_precision']:>9.4f}{m['priority_macro_recall']:>9.4f}"
              f"{m['priority_min_recall']:>9.4f}  "
              f"{r['params'].get('profile')}/{r['params'].get('score_mode', '-')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
