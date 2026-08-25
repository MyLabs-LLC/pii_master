"""Merge the two tagcount sweeps into one head pool for the cascade.

The family was searched twice. The first sweep of 250 trials ran with the
document-level floors switched on, which a bare set of tag heads cannot clear by
construction; its measurements were sound but its search direction was not, so
it was re-ranked rather than re-run (see `quiet_rerank`). The second sweep of
120 trials ran with the corrected objective and found materially better heads.

Both are real evidence and both belong in the pool the cascade chooses from.
Trial numbers restart at zero in each sweep, so the first sweep's are offset:
the cascade records which head trial it used, and `quiet_materialize` looks that
number back up to refit it. Two trials sharing a number would silently
materialise the wrong model.
"""

from __future__ import annotations

import json
from pathlib import Path

TUNING = Path("/home/lence/workspace/pii_master/projects/pii-quiet-alarm/tuning")
V1_OFFSET = 10_000
KEEP = 16


def main() -> int:
    v2 = json.loads((TUNING / "tagcount" / "trials.json").read_text(encoding="utf-8"))
    v1_path = TUNING / "tagcount_v1" / "trials_reranked.json"
    v1 = json.loads(v1_path.read_text(encoding="utf-8")) if v1_path.is_file() else []
    for r in v1:
        r["number"] += V1_OFFSET
        r["sweep"] = "v1 (doc floors on, re-ranked)"
    for r in v2:
        r["sweep"] = "v2 (corrected objective)"

    pool = v1 + v2
    ranked = sorted(pool, key=lambda r: (r["feasible"], r["metrics"]["objective"]),
                    reverse=True)
    (TUNING / "tagcount" / "best.json").write_text(
        json.dumps(ranked[:KEEP], indent=1), encoding="utf-8")
    (TUNING / "tagcount" / "trials_merged.json").write_text(
        json.dumps(ranked, indent=1), encoding="utf-8")

    print(f"merged {len(v1)} v1 + {len(v2)} v2 = {len(pool)} tagcount trials")
    print(f"{'trial':>7}{'F0.5':>9}{'P':>9}{'R':>9}{'minR':>9}  sweep")
    for r in ranked[:8]:
        m = r["metrics"]
        print(f"{r['number']:>7}{m['priority_macro_f05']:>9.4f}"
              f"{m['priority_macro_precision']:>9.4f}{m['priority_macro_recall']:>9.4f}"
              f"{m['priority_min_recall']:>9.4f}  {r['sweep']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
