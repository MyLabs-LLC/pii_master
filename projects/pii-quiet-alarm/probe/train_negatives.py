"""How many genuinely PII-free documents does the training universe contain?

The precision problem has an obvious candidate mechanism: a model that has never
seen a clean document cannot learn to stay silent on one. This counts, per
training corpus, the documents whose gold lists no sensitive tag at all -- and
separates "labelled clean" from "positive-only gold, so silence means unknown".
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path("/home/lence/workspace/pii_master")
sys.path.insert(0, str(REPO))

from training.priority_data import iter_corpus  # noqa: E402

TRAIN_ROOT = Path("/home/lence/workspace/data/1-train")

out = {}
for d in sorted(TRAIN_ROOT.iterdir()):
    if not d.is_dir():
        continue
    n = pos = complete = complete_neg = 0
    tags = Counter()
    for row in iter_corpus(d):
        n += 1
        has = bool(row.labels)
        pos += has
        if row.label_complete:
            complete += 1
            complete_neg += not has
        tags.update(row.labels)
    out[d.name] = {
        "n_rows": n,
        "gold_positive": pos,
        "label_complete_rows": complete,
        "usable_negatives": complete_neg,
        "negative_share_of_complete": round(complete_neg / complete, 4) if complete else None,
        "distinct_tags": len(tags),
    }
    print(f"{d.name:<45} n={n:>7} pos={pos:>7} complete={complete:>7} "
          f"usable_negatives={complete_neg:>7}", flush=True)

total = sum(v["usable_negatives"] for v in out.values())
print(f"\nTOTAL usable labelled-clean training documents: {total}")
out["_total_usable_negatives"] = total
Path("projects/pii-quiet-alarm/probe/train_negatives.json").write_text(
    json.dumps(out, indent=1), encoding="utf-8")
