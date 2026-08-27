"""Build the scorecard's 61-label catalogue and re-index the cached label gold.

## What this replaces

`training/quiet_cache.build_catalogue` derives its vocabulary from whatever tags
the manifests happen to carry, after `quiet_data.COLLAPSE` has folded four of them
away. This derives the vocabulary from the **scorecard** instead -- the
authoritative business tag list at

    /home/lence/workspace/data/scorecard/
        Scorecard - GAIA Catalog(Rasool-PII-PCI-PHI).csv

-- and applies **no collapse at all**. The 60 tags in that file are read from the
file rather than transcribed into this module, so the catalogue cannot drift from
the scorecard by somebody editing one and not the other.

Two stated edits on top of the 60, both argued in the project README:

* `sensitive_pci_routing_number` is **dropped** (1 training row, 0 evaluation
  rows -- the already-disabled head);
* `sensitive_pci_swift_code` is **kept** although the scorecard omits it (542 /
  130 rows, measurable on one corpus).

Total: **61**.

## Why only the labels are rebuilt

A cache entry holds features (`indices_deep` / `indptr_deep` / `n_chars`) and gold
(`label_cols` / `label_indptr` / `doc_target` / `tag_complete`). The features are
hashed from document text and do not know the catalogue exists, so changing the
label space cannot change them. Re-extracting them would mean re-reading 657,560
documents to arrive at byte-identical arrays.

So the feature arrays are copied through unchanged and only the label arrays are
recomputed, from the manifests, under the new index. `doc_target` and
`tag_complete` are properties of the corpus rather than of the vocabulary and are
also carried through.

The source cache is **never written**: 128 published results depend on it.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.quiet_cache import CACHE_ROOT as SOURCE_CACHE  # noqa: E402
from training.quiet_data import (  # noqa: E402
    EVAL_ROOT, TRAIN_ROOT, iter_quiet_corpus, list_dataset_dirs,
)

SCORECARD = Path("/home/lence/workspace/data/scorecard/"
                 "Scorecard - GAIA Catalog(Rasool-PII-PCI-PHI).csv")
DROP = ("sensitive_pci_routing_number",)
KEEP_OFF_SCORECARD = ("sensitive_pci_swift_code",)


def slug(vertical: str, name: str) -> str:
    """`PII, Driver's License Number` -> `sensitive_pii_driver_s_license_number`.

    Matches the slugs already in the manifests -- verified by
    `--check`, which fails if any scorecard tag has no gold anywhere.
    """
    s = name.lower().replace("&", " and ")
    s = re.sub(r"[()'\-,.]", " ", s)
    s = re.sub(r"\s+", "_", s.strip())
    return re.sub(r"_+", "_", f"sensitive_{vertical.lower()}_{s}")


def scorecard_labels() -> tuple[list[str], dict[str, tuple[str, str]]]:
    rows = [r for r in csv.reader(SCORECARD.open(encoding="utf-8")) if r and r[0].strip()]
    order, meta = [], {}
    for vertical, name in rows:
        tag = slug(vertical.strip(), name.strip())
        if tag in DROP:
            continue
        if tag not in meta:
            order.append(tag)
            meta[tag] = (vertical.strip(), name.strip())
    for tag in KEEP_OFF_SCORECARD:
        if tag not in meta:
            order.append(tag)
            meta[tag] = ("PCI", "SWIFT Code (kept: not on the scorecard)")
    return sorted(order), meta


def build(out_root: Path, n_features: int, check_only: bool) -> int:
    labels, meta = scorecard_labels()
    index = {t: i for i, t in enumerate(labels)}
    print(f"scorecard: {SCORECARD.name}")
    print(f"catalogue: {len(labels)} labels "
          f"(dropped {list(DROP)}, kept off-scorecard {list(KEEP_OFF_SCORECARD)})")

    # ---------------------------------------------------------------- coverage
    seen: dict[str, int] = {t: 0 for t in labels}
    unknown: dict[str, int] = {}
    for root in (TRAIN_ROOT, EVAL_ROOT):
        for d in list_dataset_dirs(root):
            for qr in iter_quiet_corpus(d):
                for t in qr.row.labels:          # RAW labels: no collapse applied
                    if t in index:
                        seen[t] += 1
                    elif t.startswith("sensitive_"):
                        unknown[t] = unknown.get(t, 0) + 1
    empty = [t for t, n in seen.items() if n == 0]
    print(f"\ntags with no gold anywhere: {len(empty)}")
    for t in empty:
        print(f"    {t}   ({meta[t][0]} / {meta[t][1]})")
    print(f"sensitive tags in the gold but NOT on the scorecard: {len(unknown)}")
    for t, n in sorted(unknown.items(), key=lambda kv: -kv[1]):
        print(f"    {t:<56} {n:>8,}   DROPPED from the catalogue")

    if check_only:
        return 0

    out_root.mkdir(parents=True, exist_ok=True)
    cat = {"labels": labels, "n_labels": len(labels), "n_features": n_features,
           "source": str(SCORECARD), "collapse": {},
           "dropped": list(DROP), "kept_off_scorecard": list(KEEP_OFF_SCORECARD),
           "tags_in_gold_not_on_scorecard": unknown,
           "tags_with_no_gold": empty,
           "scorecard_meta": {t: {"vertical": v, "name": n} for t, (v, n) in meta.items()}}
    (out_root / "catalogue.json").write_text(json.dumps(cat, indent=1), encoding="utf-8")
    print(f"\n-> {out_root / 'catalogue.json'}")

    # ------------------------------------------------------- re-index the gold
    corpora = {d.name: d for root in (TRAIN_ROOT, EVAL_ROOT)
               for d in list_dataset_dirs(root)}
    for src in sorted(SOURCE_CACHE.glob("*.npz")):
        name = src.stem
        if name not in corpora:
            print(f"  {name:<52} SKIPPED (no dataset dir)")
            continue
        with np.load(src, allow_pickle=False) as z:
            keep = {k: z[k] for k in z.files
                    if not k.startswith("label_")}     # features + doc gold, unchanged
            n_rows = len(z["doc_target"])
        indptr = np.zeros(n_rows + 1, dtype=np.int64)
        cols: list[int] = []
        for i, qr in enumerate(iter_quiet_corpus(corpora[name])):
            row = sorted({index[t] for t in qr.row.labels if t in index})
            cols.extend(row)
            indptr[i + 1] = len(cols)
        if indptr.size - 1 != n_rows:
            raise SystemExit(f"{name}: manifest has {indptr.size - 1} rows, "
                             f"cache has {n_rows}; they are not the same corpus")
        out = out_root / src.name
        np.savez(out, label_indptr=indptr,
                 label_cols=np.asarray(cols, dtype=np.int32), **keep)
        print(f"  {name:<52} {n_rows:>7,} rows  {len(cols):>9,} tag hits")
    print(f"\n-> {out_root}  ({len(list(out_root.glob('*.npz')))} cache files)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path("projects/pii-scorecard-60/cache"))
    ap.add_argument("--n-features", type=int, default=1 << 18)
    ap.add_argument("--check", action="store_true",
                    help="report coverage and write nothing")
    args = ap.parse_args()
    return build(args.out, args.n_features, args.check)


if __name__ == "__main__":
    raise SystemExit(main())
