#!/usr/bin/env python3
"""Enumerate every PII/PHI label in the nvidia/Nemotron-PII dataset.

Regenerates the counts in docs/NEMOTRON_PII_TAGS.md. Analysis-only tooling:
it needs pyarrow, which is deliberately NOT a runtime dependency of
pii_master (see docs/DESIGN.md section 5).

    pip install pyarrow
    curl -L -o train-00000-of-00001.parquet \
      https://huggingface.co/datasets/nvidia/Nemotron-PII/resolve/main/data/train-00000-of-00001.parquet
    curl -L -o test-00000-of-00001.parquet \
      https://huggingface.co/datasets/nvidia/Nemotron-PII/resolve/main/data/test-00000-of-00001.parquet
    python eval/scripts/nemotron_tags.py --data-dir .

The `spans` column holds a Python-repr string (single quotes), not JSON, so
it is parsed with ast.literal_eval.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def parse_spans(raw):
    if raw is None:
        return []
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return ast.literal_eval(raw)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=".", help="directory holding the parquet files")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = ap.parse_args(argv)

    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("pyarrow is required: pip install pyarrow", file=sys.stderr)
        return 2

    files = sorted(Path(args.data_dir).glob("*-00000-of-00001.parquet"))
    if not files:
        print(f"no Nemotron parquet files in {args.data_dir}", file=sys.stderr)
        return 2

    occ: Counter[str] = Counter()
    docfreq: defaultdict[str, set[str]] = defaultdict(set)
    locale: defaultdict[str, Counter[str]] = defaultdict(Counter)
    examples: defaultdict[str, list[str]] = defaultdict(list)
    total_docs = total_spans = 0

    for path in files:
        table = pq.read_table(path, columns=["uid", "spans", "locale"])
        rows = zip(
            table.column("uid").to_pylist(),
            table.column("spans").to_pylist(),
            table.column("locale").to_pylist(),
        )
        total_docs += table.num_rows
        for uid, raw, loc in rows:
            for span in parse_spans(raw):
                label = span["label"]
                occ[label] += 1
                docfreq[label].add(uid)
                locale[label][loc] += 1
                total_spans += 1
                value = str(span.get("text", ""))
                if len(examples[label]) < 6 and value and value not in examples[label]:
                    examples[label].append(value)

    if args.json:
        print(json.dumps({
            "files": [f.name for f in files],
            "documents": total_docs,
            "spans": total_spans,
            "labels": len(occ),
            "counts": dict(occ.most_common()),
            "docfreq": {k: len(v) for k, v in docfreq.items()},
            "locale": {k: dict(v) for k, v in locale.items()},
            "examples": dict(examples),
        }, indent=2))
        return 0

    print(f"files: {', '.join(f.name for f in files)}")
    print(f"documents: {total_docs:,}   spans: {total_spans:,}   distinct labels: {len(occ)}\n")
    print(f"{'#':>3} {'label':<32} {'spans':>9} {'%':>6} {'docs':>8} {'us':>8} {'intl':>8}")
    for i, (label, n) in enumerate(occ.most_common(), 1):
        lc = locale[label]
        print(
            f"{i:>3} {label:<32} {n:>9,} {100 * n / total_spans:>5.2f}%"
            f" {len(docfreq[label]):>8,} {lc.get('us', 0):>8,} {lc.get('intl', 0):>8,}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
