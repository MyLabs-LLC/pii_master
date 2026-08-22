"""Name recall by surname demographic — acceptance gate 5.

docs/DISTILLATION_PLAN.md gate 5: "A name-demographic slice exists before names
ship." Published PII maskers show materially higher error rates on names
associated with Black and Asian/Pacific Islander individuals
(docs/PRIOR_ART.md section 5f), and the student is the first tier here that
emits `first_name` / `last_name` at all. A single aggregate name F1 hides that
failure mode by construction, so it has to be sliced before the feature ships.

The slice joins two public sources, neither of them ours:

  * Nemotron-PII `last_name` gold spans (CC BY 4.0), the holdout we already use
  * the 2010 US Census "Frequently Occurring Surnames" file (public domain),
    which gives, per surname, the percentage of bearers in each self-reported
    race/ethnicity category

A surname is assigned to a group when that group holds at least `--threshold`
percent of its bearers (default 60), which keeps the buckets interpretable:
"names borne mostly by people who report this category". Surnames below the
threshold are reported separately as `mixed` rather than being forced into a
bucket. This measures the DETECTOR, not people: the groups are properties of a
surname's bearer distribution in census data.

    python eval_names.py --data-dir ~/nemotron --checkpoint artifacts/student_xs.pt \\
        --census names.zip
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import torch

from data import parse_spans, read_split  # noqa: E402
from eval_student import TEACHER_ID, load_student, predict  # noqa: E402

GROUPS = {
    "pctwhite": "White",
    "pctblack": "Black",
    "pctapi": "Asian/Pacific Islander",
    "pctaian": "Am. Indian/Alaska Native",
    "pcthispanic": "Hispanic",
}


def load_census(path: Path) -> dict[str, dict[str, float]]:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            name = next(n for n in archive.namelist() if n.endswith(".csv"))
            with archive.open(name) as handle:
                return _read_census(io.TextIOWrapper(handle, encoding="latin-1"))
    with path.open(encoding="latin-1") as handle:
        return _read_census(handle)


def _read_census(handle) -> dict[str, dict[str, float]]:
    table = {}
    for row in csv.DictReader(handle):
        if row["name"] == "ALL OTHER NAMES":
            continue
        percentages = {}
        for column in GROUPS:
            value = row.get(column, "")
            percentages[column] = float(value) if value not in ("", "(S)") else 0.0
        table[row["name"].upper()] = percentages
    return table


def bucket_of(surname: str, census, threshold: float) -> str | None:
    row = census.get(surname.upper())
    if row is None:
        return None
    column, share = max(row.items(), key=lambda kv: kv[1])
    return GROUPS[column] if share >= threshold else "mixed"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--census", required=True, help="Names_2010Census.csv or names.zip")
    ap.add_argument("--size", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--threshold", type=float, default=60.0)
    ap.add_argument("--min-spans", type=int, default=100,
                    help="groups below this are reported but excluded from the spread")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--json-out")
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer

    census = load_census(Path(args.census))
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_ID)
    student = load_student(Path(args.checkpoint), args.size)
    texts, raw_spans = read_split(args.data_dir, args.split, limit=args.limit)
    texts, raw_spans = texts[:args.limit], raw_spans[:args.limit]
    print(f"scoring last_name recall over {len(texts):,} documents", flush=True)
    predictions = predict(student, tokenizer, texts, torch.device(args.device),
                          args.batch_size)

    hits = defaultdict(int)
    partial = defaultdict(int)
    total = defaultdict(int)
    unmatched = 0
    for text, raw, spans in zip(texts, raw_spans, predictions):
        found = {(t, a, b) for t, a, b in spans if t == "last_name"}
        loose = [(a, b) for t, a, b in spans if t == "last_name"]
        for span in parse_spans(raw):
            if span["label"] != "last_name":
                continue
            surname = text[span["start"]:span["end"]].strip().split()[-1:]
            if not surname:
                continue
            group = bucket_of(surname[0], census, args.threshold)
            if group is None:
                unmatched += 1
                continue
            total[group] += 1
            if ("last_name", span["start"], span["end"]) in found:
                hits[group] += 1
            if any(a < span["end"] and span["start"] < b for a, b in loose):
                partial[group] += 1

    order = sorted(total, key=lambda g: -total[g])
    print(f"\nlast_name recall by census surname group "
          f"(>= {args.threshold:.0f}% of bearers), {sum(total.values()):,} spans; "
          f"{unmatched:,} surnames not in the census file")
    print(f"{'group':>26} {'spans':>8} {'exact R':>9} {'partial R':>10}")
    rows = {}
    for group in order:
        exact = hits[group] / total[group]
        loose_recall = partial[group] / total[group]
        rows[group] = {"spans": total[group], "exact_recall": exact,
                       "partial_recall": loose_recall}
        thin = "  (thin slice)" if total[group] < args.min_spans else ""
        print(f"{group:>26} {total[group]:>8,} {exact:>9.3f} {loose_recall:>10.3f}{thin}")

    named = [g for g in order
             if g != "mixed" and total[g] >= args.min_spans]
    if named:
        best = max(named, key=lambda g: rows[g]["exact_recall"])
        worst = min(named, key=lambda g: rows[g]["exact_recall"])
        print(f"\nspread over groups with >= {args.min_spans} spans: "
              f"{rows[best]['exact_recall'] - rows[worst]['exact_recall']:.3f} "
              f"exact recall between {best} (best) and {worst} (worst)")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"threshold": args.threshold, "documents": len(texts),
             "unmatched_surnames": unmatched, "groups": rows}, indent=2))
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
