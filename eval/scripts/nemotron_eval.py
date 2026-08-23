#!/usr/bin/env python3
"""Score the pipeline against a Nemotron-PII holdout — an external yardstick.

The frozen corpus in eval/corpus/ was authored alongside the detectors, so
its scores are a regression test, not a quality claim. This script runs the
same detectors against data nobody here wrote, mapped through
pii_master.crosswalk, and is the first non-tautological number we have.

Analysis-only tooling: needs pyarrow, which is deliberately NOT a runtime
dependency (docs/DESIGN.md section 5). The parquet is not vendored.

    pip install pyarrow
    curl -L -o test-00000-of-00001.parquet \
      https://huggingface.co/datasets/nvidia/Nemotron-PII/resolve/main/data/test-00000-of-00001.parquet
    python eval/scripts/nemotron_eval.py --data-dir . --out docs/BASELINE_NEMOTRON.md

Only the 12 mapped labels are scored for recall. Gold spans of the 43
unmodelled labels are not counted as misses: scoring ourselves against
categories we deliberately do not detect would measure the crosswalk, not
the detectors.

Precision needs the same care in the other direction. A prediction that does
not match mapped gold falls into one of three very different buckets, and
lumping them together is misleading -- it reported 0.49 precision for
PHONE_US when almost every "false positive" was a real fax number:

  mapped_mismatch     overlaps mapped gold, wrong boundary or type -- a real error
  unmodelled_overlap  overlaps gold of a label we do not model (fax_number,
                      tax_id, biometric_identifier). We found a genuine
                      identifier and gave it a type outside our taxonomy: a
                      labelling gap, not a spurious alarm
  spurious            overlaps no gold at all -- the only true false positive

Strict precision counts all three against us; adjusted precision excludes
unmodelled_overlap, whose gold was withheld from the denominator anyway.
Both are reported.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pii_master.classify import scan_text  # noqa: E402
from pii_master.crosswalk import RULE_MAPPED, to_entity_type  # noqa: E402

# This script measures the RULES tier. v0.3 adopted 22 more Nemotron labels
# (names, addresses, fax, routing numbers, ...) for the Stage 2 model, and
# scoring the rules against those would measure the crosswalk rather than the
# detectors -- no regex emits a PERSON_NAME, so every one of them would be a
# guaranteed miss and the committed baseline would drop for a reason that has
# nothing to do with the rules changing. So the scope here stays the 12 labels
# a rule can actually fire on. The model's score on the adopted labels is
# training/eval_student.py's "adopted" table.
NEMOTRON_TO_ENTITY = RULE_MAPPED
from pii_master.evaluation import TypeScore  # noqa: E402
from pii_master.validators import luhn_ok  # noqa: E402


def parse_spans(raw):
    if raw is None:
        return []
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return ast.literal_eval(raw)


def score(rows, limit=None):
    exact: dict[str, TypeScore] = defaultdict(TypeScore)
    partial: dict[str, TypeScore] = defaultdict(TypeScore)
    by_locale: dict[str, dict[str, TypeScore]] = {
        "us": defaultdict(TypeScore), "intl": defaultdict(TypeScore)
    }
    buckets: dict[str, Counter] = {
        "mapped_mismatch": Counter(), "unmodelled_overlap": Counter(),
        "spurious": Counter(),
    }
    confusions = Counter()
    dropped = Counter()
    # The dataset's synthetic card numbers mostly do not satisfy Luhn, and our
    # detector requires it. Measuring recall against unreachable gold would
    # report a dataset property as our failure, so track the reachable subset.
    card_gold = card_luhn = card_luhn_hit = 0
    docs = flagged = docs_with_gold = 0

    for i, (text, raw, locale) in enumerate(rows):
        if limit and i >= limit:
            break
        docs += 1
        gold: dict[str, list[tuple[int, int]]] = defaultdict(list)
        gold_all: list[tuple[str, int, int]] = []
        for span in parse_spans(raw):
            label = span["label"]
            gold_all.append((label, span["start"], span["end"]))
            mapped = NEMOTRON_TO_ENTITY.get(label)
            to_entity_type(label)          # still fails loudly on a new label
            if label == "credit_debit_card":
                card_gold += 1
                digits = "".join(c for c in str(span.get("text", "")) if c.isdigit())
                if 13 <= len(digits) <= 19 and luhn_ok(digits):
                    card_luhn += 1
                    card_luhn_hit += 1  # provisional; corrected below
                    gold.setdefault("__card_luhn__", []).append(
                        (span["start"], span["end"])
                    )
            if mapped is None:
                dropped[label] += 1
                continue
            gold[mapped.value].append((span["start"], span["end"]))

        report = scan_text(text)
        pred: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for e in report.entities:
            pred[e.type.value].append((e.start, e.end))

        if gold:
            docs_with_gold += 1
            if report.entities:
                flagged += 1

        # Correct the provisional Luhn-hit count: only spans we actually found.
        luhn_spans = set(gold.pop("__card_luhn__", []))
        if luhn_spans:
            found = {(e.start, e.end) for e in report.entities
                     if e.type.value == "CREDIT_CARD"}
            card_luhn_hit -= len(luhn_spans)
            card_luhn_hit += len(luhn_spans & found)

        # Triage every prediction that is not an exact mapped hit.
        mapped_spans = {(t, a, b) for t, v in gold.items() for a, b in v}
        for e in report.entities:
            key = (e.type.value, e.start, e.end)
            if key in mapped_spans:
                continue
            if any(e.start < b and a < e.end for _, a, b in
                   [(t, a, b) for t, v in gold.items() for a, b in v]):
                buckets["mapped_mismatch"][e.type.value] += 1
                continue
            hit = [lbl for lbl, a, b in gold_all if e.start < b and a < e.end]
            if hit:
                buckets["unmodelled_overlap"][e.type.value] += 1
                confusions[(e.type.value, hit[0])] += 1
            else:
                buckets["spurious"][e.type.value] += 1

        for entity_type in set(gold) | set(pred):
            g, p = gold.get(entity_type, []), pred.get(entity_type, [])
            for table, scores in (("exact", exact), ("partial", partial)):
                sc = scores[entity_type]
                sc.gold += len(g)
                if table == "exact":
                    tp = len(set(g) & set(p))
                else:
                    remaining, tp = sorted(p), 0
                    for gs, ge in sorted(g):
                        for idx, (ps, pe) in enumerate(remaining):
                            if gs < pe and ps < ge:
                                tp += 1
                                del remaining[idx]
                                break
                sc.tp += tp
                sc.fp += len(p) - tp
                sc.fn += len(g) - tp
            if locale in by_locale:
                sc = by_locale[locale][entity_type]
                tp = len(set(g) & set(p))
                sc.gold += len(g)
                sc.tp += tp
                sc.fp += len(p) - tp
                sc.fn += len(g) - tp

    return {
        "documents": docs,
        "documents_with_mapped_gold": docs_with_gold,
        "documents_flagged": flagged,
        "exact": dict(exact),
        "partial": dict(partial),
        "by_locale": {k: dict(v) for k, v in by_locale.items()},
        "dropped_unmodelled": dict(dropped),
        "buckets": {k: dict(v) for k, v in buckets.items()},
        "confusions": {f"{a} -> {b}": n for (a, b), n in confusions.most_common(12)},
        "card_gold": card_gold,
        "card_luhn": card_luhn,
        "card_luhn_hit": card_luhn_hit,
    }


def render(result, split, files) -> str:
    ex, pa = result["exact"], result["partial"]
    dropped_total = sum(result["dropped_unmodelled"].values())
    scored_total = sum(s.gold for s in ex.values())
    lines = [
        "# External baseline — Nemotron-PII holdout",
        "",
        "Measured with `eval/scripts/nemotron_eval.py` against",
        "[`nvidia/Nemotron-PII`](https://huggingface.co/datasets/nvidia/Nemotron-PII)"
        " (CC BY 4.0), data",
        "nobody in this repo authored. **This is the honest number**; the frozen corpus in",
        "`eval/corpus/` is a regression test, not a quality claim.",
        "",
        f"Split: `{split}` · documents scored: **{result['documents']:,}** · "
        f"source: {', '.join(files)}",
        "",
        "## Scope",
        "",
        f"Only the **{len(NEMOTRON_TO_ENTITY)} rule-mapped labels** "
        f"(`crosswalk.RULE_MAPPED`) are scored: {scored_total:,} gold spans. "
        "The 22 labels v0.3 adopted for the Stage 2 model are out of scope "
        "here — no regex emits a name — and are scored by "
        "`training/eval_student.py`.",
        f"Gold spans of every other label "
        f"({dropped_total:,} spans) are dropped, not counted as misses —"
        " scoring against categories",
        "we deliberately do not detect would measure the crosswalk, not the detectors.",
        "That omission is the Stage 2 / Track C backlog, sized in docs/NEMOTRON_PII_TAGS.md.",
        "",
        "## Span-level results (mapped types only)",
        "",
        "| type | gold | TP | FP | FN | P | R | F1 | partial R |",
        "|---|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for t in sorted(ex):
        s, p = ex[t], pa.get(t, TypeScore())
        lines.append(
            f"| `{t}` | {s.gold:,} | {s.tp:,} | {s.fp:,} | {s.fn:,} | "
            f"{s.precision:.3f} | {s.recall:.3f} | {s.f1:.3f} | {p.recall:.3f} |"
        )
    tp = sum(s.tp for s in ex.values())
    fp = sum(s.fp for s in ex.values())
    fn = sum(s.fn for s in ex.values())
    micro_p = tp / (tp + fp) if tp + fp else 0.0
    micro_r = tp / (tp + fn) if tp + fn else 0.0
    micro_f = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0
    b = result["buckets"]
    mismatch = sum(b["mapped_mismatch"].values())
    unmodelled = sum(b["unmodelled_overlap"].values())
    spurious = sum(b["spurious"].values())
    predictions = tp + mismatch + unmodelled + spurious
    adj_p = tp / (tp + mismatch + spurious) if tp + mismatch + spurious else 0.0
    adj_f = 2 * adj_p * micro_r / (adj_p + micro_r) if adj_p + micro_r else 0.0
    lines += [
        f"| **micro-average** | {scored_total:,} | {tp:,} | {fp:,} | {fn:,} | "
        f"**{micro_p:.3f}** | **{micro_r:.3f}** | **{micro_f:.3f}** | |",
        "",
        "## What the \"false positives\" actually are",
        "",
        "Strict precision counts every non-exact prediction against us. That is",
        "misleading here: most of those predictions land on a **real identifier** whose",
        "Nemotron label we do not model, so its gold was withheld from the denominator.",
        "Triaging all "
        f"{predictions:,} predictions:",
        "",
        "| bucket | count | share | meaning |",
        "|---|--:|--:|---|",
        f"| exact hit | {tp:,} | {100*tp/max(1,predictions):.1f}% | correct type and boundaries |",
        f"| mapped_mismatch | {mismatch:,} | {100*mismatch/max(1,predictions):.1f}% |"
        " overlaps mapped gold, wrong edges or type — a real error |",
        f"| unmodelled_overlap | {unmodelled:,} | {100*unmodelled/max(1,predictions):.1f}% |"
        " a genuine identifier of a label we do not model — a taxonomy gap |",
        f"| **spurious** | **{spurious:,}** | **{100*spurious/max(1,predictions):.1f}%** |"
        " **overlaps no gold at all — the only true false alarms** |",
        "",
        f"**Adjusted micro-precision (excluding unmodelled_overlap): {adj_p:.3f}**, "
        f"adjusted F1 {adj_f:.3f}.",
        "",
        "Where the unmodelled overlaps go — each row is a missing type, and the top",
        "rows are exactly the Track C backlog:",
        "",
        "| our type | actual Nemotron label | count |",
        "|---|---|--:|",
    ] + [
        f"| `{k.split(' -> ')[0]}` | `{k.split(' -> ')[1]}` | {v:,} |"
        for k, v in result["confusions"].items()
    ] + [
        "",
        "## Reading these numbers",
        "",
        f"**CREDIT_CARD recall is a dataset artifact, not a detector failure.** Only "
        f"{result['card_luhn']:,} of the {result['card_gold']:,} gold card spans "
        f"({100*result['card_luhn']/max(1,result['card_gold']):.1f}%) satisfy the Luhn",
        "checksum; the rest are card-shaped digit strings that no real payment network",
        "would issue. Our detector requires Luhn — that is what makes it our",
        "highest-precision rule — so most of this gold is unreachable by design.",
        f"**Against the Luhn-valid subset, recall is "
        f"{result['card_luhn_hit']/max(1,result['card_luhn']):.3f}.** Keep the check.",
        "",
        "**US_DRIVER_LICENSE recall is a genuine, cheap miss.** Nemotron's cue is",
        "literally \"certificate license number\"; our detector only accepts driver's-licence",
        "wording, so it fires on almost none of them. This is the Track C",
        "`LICENSE_NUMBER` umbrella (HIPAA #11) arriving as a measured number.",
        "",
        "**ACCOUNT_NUMBER / MRN / HEALTH_PLAN_ID recall (0.26-0.46) is the cue-anchoring",
        "trade-off working as designed.** Those detectors only fire next to a cue word;",
        "cue-free instances are exactly what Stage 2 is for. Precision on them stays",
        "high (0.78-0.96), which is the half of the bargain we bought.",
        "",
        "**PHONE_US** loses recall to international numbers (we validate NANP) and",
        "precision to fax numbers, which share the shape and have no type yet.",
        "",
        "## Document-level reach",
        "",
        f"Of {result['documents_with_mapped_gold']:,} documents containing at least one"
        " mapped gold span,",
        f"we flagged **{result['documents_flagged']:,}**"
        f" ({100 * result['documents_flagged'] / max(1, result['documents_with_mapped_gold']):.1f}%)"
        " with at least one entity.",
        "Nemotron carries no document label, so NONE/PII/PHI accuracy is not evaluable here.",
        "",
        "## US vs international",
        "",
        "| type | US recall | intl recall |",
        "|---|--:|--:|",
    ]
    us, intl = result["by_locale"]["us"], result["by_locale"]["intl"]
    for t in sorted(set(us) | set(intl)):
        a, b = us.get(t, TypeScore()), intl.get(t, TypeScore())
        lines.append(f"| `{t}` | {a.recall:.3f} | {b.recall:.3f} |")
    lines += [
        "",
        "Locale gaps are expected, not bugs: `PHONE_US` validates NANP and rejects",
        "international numbers, and `SSN` has no international analogue (Nemotron tags",
        "those `national_id`, which we do not model).",
        "",
        "## Reproducing",
        "",
        "```console",
        "$ pip install pyarrow   # analysis only, not a runtime dependency",
        "$ python eval/scripts/nemotron_eval.py --data-dir <dir-with-parquet> \\",
        "    --out docs/BASELINE_NEMOTRON.md",
        "```",
    ]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", help="write a markdown report here")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("pyarrow is required: pip install pyarrow", file=sys.stderr)
        return 2

    files = sorted(Path(args.data_dir).glob(f"{args.split}-*.parquet"))
    if not files:
        print(f"no {args.split} parquet in {args.data_dir}", file=sys.stderr)
        return 2

    rows = []
    for path in files:
        t = pq.read_table(path, columns=["text", "spans", "locale"])
        rows.extend(zip(t.column("text").to_pylist(),
                        t.column("spans").to_pylist(),
                        t.column("locale").to_pylist()))

    result = score(rows, limit=args.limit)
    report = render(result, args.split, [f.name for f in files])
    if args.json:
        print(json.dumps({
            "documents": result["documents"],
            "exact": {k: v.to_dict() for k, v in result["exact"].items()},
        }, indent=2))
    else:
        print(report)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
