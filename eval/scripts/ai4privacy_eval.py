"""Cross-corpus evaluation: ai4privacy/pii-masking-300k.

Every number this project reports comes from Nemotron-PII, and the model card
names that as limitation #1. This is the first evaluation on a corpus with a
different author, a different label space, a different document style and a
different locale -- which is the only way to find out whether the scores
generalise or describe one dataset.

**It is a hard shift, deliberately.** Nemotron is US narrative prose.
ai4privacy-300k is structured templated data -- JSON, XML, markdown key/value
forms -- and predominantly UK/EU:

    "building": "617", "street": "Holme Wood Lane", "city": "Doncaster"
    - Social Number: 669 398 5477        (ten digits; a US SSN has nine)
    - Phone: +16 079 662 2565            (not NANP)

So the crosswalk splits the types three ways, and the split is the point:

  IN SCOPE      types our detectors are actually built for on this data.
                A low score here is a real finding about the model.
  OUT OF SCOPE  types whose detectors are US-format by construction --
                PHONE_US wants NANP, US_DRIVER_LICENSE wants US wording. A low
                score here is a scope statement, not a quality one, and
                averaging it into the headline would be dishonest in both
                directions: it flatters nothing and it hides the real numbers.
  UNMODELLED    labels we deliberately do not emit (passports, passwords, sex,
                bare dates and times, country, state). Dropped from gold, not
                counted as misses -- scoring against categories we chose not to
                detect measures the crosswalk.

Annotation differences that are ours to absorb, not the corpus's fault:
  * names arrive as GIVENNAME1 / LASTNAME1 / LASTNAME2 like Nemotron's
    first_name / last_name, so the same merge applies;
  * addresses arrive as separate BUILDING / STREET / CITY / POSTCODE fields
    which are usually NOT adjacent in the text (they are distinct JSON keys),
    so merging cannot join them and each is scored on its own;
  * TITLE ("Sir", "Madame") is a salutation, not a name, and is dropped.

    python eval/scripts/ai4privacy_eval.py --parquet ~/ai4privacy/validation.parquet \\
        --model-dir training/artifacts/bundle_l --limit 10000
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pii_master.classify import scan_text  # noqa: E402
from pii_master.entities import DocLabel, EntityType  # noqa: E402
from pii_master.ner import merge_adjacent  # noqa: E402
from pii_master.pipeline import Pipeline  # noqa: E402

# ai4privacy label -> the set of OUR types that should count as correct.
#
# Sets, not single types, and the first draft of this file got it wrong by
# using single types. ai4privacy's `SOCIALNUMBER` covers both US-format social
# security numbers ("Social Security Number: 473-54-7641") and non-US national
# numbers ("669 398 5477", ten digits). Mapping it to NATIONAL_ID alone scored
# 854 correct SSN detections as false positives and NATIONAL_ID recall as
# 0.002 -- a number about the crosswalk, not the model. Where one foreign
# label genuinely spans two of ours, both count.
IN_SCOPE = {
    "EMAIL": {EntityType.EMAIL},
    "IP": {EntityType.IP_ADDRESS},
    "GIVENNAME1": {EntityType.PERSON_NAME},
    "GIVENNAME2": {EntityType.PERSON_NAME},
    "LASTNAME1": {EntityType.PERSON_NAME},
    "LASTNAME2": {EntityType.PERSON_NAME},
    "LASTNAME3": {EntityType.PERSON_NAME},
    "STREET": {EntityType.ADDRESS},
    "BUILDING": {EntityType.ADDRESS},
    "CITY": {EntityType.ADDRESS},
    "POSTCODE": {EntityType.ADDRESS},
    "SECADDRESS": {EntityType.ADDRESS},
    "GEOCOORD": {EntityType.GEO_COORDINATE},
    "USERNAME": {EntityType.USER_ID},
    # US-format ones are SSNs and our rules correctly say so; the rest are
    # national identity numbers. The label does not distinguish them, so both
    # are accepted.
    "SOCIALNUMBER": {EntityType.SSN, EntityType.NATIONAL_ID},
    # An identity-card number. Ours land as NATIONAL_ID or, where the document
    # frames it as a person's record number, USER_ID or MRN.
    "IDCARD": {EntityType.NATIONAL_ID, EntityType.USER_ID, EntityType.MRN},
    # A birth date written as an ISO timestamp ("1963-12-23T00:00:00" in a
    # date_of_birth field) is legitimately either of ours.
    "BOD": {EntityType.DATE_DOB, EntityType.DATE_TIME},
}

OUT_OF_SCOPE = {
    "TEL": {EntityType.PHONE_US, EntityType.FAX_NUMBER},
    "DRIVERLICENSE": {EntityType.US_DRIVER_LICENSE},
}

UNMODELLED = {
    "PASSPORT": "per-country format matrices; deferred (DESIGN.md section 6)",
    "PASS": "password -- secrets profile, not a HIPAA identifier",
    "SEX": "not a HIPAA identifier on its own",
    "DATE": "a bare date is an identifier only when tied to an individual",
    "TIME": "same",
    "COUNTRY": "not a HIPAA identifier",
    "STATE": "Safe Harbor retains state; smaller subdivisions only",
    "TITLE": "a salutation, not a name",
    "CARDISSUER": "the issuer is not the cardholder",
}

CROSSWALK = {**IN_SCOPE, **OUT_OF_SCOPE}
ALL_MAPPED = {t for types in CROSSWALK.values() for t in types}


def merge_gold(spans, text):
    """Merge adjacent same-label gold the way the detector merges predictions.

    `spans` is [(ai4privacy_label, start, end)]; merging is done on the LABEL,
    so GIVENNAME1 + LASTNAME1 join into one name exactly as first_name +
    last_name do on Nemotron.
    """
    labels = sorted({label for label, _, _ in spans})
    index = {label: i for i, label in enumerate(labels)}
    # merge_adjacent only joins types it is told are mergeable, and it keys
    # that off EntityType, so name and address parts are passed through as the
    # types they map to.
    representative = tuple(sorted(CROSSWALK[label], key=lambda t: t.value)[0]
                           for label in labels)
    packed = sorted(((index[label], s, e, 1.0) for label, s, e in spans),
                    key=lambda x: (x[1], x[2]))
    return [(labels[k], s, e) for k, s, e, _ in
            merge_adjacent(packed, text, representative)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--model-dir")
    ap.add_argument("--limit", type=int, default=10000)
    ap.add_argument("--language", default="English")
    ap.add_argument("--min-confidence", type=float, default=0.5)
    ap.add_argument("--rules-only", action="store_true")
    ap.add_argument("--json-out")
    args = ap.parse_args(argv)

    import pyarrow.parquet as pq

    table = pq.read_table(args.parquet,
                          columns=["source_text", "privacy_mask", "language"])
    rows = [(s, m) for s, m, lang in zip(table.column("source_text").to_pylist(),
                                         table.column("privacy_mask").to_pylist(),
                                         table.column("language").to_pylist())
            if args.language in ("*", lang)][:args.limit]

    if args.rules_only:
        pipeline, mode = Pipeline(), "fast (rules only)"
    else:
        from pii_master.pipeline import deep_pipeline
        pipeline = deep_pipeline(args.model_dir,
                                 min_confidence=args.min_confidence)
        mode = f"deep @{args.min_confidence:.2f}"

    # Three views, because a cross-corpus score depends on how generously the
    # label spaces are reconciled and one number hides that choice:
    #   strict   exact offsets AND a type in the accept-set. What we report.
    #   typed    exact offsets, ANY type we model. "Found the identifier,
    #            possibly labelled it something else" -- immune to crosswalk
    #            disputes, and for a redaction workflow it is the honest
    #            operational question, since the span gets masked either way.
    #   located  overlapping span, any type. "Flagged the region at all."
    strict = defaultdict(lambda: [0, 0])      # label -> [hits, gold]
    typed = defaultdict(lambda: [0, 0])
    located = defaultdict(lambda: [0, 0])
    false_positives = defaultdict(int)        # our type -> count
    unmodelled_overlap = defaultdict(int)     # landed on dropped gold
    dropped = defaultdict(int)
    doc_tp = doc_fn = doc_sensitive = 0

    for text, mask in rows:
        raw_gold, sensitive = [], False
        for span in (mask or []):
            label = span["label"]
            if label not in CROSSWALK:
                dropped[label] += 1
                continue
            sensitive = True
            raw_gold.append((label, span["start"], span["end"]))
        gold = merge_gold(raw_gold, text)

        entities = pipeline.run(text)
        by_offset = defaultdict(set)
        for e in entities:
            by_offset[(e.start, e.end)].add(e.type)

        matched = set()
        for label, gs, ge in gold:
            accept = CROSSWALK[label]
            here = by_offset.get((gs, ge), set())
            strict[label][1] += 1
            typed[label][1] += 1
            located[label][1] += 1
            if here & accept:
                strict[label][0] += 1
                matched.add((gs, ge))
            if here & ALL_MAPPED:
                typed[label][0] += 1
                matched.add((gs, ge))
            if any(gs < e.end and e.start < ge for e in entities):
                located[label][0] += 1

        gold_offsets = {(gs, ge) for _, gs, ge in gold}
        # A prediction landing on gold of a label we deliberately dropped is
        # not spurious -- it is a real identifier this corpus annotates and we
        # chose not to score. TITLE ("Madame") reads as part of a name; DATE
        # and TIME are real dates. Counting those as false positives would
        # measure the crosswalk, the same adjustment BASELINE_NEMOTRON.md
        # applies to the rules. Tracked separately, not silently forgiven.
        for e in entities:
            if (e.start, e.end) in gold_offsets:
                continue
            if any(span["start"] < e.end and e.start < span["end"]
                   for span in (mask or [])
                   if span["label"] not in CROSSWALK):
                unmodelled_overlap[e.type] += 1
            else:
                false_positives[e.type] += 1

        if sensitive:
            doc_sensitive += 1
            if scan_text(text, pipeline).label is not DocLabel.NONE:
                doc_tp += 1
            else:
                doc_fn += 1

    def roll(counts, labels):
        hits = sum(counts[l][0] for l in labels if l in counts)
        gold = sum(counts[l][1] for l in labels if l in counts)
        return hits, gold, (hits / gold if gold else 0.0)

    in_labels, out_labels = set(IN_SCOPE), set(OUT_OF_SCOPE)
    print(f"ai4privacy/pii-masking-300k validation — {mode}")
    print(f"  {len(rows):,} {args.language} documents, "
          f"{sum(c[1] for c in strict.values()):,} crosswalked gold spans")
    print(f"  dropped as unmodelled: {sum(dropped.values()):,} spans across "
          f"{len(dropped)} labels\n")

    print(f"  {'':<24} {'strict R':>9} {'typed R':>9} {'located R':>10}")
    for name, labels in (("IN SCOPE", in_labels), ("out of scope", out_labels)):
        print(f"  {name:<24} " + " ".join(
            f"{roll(c, labels)[2]:>9.3f}" for c in (strict, typed))
            + f" {roll(located, labels)[2]:>10.3f}")

    print(f"\n  {'ai4privacy label':<16} {'gold':>7} {'strict R':>9} "
          f"{'typed R':>9} {'located R':>10}  scope")
    for label in sorted(strict, key=lambda l: -strict[l][1]):
        scope = "in" if label in in_labels else "OUT (US-format detector)"
        print(f"  {label:<16} {strict[label][1]:>7,} "
              f"{strict[label][0] / max(strict[label][1], 1):>9.3f} "
              f"{typed[label][0] / max(typed[label][1], 1):>9.3f} "
              f"{located[label][0] / max(located[label][1], 1):>10.3f}  {scope}")

    total_fp = sum(false_positives.values())
    total_ovl = sum(unmodelled_overlap.values())
    total_hit = sum(c[0] for c in typed.values())
    total_pred = total_fp + total_ovl + total_hit
    print(f"\n  Of {total_pred:,} predicted spans:")
    print(f"    {total_hit:>7,} matched crosswalked gold "
          f"({total_hit / max(total_pred, 1):.1%})")
    print(f"    {total_ovl:>7,} landed on gold of a label we do not score "
          f"({total_ovl / max(total_pred, 1):.1%}) -- real identifiers, "
          f"deliberately unscored")
    print(f"    {total_fp:>7,} genuinely spurious "
          f"({total_fp / max(total_pred, 1):.1%})")
    print("  Spurious by type:")
    for entity_type, count in sorted(false_positives.items(),
                                     key=lambda kv: -kv[1])[:8]:
        print(f"    {entity_type.value:<20} {count:>6,}")

    print("\n  Document-level: does this file contain PII?")
    print(f"    {doc_sensitive:,} documents carry a crosswalked identifier")
    print(f"    RECALL {doc_tp / max(doc_sensitive, 1):.4f}  ({doc_fn:,} missed)")

    print("\n  Labels dropped as unmodelled:")
    for label, count in sorted(dropped.items(), key=lambda kv: -kv[1]):
        print(f"    {label:<14} {count:>7,}  {UNMODELLED.get(label, '?')}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "documents": len(rows), "mode": mode, "language": args.language,
            "in_scope": {
                "strict_recall": roll(strict, in_labels)[2],
                "typed_recall": roll(typed, in_labels)[2],
                "located_recall": roll(located, in_labels)[2],
                "gold": roll(strict, in_labels)[1],
            },
            "out_of_scope": {
                "strict_recall": roll(strict, out_labels)[2],
                "typed_recall": roll(typed, out_labels)[2],
                "gold": roll(strict, out_labels)[1],
            },
            "per_label": {l: {"gold": strict[l][1], "strict": strict[l][0],
                              "typed": typed[l][0], "located": located[l][0]}
                          for l in strict},
            "false_positives": {t.value: c for t, c in false_positives.items()},
            "unmodelled_overlap": {t.value: c for t, c in unmodelled_overlap.items()},
            "document_recall": doc_tp / max(doc_sensitive, 1),
            "dropped": dict(dropped),
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
