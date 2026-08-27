"""Import `Synthetic_PDF_Corpus_v2_1612` into the loader's contract.

The corpus ships its own layout — `labels/<vertical>/<type>/<id>.json` carrying
`{document: {PII_tags, PCI_tags, PHI_tags, ...}, entities: [...]}` and canonical
text under `text/`. The loader in `priority_data.iter_raw_rows` reads either a
`labels.jsonl` or a `manifest.json`, so this writes the `labels.jsonl` and changes
nothing else about the corpus.

## Why this corpus is worth adding

It is the first evaluation corpus that is **complete-gold AND uses this project's
exact taxonomy**. `metadata/taxonomy.json` is the GAIA scorecard's 60 tags —
the same list `h2h_scorecard_catalogue` builds the 61-label catalogue from — so
the tag names map by the same slug rule rather than by a hand-written guess. Every
previous mapping in this project needed one, and each was a place to be wrong.

That matters for a specific measured problem: the suite has only five corpora that
can measure per-tag precision, which is why `cascade_p88r90`'s precision confidence
interval is [0.8448, 0.9517] — wide enough that a point estimate of 0.9000 cannot
be called a pass. A sixth independent complete-gold corpus is the only honest way
to narrow that.

## What it can and cannot measure

**Tagging: complete.** Absent tag means absent, so precision-bearing per-tag
metrics are real here.

**The document question: no.** Only 20 of 1,612 documents carry no tag at all — a
1.2% negative rate, below the 30-instance floor the policies use. So this corpus is
deliberately NOT added to `h2h_eval.DOC_MEASURABLE_STEMS`; document precision and
specificity here would be computed from twenty negatives and would deserve no
weight. "Cannot measure" and "measured badly" are different claims.

**Spans: yes, and unused for now.** Every entity carries `char_start`/`char_end`
against the canonical text. This is only the second corpus in the repo with span
gold, so it is preserved in `spans.jsonl` for a future token-level run even though
the document-level task ignores it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CORPUS = Path("/home/lence/workspace/data/2-eval/Synthetic_PDF_Corpus_v2_1612")


def slug(vertical: str, name: str) -> str:
    """The same rule `h2h_scorecard_catalogue` uses. Identical by construction."""
    s = name.lower().replace("&", " and ")
    s = re.sub(r"[()'\-,.]", " ", s)
    s = re.sub(r"\s+", "_", s.strip())
    return re.sub(r"_+", "_", f"sensitive_{vertical.lower()}_{s}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--catalogue", type=Path,
                    default=Path("projects/pii-scorecard-60/cache/catalogue.json"))
    ap.add_argument("--check", action="store_true", help="report and write nothing")
    args = ap.parse_args()

    catalogue = set(json.loads(args.catalogue.read_text())["labels"])
    taxonomy = json.loads((args.corpus / "metadata/taxonomy.json").read_text())

    # Every tag the corpus can emit must land in the catalogue, or gold is being
    # silently dropped and every metric on that tag is wrong.
    unmapped = []
    for vertical, names in taxonomy.items():
        for name in names:
            if slug(vertical, name) not in catalogue:
                unmapped.append((vertical, name, slug(vertical, name)))
    n_tax = sum(len(v) for v in taxonomy.values())
    print(f"taxonomy: {n_tax} tags across {sorted(taxonomy)}")
    if unmapped:
        print(f"UNMAPPED ({len(unmapped)}):")
        for v, n, s in unmapped:
            print(f"    {v:<5} {n:<48} -> {s}")
        raise SystemExit("refusing to import: gold would be silently dropped")
    print(f"all {n_tax} taxonomy tags map into the {len(catalogue)}-label catalogue")

    rows, spans, seen = [], [], Counter()
    label_files = sorted((args.corpus / "labels").rglob("*.json"))
    for lf in label_files:
        d = json.loads(lf.read_text(encoding="utf-8"))
        doc = d["document"]
        text_rel = doc.get("canonical_text_path")
        if not text_rel or not (args.corpus / text_rel).is_file():
            raise SystemExit(f"{lf}: canonical text missing ({text_rel})")
        gold = []
        for vertical, key in (("PII", "PII_tags"), ("PCI", "PCI_tags"),
                              ("PHI", "PHI_tags")):
            for name in doc.get(key, []) or []:
                gold.append(slug(vertical, name))
        gold = sorted(set(gold))
        seen.update(gold)
        uid = str(doc["document_id"])
        rows.append({
            "uid": uid,
            "file": text_rel,
            "gold": gold,
            "split": "eval",
            "provenance": "synthetic",
            "label_provenance": "generated_controlled_field (synthetic, deidentified)",
            "source_corpus": doc.get("document_type_label", ""),
            "document_type": doc.get("document_type_label", ""),
        })
        for e in d.get("entities", []) or []:
            if e.get("char_start") is None or e.get("char_end") is None:
                continue
            tag = slug(str(e.get("category", "")), str(e.get("tag", "")))
            if tag in catalogue:
                spans.append({"uid": uid, "file": text_rel,
                              "start": int(e["char_start"]), "end": int(e["char_end"]),
                              "text": e.get("text", ""), "tag_id": tag,
                              "annotator": e.get("source", "")})

    empty = sum(1 for r in rows if not r["gold"])
    print(f"\n{len(rows):,} documents, {empty} with no tag "
          f"({empty / max(len(rows), 1):.1%} — too few document-level negatives to "
          f"measure doc precision; NOT added to DOC_MEASURABLE_STEMS)")
    print(f"{len(spans):,} entity spans over {len({s['uid'] for s in spans}):,} documents")
    print(f"{len(seen)} distinct tags present; catalogue tags with no gold here: "
          f"{len(catalogue - set(seen))}")
    print("\nmost common tags:")
    for tag, n in seen.most_common(8):
        print(f"    {tag.replace('sensitive_', ''):<46} {n:>6,}")

    if args.check:
        print("\n--check: nothing written")
        return 0

    out = args.corpus / "labels.jsonl"
    out.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows),
                   encoding="utf-8")
    sp = args.corpus / "spans.jsonl"
    sp.write_text("".join(json.dumps(s, sort_keys=True) + "\n" for s in spans),
                  encoding="utf-8")
    print(f"\n-> {out}\n-> {sp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
