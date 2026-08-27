"""Split `Synthetic_PDF_Corpus_v2_1612` into train/eval corpora in the house format.

Produces two datasets matching every convention the other eight follow:

    <n>_synthetic_pdf_train_<positives>/     documents/ labels.jsonl manifest.json README.md
    <n>_synthetic_pdf_eval_<positives>/      same

## The split is stratified and deterministic

Documents are grouped by **vertical/document-type** (`Healthcare/admission-checklist`,
`Finance/loan-agreement`, …) and 80/20 within each group, so both sides carry every
document family the corpus contains. A uniform random split over 1,612 documents
would leave whole families on one side by chance: several families have fewer than
twenty documents, and a family absent from eval is a tag nobody can measure.

Assignment is positional within each family over a `sha256(document_id)` ordering,
so re-running produces the same split, a document never moves between sides, and
every family of two or more documents contributes to eval.

**The eval side is small and that limits what it can measure.** 322 documents put
45 of the 60 tags under the 30-instance floor the policies use, so this corpus can
conclusively judge about 15 tags on its own. It is a useful addition to the suite,
not a replacement for it.

## What this costs, and it is not small

`data/3-holdout/Synthetic_PDF_Corpus_v2_1612` was the only out-of-distribution
test in this repository, and every generalisation finding measured against it —
in particular that in-distribution recall predicts out-of-distribution F1 at
r = +0.97 while precision predicts it at -0.78 — rests on **no model having seen
any of these documents**.

Once 80% of them are training data that is no longer true, and the 3-holdout copy
becomes 80% training data wearing a holdout label. It must not be scored as a
holdout again. This module refuses to delete it (that is the caller's decision)
but the split it writes is incompatible with continuing to use it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SOURCE = Path("/home/lence/workspace/data/3-holdout/Synthetic_PDF_Corpus_v2_1612")
TRAIN_ROOT = Path("/home/lence/workspace/data/1-train")
EVAL_ROOT = Path("/home/lence/workspace/data/2-eval")


def split_family(members: list[dict], frac: float) -> tuple[list[dict], list[dict]]:
    """Split ONE document family, guaranteeing eval coverage where possible.

    Hashing each id independently is deterministic but not stratified: on a family
    of eight documents it can put all eight on one side, and 20 of this corpus's
    163 families came out with no eval document that way. A family absent from
    eval is a document type nobody can measure.

    So assignment is positional within the family, over a `sha256`-ordered list —
    still deterministic and still stable under re-runs, but the proportions hold
    *within* every family and any family of two or more contributes at least one
    eval document.
    """
    ordered = sorted(members, key=lambda r: hashlib.sha256(
        r["uid"].encode()).hexdigest())
    n = len(ordered)
    n_eval = max(1, round(n * (1 - frac))) if n >= 2 else 0
    return ordered[n_eval:], ordered[:n_eval]


def size_token(n: int) -> str:
    """`1.29k` / `322` — the suffix the other corpora use for their positive count."""
    return f"{n / 1000:.2f}k" if n >= 1000 else str(n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--train-frac", type=float, default=0.80)
    ap.add_argument("--check", action="store_true", help="report, write nothing")
    args = ap.parse_args()

    rows = [json.loads(l) for l in
            (args.source / "labels.jsonl").read_text(encoding="utf-8").splitlines() if l]
    spans = defaultdict(list)
    sp = args.source / "spans.jsonl"
    if sp.is_file():
        for line in sp.read_text(encoding="utf-8").splitlines():
            if line:
                s = json.loads(line)
                spans[s["uid"]].append(s)

    # Stratify by the family the canonical path encodes: text/<vertical>/<type>/…
    fam = defaultdict(list)
    for r in rows:
        parts = Path(r["file"]).parts
        fam["/".join(parts[1:3]) if len(parts) >= 3 else "other"].append(r)

    train, ev = [], []
    for family, members in sorted(fam.items()):
        t, e = split_family(members, args.train_frac)
        train.extend(t); ev.extend(e)

    def stats(rs):
        pos = sum(1 for r in rs if r["gold"])
        tags = Counter(t for r in rs for t in r["gold"])
        return pos, tags

    tp, ttags = stats(train)
    ep, etags = stats(ev)
    print(f"source: {len(rows):,} documents, {len(fam)} families")
    print(f"train:  {len(train):,} documents ({len(train)/len(rows):.1%}), "
          f"{tp:,} with tags, {len(ttags)} distinct tags")
    print(f"eval:   {len(ev):,} documents ({len(ev)/len(rows):.1%}), "
          f"{ep:,} with tags, {len(etags)} distinct tags")
    missing = set(ttags) - set(etags)
    if missing:
        print(f"\ntags in train with NO eval instance ({len(missing)}): "
              f"{sorted(t.replace('sensitive_', '') for t in missing)}")
    thin = {t: etags[t] for t in etags if etags[t] < 30}
    print(f"eval tags under the 30-instance measurability floor: {len(thin)}")

    empty = [f for f, m in fam.items() if not split_family(m, args.train_frac)[1]]
    singles = [f for f, m in fam.items() if len(m) == 1]
    print(f"families with no eval document: {len(empty)} "
          f"(of which {len(singles)} are single-document families that cannot be split)")

    if args.check:
        print("\n--check: nothing written")
        return 0

    train_name = f"{len(train)}_synthetic_pdf_train_{size_token(tp)}"
    eval_name = f"{len(ev)}_synthetic_pdf_eval_{size_token(ep)}"
    for name, rs, root, split, tags in ((train_name, train, TRAIN_ROOT, "train", ttags),
                                        (eval_name, ev, EVAL_ROOT, "eval", etags)):
        out = root / name
        if out.exists():
            shutil.rmtree(out)
        (out / "documents").mkdir(parents=True)
        written = []
        for i, r in enumerate(rs):
            src = args.source / r["file"]
            shard = f"{i // 500:04d}"
            rel = f"documents/{shard}/{Path(r['file']).stem}.txt"
            dst = out / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            row = dict(r)
            row["file"] = rel
            row["split"] = split
            row["source_corpus"] = "Synthetic_PDF_Corpus_v2_1612"
            written.append(row)
        (out / "labels.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in written),
            encoding="utf-8")

        keep = {r["uid"] for r in rs}
        sp_rows = [s for u in keep for s in spans.get(u, [])]
        if sp_rows:
            by_uid = {r["uid"]: r["file"] for r in written}
            for s in sp_rows:
                s["file"] = by_uid[s["uid"]]
            (out / "spans.jsonl").write_text(
                "".join(json.dumps(s, sort_keys=True) + "\n" for s in sp_rows),
                encoding="utf-8")

        n_clean = sum(1 for r in rs if not r["gold"])
        (out / "manifest.json").write_text(json.dumps({
            "dataset": name, "n_documents": len(rs), "split": split,
            "label_space": "GAIA catalog sensitive_data vertical (60 tags)",
            "task": "multi-label document classification",
            "n_clean_documents": n_clean,
            "clean_fraction": round(n_clean / max(len(rs), 1), 4),
            "n_distinct_tags": len(tags),
            "tag_counts": dict(sorted(tags.items())),
            "source": "Synthetic_PDF_Corpus_v2_1612",
            "split_rule": (f"stratified by vertical/document-type, "
                           f"{args.train_frac:.0%}/{1 - args.train_frac:.0%} positionally "
                           f"within each family over a sha256 ordering — deterministic, "
                           f"and every family of 2+ documents contributes to eval"),
        }, indent=1), encoding="utf-8")

        (out / "README.md").write_text(f"""# {name}

{len(rs):,} synthetic PDF documents ({split} split) for multi-label sensitive-data
classification against the 60-tag GAIA scorecard catalogue ({len(tags)} present here).

- **{n_clean:,}** documents ({n_clean / max(len(rs), 1):.1%}) carry **no** sensitive tag.
- Text is the corpus's own canonical extraction, one `.txt` per document.
- `spans.jsonl` carries character-offset entity gold ({len(sp_rows):,} spans) —
  only the second corpus here to have any.

## Provenance

Derived from `Synthetic_PDF_Corpus_v2_1612`, a controlled synthetic release of
1,612 PDFs (Finance / Healthcare / Government / cross-industry). All documents are
synthetic and deidentified; none is a valid record.

Split **stratified by vertical/document-type**, {args.train_frac:.0%}/{1 - args.train_frac:.0%},
assigned by `sha256(document_id)` so it is deterministic and a document never
crosses sides.

## Label space

The corpus's own `metadata/taxonomy.json` **is** the GAIA scorecard's 60 tags, so
labels map to this repository's slugs by rule rather than by a hand-written table.

## Caution

Before this split, the parent corpus was the only out-of-distribution test in this
repository. Any evaluation that treats
`data/3-holdout/Synthetic_PDF_Corpus_v2_1612` as unseen is invalid once this
training split is used — 80% of those documents are now training data.
""", encoding="utf-8")
        print(f"\n-> {out}  ({len(rs):,} docs, {len(sp_rows):,} spans)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
