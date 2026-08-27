# Every model flags at least 15% of documents that contain no PII

## Results

7,126 **real files** in the formats a deployment actually meets — 3,296 PDF,
2,657 DOC, 1,559 HTML, 771 XLS, 587 PPT, plus CSV/XML/RTF — each judged to
contain no personal data. Every corpus in the sealed suite is synthetic text or
extracted plain text; this is the first measurement here on real documents.

| model | doc false-alarm rate | false positives | FP / 1,000 docs | implied precision @ 90% clean |
| --- | ---: | ---: | ---: | ---: |
| `cascade_scorecard61` | 0.1911 | 6,134 | 860.8 | 0.3468 |
| `cascade_v2_9corp` | 0.1889 | 6,051 | 849.1 | — |
| `cascade_p88r90b1` | 0.1723 | 3,698 | 518.9 | 0.5036 |
| `cascade_p90r85b1` | 0.1719 | 3,653 | 512.6 | 0.5065 |
| `v2_p90r85b1` | 0.1697 | 3,652 | 512.5 | — |
| `cascade_p80r90` | 0.1584 | 3,095 | 434.3 | 0.5426 |
| `cascade_p88r90` | 0.1575 | 3,022 | 424.1 | 0.5481 |
| `cascade_p90r90` | 0.1566 | 2,932 | 411.5 | 0.5549 |
| **`cascade_p80r70`** | **0.1506** | **2,775** | **389.4** | **0.5662** |

### What an all-negative corpus can and cannot measure

It **cannot** measure precision. Precision is TP/(TP+FP) and a corpus with no
positives supplies no TP; the ratio is undefined, not zero. An earlier note in
this project claimed this corpus would "measure precision" and that was wrong.

What it measures exactly, with no annotation ambiguity anywhere, is the
false-positive side: every tag firing here is a false positive, because there is
nothing here to correctly detect. The `implied precision` column composes that
with the sealed suite's true-positive counts at an assumed 90%-clean document
estate — the sealed suite is roughly half PII-bearing, and precision degrades
with the mixture.

### The threshold work does transfer — for what it was built to do

Baseline to best variant: **860.8 → 389.4 false positives per 1,000 documents**,
a 55% reduction, and doc false-alarm rate 0.1911 → 0.1506.

This is the first evidence in this repository that the precision tuning buys
something outside its own suite, and it partly corrects the reading in
`26-08-27_evaluation-matrix.md`. That report found precision tuning to be the
generalisation-worst direction — true, but *of micro F1*, which is dominated by
recall. The false-positive reduction the tuning was actually built for survives
transfer to real documents intact.

### The absolute numbers remain the problem

Even the best model flags **15.06% of clean files**. At a realistic 90%-clean
estate the implied precision falls to **0.35–0.57**, against sealed figures of
0.64–0.92. Nothing here is close to a 90% precision claim on a real corpus.

Where it comes from — `cascade_p88r90`, 3,022 FP across 35 tags:

| tag | documents | rate |
| --- | ---: | ---: |
| `full_name` | 762 | 10.69% |
| `phone_number` | 420 | 5.89% |
| `email` | 353 | 4.95% |
| `zip_code` | 293 | 4.11% |
| `state` | 279 | 3.92% |
| `address` | 227 | 3.19% |

The baseline fires the same tags roughly twice as often and adds
`family_name` (590) and `given_name` (496), which the tuned models largely
silence.

## TL;DR

- **7,126 real documents with no PII; every model flags at least 15% of them.**
- **Threshold tuning halves the false-alarm burden**: 860.8 → 389.4 FP per 1,000
  documents. The precision work transfers; the earlier "precision is
  generalisation-worst" finding was about micro F1, not false positives.
- **Implied precision on a 90%-clean estate is 0.35–0.57**, not the sealed
  0.64–0.92. The sealed suite's mixture flatters every precision number here.
- **`full_name` alone fires on 10.7% of clean documents** and is the single
  largest contributor for every model.
- **Precision is not measurable on this corpus** — no positives, no TP. It
  measures the false-positive side only, and an earlier claim otherwise is
  corrected here.
- **A six-document audit found 1 of 6 `full_name` firings carried a real person
  name** the judge declined to count. Treat the counts as a modest over-estimate.

---

| | |
| --- | --- |
| Date | 2026-08-27 |
| Project | `projects/pii-target-8070` |
| Request | *"do both"* — clean-document false alarms, and variant LOCO |
| Scope | 11 models × 7,126 real documents |
| CPU budget | 1 core for scoring; 8 workers for extraction only |
| Artifacts | `evaluations/clean_docs.json`, `training/h2h_cleandocs_score.py` |
| Outcome | false-alarm burden measured; **no promotion** |

## Admissibility: why 7,126 and not 10,000

Inherited from `h2h_cleandocs.admissible()` rather than re-derived, so the two
runs cannot drift:

| | count |
| --- | ---: |
| documents in the manifest | 10,000 |
| overlap with `govdocs2` / `datax` corpora | 2,874 |
| …of which sit in a **sealed** eval split | 587 |
| …of which carry gold contradicting this manifest | 782 |
| **usable — no existing gold at all** | **7,126** |

The 587 are disqualifying on their own: scoring documents that sit in a sealed
split would leak the measurement. The 782 conflicts are two independent
labelling passes disagreeing about the same files, which is a caution about the
gold generally, not only about those documents.

## What this does not measure

**Recall.** There are no positives here, so nothing says whether the tuned
models still find what they should on real documents. The `p80r70` model has the
lowest false-alarm rate *and* the lowest recall of the tuned set on the sealed
suite; on this corpus alone it looks best, and that ranking is one-sided by
construction. Pair it with the LOCO recall figures before drawing a conclusion.

## What is still open

1. **A real-document corpus with positives.** This corpus fixes the negative
   half only. Judging a few hundred real PII-bearing documents would close the
   loop and give the first honest precision number on real formats.
2. **`full_name` is the dominant failure and may be partly a taxonomy question.**
   Press contacts and named agency officials appear throughout government
   documents. Whether those count is a policy decision nobody has recorded.
3. **The 782-document gold conflict is unexplained** and touches the same
   dual-judge process the `govdocs2` and `datax` corpora rely on.
