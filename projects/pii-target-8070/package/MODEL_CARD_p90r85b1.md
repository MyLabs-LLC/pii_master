# Model Card — `pii-cascade-p90r85b1-v1`

A CPU document tagger for sensitive data (PII / PHI / PCI) over **61 labels**.
NumPy only, 28 MB, **4.03 ms p95 per document on one core**.

**This is the generalisation-first model of the set.** It is the best of seven
measured models on out-of-distribution documents, and the best on every summary
metric except precision. It deliberately does **not** meet the 90% precision
target that `pii-cascade-p88r90-v1` was built to hit.

| | this model | `p88r90` (target-hitting) |
| --- | ---: | ---: |
| micro F1 (8 sealed corpora) | **0.8564** | 0.8470 |
| micro precision | 0.8721 | **0.9000** |
| micro recall | **0.8415** | 0.8020 |
| macro F2 | **0.6865** | 0.6480 |
| macro recall | **0.6367** | 0.5802 |
| **out-of-distribution micro F1** | **0.5453** | 0.4992 |

> **It did not pass the project's per-tag ship gate and is not a `@champion`.**
> Nothing in this repository's history has.

## ⚠️ Things to improve

**1. It does not reach 90% precision, and that is the deliberate trade.** Micro
precision is 0.8721. If you have an external commitment to a 0.90 figure, take
`pii-cascade-p88r90-v1` instead and accept the generalisation cost in item 2.

**2. Out-of-distribution performance is far below the headline — for every model
here, including this one.** Scored on 1,612 synthetic PDFs
(`data/3-holdout/Synthetic_PDF_Corpus_v2_1612`) that no training corpus resembles:

| | 8 sealed corpora | holdout |
| --- | ---: | ---: |
| micro F1 | 0.8564 | **0.5453** |
| micro precision | 0.8721 | 0.7156 |
| micro recall | 0.8415 | **0.4405** |

**Precision transfers; recall does not.** Across all seven models built in this
project, holdout precision varies only 0.6606–0.7156 while holdout F1 ranges
0.4992–0.5836, tracking recall almost exactly (Pearson r = **+0.993** between
in-distribution recall and holdout F1; **−0.869** for precision). Do not quote the
0.8564 as an expected figure for a corpus unlike the training data.

**3. Whole tag families do not fire on unfamiliar document layouts.** On the
holdout, tags with substantial gold that never fire at all include
`medical_condition` (611 documents), `patient_id_number` (611),
`medical_treatment` (611), `medication` (561), `county` (610), `icd_10` (120).
The PHI clinical family in particular appears to be learnable only in the exact
rendering the training corpora use. **If your documents are clinical PDFs, measure
before deploying.**

**4. The precision claim is not conclusive even in-distribution.** The suite
aggregates equal-corpus over only five precision-measurable corpora, so the
corpus-level bootstrap gives precision **[0.8066, 0.9394]** and recall
**[0.7997, 0.9201]** — wide. A Wilson interval over pooled firings is far tighter
([0.9201, 0.9213]) but conditions on the firings made and ignores that corpora,
not firings, are what differ between deployments.

**5. Twenty of 61 tags could not reach the target box at any threshold** and keep
their best-F1 operating point instead; they are recorded as `unreachable` in
`docs/`. Their output is a hint, not a finding.

**6. Two evaluation corpora cap what any threshold can achieve.** An audit found
~31 of 39 per-tag precision failures are gold defects rather than model weakness —
`email` reaches 1.0000 precision on four corpora and ceilings at **0.5779** on
`betterdataai` at any threshold.

## Intended use and limits

**Intended.** Batch or streaming scan of text documents to decide whether they
carry sensitive data and which tags apply, on one CPU core and 4 GB — **preferred
where the document population is not well characterised in advance**, because it
is the most robust of the models measured here.

**Not intended.**

- **It is not span NER.** It says a document contains an SSN; it does not say
  where. Nothing in it localises, highlights or redacts.
- **It is not a compliance decision.** A document it calls clean is not certified
  clean; document recall is 0.7975 in-distribution and materially worse on
  unfamiliar layouts.
- **English only.** Every training corpus is English.
- **Not identity resolution.** It has no notion of a person across documents.

## Measured performance

Eight sealed `data/2-eval` corpora, 126,129 documents, equal-corpus aggregation,
one core.

| metric | value |
| --- | ---: |
| **micro F1** | **0.8564** |
| micro precision | 0.8721 |
| micro recall | 0.8415 |
| macro F0.5 | 0.7028 |
| macro F2 | 0.6865 |
| precision, macro over catalogue | 0.7218 |
| recall, macro over catalogue | 0.6367 |
| worst measurable priority recall | 0.5398 |
| document recall / precision / specificity | 0.7975 / 0.8956 / 0.8792 |
| **one-core p95** | **4.0290 ms** |

**Which corpus flatters it.** `20000_pii_holdout` and `38937_openpii` are
synthetic and generous; `10360_betterdataai` is silver-labelled — its labels are
themselves model output. **Quote the equal-corpus figure, 0.8564**, and read item
2 before assuming it holds on your data.

The bundle's verification re-scores `20000_pii_holdout_20.00k` through this
bundle's own `tagger.py` and reproduces the fixed evaluator to five decimal
places (0.935519 vs 0.935534); the residual is float16 weight storage.

**Where it is worse than the alternatives.** Precision, against
`pii-cascade-p88r90-v1` (0.8721 vs 0.9000) and `pii-cascade-p80r70-v1` (0.9165).
Against the untuned `cascade_scorecard61` it is worse on the holdout (0.5453 vs
0.5836) and on macro recall (0.6367 vs 0.6773) — **if maximum recall is what you
need, the untuned baseline is still the better model.**

## How it was made

**Thresholds only. Nothing was trained.** Gate weights, gate threshold, 61 head
weight vectors, feature hashing and read window are `cascade_scorecard61`'s,
byte-identical.

For each tag, the precision-recall curve on the *training* calibration carve was
swept and the **F1-optimal** point inside the box **(P ≥ 0.90, R ≥ 0.85)** was
taken. The `beta = 1.0` choice is what distinguishes this model from its siblings,
which use F0.5: it is the one axis a 66-configuration sweep found to matter, and
it buys recall at the cost of precision.

Latency is unchanged from the parent by construction — all 61 head scores are
computed unconditionally, so moving a threshold changes which comparisons come out
true, not how many are done.

Full reproduction steps: `HOW_TO_BUILD.md`.

## Provenance

| | |
| --- | --- |
| Derived from | `cascade_scorecard61` (`projects/pii-scorecard-60`) |
| Selection | box P ≥ 0.90, R ≥ 0.85, F1-optimal inside the box |
| Catalogue | 61 labels — the GAIA scorecard's 60, less `routing_number`, plus `swift_code` |
| Selection data | training calibration carve; sealed corpora scored once; holdout scored after selection |
| Evaluator | `training/h2h_eval.py`, unchanged |
| Promoted | **no** |
