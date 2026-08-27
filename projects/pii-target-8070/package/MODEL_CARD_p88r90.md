# Model Card — `pii-cascade-p88r90-v1`

A CPU document tagger for sensitive data (PII / PHI / PCI) over **61 labels**.
NumPy only, 28 MB, **4.03 ms p95 per document on one core**.

Built to a stated target — micro precision ≥ 0.90, micro recall ≥ 0.80, micro
F1 ≥ 0.80 — and it meets all three on the point estimate:

| | measured | target |
| --- | ---: | ---: |
| micro precision | **0.9000** | ≥ 0.90 |
| micro recall | **0.8020** | ≥ 0.80 |
| micro F1 | **0.8470** | ≥ 0.80 |

> **Read ⚠️ item 1 before quoting any of that.** Two of the three land on the bar
> to three decimal places, and the confidence interval does not clear it.
>
> **This model did not pass the project's per-tag ship gate and is not a
> `@champion`.** Nothing in this repository's history has.

## ⚠️ Things to improve

**1. The headline meets the target but does not clear it conclusively.** Micro
precision is 0.9000 against a bar of 0.90 — a margin of zero. The suite
aggregates equal-corpus, so the sampling unit is the **corpus**, and there are
only five that can measure precision. Bootstrapping those five gives:

| | point | 95% interval |
| --- | ---: | --- |
| micro precision | 0.9000 | **[0.8448, 0.9517]** |
| micro recall | 0.8020 | **[0.7495, 0.8997]** |

**Neither bound clears its bar.** A Wilson interval over pooled firings is much
tighter ([0.9344, 0.9356] on a pooled precision of 0.9350) and would say the
target is comfortably met — but Wilson conditions on the predictions made and
ignores that corpora, not firings, are what vary between deployments. **Treat
this model as "meets the target on this suite", not "clears 0.90".** If you need
a conclusive claim, the honest route is more precision-measurable corpora, not a
different threshold.

**2. Pooled and equal-corpus numbers differ a lot, and the flattering one is
wrong.** Pooled across documents this model reads P 0.9350 / R 0.8804, because
the two largest corpora are synthetic and easy. The equal-corpus figures in the
table are the ones to quote.

**3. Recall is well below the model it was derived from.** Against
`cascade_scorecard61` (same weights, different thresholds):

| | scorecard61 | this model |
| --- | ---: | ---: |
| macro recall over the catalogue | **0.6773** | 0.5802 |
| worst measurable priority recall | **0.6267** | 0.4799 |

In a domain where a missed identifier is a reportable incident this is the cost
to argue about. **If your use is "find everything, a human filters", stay on
`cascade_scorecard61`.**

**4. Twenty-six of 61 tags could not reach the box at any threshold** and keep
their best-F0.5 operating point instead. They are recorded as `unreachable` in
`docs/`. Their output is a hint, not a finding.

**5. Two evaluation corpora cap what any threshold can do.** An audit of the
per-tag precision failures found ~31 of 39 are gold defects rather than model
weakness — `email` reaches 1.0000 precision on four corpora and ceilings at
**0.5779** on `betterdataai` at any threshold. Improving the model will not move
those.

## Intended use and limits

**Intended.** Batch or streaming scan of text documents to decide whether they
carry sensitive data and which tags apply, on a budget of one CPU core and 4 GB.

**Not intended.**

- **It is not span NER.** It says a document contains an SSN; it does not say
  where. Nothing in it localises, highlights or redacts.
- **It is not a compliance decision.** A document it calls clean is not certified
  clean — document recall is 0.7975, so roughly one PII-bearing document in five
  is missed at the document level.
- **English only.** Every training corpus is English.
- **Not identity resolution.** It has no notion of a person across documents.

## Measured performance

Eight sealed `data/2-eval` corpora, 126,129 documents, equal-corpus aggregation,
one core.

| metric | value |
| --- | ---: |
| **micro F1** | **0.8470** |
| **micro precision** | **0.9000** |
| **micro recall** | **0.8020** |
| macro F0.5 | 0.7035 |
| macro F2 | 0.6480 |
| precision, macro over catalogue | 0.7530 |
| recall, macro over catalogue | 0.5802 |
| worst measurable priority recall | 0.4799 |
| document recall / precision / specificity | 0.7975 / 0.8956 / 0.8792 |
| **one-core p95** | **4.0290 ms** |

**Which corpus flatters it.** `20000_pii_holdout` and `38937_openpii` are
synthetic and generous; `10360_betterdataai` is silver-labelled — its labels are
themselves model output — and is the weakest row by a wide margin. **Quote the
equal-corpus figures above**, never a single corpus.

The bundle's verification re-scores `20000_pii_holdout_20.00k` through this
bundle's own `tagger.py` and reproduces the fixed evaluator to six decimal
places; the residual is float16 weight storage.

**Where it is worse than the alternatives.** Recall, against
`cascade_scorecard61` — see ⚠️ item 3. Against `pii-cascade-p80r90-v1` it trades
0.0004 micro F1 for +0.0156 micro precision; if you do not need the 0.90
precision figure specifically, that model is a slightly better all-rounder.

## How it was made

**Thresholds only. Nothing was trained.** The gate weights, gate threshold, 61
head weight vectors, feature hashing and read window are `cascade_scorecard61`'s,
byte-identical. For each tag the precision-recall curve on the *training*
calibration carve was swept and the F0.5-optimal point inside the box
**(P ≥ 0.88, R ≥ 0.90)** was taken; where the curve never enters the box, the tag
keeps its best F0.5 point and is recorded as unreachable.

The box parameter itself was chosen by sweeping 25 candidates **on calibration
only**, then correcting for the measured calibration→sealed gap (precision falls
~0.04, recall ~0.09 between the two). Two candidates survived that correction and
both were scored sealed once. This one met the target; `cascade_p90r90` did not
(P 0.9115, R 0.7926).

Latency is unchanged from the parent by construction: all 61 head scores are
computed unconditionally, so moving a threshold changes which comparisons come out
true, not how many are done.

Full reproduction steps: `HOW_TO_BUILD.md`.

## Provenance

| | |
| --- | --- |
| Derived from | `cascade_scorecard61` (`projects/pii-scorecard-60`) |
| Catalogue | 61 labels — the GAIA scorecard's 60, less `routing_number`, plus `swift_code` |
| Selection data | training calibration carve; sealed corpora scored once, at the end |
| Evaluator | `training/h2h_eval.py`, unchanged |
| Promoted | **no** |
