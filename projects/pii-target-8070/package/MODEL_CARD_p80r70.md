# Model Card — `pii-cascade-p80r70-v1`

A CPU document tagger for sensitive data (PII / PHI / PCI). It answers two
questions about a document: **does this contain sensitive data**, and **which of
61 tags does it carry**. NumPy only, 28 MB, **4.03 ms p95 per document on one
core**.

> **This model did not pass its ship gate, and it is not a `@champion`.**
> It is the best measured artifact of the `pii-target-8070` run, packaged on
> explicit request. Under the declared policy it is `blocked`: 141 of 180
> measurable tag × corpus pairs clear the 80% precision bar, and 44 of 55 clear
> the 70% recall bar. Section 1 says exactly where and why. Nothing in this
> repository's history has cleared a gate.

## ⚠️ Things to improve

**1. Recall was traded away for precision, and the trade is real.** Against the
model it was derived from (`cascade_scorecard61`):

| | derived from | this model |
| --- | ---: | ---: |
| macro recall over the catalogue | **0.6773** | 0.5668 |
| worst measurable priority recall | **0.6267** | 0.4830 |
| macro F2 (recall-weighted) | **0.6651** | 0.6406 |

In a domain where a missed identifier is a reportable incident, this is the
number to argue about. **If your use is "find everything, a reviewer will
filter", stay on `cascade_scorecard61`.** This model is for the opposite case:
the scanner's alarms are being acted on and false alarms are expensive.

**2. Seven tags cannot reach 80% precision at 70% recall at any threshold**, measured
on the calibration carve before this model was built: `username` (best 0.7860),
`county` (0.7385), `country_of_origin` (0.7143), `employment_status` (0.7080),
`nationality` (0.6951), `sexual_identity_and_orientation` (0.5952),
`swift_code` (0.4417). Treat their output as a hint, not a finding.

**3. Two evaluation corpora cap the result and their gold is the reason.** Of 39
per-tag precision failures, an audit found **16 are corpus-capped** — the tag
reaches ≥0.80 precision on other corpora and cannot here — and 15 more are
tags the corpus barely annotates. `email` reaches 1.0000 precision everywhere
except `betterdataai`, where it ceilings at **0.5779** at any threshold. Only
about 8 of 39 failures are attributable to the model.

**4. The document gate is unchanged and unimproved.** Document recall 0.7975,
precision 0.8956, specificity 0.8792 — identical to the model this was derived
from, because only tag thresholds moved. If your problem is the document
question, this model changes nothing for you.

**5. `pii-cascade-p80r90-v1` is the better model on the declared ranker.** That
sibling — same method, recall bar kept at 0.90 — scores micro F1 **0.8474** against
this model's 0.8394, and gives up less recall (macro 0.5872 vs 0.5668). This model
wins only on precision (0.9165 vs 0.8844) and by two precision pairs. Prefer this
one only if "80% precision" is a hard external commitment; otherwise take the
sibling.

**6. Thresholds were selected on training data, not tuned per deployment.** Your
corpus is not one of the eight here. Expect drift, and re-select on your own
calibration slice if you can.

## Intended use and limits

**Intended.** Batch or streaming scan of text documents to decide whether they
carry sensitive data and to route them. Built for a CPU budget: 1 core, 4 GB,
p95 under 8 ms.

**Not intended.**

- **It is not span NER.** It says a document contains an SSN; it does not say
  where. Nothing in it localises, highlights or redacts. If you need offsets, this
  is the wrong artifact and no amount of post-processing makes it right.
- **It is not a compliance decision.** A document this model calls clean is not
  certified clean. Document recall is 0.7975 — about one in five PII-bearing
  documents is missed at the document level.
- **English only.** Every training corpus is English.
- **Not a person-level or record-level judgement.** It has no notion of identity
  resolution across documents.

## Measured performance

Eight sealed `data/2-eval` corpora, 126,129 documents, equal-corpus aggregation,
one core.

| metric | value |
| --- | ---: |
| **micro F1** | **0.8394** |
| micro precision | **0.9165** |
| macro F0.5 | 0.7093 |
| macro F2 | 0.6406 |
| precision, macro over catalogue | 0.7616 |
| recall, macro over catalogue | 0.5668 |
| document recall / precision / specificity | 0.7975 / 0.8956 / 0.8792 |
| **one-core p95** | **4.0290 ms** |
| peak RSS | ~28 MB of weights |

**Which corpus flatters it, and what to quote instead.** `20000_pii_holdout`
returns micro F1 **0.9241** and `38937_openpii` is similarly generous; both are
synthetic. `10360_betterdataai` returns **0.5390** and is silver-labelled. **Quote
the equal-corpus figure, 0.8394.** A single-corpus number from this suite can be
made to say almost anything.

The bundle's own verification re-scores `20000_pii_holdout` through
`tagger.py` and reproduces the evaluator's micro F1 to six decimal places
(0.924094 vs 0.924100); the residual is float16 weight storage.

**Where it is worse than what it replaces.** See ⚠️ item 1. Recall, on every
recall-weighted measure.

## How it was made

Thresholds only. The gate weights, gate threshold, 61 head weight vectors,
feature hashing and read window are `cascade_scorecard61`'s, byte-identical. For
each tag, the precision-recall curve on the training calibration carve was swept
and the F0.5-optimal point **inside the box (P ≥ 0.80, R ≥ 0.70)** was taken; where
the curve never enters the box, the tag keeps its best F0.5 point and is recorded
as unreachable.

No training, no new data, no change to the evaluator, and no change in latency —
moving a threshold changes which comparisons come out true, not how many are done.

Full reproduction steps: `docs/HOW_TO_BUILD.md`.

## Provenance

| | |
| --- | --- |
| Derived from | `cascade_scorecard61` (`projects/pii-scorecard-60`) |
| Catalogue | 61 labels — the GAIA scorecard's 60, less `routing_number`, plus `swift_code` |
| Selection data | training calibration carve (`quiet_fit.carve_holdin`), sealed corpora never used for selection |
| Evaluator | `training/h2h_eval.py`, unchanged |
| Run report | `docs/26-08-26_target-80-70.md` |
| Promoted | **no** |
