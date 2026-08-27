# Model Card — `pii-content-v5d-fused-v1`

`cascade_scorecard61` fused per tag with a content tagger distilled from a fine-tuned transformer.

A CPU document tagger for sensitive data (PII / PHI / PCI) over **61 labels**.

> **This model did not pass its ship gate and is not a `@champion`.** It is
> packaged on explicit request as a measured artifact. Nothing in this
> repository's history has cleared a gate.

## ⚠️ Things to improve

**1.** **The content model earns very little and costs 56% more latency.** 6.2785 ms p95 against the cascade's 4.0290, for +0.0296 micro F1. If latency matters at all, take the cascade alone.

**2.** **It is a precision veto, not a second detector.** Across 61 tags the fusion chose `and` 37 times and `cascade` 24 times. It chose the content model alone **zero** times and `or` **zero** times — meaning the content model finds nothing the cascade misses. Do not describe this bundle as two models pooling their findings.

**3.** **It loses three priority recall gates** relative to the cascade it wraps: 22/55 against 25/55.

**4.** **Re-thresholding beats it outright.** `pii-cascade-p80r90-v1` reaches micro F1 0.8474 at 4.0290 ms with no transformer anywhere in it.

## Intended use and limits

**Intended.** Batch or streaming scan of text documents to decide whether they
carry sensitive data and which tags apply, on a CPU budget of one core.

**Not intended.**

- **It is not span NER.** It says a document contains an SSN; it does not say
  where. Nothing in it localises, highlights or redacts.
- **It is not a compliance decision.** A document it calls clean is not certified
  clean; document recall is 0.7975, so roughly one PII-bearing document in five is
  missed at the document level.
- **English only.** Every training corpus is English.
- **Not identity resolution.** It has no notion of a person across documents.

## Measured performance

Eight sealed `data/2-eval` corpora, 126,129 documents, equal-corpus aggregation,
one core.

| metric | value |
| --- | ---: |
| micro F1 | 0.7595 |
| micro precision | 0.6926 |
| macro F0.5 | 0.6282 |
| macro F2 | 0.6496 |
| recall, macro over catalogue | 0.6367 |
| precision, macro over catalogue | 0.6365 |
| worst measurable priority recall | 0.5881 |
| document recall | 0.7975 |
| document precision | 0.8956 |
| document specificity | 0.8792 |
| one-core p95 (ms) | 6.3519 |

**Which corpus flatters it.** `20000_pii_holdout_20.00k` returns micro F1
**0.8814** and `10360_betterdataai_ner_silver_eval_10.36k` returns
**0.5869**. The generous rows are synthetic; the weakest is
silver-labelled. **Quote the equal-corpus figure above**, never a single corpus.

The bundle's verification re-scores `20000_pii_holdout_20.00k` through this
bundle's own `tagger.py` and reproduces the evaluator to six decimal places.

**Where it is worse than the alternatives.** Worse than `pii-cascade-p80r90-v1` on the ranker (0.7595 vs 0.8474), worse than `cascade_scorecard61` on recall and gates, and slower than both.

The best micro F1 measured anywhere in this repo is **0.8474**
(`pii-cascade-p80r90-v1`). Compare against that before adopting this one.

## How it was made

Fine-tune ettin-68m on 8 corpora, distil to a static token table with model2vec, train a linear tagger on [mean‖max] token features, then choose a fusion rule per tag on the training calibration carve.

Full reproduction steps: `HOW_TO_BUILD.md`.

## Provenance

| | |
| --- | --- |
| Labels | 61 |
| Evaluator | `training/h2h_eval.py`, unchanged |
| Selection data | training calibration carve only; sealed corpora never used for selection |
| Promoted | **no** |
