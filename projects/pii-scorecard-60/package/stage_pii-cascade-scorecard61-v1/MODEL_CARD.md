# Model Card — `pii-cascade-scorecard61-v1`

The 61-label baseline: the first model in this repo built on the GAIA scorecard taxonomy, before any precision work.

A CPU document tagger for sensitive data (PII / PHI / PCI) over **61 labels**.

> **This model did not pass its ship gate and is not a `@champion`.** It is
> packaged on explicit request as a measured artifact. Nothing in this
> repository's history has cleared a gate.

## ⚠️ Things to improve

**1.** **Its precision is poor and that is fixable without retraining.** Micro precision is 0.6360. Two sibling models built from these exact weights — only the 61 thresholds differ — reach 0.8844 and 0.9165. **If you are choosing a model today, choose one of those.** This one is packaged as the reference point they are measured against, not as a recommendation.

**2.** **Several tags fire almost indiscriminately.** `sexual_identity_and_orientation` produces 42,131 false positives against 173 true positives across the sealed suite; `geolocation` 19,726 against 226; `religion` 17,071 against 293. The threshold rule that selected them recorded these as successes because it had no precision floor.

**3.** **Only 25 of 55 measurable priority tag×corpus pairs clear 0.90 recall.** It does not pass its own declared gate.

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
| micro F1 | 0.7299 |
| micro precision | 0.6360 |
| macro F0.5 | 0.6230 |
| macro F2 | 0.6651 |
| recall, macro over catalogue | 0.6773 |
| precision, macro over catalogue | 0.6244 |
| worst measurable priority recall | 0.6267 |
| document recall | 0.7975 |
| document precision | 0.8956 |
| document specificity | 0.8792 |
| one-core p95 (ms) | 4.0290 |

**Which corpus flatters it.** `38937_openpii_pii_eval_38.94k` returns micro F1
**0.8715** and `10360_betterdataai_ner_silver_eval_10.36k` returns
**0.5390**. The generous rows are synthetic; the weakest is
silver-labelled. **Quote the equal-corpus figure above**, never a single corpus.

The bundle's verification re-scores `20000_pii_holdout_20.00k` through this
bundle's own `tagger.py` and reproduces the evaluator to six decimal places.

**Where it is worse than the alternatives.** Nothing this bundle does is better than `pii-cascade-p80r90-v1` except recall: macro recall 0.6773 vs 0.5872, worst priority recall 0.6267 vs 0.4921. If recall is what you need, that difference is the reason to take this one.

The best micro F1 measured anywhere in this repo is **0.8474**
(`pii-cascade-p80r90-v1`). Compare against that before adopting this one.

## How it was made

Refit from `cascade_balanced`'s hyperparameters onto the 61-label scorecard catalogue: gate refit balanced and regularised, 61 heads refit, thresholds selected by the group-recall-cap rule with the estimability correction.

Full reproduction steps: `HOW_TO_BUILD.md`.

## Provenance

| | |
| --- | --- |
| Labels | 61 |
| Evaluator | `training/h2h_eval.py`, unchanged |
| Selection data | training calibration carve only; sealed corpora never used for selection |
| Promoted | **no** |
