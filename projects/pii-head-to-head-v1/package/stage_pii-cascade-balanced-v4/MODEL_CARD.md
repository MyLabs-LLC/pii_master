# Model Card — `pii-cascade-balanced-v4`

A 58-label cascade whose thresholds were re-derived honestly on training data, correcting a predecessor tuned on the evaluation set.

A CPU document tagger for sensitive data (PII / PHI / PCI) over **58 labels**.

> **This model did not pass its ship gate and is not a `@champion`.** It is
> packaged on explicit request as a measured artifact. Nothing in this
> repository's history has cleared a gate.

## ⚠️ Things to improve

**1.** **58 labels, not 61.** It collapses `given_name`, `family_name` and `middle_name` into `full_name`, and `street_number_and_name` into `address`. If you need those distinctions this model cannot make them, and its numbers are not comparable with the 61-label bundles.

**2.** **It loses four priority recall gates relative to its own predecessor** — 25/55 against 29/55 — because correcting the thresholds raised them. That was the honest cost of removing an evaluation-set-tuned shortcut.

**3.** **Its worst measurable priority recall is 0.6410.** One priority tag×corpus pair sits there; the aggregate does not tell you which.

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
| micro F1 | 0.7862 |
| micro precision | 0.7188 |
| macro F0.5 | 0.6403 |
| macro F2 | 0.6641 |
| recall, macro over catalogue | 0.6655 |
| precision, macro over catalogue | 0.6461 |
| worst measurable priority recall | 0.6410 |
| document recall | 0.7975 |
| document precision | 0.8956 |
| document specificity | 0.8792 |
| one-core p95 (ms) | 3.9696 |

**Which corpus flatters it.** `38937_openpii_pii_eval_38.94k` returns micro F1
**0.8843** and `10360_betterdataai_ner_silver_eval_10.36k` returns
**0.6973**. The generous rows are synthetic; the weakest is
silver-labelled. **Quote the equal-corpus figure above**, never a single corpus.

The bundle's verification re-scores `20000_pii_holdout_20.00k` through this
bundle's own `tagger.py` and reproduces the evaluator to six decimal places.

**Where it is worse than the alternatives.** `pii-cascade-balanced-v3` has better recall (worst priority recall 0.7221 vs 0.6410) — but v3's advantage comes from two thresholds selected on the sealed evaluation set, so it is not an advantage you can trust. That is the whole reason this model exists.

The best micro F1 measured anywhere in this repo is **0.8474**
(`pii-cascade-p80r90-v1`). Compare against that before adopting this one.

## How it was made

Thresholds re-selected on the training calibration carve with a corrected group-recall cap: a source group may only set the cap if at least 10 of its positives fall at or below the candidate cut. Five of 58 thresholds moved.

Full reproduction steps: `HOW_TO_BUILD.md`.

## Provenance

| | |
| --- | --- |
| Labels | 58 |
| Evaluator | `training/h2h_eval.py`, unchanged |
| Selection data | training calibration carve only; sealed corpora never used for selection |
| Promoted | **no** |
