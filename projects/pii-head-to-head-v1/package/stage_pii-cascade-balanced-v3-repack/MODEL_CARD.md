# Model Card — `pii-cascade-balanced-v3-repack`

A 58-label cascade. **Two of its thresholds were selected on the sealed evaluation corpora.** Repackaged here for completeness, with that defect stated.

A CPU document tagger for sensitive data (PII / PHI / PCI) over **58 labels**.

> **This model did not pass its ship gate and is not a `@champion`.** It is
> packaged on explicit request as a measured artifact. Nothing in this
> repository's history has cleared a gate.

## ⚠️ Things to improve

**1.** **Two thresholds were tuned on the evaluation set, and the script that did it was never checked in.** `military_identification_number` and `sexual_identity_and_orientation` were swept on a held-in half of the same eight corpora this model is scored on. Their numbers here are selection results, not measurements, and the split cannot be reproduced.

**2.** **Its apparent advantage over `pii-cascade-balanced-v4` is that defect.** v3 keeps both `military_identification_number` gates that v4 loses. An honest re-derivation on training data chose a threshold 2.5 points higher and lost them. **Prefer v4.**

**3.** **58 labels**, with the name and address taxonomy collapsed — see the v4 card.

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
| micro F1 | 0.7533 |
| micro precision | 0.6634 |
| macro F0.5 | 0.6253 |
| macro F2 | 0.6657 |
| recall, macro over catalogue | 0.6887 |
| precision, macro over catalogue | 0.6258 |
| worst measurable priority recall | 0.7221 |
| document recall | 0.7975 |
| document precision | 0.8956 |
| document specificity | 0.8792 |
| one-core p95 (ms) | 3.9164 |

**Which corpus flatters it.** `38937_openpii_pii_eval_38.94k` returns micro F1
**0.8713** and `10360_betterdataai_ner_silver_eval_10.36k` returns
**0.6461**. The generous rows are synthetic; the weakest is
silver-labelled. **Quote the equal-corpus figure above**, never a single corpus.

The bundle's verification re-scores `20000_pii_holdout_20.00k` through this
bundle's own `tagger.py` and reproduces the evaluator to six decimal places.

**Where it is worse than the alternatives.** Compared with `pii-cascade-balanced-v4`, this model is not better in any way you can defend; where it looks better it is because it saw the test.

The best micro F1 measured anywhere in this repo is **0.8474**
(`pii-cascade-p80r90-v1`). Compare against that before adopting this one.

## How it was made

Derived from `pii-cascade-balanced-v2` by hand-editing two tag thresholds after a sweep on evaluation data. Documented here as an anti-pattern.

Full reproduction steps: `HOW_TO_BUILD.md`.

## Provenance

| | |
| --- | --- |
| Labels | 58 |
| Evaluator | `training/h2h_eval.py`, unchanged |
| Selection data | training calibration carve only; sealed corpora never used for selection |
| Promoted | **no** |
