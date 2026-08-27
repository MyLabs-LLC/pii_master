# Model Card — `pii-content-v5e-fused-v1`

`cascade_p80r70` fused per tag with the same content tagger. The highest micro precision measured in this repo — and still not the model to use.

A CPU document tagger for sensitive data (PII / PHI / PCI) over **61 labels**.

> **This model did not pass its ship gate and is not a `@champion`.** It is
> packaged on explicit request as a measured artifact. Nothing in this
> repository's history has cleared a gate.

## ⚠️ Things to improve

**1.** **Fusing made the model it wraps very slightly worse.** `cascade_p80r70` alone scores micro F1 0.8394; this scores 0.8373. The content arm buys +0.0025 micro precision and costs 0.019 macro recall and 59% more latency (6.4107 ms vs 4.0290).

**2.** **This is the experiment that closed the question.** Once thresholds already deliver precision, a precision veto has almost nothing left to remove. The two gains were never independent.

**3.** **Same fusion pattern as v5d**: `and` 29 tags, `cascade` 32, `content` zero, `or` zero.

**4.** **Lowest recall of any bundle here**: macro recall 0.5474, worst priority recall 0.4729.

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
| micro F1 | 0.8373 |
| micro precision | 0.9190 |
| macro F0.5 | 0.6954 |
| macro F2 | 0.6214 |
| recall, macro over catalogue | 0.5474 |
| precision, macro over catalogue | 0.7541 |
| worst measurable priority recall | 0.4729 |
| document recall | 0.7975 |
| document precision | 0.8956 |
| document specificity | 0.8792 |
| one-core p95 (ms) | 6.4107 |

**Which corpus flatters it.** `20000_pii_holdout_20.00k` returns micro F1
**0.9212** and `10626_ai4privacy_pii_masking_eval_10.63k` returns
**0.7323**. The generous rows are synthetic; the weakest is
silver-labelled. **Quote the equal-corpus figure above**, never a single corpus.

The bundle's verification re-scores `20000_pii_holdout_20.00k` through this
bundle's own `tagger.py` and reproduces the evaluator to six decimal places.

**Where it is worse than the alternatives.** Worse than `pii-cascade-p80r70-v1` — which it is built from — on micro F1, macro F0.5, macro F2, recall and latency. It is better only on micro precision, by 0.0025.

The best micro F1 measured anywhere in this repo is **0.8474**
(`pii-cascade-p80r90-v1`). Compare against that before adopting this one.

## How it was made

As v5d, but fused into `cascade_p80r70` instead of `cascade_scorecard61`.

Full reproduction steps: `HOW_TO_BUILD.md`.

## Provenance

| | |
| --- | --- |
| Labels | 61 |
| Evaluator | `training/h2h_eval.py`, unchanged |
| Selection data | training calibration carve only; sealed corpora never used for selection |
| Promoted | **no** |
