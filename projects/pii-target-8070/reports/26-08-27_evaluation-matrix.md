# Eleven models, nine corpora: the sealed suite ranks them backwards

## Results

99 measurements — every model in this repository against every evaluation corpus,
one fixed evaluator, one core.

**Eight corpora are the sealed suite** that decides gates. **The ninth,
`Synthetic_PDF_Corpus_v2_1612`, is an out-of-distribution holdout**: 1,612
synthetic PDFs no training corpus resembles, held outside `data/2-eval` and scored
only after every model was already selected.

| model | labels | sealed F1 | sealed P | sealed R | **holdout F1** | holdout P | holdout R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cascade_balanced_v4` | *58* | *0.7862* | *0.7188* | *0.8745* | ***0.6574*** | *0.8361* | *0.5416* |
| `cascade_balanced_v3` | *58* | *0.7533* | *0.6634* | *0.8856* | *0.6360* | *0.7545* | *0.5497* |
| `cascade_scorecard61` | 61 | 0.7299 | 0.6360 | **0.8866** | **0.5836** | 0.6606 | **0.5227** |
| `cascade_p90r85b1` | 61 | **0.8564** | 0.8721 | 0.8415 | 0.5453 | 0.7156 | 0.4405 |
| `cascade_p88r90b1` | 61 | 0.8557 | 0.8634 | 0.8483 | 0.5427 | 0.7100 | 0.4392 |
| `v5d_fused` | 61 | 0.7595 | 0.6926 | 0.8592 | 0.5296 | 0.6938 | 0.4282 |
| `cascade_p80r90` | 61 | 0.8474 | 0.8844 | 0.8141 | 0.5167 | 0.7108 | 0.4059 |
| `cascade_p80r70` | 61 | 0.8394 | 0.9165 | 0.7777 | 0.5034 | 0.7215 | 0.3866 |
| `cascade_p90r90` | 61 | 0.8464 | 0.9115 | 0.7926 | 0.5006 | 0.7156 | 0.3849 |
| `cascade_p88r90` | 61 | 0.8470 | **0.9000** | 0.8020 | 0.4992 | 0.7061 | 0.3861 |
| `v5e_fused` | 61 | 0.8373 | **0.9190** | 0.7726 | **0.4873** | 0.7308 | 0.3655 |

Sealed figures are equal-corpus over the eight. **The two 58-label rows are in
italics because they are not comparable** — see below.

### The sealed suite ranks the 61-label models almost exactly backwards

Across the nine models that share the 61-label catalogue:

| measured on the sealed 8 | Pearson r with **holdout** micro F1 |
| --- | ---: |
| micro **recall** | **+0.9359** |
| macro recall | **+0.9656** |
| micro F1 | **−0.6024** |
| micro **precision** | **−0.7793** |

The best sealed model (`p90r85b1`, F1 0.8564) is fourth on the holdout. The best
sealed precision (`v5e_fused`, 0.9190) is **last**. The untuned baseline — worst
sealed F1 of the tuned set at 0.7299 — is **best** of the 61-label models
out-of-distribution.

**Precision transfers; recall does not.** Holdout precision spans 0.6606–0.7308
across every 61-label model, a range of 0.07. Holdout F1 spans 0.4873–0.5836 and
tracks recall.

### Every model loses roughly a third on unfamiliar documents

| | best sealed | best holdout | drop |
| --- | ---: | ---: | ---: |
| 61-label micro F1 | 0.8564 | 0.5836 | **−0.27** |
| 58-label micro F1 | 0.7862 | 0.6574 | −0.13 |

No model in this repository has been shown to work on documents unlike its
training data. That is the single most important line in this report.

## TL;DR

- **99 measurements**, 11 models × 9 corpora, one evaluator, recorded in
  `26-08-27_Experiment-Log.xlsx` (99 Experiment Log rows, 5,985 per-tag rows).
- **The sealed suite is anti-correlated with generalisation.** Optimising sealed
  micro F1 (r = −0.60) or precision (r = −0.78) predicts *worse* out-of-distribution
  performance; recall predicts it almost perfectly (r = +0.97 on macro recall).
- **The untuned `cascade_scorecard61` is the best 61-label model on the holdout**
  (0.5836), despite being the weakest tuned model on the sealed suite.
- **`v5e_fused` has the highest sealed precision (0.9190) and the worst holdout
  score (0.4873).** The precision-first direction is the generalisation-worst
  direction, consistently, across all nine.
- **The 58-label models look best on the holdout and the comparison is unfair** —
  they are scored on a collapsed label space that merges the name subtypes, which
  is a materially easier task. Do not read that as evidence for the old taxonomy.
- **Everything drops ~0.27 micro F1** moving from the sealed suite to unfamiliar
  documents.

---

| | |
| --- | --- |
| Date | 2026-08-27 |
| Project | `projects/pii-target-8070` |
| Run | `26-08-27-1` |
| Request | *"do an eval run from all these models over all 9 eval datasets"* |
| Scope | 11 models × 9 corpora = 99 measurements |
| CPU budget | 1 core for every published latency |
| Outcome | matrix recorded; **no promotion** — nothing has cleared a gate |

## Why the 58-label rows cannot be compared

`cascade_balanced_v3` and `v4` emit a **collapsed** catalogue: `given_name`,
`family_name` and `middle_name` are folded into `full_name`, and
`street_number_and_name` into `address`. They are scored against gold collapsed
the same way, which is correct for them and is a **different, easier question** —
the distinctions removed are precisely the ones two independent judges agree on
only 0.018–0.40 of the time.

So `v4`'s holdout 0.6574 against `scorecard61`'s 0.5836 is not evidence that the
58-label model generalises better. It is mostly evidence that four hard tags were
deleted from its exam. They are included here because excluding them would hide
two shipped artifacts, and italicised because including them naively would mislead.

## What the holdout actually exposes

Whole tag families never fire on unfamiliar layouts. For `cascade_p88r90`, tags
with substantial gold and **zero** predictions across 1,612 documents include
`medical_condition` (611), `patient_id_number` (611), `medical_treatment` (611),
`medication` (561), `county` (610), `icd_10` (120), `credit_card_number` (41).

The PHI clinical family is the clearest case: those tags exist in the catalogue
only because the training corpora supply them, and this corpus renders them in a
layout the model has never seen. **A clinical-PDF deployment should measure before
trusting any model here.**

## What this changes about the earlier conclusions

Nothing measured earlier is retracted — the sealed numbers stand and
`cascade_p88r90` does meet the 90/80/80 target on the eight. What changes is what
the sealed numbers *mean*:

- The threshold work (micro F1 0.7299 → 0.8564, precision 0.6360 → 0.9165) is a
  real in-distribution gain and an out-of-distribution **loss** in every case.
- The target itself selects against generalisation. It was set on the eight, and
  the model built to hit it is the second-worst of eleven on the holdout.
- The recommendation stands but the reason has changed: prefer `p90r85b1` over
  `p88r90` not merely because it scores higher, but because higher recall is the
  only in-distribution signal that predicts out-of-distribution behaviour.

## What is still open

1. **One holdout is one data point.** These conclusions rest on a single
   out-of-distribution corpus. A second, of a different kind, would say whether
   "recall predicts generalisation" is a property of this problem or of this PDF
   corpus.
2. **The clinical PHI gap is unexplained.** Zero firings on 611-document tags is
   not a threshold effect — it is a representation failure, and nobody has looked
   at why.
3. **No model has cleared a gate**, in any project, across this entire lineage.
4. **The recall cost is still unpriced** in incidents rather than metrics.
