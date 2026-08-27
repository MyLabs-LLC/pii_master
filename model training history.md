# Model training history

One row per change, one change per row, newest last. Every row records what was
altered, what it did to the numbers, and whether it was kept — **including the
ones that failed**, because a rejected idea is the cheapest thing in here and the
most expensive to rediscover.

## Rules this log follows

1. **One change at a time.** A row that alters two things cannot tell you which
   one worked.
2. **Numbers come from a recorded evaluation**, never retyped from memory. Each
   row cites the arm file it was read from.
3. **Selection happens on the training calibration carve; the sealed corpora are
   scored once per candidate.** A row that reports a sealed number obtained by
   trying many variants against the sealed set is marked `LEAKY` and its result
   is not usable.
4. **Failures stay.** Struck-through rows were reverted.

## The target

| | target | best so far | model |
| --- | ---: | ---: | --- |
| micro precision | ≥ 0.90 | **0.9000** | `cascade_p88r90` |
| micro recall | ≥ 0.80 | **0.8020** | `cascade_p88r90` |
| micro F1 | ≥ 0.80 | **0.8470** | `cascade_p88r90` |

**Met on the point estimate, not conclusively.** The suite aggregates
equal-corpus and only five corpora can measure precision, so the corpus-level
bootstrap gives precision **[0.8448, 0.9517]** and recall **[0.7495, 0.8997]**.
Neither lower bound clears its bar. Closing *that* gap needs more
precision-measurable corpora, not a better threshold.

All figures below are equal-corpus over the eight sealed `data/2-eval` corpora,
one CPU core, evaluated by `training/h2h_eval.py` (unchanged throughout).

---

## Baseline

| # | change | micro F1 | micro P | micro R | macro recall | p95 ms | kept |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | `cascade_scorecard61` — 61-label scorecard taxonomy, thresholds from the group-recall-cap rule | 0.7299 | 0.6360 | 0.8866 | 0.6773 | 4.029 | baseline |

Source: `projects/pii-target-8070/evaluations/arm_baseline_scorecard61.json`

---

## Changes

### 1 — Taxonomy: 58 collapsed labels → 61 scorecard labels
**What changed.** Restored `given_name`, `family_name`, `middle_name` and
`street_number_and_name` as tags in their own right; dropped `routing_number`
(1 training row, 0 eval rows); kept `swift_code` though the scorecard omits it.
Only the label arrays of the feature cache were rebuilt — features are hashed
from text and cannot depend on the label space.

**Result.** macro F2 0.6641 → 0.6651 against a target of 0.6641 and a measured
ceiling of 0.6754. The taxonomy change cost the ranker nothing.

**Kept.** Every model after this point is 61-label.
Evidence: `projects/pii-scorecard-60/reports/26-08-26_scorecard-61-taxonomy.md`

---

### 2 — Threshold rule: precision floor on the group-recall cap
**What changed.** The selector honoured a 0.75 recall floor at *any* precision
cost and recorded 0.5%-precision operating points as `floor_met` successes. Three
tags produced **78,928 false positives for 692 true positives**.

**Result.** Superseded before it shipped by change 3, which constrains precision
and recall jointly and subsumes it. Retained as an opt-in
(`h2h_thresholds_v4.select_per_label(min_precision=...)`).

**Not kept** — not wrong, just dominated.

---

### 3 — Threshold selection: target a (precision, recall) box
**What changed.** Replaced "F0.5 optimum subject to a recall floor and a
group-recall cap" with "F0.5 optimum among points satisfying `P ≥ p` **and**
`R ≥ r`". No retraining: gate, weights, hashing and read window byte-identical.

**Result.** The single largest gain in this history.

| box | micro F1 | micro P | micro R | macro recall |
| --- | ---: | ---: | ---: | ---: |
| — (baseline) | 0.7299 | 0.6360 | 0.8866 | 0.6773 |
| P≥0.80, R≥0.70 | 0.8394 | 0.9165 | 0.7777 | 0.5668 |
| P≥0.80, R≥0.90 | 0.8474 | 0.8844 | 0.8141 | 0.5872 |

**Kept.** Micro F1 +0.11, micro precision +0.28, latency unchanged, nothing
trained. The model was never the problem — tags were shipping at precision the
model did not require (`gender_and_sex` at 0.2689 where 0.9894 was reachable at
the same recall).

Evidence: `projects/pii-target-8070/reports/26-08-26_target-80-70.md`

---

### 4 — ~~Content model: fine-tuned transformer, distilled, fused~~
**What changed.** Fine-tuned `kalyan-ks/ettin-68m-nemotron-pii` on all 8 corpora
(token head on the 85,593-doc span corpus, then a 61-way document head on
451,548 rows), distilled the encoder to a static token table with model2vec,
trained a linear tagger on `[mean ‖ max]` token features, fused per tag.

**Result.** The fine-tune itself worked — 50 of 61 tags improved, mean +0.41 F1
on held-in data. The *fusion* did not.

| fused into | micro F1 | vs the model it wraps | p95 ms |
| --- | ---: | ---: | ---: |
| `cascade_scorecard61` | 0.7595 | +0.0296 | 6.279 |
| `cascade_p80r70` | 0.8373 | **−0.0021** | 6.411 |

**Reverted.** Across 61 tags the fusion chose the content model **on its own zero
times** and `or` **zero times** — its entire contribution is a precision veto,
and once thresholds already deliver precision there is nothing left for it to
remove. It cost +56% latency for a loss.

Evidence: `projects/pii-content-v5/reports/26-08-26_content-chain-v5.md`

---

### 5 — Box parameter tuned to the aggregate target
**What changed.** The per-tag box and the aggregate target are different things —
micro metrics pool every decision and are dominated by high-volume tags. Swept 25
boxes **on the calibration carve only**, then corrected for the measured
calibration→sealed gap (precision falls ~0.04, recall ~0.09) and scored the two
surviving candidates sealed, once each.

| box | micro P | micro R | micro F1 | |
| --- | ---: | ---: | ---: | --- |
| P≥0.90, R≥0.90 | 0.9115 | 0.7926 | 0.8464 | recall short |
| **P≥0.88, R≥0.90** | **0.9000** | **0.8020** | **0.8470** | **meets target** |

**Kept.** `cascade_p88r90`, packaged as `pii-cascade-p88r90-v1`.

All 25 boxes hit the target *on calibration*, which is exactly why the
calibration→sealed correction mattered: calibration said everything worked and
the sealed set disagreed by 0.04–0.09.

Evidence: `projects/pii-target-8070/probe/box_sweep.json`,
`probe/ci_cascade_p88r90.json`

---

---

### 6 — Full mechanism sweep: 66 configurations, two rounds

**What changed.** Round 1 swept **21 distinct decision mechanisms** on the
calibration carve, one change each: F-beta variants inside the box, prevalence
priors, per-tag standardisation, document-length normalisation, top-k caps,
score margins, parent/child taxonomy consistency both directions, gate-margin
conditioning, disabling unreachable tags, per-vertical boxes, co-occurrence
pruning, and box ensembles. Round 2 crossed the one axis round 1 found (`beta`)
with the box parameters — 45 more configurations.

**Two sanity checks passed.** Per-tag z-scoring and length normalisation both
moved the result by ±0.0000/−0.0001, which is correct: per-tag threshold
selection is invariant to monotone per-tag transforms. A harness that showed a
gain there would have been broken.

**What failed, and it is most of it.** Top-k caps were catastrophic (−0.19 to
−0.38 calibration F1 — documents genuinely carry many tags). Co-occurrence
pruning −0.22. Disabling unreachable tags −0.09; they contribute real recall.
Prevalence priors, margins and gate-conditioning all bought precision at more
recall than they returned.

**Sealed results, 8 corpora:**

| model | micro F1 | micro P | micro R | macro F2 | macro recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cascade_scorecard61` | 0.7299 | 0.6360 | 0.8866 | 0.6651 | 0.6773 |
| `cascade_p80r70` | 0.8394 | 0.9165 | 0.7777 | 0.6406 | 0.5668 |
| `cascade_p80r90` | 0.8474 | 0.8844 | 0.8141 | 0.6564 | 0.5872 |
| `cascade_p88r90` | 0.8470 | 0.9000 | 0.8020 | 0.6480 | 0.5802 |
| `cascade_p90r90` | 0.8464 | 0.9115 | 0.7926 | 0.6421 | 0.5739 |
| `cascade_p90r85b1` | 0.8564 | 0.8721 | 0.8415 | 0.6865 | 0.6367 |
| `cascade_p88r90b1` | 0.8557 | 0.8634 | 0.8483 | 0.6877 | 0.6381 |

**Outcome.** `cascade_p88r90` remains **the only model that hits the target** —
66 configurations found nothing better for `P ≥ 0.90`. The two `beta=1.0`
finalists are better models on every other measure (micro F1 0.8564/0.8557,
macro F2 0.6865/0.6877, macro recall 0.6367/0.6381) and miss only on precision.

**Kept:** `cascade_p88r90` for the target; `cascade_p90r85b1` noted as the better
all-round model.

Evidence: `probe/mechanism_sweep.json`, `probe/round2_sweep.json`

---

### 7 — Out-of-distribution holdout, and what it says about all of the above

**What changed.** Added `Synthetic_PDF_Corpus_v2_1612` (1,612 synthetic PDFs,
extracted text) as a holdout at `data/3-holdout/`, **outside** the scored suite
and scored only after selection. Its own taxonomy IS the GAIA scorecard's 60
tags, so all 60 mapped by the same slug rule with no hand-written mapping.

| model | 8-corpus micro F1 | **holdout micro F1** | holdout P | holdout R |
| --- | ---: | ---: | ---: | ---: |
| `cascade_scorecard61` | 0.7299 | **0.5836** | 0.6606 | 0.5227 |
| `cascade_p80r90` | 0.8474 | **0.5167** | 0.7108 | 0.4059 |
| `cascade_p88r90` | 0.8470 | **0.4992** | 0.7061 | 0.3861 |
| `cascade_p90r85b1` | 0.8564 | **0.5453** | 0.7156 | 0.4405 |
| `cascade_p88r90b1` | 0.8557 | **0.5427** | 0.7100 | 0.4392 |

**The ranking reverses.** The baseline is best on the holdout (0.5836); the
target-hitting model is worst (0.4992).

**What predicts out-of-distribution performance**, across the five models scored
on both:

| measured on the 8 corpora | Pearson r with holdout micro F1 |
| --- | ---: |
| micro **recall** | **+0.9928** |
| macro recall | +0.9831 |
| micro F1 | **−0.7548** |
| micro **precision** | **−0.8689** |

**Optimising precision or micro F1 on the sealed suite predicts *worse*
generalisation.** Holdout precision barely varies across every model built here
(0.6606–0.7156) — precision transfers for free. Recall is the quantity actually
at risk, and every threshold gain in entries 3, 5 and 6 was bought with it.

**Not reverted, but reframed.** The gains on the eight are real and the target is
met there. They are also, in part, a fit to the character of those eight corpora.
Any claim that a model here generalises must be checked against the holdout
first.

Evidence: `evaluations/holdout_*.json`, `training/h2h_score_holdout.py`

### 8 — Ninth corpus into training, and leave-one-corpus-out to replace the holdout it consumed

**What changed.** `Synthetic_PDF_Corpus_v2_1612` was split 80/20 (stratified by
vertical/document-type) into `1290_synthetic_pdf_train_1.27k` and
`322_synthetic_pdf_eval_318`, caches rebuilt, and the cascade retrained on
532,721 rows across nine sources.

**The retrain bought almost nothing, and the headline hid it.**

| measured on | before | after |
| --- | ---: | ---: |
| the original 8 corpora | 0.7299 | **0.7305** (+0.0007) |
| the 9-corpus headline | — | 0.7627 |

The headline rose because the new corpus scores **0.9233 on itself**, not
because the model improved. Nothing in the sealed suite could distinguish those
two explanations — which is the whole problem with the sealed suite.

**The split also destroyed the only out-of-distribution test in the repository**
(entry 7 rests on it). Leave-one-corpus-out replaces it: for each source, refit
the gate, all 61 heads and the thresholds on the other eight, then score that
source's sealed split.

| held out | in-distribution | **LOCO** | gap |
| --- | ---: | ---: | ---: |
| `synthetic_pdf` | 0.9233 | 0.5856 | **0.3377** |
| `betterdataai` | 0.5414 | 0.3388 | 0.2026 |
| `pii2` | 0.7702 | 0.6035 | 0.1668 |
| `ai4privacy` | 0.6201 | 0.5048 | 0.1153 |
| `openpii` | 0.8712 | 0.7754 | 0.0958 |
| `pii_holdout` | 0.8497 | 0.8053 | 0.0444 |

`datax`, `nemotron` and `govdocs2` are excluded — positive-only or partial gold
cannot support precision, so they report no micro F1.

**Mean transfer gap 0.1604.** Expect **~0.60 micro F1 on a new source**, against
a 0.7627 headline — about **21% of every published point is source familiarity**.

**LOCO is validated against the holdout it replaces**: it scores `synthetic_pdf`
at 0.5856; the genuine out-of-distribution measurement taken before the split was
**0.5836**. Within 0.002.

**Recall is the half that breaks**, confirming entry 7 from the opposite
direction: micro precision 0.6772 → 0.5768 (−0.1004), micro recall 0.8996 →
0.6441 (**−0.2556**). Recall falls 2.5× harder. The 90/80/80 box was measured
in-distribution and its recall leg is the one that will not hold.

Evidence: `evaluations/loco.json`, `evaluations/loco_gap.json`,
`training/h2h_loco.py`, `reports/26-08-27-2_leave-one-corpus-out.md`

### 9 — The thresholds under transfer, and the correction to entry 7

**What changed.** Nothing in any model. The nine LOCO folds were re-run deriving
**three threshold policies per fold** from that fold's own calibration carve —
baseline, `p88r90` (box P>=0.88 R>=0.90, beta 0.5) and `p90r85b1` (box P>=0.90
R>=0.85, beta 1.0). Same weights, same gate, same rows; only the thresholds move.
The `baseline` column reproduces entry 8 exactly on all six measurable folds,
which is what makes the comparison trustworthy.

| policy | LOCO micro F1 | precision | recall | folds won | vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.6022 | 0.5768 | 0.6441 | 2 | — |
| `p88r90` | 0.5972 | **0.6791** | 0.5407 | 0 | −0.0050 |
| **`p90r85b1`** | **0.6303** | 0.6662 | 0.6067 | **4** | **+0.0281** |

**Entry 7 was wrong, and this is why.** Entry 7 concluded from the PDF holdout
that the sealed suite ranks models backwards and that the untuned baseline
generalises best. That corpus behaves exactly as entry 7 said — in these folds
it is the one source where the baseline (0.5856) beats `p88r90` (0.5034). It is
**not representative**: on four of six sources the ordering runs the other way.
A confident generalisation claim rested on a single corpus and six refute it.
Entry 7's own open question, *"one holdout is one data point"*, was correct.

**`p90r85b1` is the champion on transfer as well as in-distribution.** The
caveat that the holdout ranked it fourth is withdrawn. **`p88r90` should not
ship**: it hits the 90/80/80 box in-distribution and is worst of the three on
unseen data, because `beta = 0.5` picks too aggressively inside the box and
spends twice the recall for the same precision.

Evidence: `evaluations/loco_variants.json`, `loco_variants_agg.json`,
`training/h2h_loco_variants.py`, `reports/26-08-27-4_do-the-thresholds-transfer.md`

### 10 — 7,126 real documents containing no PII

**What changed.** Nothing in any model. First measurement on **real files** —
3,296 PDF, 2,657 DOC, 1,559 HTML, 771 XLS, 587 PPT — all judged PII-free. Every
suite corpus is synthetic or extracted plain text.

| model | doc false-alarm rate | FP / 1,000 docs | implied precision @ 90% clean |
| --- | ---: | ---: | ---: |
| `cascade_scorecard61` | 0.1911 | 860.8 | 0.3468 |
| `cascade_p90r85b1` | 0.1719 | 512.6 | 0.5065 |
| `cascade_p88r90` | 0.1575 | 424.1 | 0.5481 |
| `cascade_p80r70` | **0.1506** | **389.4** | **0.5662** |

**The threshold work halves the real-document false-alarm burden** — 860.8 →
389.4 FP per 1,000. Combined with entry 9, the precision half of the tuning is
confirmed to transfer twice over.

**Every model still flags at least 15% of clean files**, and implied precision on
a 90%-clean estate is **0.35–0.57** against sealed figures of 0.64–0.92. The
sealed suite is roughly half PII-bearing; a real estate is not, and precision
degrades with the mixture. `full_name` alone fires on 10.7% of clean documents.

**An all-negative corpus cannot measure precision** — no positives, no TP, so the
ratio is undefined rather than zero. It measures the false-positive side, which
composes with sealed true positives into the implied-precision column. Of 10,000
documents only 7,126 were admissible: 2,874 overlap `govdocs2`/`datax`, of which
587 sit in a **sealed** split and 782 carry gold contradicting this manifest.

Evidence: `evaluations/clean_docs.json`, `training/h2h_cleandocs_score.py`,
`reports/26-08-27-3_clean-documents-false-alarms.md`

### 11 — `cascade_p90r85b1` packaged as champion

**What changed.** No model. `cascade_p90r85b1` was designated champion on the
entry-9 evidence and packaged as `pii-cascade-p90r85b1-champion-v2` (30.1 MB,
25 files, all checksums verifying, 2.77 ms p95 at the standard 10,000-char depth
on one core — measured for this model, not inherited from a sibling).

**The card is generated, not written.** `training/h2h_package_champion.py`
contains no literal metric: every figure is read from a recorded evaluation at
build time, and a missing artifact fails the build. This exists because the two
hand-written cards in this project shipped wrong numbers (`p88r90` 5 of 12,
`p90r85b1` 3 of 12) while the six generated ones had none.

**It caught two errors of its own**, which is the argument for generating:

* the draft card claimed `p88r90` is "worse on every unseen source". It is worse
  on **five of six** — `p88r90` edges it on `pii2`, 0.5734 to 0.5731. The claim
  is now computed from the fold data rather than asserted.
* the `TRAINING.md` reproduce command did not run as written. It now does, and
  was **executed** to confirm it: re-running reproduces the shipped numbers to
  **1e-9** on the corpora they share.

**`cascade_p90r85b1` was shipped over `v2_p90r85b1`** (the 9-corpus retrain)
because the two are indistinguishable on the corpora they share (0.8564 vs
0.8558, v2 better on 1 of 5) while only `cascade_p90r85b1` has a genuine
out-of-distribution measurement — `v2` trained on the corpus that used to
provide it. `v2`'s apparent 0.8779 is the entry-8 inflation again.

**A provenance trap, recorded in the bundle.** Re-running the reproduce command
today gives a headline of **0.8057**, not 0.8564 — because the cache was rebuilt
with a ninth corpus, so an equal-corpus mean now spans 9 sources where the
shipped figure spans 8. The model is bit-identical. `TRAINING.md` states this so
nobody later reads it as drift.

**It has not passed the per-tag ship gate**, and the card says so in the first
screen. Champion here means *best measured*, not *cleared a bar*.

Evidence: `dist/pii-cascade-p90r85b1-champion-v2/`,
`training/h2h_package_champion.py`,
`evaluations/repro_check_cascade_p90r85b1.json`,
`evaluations/latency_cascade_p90r85b1.json`

## What the evidence says not to try again

- **Adding a transformer.** Measured at change 4: a full GPU chain bought +0.03
  micro F1 fused into the baseline and *lost* ground fused into a re-thresholded
  cascade, at +56% latency. The cascade was already finding the identifiers; what
  it lacked was knowing when to stay quiet.
- **Modelling the last per-tag precision failures.** An audit of the 39 failures
  found **~31 are gold defects**, not model weakness. `email` reaches 1.0000
  precision on four corpora and ceilings at **0.5779** on `betterdataai` at any
  threshold. No architecture fixes a label.
- **Trusting calibration numbers.** Every box hit the target on calibration; the
  sealed set was 0.04–0.09 worse. Always correct for the gap or measure it.
- **Top-k caps, co-occurrence pruning, disabling weak tags.** Measured at entry 6:
  −0.38, −0.22 and −0.09 calibration F1 respectively. Documents in this domain
  genuinely carry many tags, and the weak tags still carry recall.
- ~~**Reading micro F1 on the eight as a measure of quality.**~~ Entry 7 found it
  correlates −0.75 with out-of-distribution performance. **Entry 9 overturns
  this**: that correlation came from one holdout corpus, and across six
  leave-one-corpus-out folds the sealed ordering largely holds. The sealed suite
  still overstates absolute performance by ~0.16 (entry 8), but it does not rank
  models backwards.
- **Writing a model card by hand.** Entry 11. Two of two hand-written cards
  shipped wrong numbers; six of six generated ones did not. Generate the card
  from recorded evaluations and let a missing artifact fail the build.
- **Generalising from a single out-of-distribution corpus.** Entries 7 and 9,
  the most expensive mistake in this log. One holdout supported a confident,
  wrong claim about what predicts generalisation, and it stood for two entries.
  A generalisation claim needs several sources; LOCO gives nine for free.
- **Reading a headline that moved because a corpus was added to training.**
  Entry 8: adding the ninth source moved the original eight by +0.0007 while the
  headline rose 0.03, entirely because the new corpus scores 0.9233 on itself.
  Always re-report the unchanged subset alongside the new headline.
- **Spending the last out-of-distribution corpus on training.** Entry 8: it cost
  the only generalisation measurement here, and only LOCO recovered it. If a
  source is the sole holdout, splitting it needs the replacement measurement
  built *first*.

## Open, in value order

1. **The confidence interval, not the point estimate.** Precision
   [0.8448, 0.9517] over five corpora. More precision-measurable corpora is the
   only honest fix.
2. **`betterdataai` and `ai4privacy` gold.** 29 of 39 per-tag precision failures
   sit there. Either correct the labelling or declare the affected pairs
   unmeasurable in `suite.yaml`, as `datax` and `govdocs2` already are.
3. **Recall is unpriced.** macro recall 0.6773 → 0.5802 between baseline and
   `p88r90`. Nobody has said what that costs in missed reportable identifiers.
4. **Recall on real documents is unmeasured.** Entry 10's corpus has no
   positives; entries 8 and 9 have no real file formats. Judging a few hundred
   real PII-bearing PDFs would give the first honest precision *and* recall
   number on the formats a deployment actually meets. This is now the highest-
   value open item.
5. **The `synthetic_pdf` reversal is unexplained.** Entry 9: one source of six
   inverts the threshold ordering, consistently and by −0.08 for `p88r90`.
   Knowing why would be worth more than another sweep.
6. **`full_name` fires on 10.7% of clean documents** and is the largest error
   source for every model. Partly a taxonomy question — whether a named agency
   official or press contact counts is a policy decision nobody has recorded.
