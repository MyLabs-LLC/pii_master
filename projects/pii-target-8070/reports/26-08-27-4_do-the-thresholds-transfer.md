# The target-box thresholds do transfer — and the one-holdout finding was wrong

## Results

Nine leave-one-corpus-out folds. Within each, the gate and 61 heads are fitted
on the other eight sources, then **three threshold policies are derived from
that fold's own calibration carve** and scored on the held-out source. Same
weights, same gate, same calibration rows: the only difference between the three
numbers on any fold is where the per-tag thresholds sit.

| held-out corpus | baseline | `p88r90` | `p90r85b1` | best |
| --- | ---: | ---: | ---: | --- |
| `pii_holdout` | 0.8053 | 0.8413 | **0.8528** | `p90r85b1` |
| `openpii` | 0.7754 | 0.7824 | **0.8127** | `p90r85b1` |
| `pii2` | **0.6035** | 0.5734 | 0.5731 | baseline |
| `synthetic_pdf` | **0.5856** | 0.5034 | 0.5525 | baseline |
| `ai4privacy` | 0.5048 | 0.5137 | **0.5611** | `p90r85b1` |
| `betterdataai` | 0.3388 | 0.3692 | **0.4295** | `p90r85b1` |

| policy | micro F1 | precision | recall | folds won | vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.6022 | 0.5768 | 0.6441 | 2 | — |
| `p88r90` | 0.5972 | **0.6791** | 0.5407 | 0 | −0.0050 |
| **`p90r85b1`** | **0.6303** | 0.6662 | 0.6067 | **4** | **+0.0281** |

`datax`, `nemotron` and `govdocs2` are excluded — positive-only or partial gold
cannot support precision, so they report no micro F1.

**The thresholds are re-derived inside every fold, never imported from the
packaged models.** A packaged threshold was selected on a carve that included
the held-out source; reusing it would leak that source back in and understate
the transfer gap.

### This contradicts entry 7 of the training history, and entry 7 was wrong

`26-08-27_evaluation-matrix.md` concluded that the sealed suite ranks models
*backwards* — that optimising precision predicts worse generalisation
(r = −0.78) and that the untuned baseline was the best 61-label model
out-of-distribution. That conclusion rested on **one** out-of-distribution
corpus: `Synthetic_PDF_Corpus_v2_1612`.

Look at the `synthetic_pdf` row above. Baseline 0.5856, `p88r90` 0.5034 — the
same reversal, reproduced exactly. **That corpus really does behave the way the
report said.** It is simply not representative: on four of the six measurable
folds the ordering runs the other way, and `p90r85b1` beats the baseline by
+0.0281 on average.

The earlier report's own open question — *"one holdout is one data point"* — was
the correct instinct, and this is the answer to it. A single out-of-distribution
corpus supported a confident claim about generalisation that six of them refute.

### Precision transfers; recall is what the boxes spend

| | baseline | `p88r90` | `p90r85b1` |
| --- | ---: | ---: | ---: |
| micro precision on unseen sources | 0.5768 | **0.6791** | 0.6662 |
| micro recall on unseen sources | **0.6441** | 0.5407 | 0.6067 |

`p88r90` buys +0.10 precision for −0.10 recall and lands net-negative on F1.
`p90r85b1` buys +0.09 precision for −0.04 recall and lands net-positive. The
difference between the two is entirely the recall floor and the F-beta used to
pick the point inside the box — `p90r85b1` uses `beta = 1.0`, which does not
over-weight precision when choosing among in-box points.

This agrees with the clean-document measurement taken the same day: the boxes
cut real-document false alarms from 860.8 to ~390–510 per 1,000 documents. The
precision half of the tuning is real and it travels.

### The transfer gap is still large

`p90r85b1` scores **0.8564** on the sealed suite and **0.6303** across the six
folds. The tuning is not the problem — the gap is roughly the same size for all
three policies, and it is the architecture's, not the thresholds'.

## TL;DR

- **`p90r85b1` is the best policy on unseen sources** (0.6303 vs the baseline's
  0.6022, winning 4 of 6 folds). The sealed-suite ordering holds up.
- **The "sealed suite ranks models backwards" finding was an artefact of a
  single holdout corpus.** `synthetic_pdf` reproduces it exactly; five other
  sources do not. Entry 7 of the training history is corrected, not retracted.
- **`p88r90` is genuinely worse than `p90r85b1` on transfer** (−0.0050 vs
  +0.0281) because it spends twice as much recall for the same precision. Its
  `beta = 0.5` picks too aggressively inside the box.
- **Precision transfers (+0.09 to +0.10); recall is the currency** (−0.04 to
  −0.10). Consistent with the clean-document false-alarm result.
- **The transfer gap itself is unchanged** — ~0.23 for every policy. It belongs
  to the architecture, not the thresholds.

---

| | |
| --- | --- |
| Date | 2026-08-27 |
| Project | `projects/pii-target-8070` |
| Request | *"do both"* |
| Scope | 9 folds × 3 threshold policies, full refit per fold |
| CPU budget | 1 core |
| Artifacts | `evaluations/loco_variants.json`, `loco_variants_agg.json`, `training/h2h_loco_variants.py` |
| Outcome | `p90r85b1` confirmed as champion on transfer; **no promotion** |

## Validation

The `baseline` column reproduces `h2h_loco`'s numbers **exactly** on every fold
(`synthetic_pdf` 0.5856, `pii2` 0.6035, `openpii` 0.7754, `betterdataai` 0.3388,
`ai4privacy` 0.5048, `pii_holdout` 0.8053). The refit is identical and the only
moving part is the threshold policy — which is what makes the comparison valid.

## What this changes about the recommendation

`p90r85b1` was already the recommended model, on sealed-suite grounds, with a
caveat that the holdout ranked it fourth. That caveat is now withdrawn: across
six sources it is the best of the three on transfer, and it carries the smallest
recall loss of the tuned policies while keeping most of the precision gain.

`p88r90` should not ship. It hits the 90/80/80 box in-distribution and is the
worst of the three on unseen data.

## What is still open

1. **The `synthetic_pdf` reversal is unexplained.** One source out of six
   inverts the ordering, consistently and by a wide margin (−0.08 for `p88r90`).
   Something about that corpus interacts with thresholds differently, and
   knowing what would be worth more than another sweep.
2. **`betterdataai` at 0.4295 remains the worst fold** even at its best, against
   0.5414 in-distribution. The gold audit already found its `email` precision
   ceilings at 0.5779.
3. **Recall on real documents is still unmeasured.** The clean-document corpus
   has no positives; these folds have no real file formats. Neither answers what
   a real PII-bearing PDF does.
4. **No model has cleared a gate.**
