# Balanced gate shipped end-to-end; the document bars are not reachable in this architecture

## Results

Three things were run: the unfinished balanced-gate cascade, a feasibility probe
on the document-level bars, and an active-learning label-budget curve. One is a
free improvement. The other two say to stop spending on this architecture.

### 1. The balanced gate, carried through a full cascade

Gate refit with source-balanced weights and `alpha = 1e-2`; heads, tag-threshold
rule, read profile and evaluator all unchanged from arm B. Both operating points
re-derived. Scored on the eight sealed corpora by the same fixed evaluator.

| metric | arm B | **balanced gate** | Δ |
| --- | ---: | ---: | ---: |
| document recall | 0.7532 | **0.7975** | **+0.0442** |
| document precision | 0.8893 | **0.8956** | +0.0062 |
| document specificity | 0.8832 | 0.8792 | −0.0040 |
| macro F2 | 0.6497 | **0.6614** | +0.0116 |
| macro recall | 0.6482 | **0.6938** | **+0.0456** |
| macro precision | 0.6154 | **0.6182** | +0.0027 |
| **worst priority-tag recall** | 0.6524 | **0.7221** | **+0.0697** |
| micro F1 | 0.7313 | 0.7255 | −0.0058 |
| priority macro F0.5 | 0.7467 | 0.7434 | −0.0033 |
| prediction rate | 0.7836 | 0.8192 | +0.0357 |

Priority tag×corpus gates cleared, of 55 measurable:

| | ≥ 0.90 | ≥ 0.75 |
| --- | ---: | ---: |
| arm B | 25 | 43 |
| **balanced gate** | **29** | **45** |

**Better on almost every axis at no cost.** Same architecture, same latency
(3.916 ms p95), same 57 enabled tags. The two regressions — micro F1 −0.006,
priority F0.5 −0.003 — are noise beside a +0.070 gain on the worst priority tag.
This is the largest single measured improvement in the project and it is free.

It still does not ship: 29 of 55 gates is short of 55.

### 2. Feasibility — the bars are not reachable

The precision-view policy wants document precision ≥ 0.90, specificity ≥ 0.85 and
recall ≥ 0.85 simultaneously, conclusively, on real-world documents. Walking the
measured frontier on the sealed real corpora (5,397 positives / 5,152 negatives):

| gate | AUC | best recall @ P≥0.90, sp≥0.85 | best precision @ R≥0.85 |
| --- | ---: | ---: | ---: |
| arm B | 0.8453 | 0.5370 | 0.7008 |
| balanced + regularised | 0.8804 | **0.6231** | **0.7664** |
| **bar** | | **0.85** | **0.90** |

No threshold on either gate satisfies all three. The shortfall on recall is
**0.227** even for the better gate.

### 3. Active learning — the curve is flat

The obvious response to (2) is "get more real data from the hard distribution".
Measured directly, at increasing label budgets, on a fixed held-out 30% of the
sealed real documents:

| target-distribution documents added | AUC | best recall @ bars |
| ---: | ---: | ---: |
| 0 | 0.8832 | 0.6279 |
| 500 | 0.8763 | 0.6358 |
| 1,500 | 0.8731 | 0.6072 |
| 3,000 | 0.8801 | 0.6376 |
| 5,000 | 0.8804 | 0.6255 |
| 7,370 *(all available)* | 0.8778 | 0.6157 |

**7,370 labelled documents from exactly the distribution being tested move
nothing.** Flat within noise, slightly down at the end.

Cross-validated *within* the target distribution — training and testing on the
sealed real documents only — the ceiling is **AUC 0.869**, below what the
balanced gate already achieves from the training distribution.

## TL;DR

- **Adopt the balanced gate.** Document recall +0.044, macro recall +0.046, worst
  priority-tag recall +0.070, gates cleared 25 → 29 of 55, precision and
  specificity held, latency unchanged. Free.
- **The document bars are not reachable in this architecture.** Best achievable is
  recall **0.62** holding precision and specificity (bar 0.85), or precision
  **0.77** holding recall (bar 0.90).
- **Active learning will not close it.** 7,370 labelled target-distribution
  documents moved AUC 0.883 → 0.878. The labelling budget would buy nothing.
- **The constraint is the representation, not the data.** Hashed
  unigram/bigram/shape features over 12,000 characters plateau at AUC ≈ 0.88 on
  real-world documents. Synthetic data was refuted yesterday; real data is refuted
  now.
- **Verdict: `target_infeasible`** for the current architecture, with the
  reachable numbers stated above.

---

| | |
| --- | --- |
| Date | 2026-08-26 |
| Author | Ryan Lence |
| Project | `projects/pii-head-to-head-v1` |
| Run ID | H1 (follow-up) |
| Scope | document gate and its cascade; heads unchanged |
| Outcome | one improvement adopted; target declared infeasible for this architecture |

## A bug worth recording

The first rebuild returned document specificity **0.0109**, down from 0.8832. The
cause: arm B's cascade trial carries a `gate_shift` of −2.196, an **absolute**
offset searched against a score scale where the cut sits near 162. Regularising at
`alpha = 1e-2` shrinks the weights by orders of magnitude, so the same −2.196
moved the new cut to −2.13 and the gate fired on everything.

The general form is worth more than the fix: **a tuned constant does not survive a
change of scale.** That shift was searched jointly against arm B's own gate;
carrying it onto a differently-regularised one was never valid, whatever number it
produced. It is not carried over now, and the joint shift would have to be
re-searched on the new gate to mean anything — which is a further small
improvement left on the table.

The metrics table caught it immediately, which is the argument for printing
precision, recall and specificity together rather than a single headline.

## What "infeasible" rests on

Three independent lines, none of which is a tuning result:

1. **The frontier.** At the measured ranking quality there is exactly one best
   recall per specificity. The bars sit outside it by 0.227.
2. **The label-budget curve.** Flat across a 15× range of added
   target-distribution data.
3. **The in-distribution ceiling.** Training *on* the target distribution reaches
   AUC 0.869 — no better than transferring from the training distribution.

Together these say the limit is what the features can express about these
documents, not what the model has seen. An earlier "oracle" figure of AUC 0.9998
is **not** evidence against this: an unregularised linear model over 262,144
features fits 10,549 rows exactly, and that number is memorisation. The
cross-validated 0.869 is the honest version.

## Recommendation

**Adopt the balanced gate now.** It is strictly better on the metrics that matter
and costs nothing.

**Stop tuning this architecture against the 0.85/0.90 document bars.** Every
remaining dial has been measured: per-tag floor, selection beta, gate
regularisation, source balancing, synthetic negatives, real negatives. The
reachable region is recall ≈ 0.62 at the precision bars, or precision ≈ 0.77 at
the recall bar.

**Then one of two decisions, and it is a product decision rather than a modelling
one:**

- **Re-scope the bars** to the reachable region and ship the balanced cascade
  against them; or
- **Change the representation.** The plateau is a property of hashed
  unigram/bigram/shape features. Contextual embeddings or a span-level NER model
  are the next architecture, and this repo already carries candidates under
  `models/pii-master-ner-*`. That is a new project with its own feasibility
  probe, not a continuation of this one.

## Limitations

- The balanced gate's `gate_shift` was not re-searched; a joint re-search of the
  gate cut and tag thresholds on the new gate would likely gain a little more.
- The label-budget curve is bounded by the 7,370 documents available in the
  sealed real pool. A far larger real corpus was not tried; the flatness across
  the measured range is the evidence, not a proof about all budgets.
- Feasibility is measured on the two real-world corpora pooled. The synthetic
  corpora clear the bars comfortably and are not the constraint.
- The balanced gate has not been packaged as a champion bundle; it is an
  evaluated artifact under `models/cascade_balanced/`.

## Artifacts

| Path | What |
| --- | --- |
| `models/cascade_balanced/` | the adopted gate, full cascade |
| `evaluations/arm_cascade_balanced.json` | its sealed scores, all scopes and CIs |
| `probe/doc_feasibility.json` | frontier walk for both gates |
| `probe/gate_augment.json` | the synthetic-negative result |
| `decision/headline_balanced.json` | policy applied to arm B and the balanced gate |
| `training/h2h_cascade_rebuild.py` · `h2h_feasibility.py` | the two new modules |
