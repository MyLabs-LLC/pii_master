# Pipeline spec — pii-target-8070

Can the shipped cascade be made to hit **80% precision and 70% recall on every
tag** without retraining anything?

The question came from the user as a target, not a hypothesis. Before building
anything it was turned into a measurement: for each tag, does its precision-recall
curve on the calibration carve ever pass through the box? A threshold can only
move a tag along its own curve, so that question has a definite answer per tag and
it separates "needs a threshold" from "needs a better model" before any budget is
spent.

| target | reachable by threshold alone | needs a model or better gold |
| --- | ---: | ---: |
| **P ≥ 0.80, R ≥ 0.70** | **54 of 61** | 7 |
| P ≥ 0.80, R ≥ 0.90 | 38 of 61 | 23 |

Both are built and measured, because the 16-tag difference between them *is* the
price of the recall relaxation and it should be reported rather than assumed.

## Objective

| | |
| --- | --- |
| Task family | multi-label document tagging over the 61-label scorecard catalogue |
| Metric to optimize | `micro_f1`, then `precision_micro` — the precision-led order |
| Gate | **two arms, two declared policies**: `policy_p80r70.yaml` and `policy_p80r90.yaml`, both also carrying the 8 ms one-core serving requirement |
| Baseline | `cascade_scorecard61` — micro F1 0.7299, micro precision 0.6360 |

## What changes, and what emphatically does not

Only the 61 tag thresholds. The gate weights, the gate threshold, the 61 head
weight vectors, the feature hashing and the read window are `cascade_scorecard61`'s,
byte-identical. Nothing is refit and no new data is used.

That is what makes the arms directly comparable with the incumbent, and it is why
their latency is **carried rather than re-measured**: `predict_cascade` scores every
head unconditionally and the thresholds only move the comparison, so the
per-document instruction sequence is the same. Re-measuring would be measuring the
same code.

## The selection rule

`training/h2h_target_box.py`. Per tag, on the calibration carve:

- sweep the curve, find the points with `P ≥ p_target` **and** `R ≥ r_target`;
- among those take the **F0.5 optimum**, so the choice inside the box is
  precision-led rather than arbitrary;
- if the curve never enters the box, keep the tag's best F0.5 point and **report
  it as unreachable** — not silently parked somewhere flattering;
- under 30 calibration positives, `not_measurable` — an absence of evidence, named
  as one.

## Making precision gateable at all

`h2h_eval` exports exactly one metric per tag × corpus scope: `recall`. That is
the right default — most of this suite's corpora carry positive-only tag gold,
where a precision figure means nothing — but it makes "80% precision across the
board" ungateable: a hard constraint on a metric no scope carries is
`NOT_MEASURABLE` everywhere, which blocks every arm while saying nothing about any
of them.

The numbers already existed. `evaluate_corpus` writes `tp`/`fp`/`fn` per tag per
corpus, and `per_corpus[...]["can_measure_precision"]` already declares which
corpora may answer. `training/h2h_precision_scopes.py` lifts those into `scopes`
for the five corpora that can, and leaves the other three out. The evaluator is
not modified — it is a forbidden surface, and every existing number is unchanged.

**The interval is different and that is stated, not hidden.** Recall scopes carry a
document-bootstrap lower bound; per-tag precision has no bootstrap, so the derived
scopes carry a **Wilson score** lower bound at the same 95%. Two estimators in one
policy, answering "conclusively above the bar?" by different routes.

## Data

Training carve for selection (`quiet_fit.carve_holdin`, training corpora only);
the eight sealed `data/2-eval` corpora scored once, afterwards. The catalogue,
suite and evaluator are `pii-scorecard-60`'s, unchanged.

## Budget & guardrails

No search, no training, no new data. Two deterministic re-selections. Nothing is
promoted unless it clears a declared gate.
