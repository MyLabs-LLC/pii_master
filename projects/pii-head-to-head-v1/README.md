# Pipeline spec — pii-head-to-head-v1

A head-to-head between the two shipped sensitive-data document taggers, both
**re-tuned from scratch on the full `1-train` corpus** under one loader, one
catalogue and one evaluator, then scored on all eight sealed `2-eval` corpora.

The two models were built to opposite objectives. `pii-priority-fusion-1k-v1`
was selected for recall (55/55 measurable priority gates at recall >= 0.90, but
equal-corpus macro F2 0.4835 and document specificity 0.0005).
`pii-steady-aim-cascade-v1` was selected for precision (priority macro F0.5
0.7474, document specificity 0.8798, document recall 0.7582). Neither number was
produced under conditions the other model saw, and the two lineages did not even
load the training data the same way. This run removes every one of those
differences except the models themselves.

## What is held identical

| | |
| --- | --- |
| Training rows | all 8 corpora of `/home/lence/workspace/data/1-train`, 531,431 rows |
| Evaluation rows | all 8 corpora of `/home/lence/workspace/data/2-eval`, 126,129 rows, sealed |
| Loader | `training.quiet_data.iter_quiet_corpus` for **both** lineages |
| Catalogue | the frozen 58 collapsed labels, identical label space for both |
| Fit / calibration carve | `quiet_fit.carve_holdin`, 15% by stable document hash |
| Evaluator | one fixed scorer, `training/h2h_eval.py`, applied to every arm |
| Latency | p95 on exactly one core, nothing else running |

The loader is the substantive change. The priority lineage previously used
`priority_data.normalize_row`, which marks the dual-judge corpora
`label_complete=False` and so discarded 20,714 real-world clean documents, and
which inferred "this corpus has a complete catalogue" from a folder-name prefix
that an external rename had broken -- costing 54,812 of 70,600 negatives without
raising anything. Retrained under the corrected loader, both models see
identical rows carrying identical gold. Without that, a head-to-head compares
two training sets as much as two models.

## Arms

| Arm | Lineage | Read window | Search budget |
| --- | --- | ---: | --- |
| A | priority fusion | 1,000 chars (as shipped) | hash 300 + tfidf 300 + embeddingbag 300 + fusion 100 |
| B | steady-aim cascade | 12,000 chars (`deep`, as shipped) | docgate 250 + tagdisc 250 + tagcount 150 + cascade 350 |
| C | priority fusion | 12,000 chars (confound control) | shares A's component search; own fusion re-selection |

Arm C exists because the two shipped models do not read the same amount of a
document, and on `govdocs2` -- which averages 149,000 characters -- a 12x
difference in input could decide the result on its own. With C in the table, a
gap between A and B can be split into the part that is the read window and the
part that is the model.

Arm B's profile search is pinned to `deep` so the arm stays at 12,000
characters. That concentrates all 1,000 trials on one profile rather than
spreading them over three, and it inherits the lineage's known blind spot: no
`fast` or `std` cascade has ever been evaluated in it. Recorded, not fixed.

## The decision, written before the numbers

`profiles/sensitive-data.yaml`, unchanged:

* **gate** -- per-priority-tag recall >= 0.90 on a `ci_lower` basis, `min_support: 30`
* **headline ranker** -- equal-corpus macro F2
* **serving constraint** -- one-core p95 <= 5 ms

Macro F2 favours the recall-first arm by construction, and F0.5 is where the
precision-first arm's objective lives. Both ladders are reported in full and the
headline is named here rather than chosen once the results are in.

## Metrics recorded on every arm x corpus

Per tag: support, tp, fp, fn, predicted, precision, recall, F0.5, F1, F2, F3,
sorted worst-first. Per corpus: macro and micro {P, R, F0.5, F1, F2, F3},
priority-tag macro, `f2_min`, `f2_median`, `n_tags_f2_zero`,
`n_tags_f2_below_10pct`, `prediction_rate`, `tags_predicted_zero_times`.
Document level (precision, recall, specificity, F1) on the three corpora holding
genuine negatives. Bootstrap CIs at 95% over 1,000 resamples.

A precision-bearing metric on positive-only gold is NOT_MEASURABLE, never 0.0.
`mp suite check` puts precision at 5/8 corpora and recall at 7/8.

## Known property reproduced rather than fixed

The fusion recipe calibrates its component thresholds on 20,000-character scores
and then serves at 1,000. That mismatch is part of the shipped recipe; it is
reproduced faithfully here, and arm C is partly there to measure what it costs.

## Ship

Nothing. This is a measurement run: no promotion, no packaging, no deployment.
Packaging a winner is a separate decision with its own approval.
