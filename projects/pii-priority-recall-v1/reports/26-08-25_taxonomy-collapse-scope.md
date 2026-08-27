# Scope — taxonomy-collapse re-run

## Results

**Measured, not estimated: the collapse lowers the headline score.** Folding
`given_name` / `family_name` / `middle_name` into `full_name` and
`street_number_and_name` into `address`, in both gold and predictions, and
dropping the four folded tags from each catalogue:

| Arm | macro F2 | → collapsed | micro F1 | → collapsed | Gates |
| --- | ---: | ---: | ---: | ---: | ---: |
| `champion_1k` | 0.4835 | **0.4745** (−0.0090) | 0.3812 | 0.3642 (−0.0170) | 55/55 → 55/55 |
| `perlabel_v4` | 0.4932 | **0.4855** (−0.0077) | 0.4249 | 0.4123 (−0.0126) | 55/55 → 55/55 |

This was re-scored from cached predictions under the collapsed mapping. **No
evaluator was changed and no model was selected** — it is a probe, run so the
payoff could be known before committing.

### The mechanism works exactly as predicted — and still loses

The merge does what the agreement analysis said it would. On the corpora where
annotators used `given_name` without `full_name`, a `full_name` prediction was
being scored as a false positive; after the fold it is a true positive:

| Corpus | Tag | Support | Precision |
| --- | ---: | ---: |
| `pii_holdout_20k` | `full_name` | 7,836 → 11,449 | **0.388 → 0.571** |
| `pii2_eval_30k` | `full_name` | 6,811 → 9,994 | **0.229 → 0.339** |
| `pii_holdout_20k` | `address` | 4,629 → 6,566 | **0.232 → 0.329** |
| `openpii_pii_eval_38k` | `full_name` | 29,335 → 30,464 | 0.753 → 0.782 |
| `ai4privacy` | `full_name` | 5,114 → 5,470 | 0.483 → 0.516 |
| `pii2_eval_30k` | `address` | 2,553 → 2,841 | 0.085 → 0.094 |

`full_name` precision rises **47%** on `pii_holdout_20k` and **48%** on
`pii2_eval_30k`. Recall is unchanged throughout. That is a real measurement
error being corrected on the highest-support priority tag in the project.

**It loses anyway, because of the denominator.** `macro_f2` averages over the
frozen corpus catalogue, so removing four tags changes the mean regardless of
what else happens — and those four tags are *not* weak:

| Corpus | `family` | `given` | `middle` | `street` | Effect of dropping them |
| --- | ---: | ---: | ---: | ---: | ---: |
| `openpii` | 0.908 | 0.928 | 0.889 | 0.866 | **−0.0344** |
| `pii_holdout_20k` | 0.870 | 0.889 | 0.830 | 0.845 | −0.0219 |
| `pii2` | 0.786 | 0.747 | 0.701 | 0.721 | −0.0178 |
| `ai4privacy` | 0.468 | 0.534 | 0.096 | 0.656 | +0.0141 |
| `betterdataai` | 0.071 | 0.171 | 0.000 | 0.426 | +0.0136 |
| | | | | | **net −0.0093** |

On three of the five complete corpora the model scores **0.70–0.93** on the
name components *separately*. The collapse throws that credit away, and the
merge gain on two tags does not cover the loss of four.

### So the finding is about what you want to measure

The agreement probe and this probe are both right, and they are not measuring
the same thing:

- **Agreement** rises under the collapse (macro 0.514 → 0.635, weighted 0.469 →
  0.788) because two *judges* cannot agree on which name tag to use.
- **The model's score** falls, because the model *can* reproduce the
  distinction — it was trained on these corpora and has learned each one's
  convention.

That is the uncomfortable part worth stating plainly: the model scoring 0.89 on
`given_name` where two independent judges agree at 0.29 does not mean the model
knows something the judges do not. It means it has learned **this corpus's
labelling convention**, which is exactly the thing that does not generalise.
Collapsing removes the reward for convention-fitting — and the score falls
because a good part of the score was convention-fitting.

**A lower number that measures a concept the labels can pin down is worth more
than a higher one that rewards reproducing an arbitrary convention.** But that
is a judgement about the contract, not something a probe settles, and it is
yours to make.

## TL;DR

- **The collapse lowers macro F2 by 0.008 and micro F1 by 0.013.** Measured on
  cached predictions for both arms; gates stay 55/55 either way.
- The intended mechanism works — `full_name` precision **+47%** on
  `pii_holdout_20k`, **+48%** on `pii2_eval_30k`, recall unchanged — but is
  outweighed by dropping four tags the model scores 0.70–0.93 on.
- **The earlier "ceiling rises to 0.635" claim was about judge agreement, not
  about the model's score.** Both are true; they are different quantities. The
  score falls precisely because part of it was reproducing an arbitrary
  labelling convention that two independent judges could not agree on.
- **Compute is not the constraint** — all 12 arms have cached predictions, so a
  full re-run is ~20 minutes. The cost is that every published number becomes
  non-comparable, including the eight `verify.py` pins and the shipped model
  card.
- **Recommendation: option C.** Keep the contract, report the collapsed view
  beside it. If the goal is a number that generalises rather than a number that
  is high, option B is defensible — but it should be chosen for that reason, not
  in the expectation that the score goes up.

## Cost of the full re-run

Not the constraint. **All 12 arms already have cached predictions** (126,129
rows each), and the collapse is a re-scoring, not an inference:

| Work | Cost |
| --- | --- |
| Re-score 12 arms × 8 corpora, with bootstrap | **~2–3 minutes**, no inference |
| Rebuild `evaluation_catalogue.json` (5 corpora lose 4 tags) | minutes |
| Rebuild suite / arms / decision, re-run `mp decide` | minutes |
| Re-run selection + materialize under the new contract | ~15 minutes |
| Rewrite both existing reports, `verify.py`, the model card | the real work |

**The expensive part is that every published number becomes non-comparable.**
`decision/verify.py` pins eight published numbers from the 1,000-trial run;
under a changed contract it must either be re-baselined or explicitly scoped to
the old taxonomy. The `MODEL_CARD.md` and the shipped bundle quote the old
numbers. Two reports quote them. That is the cost, and it is not measured in
compute.

## Options

**A — Don't collapse.** Keep the contract. Record that `full_name` precision is
understated by up to 47% on two corpora through annotation convention, as a
known caveat on the metric. Zero cost, zero churn; the caveat lives in prose
where it will eventually be forgotten.

**B — Collapse and re-baseline everything.** ~20 minutes of compute plus the
documentation churn. Headline drops to ~0.486. Every number in the project
changes and every prior report needs a scope note. Buys a metric that measures
name *presence*, which two judges agree on at F1 0.992.

**C — Report both (recommended).** Keep the current taxonomy as the contract so
nothing is invalidated, and add the collapsed view as a standing diagnostic
beside it — the probe that produced this table already computes it. Costs one
extra column in the experiment log. Makes the convention-fitting visible on
every future run instead of resting on one report nobody re-reads.

**D — Collapse the gold, keep the tags.** Incoherent, named only to rule it
out: folding gold while leaving `given_name` in the catalogue makes every
`given_name` prediction an unfixable false positive.

## Implemented — option C

The contract is unchanged. The collapsed view is now a **standing diagnostic**,
written beside the gate numbers rather than living in this report:

- `emit_diagnostic()` writes `evaluations/<family>/collapsed/summary.json` plus
  a per-corpus file, explicitly labelled `role: diagnostic - NOT a gate`.
- `record_in_run()` folds it into `run.json` as
  `run_summary.<family>.collapsed_taxonomy`, and appends the per-corpus delta to
  each arm's `verdict`, so it appears in the Experiment Log's existing column.
  The three positive-only corpora get no note — they cannot measure macro F2,
  and a corpus reports only what its gold can measure.
- **`final_bootstrap` calls it automatically**, so every future gate run emits
  it without anyone remembering. Verified by deleting `perlabel_v3/collapsed/`
  and watching a plain bootstrap run restore it.

Backfilled across all 12 arms. The Experiment Log's fixed schema has no room
for a new metric column, so the diagnostic rides in `Verdict / Notes` — which
already carried each arm's standing summary — rather than by editing the
shared report writer.

### What the backfill revealed

Running it across all 12 arms produced the cleanest confirmation of the thesis
so far. **`current_rules` is the only arm that gains from the collapse:**

| Arm | macro F2 Δ | micro F1 Δ |
| --- | ---: | ---: |
| **`current_rules`** | **+0.0114** | **+0.0365** |
| `hash_sgd_f2` | −0.0007 | −0.0011 |
| `tfidf_linear` | −0.0011 | +0.0001 |
| `hybrid_priority_001` | −0.0058 | −0.0152 |
| `perlabel_v4` | −0.0077 | −0.0126 |
| `hybrid_priority` | −0.0082 | −0.0156 |
| `champion_1k` | −0.0090 | −0.0170 |
| `hash_sgd` | −0.0108 | −0.0139 |
| `embeddingbag_asl` | −0.0111 | −0.0123 |

The rules engine has no name-component detectors at all. It never learned the
given/family/middle convention, so it was being penalised for a distinction it
cannot make — and folding that distinction away is pure gain for it. Every
learned arm loses, and the fusion arms, which reproduce the convention best,
lose most.

**That ordering is the measurement working.** The amount an arm loses under the
collapse is a direct estimate of how much of its score was convention-fitting
rather than PII detection. It is now computed on every run.

## Reproducing

```
training/simulate_taxonomy_collapse.py --bootstrap          # the scoping comparison
training/simulate_taxonomy_collapse.py --emit --families …  # write the standing diagnostic
projects/pii-priority-recall-v1/taxonomy_collapse_scope.json
projects/pii-priority-recall-v1/evaluations/<family>/collapsed/summary.json
```
