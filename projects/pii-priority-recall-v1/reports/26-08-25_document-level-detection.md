# Document-level PII detection

## Results

**The pivot does not work, and the reason is worth more than the pivot would
have been.** On real documents with real negatives, every learned arm has a
specificity of approximately **zero** — it flags essentially every document,
including every genuinely PII-free one. Its document-level accuracy equals the
always-say-yes baseline to three decimal places.

Measured on the two corpora that carry a document-level verdict *with*
negatives — `datax` (prevalence 0.473) and `govdocs2` (0.594), both real
business and government documents, dual-judged:

| Arm | F1 | Precision | Recall | **Specificity** | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| `current_rules` | 0.4573 | 0.5588 | 0.3869 | **0.6549** | **0.5166** |
| `hash_sgd` | 0.6941 | 0.5341 | 0.9994 | 0.0021 | 0.5343 |
| `champion_1k` | 0.6940 | 0.5340 | 0.9994 | 0.0015 | 0.5340 |
| `perlabel_v4` | 0.6939 | 0.5341 | 0.9987 | 0.0026 | 0.5341 |
| *always say yes* | *0.6934* | *0.5335* | *1.0000* | *0.0000* | *0.5335* |
| **two independent judges** | **0.9487** | — | — | — | **0.9545** |

Read the last two rows together. The learned arms are **indistinguishable from
a constant "yes"** — precision equals prevalence, recall is 1.0, specificity is
0. Meanwhile two independent judges answer the same question at F1 **0.9487**,
Cohen's κ **0.9057**. The task is not hard to measure and not hard to do. These
models simply do not do it.

`current_rules` — the arm dismissed at macro F2 0.0985 — is the **only** one
with any discrimination: specificity 0.655, and the only accuracy above the
majority-class baseline.

### Why my earlier number was wrong

Scored the obvious way, across the five `label_complete` corpora, document-level
detection looks superb: `champion_1k` F1 **0.9824**, accuracy 0.9676. That
number is meaningless. Four of those five corpora have **100% PII prevalence** —
every document is positive, so a constant "yes" scores F1 1.0:

| Corpus | Prevalence | `champion_1k` predicted positive |
| --- | ---: | --- |
| `ai4privacy_pii_masking_eval_10k` | 1.0000 | 10,625 / 10,626 |
| `betterdataai_ner_silver_eval_10k` | 1.0000 | 10,360 / 10,360 (all) |
| `openpii_pii_eval_38k` | 1.0000 | 38,937 / 38,937 (all) |
| `pii_holdout_20k` | 1.0000 | 20,000 / 20,000 (all) |
| `pii2_eval_30k` | 0.8383 | 30,000 / 30,000 (all) |

On `pii2`, the one complete corpus with genuine negatives, precision is
**0.8383** — exactly its prevalence. The model flags all 30,000 documents.
A 0.98 on an all-positive corpus is a measurement artifact, and reporting it as
a capability would have been the "suspiciously perfect score" failure.

### Root cause: the training set has almost no negatives

| Split | Rows | Positive | Prevalence |
| --- | ---: | ---: | ---: |
| **Train, overall** | 554,247 | 457,421 | **0.8253** |
| `openpii_pii_train_155k` | 155,744 | 155,744 | **1.0000** |
| `ai4privacy_pii_masking_train_42k` | 42,504 | 42,504 | **1.0000** |
| `betterdataai_ner_silver_train_41k` | 41,436 | 41,436 | **1.0000** |
| `pii_trainset_100k` | 100,000 | 100,000 | **1.0000** |
| `pii2_train_150k` | 150,000 | 100,003 | 0.6667 |
| `26095_govdocs2-dualjudge-train80` | 26,095 | 13,225 | 0.5068 |
| `16000_datax-dualjudge-trainset` | 16,000 | 4,509 | 0.2818 |
| `nemotron_train_22k` | 22,468 | 0 | 0.0000 |

**339,684 training documents — 61% of the set — come from corpora where every
single document contains PII.** Real documents run 28–51% positive, so the
training mix is badly misaligned with the deployment distribution.

But "no negatives" would overstate it, and the precise version matters:
**96,826 negatives do exist in the training set** (17.5%) — 49,997 in `pii2`,
22,468 in `nemotron`, 12,870 in `govdocs2`, 11,491 in `datax`. Positives
outnumber them 4.7 : 1. So the models were shown negatives; they were shown
4.7× as many positives, and nothing in the objective rewarded getting the
negatives right.

### The recall gate is the other half of the cause

Skew alone does not force specificity to 0.0015. The threshold policy does.
Every priority tag is tuned so its recall clears 0.90 on a bootstrap lower
bound, on every corpus — which drives thresholds down until the model fires on
almost any document. **A gate that constrains only recall selects, by
construction, for a model that says yes.** That is the same mechanism as the
precision collapse in the threshold report, seen at document level instead of
per tag.

The corroboration is in the table above: `perlabel_v4`, which raised every
priority threshold to the gate boundary, has specificity 0.0026 against
`champion_1k`'s 0.0015 — a real improvement in the right direction, and still
two orders of magnitude short of useful. And `current_rules`, the one arm never
tuned against the recall gate at all, is the one with specificity 0.655.

### Disputed documents

Both dual-judge corpora carry documents the two judges could not agree on: 1,203
in `datax`, 643 in `govdocs2`. They are excluded from the headline above and
reported separately, because gold two annotators disagreed about is not gold.
Every learned arm flags **100%** of them; `current_rules` flags 28.6% / 38.4%.
That is the same finding again rather than a new one — an arm that flags
everything flags the ambiguous cases too.

## TL;DR

- **Document-level detection fails on real documents.** Every learned arm has
  specificity ≈ 0.00 and accuracy equal to the always-say-yes baseline
  (0.534). Precision equals prevalence. They flag everything.
- **The task itself is well-measured and very doable** — two independent judges
  agree at F1 **0.9487**, κ **0.9057**. This is a model failure, not a
  measurement failure, and it is the opposite of the tagging metric where the
  labels were the ceiling.
- **My earlier 0.98 was an artifact.** Four of the five `label_complete`
  corpora are 100% PII-positive; a constant "yes" scores 0.98 there. The
  measurement had to move to the two corpora with real negatives before it
  said anything.
- **Two causes, not one.** 61% of the training set comes from all-positive
  corpora (train prevalence 0.825 vs 0.28–0.51 in real documents) — though
  96,826 negatives do exist, outnumbered 4.7 : 1. And the recall-only gate
  selects for a model that says yes: `perlabel_v4`, tuned to the gate boundary,
  has 1.7× the specificity of `champion_1k`, while `current_rules`, never tuned
  against the gate, has 400×.
- **`current_rules` is the only arm that discriminates** (specificity 0.655).
  The weakest arm on the tagging metric is the only usable one on this task.
- So the pivot recommended in the previous report is **not** available off the
  shelf — but it is reachable, and the fix is a data problem rather than a
  modelling one.

## What this changes

The previous report recommended re-anchoring on document-level detection
because the labels support it. That still holds — judges agree at 0.95 — but it
cannot be done with these models as they stand. The order of work changes:

1. **Add specificity to the gate first.** This is the cheaper half of the cause
   and nothing else works around it: while the only hard constraint is per-tag
   recall, the selection procedure will keep choosing arms that flag
   everything, whatever the training mix looks like. A document-level
   specificity floor on the dual-judge corpora is a one-line addition to
   `policy.yaml` and it re-scores from cached predictions.
2. **Then rebalance the training mix.** 96,826 negatives already exist at
   4.7 : 1 against; reweighting or resampling toward the 0.28–0.51 prevalence of
   real documents costs no new data collection.
3. **Re-measure document-level detection on the dual-judge corpora only.** The
   all-positive corpora cannot answer this question and will report 0.98
   whatever happens.
4. Tagging work stays where the previous report left it: `target_infeasible`,
   renegotiated to the label-agreement ceiling.

## Caveats

- The judge ceiling (F1 0.9487, κ 0.9057) is measured on the 946 `datax`
  documents both judges labelled, now frozen inside that dataset folder. The
  same figures on the broader 4,708-document datax sample were 0.9472 / 0.9057,
  so the number is stable.
- Predictions here are **not** restricted to the tagging catalogue: the judge
  was asked about ~60 sensitive types, so the matching question of the model is
  whether it emits any tag at all. Restricting to the catalogue would understate
  detection.
- `current_rules`' better specificity is not evidence it is a better product.
  It is cue-anchored and conservative, which is the right shape for this task
  and the wrong shape for per-tag recall — it fails 53 of 55 tagging gates.

## Data layout

Applied the self-containment rule while doing this: the dual-judge per-judge
PII labels now live at
`2-eval/4000_datax-dualjudge-evalset-1.32k/judge_labels.json` (946 documents,
186 KB — document text discarded), keyed by that corpus's own `doc_id`. They
had been read from `datax/data/`, which moved to `3-junk/` mid-session; the
label-agreement ceiling and the `target_infeasible` verdict both rest on them,
so they must not live in a directory named for deletion. `zip -r` on the
dataset folder now carries everything needed to reproduce the agreement
analysis. `govdocs2` needed nothing — its document-level verdict is already in
its own manifest.

## Reproducing

```
training/doc_level_eval.py          # both parts: complete corpora, then dual-judge
projects/pii-priority-recall-v1/doc_level.json
```
