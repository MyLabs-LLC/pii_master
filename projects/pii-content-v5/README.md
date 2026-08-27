# Pipeline spec — pii-content-v5

Add a **content** signal to the sensitive-data cascade. Every model in this repo
so far reads documents as a bag of hashed character/word n-grams: it knows that
`123-45-6789` looks like an SSN, and nothing about the sentence around it. This
project tests whether a transformer's contextual signal, compressed into
something that can serve on one CPU core, adds anything that bag does not already
have.

## The chain, and why it has four stages

| stage | model | what it is | where it runs |
| --- | --- | --- | --- |
| **v5a** | `pii-content-v5a-ettin-ft` | `kalyan-ks/ettin-68m-nemotron-pii` fine-tuned on this repo's corpora | GPU |
| **v5b** | `pii-content-v5b-m2v` | that encoder distilled by model2vec into a **static token embedding table** | GPU → CPU |
| **v5c** | `pii-content-v5c-tagger` | a document tagger trained on those token embeddings | CPU |
| **v5d** | `pii-content-v5d-fused` | v5c fused with `cascade_scorecard61` | 1 core |

**v5a is a teacher, not a product, and this must stay explicit.** The shipping
constraint is p95 **≤ 8 ms** on one CPU core (revised from the repo README's
original 5 ms on 2026-08-26, then from an interim 10 ms). A 68M-parameter
ModernBERT over a 12,000-character document is orders of magnitude past any of
those — this repo's own distilled dilated CNN already costs 8.5–15.5 ms. The
transformer can never be the served artifact. Only v5b/v5c can, which is the entire reason
model2vec is in the chain rather than "just deploy the encoder".

**The content path reads 6,000 characters, not 12,000.** Measured on the real
static table: 12,000 chars and 6,000 chars both yield the tokenizer's full 1,024
tokens, but cost 6.946 ms and 2.533 ms respectively on one core. Everything past
~6,000 characters is scanned and then discarded by truncation, so reading the
cascade's full window would deliver the model no extra information and miss even
a 10 ms budget. It also matches training: `v5_finetune` truncates at 1,024 tokens,
and truncation keeps the *first* 1,024 — the same ones 6,000 characters yield.

**The distillation is token-based, not sentence-based.** model2vec's `StaticModel`
mean-pools token vectors into one document vector by default. Over a 12,000-
character document that averages a lone SSN into nothing — the exact failure the
cascade's per-tag heads exist to avoid. v5c therefore consumes **per-token**
vectors and does its own aggregation, and v5b is evaluated on whether the token
table preserves identifier-bearing tokens, not on sentence-similarity benchmarks.

## Objective

| | |
| --- | --- |
| Task family | multi-label document tagging (61 labels), with the existing document gate in front |
| Metric to optimize | `micro_f1`, then `precision_micro` — the precision-led ranking |
| Direction | maximize |
| Target | **beat `cascade_scorecard61` on micro F1 (0.7299) without losing priority-recall gates (25/55) or the 8 ms fused budget** |
| Gate | per-tag recall ≥ 0.90 `ci_lower` on the 16 priority tags; one-core fused p95 ≤ 8 ms, **measured** by `v5_fuse`, not inferred by adding the two arms' separate numbers |

The target is a *non-regression* target on purpose. Nothing in this project's
history has cleared the ship gate; a new architecture that improves the ranker
while holding the gate count is the result worth having, and one that improves
the ranker by losing gates is the failure mode already seen twice.

## Model / code

### v5a — fine-tuning, two stages

Chosen because **only one of eight training corpora carries token-level gold**.
`85593_pii_trainset_85.59k/spans.jsonl` has character offsets whose `tag_id`s are
already this repo's slugs (`sensitive_pii_family_name`); the other seven carry
document-level `gold` lists. Training a token head on all eight is not possible
against gold that does not exist.

1. **Span stage** — keep the existing `ModernBertForTokenClassification` head and
   continue training on the 85,593-document span corpus, re-headed from its 55
   Nemotron entity types (107 BIO labels) onto this repo's tag ids.
2. **Document stage** — swap in a 61-way sigmoid document head and train on all 8
   corpora, 531,431 rows.

Both stages update the encoder, and the encoder is what v5b distils, so the
static table inherits span-level *and* document-level signal.

**Measured outcome (checkpoint, 2026-08-26).** Both models scored on the *same*
2,400 calibration documents: 50 of 61 tags improved by >0.02, 8 regressed, mean
delta **+0.4057** (+0.5375 restricted to support ≥ 50). The span stage alone had
caused severe single-corpus forgetting — CVV 0.8780 → 0.0581, religion 0.8302 →
0.1290 — and the document stage largely repaired it (CVV back to 0.4545). Residual
regressions: `mac_address` −0.25, `password` −0.19, CVV −0.13.

A correction worth carrying: the first feasibility probe read **0.0000** on
`ssn`/`mrn`/`itin`/`cvv`/`health_plan_beneficiary_number` and that was reported as
the central risk. It was an artifact of that probe's corpus, where those tags had
2–12 gold documents. Re-measured with real support the as-shipped model already
scored SSN 0.5730, ITIN 0.8698, CVV 0.8780.

The upstream label space overlaps ours heavily (ssn, email, phone_number,
credit_debit_card, cvv, medical_record_number, health_plan_beneficiary_number,
password, pin, ipv4/ipv6, mac_address, street_address, postcode, first/last_name,
…) but is missing `iban`, `passport_number`, `visa_number`,
`military_identification_number`, `income`, `marital_status` and the PHI clinical
set (`icd_code`, `medical_condition`, `medication`, `patient_id_number`). Those
tags come only from the fine-tune, so per-tag results on them are the ones to
read first.

### v5b — distillation

`model2vec.distill` over the fine-tuned encoder. Output is a vocabulary-sized
static embedding matrix. Recorded: vocabulary size, dimensionality, PCA variance
retained, and on-disk bytes — the last one is a serving constraint, not trivia
(the shipped cascade is 28 MB).

### v5c — the tagger

Per-token vectors → aggregation → 61 per-tag decisions, trained on the same 8
corpora under `quiet_fit.carve_holdin` (15% by stable document hash, training
corpora only). Thresholds selected by
`h2h_thresholds_v4.select_per_label(corrected_cap=True)`, the same rule as
`cascade_scorecard61`, so the two arms differ by features and not by calibration
policy.

### v5d — fusion

Fused with `projects/pii-scorecard-60/models/cascade_scorecard61` — the 61-label
model, so the two share a label space exactly and no tag has nowhere to go. (The
originally named `pii-cascade-balanced-v3` is 58-label and its two hand-set
thresholds were selected on the sealed eval set; fusing into it would carry both
problems forward.) Fusion policy is explicit and per-tag, and is itself selected
on the calibration carve.

## Data

| | |
| --- | --- |
| Training | `/home/lence/workspace/data/1-train`, all 8 corpora, 531,431 rows |
| Span supervision | `85593_pii_trainset_85.59k/spans.jsonl` only |
| Evaluation | `/home/lence/workspace/data/2-eval`, all 8 corpora, sealed |
| Catalogue | the 61-label scorecard catalogue, `projects/pii-scorecard-60/cache/catalogue.json` |
| Suite | copied from `pii-scorecard-60`, unchanged |
| Policies | copied from `pii-scorecard-60`, then **two edits**: the serving constraint raised to 8 ms, and both renamed `pii-content-v5-*` (the copies still carried scorecard-60's names, so their decision records would have claimed to be another project's) |

## What may change

The encoder weights (v5a), the distillation hyperparameters (v5b), the tagger and
its thresholds (v5c), the fusion policy (v5d).

Not allowed to change: `training/h2h_eval.py`, the sealed corpora, the catalogue,
the suite, and the two policy files. `cascade_scorecard61` is frozen — v5d fuses
with it, it is not refit.

## Budget & guardrails

Staged, with a checkpoint between each: v5a and v5b are worthless if v5c cannot
beat the cascade, and v5c is worthless if it cannot serve inside 8 ms. Each stage
reports before the next is funded.

## Tracking

MLflow at `sqlite:///projects/pii-content-v5/mlflow.db`, experiment
`pii-content-v5`. One parent run, one child per stage.

## Reporting

`.md` + `.pdf` + `.commands.txt` + the experiment-log workbook, under
`projects/pii-content-v5/reports/`.
