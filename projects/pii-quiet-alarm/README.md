# Pipeline spec — pii-quiet-alarm

A new lineage, not a continuation of `pii-priority-recall-v1`. That run's gate
was per-tag recall and its ranker was macro F2; it shipped
`pii-priority-fusion-1k-v1`, which clears 55/55 recall gates and flags
**99.95% of genuinely PII-free documents** as containing PII. This run keeps
the detection value and removes the false alarms.

Assembled 2026-08-25 from the repository, the `pii-priority-recall-v1` lineage,
every manifest under `/home/lence/workspace/data/{1-train,2-eval}`, and four
training-free probes recorded under `probe/`. **No training, evaluation beyond
those probes, dependency installation, or model download starts until the user
approves this specification.**

---

## The finding this run is built on

Two probes, run before the gate, changed what the run should be.

**1. The precision failure has a data cause, not a model cause.** Of 554,247
training documents, only **49,997 carry a labelled-clean negative**, and all of
them come from one corpus (`pii2_train_150k`). Six of the eight training
corpora contain **zero** documents annotated as PII-free. A model fitted on a
world where every document contains PII has learned the correct answer for that
world, and "everything has PII" is what it says.

| Training corpus | rows | doc-positive | labelled-clean negatives |
| --- | ---: | ---: | ---: |
| `openpii_pii_train_155k` | 151,708 | 151,708 | **0** |
| `pii2_train_150k` | 148,775 | 98,814 | 49,961 |
| `pii_trainset_100k` | 85,593 | 85,593 | **0** |
| `ai4privacy_pii_masking_train_42k` | 42,504 | 42,504 | **0** |
| `betterdataai_ner_silver_train_41k` | 41,429 | 41,429 | **0** |
| `26095_govdocs2-dualjudge-train80-14.25k` | 23,693 | 11,386 | **0** → **12,148** *(corrected loader)* |
| `nemotron_train_22k` | 21,743 | 0 | **0** |
| `16000_datax-dualjudge-trainset-5.37k` | 15,986 | 7,495 | **0** → **8,491** *(corrected loader)* |
| **total** | **531,431** | **438,929** | **49,961 → 70,600** |

> **Row counts are post-decontamination.** An external pass rewrote all eight
> training manifests at 2026-08-25 11:32 — during this session, between two
> censuses — removing the **22,816** training rows that duplicate an evaluation
> document, and verifying **0 remaining overlaps** in every corpus
> (`/home/lence/workspace/data/{train_dedup_summary,verification_results}.json`,
> backup in `backup_pre_eval_dedup/`). Evaluation manifests were **not**
> touched; every `n` in `suite.yaml` still matches. This is the leakage removal
> step 3 below called for, done more thoroughly and on disk rather than in
> memory, so that step is now a *verification* rather than a repair.
>
> Because the corpus moved underneath a run that had already started,
> `projects/pii-quiet-alarm/data_snapshot.json` pins every manifest's SHA-256
> and row counts. `training/quiet_freeze.py verify` re-checks it before the
> first fit and again before promotion; a mid-run change fails the run rather
> than being silently averaged in.

**2. The negatives exist; the loader could not see them, and the document-level
gold field was the wrong one.** The dual-judge corpora carry
`pii_entities` / `pii_classes` / `pii_sensitivity`, which are mutually
consistent on every row of all four dual-judge directories — empty entities and
empty classes occur **if and only if** `pii_sensitivity == "none"`. That is a
judge assertion of absence. `training/priority_data.normalize_row` reads only
`gold` and `pii_entities`, treats the corpora as positive-only, and therefore
discards every one of them.

The prior lineage's document-level evaluation instead read the manifest's
`label` field. That field is the judges' **document-type** verdict, and it is
orthogonal to PII presence:

| | no PII entities | has PII entities |
| --- | ---: | ---: |
| `label = positive` (govdocs2 eval) | **1,501** | 2,032 |
| `label = negative` (govdocs2 eval) | 1,193 | **1,220** |

So the number the whole customer complaint turns on was measured against a
field that does not answer the question. Corrected, and re-scored from the
frozen champion's own recorded predictions (`probe/doc_baseline_corrected.json`):

| Corpus | n | prevalence | doc recall | doc precision | doc specificity |
| --- | ---: | ---: | ---: | ---: | ---: |
| `4000_datax-dualjudge-evalset-1.32k` | 4,000 | 0.478 | 1.0000 | 0.4783 | **0.0000** |
| `6589_govdocs2-dualjudge-eval20-3.53k` | 6,549 | 0.532 | 0.9983 | 0.5320 | **0.0016** |
| `pii2_eval_30k` | 30,000 | 0.838 | 1.0000 | 0.8383 | **0.0000** |
| **equal-corpus** | | | **0.9994** | **0.6162** | **0.0005** |

Correcting the loader unlocks **20,639 real-world judge-asserted PII-free
training documents** (datax 8,491 + govdocs2 12,148) on top of pii2's 49,961 —
**70,600 negatives**, 29% of them real business documents rather than
synthetic — and **10,003 evaluation negatives** across three corpora, two of
them real-world.

A third outcome, and the reason the contract is tested rather than assumed:
**217 rows are ambiguous** — the judge listed an entity (usually a bare `State`
or `ZIP Code`) and then rated the document `sensitivity: none`. Reading those
as positive penalises correct silence; reading them as negative teaches silence
over real entities. They are excluded from document-level gold entirely,
counted by name, and capped at 2% of any corpus by
`training.quiet_data.assert_absence_contract`, which fails the run rather than
warning if the judge fields ever change shape.

**3. Threshold-tuning the existing champion is not the route.**
`probe/precision_headroom.json` sweeps the shipped model's own per-label
thresholds across 41 operating points. Its document-level score barely
separates: reaching specificity 0.90 on `pii2_eval_30k` drops document recall
to 0.22, and on govdocs2 to 0.34. The artifact does not have a precision
operating point to move to; a precision model has to be **fitted**, with
negatives, against a precision objective.

Tag-level, the same sweep is more encouraging — priority macro precision moves
0.0299 → 0.4115 on pii2 and 0.1374 → 0.5236 on `pii_holdout_20k` while macro
recall stays near 0.74–0.78 — and that is with a single crude global threshold
scale, not the per-label selection this run will do.

---

## Objective

**Precision-first multi-label sensitive-data tagging with a document-level
gate.** The question a customer actually asks first is *does this document
contain sensitive PII at all* — and it is both the best-measured question in
the suite and the one the current artifact answers worst. It becomes a hard
gate. The 16 priority tags remain the thing worth finding, and a
precision-weighted score over them decides the champion.

### Hard ship gates

Every one is judged on the **bootstrap lower bound** (`basis: ci_lower`) with a
30-instance minimum, per corpus, never pooled. A pair nobody can measure is
`NOT_MEASURABLE` and named — never counted as a pass.

| # | Gate | Basis | Measurable on |
| --- | --- | --- | --- |
| 1 | document-level **precision ≥ 0.90** | `ci_lower` | the 3 corpora with genuine negatives |
| 2 | document-level **specificity ≥ 0.85** | `ci_lower` | the 3 corpora with genuine negatives |
| 3 | document-level **recall ≥ 0.85** | `ci_lower` | all 7 corpora with positives |
| 4 | per-priority-tag **recall ≥ 0.75** | `ci_lower` | 55 measurable tag×corpus pairs |
| 5 | one-core **p95 ≤ 5 ms** on a 10 KB document | point | every arm |

Gates 1 and 2 interact deliberately, and the interaction is the point.
Precision depends on prevalence; specificity does not. On datax
(prevalence 0.478) precision ≥ 0.90 at recall 0.75 implies specificity ≥ ~0.924,
so **precision** is the binding bar there. On pii2 (prevalence 0.838)
**specificity** is the binding bar. Carrying both means neither a
low-prevalence nor a high-prevalence corpus can hide a noisy model.

Gates 3 and 4 are the two-sided protection: without them, the degenerate way to
win a precision-first run is to predict nothing. Gate 4 was relaxed from the
prior lineage's 0.90 to **0.75 by explicit user decision**, to buy precision
headroom. It is the one number in this spec that trades away a safety property,
and the model card will say so plainly.

### Ranking — precision decides the champion

Applied lexicographically, only among arms that cleared every hard gate:

1. **equal-corpus priority macro F0.5** over the 16 priority tags — precision
   weighted twice recall, macro-averaged so a dead rare tag cannot hide behind
   a strong common one. *This is the number that picks the winner.*
2. **equal-corpus macro F0.5** over the full supported catalogue.
3. **one-core p95 latency**, lower wins.

There is no F2 anywhere in the ranking. That is the whole change.

### Priority catalogue (0.75 conclusive recall floor)

`sensitive_pii_social_security_number` ·
`sensitive_pci_individual_taxpayer_identification_number_itin` ·
`sensitive_phi_medical_record_number_mrn` ·
`sensitive_phi_health_plan_beneficiary_number` ·
`sensitive_phi_patient_id_number` · `sensitive_pci_bank_account_number` ·
`sensitive_pci_credit_card_number` · `sensitive_pci_iban` ·
`sensitive_pii_passport_number` · `sensitive_pii_driver_s_license_number` ·
`sensitive_pii_military_identification_number` · `sensitive_pii_visa_number` ·
`sensitive_pii_password` ·
`sensitive_pii_personal_identification_number_pin` ·
`sensitive_pii_full_name` · `sensitive_pii_address`

### Taxonomy collapse — headline collapsed, both reported

Two independent judges agree at **F1 0.99** on whether a *name is present* and
at **0.02–0.40** on *which* name tag it is. The same holds for street-address
components. Scoring that distinction charges the model twice — a false positive
and a false negative — for a disagreement the gold itself cannot resolve.

```
sensitive_pii_given_name   ─┐
sensitive_pii_family_name  ─┼─> sensitive_pii_full_name
sensitive_pii_middle_name  ─┘
sensitive_pii_street_number_and_name ──> sensitive_pii_address
```

The **collapsed** catalogue is the headline; the **uncollapsed 61-tag** numbers
are reported beside it in every table, so nothing is hidden. The collapse is
frozen and audited before any model is fitted, and is applied identically to
gold and to predictions. In the prior lineage's probe it moved the measurable
ceiling from 0.51 to **0.64 macro / 0.79 support-weighted**.

### Diagnostics reported on every arm

Per-tag precision/recall/F0.5 worst-first per corpus; `severity_recall_min` and
`severity_recall_mean`; `f05_min`, `f05_median`, `n_tags_f05_zero`,
`n_tags_predicted_zero_times`; macro–micro gap; **prediction rate** (the thing
that tells a *silent* tag from an *unlearned* one, and the degenerate-win
detector for this run); calibration; document-level confusion matrix per
corpus; p50/p95/p99 latency; docs/sec; peak RSS; artifact size.

---

## Feasibility — probed before the budget, not after

| Target | Verdict | Ceiling evidence | Supported |
| --- | --- | --- | ---: |
| document specificity **0.90** | **plausible** | two judges over the same 946 datax documents: raw agreement 0.9545, Cohen κ 0.9080, judge-vs-judge F1 0.9487 | **0.955** |
| priority macro precision **0.90** | **unlikely** | inter-judge macro F1 0.4582 across 12 tags (support-weighted 0.4728); best point on the champion's own sweep 0.5236 | **0.458** |

Recorded in `probe/feasibility_doc_specificity.json` and
`probe/feasibility_tag_precision.json`.

**This split is why the gates and the ranker are shaped the way they are.** The
document-level question is well-measured and the target sits under every bound
the probe found. Tag-level precision is bounded near 0.46–0.52 **by the gold,
not by the model**: a correct tag the corpus failed to list is scored as a false
positive, and the judges themselves only agree at 0.46. So tag precision is
*ranked*, never *gated*, and no tag-level precision target is promised.

The honest reading: the gates are reachable; the tag-level ranker will top out
around 0.5–0.6 measured, and a report claiming more than that would be
measuring gold noise. The collapse above is the single largest lever on that
ceiling and is why it is being taken.

---

## Model / code

The packaged `pii-priority-fusion-1k-v1` is the baseline arm, re-scored under
the new policy. Five candidates, all CPU-only, all fitted on the same corrected,
deduplicated, negative-bearing training universe.

- **Candidate `docgate` — the new mechanism.** A single binary
  *does this document contain sensitive PII* head, hashed word+char features,
  fitted on all **70,711** negatives against the positives, calibrated, and
  thresholded for precision. This is the piece that has never existed; the
  document-level gates are its job.
  Search: hash width, n-gram ranges, loss, regularization, negative
  reweighting, source balance, real-vs-synthetic negative mix, read window,
  calibration family, operating threshold.
- **Candidate `precision_hash` — speed control.** The prior lineage's hashed
  cue model refitted with negatives and **per-label F0.5** threshold selection
  instead of recall-max.
  Search: as above plus per-label thresholds and priority weighting.
- **Candidate `tfidf_f05` — sparse quality control.** Shared word+char TF-IDF
  encoder, calibrated linear per-tag heads, per-label F0.5 operating points.
- **Candidate `embeddingbag_f05` — compact neural challenger.** Hashed subword
  EmbeddingBag with masked multi-label heads and an asymmetric loss tuned
  toward precision rather than recall.
- **Candidate `quiet_cascade` — the promotion candidate.** A two-stage cascade:
  `docgate` decides whether the document is flagged at all; the per-tag heads
  are consulted only when it fires, with the repository's checksum-validated
  and context rules as a precision-raising confirmation layer on the priority
  tags. This is also the **fastest** arm by construction — a clean document
  exits after stage one — which is why the latency gate and the precision gate
  are expected to be satisfied by the same design rather than traded against
  each other.

Fusion order, boosts and thresholds are learned on held-in calibration data
only. Evaluation manifests and sealed predictions never tune them.

---

## Data

### Training universe

All eight directories under `/home/lence/workspace/data/1-train/*`, with
provenance retained. Combined means all compatible evidence is used, not blind
concatenation:

1. Freeze the canonical label/alias crosswalk **and the taxonomy collapse**
   before fitting; audit both.
2. **Correct the loader** so a judge assertion of absence
   (`pii_entities == [] and pii_classes == [] and pii_sensitivity == "none"`)
   is admitted as a document-level negative, and re-verify the biconditional
   holds on every row of all four dual-judge directories before relying on it.
   This is the single code change the whole run depends on; it gets its own
   commit, its own test, and its own audit record.
3. Globally exact-deduplicate within training, and remove any exact or near
   duplicate of **any** evaluation document from training. The documented
   1,642-document `pii_trainset_100k` ↔ `pii_holdout_20k` overlap is removed
   from **training**, never from evaluation. **Re-run in full**: the prior
   audit's exclusion list never saw the 20,714 newly admitted rows.
4. Preserve multi-source membership; source-balanced sampling and weights so
   the largest or most synthetic corpus cannot dominate. Negative supply is
   balanced explicitly between real (20,714) and synthetic (49,997) — a
   document gate fitted only on synthetic negatives will not transfer to the
   customer's real files, and govdocs2 is the closest thing in the suite to
   those files.
5. Masked / positive-unlabelled losses stay in place for partial-label corpora.
   **An unannotated tag is still never a negative.** The correction admits
   *document-level* absence where a judge asserted it; it does not turn
   coarse-labelled corpora into complete ones.
6. Carve held-in train/calibration splits by source/entity/hash group. The
   external evaluation directories remain sealed from proposal and tuning.

Targeted offline augmentation may add hard-negative variants for weak priority
tags — documents that look like a match and are not (formatting, cue words
without values, near-miss checksums, business phone/address without personal
context). These are the examples that buy precision. They stay in held-in
training, carry an oracle/provenance marker, and pass the exact-label/slug
leakage scan before use.

### Sealed evaluation matrix — one corpus at a time

All eight directories under `/home/lence/workspace/data/2-eval/*`, scored
independently. What each can measure is declared in `suite.yaml` and enforced:
three corpora carry document-level negatives; five have prevalence 1.0 and
report `NOT_MEASURABLE` for the document-level precision gates rather than a
free pass; `nemotron_eval_6k` recovers no sensitive-tag positives at all and
reports nothing a gate reads.

With five candidates plus the baseline across eight corpora the matrix is
**48 arms**, one MLflow child run and one Experiment Log row each.

---

## What may change

- **Config:** candidate hyperparameters, source/provenance weights, negative
  mix, masked-loss settings, read window, calibration, per-tag thresholds,
  cascade thresholds, fusion weights.
- **Algorithm:** the five candidates above.
- **Code** (git-backed, one kept commit per focused edit):
  - the loader correction and its audit/test under `training/`
  - the frozen taxonomy collapse and its audit
  - new candidate trainers, the document-gate trainer, the cascade, a fixed
    prediction contract, calibrator, evaluator adapter, one-core benchmark,
    exporter and package verification under `training/`
  - narrowly scoped serving hooks in `src/pii_master/` only if the cascade
    needs them
  - tests under `tests/`
  - run/spec/audit/report/package artifacts under `projects/pii-quiet-alarm/`
- **Data:** training-only deduplication, label masks, source weights, negative
  balancing, hard-negative augmentation.

Evaluation data, evaluation labels, scorer semantics, split membership, metric
formulas, minimum support and every gate threshold are **immutable**.

The workspace has unrelated untracked files. Code trials run in an isolated
clean clone/worktree under `/tmp`; only individually gated commits are applied
back. Existing user files are not staged, overwritten or removed.

---

## Budget & guardrails

- **Search budget: 1,000 trials maximum** — the user's approved figure.
  `docgate` 250 · `precision_hash` 200 · `tfidf_f05` 175 ·
  `embeddingbag_f05` 175 · `quiet_cascade` calibration/fusion 200.
  Reallocation between candidates is allowed; the 1,000 total is not.
- **Wall-clock cap:** 12 hours for the full loop.
- **Early stop:** only when every hard gate passes **and** priority macro F0.5
  stops improving by more than 0.002 over 100 consecutive trials.
- **Proposal width:** K = 2–3 diverse, single-mechanism candidates per round.
- **Training CPU:** up to all 32 cores, because sweep wall-clock matters.
  **Every published latency and throughput number is re-measured on exactly one
  core**, and the report says so.
- **Acceptance:** a kept edit must satisfy the two-split non-regression rule
  `Δ_in ≥ 0 ∧ Δ_ho ≥ 0 ∧ max(Δ_in, Δ_ho) > 0` on priority macro F0.5, must not
  turn any previously passing gate into a failure, and must not reduce any
  priority tag's measurable recall below 0.75. Merged edits are re-evaluated
  together.
- **Evaluator immutability:** no proposal may edit evaluation manifests, gold,
  split membership, metric formulas, minimum support, or any threshold above.
- **Leakage:** exact and near-duplicate audit before fitting, re-run over the
  newly admitted rows; the exact-label/slug scan on every augmented example.
  Any suspiciously perfect score halts promotion for investigation.
- **Licensing:** sources include AI4Privacy and NonCommercial material. The
  result is research/internal-only and must not be redistributed or
  commercially deployed until rights are confirmed.

---

## Orchestration · Tracking · Reporting · Packaging

- **Mode:** direct standalone engine; no ZenML. Every Python invocation goes
  through the skill's Python 3.13 `uv run` bootstrap.
- **MLflow experiment:** `pii-quiet-alarm`, local file store under the project.
  Save models: **yes**, with signature and input example; a registry version per
  accepted candidate. Extra artifacts: resolved spec/config, data-quality and
  leakage records, the loader-correction audit, the frozen collapse map,
  evaluation JSON and per-tag tables, calibration/threshold tables,
  document-level confusion matrices, worst-first plots, environment lock, run
  report, command history, experiment workbook. No reference profile —
  deployment and monitoring are out of scope.
- **Registered model:** `pii_quiet_alarm`; only a fully verified, gate-passing,
  latency-passing artifact is promoted to `@champion` via `promotion_v2`.
- **Reporting:** Markdown + PDF + XLSX plus the complete command history, under
  `projects/pii-quiet-alarm/reports/`. `run.json` is opened before the first
  post-approval command and records successes **and failures** as they occur.
  `## Results` first, `## TL;DR` second; one row per model × corpus; command
  output lives only in the `.commands.txt` archive.
- **Packaging:** a passing champion is packaged under
  `projects/pii-quiet-alarm/dist/` with a complete `MODEL_CARD.md`, a
  self-contained entry point, config, weights, licence/provenance notes,
  evidence and bundle-owned verification; the zip must reproduce its recorded
  holdout metrics within the fixed 0.01 tolerance, re-scored **through the
  bundle's own code**.
- **If no candidate clears every gate:** do not promote or package a false
  champion. Report the best non-passing candidate, the exact failed pairs, and
  the number the evidence does support.

## Modules / loop

Full loop: **measure → weakness mining → tune/classify/datagen → validate →
package**. Deployment and monitoring are deliberately excluded.
`pii-priority-fusion-1k-v1` is the baseline. This is a new lineage because its
gate, ranker, taxonomy, loader semantics and document-level gold all changed.
