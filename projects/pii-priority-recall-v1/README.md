# Pipeline spec — pii-priority-recall-v1

Intake assembled on 2026-08-25 from the repository, the unfinished
`projects/pii-stage2` lineage, and every manifest under
`/home/lence/workspace/data/{1-train,2-eval}`. This is a new versioned loop:
the prior approval does not cover the new datasets, targets, code, or budget.
**No training, evaluation, dependency installation, or model download starts
until the user approves this specification.**

## Objective

- **Task family:** recall-first multi-label sensitive-data tagging. Native
  document-type/coarse-entity labels in auxiliary corpora are learned through
  masked auxiliary heads; missing sensitive-tag annotations are never treated as
  negatives.
- **Hard ship gate:** on every evaluation corpus separately, every measurable
  priority tag must have recall >= **0.90**. A tag/corpus pair is measurable only
  with at least 30 positive examples. Thin or absent pairs are reported as
  `INSUFFICIENT EVIDENCE`, never passed. A point estimate >= 0.90 whose bootstrap
  95% lower bound is below 0.90 is `INCONCLUSIVE`, not ship-ready.
- **Primary optimization metric:** equal-corpus mean **macro F2** over the full
  supported sensitive-tag catalogue after all hard recall gates are satisfied.
  Each corpus also reports its own macro F2; no pooling may hide a weak corpus.
- **Target:** all measurable priority recall gates pass and equal-corpus macro F2
  >= **0.90**. Until the hard gates pass, selection ranks candidates by the
  worst priority-tag recall, then the number of passed priority gates.
- **Tie-breaker:** equal-corpus mean **micro F1**, as requested. If still tied,
  lower one-core p95 latency wins.
- **Latency gate:** the promotable `fast` artifact must process a 10 KB document
  at <= **5 ms p95 on one CPU core**, remain under 4 GB peak RSS, and report
  docs/sec. A quality-only arm may be reported up to 25 ms p95 but cannot win the
  `fast` alias.
- **Read-depth ladder:** benchmark paired passes at **1,000**, **2,500**,
  **10,000**, and **20,000** characters for every finalist. Record recall/F2
  deltas against 20k together with p50/p95/p99 latency, docs/sec, peak RSS, and
  artifact size. A shorter pass may win only if it preserves every priority
  recall gate; speed never excuses a missed high-value tag.
- **Diagnostics:** per-tag precision/recall/F2 worst-first per corpus,
  `severity_recall_min`, `severity_recall_mean`, `f2_min`, `f2_median`,
  `n_tags_f2_zero`, `n_tags_f2_below_10pct`, micro/macro gap, top-k ladder,
  calibration, prediction rate, p50/p95/p99 latency, throughput, peak RSS, and
  artifact size.

### Priority catalogue (90% recall gate)

1. `sensitive_pii_social_security_number`
2. `sensitive_pci_individual_taxpayer_identification_number_itin`
3. `sensitive_phi_medical_record_number_mrn`
4. `sensitive_phi_health_plan_beneficiary_number`
5. `sensitive_phi_patient_id_number`
6. `sensitive_pci_bank_account_number`
7. `sensitive_pci_credit_card_number`
8. `sensitive_pci_iban`
9. `sensitive_pii_passport_number`
10. `sensitive_pii_driver_s_license_number`
11. `sensitive_pii_military_identification_number`
12. `sensitive_pii_visa_number`
13. `sensitive_pii_password`
14. `sensitive_pii_personal_identification_number_pin`
15. `sensitive_pii_full_name`
16. `sensitive_pii_address`

Name component tags (`given_name`, `family_name`, `middle_name`) and address
component tags (`street_number_and_name`, `city`, `state`, `zip_code`) remain in
the full catalogue and are reported individually. They do not silently substitute
for the two requested generic tags; any hierarchy/alias mapping is fixed and
audited before model fitting.

## Model / code

The current packaged deep scanner is the external baseline. Three learned
candidates are trained independently on the same globally deduplicated combined
training universe; a fourth hybrid candidate fuses the best learned scorer with
the existing checksum/context rules.

- **Candidate `hash_sgd` — speed control**
  - Streaming word + character hashing features; one-vs-rest calibrated SGD.
  - Search: word/character n-grams, hash width, loss, regularization, class and
    source weights, read window, and per-label threshold.
  - Expected strength: smallest model, quickest fit and inference.
- **Candidate `tfidf_linear` — sparse quality control**
  - Shared word + character TF-IDF encoder with calibrated linear per-tag heads.
  - Search: feature caps, sublinear TF, min/max document frequency, n-grams,
    regularization, priority weights, and per-label threshold.
  - Expected strength: strong lexical/context quality while remaining CPU-fast.
- **Candidate `embeddingbag_asl` — compact neural challenger**
  - Small hashed subword EmbeddingBag/temporal-pooling encoder with masked
    multi-label heads, asymmetric/focal loss, priority-label weighting, and ONNX
    export.
  - Search: embedding width, pooling, dropout, loss asymmetry, learning rate,
    source sampler, priority oversampling, read window, and thresholds.
  - Expected strength: names, addresses, and cue-light identifiers without a
    transformer-sized latency or memory cost.
- **Candidate `hybrid_priority` — promotion candidate**
  - Best learned scorer plus the repository's checksum-validated/context rules,
    safe hierarchy closure, and label-specific calibration/thresholds.
  - Fusion order and boosts are learned on held-in calibration data only.
    Evaluation manifests and sealed holdout predictions never tune them.

All learned candidates use the same combined corpora, split masks, canonical
taxonomy, deduplication result, priority augmentation, and calibration protocol.
No candidate gets extra holdout information.

## Data

### Combined training universe

Every dataset directory under `/home/lence/workspace/data/1-train/*` is ingested
with provenance retained:

- `betterdataai_ner_silver_train_41k`
- `ai4privacy_pii_masking_train_42k`
- `nemotron_train_22k`
- `openpii_pii_train_155k`
- `pii_trainset_100k`
- `pii2_train_150k`
- `16000_datax-dualjudge-trainset-5.37k`
- `26095_govdocs2-dualjudge-train80-14.25k`

“Combined” means all compatible evidence is used, not blind concatenation:

1. Freeze a canonical label/alias crosswalk before fitting.
2. Verify paths, hashes, missing files, class support, provenance, real/synthetic
   mix, and label consistency.
3. Globally exact-deduplicate within training and remove any exact/near duplicate
   of **any** evaluation document from training. The documented 1,642-document
   overlap between `pii_trainset_100k` and `pii_holdout_20k` must be removed from
   training, not from evaluation.
4. Preserve multi-source membership and use source-balanced sampling/weights so
   the largest or most synthetic corpus cannot dominate.
5. Use masked/positive-unlabelled losses for corpora with partial, coarse, or
   native document-type labels. Never interpret an unannotated tag as a negative.
6. Carve held-in train/calibration splits by source/entity/hash group. The fixed
   external evaluation directories remain sealed from proposal and tuning.

Targeted offline augmentation may add formatting, OCR, separator, country,
cue/no-cue, casing, name, and address variants for weak priority tags. Generated
examples stay in held-in training only, carry an oracle/provenance marker, and
must pass the exact-label/slug leakage scan before use.

### Sealed evaluation matrix — one corpus at a time

Every candidate is run independently against every directory under
`/home/lence/workspace/data/2-eval/*`:

- `openpii_pii_eval_38k`
- `6589_govdocs2-dualjudge-eval20-3.53k`
- `ai4privacy_pii_masking_eval_10k`
- `pii_holdout_20k`
- `nemotron_eval_6k`
- `4000_datax-dualjudge-evalset-1.32k`
- `betterdataai_ner_silver_eval_10k`
- `pii2_eval_30k`

Sensitive-catalogue corpora receive the full recall/F2/F1 evaluation. Coarse or
native document-type corpora receive only metrics supported by their gold,
including safely mapped PII entities where possible. Unsupported priority tags
are `NOT ASSESSABLE`; they are never inferred from model predictions or counted
as passed. No pooled evaluation replaces the eight individual rows.

Every model x dataset cell becomes one MLflow child run and one Experiment Log
row. With four candidates and eight corpora, the final matrix has **32 arms**
before threshold/calibration diagnostics.

## What may change

- **Config:** candidate hyperparameters, source/provenance weights, masked-loss
  settings, priority oversampling, read window, calibration, per-tag thresholds,
  and evidence-fusion weights.
- **Algorithm:** the four candidates above.
- **Code (git-backed, one kept commit per focused edit):**
  - new unified loaders/crosswalk and leakage/data-quality checks under `training/`
  - new candidate trainers, fixed prediction contract, calibrator, evaluator
    adapter, one-core benchmark, exporter, and package verification under
    `training/`
  - only evidence-backed, narrowly scoped additions to `src/pii_master/` if the
    final hybrid needs a serving hook
  - tests under `tests/`
  - project run/spec/audit/report/package artifacts under
    `projects/pii-priority-recall-v1/`
- **Data:** training-only deduplication, label masks, source weights, and targeted
  synthetic augmentation. Evaluation data, evaluation labels, scorer semantics,
  and split membership are immutable.

The current workspace has unrelated untracked files. Code trials therefore run
in an isolated clean clone/worktree under `/tmp`; only individually gated commits
are applied back. Existing user files are not staged, overwritten, or removed.

## Budget & guardrails

- **Search budget:** 300 trials for each of the three learned models (900 total),
  plus up to 100 focused calibration/fusion candidates: **1,000 maximum trials**.
  This is the user's approved expanded budget; the loop may stop early only when
  the complete recall/F2 target is reached or the wall-clock cap expires.
- **Wall-clock cap:** 12 hours for the full loop. Stop earlier only when all
  measurable priority gates and the macro-F2 target pass.
- **Proposal width:** K=2-3 diverse, single-mechanism candidates per round.
- **Training CPU:** up to all 32 cores because sweep wall-clock matters; every
  published latency and throughput measurement is repeated on exactly one core.
  No working NVIDIA GPU is currently available.
- **Acceptance:** a kept edit must not regress equal-corpus macro F2 on held-in or
  sealed evaluation, must not turn any previously passing priority tag/corpus
  pair into a failure, and must satisfy the loop's two-split non-regression rule.
  Merged edits are re-evaluated together.
- **Evaluator immutability:** no proposal may edit evaluation manifests, gold,
  split membership, metric formulas, minimum support, or the 0.90 gate.
- **Leakage:** exact and near-duplicate audit runs before fitting; any suspiciously
  perfect score halts promotion for investigation.
- **Licensing:** the combined sources include AI4Privacy and NonCommercial data.
  The resulting model is research/internal-only and must not be redistributed or
  commercially deployed until the relevant rights are confirmed.

## Orchestration

- **Mode:** direct standalone engine. No local ZenML server is installed or
  running, and adding orchestration does not improve model quality or inference
  latency.
- Every Python invocation uses the skill's Python 3.13 `uv run` bootstrap.

## Tracking

- **MLflow experiment:** `pii-priority-recall-v1`
- **Backend:** local file store under the project (no tracking URI was configured).
- **Save models:** yes. Log each candidate's reproducible best/final artifact with
  signature and input example; create a versioned history for accepted candidates.
- **Extra artifacts proposed for approval:** resolved spec/config, data-quality
  and leakage records, evaluation JSON/per-tag tables, calibration/threshold
  tables, worst-first plots, environment lock, run report, command history, and
  experiment workbook. No reference profile because deployment/monitoring is out
  of scope.
- **Registered model:** `pii_priority_recall`; promote only a fully verified,
  gate-passing, latency-passing artifact to `@champion`.

## Deploy and package

- **Deploy target:** none yet.
- A passing champion is still packaged under
  `projects/pii-priority-recall-v1/dist/` with a complete `MODEL_CARD.md`,
  self-contained entry point, config, weights, licence/provenance notes, evidence,
  and bundle-owned verification; the zip must reproduce its recorded holdout
  metrics within the fixed 0.01 tolerance.
- If no candidate clears every required gate, do not promote or package a false
  champion. Report the best non-passing candidate and the exact failed pairs.

## Reporting

- **Format:** Markdown + PDF + XLSX (default), plus complete command history.
- **Location:** `projects/pii-priority-recall-v1/reports/`.
- `run.json` is opened before the first post-approval command and records successes
  and failures as they occur.
- `## Results` is first and `## TL;DR` second. Every model x corpus arm has its
  own row; command output stays only in the `.commands.txt` archive.

## Modules / loop

Full loop: **eval -> weakness mining -> tune/classify/datagen -> eval -> package**.
Deployment and monitoring are deliberately excluded. The current `pii-stage2`
champion is the baseline; this run is a new lineage because its task catalogue,
datasets, evaluator matrix, priority gate, and budget changed.
