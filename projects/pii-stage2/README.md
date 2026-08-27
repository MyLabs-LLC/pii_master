# Pipeline spec — pii-stage2

Intake from `/sdlc-model-pipeline` on 2026-08-22 after reviewing `docs/`.
Working tree was clean at assembly. **No compute until the approval gate in this
file is signed.**

## Objective

- **Task family:** sensitive-data **tagging** (span-level PII/PHI) plus
  document-level **classification** (`NONE` / `PII` / `PHI`). Tagging is the
  ship/no-ship problem; document labels are derived from spans and scored as a
  secondary diagnostic on the frozen corpus only (Nemotron has no document
  labels).
- **Metric to optimize (compare models / loop progress):** `macro_f2` on the
  mapped Nemotron types, via `model_pipeline.evaluate("tagging", ...,
  severity_tags=HIGH_SEVERITY)`. F2 (β=2) weights recall 4× precision; macro
  over the catalogue so a dead rare class cannot hide behind EMAIL/URL.
- **Ship/no-ship gate:** `severity_recall_min` ≥ **0.90** on high-severity tags
  with support ≥ 30 on the sealed holdout. Report `severity_recall_mean` beside
  the min. Tags below support 30 are **named as excluded**, never treated as a
  pass.
- **High-severity tags:** `SSN`, `MRN`, `HEALTH_PLAN_ID`, `ACCOUNT_NUMBER`,
  `US_DRIVER_LICENSE` / `LICENSE_NUMBER`, `CREDIT_CARD` **on the Luhn-valid
  subset only**, plus Track C types once they exist (`BANK_ROUTING`,
  `FAX_NUMBER`, VIN). Credentials (`password`, `api_key`, `cvv`, `pin`) stay
  out of the HIPAA gate until a secrets profile exists.
- **Target (early stop):** `macro_f2` ≥ **0.92** **and** `severity_recall_min`
  ≥ **0.90** on `D_ho`.
- **Direction:** maximize both.
- **Diagnostics (never the gate):** micro-F1 on the 12 mapped types (today's
  committed headline: rules 0.788 / `m`+checksum-first 0.901), per-tag F2
  worst-first with precision beside recall, document PHI recall on the frozen
  corpus, 10 KB p95 latency, `n_tags_f2_zero`.
- **Latency constraint (hard, not a metric to game):** `fast` = rules-only
  ≤ 5 ms p95 / 10 KB / 1 core; `deep` = rules + student ≤ 25 ms p95 / 10 KB /
  1 core. An arm that misses its latency budget cannot promote, regardless of
  F2.

**Evaluator caveat (completeness):** Nemotron `credit_debit_card` gold is 88%
Luhn-invalid. Gating `CREDIT_CARD` recall on the full gold makes the threshold
unreachable by design. The gate uses the Luhn-valid subset (rules already
measure 0.998 recall there). The unreachable mass is reported, not swept under
a pass.

## Model / code

**Baseline (existing champion, eval-only):** Stage 1 rules engine in
`src/pii_master/` — stdlib, cue-anchored + checksum validators. Committed
external number: micro P 0.854 / R 0.732 / F1 0.788 on 100k Nemotron-PII test
docs (`docs/BASELINE_NEMOTRON.md`). Frozen-corpus scores are a regression
test, not a quality claim (`docs/BASELINE_M1.md`).

**Candidate `cnn_m` (primary challenger, already trained, not in the package):**
depthwise-separable CNN student `m` (d=128, 6 layers, 6.57 M, fp32 ONNX at
`training/artifacts/student_m.onnx`). Distilled from
`kalyan-ks/ettin-68m-nemotron-pii`. Holdout under checksum-first fusion:
micro F1 **0.901**. E2E p95 **8.04 ms** / 10 KB / 1 core (`deep` budget).
Ship **fp32**; dynamic int8 is 5–12× slower on this architecture.

**Candidate `cnn_xs` (control, eval-only unless it beats `m` on the gate):**
d=64, 4 layers. Docs: mapped F1 ties rules (0.788) alone; cascade 5.21 ms so
it does **not** enter `fast`. Kept as a latency control.

**Candidate `ettin_68m` (quality ceiling, eval-only, not a serving candidate):**
`kalyan-ks/ettin-68m-nemotron-pii` (MIT, 68.5 M, reported F1 0.963 on the
same Nemotron test split). Scores the ceiling; fails the 5 ms / 25 ms CPU
budgets by construction. Do not deploy it.

**Candidate `hf_gliner` (scout, eval-only):** `nvidia/gliner-PII` (trained on
Nemotron, reported strict F1 0.87 on Nemotron — *below* our student on mapped
types, 570 M params). Optional ONNX export (`ineersa/gliner-PII-onnx`) for a
CPU latency number. Expected to miss the serving budget; run on a **sample**
of `D_ho` (not the full 100k) unless the sample is already over budget.

**Kaggle 2024 PII Detection (scout lessons, not a candidate to serve):**
1st place = DeBERTa-v3-large ensemble + distillation + Optuna-weighted voting;
3rd = DeBERTa + Mixtral synthetic essays; 4th = Llama-3-70B synthetic data
beating their ensemble. Taxonomy is educational-essay PII (7 types), not
HIPAA. Transferable: synthetic-data diversity, name-swap augmentation,
post-processing, distillation. Not transferable: the models (PRIOR_ART:
100–400× over the 5 ms box).

**Skill-chosen extras (loop proposals, not Day-0 fits):**

1. Integrate `OnnxNerDetector` + **checksum-first** fusion (checksum rules >
   model > cue-anchored rules) + confidence threshold + `--revalidate` of
   model spans of checksummed types — `docs/DISTILLATION_RESULTS.md` §5–6,
   the missing §8 contract.
2. Candidate **windows** (reuse Stage 1 pre-scan + capitalized runs + street
   suffixes) so tokenize+decode stop dominating e2e (3.7 ms of the `xs`
   cascade is tokenizer/decoder).
3. Vectorize `decode_spans`.
4. Track C format-anchored detectors: `FAX_NUMBER`, `BANK_ROUTING` (ABA),
   `MAC_ADDRESS`, `SWIFT_BIC`, `VEHICLE_ID` (VIN), `LICENSE_NUMBER` umbrella.
5. Student hyperparams that were never swept: `lr`, `alpha`, `T`, `epochs`
   (loss still falling at epoch 3), `--soft-scope`.
6. Gazetteer tier (Aho-Corasick / FlashText) for names/cities/hospitals
   before spending more neural budget (`docs/PRIOR_ART.md` §7).

- **Framework:** rules = stdlib; student = PyTorch train / onnxruntime CPU
  serve, `intra_op_num_threads=1`. Optional extra: `pip install -e ".[ml]"`
  must not become a default-install dependency.
- **CPU vs GPU:** train on GPU if a new student fit is proposed; every
  quoted latency is 1 core. `set_cpus(1)` for any number that goes in the
  Experiment Log.
- **Entry point (script candidates):** `training/eval_student.py` and
  `eval/scripts/nemotron_eval.py` wrapped so they write `results.json` for
  `model_pipeline`. New fits go through `training/train.py`.

## Data

- **Sealed holdout `D_ho`:** `nvidia/Nemotron-PII` test split, local parquet
  `/home/lence/nemotron/test-00000-of-00001.parquet` (also in the HF hub
  cache). 100,000 documents, nobody in this repo authored. Score through the
  existing mapped-type protocol in `eval/scripts/nemotron_eval.py` (frozen
  match rule: exact `(type, start, end)`). The loop proposer never sees
  per-document `D_ho` failures — only aggregate metrics and the held-in
  evidence bundle.
- **Held-in `D_in`:** Nemotron **train** split
  `/home/lence/nemotron/train-00000-of-00001.parquet`. Weakness mining,
  threshold sweeps, and student hyperparam trials live here. A further
  stratified slice of train may be carved for selection; do not touch `D_ho`
  for that.
- **Regression corpus (not for optimisation):**
  `eval/corpus/frozen_v1.jsonl` (39 docs, append-only) +
  `eval/corpus/frozen_v1.scores.json` (`pii-master eval --fail-under`).
  Integration must add the adversarial frozen rows from
  `docs/DISTILLATION_RESULTS.md` §6 (`none-007` false PHI, truncated MRN on
  `phi-001`).
- **Split owner:** the harness. Do not reshuffle Nemotron test.
- **Crosswalk:** `src/pii_master/crosswalk.py`. Unmodelled gold is dropped,
  not counted as FN. GDPR special-category labels stay `None` (HIPAA
  profile).
- **n2c2/i2b2 2014:** DUA-restricted; out of tree; not this run.
- **`data_quality` (required this run):** leakage scan on any synthetic
  add; class balance / support per tag; real vs synth mix (Nemotron = 100%
  synth — say so); frozen-corpus tautology (authored beside detectors).

## What may change

- **Hyperparameters** (`change_mode="config"`): student `lr` (log, ~1e-4–1e-2),
  `alpha` (0.3–0.9), `T` (1–5), `epochs` (3–8), `--soft-scope`
  (`word_homogeneous` / `word_start`), fusion confidence threshold, window
  width. Rules detector confidences are ordinal and not a search surface
  unless an evidence bundle names a specific validator constant.
- **The algorithm:** the candidate list above. Each eval-only arm is one
  tracked run. Only `cnn_m` (and a new student fit if proposed) consume the
  trial budget.
- **Code** (`change_mode="code"`): keep = `git commit`, discard = revert.
  Requires a clean tree at the start of each code trial. `out_dir` lives
  **outside** the repo: `/home/lence/pii-stage2-runs`.
  - **Editable files:**
    - `src/pii_master/` — `OnnxNerDetector`, `pipeline.py` fusion policy,
      Track C detectors (`detectors/*`, `entities.py`, `validators.py`,
      `crosswalk.py`, `cli.py` for `--deep`)
    - `training/` — train/export/eval/decode/bench, window nominator
    - `eval/` — corpus **append-only**, fail-under scores file when a real
      improvement lands, reporting wrappers. **Do not change the span-match
      rule** in `nemotron_eval.py` / `evaluation.py` to move a number.
- **The data:** append frozen-corpus hard cases (false-PHI / truncated-span
  regressions). Optional: Kaggle-style name-swap / persona templates on
  `D_in` only, with a leakage scan (label/slug must not appear verbatim).
  Do not regenerate Nemotron; do not vendor n2c2.

Anything not listed here may not change — including the sealed split, the
match protocol, and `fast` staying rules-only until a student actually fits
5 ms e2e.

## Budget & guardrails

- **Primary stop:** target threshold (`macro_f2` ≥ 0.92 ∧
  `severity_recall_min` ≥ 0.90 on `D_ho`).
- **Hard cap (required so the loop cannot run unbounded):** `max_trials: 30`
  for the improvement loop (code/config/data edits). Eval-only arms
  (rules, `cnn_xs`, ettin, GLiNER sample) do not count. `max_wallclock_s:
  28800` (8 h).
- **One change at a time:** each trial touches one declared surface; keep
  only under the non-regression rule `Δ_in ≥ 0 ∧ Δ_ho ≥ 0 ∧ max(Δ_in, Δ_ho)
  > 0`. Merged edits are re-evaluated together.
- **Approval:** no compute until this spec is approved. Code-mode and any
  new training job persist an `ApprovalRecord` under
  `projects/pii-stage2/approvals/`.
- **CPU budget:** 1 core for every latency and every Experiment Log row;
  raise only for a training sweep and say so in the report.

## Tracking

- **MLflow experiment:** `pii-stage2`
- **Tracking backend:** local file-store `./mlruns` (no server). Tracking
  is never off.
- **Save the model:** yes — `log_model` (pyfunc or onnx flavor) with
  signature + `input_example` on each accepted candidate; register as
  `pii_master` and promote `@champion` only after the gate **and** the
  latency budget. File-store registry is local to this machine.
- **Extra artifacts:** `evaluation/` (JSON + per-tag F2 tables) and
  `plots/` (per-tag F2 worst-first, confusion, latency). No report upload
  to MLflow unless re-asked. No reference profile (deploy = none yet).
- **Registered-model name:** `pii_master`

## Deploy

- **Target:** none yet. Stop after the gate. Do not convert to ONNX as a
  serving step — the student is already ONNX fp32; a new conversion is only
  in scope if a new student is fitted.
- **Monitoring:** skipped (no ship).

## Reporting

- **Format:** `.md` + `.pdf` + `.xlsx`
- **Where:** `projects/pii-stage2/reports/` — `YY-MM-DD_<slug>.md` (+
  `.pdf`), `YY-MM-DD_<slug>.commands.txt`,
  `YY-MM-DD_Experiment-Log.xlsx`
- **One exported line per model × dataset.** Arms at minimum: rules ×
  Nemotron, rules × frozen, `cnn_m` × Nemotron, `cnn_m` × frozen, `cnn_xs`
  × Nemotron, ettin × Nemotron (sample if needed), GLiNER × Nemotron sample.
- **`run.json`** is kept open from the first command.

## Modules / loop

This is a **full loop** (code + data + hyperparameters + algorithm), not a
single module:

1. **Measure** — eval module: score rules and the already-trained students
   on `D_in` and `D_ho` with bootstrapped CIs; compute `macro_f2` /
   `severity_recall_*` (the committed reports lead with micro-F1 — this run
   adds the skill gate).
2. **Weakness mining** — evidence bundle from per-tag F2 worst-first
   (expected: cue-free ACCOUNT/MRN/PLAN, `US_DRIVER_LICENSE`, names,
   tokenizer-cost on `fast`).
3. **Propose** — K=2–3 diverse minimal edits routed to tune (config), tune
   (code: integrate / windows / Track C), or datagen (frozen adversarial
   rows).
4. **Validate** — same evaluator, both splits, non-regression rule.
5. **Ship** — skipped this run (`deploy: none yet`). A passing `@champion`
   still gets a model card + zip under `projects/pii-stage2/dist/` if
   promotion is requested later.
6. **Observe** — skipped.

## Known doc defects this spec is not allowed to “fix” by editing numbers

See the docs review in the intake transcript. In short: do not hand-edit
`docs/BASELINE_*.md`; regenerate. Do not collapse the frozen corpus. Do not
adopt int8 because DESIGN.md still says to.
