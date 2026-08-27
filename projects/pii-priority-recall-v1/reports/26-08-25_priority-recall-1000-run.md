# PII priority recall — 1,000-trial run

## Results

The selected artifact is the **1,000-character per-label fusion**. It passed all
**55/55 measurable priority tag–corpus recall gates conclusively**: every point
recall was at least 0.90 and every 95% document-bootstrap lower bound was at
least 0.90. Worst observed priority recall was **0.9888** and the lowest lower
bound was **0.9811**.

It did not reach the quality aspiration: equal-corpus macro F2 is **0.4835**
(95% corpus-bootstrap interval **0.3501–0.5892**) versus the 0.90 target. The
micro F1 tie-break is **0.3812** (95% CI **0.2347–0.5348**). This is therefore a
high-recall internal/research candidate, not a precision-ready production
redactor.

| Model / operating point | Macro F2 | Micro F1 | Priority point gates | Worst recall |
| --- | ---: | ---: | ---: | ---: |
| Current rules baseline | 0.0985 | 0.2000 | 2/55 | 0.0000 |
| Hash recall-max, 20k | 0.4234 | 0.3452 | 55/55 | 0.9966 |
| Hash F2, 20k | 0.4871 | 0.5098 | 49/55 | 0.1800 |
| TF-IDF linear, 20k | 0.4832 | 0.4929 | 50/55 | 0.1600 |
| EmbeddingBag ASL, 20k | 0.4647 | 0.4691 | 50/55 | 0.4200 |
| Two-head priority hybrid, 20k | 0.4466 | 0.3634 | 55/55 | 0.9966 |
| Full per-label fusion, 20k | 0.4798 | 0.3791 | 55/55 | 0.9966 |
| **Selected full fusion, 1k** | **0.4835** | **0.3812** | **55/55** | **0.9888** |

### Requested read-pass speed comparison

One-core end-to-end latency was measured on the same deterministic stratified
sample of 1,000 documents. Quality was re-scored on all eight holdouts at each
read depth. `c_read` is the text actually read by the model.

| Read pass | p50 ms | p95 ms | p99 ms | Mean ms | Docs/s | p95 speed vs 20k | p95 saved vs 20k | Macro F2 | Micro F1 | Priority gates | Worst recall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **1,000** | **0.825** | **2.200** | **2.386** | **1.088** | **919.1** | **4.01×** | **6.629 ms** | **0.4835** | **0.3812** | **55/55** | **0.9888** |
| 2,500 | 1.055 | 4.211 | 4.620 | 1.754 | 570.1 | 2.10× | 4.618 ms | 0.4797 | 0.3801 | 55/55 | 0.9686 |
| 10,000 | 1.075 | 6.766 | 8.220 | 2.440 | 409.9 | 1.30× | 2.063 ms | 0.4799 | 0.3792 | 55/55 | 0.9966 |
| 20,000 | 1.064 | 8.829 | 11.885 | 2.822 | 354.4 | 1.00× | 0.000 ms | 0.4798 | 0.3791 | 55/55 | 0.9966 |

The 1k pass is **4.01× faster at p95** than 20k and saves 6.63 ms/document. It
also has slightly higher macro F2 (+0.0036), but lowers the worst priority
recall by 0.0079. The 1k artifact passes the fast gate because its effective
read ceiling is 1k and p95 is 2.20 ms. Full 10k and 20k fusion passes exceed
the 5 ms p95 aspiration (6.77 and 8.83 ms); they must not be described as
meeting it.

### Holdout matrix, one corpus at a time

| Holdout corpus | Rows | Gold mode | Macro F2 | Micro F1 | Conclusive priority gates | Worst recall |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 4000_datax-dualjudge-evalset-1.32k | 4,000 | Positive-only | N/A | N/A | 1/1 | 1.0000 |
| 6589_govdocs2-dualjudge-eval20-3.53k | 6,589 | Positive-only | N/A | N/A | 2/2 | 0.9888 |
| ai4privacy_pii_masking_eval_10k | 10,626 | Complete | 0.4995 | 0.3529 | 10/10 | 0.9991 |
| betterdataai_ner_silver_eval_10k | 10,360 | Complete | 0.2313 | 0.2022 | 4/4 | 1.0000 |
| nemotron_eval_6k | 5,617 | Positive-only | N/A | N/A | 0/0 | N/A |
| openpii_pii_eval_38k | 38,937 | Complete | 0.6297 | 0.6613 | 9/9 | 0.9997 |
| pii2_eval_30k | 30,000 | Complete | 0.4930 | 0.2082 | 15/15 | 0.9958 |
| pii_holdout_20k | 20,000 | Complete | 0.5638 | 0.4813 | 14/14 | 0.9944 |

OpenPII is the most flattering complete corpus and BetterDataAI is the weakest.
The number to quote across corpora is the equal-corpus macro F2 of 0.4835, not
OpenPII's 0.6297.

### Priority-tag evidence

Each row is the minimum across separately scored corpora having at least 30
positives for that tag. The confidence value is the lowest lower 95% bootstrap
bound across those corpora.

| Priority tag | Corpora measured | Total support | Minimum recall | Lowest 95% bound | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Social Security number | 4 | 12,551 | 0.9958 | 0.9861 | PASS |
| ITIN | 3 | 10,543 | 1.0000 | 1.0000 | PASS |
| Medical record number | 3 | 1,104 | 1.0000 | 1.0000 | PASS |
| Health-plan beneficiary number | 2 | 890 | 0.9944 | 0.9850 | PASS |
| Patient ID | 1 | 326 | 1.0000 | 1.0000 | PASS |
| Bank account number | 3 | 1,438 | 0.9983 | 0.9946 | PASS |
| Credit-card number | 4 | 12,156 | 1.0000 | 1.0000 | PASS |
| IBAN | 2 | 740 | 1.0000 | 1.0000 | PASS |
| Passport number | 4 | 19,419 | 0.9991 | 0.9971 | PASS |
| Driver's licence number | 4 | 12,666 | 1.0000 | 1.0000 | PASS |
| Military ID | 2 | 18,090 | 0.9998 | 0.9996 | PASS |
| Visa number | 3 | 18,317 | 0.9998 | 0.9995 | PASS |
| Password | 4 | 1,723 | 0.9992 | 0.9975 | PASS |
| PIN | 3 | 613 | 1.0000 | 1.0000 | PASS |
| Full name | 6 | 60,902 | 0.9972 | 0.9952 | PASS |
| Address | 7 | 26,473 | 0.9888 | 0.9811 | PASS |

## TL;DR

- Exactly **1,000** MLflow-tracked trials were completed: 300 hash, 300 TF-IDF,
  300 EmbeddingBag ASL, and 100 fusion candidates.
- The selected 1k fusion passed every measurable 90% priority-recall gate with
  conclusive 95% bootstrap evidence and runs at 2.20 ms p95 on one core.
- It is 4.01× faster at p95 than the same 20k fusion.
- Macro F2 remains only 0.4835, so use it for high-recall triage with downstream
  review—not as a stand-alone redactor or proof that a document is PII-free.

## Run metadata

| Field | Value |
| --- | --- |
| Date | 2026-08-25 |
| Project | pii-priority-recall-v1 |
| Task | 61-label sensitive PII/PCI/PHI document tagging |
| Selection | Priority gates → equal-corpus macro F2 → micro F1 → one-core p95 |
| Training inputs | All eight `/home/lence/workspace/data/1-train/*` directories |
| Evaluation inputs | Eight `/home/lence/workspace/data/2-eval/*` directories, separate arms |
| Approved search budget | 1,000 trials |
| Trials completed | 1,000 |
| MLflow backend | Local project file store |
| Source commit | d662a3a |

## Method and model search

The unified loader retained source provenance and annotation completeness. A
missing label in partial/coarse corpora was never treated as a negative. The
frozen evaluator used document-level F2 with beta 2, equal corpus weighting,
and micro F1 only after the recall gate and macro F2 comparison.

The 1,000-trial allocation was exhausted exactly as approved:

- 300 compact hash/cue operating points.
- 300 sparse TF-IDF-derived linear operating points.
- 300 low-rank EmbeddingBag ASL operating points.
- 100 per-label Boolean fusion/calibration candidates.

All candidates trained from the same combined, leakage-filtered training
universe and were evaluated against each holdout independently. Priority tags
in the winner use the recall-max head; other labels use the best held-in
per-label fusion strategy. No evaluation label tuned thresholds or fusion.

## Data quality and leakage controls

- 554,247 indexed training rows and 126,129 indexed evaluation rows were
  audited; the evaluation side contains 121,179 unique text hashes.
- 19,668 exact text hashes crossed train/evaluation. All 22,816 affected
  training rows were excluded from fitting; evaluation was left untouched.
- Complete-catalogue metrics are reported only for five corpora. Three
  partial/coarse corpora contribute known-positive recall only.
- Eleven unreadable/missing GovDocs evaluation files were recorded rather than
  silently counted as negatives.
- The data include real, synthetic, silver, and machine-judged sources. Public
  benchmark contamination beyond exact supplied-data overlap remains possible.

## Limitations and decision

**Decision:** package the 1k fusion as a versioned research/internal champion
candidate because the hard priority-recall and effective-latency gates pass.
Do not claim that the macro-F2 target passed.

The model emits document tags, not spans. It cannot redact values or explain a
positive. Its first-1,000-character ceiling can miss late-document PII; upstream
chunking or a longer pass is required for that threat model. The short pass's
holdout success does not prove late-page coverage.

The exceptionally high recall is bought with low precision on some labels, as
reflected in macro F2. A downstream verifier, reviewer, or span model is needed
for blocking and redaction. Dataset licence constraints make the current bundle
research/internal-only until all redistribution and commercial rights are
confirmed.

## Reproducibility and artifacts

- Frozen data/evaluator records: `../data/` and `../evaluations/champion_1k/`.
- Read-depth benchmark: `../benchmarks/read_depth.json`.
- Model directory: `../models/champion_1k/`.
- MLflow records: `../mlruns/`.
- Full command archive: `26-08-25_priority-recall-1000-run.commands.txt`.
- Experiment workbook: `26-08-25_Experiment-Log.xlsx`.
- Delivery bundle: `../dist/pii-priority-fusion-1k-v1.zip`.

Every Python command ran through the pipeline's pinned Python 3.13 environment.
The command archive includes successes and failures; raw command output is kept
there instead of expanding this report.

The final delivery check re-scored all 126,129 indexed holdout rows through the
bundle's own `tagger.py`. It reproduced macro F2 exactly at 0.48345749382010117;
the packaged and sealed prediction files were byte-identical with SHA-256
`3bfd3f776d8b7c09a7f11daafb8b5c4d2316571e724232506857be2f0168c194`.
