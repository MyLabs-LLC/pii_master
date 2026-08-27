# Final: sixteen models measured, one packaged, and a target declared infeasible

## Results

One run, two architectures re-tuned from scratch on the full 531,431-row corpus,
**2,000 search trials**, **sixteen models scored on eight sealed corpora — 128
measured results**, all under one loader, one 58-label catalogue, one
fit/calibration carve and one fixed evaluator.

### The scoreboard

| model | read window | macro F2 | micro F1 | priority F0.5 | doc P / sp / R | p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **pii-cascade-balanced-v2** *(packaged)* | 12,000 | **0.6614** | 0.7255 | 0.7434 | **0.896 / 0.879 / 0.797** | 3.92 ms |
| steady-cascade (arm B) | 12,000 | 0.6497 | **0.7313** | **0.7467** | 0.889 / 0.883 / 0.753 | 3.92 ms |
| fusion-12k (arm C) | 12,000 | 0.4809 | 0.3647 | 0.2054 | 0.616 / 0.001 / 1.000 | 4.23 ms |
| fusion-1k (arm A) | 1,000 | 0.4800 | 0.3652 | 0.2060 | 0.616 / 0.001 / 1.000 | **1.16 ms** |
| *12 operating points* | 12,000 | 0.63–0.66 | 0.73–0.78 | 0.75–0.77 | unchanged | 3.92 ms |

### The three findings that decided the run

**1. The cascade beats the fusion decisively — including on the metric chosen to
favour the fusion.** macro F2 0.6497 vs 0.4809, on all five corpora that can
measure it, with non-overlapping intervals. The fusion's only advantage is recall,
and it buys that by firing on 99.99% of documents: **its document precision equals
the corpus base rate to four decimals on all three corpora that have negatives**,
so its document-level output carries zero information.

**2. Source-balancing the gate is a free win.** Real-world documents are 7.8% of
the fit rows carrying document gold. Equalising their contribution to the loss
lifted document recall 0.7532 → 0.7975, macro recall 0.6482 → 0.6938, worst
priority-tag recall 0.6524 → 0.7221, and priority gates cleared 25 → 29 of 55 —
at no cost in latency and essentially none in precision.

**3. The document-level bars are not reachable in this architecture.** Best
simultaneously achievable is recall **0.623** holding precision ≥ 0.90 and
specificity ≥ 0.85 (bar 0.85), or precision **0.766** holding recall ≥ 0.85 (bar
0.90). Three independent lines say this is the representation, not the data:

| evidence | result |
| --- | --- |
| frontier walk on measured scores | bars sit 0.227 outside it |
| synthetic near-miss negatives (1,948 verified) | AUC 0.8453 → 0.8461; generated vs real separate at **AUC 1.0000** |
| real target-distribution data (7,370 labelled) | AUC 0.8832 → 0.8778, flat across a 15× budget range |
| in-distribution cross-validated ceiling | AUC 0.869 — no better than transferring |

Formal verdict recorded: **`unlikely`**, ceiling ≈ 0.623 against a 0.85 target.

### What was packaged

`projects/pii-head-to-head-v1/dist/pii-cascade-balanced-v2/` and its `.zip`,
re-scored through its own `tagger.py` after packaging: priority macro F0.5 on
`pii_holdout_20k` expected 0.841073, measured 0.841073, **delta 2.5e-08**.

It is packaged as the run's best measured artifact, **not** promoted to
`@champion` — it fails hard constraints in both declared policies, and this run's
rule is that a gate-failing artifact is not promoted.

## TL;DR

- **Sixteen models, 128 sealed results, 2,000 trials.** Full history in
  `26-08-25_Experiment-Log.xlsx`: 152 Experiment Log rows, 3,384 metric rows with
  confidence intervals, 5,980 per-tag rows, 8 data-quality rows.
- **The cascade wins**, on every summary metric that prices precision at all,
  including the recall-weighted one picked to favour its opponent.
- **The fusion is not a detector.** Document precision = base rate; zero
  information at the document level.
- **Balanced gate adopted**: +0.044 document recall, +0.070 worst priority-tag
  recall, 4 more gates cleared, free.
- **Target declared infeasible** for this architecture, with the reachable
  numbers stated. Synthetic data and real data both refuted as remedies.
- **Packaged** `pii-cascade-balanced-v2` with card, zip, checksums and a
  post-packaging reproduction check.
- **Nothing promoted.** No arm clears its declared gates.

---

| | |
| --- | --- |
| Date | 2026-08-26 |
| Author | Ryan Lence |
| Project | `projects/pii-head-to-head-v1` |
| Run ID | H1 |
| Scope | 2 lineages × 1,000 trials; 16 models × 8 sealed corpora; datagen; feasibility |
| Request | *"do a head to head on both models. train them on this full dataset … then eval them on each of the datasets in eval … record all the metrics … also the results of each pii tag … each run needs to be recorded in the log"* |
| Outcome | cascade wins; balanced gate adopted and packaged; document bars infeasible here |

## The history, and where to read it

`reports/26-08-25_Experiment-Log.xlsx` is the record. Four tabs:

| tab | rows | holds |
| --- | ---: | --- |
| Experiment Log | 152 | one row per model × corpus, carried forward across runs |
| Run Results | 3,384 | every metric with its 95% interval, one row per metric per arm |
| Per Tag Results | 5,980 | support, predicted, found, missed, false-positive, P, R, F0.5, F1, F2, F3 for every model × corpus × tag |
| Data Quality Log | 8 | every corpus, its gold mode, leakage assessment |

The twelve operating points from the precision ladder are logged as twelve
distinct models, which is what lets the recall/precision trade be read as rows
rather than reconstructed. One earlier ladder was computed with the head trial's
`margin`/`min_support` instead of the cascade trial's; it is **excluded** from the
log and kept under `superseded/` rather than shipped as a second, conflicting
"floor 0.75" row.

## Four reports, in order

1. `26-08-25_head-to-head.md` — the head-to-head itself, both declared decisions
2. `26-08-25-1_gate-diagnosis.md` — why real-world recall is 20 points below the
   held-in estimate; three hypotheses refuted
3. `26-08-26_datagen-near-miss-negatives.md` — synthetic negatives refuted, with
   the separability mechanism measured
4. `26-08-26-1_balanced-gate-and-feasibility.md` — the balanced gate end-to-end
   and the infeasibility verdict

## What I got wrong, recorded

Four corrections, each caught by measurement rather than argument. They are worth
keeping because three of them were confident and wrong in the same direction —
assuming a modelling defect where the truth was a property of the data or the
harness.

- **"The gate overfits."** Reasonable given `alpha` 7.2e-7 over 262,144 features
  and 7.8% real data. Refuted: sweeping alpha four orders of magnitude moved the
  gap 0.04, and removing govdocs2 from training entirely changed nothing.
- **"It is source-directory leakage."** Refuted: grouped carving moved held-in
  AUC 0.9894 → 0.9864.
- **"Active learning will close it."** Refuted by the label-budget curve, before
  a labelling budget was spent. My earlier "oracle" AUC of 0.9998 was
  memorisation — an unregularised linear model fitting 10,549 rows over 262k
  features — and the honest cross-validated ceiling is 0.869.
- **A `gate_shift` carried across a change of scale.** Arm B's absolute −2.196
  offset, applied to a gate regularised at `alpha` 1e-2, moved the cut to −2.13
  and dropped document specificity 0.883 → 0.011. A tuned constant does not
  survive a change of scale.

Two false positives in my own verification code were also fixed: the model's
`ipv6` pattern matches timestamps like `10:15:30` (it was discarding log
extracts, the most adversarial generated class), and a data dictionary opening
"Data Dictionary" is a realistic document rather than label leakage.

## Verification discipline

- Cached-feature predictions were proved identical to each model's own
  `predict()` on sampled real documents — **0 mismatches of 96 per arm**. The
  first attempt failed arm B with 2 of 96, exposing that `.docx` extraction is
  read-limit dependent (564 of 4,000 datax documents affected, 14.1%).
- The evaluator was checked against closed-form values before producing any
  number, including that precision-bearing metrics are `None` on positive-only
  gold rather than 0.0, and that a prevalence-1.0 corpus emits no document scope.
- The generated corpus was scanned before any document reached training: 52 of
  2,000 rejected, including **39 carrying real identifiers under a negative
  label**.
- The packaged bundle was re-scored through its own entry point: delta 2.5e-08.

## Recommendation

**Ship `pii-cascade-balanced-v2` for triage** if the reachable numbers are
acceptable — document precision 0.896 at 0.797 recall, 3.92 ms on one core.

**Do not spend more on this architecture against the 0.85/0.90 bars.** Every dial
is measured: per-tag floor, selection beta, gate regularisation, source
balancing, synthetic negatives, real negatives.

**Then a product decision, not a modelling one:** re-scope the bars to the
reachable region, or move to a representation that can clear them — contextual
embeddings or span-level NER, candidates for which already sit in this repo under
`models/pii-master-ner-*`. That is a new project with its own feasibility probe.

## Limitations

- Nothing here is promoted; no registry version exists.
- The balanced gate's joint `gate_shift` was not re-searched — a small gain left.
- No `fast` or `std` read profile was evaluated in this lineage; the
  quality/latency trade below 12,000 characters is unmeasured.
- The label-budget curve is bounded by the 7,370 documents in the sealed real
  pool; flatness across that range is the evidence, not a proof for all budgets.
- Search wall-clock is not comparable between lineages — they ran concurrently.
  Only the latency benches were measured on a quiet machine, and they are the
  only timings quoted.

## Artifacts

| Path | What |
| --- | --- |
| `dist/pii-cascade-balanced-v2/` + `.zip` | the packaged model, card, checksums |
| `reports/26-08-25_Experiment-Log.xlsx` | the full history, four tabs |
| `reports/*.md` · `*.pdf` | four run reports |
| `reports/26-08-25_head-to-head.commands.txt` | every command, unabridged |
| `run.json` | 128 arms, both decisions, data quality |
| `evaluations/` | 16 model-level arms with all scopes and CIs |
| `probe/` | feasibility, gate diagnosis, grouped split, augmentation |
| `tuning/` | all 2,000 trials |
| `datagen/2000_26082521_claude/` | the generated corpus and its verification |
| `mlflow.db` | every trial as a tracked run |
