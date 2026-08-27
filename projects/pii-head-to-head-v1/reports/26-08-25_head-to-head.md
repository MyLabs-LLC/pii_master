# Head-to-head: priority-fusion vs steady-aim-cascade, both re-tuned on the full corpus

## Results

Both lineages were re-tuned from scratch — 2,000 trials, 1,000 each — on all
531,431 rows of `1-train`, under one loader, one 58-label catalogue, one
fit/calibration carve and one fixed evaluator, then scored on all eight sealed
`2-eval` corpora (126,129 rows).

### Model scoreboard

| Measure | fusion-1k (arm A) | steady-cascade (arm B) | fusion-12k (arm C) |
| --- | ---: | ---: | ---: |
| Read window (chars) | 1,000 | 12,000 | 12,000 |
| **macro F2** (headline ranker) | 0.4800 | **0.6497** | 0.4809 |
| macro F1 | 0.4066 | **0.6235** | 0.4037 |
| macro F0.5 | 0.3712 | **0.6147** | 0.3663 |
| macro F3 | 0.5308 | **0.6682** | 0.5344 |
| macro precision | 0.3573 | **0.6154** | 0.3512 |
| macro recall | 0.7946 | 0.6482 | **0.8071** |
| micro F1 | 0.3652 | **0.7313** | 0.3647 |
| micro F2 | 0.5443 | **0.8025** | 0.5459 |
| micro F0.5 | 0.2797 | **0.6730** | 0.2788 |
| micro precision | 0.2429 | **0.6397** | 0.2418 |
| micro recall | 0.9031 | 0.8598 | **0.9152** |
| priority macro F0.5 (contra ranker) | 0.2060 | **0.7467** | 0.2054 |
| priority macro precision | 0.1829 | **0.7478** | 0.1824 |
| priority macro recall | **0.9991** | 0.7622 | **0.9991** |
| worst priority-tag recall | 0.9966 | 0.6524 | **0.9974** |
| document precision | 0.6163 | **0.8893** | 0.6163 |
| document recall | **1.0000** | 0.7532 | **1.0000** |
| document specificity | 0.0006 | **0.8832** | 0.0007 |
| prediction rate | 0.9999 | 0.7836 | 0.9999 |
| dead tag×corpus pairs (F2 = 0) | 8 | 16 | **6** |
| median tag F2 | 0.4004 | **0.6667** | 0.3971 |
| one-core p95 (ms) | **1.157** | 3.916 | 4.232 |
| one-core docs/s | **1361** | 314 | 270 |

**The cascade wins every summary metric that prices precision at all — including
macro F2, the metric chosen because it favours recall-first models.** It leads on
all five corpora whose gold can measure F2, with non-overlapping 95% intervals on
every one:

| Corpus | n | fusion-1k | steady-cascade | fusion-12k |
| --- | ---: | ---: | ---: | ---: |
| betterdataai_ner_silver | 10,360 | 0.2485 [0.1719, 0.2728] | **0.3122** [0.2603, 0.3275] | 0.2485 |
| ai4privacy_pii_masking | 10,626 | 0.5238 [0.5158, 0.5313] | **0.7476** [0.7386, 0.7560] | 0.5242 |
| pii_holdout | 20,000 | 0.5512 [0.5464, 0.5563] | **0.7411** [0.7272, 0.7530] | 0.5699 |
| pii2_eval | 30,000 | 0.4808 [0.4749, 0.4859] | **0.7822** [0.7783, 0.7858] | 0.4659 |
| openpii_pii_eval | 38,937 | 0.5957 [0.5881, 0.6067] | **0.6656** [0.6644, 0.6669] | 0.5962 |
| datax · nemotron · govdocs2 | 15,206 | *NOT MEASURABLE* | *NOT MEASURABLE* | *NOT MEASURABLE* |
| **equal-corpus mean** | | **0.4800** | **0.6497** | **0.4809** |

Three of the eight corpora carry positive-only or partial tag gold, so no
precision-bearing metric is computable on them. They report `NOT MEASURABLE`,
never `0.0`, and are excluded from the mean.

### The decisions, both declared before the numbers

**Headline policy** (per-priority-tag recall ≥ 0.90 conclusive; one-core p95 ≤ 5 ms;
ranked on macro F2):

| Arm | Verdict | Hard constraints failed |
| --- | --- | --- |
| arm-A | FEASIBLE | 0 of 2 |
| arm-B | **blocked** | 1 of 2 — priority tag recall: **25 of 55** measurable pairs passed |
| arm-C | **WINNER** | 0 of 2 |

**Contra-view policy** (document precision ≥ 0.90, specificity ≥ 0.85, recall ≥ 0.85,
per-priority-tag recall ≥ 0.75, all conclusive; ranked on priority macro F0.5):

| Arm | Verdict | Hard constraints failed |
| --- | --- | --- |
| arm-A | blocked | 2 of 5 — doc precision 0/3 pairs, doc specificity 0/3 pairs |
| arm-B | blocked | 3 of 5 — doc precision 1/3, doc recall 1/3, priority recall 43/55 |
| arm-C | blocked | 2 of 5 — doc precision 0/3 pairs, doc specificity 0/3 pairs |

**No arm ships under the contra-view policy.** Selected under the headline policy:
**arm-C**, by 0.0009 macro F2 over arm-A — which is not a real ordering; see below.

### The finding that matters most

**The fusion arms clear the 0.90 recall gate by saturation, not by capability.**
Their document specificity is **0.0006** — across 10,003 documents that judges
confirmed contain no PII, the fusion stays silent on about **six**. Its prediction
rate is 0.9999. A model that says "this contains PII" about essentially every
document has a per-tag recall near 1.0 by construction, and that is exactly the
degenerate optimum the profile's advisory metrics exist to expose.

Read the two columns together:

| | fusion (A/C) | cascade (B) |
| --- | ---: | ---: |
| priority macro **recall** | 0.9991 | 0.7622 |
| priority macro **precision** | 0.1829 | 0.7478 |
| document specificity | 0.0006 | 0.8832 |

The gate is satisfied and the model is not useful. The cascade fails the same gate
honestly: 30 of 55 priority tag×corpus pairs fall below 0.90, and at the
contra-view's 0.75 floor only 12 of 55 still fail. Its worst pairs are real gaps,
not thin scopes — `sensitive_pii_address@govdocs2` at 0.5252 (n=891),
`sensitive_pii_full_name@govdocs2` at 0.5493 (n=3,215).

## TL;DR

- **Both models were re-tuned from scratch on the full 531,431-row corpus** —
  2,000 trials total — under one loader, one catalogue, one fit/calibration carve
  (451,548 / 79,883, identical rows for both) and one fixed evaluator.
- **The steady-aim cascade is the better model, decisively.** macro F2 **0.6497 vs
  0.4809**, micro F1 **0.7313 vs 0.3647**, priority macro F0.5 **0.7467 vs 0.2054**.
  It leads on all five corpora that can measure F2, with non-overlapping intervals.
- **It wins even on macro F2**, the recall-weighted metric picked because it
  favours the fusion's design. That was not the expected outcome.
- **The fusion's only win is recall, and it buys it by firing on everything** —
  prediction rate 0.9999, document specificity 0.0006. It passes the 0.90 per-tag
  recall gate degenerately.
- **Under the declared headline policy the winner is arm-C**, because the cascade
  fails the 0.90 recall gate (25/55 pairs) and the fusion does not. **Under the
  contra-view policy nothing ships.** Neither model is deployable as it stands.
- **The read window is not the story.** Arm C reads 12× more of every document
  than arm A and gains **+0.0009 macro F2**. The control ruled itself out.
- **Nothing was promoted, packaged or deployed** — this was a measurement run.

---

| | |
| --- | --- |
| Date | 2026-08-25 |
| Author | Ryan Lence |
| Project | `projects/pii-head-to-head-v1` |
| Branch | `main` |
| Run ID | H1 |
| Scope | two lineages re-tuned on the full `1-train`; 3 arms × 8 sealed corpora = 24 measured results |
| Request | *"do a head to head on both models. train them on this full dataset … then eval them on each of the datasets in eval … recored all the metrics in the log F1, F3, micro_F1, F0.5, F2, precission, recall, etc.. also make sure to record the results of each pii tag as well from the eval each run needs to be recorded in the log"* |
| Outcome | Cascade wins on quality by a wide margin; no arm clears both declared policies; nothing promoted |

## What was actually compared

The two shipped artifacts were **not** compared as they stand. Each lineage's
whole search was re-run on the full training corpus and the winner of each search
is the arm. That is what "train them on this full dataset" was taken to mean, and
it is the only version with a clean answer: the shipped artifacts were selected
under different loaders, different splits and different catalogues, so a gap
between them is partly a gap between two training setups.

Everything that could be held identical was:

| | |
| --- | --- |
| Training rows | all 8 corpora of `1-train` — **531,431 rows** |
| Fit / calibration carve | **451,548 / 79,883** — the same rows for both lineages |
| Catalogue | the frozen **58 collapsed labels**, same index order |
| Evaluation | all 8 corpora of `2-eval` — **126,129 rows**, sealed |
| Loader | `quiet_data.iter_quiet_corpus` for both |
| CPU | training on 32 cores; **every published latency on exactly one core** |

### The loader is the substantive change

The priority lineage previously loaded through `priority_data.normalize_row`,
which marks the dual-judge corpora `label_complete=False` — discarding the 20,714
real-world clean documents they contain — and decided whether a corpus carried a
complete catalogue by testing `dataset_dir.name.startswith("pii")`, a test an
external directory rename had silently broken, costing 54,812 of 70,600 negatives
without raising anything.

Both lineages here load through the corrected path, so both see the same rows
carrying the same gold. Without that, this would have compared two training sets
as much as two models.

## Arms

| Arm | Lineage | Read window | Search |
| --- | --- | ---: | --- |
| A | priority fusion | 1,000 chars (as shipped) | hash 300 · tfidf 300 · embeddingbag 300 · fusion 100 |
| B | steady-aim cascade | 12,000 chars (`deep`, as shipped) | docgate 250 · tagcount 150 · tagdisc 250 · cascade 350 |
| C | priority fusion | 12,000 chars | shares A's components; differs only in `read_window_override` |

**Arm C is a control, not a third model.** A and C share every count, every
component weight, every per-label threshold and every fusion strategy; only the
serving window differs. On govdocs2 that window is worth a lot of input — arm A
sees a mean of 233 hashed features per document, arm C 483, arm B 911 — so
without C, a gap between A and B could have been the read window rather than the
architecture.

**The control ruled itself out.** Twelve times the input moved the headline by
+0.0009. The per-corpus view shows why the mean barely moves: C gains on
pii_holdout (0.5699 vs 0.5512) and loses on pii2_eval (0.4659 vs 0.4808), both
with non-overlapping intervals, and the two offset. The 0.0009 that decided the
headline policy's winner is the residue of two real but opposing effects, **not
evidence that arm C is the better model.** Nothing should be built on that
ordering.

The reason the window buys so little is in the same table: prediction rate 0.9999.
The fusion already fires on every document at 1,000 characters, so more text
cannot make it fire more.

## Verification

**The cached-feature path was proved identical to the serving path, not assumed.**
Every arm re-scored 96 sampled documents by reading the real file and calling the
model's own `predict()`: **0 mismatches on all three arms**.

That check earned its place. On its first run it failed arm B with 2 of 96
disagreeing, all `.docx`. The cause is real and worth recording:
`_xml_archive_text` concatenates zip-XML parts until it has `limit` characters, so
for Office files the **extraction** — not just the truncation — depends on the
limit. One datax file yields 10,788 characters at `limit=12000` and 11,995 at
`limit=20000`, diverging in the tail. Measured scope:

| Corpus | Documents affected | Extensions |
| --- | ---: | --- |
| datax eval | 564 of 4,000 (14.1%) | all `.docx` |
| govdocs2 eval | 0 of 6,578 (0.00%) | — |

Each arm is now verified — and was trained — at its own lineage's read limit
(20,000 for the fusion, 12,000 for the cascade). That is the only internally
consistent choice: arm B's gate and heads were *fitted* on the 12,000-limit
extraction, so scoring against 20,000-limit text would measure a model on input
it never saw.

**Residual confound, stated rather than buried:** on those 564 datax documents
arm C sees slightly more tail text than arm B. It is confined to one of eight
corpora, affects only the tail, and datax carries positive-only tag gold — so it
can touch recall and document metrics but never a precision-bearing number.

**The evaluator was checked against closed-form values before it produced a
number** (`training/h2h_eval_selftest.py`): all four F-betas against their closed
forms and correctly ordered; macro excluding tags with no gold; micro pooling
correctly; every CI bracketing its point estimate; and the three properties that
would otherwise flatter a model — precision-bearing metrics `None` on
positive-only gold rather than `0.0`, no document scope on a prevalence-1.0
corpus, and a sub-`min_support` tag scoped with `None` plus its support.

**Reproduction fidelity.** Retrained from scratch, arm A lands within 0.004 of the
shipped `pii-priority-fusion-1k-v1` on both its headline numbers (macro F2 0.4800
vs 0.4835; priority macro F0.5 0.2060 vs 0.2057) and passes the same 55 measurable
gates. Arm B benches at 3.916 ms / 314 docs/s against the shipped cascade's
4.11 ms / 310 docs/s, and its materialisation reproduced its own trial with drift
**0.00000** against a 0.02 tolerance.

## Latency

| Arm | p50 | p95 | p99 | docs/s | 5 ms budget |
| --- | ---: | ---: | ---: | ---: | --- |
| A — fusion @ 1k | 0.825 | 1.157 | 1.233 | 1,361 | PASS |
| B — cascade @ 12k | 3.227 | 3.916 | 4.094 | 314 | PASS |
| C — fusion @ 12k | 3.822 | 4.232 | 4.495 | 270 | PASS |

One core, nothing else running, 200 real documents of ≥ 10 KB from the two
real-world corpora, 5 repeats. All three clear the budget.

The cascade's document gate saves less than it looks: silent p95 3.462 against
firing p95 4.038, at a 0.370 fire rate. At a 12,000-character window, tokenising
the read window dominates; the gate skips 58 head dot products, which were never
the expensive part.

## Two macro averages, and why both are reported

`macro_*_catalogue` averages over every tag the corpus's gold contains, counting a
tag the model never emits as a real 0 — the domain profile's definition, and the
one that ranks the arms. `macro_*_support30` averages over tags with at least 30
instances. Both are carried on all 24 rows. Reporting one without the other is how
a dead tail hides — and the tail is real here: 16 dead tag×corpus pairs for the
cascade against 6–8 for the fusion, the one tail metric where the fusion leads.

## Limitations

- **No arm is deployable.** Under the contra-view policy every arm fails; under
  the headline policy the two that pass do so degenerately. The honest reading is
  that neither lineage currently satisfies both halves of the requirement.
- **The headline winner is inside the noise.** Arm C beats arm A by 0.0009 macro
  F2, from two offsetting per-corpus differences. Treat A and C as tied.
- **Arm B's profile search was pinned to `deep`** so the arm stayed at its shipped
  12,000-character window. That concentrated all 1,000 trials on one profile, and
  it inherits the lineage's known blind spot: no `fast` or `std` cascade has ever
  been evaluated in it.
- **The fusion's train/serve mismatch is reproduced, not fixed.** Component
  thresholds are calibrated on 20,000-character scores and served at 1,000. Arm C
  bounds its cost at +0.0009 macro F2, so it is not the fusion's problem — but it
  remains a defect in the recipe.
- **Three of eight corpora cannot measure precision**, and nemotron cannot measure
  any tag metric at all. The equal-corpus headline rests on five corpora.
- **betterdataai carries silver (model-generated) labels.** It is the weakest
  corpus for both arms and should never decide a gate alone.
- **Search wall-clock is not comparable between lineages** — both ran while the
  other was running. Only the latency benches were measured on a quiet machine,
  which is the only timing quoted.

## Artifacts

| Path | What |
| --- | --- |
| `reports/26-08-25_head-to-head.md` | this report |
| `reports/26-08-25_head-to-head.pdf` | rendering of it |
| `reports/26-08-25_head-to-head.commands.txt` | complete unabridged command history |
| `reports/26-08-25_Experiment-Log.xlsx` | 24 log rows, per-metric CIs, per-tag tab, data-quality tab |
| `reports/_tables.md` | every generated table, including the full per-tag worst-first listings |
| `run.json` | 24 arms, 24 commands, both decisions, data quality |
| `decision/headline.json`, `decision/precision_view.json` | both recorded `DecisionResult`s |
| `evaluations/arm_{A,B,C}.json` | model-level arms with all scopes and CIs |
| `evaluations/latency_{A,B,C}.json` | one-core bench evidence |
| `models/{cascade,fusion_1000,fusion_12000}/` | the three trained artifacts |
| `tuning/*/trials.json` | all 2,000 trials |
| `mlflow.db` | every trial as a tracked run under `pii-head-to-head-v1` |

Command output is in the `.commands.txt`, not in this report.
