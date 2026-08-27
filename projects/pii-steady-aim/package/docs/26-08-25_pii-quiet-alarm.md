# pii-quiet-alarm — precision-first sensitive-data tagging

Run `pii-quiet-alarm-2026-08-25` · commit `3ce3c8c` · 1,000 trials · 24 measured
arms (3 models × 8 corpora) · MLflow experiment `pii-quiet-alarm`

## Results

### Headline — equal-corpus, sealed evaluation

| Arm | priority macro F0.5 | macro F0.5 | doc precision | doc specificity | doc recall | one-core p95 | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| **quiet-cascade** | **0.8233** | **0.7766** | **0.8806** | **0.8693** | 0.7729 | 3.93 ms | **blocked** |
| quiet-nogate *(ablation)* | 0.8205 | 0.7746 | 0.7473 | 0.5783 | 0.8943 | 3.97 ms | blocked |
| champion_1k *(prior champion)* | 0.2057 | 0.4163 | 0.6162 | 0.0005 | 0.9994 | 1.81 ms | blocked |

`quiet-nogate` carries **identical weights and thresholds** to `quiet-cascade`
with the document gate disabled, so every difference between those two rows is
the gate alone.

### The gate ladder — why nothing shipped

Each constraint is judged per corpus on the bootstrap **lower bound**, with a
30-instance minimum. `n/m` is passing over measurable.

| Arm | doc precision ≥ 0.90 | doc specificity ≥ 0.85 | doc recall ≥ 0.85 | priority tag recall ≥ 0.75 | p95 ≤ 5 ms |
| --- | --- | --- | --- | --- | --- |
| quiet-cascade | **FAIL** 1/3 (worst 0.7893) | **FAIL** 2/3 (worst 0.8299) | **FAIL** 1/3 (worst 0.6604) | **FAIL** 36/55 (worst 0.0000) | PASS |
| quiet-nogate | **FAIL** 1/3 (0.5892) | **FAIL** 0/3 (0.4533) | **FAIL** 2/3 (0.8229) | **FAIL** 36/55 (0.0000) | PASS |
| champion_1k | **FAIL** 0/3 (0.4632) | **FAIL** 0/3 (0.0000) | PASS 3/3 | PASS 55/55 | PASS |

**No feasible arm. Promotion refused; no champion packaged.**

### Document level, per corpus — the three that hold genuine negatives

Point estimate with the 95% bootstrap lower bound in brackets. The other five
corpora have prevalence 1.0 and report these as `NOT_MEASURABLE`, never as a
pass.

| Arm | Corpus | precision | specificity | recall |
| --- | --- | ---: | ---: | ---: |
| quiet-cascade | pii2 (synthetic) | 0.9783 (0.9764) | 0.8926 (0.8841) | 0.9324 (0.9295) |
| quiet-cascade | datax (real) | 0.8089 (0.7893) | 0.8462 (0.8299) | 0.7104 (0.6905) |
| quiet-cascade | govdocs2 (real) | 0.8545 (0.8414) | 0.8692 (0.8574) | 0.6759 (0.6604) |
| champion_1k | pii2 (synthetic) | 0.8383 (0.8342) | 0.0000 (0.0000) | 1.0000 (1.0000) |
| champion_1k | datax (real) | 0.4783 (0.4632) | 0.0000 (0.0000) | 1.0000 (1.0000) |
| champion_1k | govdocs2 (real) | 0.5320 (0.5205) | 0.0016 (0.0003) | 0.9983 (0.9968) |

The split is the finding: on synthetic documents the cascade clears every
document bar comfortably; on **real business documents it stays shut too often**
(recall 0.71 and 0.68 against a 0.85 floor).

### Priority-tag quality, per corpus

| Corpus | quiet-cascade | champion_1k |
| --- | --- | --- |
| pii2 | F0.5 0.965 · P 0.966 · R 0.967 | F0.5 0.045 · P 0.038 · R 1.000 |
| pii_holdout | F0.5 0.942 · P 0.964 · R 0.864 | F0.5 0.185 · P 0.159 · R 0.999 |
| openpii | F0.5 0.907 · P 0.940 · R 0.799 | F0.5 0.385 · P 0.341 · R 1.000 |
| ai4privacy | F0.5 0.880 · P 0.929 · R 0.742 | F0.5 0.184 · P 0.157 · R 1.000 |
| betterdataai | F0.5 0.422 · P 0.632 · R 0.319 | F0.5 0.228 · P 0.218 · R 1.000 |

**pii2 is the flattering corpus (0.965); betterdataai is the weakest (0.422).**
Quote the equal-corpus **0.8233**, not either end.

Prediction rate — the share of documents receiving any tag — falls from
**0.9998** (champion_1k tags essentially every document) to **0.7876**.

### The 19 failing priority pairs

Worst first, of 55 measurable pairs. Two clusters, with different causes.

| Tag | Corpus | recall | ci_low | support |
| --- | --- | ---: | ---: | ---: |
| medical_record_number_mrn | betterdataai | 0.0132 | 0.0000 | 151 |
| password | betterdataai | 0.1000 | 0.0200 | 50 |
| address | betterdataai | 0.3110 | 0.2559 | 254 |
| address | govdocs2 | 0.3636 | 0.3322 | 891 |
| personal_identification_number_pin | ai4privacy | 0.5172 | 0.4345 | 145 |
| address | datax | 0.5600 | 0.5086 | 350 |
| full_name | govdocs2 | 0.6208 | 0.6037 | 3,215 |
| driver_s_license_number | openpii | 0.6206 | 0.6102 | 8,538 |
| full_name | ai4privacy | 0.6331 | 0.6192 | 5,470 |
| password | ai4privacy | 0.7020 | 0.6779 | 1,245 |
| social_security_number | ai4privacy | 0.7113 | 0.6925 | 2,400 |
| passport_number | ai4privacy | 0.7205 | 0.6942 | 1,102 |
| address | pii2 | 0.7286 | 0.7128 | 2,841 |
| driver_s_license_number | pii_holdout | 0.7315 | 0.7147 | 2,808 |
| military_identification_number | openpii | 0.7307 | 0.7237 | 13,763 |
| passport_number | openpii | 0.7363 | 0.7284 | 13,763 |
| address | ai4privacy | 0.7461 | 0.7295 | 2,529 |
| visa_number | openpii | 0.7404 | 0.7334 | 13,763 |
| driver_s_license_number | ai4privacy | 0.7657 | 0.7393 | 1,101 |

**Cluster A — marginal (11 pairs, 0.70–0.77).** These sit just under a 0.75
floor. Their thresholds were chosen to hit exactly 0.75 on *pooled* held-in
calibration, so any distribution shift lands them on the wrong side. This is a
calibration-transfer problem, not a capability limit.

**Cluster B — structural (8 pairs).** `address` fails on all five corpora it is
measurable on, and MRN/password collapse on betterdataai specifically. Address
is the tag the frozen collapse merged `street_number_and_name` into, and
betterdataai carries silver (model-generated) labels.

---

## TL;DR

The precision problem was a **data defect, not a model defect**, and fixing it
moved the numbers a long way — but the run did not clear the bars it set for
itself, so nothing was promoted.

- **Priority macro F0.5: 0.2057 → 0.8233** (4.0×). Priority-tag precision on the
  weakest complete corpus went from 0.218 to 0.632; on the strongest, 0.038 to
  0.966.
- **Document specificity: 0.0005 → 0.8693.** The prior champion flagged 99.95%
  of genuinely PII-free documents; the cascade correctly stays silent on 87% of
  them. Document precision 0.6162 → 0.8806.
- **Still fast: 3.93 ms p95 on one core** against a 5 ms budget, reading 12,000
  characters — 12× deeper than the prior champion's 1,000, and the depth is what
  makes it work on real documents.
- **Nothing shipped.** Of five hard gates the best arm clears one. Document
  recall on real business documents (0.66–0.69) and 19 of 55 priority-tag recall
  pairs fall short. The gates were written before the numbers arrived and were
  not moved afterwards.
- **The route is `budget_exhausted`, not `target_infeasible`.** Both failure
  clusters have named, unexplored mechanisms — see *What to do next*.

---

## What was found before any model was fitted

Two probes changed the shape of the run, and both are corrections to the prior
lineage rather than new modelling.

### 1. Six of eight training corpora contained zero clean documents

Of 531,431 training documents, only **49,961** carried a labelled-clean
negative, all from one corpus. A model fitted where every document contains PII
has learned the correct answer for that world, and "everything has PII" is what
the prior champion says.

The negatives existed and the loader could not see them.
`priority_data.normalize_row` reads only `gold` and `pii_entities`, so it
discarded every clean document in the four dual-judge directories. Admitting a
judge assertion of absence — empty entities **and** empty classes **and** an
explicit `sensitivity: "none"` — recovers **20,639 real-world PII-free business
documents** for training (70,600 negatives total, 29% real rather than
synthetic) and **10,003** for evaluation.

A third outcome had to be invented: **217 rows are ambiguous**, listing an
entity (usually a bare `State`) and then rating the document `sensitivity:
none`. Reading them as positive penalises correct silence; as negative, teaches
silence over real entities. They are excluded, counted by name, and capped at 2%
of any corpus by a contract that fails the run rather than warning.

### 2. The document-level gold field was the wrong one

The prior lineage read the manifest's `label` for its document-level numbers.
That field is the judges' **document-type** verdict, and it is orthogonal to PII
presence:

| govdocs2 eval | no PII entities | has PII entities |
| --- | ---: | ---: |
| `label = positive` | **1,501** | 2,032 |
| `label = negative` | 1,193 | **1,220** |

Re-scored against the fields that do answer the question, the frozen champion's
document specificity is **0.0005**, not the 0.53 precision the earlier reading
suggested.

### 3. Re-thresholding the old champion was never going to work

A 41-point sweep of the shipped model's own per-label thresholds
(`probe/precision_headroom.json`): reaching specificity 0.90 drops document
recall to 0.22 on pii2 and 0.34 on govdocs2. Its document score does not
separate. A precision model had to be **fitted**, with negatives, against a
precision objective.

---

## Feasibility, probed before the budget was approved

| Target | Verdict | Ceiling evidence | Supported |
| --- | --- | --- | ---: |
| document specificity 0.90 | **plausible** | two judges over 946 datax documents: raw agreement 0.9545, Cohen κ 0.9080 | 0.955 |
| priority macro precision 0.90 | **unlikely** | inter-judge macro F1 0.4582 across 12 tags | 0.458 |

Both verdicts held up. Document specificity reached 0.869 — close to the 0.955
the probe supported. Tag-level precision reached **0.8806 equal-corpus**, well
*above* the 0.458 the inter-judge probe suggested, because the frozen collapse
removed the naming disagreement that ceiling was mostly made of. The probe was
right about the mechanism and conservative about the size.

---

## How the search was spent

1,000 trials, all MLflow-tracked, all scored on held-in calibration carved from
the training corpora by a stable document hash. The eight evaluation
directories were not reachable from the tuning module and were scored **once**.

| Family | Trials | Mechanism | Best held-in priority F0.5 |
| --- | ---: | --- | ---: |
| `docgate` | 300 | discriminative binary "contains sensitive PII" head | — (document metrics only) |
| `tagcount` | 370 | Bernoulli log-odds from counts, per-label F0.5 cuts | 0.7073 |
| `tagdisc` | 147 | one-vs-rest discriminative per-tag heads | 0.9347 |
| `cascade` | 183 | gate in front of heads, joint operating point | 0.9367 |

Two process notes, both of which cost budget:

- **250 of the tagcount trials were scored with a mis-specified objective.** The
  document-level floors were left on for a family that decides "this document
  has PII" by firing at least one tag — which it cannot clear by construction.
  All 250 came back infeasible and were then ranked by smallest deficit, which
  rewards heads that fire *less*: the opposite of what the cascade needs. The
  measurements were sound, so the family was **re-ranked rather than re-run**,
  and a corrected 120-trial sweep followed. The correction was worth it: best
  priority F0.5 went from 0.640 (re-ranked) to 0.707 (corrected search).
- **29 tagdisc trials were spent on two aborted runs** — the first serial, the
  second killed to switch per-label fitting from process to thread parallelism
  after 12 worker processes each pickling a 450,000-row sparse matrix proved
  slower than the serial loop it replaced.

**Only `deep` + `tagdisc` cascades were feasible on calibration.** No `fast` or
`std` profile cascade, and no count-based head cascade, cleared the document
floors — the top-8 gates were all `deep`, so profile-mismatched pairs were
pruned (22 trials). That is a real limitation of how the cascade sampled its
components, and it means the run has *no* evidence about whether a cheaper read
profile could have worked with a differently-selected gate.

---

## What the gate bought, isolated

`quiet-nogate` is the same artifact with the document gate switched off:

| | with gate | without gate | Δ |
| --- | ---: | ---: | ---: |
| doc precision | 0.8806 | 0.7473 | **+0.1333** |
| doc specificity | 0.8693 | 0.5783 | **+0.2910** |
| doc recall | 0.7729 | 0.8943 | −0.1214 |
| priority macro F0.5 | 0.8233 | 0.8205 | +0.0028 |
| prediction rate | 0.7876 | 0.8622 | −0.0746 |

The gate is doing exactly the job it was added for and paying exactly the price
that was expected: **+29 points of specificity for −12 points of document
recall.** It is also the reason document recall misses its floor.

---

## Where this run is worse than the model it does not replace

`champion_1k` remains better on two things, and anyone depending on them should
not move:

- **Priority-tag recall: 55/55 gates versus 36/55.** If a missed identifier is
  the only cost that matters and false alarms are free, the old model is still
  the right one.
- **Latency: 1.81 ms versus 3.93 ms p95.** Both clear the 5 ms budget, but the
  cascade reads 12,000 characters to the champion's 1,000.

It is worse on everything the run was about: it tags 99.98% of all documents,
has document specificity indistinguishable from zero, and its priority-tag
precision on the weakest complete corpus is 0.218.

---

## What the evidence supports

The gates as written are not met. What **is** established, on 126,129 sealed
documents:

| Bar | Written | Achieved (worst corpus, ci_lower) |
| --- | ---: | ---: |
| doc precision | 0.90 | **0.79** |
| doc specificity | 0.85 | **0.83** |
| doc recall | 0.85 | **0.66** |
| priority tag recall | 0.75 on 55/55 | **0.75 on 36/55** |
| priority macro F0.5 | *(ranker)* | **0.8233** |

A gate set of *doc precision ≥ 0.78, specificity ≥ 0.82, recall ≥ 0.65, priority
tag recall ≥ 0.75 on 36/55 pairs* is what this artifact clears today. Whether
that is a shippable product is a decision about the cost of a missed identifier
against the cost of a false alarm — which is exactly the trade the 0.75 floor
was chosen to make, and it is the user's call, not this run's.

## What to do next

Ordered by expected value. Each targets a measured failure, not a hunch.

1. **Select per-label thresholds against the worst source group, not pooled
   calibration.** Eleven of the 19 failures sit between 0.70 and 0.77 against a
   0.75 floor — thresholds tuned to hit the floor exactly on pooled data, then
   landing just under it on shifted data. Selecting so the floor holds on the
   weakest held-in source group directly attacks the largest cluster, costs
   precision only where the margin was illusory, and needs no new fitting —
   only a different selection rule over scores this run already computes.
2. **Give the document gate a real-world recall term.** It was selected on real
   calibration data but ranked by precision; document recall on real business
   documents (0.66–0.69) is now the binding failure. The gate's own sweep has
   operating points at higher recall, and the ablation shows the cost is
   specificity, currently 2 points above its floor with nowhere to spend it.
3. **Treat `address` as its own problem.** It fails on all five corpora where it
   is measurable, 0.31 to 0.75, and it is the tag the collapse merged
   `street_number_and_name` into. Either the collapse is wrong for address, or
   address needs cue features the hashed representation is not capturing.
4. **Decide what betterdataai is for.** It is silver-labelled, it is the weakest
   corpus by a wide margin (F0.5 0.422 against 0.965), and MRN/password collapse
   there and nowhere else. Either its labels are noise that should not gate a
   promotion, or they are signal the model genuinely misses. The run cannot tell
   these apart and should not pretend to.
5. **Re-open the cheaper read profiles.** No `fast` or `std` cascade was ever
   evaluated, because component pairing pruned them. A 0.45 ms p95 artifact at
   materially lower quality may still be the right product for some callers, and
   right now there is no measurement either way.

---

## Provenance and limits

- **Data.** 531,431 training documents (438,929 positive, 70,600 negative);
  126,129 evaluation documents (110,469 positive, 10,003 negative). Manifest
  digests pinned in `data_snapshot.json`; leakage 0 on every corpus after an
  external decontamination pass removed 22,816 leaking training rows.
- **Two mid-run data events**, both caught and both recorded. An external pass
  rewrote all eight training manifests at 11:32, and a second renamed every
  dataset directory at 12:55. The rename silently broke a loader heuristic that
  inferred label completeness from the folder-name prefix, erasing 54,812 of the
  70,600 negatives with no error raised and no failing test — the one test
  guarding them looked corpora up by literal name and skipped itself. Both are
  fixed at commit `3ce3c8c`; counts were verified identical to the pre-rename
  freeze before the search results were used.
- **Latency was re-measured on a quiescent machine.** An earlier reading of
  8.57 ms p95 for the baseline was taken while three searches saturated all 32
  cores; the same artifact measures 1.81 ms quiet. Every latency in this report
  is single-core with nothing else running.
- **Deviation from the run-record convention.** Commands were issued directly
  rather than through `mp run`, so there is no captured stdout per command. The
  long-running steps do have captured logs, and every number here traces to a
  JSON artifact or to `mlflow.db`; none is reconstructed from memory.
- **This is document classification, not span NER.** It cannot locate or redact
  a value, and a positive tag does not identify which text triggered it.
- **Licensing.** Sources include AI4Privacy and NonCommercial material.
  Research/internal only until redistribution rights are confirmed.
