# Gate diagnosis: why real-world document recall is 20 points below the held-in estimate

## Results

Arm B's document gate scores **0.8564** document recall on held-in real-world
calibration and **0.6495–0.6790** on the sealed real-world corpora, while on
synthetic data the same gate moves 0.9348 → 0.9311 — no gap at all. Three
hypotheses were tested. **All three are refuted.** The cause is a property of the
data, and it is deliberate.

### What was ruled out

| # | Hypothesis | Test | Verdict |
| --- | --- | --- | --- |
| a | Gate weights overfit (`alpha` 7.2e-7 over 262,144 features) | ROC-AUC held-in vs sealed, `alpha` swept 7.2e-7 → 1e-2 | **refuted** — ΔAUC moves only −0.113 → −0.074 across four orders of magnitude |
| b | The threshold does not transfer | recall at the held-in cut vs at a sealed cut of matched specificity | **refuted** — re-cutting on sealed is *worse* (0.4979 vs 0.6504); at matched specificity the gap is larger, not smaller |
| c | The gold shifted between halves | judge fields, provenance mix, entity composition | **refuted** — same judges (gemini + opus), 63.7% vs 63.3% dual-judge agreement, 3.25 entities per positive in both |

Two further candidates raised during the investigation, also refuted:

| Candidate | Test | Verdict |
| --- | --- | --- |
| Source-directory leakage across the held-in carve | carve whole directories instead of documents | **refuted** — held-in AUC 0.9894 → 0.9864. Barely moves |
| Non-random train/eval directory assignment | directory-number structure | **refuted** — 198 eval directories form 162 interleaved runs across 0–999, longest run 4 |

The decisive test was leave-one-corpus-out. A gate fitted with **govdocs2 removed
from training entirely** still scores:

| gate fitted on | govdocs2 held-in AUC | govdocs2 sealed AUC | Δ |
| --- | ---: | ---: | ---: |
| all corpora | 0.9894 | 0.8434 | −0.1460 |
| **without govdocs2** | 0.9879 | 0.8378 | −0.1501 |
| without any real corpus | 0.9865 | 0.8454 | −0.1411 |

A model that has never seen a single govdocs2 document separates one half at
0.99 and the other at 0.84. Nothing about training can explain that.

### What it actually is

The two halves are not equally separable, and both classes move toward each
other. Scored by a gate trained on **synthetic data only**:

| | n | p10 | p25 | **p50** | p75 | p90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train positives | 1,645 | 292 | 462 | **759** | 1,163 | 1,609 |
| eval positives | 3,484 | −352 | 43 | **520** | 1,085 | 1,567 |
| train negatives | 1,849 | −1,285 | −957 | **−609** | −314 | −111 |
| eval negatives | 3,065 | −1,052 | −725 | **−362** | −39 | 313 |

**Class separation collapses from 1,368 to 882 — a 36% reduction.** The eval
half's positives carry subtler evidence *and* its negatives look more like PII.
The 90th percentile of eval negatives (313) sits above the 10th percentile of
eval positives (−352) and near their 25th (43).

Everything else matches: 100% `.txt` in both, identical length and feature-density
distributions, and positive entity composition identical to within a ratio of
0.78–1.06 across all sixteen common entity types (`full name` 91.7% vs 91.7%,
`phone number` 45.2% vs 45.0%, `email` 36.4% vs 37.6%).

**The eval half is adversarial by construction**, which is what the suite already
says of it: `role: adversarial`, "the nearest thing in the suite to the
customer's own corpus — heterogeneous real files, mostly mundane, mostly clean."

## TL;DR

- The 20-point gap is **not a defect and not fixable by training**. A gate that
  never saw govdocs2 reproduces it exactly.
- It is the difference between a **random** sample of real documents (the
  training half) and an **adversarially selected** one (the sealed half). Class
  separation is 36% narrower in the sealed half.
- **The sealed number is the honest one.** 0.6495 document recall is what this
  architecture delivers on hard real-world documents; 0.8564 is what the
  training-side real corpus flatters it into looking like.
- **One real improvement survives**: source-balancing the gate's loss (real
  documents are 7.8% of fit rows) plus `alpha = 1e-2` lifts sealed real AUC
  0.8453 → 0.8804 and sealed datax recall 0.5714 → 0.6529. Worth adopting.
- **The grouped-split fix should not be adopted** — it costs complexity and buys
  0.003 AUC.
- The search's held-in objective will stay optimistic on real-world data as long
  as the training-side real corpus is not adversarial. That is the thing to fix,
  and it is a **data** problem, not a split or regularisation problem.

---

| | |
| --- | --- |
| Date | 2026-08-25 |
| Author | Ryan Lence |
| Project | `projects/pii-head-to-head-v1` |
| Run ID | H1 (follow-up to the head-to-head) |
| Scope | document gate only; the 58 tag heads are untouched |
| Outcome | three hypotheses refuted; cause identified as adversarial construction of the sealed half; one adoptable improvement |

## Why this mattered enough to chase

The gate composes multiplicatively with everything downstream: a tag can only
fire on a document the gate admits. At 0.6495 document recall on govdocs2, **35%
of PII-bearing real-world documents never reach the tag heads**, and no per-tag
threshold can recover them. Every tag metric on real documents is capped by this
number, so a 20-point error in understanding it would have misdirected the next
round of work entirely — which is exactly what it nearly did.

## What I got wrong, and what it cost

I recommended this investigation on the hypothesis that the gate was overfitting:
`alpha = 7.2e-7` against 262,144 features and only 7.8% real training rows is a
textbook setup for it. That reasoning was sound and the conclusion was wrong.

I then proposed the grouped split as the durable fix, on the evidence that
govdocs2 carries 793 source directories and `carve_holdin` splits by document.
That was also wrong — the leak it would have prevented does not exist.

Both were caught by measurement rather than argument, which is the only reason
they cost an hour instead of a sprint. The leave-one-corpus-out test is the one
that should have been run first: it is a single fit, and it falsifies every
training-side explanation at once.

## Recommendations

**1. Adopt the source-balanced gate.** Real documents are 33,637 of 432,903 fit
rows with document gold — 7.8% of the loss for the half of the problem that
matters. Equal-group weighting with `alpha = 1e-2`:

| | arm B gate | balanced + regularised |
| --- | ---: | ---: |
| sealed real AUC (pooled) | 0.8453 | **0.8804** |
| govdocs2 sealed AUC | 0.8434 | **0.8746** |
| datax sealed AUC | 0.8504 | **0.8888** |
| datax sealed recall @ spec 0.95 | 0.5714 | **0.6529** |

Threshold-free on both corpora, so it is a genuine ranking improvement rather
than an operating-point move.

**2. Do not adopt the grouped split.** It buys 0.003 AUC.

**3. Stop quoting held-in real-world numbers.** The search's `doc_real_*` metrics
are measured on a non-adversarial sample and run ~0.14 AUC optimistic. Either
report them with that offset stated, or move an adversarial slice into the
held-in split so the search can see what it is actually choosing.

**4. The real lever is data.** Closing this gap needs adversarial near-miss
negatives — mundane real documents that read as sensitive — in *training*, not a
different regulariser. That is a datagen task, and it is the highest-value one
available: it is the only thing that lifts the ceiling every tag metric sits
under.

## Limitations

- Everything here concerns the **document gate**. The tag heads were not refit
  and their thresholds were not moved.
- The balanced+regularised gate is validated on ranking (AUC) and on
  specificity-matched recall. It has **not** been carried through a full cascade
  materialisation, so its effect on the end-to-end sealed metrics is not yet
  measured.
- `datax` has only 12 source groups, too few for the grouped-split test to say
  much; that arm of the test rests on govdocs2's 793.
- The adversarial-construction conclusion is inferred from score distributions
  and the suite's own `role: adversarial` declaration, not from documentation of
  how the eval half was sampled. If that sampling procedure is written down
  somewhere, it is worth confirming against it.

## Artifacts

| Path | What |
| --- | --- |
| `probe/gate_diagnosis.json` | 15 gate fits: alpha × balance, AUC and recall, held-in vs sealed |
| `probe/grouped_split.json` | per-document vs grouped carve, per corpus |
| `training/h2h_gate_diag.py` | the (a)/(b)/(c) separation |
| `training/h2h_grouped_split.py` | the grouped carve and its refutation |
| `reports/26-08-25_head-to-head.md` | the head-to-head this follows from |
