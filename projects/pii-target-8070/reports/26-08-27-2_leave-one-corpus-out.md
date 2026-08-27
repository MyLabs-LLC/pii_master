# Leave-one-corpus-out: 21% of every published number comes from having seen the source

## Results

Nine folds. Each trains the cascade from scratch on eight of the nine sources —
gate, 61 heads, and threshold selection on that fold's own calibration carve —
then evaluates on the ninth source's sealed split. The model has never seen a
document from that source, nor anything drawn from its generator.

| held-out source | n eval | in-distribution | **LOCO** | gap |
| --- | ---: | ---: | ---: | ---: |
| `synthetic_pdf` | 322 | 0.9233 | 0.5856 | **0.3377** |
| `betterdataai` | 10,360 | 0.5414 | 0.3388 | 0.2026 |
| `pii2` | 30,000 | 0.7702 | 0.6035 | 0.1668 |
| `ai4privacy` | 10,626 | 0.6201 | 0.5048 | 0.1153 |
| `openpii` | 38,937 | 0.8712 | 0.7754 | 0.0958 |
| `pii_holdout` | 20,000 | 0.8497 | 0.8053 | **0.0444** |

`in-distribution` is `cascade_v2_9corp` — trained on all nine — on the same corpus.

**Three folds are excluded, not failed.** `datax` (positive-only gold),
`nemotron` (partial gold, no sensitive tags recoverable) and `govdocs2`
(positive-only) cannot support precision, so they report no micro F1. Scoring
their precision as `0.0` would charge the model for annotation gaps; the
evaluator reports `NOT_MEASURABLE` instead.

**Mean transfer gap 0.1604** (median 0.1411, worst 0.3377). The nine-corpus
headline of 0.7627 becomes **0.6022** on an unseen source — roughly **21% of
every published in-distribution point is source familiarity rather than
capability.**

### Recall is the fragile half; precision transfers

| | in-distribution | LOCO | Δ |
| --- | ---: | ---: | ---: |
| micro precision | 0.6772 | 0.5768 | −0.1004 |
| micro recall | 0.8996 | 0.6441 | **−0.2556** |

Recall falls 2.5× harder than precision. This is the same result the 11×9
evaluation matrix reached from the opposite direction — there, holdout precision
spanned only 0.07 across every 61-label model while holdout F1 tracked recall at
r = +0.97. Two independent measurements, one conclusion.

### LOCO recovers the measurement the corpus split destroyed

Splitting `Synthetic_PDF_Corpus_v2_1612` 80/20 into training and eval consumed
the only out-of-distribution test in this repository. LOCO scores that source
**0.5856**; the genuine out-of-distribution measurement, taken before the split,
was **0.5836**. It recovers the lost number to within 0.002 — so LOCO is a
working replacement for the holdout, not merely a proxy for it.

## TL;DR

- **Expect ~0.60 micro F1 on a genuinely new source**, not the 0.7627 headline.
  The mean transfer gap is 0.1604.
- **Recall collapses on transfer (−0.2556); precision largely holds (−0.1004).**
  A precision-tuned model will keep most of its precision on new data and lose
  about a quarter of its recall.
- **This directly threatens the 90/80/80 target.** The box was measured
  in-distribution and the recall leg is the one that breaks out-of-distribution.
- **The gap is very uneven.** `pii_holdout` transfers nearly free (0.0444);
  `betterdataai` drops to 0.3388 absolute. Published numbers for the
  high-gap sources are interpolations, not predictions.
- **LOCO is validated** against the destroyed holdout to within 0.002.
- **The threshold variants are unmeasured under LOCO** — this run uses the
  baseline policy. Their transfer is inferred, not known.

---

| | |
| --- | --- |
| Date | 2026-08-27 |
| Project | `projects/pii-target-8070` |
| Request | *"yes retrain and eval all the models. this needs to generalize."* |
| Scope | 9 leave-one-corpus-out folds, full refit each |
| CPU budget | 1 core |
| Artifacts | `evaluations/loco.json`, `evaluations/loco_gap.json` |
| Outcome | transfer gap measured; **no promotion** |

## Why this run exists

Every headline in this repository is measured on corpora whose training halves
the model was fitted on. That answers "how good is it here" and is silent on
"how good is it somewhere new". The one corpus that could have answered the
second question became 80% training data in the preceding step.

The retrain that preceded this run made the point sharply: adding the PDF corpus
improved the original eight by **+0.0007** (0.7299 → 0.7305), while the
nine-corpus headline rose to 0.7627 almost entirely because the new corpus
scores 0.9233 on itself. A headline that moves because a source was added to
training is not a capability gain, and nothing in the sealed suite could
distinguish the two. LOCO can.

## What is still open

1. **The threshold variants are unmeasured under transfer.** `p88r90` and
   `p90r85b1` trade recall for precision, and recall is the leg that breaks.
   Nine folds × re-derived thresholds would settle it.
2. **No precision measurement exists on real file formats.** All nine sources
   are extracted or synthetic text. `clean_docs_10000_no_pii_phi_pfi` holds
   7,126 documents (PDF/DOC/XLS/PPT/HTML) never seen by any model, judged to
   contain no PII — pure negatives, on which precision is unambiguous. The
   other 2,874 overlap the govdocs2 corpora and must be excluded.
3. **`betterdataai` at 0.3388 is unexplained.** A source that the other eight
   cannot teach at all is either genuinely distinct or defective, and the gold
   audit already found its `email` precision ceiling at 0.5779 against 1.0000
   elsewhere.
4. **No model has cleared a gate**, in any project, across this lineage.
