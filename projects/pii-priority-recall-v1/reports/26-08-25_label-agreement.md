# Label agreement between corpora

## Results

**The model is at 96% of the ceiling the labels can measure.** Two independent
judge models labelled the same **4,708 datax documents**, and across the 20 PII
tags with support ≥ 30 they agree with each other at **macro F1 0.5138**
(support-weighted **0.4686**). `perlabel_v4` scores macro F2 **0.4932**.

That is the whole explanation for the plateau. Five model families and 1,000
trials converged to 0.42–0.49 not because the search was inadequate but because
**no model can be measured above the reliability of the labels measuring it**,
and that reliability is ≈0.51.

| Evidence | Value |
| --- | ---: |
| Target as written | 0.90 |
| Champion's bootstrap CI upper bound (optimistic) | 0.5892 |
| **Inter-judge macro F1 (model-independent)** | **0.5138** |
| `perlabel_v4` today | 0.4932 |
| **Verdict** | **unlikely** |

Recorded in `feasibility_agreement.json`; the loop is now
`target_infeasible` with `renegotiated_target = 0.5138`.

### The disagreement is mostly taxonomic, not perceptual

The per-tag numbers looked alarming until the pattern showed itself. The two
judges are not disagreeing about what is in the documents. They are disagreeing
about **which tag to file it under**.

| Tag | Judge A + | Judge B + | F1 |
| --- | ---: | ---: | ---: |
| `full_name` | 455 | 1,809 | 0.3993 |
| `given_name` | 1,513 | 261 | 0.2864 |
| `family_name` | 1,547 | 312 | 0.3141 |
| `middle_name` | 329 | 3 | 0.0181 |
| **collapsed to one NAME class** | **1,878** | **1,884** | **0.9920** |

Judge A decomposes a person's name into given/family/middle. Judge B records
`full_name`. Both see the same person: they agree on **name presence at F1
0.9920, Cohen's κ 0.9867** — 1,866 of ~1,880 documents. Split across four tags,
that near-perfect agreement reads as F1 0.018–0.40, and the fixed evaluator
scores the difference as *model error*.

`sensitive_pii_full_name` is the project's **highest-support priority tag**
(60,902). Its measured ceiling is 0.399. That single taxonomy choice is the
largest measurement artifact in the run.

Address shows the same effect more weakly — `address` 0.1286 and
`street_number_and_name` 0.6953 individually, **0.7396** collapsed — so part is
taxonomic and part is genuine disagreement about what counts as an address.

**Format-anchored tags need no collapsing and already agree well:** `email`
0.9095, `phone_number` 0.8939, `age` 0.8515. Semantically-defined tags are
where the reliability goes: `employment_status` 0.0440,
`country_of_residence` 0.3448, `zip_code` 0.5661, `medical_condition` 0.5682.

### What collapsing the taxonomy would buy

Merging the four name tags into one and the two street-address tags into one —
changing nothing else:

| Taxonomy | Macro F1 ceiling | Support-weighted |
| --- | ---: | ---: |
| Current (20 tags) | 0.5138 | 0.4686 |
| **Collapsed (16 classes)** | **0.6354** | **0.7879** |

The weighted figure nearly doubles because NAME is the largest class by
support. **This is a change to the evaluator, so it is not something to do
quietly** — it needs explicit approval and a re-run of every arm under the
corrected definition, exactly as widening a policy mid-run would.

### Cross-corpus: the corpora also disagree with each other

The eight corpora share no documents, so inter-annotator agreement is undefined
across them. What *is* measurable: hold the model fixed and ask what score
threshold each corpus demands to reach 0.90 recall on the same tag. If they
mean the same thing by a tag, the thresholds should be similar.

| Tag | Corpora | Threshold spread | Prevalence spread |
| --- | ---: | ---: | ---: |
| `medical_record_number_mrn` | 3 | **3.8×** | 1.5× |
| `full_name` | 6 | **3.1×** | 2.6× |
| `credit_card_number` | 4 | **2.9×** | 18.4× |
| `password` | 5 | 2.1× | **29.0×** |
| `address` | 7 | 2.0× | 17.0× |
| `military_identification_number` | 3 | 1.5× | **167.7×** |
| `passport_number` | 4 | 1.2× | 36.8× |
| `iban` | 2 | 1.1× | 3.1× |

Threshold spread is the more trustworthy column — prevalence can differ for
honest compositional reasons (a medical corpus really does hold more MRNs),
whereas a 3.8× difference in the score level at which the *same model* finds a
corpus's positives means those positives look materially different. MRN,
`full_name` and `credit_card` are where the corpora least agree about what the
tag means.

This is the mechanism behind the 0.63 / 0.23 spread on `openpii` versus
`betterdataai_ner_silver` noted in the previous report: the same model, the
same tags, different definitions.

## TL;DR

- **Two independent judges over the same 4,708 documents agree at macro F1
  0.5138.** The champion scores 0.4932 — **96% of the ceiling**. The 0.90
  target was never reachable, and this evidence is model-independent, so it
  does not depend on any arm's defects.
- **Most of the disagreement is taxonomic.** The judges agree on whether a name
  is present at **F1 0.9920 / κ 0.9867**; split across `full_name` /
  `given_name` / `family_name` / `middle_name` that reads as 0.018–0.40. The
  evaluator scores that bookkeeping difference as model error, on the
  highest-support tag in the project.
- **Collapsing names and street-address raises the ceiling to macro 0.6354 /
  weighted 0.7879** — the single largest lever available, and roughly 3× more
  headroom than the per-label threshold work bought. It is an evaluator change
  and needs sign-off plus a full re-run.
- Format-anchored tags are fine: `email` 0.909, `phone` 0.894, `age` 0.852.
  Semantic tags are where reliability collapses: `employment_status` 0.044.
- Loop recorded as `target_infeasible`, `renegotiated_target = 0.5138`.

## Caveats worth stating

- Agreement was measured on **datax**, whose judge vocabulary is close to but
  not identical with the 16 gated priority tags; the champion's macro F2 is
  measured over eight corpora. These are indicative of the same quantity, not
  the identical one. That five independent model families all plateaued just
  under 0.51 is corroboration, not proof.
- The ceiling is an **F1** agreement compared against an **F2** headline. For
  two symmetric judges precision ≈ recall so F1 ≈ F2; the asymmetric tags
  (`middle_name`, 329 vs 3) are taxonomy mismatches where "which judge is gold"
  is arbitrary anyway.
- Both judges are LLMs, so this bounds *this* labelling process. Human
  double-annotation might agree better — or, on tags like `employment_status`,
  worse.

## Reproducing

```
training/label_agreement_probe.py     # Part A inter-judge, Part B cross-corpus
projects/pii-priority-recall-v1/label_agreement.json
projects/pii-priority-recall-v1/feasibility_agreement.json
```
