# Model Card — `pii-steady-aim-cascade-v1`

A 58-label CPU document tagger with a document-level gate in front of it. It
answers two questions: *does this document contain sensitive PII at all* (one
dot product, and the question this model was built for), and *which of 58
PII/PHI/PCI tags does it carry*. On the eight sealed evaluation corpora it
scores **priority macro F0.5 0.7474**, **document precision 0.8870** and
**document specificity 0.8798**, at **4.11 ms p95 per document on a single CPU
core** (310 docs/s), from a 28 MB NumPy-only artifact.

- **Version:** v1 — first release of the `pii-steady-aim` lineage
- **Run:** `pii-steady-aim` (2026-08-25), 1,000 MLflow-tracked trials, 16
  measured arms
- **Licence:** research / internal only — see *Training data and licensing*
- **Replaces:** nothing. It does **not** supersede
  `pii-priority-fusion-1k-v1`; see *Where this model is worse*.
- **Registry:** **not promoted.** No `@champion` alias was assigned, because the
  artifact does not clear the run's gate set.

---

## ⚠️ Things to improve

### 1. It does not pass the gates it was measured against — read this first

This bundle was packaged on explicit request as the run's best artifact, **not**
as a gate-passing champion. Of five hard constraints in
`pii-steady-aim-precision-gate` it clears one. Every bar is judged per corpus on
the bootstrap lower bound with a 30-instance minimum:

| Gate | Required | Achieved (worst corpus) | |
| --- | ---: | ---: | --- |
| document precision | ≥ 0.90 | **0.8024** (datax) | 1 of 3 corpora |
| document specificity | ≥ 0.85 | **0.8486** (datax) | 2 of 3 corpora |
| document recall | ≥ 0.85 | **0.6534** (datax) | 1 of 3 corpora |
| per-priority-tag recall | ≥ 0.75 | **0.2000** (password @ betterdataai) | 42 of 55 pairs |
| one-core p95 | ≤ 5 ms | 4.11 ms | **PASS** |

Do not describe this model as meeting a 0.90 document-precision bar or a 0.75
per-tag recall floor. What it does clear, on 126,129 sealed documents, is
roughly *document precision 0.80, specificity 0.85, recall 0.65, and per-tag
recall 0.75 on 42 of 55 measurable pairs*.

### 2. Document recall on real business documents is the weakest number

0.6743 on datax and 0.6696 on govdocs2 — the two corpora made of real files
rather than generated text. **Roughly one in three real documents that do
contain sensitive PII is passed as clean.** This is the direct cost of the gate
that produces the specificity: with the gate disabled, document recall rises to
0.89 and specificity collapses to 0.58 (measured in the sibling run
`pii-quiet-alarm`, `docs/`). Anything that must not miss a document needs a
second pass, a lower gate threshold, or human review.

### 3. Thirteen tag × corpus pairs sit under the 0.75 recall floor

Worst first: `password`@betterdataai 0.32, `medical_record_number_mrn`
@betterdataai 0.36, `address`@govdocs2 0.55, `address`@betterdataai 0.63,
`personal_identification_number_pin`@ai4privacy 0.66, `full_name`@govdocs2 0.61,
`address`@datax 0.65, `full_name`@ai4privacy 0.64, `iban`@ai4privacy 0.72,
`driver_s_license_number`@openpii 0.69, `social_security_number`@ai4privacy
0.74, `driver_s_license_number`@ai4privacy 0.75, `passport_number`@ai4privacy
0.76.

`address` and `full_name` fail across several corpora and are the two tags the
frozen taxonomy collapse merged components into — either the collapse is wrong
for them, or they need cue features the hashed representation does not capture.
Full table in `docs/`.

### 4. betterdataai is far weaker than every other corpus

Priority macro F0.5 **0.3570** there against 0.9296 on pii2. Its labels are
silver (model-generated). The run cannot separate "the labels are noise" from
"the model genuinely misses these", and does not pretend to.

### 5. It has never been evaluated at a cheaper read depth

Only the `deep` read profile (12,000 characters, 2,048 tokens) was ever
evaluated as a cascade — the top document gates were all `deep`, so
profile-mismatched component pairs were pruned before scoring. A `fast`
(1,000-character) variant would be roughly 9× quicker and is completely
unmeasured. Absence of evidence, not evidence of absence.

---

## Measured performance

Equal-corpus mean across the corpora whose gold can measure each quantity.
Precision-bearing tag metrics come from the five label-complete corpora;
document precision, specificity and recall come from the three that hold genuine
negatives. The other cells are **not measurable**, never zero.

| Measure | Result |
| --- | ---: |
| Equal-corpus priority macro F0.5 (16 priority tags) | **0.7474** |
| Equal-corpus macro F0.5 (full catalogue) | 0.6932 |
| Equal-corpus priority macro precision | 0.7520 |
| Equal-corpus priority macro recall | 0.8076 |
| Equal-corpus document precision | 0.8870 |
| Equal-corpus document specificity | 0.8798 |
| Equal-corpus document recall | 0.7582 |
| One-core p95 latency, 10 KB document | 4.11 ms |
| One-core throughput | 310 documents/second |
| Prediction rate (documents receiving ≥1 tag) | 0.7735 |

### Per corpus — quote the equal-corpus number, not the best row

| Corpus | priority F0.5 | precision | recall | n |
| --- | ---: | ---: | ---: | ---: |
| pii2 | 0.9296 | 0.9268 | 0.9597 | 30,000 |
| pii_holdout | 0.8473 | 0.8469 | 0.9141 | 20,000 |
| openpii | 0.8325 | 0.8388 | 0.8596 | 38,937 |
| ai4privacy | 0.7707 | 0.7941 | 0.7726 | 10,626 |
| betterdataai | 0.3570 | 0.3533 | 0.5320 | 10,360 |
| datax · govdocs2 · nemotron | not measurable (positive-only or coarse gold) | | | 16,206 |

**pii2 is the flattering corpus.** It is synthetic and shares generators with
training-side material. Quote **0.7474**.

### Against the model it does not replace

| | `pii-priority-fusion-1k-v1` | this model |
| --- | ---: | ---: |
| priority macro F0.5 | 0.2057 | **0.7474** |
| document precision | 0.6162 | **0.8870** |
| document specificity | 0.0005 | **0.8798** |
| document recall | **0.9994** | 0.7582 |
| priority tag recall gates | **55/55** | 42/55 |
| one-core p95 | **2.03 ms** | 4.11 ms |

---

## Where this model is worse

Nearly every promotion trades something; this one trades a lot, and the previous
model remains the better choice for some callers.

- **Recall.** `pii-priority-fusion-1k-v1` clears all 55 per-tag recall gates and
  has document recall 0.9994. If a missed identifier is a reportable incident
  and a false alarm merely costs a reviewer a minute, **stay on the old model.**
  This one misses about a third of real PII-bearing business documents.
- **Latency.** 4.11 ms against 2.03 ms p95, because it reads 12,000 characters
  to the old model's 1,000.
- **It is only better at being quiet.** That is the whole point of it — the old
  model tags 99.98% of all documents and has document specificity 0.0005 — but
  "better" here means one specific thing, not everything.

## Intended use and limits

Intended for internal or research triage, routing, and reducing reviewer load on
document sets where most files are clean and false alarms are expensive. The
document gate is the primary interface; the tag list is a hint for a reviewer.

**Human review or a downstream span detector is required** before redaction,
blocking, disclosure, or any decision affecting a person.

Do not use it as proof that a document is free of PII (it misses roughly one in
three real positive documents), as a compliance certification, as an
access-control boundary, or for automated legal, medical, employment, credit,
insurance or identity decisions.

**This is document classification, not span NER.** It cannot locate or redact a
value, and a positive tag does not identify which text triggered it. That is the
adjacent capability people most often assume; it is not present.

## Architecture and inference

Two stages over one shared feature extraction:

1. **Document gate** — a single linear model over 2^18 hashed, value-redacted
   word/character/shape features. If its score is below threshold the document
   is reported clean and stage 2 never runs.
2. **Per-tag heads** — 57 enabled one-vs-rest linear heads with per-label
   thresholds, each chosen so the recall floor holds on the **worst training
   source group** rather than on the pooled average.

- Runtime: Python 3.10+ and NumPy. No other dependency.
- Read profile: 12,000 characters, 2,048 tokens, 1,024 hashed features per doc.
- Catalogue: 58 collapsed labels (name components → `full_name`, street →
  `address`).
- Features are value-redacted: digits are normalised to `0` and shape cues fire
  on the *form* of an identifier, so no observed value is retained in a feature.
- Integrity: verify all members against `SHA256SUMS`.

## Training data and licensing

Fitted on 451,845 documents (85% of a 531,431-document universe; the remaining
15% carried the calibration split that chose every threshold). 70,600 of those
documents are labelled PII-free, 20,639 of them real business documents that a
previous loader discarded — recovering them is the change that made this model
possible.

Evaluation used 126,129 documents across eight sealed directories, scored one
corpus at a time, with 0 exact overlaps against training after an audit removed
22,816 leaking training rows.

Sources include AI4Privacy and NonCommercial material, and several corpora carry
synthetic or model-generated (silver) labels. **Use is restricted to research
and internal evaluation until redistribution and commercial rights are
confirmed.** Real-world drift, OCR corruption, new jurisdictions and adversarial
formatting all require independent validation.

## Provenance

Run `pii-steady-aim`, 2026-08-25. Approved budget: 1,000 trials, all spent.
Source commit recorded in `models/metadata.json`. Full per-corpus, per-tag,
bootstrap and threshold evidence in `docs/`.

Post-packaging verification re-scored a sealed corpus **through this bundle's
own `tagger.py`** and reproduced its recorded priority macro F0.5; the result is
in `models/verification.json`.
