# Model Card — `pii-cascade-balanced-v2`

A CPU document tagger for sensitive data (PII / PHI / PCI): a document-level gate
in front of 57 per-tag heads. NumPy only, 28 MB, 3.92 ms p95 per document on one
core.

Selected as the best of **sixteen** models measured on eight sealed corpora in
one head-to-head run — two architectures re-tuned from scratch over 2,000 trials
on the full 531,431-row training corpus, plus twelve operating points and a gate
variant. It is the best measured artifact of that run.

**It does not clear the gates it was measured against.** Section 1 says which,
by how much, and — new in this version — why the evidence says those gates are
not reachable with this architecture at all.

---

## ⚠️ Things to improve

### 1. It clears 29 of 55 priority recall gates, and the rest are not reachable here

The declared policy requires every measurable priority tag × corpus pair to clear
recall 0.90 on a bootstrap lower bound. This model clears **29 of 55** (the model
it derives from cleared 25). At the relaxed 0.75 floor it clears **45 of 55**.

The document-level policy is missed too, and this is the part worth reading. It
asks for document precision ≥ 0.90, specificity ≥ 0.85 and recall ≥ 0.85
simultaneously on real-world documents. Walking the full frontier of the measured
score distribution, the best simultaneously achievable is:

| | reachable | bar | shortfall |
| --- | ---: | ---: | ---: |
| recall, holding precision ≥ 0.90 and specificity ≥ 0.85 | **0.623** | 0.85 | 0.227 |
| precision, holding recall ≥ 0.85 | **0.766** | 0.90 | 0.134 |

**No threshold on this model satisfies all three.** That is a property of the
ranking quality, not of the operating point, so no re-cutting fixes it. See
§ 2 for what does.

### 2. The limit is the representation, and it was measured three ways

Do not spend a labelling budget trying to close § 1. Three independent lines say
the ceiling is what hashed unigram/bigram/shape features can express about
real-world documents, not what the model has been shown:

- **Synthetic near-miss negatives failed.** 1,948 verified LLM-generated
  documents that read as sensitive and contain no personal data moved sealed
  real AUC 0.8453 → 0.8461. The gate learned them completely (held-out fire rate
  8.2% → 0.0%) and transferred nothing. Generated and real negatives separate at
  **AUC 1.0000** — style is a free shortcut.
- **Real target-distribution data failed.** Adding up to **7,370 labelled
  documents from exactly the distribution being tested** moved AUC 0.8832 →
  0.8778. Flat across a 15× range of label budget.
- **The in-distribution ceiling is 0.869.** Training *on* the target distribution
  and cross-validating within it does no better than transferring from the
  training distribution.

Closing § 1 needs contextual embeddings or a span-level NER model. That is a new
architecture with its own feasibility probe, not a tuning pass on this one.

### 3. Document recall on real business documents is the weakest number

Equal-corpus document recall is 0.797, but that average hides the split:

| corpus | document recall |
| --- | ---: |
| pii2 (synthetic) | 0.955 |
| datax (real) | 0.719 |
| govdocs2 (real) | 0.700 |

On real-world documents roughly **30% of PII-bearing files are rejected at the
gate**, and their tags are never scored. Every per-tag number on real documents
is capped by that.

### 4. The operating point was not re-optimised for this gate

Only the gate changed from the model this derives from; the tag thresholds and
the gate cut were re-derived under the *inherited* rule, and the cascade's joint
`gate_shift` was **not** re-searched — it is an absolute offset tuned against a
different score scale and carrying it over is invalid. A joint re-search would
likely gain a little more. Measured operating points at other recall floors are
in `docs/` and in the run's workbook.

### 5. betterdataai is far weaker than every other corpus

macro F2 0.31 there against 0.67–0.78 elsewhere. Its labels are silver
(model-generated), so it should never decide a gate alone.

### 6. It has never been evaluated at a cheaper read depth

Every number here is at the 12,000-character `deep` profile. No `fast` (1,000) or
`std` (4,000) cascade has been evaluated in this lineage, so nothing is known
about the quality/latency trade below 12,000 characters.

---

## Measured performance

Equal-corpus means over eight sealed corpora (126,129 documents), by a fixed
evaluator. A corpus contributes to a metric only if its gold can measure it:
three of the eight carry positive-only or partial tag gold, so precision-bearing
metrics are **NOT MEASURABLE** there and are excluded — never counted as zero.

| Measure | Result |
| --- | ---: |
| macro F2 (the run's declared ranker) | **0.6614** |
| macro F1 | 0.6398 |
| macro F0.5 | 0.6183 |
| macro precision / recall | 0.6182 / 0.6938 |
| micro F1 | 0.7255 |
| priority macro F0.5 (16 priority tags) | 0.7434 |
| worst measurable priority-tag recall | 0.7221 |
| document precision | 0.8956 |
| document specificity | 0.8792 |
| document recall | 0.7975 |
| prediction rate | 0.8192 |
| one-core p95 / throughput | **3.92 ms / 314 docs/s** |

### Per corpus — quote the equal-corpus number, not the best row

| corpus | n | macro F2 | micro F1 |
| --- | ---: | ---: | ---: |
| betterdataai_ner_silver | 10,360 | 0.32 | 0.66 |
| ai4privacy_pii_masking | 10,626 | 0.75 | 0.66 |
| pii_holdout | 20,000 | 0.76 | 0.82 |
| pii2_eval | 30,000 | 0.78 | 0.67 |
| openpii_pii_eval | 38,937 | 0.67 | 0.85 |
| datax · nemotron · govdocs2 | 15,206 | *NOT MEASURABLE* | *NOT MEASURABLE* |

**`pii2_eval` and `openpii` flatter this model.** Both are synthetic and
prevalence-heavy. The corpus nearest a real customer estate is `govdocs2`, and it
cannot measure macro F2 at all — its document-level numbers (recall 0.700,
precision 0.869) are the honest guide to real-world behaviour.

---

## Where this model is worse

Against `pii-steady-aim-cascade-v1`, which it supersedes, two metrics regress:

| | v1 | this | Δ |
| --- | ---: | ---: | ---: |
| micro F1 | 0.7313 | 0.7255 | **−0.0058** |
| priority macro F0.5 | 0.7467 | 0.7434 | **−0.0033** |

Both are small beside the gains (document recall +0.044, macro recall +0.046,
worst priority-tag recall +0.070, gates cleared 25 → 29). **If micro F1 is your
headline and real-world recall is not, stay on v1.**

Against `pii-priority-fusion-1k-v1`, which it does *not* supersede in one
respect: that model has higher raw recall (macro 0.795 vs 0.694). It achieves it
by firing on 99.99% of documents — its document precision equals the corpus base
rate to four decimals, meaning its document-level output carries **zero
information**. If you need maximum recall and will review every document anyway,
you do not need a model.

---

## Intended use and limits

**Use it for** triaging a document estate: deciding which files a human or a
heavier pipeline should look at, and labelling likely sensitive-data categories.

**Do not use it for**

- **Redaction, or as a system of record.** It predicts document-level tags, not
  entity spans. It cannot tell you *where* an identifier is, only that the
  document probably has one. Span NER is the adjacent thing people assume this
  does; it does not.
- **A compliance guarantee.** It misses roughly 30% of PII-bearing real-world
  documents at the gate (§ 3). A "clean" verdict is not evidence of absence.
- **Anything the declared policy gates on.** It clears neither the 0.90 per-tag
  recall bar nor the document-level bars (§ 1).
- **Non-English text.** The training corpora are English; nothing else was
  measured.

**Licensing:** research/internal only. Training sources include AI4Privacy
(NonCommercial) and machine-judged corpora. Not cleared for commercial use.

---

## Architecture and inference

A document gate — one weight vector over 2^18 hashed features — in front of 57
one-vs-rest linear per-tag heads. Inference is two dot products in the order that
makes the document decision cheap: extract features once, score the gate, stop if
the document reads clean, and only then score the tag heads.

```python
from tagger import Tagger
t = Tagger()
t.has_pii(text)    # bool — one dot product, the cheap question
t.predict(text)    # list — the tags, empty when the gate stays shut
```

| | |
| --- | --- |
| read window | 12,000 characters (`deep`) |
| tokens / features per document | 2,048 / 1,024 |
| hashed feature space | 262,144 |
| labels | 58 collapsed (57 enabled; one disabled for thin support) |
| gate threshold | 0.0620 |
| runtime | NumPy only |

The gate's short-circuit saves less than it appears: silent p95 3.46 ms against
firing 4.04 ms. At a 12,000-character window, tokenising dominates; the gate
skips 58 dot products, which were never the expensive part.

**Taxonomy collapse** (applied identically to gold and predictions): given,
family and middle name → `full_name`; street number and name → `address`. Two
independent judges agree at F1 0.99 on whether a name is present and 0.02–0.40 on
which name tag it is, so the uncollapsed distinction is mostly gold noise scored
as model error.

---

## Training data and licensing

531,431 documents across eight corpora, all eight used. 451,548 rows fitted,
79,883 held in for calibration. The gate's loss is **source-balanced**: real-world
documents are 33,637 of 432,903 fit rows with document gold (7.8%), and equalising
their contribution is the single change that produced this version.

The evaluation corpora were never trained on, never used to select an operating
point, and were scored once per arm by a fixed evaluator. Cached-feature
predictions were verified identical to the model's own `predict()` on sampled
documents — 0 mismatches of 96.

---

## Provenance

| | |
| --- | --- |
| Run | `pii-head-to-head-v1` (H1), 2026-08-25/26 |
| Derived from | `pii-steady-aim-cascade-v1`, gate refit balanced + regularised |
| Models measured in the run | 16, on 8 sealed corpora — 128 scored results |
| Search | 2,000 trials across two lineages |
| Registry | not registry-promoted; packaged as the run's best measured artifact |
| Evidence | `docs/` — four run reports, decisions, feasibility probe, per-tag tables |

**Not promoted to `@champion`.** It fails hard constraints in both declared
policies (§ 1), and this run's own rule is that a gate-failing artifact is not
promoted. It is packaged because it is the best measured model of the run and
because § 2 establishes that further tuning of this architecture will not change
that.
