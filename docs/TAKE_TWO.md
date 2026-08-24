# Take two — knowledge-augmented detection with privacy-risk indicators

**Status:** v0.4. The first branch (`cursor/improvement-plan-04f6`) sequenced
"harden the rules, then distill a student." Main shipped that cascade. This
branch is a different idea, taken from the current sanitization literature
rather than from another training run.

---

## 1. What the papers actually say

Surveyed 2026-08-23. The field's unsolved problem is not "a better CNN."
docs/PRIOR_ART.md already recorded that: **cross-domain generalisation and
quasi-identifier combinations**, not architecture. The papers that changed
what this repo should build next:

| Paper | Finding we used |
|---|---|
| **Papadopoulou, Lison, Anderson, Øvrelid, Pilán** — *Neural Text Sanitization with Privacy Risk Indicators* (arXiv 2310.14312; LRE 2026) | Two-step sanitization: (1) a privacy-oriented recognizer that **combines NER with a person-term gazetteer**, (2) **privacy-risk indicators** that score spans *in combination*, not just in isolation. Direct vs quasi identifiers, as in TAB. |
| **Pilán et al.** — Text Anonymization Benchmark (CL 48(4) 2022) | An entity is protected only if **every mention** is found. Aggregate F1 hides the operational failure. |
| **Golle (2006)** / HIPAA Safe Harbor | Gender + DOB + ZIP uniquely identifies 63–78% of the US. Combinations, not sums. |
| **SHIELD** (arXiv 2605.03301, 2026) | Structured PHI (dates, phones, IDs) transfers across institutions; institution-specific identifiers do not. Optimal deployment pairs a broad model with **specialised structure detectors**. |
| **PIIBench** (arXiv 2604.15776) + follow-up (arXiv 2605.25816) | Data diversity beats architectural complexity. A DeBERTa on diverse data reaches 0.65 F1; eight published systems sit below 0.14. |
| **PRIOR_ART.md §7** (this repo, unused) | Add a gazetteer tier *before* any neural tier. Aho-Corasick / hash lookup is O(document). |

What we did **not** pick, and why:

- GLiNER2-PII / LOGICAL / OpenBioNER — 15–300 M transformers, 100+ ms on CPU.
  They are the accuracy ceiling, not a serving candidate (PRIOR_ART §2).
- Retraining the student on more data — already tried (ai4privacy mix). The
  remaining miss is names and format-anchored types the rules never covered.
- LLM-in-the-loop (the 2026 multilingual de-id paper) — forbidden on the
  serving path (DESIGN.md §5).
- Papadopoulou's LLM-probability, web-search, and perturbation indicators —
  they need a language model or a network call. We took the one indicator
  that fits the box: **span classification + combination risk**.

---

## 2. What shipped

Three additions, all stdlib, all in the `fast` path.

### 2.1 Structure-verified detectors

Checksum / format validators for HIPAA rows that were model-only:

| Type | Validator | Cue |
|---|---|---|
| `BANK_ROUTING` | ABA 9-digit checksum | routing / ABA / RTN |
| `VEHICLE_ID` | ISO 3779 VIN check digit | none (17-char + checksum) |
| `SWIFT_BIC` | ISO 9362 structure | SWIFT/BIC, or a digit-bearing standalone |
| `MAC_ADDRESS` | 6 hex octets, one separator | none |
| `FAX_NUMBER` | NANP + fax/facsimile cue | required (else it stays `PHONE_US`) |
| `TAX_ID` | US EIN prefix + 9 digits | EIN / tax id / TIN |

These types join `CHECKSUMMED_TYPES`. A model span of the same type is
re-validated before it may ship, so the student can no longer emit a routing
number no bank would accept.

All-alpha English words (`SOFTWARE`, `withholding`, `Beneficiary`) are
structurally valid BICs. Cue-less SWIFT therefore requires a digit;
`DEUTDEFF` still fires when the sentence says `SWIFT`.

### 2.2 Context-gated gazetteer

A hash-set lookup over a compact, diverse public-domain name list (SSA /
Census given names and surnames). It is *not* "every capitalised token":

- `PERSON_NAME` only on first+last (both in the list), first + middle
  initial + capitalised last, or title + gazetteer name. The title is not
  part of the span (`Dr. Jane Doe` → `Jane Doe`).
- `ADDRESS` only on house-number + street + suffix, optional `, City`.

Single gazetteer hits stay silent. That is why `The Young team meets in
Park Hall after the May review` is `NONE`.

Detector names stay under `regex/`, so fusion treats them as cue-tier
rules: checksum facts outrank them, and in `--deep` the student still
outranks them on overlap. Rules-only fusion ranking stays off.

### 2.3 Privacy-risk indicators

`DocumentReport` now carries `direct_count`, `quasi_count`, and
`reidentification_combos`. Combinations that add risk (longest first, no
double-counting):

| Combo | Bonus |
|---|--:|
| name + DOB + address | 20 |
| name + DOB | 12 |
| name + address | 12 |
| DOB + address | 12 |
| DOB + phone | 8 |

The document **label** is unchanged. A combo raises the explainable risk
score and writes a reason; it does not invent PHI. False PHI is still the
failure class Track A closed.

Eval reports **document leakage rate**: the fraction of
identifier-bearing documents that missed at least one *detectable* gold
span. A rise is a CI regression; a drop is not. Types still in
`MODEL_ONLY_TYPES` do not count as leaks on the rules path.

---

## 3. What it measures on the frozen corpus

Rules-only, after this change (append-only corpus, new rows at
`pii-014`–`pii-019` and `none-015`–`none-016`):

- Document accuracy **1.00**, PHI recall **1.00**, leakage **0.00**.
- `PERSON_NAME` and `ADDRESS` move from recall 0 (undetectable) to
  exact 1.00 on the rows the corpus already had (`Jane Doe`,
  `John Q. Patient`, `44 Elm Street, Springfield`).
- The six new structure types each have a frozen TP and a frozen FP.

The Nemotron holdout is not re-run here: this is a rules-tier change, and
the committed `docs/BASELINE_NEMOTRON.md` is the rules-only number on the
original 12 types, which this does not touch.

---

## 4. What this is not

It is not a claim that a 500-name gazetteer replaces the student. Out-of-
domain names will still miss; that is why `--deep` exists. It is the
fast-path recall the literature said we were leaving on the table, plus
the combination-risk score the literature said a sum-of-weights cannot
express.
