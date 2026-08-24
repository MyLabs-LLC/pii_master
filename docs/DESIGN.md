# pii_master — Design & Roadmap

**Status:** v0.4 — Stage 1 hardened; evaluation + benchmark harnesses and a measured
baseline (docs/BASELINE_M1.md) committed, plus an **external** baseline on a
Nemotron-PII holdout (docs/BASELINE_NEMOTRON.md). **Stage 2 is built and shipped
behind `--deep`**: distilled, exported, fused, and measured — docs/DISTILLATION_PLAN.md,
docs/DISTILLATION_RESULTS.md, docs/STAGE2_INTEGRATION.md.
Post-M1 review: docs/IMPROVEMENT_PLAN.md. Take two (gazetteer + structure
validators + privacy-risk indicators): docs/TAKE_TWO.md.
**Production constraint:** inference must run on **1 CPU core / 4 GB RAM** and finish in
**5 ms per document (p95), end-to-end**. Training may use a GPU.

This document is the answer to the question *"how would you start solving document
classification of PII and PHI, step by step?"* — written as an executable design. Section
order mirrors the build order.

---

## 1. Problem statement & goals

Given a document, decide:

1. **Document label** — does it contain personally identifiable information (PII),
   protected health information (PHI), or neither?
2. **Evidence** — which spans of text triggered that decision (entity type, offsets,
   confidence, which detector found it)?
3. **Risk score** — an explainable 0–100 number so downstream policy can triage
   (block, quarantine, redact, allow).

Goals, in priority order:

- **Recall-first for PHI.** A missed medical record number leaking into a data lake is a
  reportable incident; a false alarm costs a reviewer minutes. When precision and recall
  conflict, we choose operating points that favor recall for PHI-relevant types, and we
  keep precision high by *validating* candidates (checksums, known-invalid ranges) rather
  than by narrowing recall.
- **Explainability.** Every document label must decompose into "these spans, found by
  these detectors, contributed this much." No opaque scores.
- **Fit the box.** The production container is 1 CPU core and 4 GB of memory, and the
  whole pipeline must answer in 5 ms per document at p95. Every architectural choice is
  made under that budget (§5); the latency ceiling, not memory, is the binding
  constraint.
- **Composability.** Detection strategies will change (rules today, learned models
  tomorrow); the contracts between stages must not.

Non-goals for v1 are listed in §12.

## 2. Definitions: PII vs PHI

**PII** is any information that can identify an individual, alone or in combination:
direct identifiers (SSN, passport number) and quasi-identifiers (DOB, ZIP code, gender —
individually weak, jointly identifying).

**PHI** (per HIPAA) is individually identifiable health information created, received, or
maintained by a covered entity or business associate. The critical property: **PHI is
PII plus health-context linkage.** An SSN in a bank statement is PII; the same SSN in a
discharge summary is PHI. Consequently our labels are ordered — `NONE < PII < PHI` — and
PHI implies PII is present.

The PHI taxonomy anchor is the **18 HIPAA Safe Harbor identifiers**
(45 CFR §164.514(b)(2)), which enumerate what must be removed for a dataset to be
considered de-identified:

| #  | Identifier |
|----|------------|
| 1  | Names |
| 2  | Geographic subdivisions smaller than a state (street, city, county, ZIP*) |
| 3  | All elements of dates (except year) directly related to an individual — birth date, admission date, discharge date, death date; all ages over 89 |
| 4  | Telephone numbers |
| 5  | Fax numbers |
| 6  | Email addresses |
| 7  | Social Security numbers |
| 8  | Medical record numbers |
| 9  | Health plan beneficiary numbers |
| 10 | Account numbers |
| 11 | Certificate/license numbers |
| 12 | Vehicle identifiers and serial numbers, including license plates |
| 13 | Device identifiers and serial numbers |
| 14 | Web URLs |
| 15 | IP addresses |
| 16 | Biometric identifiers (finger/voice prints) |
| 17 | Full-face photographs and comparable images |
| 18 | Any other unique identifying number, characteristic, or code |

\* first 3 digits of ZIP retainable under population-size conditions.

Every entity type we ever add maps to one of these rows (or is PII-only, like a
non-medical account credential). That mapping lives in code (`entities.TAXONOMY`) so the
classifier and reports can cite it.

## 3. Why this is hard

- **Context-dependence.** `03/14/1985` is nothing; `DOB: 03/14/1985` is HIPAA identifier
  #3. `10.2.1.4` is a software version in a changelog and an IP address in an access log.
  `Chart #4829471` is an MRN; `chart topper` is prose. Rules can encode local context
  cues; genuinely resolving ambiguity needs learned models (Stage 2).
- **Precision/recall asymmetry.** The cost matrix is lopsided (missed PHI ≫ false alarm),
  but a scanner that cries wolf gets turned off — which is the worst recall of all. So
  precision is not optional; it just isn't the tie-breaker.
- **Format explosion.** SSNs appear hyphenated, spaced, or bare; phone numbers in a dozen
  notations; dates in endless formats. Some identifiers (MRNs, health plan IDs) have **no
  universal format at all** — every hospital system mints its own.
- **Quasi-identifier combinations.** No single field identifies anyone, but
  {ZIP, birth date, sex} famously identifies most of the US population. Document-level
  classification must eventually reason about co-occurrence, not just individual spans.
- **Document formats.** Real documents are PDFs, Word files, spreadsheets, and scans. OCR
  noise (`O`↔`0`, `l`↔`1`) breaks both regexes and checksums.
- **Multilinguality & locale.** Names, addresses, and national ID formats differ per
  country; v1 is deliberately US/English-scoped and says so.

## 4. System architecture: a staged pipeline

```
                     ┌────────────────────────────────────────────────┐
                     │                 pii_master                     │
 document text ──►   │                                                │
 (v1: plain text;    │  Stage 1: Rules & validators        [v1, now]  │
  M4+: PDF/DOCX/OCR) │    regex candidates → checksum/range           │
                     │    validation → typed spans                    │
                     │                    │                           │
                     │  Stage 2: Learned NER               [M2]       │
                     │    small transformer, GPU-trained,             │
                     │    int8 ONNX on CPU → typed spans              │
                     │                    │                           │
                     │  Stage 3: Aggregation               [v1 basic] │
                     │    merge spans → doc label (NONE/PII/PHI)      │
                     │    → explainable risk score                    │
                     └────────────────────┬───────────────────────────┘
                                          ▼
                       JSON report: label, risk, entities, reasons
```

The load-bearing decision is the **shared contract**: every detection strategy — regex
today, ONNX NER at M2, anything later — implements one structural interface
(`Detector.detect(text) -> list[Entity]`) and emits the same frozen `Entity` record
(type, char offsets, text, confidence, detector provenance). Stages compose because they
only exchange `Entity` lists:

- Stage 1 and Stage 2 both *produce* entities; neither knows the other exists.
- Stage 3 *consumes* entities regardless of origin, resolves overlaps, and classifies.
- Rules stay in the pipeline forever: post-M2 they act as high-precision overrides
  (a Luhn-valid card number found by regex outranks a model's vague "NUMBER" span), and
  checksum validators re-verify model-proposed spans.

Cheap-first ordering also serves the CPU budget: the regex stage can act as a pre-filter
(e.g., a document with zero digit runs and zero `@` cannot contain most identifier types),
letting Stage 2 skip or subset documents when throughput demands it.

## 5. Performance budget: 1 CPU core / 4 GB RAM / 5 ms per document

The serving budget has three sides:

- **1 CPU core** and **4 GB RAM** (the production container), and
- **≤ 5 ms per document at p95, end-to-end** — Stage 1 + Stage 2 + Stage 3 combined —
  measured by the in-repo benchmark harness (`pii-master bench`). The reference
  document is a *typical* one: ≤ 10 KB of extracted text (roughly 2–5 pages). Larger
  documents get a pro-rated allowance (5 ms per 10 KB) until a streaming design lands;
  the harness reports per-size buckets so both numbers stay visible.

Estimates are labeled as such; **measured** numbers for the current pipeline live in
docs/BASELINE_M1.md and are regenerated with `pii-master bench`. The standing rule: *no
optimization claim without a reproducible benchmark script in-repo.*

| Component | Memory target | Latency target (1 core, 10 KB doc) | Notes |
|---|---|---|---|
| Stage 1 rules | < 50 MB total RSS | ≤ ~2 ms (measured: see BASELINE_M1.md) | Compiled regex, one `finditer` pass per detector; stdlib only |
| Stage 2 NER (M2) | model artifact ≤ ~50 MB; steady-state RSS ≤ ~1 GB | ≤ ~3 ms within the shared 5 ms ceiling | Tiny model + candidate-window cascade — see §8 |
| Stage 3 aggregation | negligible | ≤ ~0.1 ms | O(n log n) in span count |

The 5 ms ceiling — not memory — is the binding constraint, and it has teeth:

- Full-document inference with even a distilled BERT-class encoder on one CPU core
  costs **tens of milliseconds** for a multi-hundred-token document (estimate to be
  confirmed on the harness) — an order of magnitude over budget. Stage 2 therefore
  cannot be "run a transformer over every token of every document"; it must be a tiny
  model, run selectively (§8).
- Stage 1 must consume well under half the ceiling so the model layer has room; the
  benchmark gates this per commit (`pii-master bench --fail-over-budget` in CI).
- Anything super-linear in document length must chunk with early exit.

Memory ledger (Stage 2 worst case): Python interpreter + onnxruntime (~200 MB) + model
(tens of MB int8) + tokenizer + working buffers + document text ≪ 4 GB, leaving headroom
for the serving layer. Nothing in this design requires loading more than one document in
memory at a time.

What the budget **rules out**: LLM inference in the hot path, full-document
transformer inference of any size, large encoder models, and per-document network calls.
GPU appears only in training/export/distillation pipelines (M2), never in the serving
container.

## 6. Entity taxonomy

v1 types (implemented in `src/pii_master/entities.py`):

| EntityType | PII | PHI-specific† | HIPAA row (§2) | Detection (v1) | Base confidence |
|---|---|---|---|---|---|
| `EMAIL` | yes | no | #6 | regex | 0.95 |
| `PHONE_US` | yes | no | #4 | regex + NANP validation | 0.85 |
| `SSN` | yes | no | #7 | regex + invalid-range rules | 0.90 hyphenated / 0.70 spaced |
| `CREDIT_CARD` | yes | no | #10 | regex candidates + Luhn checksum | 0.80, 0.95 with known IIN |
| `IP_ADDRESS` | yes (weak) | no | #15 | regex + `ipaddress` validation (v4 and v6) | 0.70 |
| `DATE_DOB` | yes | no | #3 | date regex + birth-cue proximity | 0.90 |
| `MRN` | yes | **yes** | #8 | cue-anchored regex | 0.85 |
| `URL` | yes (weak) | no | #14 | regex | 0.85 |
| `ACCOUNT_NUMBER` | yes | no | #10 | cue-anchored regex | 0.80 |
| `HEALTH_PLAN_ID` | yes | **yes** | #9 | cue-anchored regex (health-flavored cues only) | 0.80 |
| `US_DRIVER_LICENSE` | yes | no | #11 | cue-anchored regex | 0.80 |

v0.3 model-tier types (originally `--deep` only; `entities.STAGE2_TYPES`).
Take two added a rules path for names, addresses, fax, routing, SWIFT, VIN,
MAC and EIN; `MODEL_ONLY_TYPES` is what no rule can emit. None is
`phi_specific` — a name or an address is an identifier in any context, so
adopting them widens PII coverage without widening what may be called PHI:

| EntityType | HIPAA row | Nemotron labels it absorbs | Risk weight |
|---|---|---|--:|
| `PERSON_NAME` | #1 | `first_name`, `last_name` | 15 |
| `ADDRESS` | #2 | `street_address`, `city`, `county`, `postcode` | 20 |
| `GEO_COORDINATE` | #2 | `coordinate` | 20 |
| `DATE_TIME` | #3 | `date_time` | 5 |
| `FAX_NUMBER` | #5 | `fax_number` | 10 |
| `BANK_ROUTING` | #10 | `bank_routing_number` | 10 |
| `SWIFT_BIC` | #10 | `swift_bic` | 5 |
| `VEHICLE_ID` | #12 | `vehicle_identifier`, `license_plate` | 15 |
| `DEVICE_ID` | #13 | `device_identifier` | 10 |
| `BIOMETRIC_ID` | #16 | `biometric_identifier` | 25 |
| `MAC_ADDRESS` | #18 | `mac_address` | 10 |
| `NATIONAL_ID` | #18 | `national_id` | 30 |
| `TAX_ID` | #18 | `tax_id` | 20 |
| `USER_ID` | #18 | `user_name`, `customer_id`, `employee_id`, `unique_id` | 10 |

Multi-label collapses are deliberate. Nemotron tags "Jane Doe" as two spans and
"44 Elm Street, Springfield" as two more; `ner.merge_adjacent` rejoins same-type
spans separated only by whitespace or a comma, so one real-world identifier is
one entity in the report. The gap test is tight enough that "Jane Doe and John
Smith" stays two names.

Three Nemotron groups are still **not** adopted, and the reasons differ:
`state` (HIPAA #2 is subdivisions *smaller* than a state, so a state is
retainable — tagging it would be wrong, not conservative); the five credential
labels and the four GDPR special-category attributes (not HIPAA identifiers —
M3 policy profiles, §9); and the quasi-identifier attributes `date` / `time` /
`age` (identifiers only when tied to an individual or above 89, a distinction
the model cannot draw yet, so emitting them would flood every document).

`HEALTH_PLAN_ID` cues are deliberately restricted to unambiguous health wording
("health plan id", "beneficiary number", "subscriber id"); generic cues like "member id"
(gym, loyalty program) or "policy number" (any insurance) are excluded so the
phi-specific flag stays honest. Cue-free plan IDs are Stage 2 work.

† *PHI-specific* means the entity alone establishes health-context linkage: an MRN has no
non-medical reading, so its presence makes a document PHI outright. All other types
become PHI only when medical context co-occurs (§9).

Still deferred, with reasons:

- **Bare dates / ages > 89** (#3) — quasi-identifiers with enormous false-positive
  surface; handled at M3 with co-occurrence logic. v1 only takes dates with an explicit
  birth cue.
- **PASSPORT** (#11) — per-country format matrices; mechanical but voluminous. The
  driver's-license detector covers only the cue-anchored case; per-state format
  validation is likewise deferred.
- **Cue-free ACCOUNT_NUMBER / HEALTH_PLAN_ID / MRN** — formatless without issuer
  context; contextual NER at M2. v1 ships the cue-anchored versions.
- **PHOTO** (#17) — requires a non-text pipeline (out of scope until the ingestion
  phases). `DEVICE_ID`, `VEHICLE_ID` and `BIOMETRIC_ID` shipped at v0.3 as model
  types; a VIN check-digit validator remains a worthwhile Stage 1 add that would
  let the rules tier confirm what the model proposes.
- **IPv4-mapped IPv6 textual form** (`::ffff:192.0.2.1`) — rare in documents; the v6
  candidate pattern excludes dotted tails for simplicity.

## 7. Stage 1 design: rules + validators (v1, this repo)

**Principle: broad candidate regex, strict validator.** The regex over-generates; a pure
function then rejects or scores each candidate. This keeps recall in the pattern and
precision in the validator, and validators are unit-testable in isolation
(`src/pii_master/validators.py`).

**Pre-scan windows.** Running a dozen back-tracking patterns over every position of
every document measured ~7 ms for a 10 KB document — over the entire 5 ms budget on its
own (docs/BASELINE_M1.md). Each detector therefore narrows where its pattern runs using
a cheap pre-scan: literal hints found with C-speed `str.find` (a cue word, `@`, `http`),
a shared cached digit-run scan for the numeric types, or a cheap seed regex (IPv6's
colon-pair finder). The full pattern then runs only inside small windows around the
hits. This is the same candidate-window idea Stage 2 uses (§8), applied one layer down.
Windowed and full scans are asserted equivalent over the whole frozen corpus in
`tests/test_prescan_equivalence.py`.

**Boundary strategy.** All numeric patterns use explicit lookarounds `(?<!\d)` / `(?!\d)`
(and variants including `.` or `-` where relevant) instead of `\b`. Word-boundary `\b`
treats `-` and `.` as boundaries, so a 16-digit account number would otherwise yield an
embedded "SSN" match on digits 4–12. Lookarounds reject candidates embedded in longer
digit runs.

Per-type design:

- **EMAIL** — pragmatic pattern `[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}`, not
  RFC 5322 (which admits quoted-string monstrosities nobody writes). Validator rejects
  local parts starting/ending with `.`.
- **PHONE_US** — NANP shapes with optional `+1`, `(area)`, and `-. ` separators.
  Validator: area code and exchange must start with 2–9 (NANP rule); kills most
  order-number/timestamp collisions. 10 bare digits remain the known FP class → 0.85.
- **SSN** — `(\d{3})([- ])(\d{2})\2(\d{4})` with a backreferenced separator (consistent
  `-` or ` `, never mixed). Validator rejects known-never-issued ranges: area `000`,
  `666`, `900–999`; group `00`; serial `0000`. Post-2011 randomization means the area no
  longer encodes geography, so we do *not* reject on geography; the famous test SSN
  `078-05-1120` is structurally valid and is deliberately detected. Bare 9-digit runs are
  **not** matched in v1 — the FP flood outweighs recall; revisit with Stage 2 context.
- **CREDIT_CARD** — candidate: 13–19 digits with optional uniform ` ` or `-` grouping.
  Validator: consistent separator, length 13–19 after stripping, and the **Luhn mod-10
  checksum as a hard reject** (a 1-in-10 random-number survivor rate, multiplied by the
  other constraints, makes this the highest-precision detector). Known IIN prefixes
  (Visa 4, Mastercard 51–55/2221–2720, Amex 34/37, Discover 6011/65) boost confidence
  0.80 → 0.95 but never reject — IIN ranges churn.
- **IP_ADDRESS** — dotted-quad candidates validated by stdlib `ipaddress.IPv4Address`.
  Version strings like `10.2.1.4` are irreducible FPs at the rules level → 0.70; Stage 2
  context (surrounding tokens like `version`, `v`, `release`) is the real fix.
- **DATE_DOB** — date shapes (`MM/DD/YYYY`, `M-D-YYYY`, `YYYY-MM-DD`, `March 14, 1985`)
  become entities **only** when a birth cue (`DOB`, `date of birth`, `birthdate`,
  `born`) appears in the ~40 characters before the match. Calendar validation via
  `datetime.date`; years bounded to [1900, current]. Bare dates: see §6 deferral.
- **MRN** — cue-anchored only: `MRN`, `Medical Record No/Number/#`, or `Chart` **carrying
  a number marker** (`#`, `no.`, `number`) followed by a 5–12 char alphanumeric ID
  containing ≥3 digits. Bare `chart <id>` is a table-of-contents collision and is
  rejected. The entity span covers the ID, not
  the cue. Formatless identifiers get detected *by their labels* in v1; cue-free MRN
  detection is a Stage 2 objective.
- **ACCOUNT_NUMBER / HEALTH_PLAN_ID / US_DRIVER_LICENSE** — same cue-anchored pattern
  family as MRN (shared `CueAnchoredIdDetector` base): cue phrase, optional separator,
  then an ID with a minimum digit count. Because `HEALTH_PLAN_ID` is `phi_specific` —
  one hit escalates the whole document to PHI — its cues must be unambiguously health
  on their own: `health plan`, `beneficiary`, or a health qualifier before
  `subscriber`/`member`. Bare `subscriber id` (magazine, SaaS) and `member id` (gym,
  loyalty) are rejected.

**Lexical suppressors.** Two documented false-positive classes are cheaply recoverable
without a model, so the validators reject them: a dotted quad behind a
`build`/`release`/`version`/`v` token is a version string, not an IPv4 address; and a
**bare** 10-digit run behind `confirmation`/`order`/`tracking`/`invoice`/`ref` is a
reference number, not a phone. Formatted phone numbers keep their shape and are never
suppressed. Residual ambiguity in both classes stays Stage 2's problem.
- **URL** — `http(s)://` or `www.` candidates; the pattern's final character class
  excludes closing punctuation so a sentence-ending period or bracket is not swallowed.
- **IPv6** — hex-and-colon candidate runs (≥2 colons) validated by stdlib
  `ipaddress.IPv6Address`; bare `::` (no hex digit) is rejected, and leading lookarounds
  keep `std::vector`-style code tokens out.

**Confidence model.** v1 confidences are **ordinal detector certainty, not calibrated
probabilities**: hand-set base values per detector, adjusted by validators (Luhn+IIN
boost, spaced-SSN penalty). They exist so overlap resolution and risk scoring have a
consistent preference order. Real calibration (isotonic/Platt on a labeled dev set)
arrives with Stage 2, where scores come from a model that can be calibrated.

**Overlap resolution** (`pipeline.py`): collect candidates from all detectors, sort by
`(-confidence, -length, start, detector_name)`, greedily accept spans that don't overlap
an accepted span, return sorted by position. Deterministic, O(n log n), explainable.
Documented replacement point: M2 may want nested spans of *different* types (a phone
number inside an email's quoted display name); only this function changes.

## 8. Stage 2 design: learned NER under the CPU budget (M2)

> **Built at v0.3.** This section is the design as written; what was actually
> trained, measured and shipped is in docs/DISTILLATION_PLAN.md (the arithmetic
> that chose the architecture), docs/DISTILLATION_RESULTS.md (the training run
> and its five acceptance gates) and docs/STAGE2_INTEGRATION.md (the serving
> change: `ner.OnnxNerDetector`, the fusion policy, and the latency the shipped
> cascade actually has). Three predictions in this section did not survive
> contact with measurement and are worth flagging where you read them:
> **int8 quantization made the model an order of magnitude slower**, not faster,
> because it is normalisation-bound rather than matmul-bound; the winning
> student was a **dilated CNN, not a micro-transformer**, and the margin was not
> close; and **candidate windows turned out not to be necessary** — the student
> reads whole documents inside the budget, so the cascade in the box below is
> simpler than the one designed here.

Rules cannot find names, addresses, or cue-free MRNs, and cannot disambiguate
context-dependent types. Stage 2 adds a learned token classifier — built under the
constraint that **the GPU is for training; the container is 1 CPU core and the whole
pipeline has 5 ms per document.**

That ceiling changes the model class. A BERT-class encoder — even distilled — costs tens
of milliseconds per full document on one core, so the design is a **cascade: a tiny
model, run selectively**:

1. **Candidate windows, not full documents.** Stage 1 output plus cheap lexical triggers
   (capitalized token runs, digit-dense regions, cue words like "name:", street
   suffixes) nominate short windows (~tens of tokens). The model scores only those
   windows; documents and regions with no triggers never touch the model. This bounds
   model cost by trigger density, not document length.
2. **Tiny student models.** Candidate architectures, in order of preference under the
   budget: (a) a char/word CNN or small BiLSTM tagger (~1–5 M params — the historically
   strong architecture class for clinical de-identification), (b) a 2–4-layer
   distilled micro-transformer with a short window. Either is trained by
   **distillation on GPU**: a large teacher (fine-tuned DeBERTa-class or an
   LLM-labeled corpus) produces soft labels; the student learns them. **The choice is
   settled by measured 1-core latency vs span-F1 on the harness — not assumed in
   advance.**
2. **Data.**
   - *Primary candidate:* **`nvidia/Nemotron-PII`** (CC BY 4.0, commercially usable) —
     200k synthetic persona-grounded documents, span-annotated, covering **both PII and
     PHI**, structured and unstructured, US and international locales. Its complete
     55-label inventory and the crosswalk to our taxonomy are in
     **docs/NEMOTRON_PII_TAGS.md**; 12 of its labels map onto our 11 entity types
     (23.6% of spans), and the remaining 43 labels are the measured shape of what
     Stage 2 must add.
   - *Additional PII:* the `ai4privacy/pii-masking` dataset family on Hugging Face, plus
     synthetic generators (Gretel-style or Faker-templated) for format coverage.
   - *PHI benchmark:* the **n2c2/i2b2 2014 de-identification corpus** (real clinical
     notes with gold PHI spans) — the standard benchmark. **Access requires a data use
     agreement with Harvard DBMI; it cannot be redistributed in or evaluated by this
     public repo.** Nemotron-PII and synthetic clinical notes fill the gap for open CI.
   - Alignment: our taxonomy → dataset label crosswalks live next to the training code.
     The label-space policy must be explicit: the ~43 unmodelled Nemotron labels either
     collapse to `O` or get adopted, and the special-category attributes among them
     (`race_ethnicity`, `religious_belief`, `political_view`, `sexuality`) belong to a
     GDPR profile (M3), not the HIPAA one.
3. **Export & quantization.** Export to ONNX; apply dynamic int8 quantization
   (onnxruntime). Expected artifact for a tiny student: single-digit-to-tens of MB.
   Serving: `onnxruntime` CPU EP, `intra_op_num_threads=1`, candidate windows batched
   into one session run per document where possible, span de-duplication across
   overlapping windows.
4. **Fusion with rules.** The ONNX detector is just another `Detector`. Fusion policy in
   Stage 3: checksum-validated rule spans (CREDIT_CARD, SSN) outrank model spans on
   overlap; model spans add the types rules can't see; checksum validators re-verify any
   model span of a checksummed type before it can carry high confidence.
5. **Release gate.** A model ships only if the end-to-end pipeline stays within the
   **5 ms/doc p95** budget on the 1-core benchmark harness *and* span-level F1 beats the
   rules-only baseline on the frozen test set. If the tiny student cannot beat the rules
   baseline, the rules ship alone — the cascade makes the model additive, never
   load-bearing.

## 9. Stage 3 design: aggregation, document labels, risk scoring

**Label decision** (v1, implemented in `classify.py`):

1. No entities → `NONE`.
2. Any entity whose taxonomy row is `phi_specific` (currently `MRN` and
   `HEALTH_PLAN_ID`) → `PHI`. The rule reads `entities.TAXONOMY`, so adding a
   PHI-specific type needs no change in `classify.py`.
3. Otherwise, ≥1 entity **and** medical context present → `PHI`. The medical-context
   test is a ~27-term **word-bounded** scan (`patient`, `diagnosis`, `discharge`,
   `icd-10`, `prescription`, …) — a deliberately cheap stand-in, replaced by real
   context modeling at M2/M3. Two rules govern the term list, both learned the hard
   way (docs/IMPROVEMENT_PLAN.md §3.1): terms match on word boundaries, never as
   substrings (`impatient` used to match `patient`), and a term must be
   unambiguously health-flavored alone (bare `provider` was dropped for
   `healthcare provider`, since "cloud provider" is the common reading). Residual
   weakness, accepted for recall: `treatment`, `discharge`, `admission` and adjectival
   `patient` still have non-clinical readings.
4. Otherwise → `PII`.

Every rule that fires appends a human-readable line to `report.reasons` — the report
explains itself.

**Risk score** (0–100, explainable, defined in one docstring):

```
score = clamp( Σ_over_types  min(count_t, 3) × weight_t × mean_confidence_t
               + 15 if label == PHI else 0,
               0, 100 )
```

Per-type counts cap at 3 so a CSV with 500 emails doesn't outscore a document with one
SSN; weights (5–30) live in `entities.TAXONOMY`; the PHI bonus reflects regulatory
exposure. All hand-set v1 heuristics — the structure (additive, capped, per-type
attributable) is the durable part.

**Planned evolution (M3):** quasi-identifier co-occurrence scoring (name + DOB +
diagnosis ≫ each alone), per-section/page attribution for long documents, and
configurable **policy profiles** (HIPAA Safe Harbor vs GDPR personal-data categories vs
CCPA) mapping the same entity evidence to regime-specific labels.

## 10. Evaluation methodology

- **Span level:** precision/recall/**F1 and F2** **per entity type** plus a pooled
  micro row, entity-level matching in the seqeval style — exact-boundary and
  partial-credit (overlap) variants reported separately, because boundary
  sloppiness and type confusion are different bugs.
- **Why both F-scores.** §1 ranks recall first for PHI: a missed medical record
  number is a reportable incident, a false alarm costs a reviewer minutes. F1
  prices those two errors identically, so it is the wrong single headline for
  this system. **F2 weights recall four times as heavily**, which is closer to
  the real cost matrix. Both are reported and neither is *the* number, because
  §1 is equally clear that precision is not optional — a scanner that cries
  wolf gets turned off, which is the worst recall of all. The gap between the
  two columns is exactly the tradeoff a confidence threshold is choosing, and
  showing one column hides the choice. Micro rather than macro throughout, so a
  43-span type cannot swing the headline.
- **Document level:** confusion matrix over NONE/PII/PHI; the headline operating metric
  is **PHI recall** at a stated precision floor. Note this is *already* a
  recall-first metric, and it saturates at 1.00 in every shipped configuration —
  so span-level F2 is where the recall tradeoff is actually visible, and where
  it should be argued.
- **Error taxonomy:** every FN/FP triaged as boundary error / type confusion / context
  miss / validator over-reject — each maps to a different fix.
- **Test assets:** the **frozen hard-case corpus** lives in `eval/corpus/` (gold spans
  authored as inline `[[TYPE:...]]` markup so offsets can never drift) and grows
  monotonically — cases are added, never removed, so scores are comparable across
  versions. It deliberately contains gold entities the current system cannot detect
  (PERSON_NAME, ADDRESS) so Stage 2's job shows up as measured recall 0 today, and
  known-FP hard negatives (version strings, bare 10-digit numbers) so precision costs
  are quantified, not anecdotal. `pii-master eval` scores it. Public-dataset evals
  (ai4privacy; n2c2 where the DUA permits) run alongside but never replace the frozen
  set.
- **Targets** (goals, not achieved results): >0.95 recall at >0.90 precision on
  checksum/format types (SSN, CREDIT_CARD, EMAIL); >0.90 recall on MRN-with-cue;
  document-level PHI recall >0.95 on the frozen corpus.

## 11. Roadmap & milestones

- **M0 — Scaffold + Stage 1 (this commit).** Package, taxonomy, 7 rule detectors,
  pipeline, classifier, CLI, test suite. *Exit: `pytest` green; CLI classifies a PHI
  sample correctly end-to-end.*
- **M1 — Measure before learning** *(delivered in v0.2)*. Benchmark harness
  (`pii-master bench`: per-doc latency percentiles vs the 5 ms budget, throughput, RSS,
  on 1 core); frozen hard-case corpus (`eval/corpus/`, append-only) + span/document
  scoring (`pii-master eval`); Stage 1 hardening (IPv6, URL, cue-anchored driver's
  licenses and account/plan IDs, more date formats). *Exit: baseline report
  (rules-only P/R/F1 per type + measured latency) committed — docs/BASELINE_M1.md.*
- **M1.5 — Harden Stage 1 and make evaluation a gate** *(planned; see
  docs/IMPROVEMENT_PLAN.md)*. Close known false-PHI leaks (loose MRN/plan cues,
  substring medical-context), add CI + an eval `--fail-under` snapshot, implement
  the error taxonomy in §10, and take the remaining format-anchored regex wins
  (fax, ABA routing, MAC, SWIFT, VIN). *Exit: no known reproducible false PHI;
  pytest + eval + bench run on every push; 10 KB p95 still ≤ ~2 ms.*
- **M2 — Learned NER under the 5 ms cascade** *(delivered in v0.3)*. GPU
  distillation pipeline; fp32 ONNX export (int8 measured *slower*, see §8);
  `ner.OnnxNerDetector` implementing the `Detector` protocol; a named fusion
  policy in `pipeline.py`; three guards on model output (confidence floor,
  checksum re-validation, tier precedence); 14 new HIPAA-mapped entity types;
  `--deep` on `scan`, `eval` and `bench`. *Exit met: the fused cascade beats the
  rules-only baseline on the external Nemotron holdout and the frozen corpus
  holds, within budget on one core — docs/STAGE2_INTEGRATION.md.* Calibration
  is the one piece deferred: confidences remain ordinal (§7), and the
  confidence floor is a threshold on an uncalibrated score.
- **M3 — Risk & policy.** Co-occurrence scoring, policy profiles (HIPAA/GDPR), config
  system, redaction-ready span output. *Exit: same document classifiable under two
  regimes with distinct, explained outcomes.*
- **M4 — PDF/DOCX ingestion.** Text-layer extraction with offset mapping back to source
  (page/bbox) so evidence remains locatable. *Exit: PDF in, span-attributed report out.*
- **M5 — OCR path.** Scanned-document support; OCR-noise-tolerant validators (confusable
  characters). *Exit: scanned PHI sample correctly labeled.*
- **M6 — Feedback loop.** Reviewer corrections captured as labeled data; active-learning
  selection; periodic retrain. *Exit: correction→retrain path exercised once end-to-end.*
- **M7 — Beyond US/English.** Locale packs (national ID formats + per-locale NER),
  starting with the EU. *Exit: one non-US locale at parity on its frozen corpus.*

## 12. Non-goals (v1) & open questions

**v1 non-goals:** redaction/masking output (report only); PDF/DOCX/OCR (M4/M5); non-US
identifier formats (M7); calibrated probabilities (M2); streaming/chunked processing of
multi-GB files; any network calls at inference time (permanent non-goal).

**Open questions:**
- Nested spans of different types — allow, or keep flat spans and let the report
  carry alternates? Still open after M2: fusion resolves overlaps by tier and
  drops the loser, so a model `NATIONAL_ID` proposal over a rule-detected `SSN`
  is discarded rather than recorded as an alternate reading.
- Should a fast "doc-label-only" mode exist that skips span extraction when the first
  PHI-specific hit short-circuits? (Speed vs evidence completeness.)
- ~~Where does the confidence threshold live?~~ **Answered at v0.3:** in the
  library, as an opinionated default (`min_confidence=0.5`) that policy config
  can override. It had to ship somewhere — an unthresholded student emits a
  false PHI on the frozen corpus — and a default a caller can move is a better
  answer than a constant nobody can reach. It is a threshold on an *ordinal*
  score until calibration lands, which is the honest caveat.

## 13. References

- HIPAA Safe Harbor de-identification: 45 CFR §164.514(b); HHS guidance:
  https://www.hhs.gov/hipaa/for-professionals/special-topics/de-identification/index.html
- n2c2 (formerly i2b2) de-identification corpora, Harvard DBMI data portal (DUA
  required): https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/
- Prior-art survey (what to adopt/avoid, measured CPU latencies): docs/PRIOR_ART.md
- Stage 2 distillation plan (teacher, student ladder, runbook): docs/DISTILLATION_PLAN.md
- Nemotron-PII dataset (CC BY 4.0): https://huggingface.co/datasets/nvidia/Nemotron-PII
  — full tag inventory and crosswalk: docs/NEMOTRON_PII_TAGS.md
- ai4privacy PII masking datasets: https://huggingface.co/ai4privacy
- seqeval (entity-level sequence evaluation): https://github.com/chakki-works/seqeval
- ONNX Runtime quantization: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- Luhn algorithm: ISO/IEC 7812-1 (check digit for identification card numbers)
