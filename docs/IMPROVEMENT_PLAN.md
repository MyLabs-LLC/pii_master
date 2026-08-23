# Improvement plan — post-M1 review

**Status:** review of v0.2 (M1), 2026-08-22.
**Companion docs:** [DESIGN.md](DESIGN.md) (architecture + original roadmap),
[BASELINE_M1.md](BASELINE_M1.md) (measured P/R/F1 + latency),
[NEMOTRON_PII_TAGS.md](NEMOTRON_PII_TAGS.md) (Stage 2 data crosswalk).

This is not a rewrite of the roadmap. The staged architecture, 5 ms / 1-core
budget, recall-first PHI policy, and Entity contract are the right load-bearing
decisions and should stay. This plan is what to do *next*, in what order, and
what not to start yet — based on reading the code, the frozen corpus, and a
handful of probes against the live pipeline.

---

## 1. Review summary

M1 did what it claimed: a stdlib-only rules engine, pre-scan windows that
brought 10 KB p95 from 7.9 ms to 1.2 ms, an append-only gold corpus, and
reproducible `eval` / `bench` harnesses. The design writing is unusually
honest about what rules cannot do.

The gap is that the **quality story is still tautological** and a few
Stage 1 decisions leak false PHI. The frozen corpus (31 docs) was authored
alongside the detectors, so perfect scores on format types mean “the rules
do what they claim on their own hard cases,” which BASELINE_M1 already
says. What it does *not* yet say is that several cheap, reproducible false
positives sit just outside that corpus — and two of them escalate a
document to PHI.

Jumping to Stage 2 NER before closing those leaks will train a student on
noisy fusion labels and spend the remaining 3.7 ms of budget on problems
rules can still fix.

**Recommended sequence:** harden Stage 1 and make evaluation honest
(this plan’s Track A–C), then start Stage 2 with a frozen label-space
policy (Track D). M3–M7 stay as designed, with a few scope refinements.

---

## 2. What is already strong

Keep these; later work should compose with them, not replace them.

| Choice | Why it should stay |
|---|---|
| Shared `Entity` / `Detector` protocol | Stage 2 plugs in without touching Stage 3 |
| Broad regex + isolated validators | Precision lives in testable functions; Stage 2 can reuse them |
| Pre-scan windows + `test_prescan_equivalence` | The 6.4× speedup is behavior-preserving, not a silent recall cut |
| Cue-anchored DOB / MRN / plan / license / account | Matches Nemotron: 96.4% of `date_of_birth` spans have a cue in the 40-char window |
| Append-only frozen corpus + inline `[[TYPE:]]` markup | Offsets cannot drift; Stage 2 gaps show up as recall 0 |
| Checksum types (Luhn, SSN ranges, NANP, `ipaddress`) | Highest-precision detectors; fusion policy should keep them as overrides |
| Bench gate (`--fail-over-budget`) and scan gate (`--fail-on-detect`) | The right CI shape — they just are not wired up yet |
| Stdlib-only runtime | Fits the 1-core / 4 GB box; keep training extras in optional extras |

---

## 3. Findings

### 3.1 False PHI (highest severity)

PHI is the headline label. A false PHI is worse than a false PII: it
triggers the wrong policy and trains reviewers to ignore the scanner.

Probed against the current `scan_text` (not in the frozen corpus):

| Input | Result | Cause |
|---|---|---|
| `The chart 4829471 is in the appendix.` | `PHI` via `MRN` | `MrnDetector` cue is `chart\s*#?` — “chart” alone, no `#` / “number”, is enough |
| `subscriber id A9-3321-77 for the magazine` | `PHI` via `HEALTH_PLAN_ID` | “subscriber id” is treated as unambiguously health-flavored; it is not |
| `SSN 123-45-6789 from the cloud provider.` | `PHI` | `has_medical_context` is a raw substring scan; `provider` matches |

Related, currently harmless only because no identifier is present:

- `The impatient customer called about shipping.` → medical context **True**
  (`patient` is a substring of `impatient`).
- `The cloud provider invoice is attached.` → medical context **True**.

The medical-context list also includes the two-letter token `rx`. It did
not fire on `matrix` / `proxy` in probes, but a substring test on a
two-letter token is one vocabulary change away from a collision. Word
boundaries are the fix, not a longer deny-list.

DESIGN.md §9 and the `DocumentClassifier` docstring still say
“PHI-specific entity (v1: MRN)”. `HEALTH_PLAN_ID` is also `phi_specific`
— correct in `TAXONOMY`, stale in the comments.

### 3.2 Known false PII, still unmitigated

These are documented in BASELINE_M1 and present in the corpus
(`none-003`, `none-006`). They are the two document-level misses
(accuracy 0.94). Rules cannot *fully* kill them, but cheap lexical
suppressors recover most of the mass without a model:

- **IPv4 vs version string** — `Build 10.2.1.4` / `release 1.2.3`. A
  preceding `build`, `release`, `version`, `v`, `shipped` token is a
  high-precision reject. Residual FPs become Stage 2’s problem.
- **Bare 10-digit PHONE_US vs confirmation / order / tracking** — a
  preceding `confirmation`, `order`, `tracking`, `invoice`, `ref`
  token is the same shape of fix. Keep formatted NANP (`(415) 555-2671`)
  as-is.

Do not wait for a transformer to apply a 6-word deny-list.

### 3.3 Evaluation is not yet a product

DESIGN.md §10 describes an error taxonomy (boundary / type confusion /
context miss / validator over-reject) and external-dataset evals. Neither
is implemented.

- `evaluation.py` reports exact/partial P/R/F1 and a 3×3 confusion
  matrix. It does not triage FN/FP, so a regression is a number, not a
  work item.
- There is no score snapshot / `--fail-under` gate. `bench` can fail CI;
  `eval` cannot. A detector change can drop PHI recall silently.
- There is **no CI workflow**. `.github/` does not exist. The README
  presents `pytest`, `eval`, and `bench --fail-over-budget` as gates;
  none of them run on push.
- The frozen set is 31 short, English, US-format sentences. It is the
  right *unit* corpus. It is the wrong *only* corpus. Nemotron-PII is
  already surveyed and licensed (CC BY 4.0); there is no harness that
  scores our 11 types against a Nemotron holdout.

Until an external slice exists, treat frozen-corpus F1 as a smoke test,
not a quality claim.

### 3.4 Stage 1 coverage still on the table

NEMOTRON_PII_TAGS.md §“Findings that change our plans” already names
format-anchored types that match the existing detector shape
(regex + checksum/range). None are implemented:

| Type | HIPAA | Why it is a Stage 1 add, not M2 |
|---|---|---|
| `BANK_ROUTING` (ABA) | #10 | 9 digits + ABA checksum |
| `VEHICLE_ID` (VIN) | #12 | 17-char + VIN check digit |
| `MAC_ADDRESS` | #18 | canonical hex-colon/hex-dash |
| `SWIFT_BIC` | #10 | 8/11-char ISO 9362 |
| `DATE_TIME` | #3 | ISO-8601 / `YYYY-MM-DDTHH:MM` |
| `GEO_COORDINATE` | #2 | lat/long with hemisphere or signed decimal |
| `FAX_NUMBER` | #5 | same shape as phone; HIPAA lists it separately |
| `US_ZIP` | #2 | 5 / 9 digit; Safe Harbor keeps only first 3 under conditions |

Also noted there, still open:

- `certificate_license_number` → `US_DRIVER_LICENSE` is a **narrowing**.
  A `LICENSE_NUMBER` rename (or a parent type) matches HIPAA #11.
- `phone_number` includes international; `PHONE_US` rejects them.
- Cue-free account / plan / MRN spans will stay as FN until Stage 2 —
  do not try to regex them.

`entities.py`’s module docstring still lists `US_DRIVER_LICENSE`,
`HEALTH_PLAN_ID`, IPv6, and URL as deferred. They shipped in v0.2.

### 3.5 API and runtime nits (small, cheap)

- `scan_text` constructs a new `Pipeline()` (and 12 detectors, 12
  compiled patterns) on every call. `eval`, `bench`, and the CLI all go
  through it. A module-level default pipeline, or `scan_text(text,
  pipeline=...)`, removes avoidable setup before Stage 2 adds ONNX
  session cost.
- `_digit_run_windows` is `@lru_cache(maxsize=8)` keyed on the **full
  document string**. Correct for the current single-doc path; in a
  batch worker it retains up to eight document bodies. Cache windows by
  `id(text)` plus length, or pass a per-scan scratch object.
- Overlap resolution is O(n × accepted) via `any(candidate.overlaps)`.
  Fine at current span counts; a sweep-line becomes the M2 replacement
  point once the NER detector adds names on every capitalized run.
- CLI `scan` is file-or-stdin only: no recursive directory walk, no
  `--types` filter, no `--redact` (redaction is a v1 non-goal and can
  stay one).
- No `py.typed`, no ruff/mypy, no package URLs/classifiers in
  `pyproject.toml`.
- `test_prescan_equivalence` calls `sample_texts()` at collection time
  *and* inside the test (the `range(len(sample_texts()))` default).
  Harmless, slightly wasteful; parametrize from a fixture once.

### 3.6 Stage 2 is designed, not sequenced

DESIGN.md §8 is the right model class (tiny student, candidate windows,
int8 ONNX, fusion with checksum overrides, ship-only-if-it-beats-rules
*and* stays ≤ 5 ms). Three decisions are still open and will block
training if left implicit:

1. **Label-space policy** for the 43 Nemotron labels we do not model.
   Adopting HIPAA-mapped ones and collapsing GDPR special-category
   attributes (`race_ethnicity`, `religious_belief`, `political_view`,
   `sexuality`) to `O` for the HIPAA profile is the coherent default —
   write it down as code, not a paragraph.
2. **Window nominator.** Stage 1 already has the machinery (hints,
   digit-runs, cue words). Stage 2 windows should be the same function
   plus capitalized-token runs and street-suffix triggers — one
   implementation, two consumers.
3. **Public CI yardstick.** n2c2/i2b2 2014 is the clinical gold
   standard and cannot live in this repo. A Nemotron-PII *mapped*
   holdout (the 12 labels that crosswalk today, plus whatever Track A
   adds) is the thing CI can actually run.

---

## 4. Recommended tracks

Tracks A–C are the M1.5 work. Track D is M2 with a tighter entry
criterion. Tracks E–F refine M3–M7; they do not move earlier.

### Track A — Stop lying about PHI

**Goal:** no known, reproducible false-PHI from rules. Frozen-corpus
PHI recall stays 1.00.

1. **Word-bounded medical context.** Replace `term in lower` with a
   compiled `\b` alternation (or a lowercase-split membership test for
   single tokens and a phrase scan for multi-word terms). Drop `rx` or
   require it as a standalone token next to a digit/dose cue. Add
   probes for `impatient`, `cloud provider`, `proxy`, `matrix` to
   `tests/test_classify.py`.
2. **Tighten `MrnDetector`.** Require `chart` to be `chart #` /
   `chart no/number`. Bare `chart <alnum>` is a table-of-contents
   collision. Append `The chart 4829471 is in the appendix.` to the
   frozen corpus as a NONE (or as a non-MRN PII if some other type
   fires).
3. **Tighten `HealthPlanIdDetector`.** Drop standalone `subscriber` or
   require a health-flavored neighbor (`health plan`, `insurance`,
   `medicare`, `medicaid`, `beneficiary`). Magazine/SaaS “subscriber
   id” must not be PHI. Append a NONE case.
4. **Lexical suppressors for the two documented PII FPs.**
   `IpAddressDetector.validate`: reject when a version/build/release
   cue sits in a short left window. `UsPhoneDetector.validate`: reject
   *bare* 10-digit runs with an order/confirmation/tracking/invoice
   cue. Formatted phones stay. These are validator changes, so
   `test_prescan_equivalence` still holds.
5. **Stale docs/comments.** `entities.py` deferred list,
   `DocumentClassifier` “v1: MRN” wording, DESIGN.md §9 if it still
   names only MRN.

Exit: new corpus rows stay append-only; document accuracy on the
frozen set does not go down; the three probes above are `NONE` or
non-PHI as intended; `pytest` green.

### Track B — Make evaluation a gate

**Goal:** a detector change that drops quality or blows the budget
fails CI, with an error class attached.

1. **GitHub Actions** on push/PR: `pip install -e ".[dev]"` → `pytest`
   → `pii-master eval eval/corpus/frozen_v1.jsonl --json` →
   `pii-master bench --fail-over-budget`. Pin Python 3.11 to match
   BASELINE_M1.
2. **Eval regression file.** Commit
   `eval/corpus/frozen_v1.scores.json` (per-type P/R/F1 + doc accuracy
   + PHI recall). `pii-master eval --fail-under
   eval/corpus/frozen_v1.scores.json` exits 1 on any drop. Raising a
   score is a deliberate scores-file edit, same as the append-only
   corpus rule.
3. **Error taxonomy in `evaluation.py`.** For each FN/FP, classify
   `boundary` / `type_confusion` / `context_miss` /
   `validator_over_reject` / `undetectable` (gold type in
   `FUTURE_TYPES`). Render a one-line histogram. This is the
   DESIGN.md §10 item that turns a failed gate into a ticket.
4. **Nemotron mapped eval (optional extra, analysis-only deps).** A
   script in `eval/scripts/` that (a) downloads or accepts local
   parquet, (b) applies the crosswalk table already in
   NEMOTRON_PII_TAGS.md, (c) scores only the mapped types, (d) writes
   `docs/BASELINE_NEMOTRON.md`. Do **not** vendor the 300 MB parquet.
   This is the first non-tautological quality number.

Exit: CI green on main; scores file exists; `eval --fail-under`
tested by a deliberate local downgrade.

### Track C — Cheap Stage 1 coverage

**Goal:** pick up format-anchored HIPAA identifiers without spending
Stage 2 budget. Each new type follows the existing pattern: candidate
regex, validator, table-driven tests, one frozen-corpus row, taxonomy
entry, bench still under 2 ms p95 at 10 KB.

Priority order (impact × implementation cost, from the Nemotron gap
table):

1. `FAX_NUMBER` — HIPAA #5 is a missing row, not a new shape. Detect
   as phone-shaped + fax cue (`fax`, `facsimile`, `telefax`). Without
   the cue, keep emitting `PHONE_US`.
2. `BANK_ROUTING` — ABA checksum is the Luhn of this family.
3. `MAC_ADDRESS` — regex + octet range; pre-scan on `:` / `-` hex
   pairs, same idea as IPv6’s colon-pair seed.
4. `SWIFT_BIC` — 8/11-char charset; reject if it fails the structure.
5. `VEHICLE_ID` — VIN check digit; 17-char window.
6. `US_ZIP` — 5 / 5+4; do **not** treat a bare 5-digit run as PII
   (the ZIP/order-number collision is the phone problem again). Cue-
   or address-adjacent only until Stage 2.
7. `DATE_TIME` — ISO timestamps. Distinct from `DATE_DOB`. HIPAA #3
   only becomes PHI with medical context, same as today.
8. `GEO_COORDINATE` — lower priority; useful, rarer in the documents
   this scanner will see first.

Also in this track, not a new detector:

- Decide `US_DRIVER_LICENSE` vs `LICENSE_NUMBER`. Recommendation:
  keep the current type, add `LICENSE_NUMBER` as the HIPAA #11
  umbrella, and treat driver’s-license cues as a subtype. Avoid a
  breaking rename of a shipped enum value.
- IPv4-mapped IPv6 (`::ffff:192.0.2.1`) if it stays cheap; it is
  called out as a v1 hole in DESIGN.md §6.

Exit: each type has validator unit tests, at least one frozen TP and
one frozen FP, taxonomy + HIPAA row, 10 KB p95 still ≤ ~2 ms.

### Track D — Stage 2, with an entry ticket

> **Closed at v0.3.** The entry ticket was met, the student was trained
> (docs/DISTILLATION_RESULTS.md) and integrated (docs/STAGE2_INTEGRATION.md).
> Items 1–4 below shipped. Item 5, **calibration, did not** — confidences are
> still ordinal, and `min_confidence` is a threshold on an uncalibrated score
> chosen by sweeping it on the holdout rather than by fitting isotonic
> regression. That is the honest state, and it is the first thing to fix.
> Track C (format-anchored Stage 1 types) is *not* closed by this: several of
> its types now have model coverage, which lowers their urgency but does not
> give them a checksum. A `BANK_ROUTING` the model proposes cannot be verified
> the way a Luhn-valid card can, so it never earns the top fusion tier.

Do not start GPU distillation until Tracks A and B have landed and
the label-space policy is a module, not a markdown table.

**Entry ticket**

- [ ] Track A false-PHI probes are tests.
- [ ] CI + `--fail-under` on the frozen set.
- [ ] `src/pii_master/crosswalk.py` (or `training/label_map.py`):
      Nemotron label → `EntityType | None`. `None` means collapse to
      `O`. HIPAA-mapped labels adopted as we add them in Track C;
      GDPR special-category labels stay `None` until M3.
- [ ] Written decision on window nomination (reuse Stage 1 pre-scan
      + capitalized runs + street suffixes).

**Then, in this order**

1. **Data prep, no model.** Script that maps Nemotron spans through
   the crosswalk, writes a token-classification dataset (BIO or
   span JSONL), and reports label distribution *after* collapse.
   Refuse to train if a mapped type has fewer than a stated floor of
   spans.
2. **Teacher only if needed.** Nemotron is already span-annotated.
   A DeBERTa/LLM teacher is for *our* leftover types or for n2c2
   once a DUA exists — not a prerequisite for a first student.
3. **Student bake-off on the harness, not on vibes.** Char/word CNN
   or small BiLSTM vs a 2–4 layer micro-transformer, both int8 ONNX,
   `intra_op_num_threads=1`, windows only. Winner is the one that
   (a) raises frozen-corpus recall on `PERSON_NAME` / `ADDRESS` and
   cue-free MRN/account/plan, (b) does not drop checksum-type
   precision (fusion policy), (c) keeps e2e p95 ≤ 5 ms at 10 KB.
4. **Fusion.** Checksum-validated rule spans outrank model spans.
   Model spans of SSN/card/ABA/VIN are re-validated before they may
   carry high confidence. This is already in DESIGN.md §8; implement
   it as a named policy in `pipeline.py`, not a comment.
5. **Calibration** on a held-out mapped slice (isotonic or Platt).
   Until then, keep calling scores ordinal.

If the student cannot beat rules-only on the frozen set inside the
budget, **do not ship it**. The cascade is additive.

### Track E — Risk, policy, product (M3, refined)

After Stage 2 spans exist (or in parallel *after* Track A, for the
rules-only path):

- Quasi-identifier co-occurrence: `{DATE_DOB, US_ZIP, PERSON_NAME}`
  and `{DATE_DOB, PHONE_US, medical context}` should move risk more
  than the sum of caps. Keep the score additive-and-attributable.
- Policy profiles: `hipaa_safe_harbor` (default), `gdpr_personal`,
  `gdpr_special` (the four Nemotron attributes), `secrets` (password /
  api_key / cvv / pin / cookie — out of PII/PHI but exactly what
  `--fail-on-detect` CI users want). Same entities, different labels.
- Confidence threshold as policy config, not a library constant
  (answers DESIGN.md §12).
- Redaction-ready output: `scan --redact` emitting a copy with
  `entities` spans replaced by `[TYPE]`. Still no PDF. This is the
  smallest product increment that makes the CLI useful beyond a
  classifier.

### Track F — Ingestion and locales (M4–M7, unchanged intent)

Do not start M4 until the text path has an external-dataset number
(Track B.4) and a redaction story (Track E). Offset mapping through
PDF text layers is wasted if the types and labels are still moving.

- M4 PDF/DOCX: text-layer extraction + page/bbox. Structured tables
  are how the 3.6% cue-less DOBs in Nemotron hide (column header
  “DOB”, value two cells down) — that is the real M4 payoff, not
  pretty page numbers.
- M5 OCR: confusable-character validators (`O`/`0`, `l`/`1`) on
  checksum types first; do not OCR-train a new NER as the first step.
- M6 feedback: reviewer corrections as JSONL in the same markup as
  the frozen corpus, so they append to eval automatically.
- M7 locales: start with the Nemotron `intl` slice and `national_id`,
  not a blank-page EU pack.

---

## 5. What not to do yet

- **Do not train a Stage 2 model this cycle.** The false-PHI leaks and
  the missing eval gate are cheaper and will poison fusion labels if
  left open.
- **Do not add an LLM to the serving path.** Permanent non-goal
  (DESIGN.md §5).
- **Do not regex names or street addresses.** BASELINE_M1 recall 0.00
  on those rows is a feature.
- **Do not match bare 9-digit SSNs or cue-free MRNs.** The v1 FP
  argument still holds.
- **Do not vendor n2c2** or any DUA-restricted text. A private eval
  harness can live out of tree.
- **Do not collapse the frozen corpus** to raise scores. Append the
  new NONE/PHI-leak cases; let the tables get harder.
- **Do not spend the 3.7 ms Stage 2 budget on types Track C can take.**
  Every format-anchored detector left for the model is a window the
  model should have spent on `PERSON_NAME`.

---

## 6. Suggested sequencing

```
Track A  false-PHI + lexical suppressors     ─┐
Track B  CI + eval gate + error taxonomy     ─┼─ M1.5, rules-only
Track C  format-anchored types (C.1–C.5 first)┘
                    │
                    ▼
Track D  crosswalk module → mapped dataset → student bake-off
                    │
                    ▼
Track E  policy profiles + redaction
Track F  PDF / OCR / locales, in that order
```

A and B can proceed in parallel. C can start as soon as A’s detector
tweaks land (so new types are not built on loose cues). D’s entry
ticket is A+B plus the crosswalk module.

---

## 7. Exit criteria for “M1.5 done”

The original M2 exit (“model beats rules-only AND p95 ≤ 5 ms”) stays.
M1.5 is done when all of the following are true:

- The three false-PHI probes in §3.1 do not produce `PHI`.
- `impatient` / `cloud provider` do not set medical context.
- `.github/workflows/ci.yml` runs pytest, eval `--fail-under`, and
  bench `--fail-over-budget` on every push.
- Frozen corpus has grown (not shrunk) with the new hard cases;
  BASELINE_M1 regenerated, not edited by hand.
- Error-taxonomy histogram is in the `eval` text report.
- At least `FAX_NUMBER` and `BANK_ROUTING` are implemented, tested,
  and on the taxonomy.
- 10 KB p95 remains ≤ ~2 ms on the harness (Stage 2 still has room).
- A Nemotron-mapped baseline exists as a committed report, even if
  the first numbers are ugly.

That is the improvement plan: make the rules honest, make the numbers
ungameable, take the remaining regex wins, then — and only then —
spend the rest of the 5 ms on a tiny model for names, addresses, and
cue-free identifiers.
