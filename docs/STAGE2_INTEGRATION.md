# Stage 2 integration — the student in the shipped pipeline

**Status:** shipped at v0.3. [DISTILLATION_RESULTS.md](DISTILLATION_RESULTS.md)
ends with a student that passed five acceptance gates and lived entirely in
`training/`. This is the change that put it in `src/pii_master/`, and the
measurements that describe what shipped rather than what was trained.

The results doc's section 6 named three things the integration must carry. Two
are guards — a confidence threshold and checksum re-validation — and section 4
below shows they are load-bearing rather than ceremonial. The third was "add
these documents to the frozen corpus as regression rows"; they were already in
it (`none-007`, `none-011`, `phi-001` — the corpus is append-only and those
cases predate the student). What was missing was a *deep-mode* assertion over
them, since `pii-master eval` only ever ran the rules. That is now
`pii-master eval --deep`, plus bundle-gated tests in `tests/test_ner.py` that
assert those documents do not produce PHI with the model in the pipeline.

---

## 1. What shipped

| piece | where |
|---|---|
| `OnnxNerDetector` — the student behind the existing `Detector` protocol | `src/pii_master/ner.py` |
| Vectorised BIO span decoder | `ner.decode_spans` |
| Word-edge snapping for span boundaries | `ner.snap_to_word_edges` |
| Confidence calibration (isotonic, fitted per bundle) | `ner.calibrate`, `training/calibrate.py` |
| An `l` rung on the size ladder | `training/model.py` |
| Fusion precedence, named rather than commented | `pipeline.fusion_rank` |
| 14 new HIPAA-mapped entity types | `entities.EntityType`, `entities.MODEL_ONLY_TYPES` |
| 22 more Nemotron labels adopted | `crosswalk.MODEL_MAPPED` |
| `--deep` on `scan`, `eval` and `bench` | `cli.py`, `bench.py` |
| Serving bundle export + a torch-vs-ONNX equivalence check | `training/export.py --bundle` |
| The shipped cascade scored on the external holdout | `eval/scripts/nemotron_deep_eval.py` |

Serving stays two modes, and the default did not move:

| mode | tiers | dependencies | budget |
|---|---|---|--:|
| `fast` (default) | rules | none (stdlib) | 5 ms / 10 KB |
| `deep` (`--deep`) | rules + student | `pii-master[ml]` | 25 ms / 10 KB |

Two students clear every gate and ship as a ladder: **`l`** (recommended,
14.80 ms p95, model-tier F1 0.926) and **`m`** (7.99 ms, 0.904). A third,
`xs`, fits the strict 5 ms contract and is **not** shipped because it fails the
PHI-recall gate — §7.5.

`onnxruntime` and `tokenizers` are an optional extra, and CI has a job that
installs the package *bare* and asserts it still classifies — because a single
module-scope `import numpy` in `pii_master/` would break the zero-dependency
claim silently, and no other test would notice.

## 2. The decoder is 27x faster, and that is what fits the budget

The results doc's section 4 measured the `xs` cascade at 5.21 ms p95 and
concluded: *"the promotion rule — a student under 4 ms is promoted into `fast` —
does not fire on this measurement... 3.7 ms of that is tokenizer and decoder,
not the model."* It named two fixes. One of them, vectorising `decode_spans`,
is done:

| stage | results doc (`xs`) | shipped (`xs`) | shipped (`m`) |
|---|--:|--:|--:|
| rules (Stage 1) | 0.51 ms | 0.70 ms | 0.69 ms |
| tokenize | 2.59 ms | 2.48 ms | 2.54 ms |
| student (ONNX, 1 core) | 1.04 ms | 1.24 ms | 4.52 ms |
| **decode spans** | **1.08 ms** | **0.20 ms** | **0.20 ms** |
| **total p95** | **5.21 ms** | **4.68 ms** | **8.07 ms** |

(30 documents x 10 KB, ~1,659 tokens, `taskset -c 0`,
`intra_op_num_threads=1`, same synthetic corpus `pii-master bench` uses. The
rules and student columns differ from the results doc's simply because this is
a different machine; the decode column is the change.)

End to end through the shipped CLI, `pii-master bench [--deep]`, same core:

| mode | 10 KB p95 | 100 KB p95 | peak RSS | budget |
|---|--:|--:|--:|--:|
| `fast` (rules) | 0.70 ms | 7.80 ms | 20 MB | 5 ms — PASS |
| `deep` (`xs`, not shipped) | **4.79 ms** | 55.62 ms | 109 MB | 25 ms — PASS |
| `deep` (`m`) | **7.99 ms** | 105.22 ms | 112 MB | 25 ms — PASS |
| `deep` (`l`, recommended) | **14.80 ms** | 201.92 ms | 128 MB | 25 ms — PASS |

`deep` with the `xs` student comes in at **4.79 ms — inside the strict 5 ms
production contract**, not just the 25 ms deep budget. The promotion rule the
results doc said "does not fire on this measurement" now fires. Whether that
student should actually ship is a separate question, and the answer turned out
to be no: §7.5.

The old decoder was a Python loop over every token — ~1,700 iterations, each
doing a dict lookup and two `str.isspace()` calls. The new one does the whole
document in numpy: a whitespace bitmap over the text, a word-start test that is
an array index, span boundaries from `cumsum` over a break mask, and confidence
from `bincount`. Per-span Python work remains, but there are dozens of spans and
thousands of tokens, so the loop that mattered is gone.

**Correctness is not asserted, it is tested.** `training/decode.py` — the
readable loop that produced every number in the results doc — is *imported* by
`tests/test_ner.py` and fuzzed against the vectorised version over 400
randomly-labelled documents. Not a reimplementation of it in the test file: the
real file, so the two cannot drift.

The remaining lever is tokenization, now **53% of the `xs` cascade** and the
largest single term in it. Running the
student on candidate windows (Track D.2) would cut it, and it was left undone
deliberately: the model has a 253-token receptive field, windows truncate
context, and trading measured quality for latency the budget does not need is a
bad trade. `deep` fits its 25 ms budget with 3x headroom either way.

## 3. Fusion: the narrow reading of the policy

Precedence is **checksum-validated rule > model > cue-anchored rule**, in
`pipeline.fusion_rank`, with two narrow promotions of a cue-anchored rule span
back to the top tier: `truncates` (§7.4) and `erases_phi` (§7.6). Both were
added because of failures measured on the frozen corpus, and both are scoped so
tightly that neither costs anything on the holdout. The results doc measured the alternative (all rules
outrank the model) at 0.028 F1 worse, concentrated exactly on the types where
the rules were weakest — `US_DRIVER_LICENSE` 0.001, `ACCOUNT_NUMBER` 0.407,
`HEALTH_PLAN_ID` 0.448 — and made no difference on the checksummed types, which
the model cannot displace under either reading.

Implementation is one extra key on the sort the pipeline already did, plus the
promotion rule §7.4 describes:

```python
ranks = [
    TIER_CHECKSUM_RULE
    if fusion_rank(e) == TIER_CUE_RULE and truncates(e, model_spans)
    else fusion_rank(e)
    for e in candidates
] if model_spans else [0] * len(candidates)
```

The `if model_spans else` is load-bearing. With rules alone every candidate
would fall in tier 0 or tier 2, and ranking them unconditionally would silently
re-order rule-vs-rule overlaps that v0.2 resolved by confidence — changing
shipped behaviour for a reason nothing measured. The finding is about
rule-vs-*model* precedence, so that is the only case where it fires.
Rules-only output is byte-identical to v0.2, and a test pins it.

## 4. The guards, and evidence that they matter

Gate 3 of the results doc ran the raw student over the frozen corpus and found
it did not survive contact: `666` tagged as an MRN (which is `phi_specific`, so
that is a **false PHI** — the exact class Track A of the improvement plan
closed), `MRN: 4829471` truncated to `4829` plus a spurious `national_id`, a
fragment `-3321-77` tagged as an SSN. Three guards ship because of that:

1. **Confidence floor** (`min_confidence`, default **0.75**) — mean per-token
   probability over the span. Swept, not guessed; §7.4 for why the frozen
   corpus and not the holdout decided the value.
2. **Checksum re-validation** — a model span of `SSN`, `CREDIT_CARD`,
   `EMAIL`, `IP_ADDRESS` or `URL` must pass the same validator the rules use.
   88% of Nemotron's gold card numbers fail Luhn, so a student trained on that
   corpus *learns to emit invalid cards*.
3. **Tier precedence** (§3).

`tests/test_ner.py` asserts the four hard negatives do not produce PHI in deep
mode, and — separately — that a detector with the guards *off* produces
different output on the frozen corpus than one with them on. Without that second
test, the first would pass just as happily if the guards were dead code.

A fourth guard is not in that list because it is not about the model's output at
all: `load_bundle` calls `tokenizer.no_truncation()`. A tokenizer saved from the
teacher restores its truncation config — 1,024 tokens, roughly 6 KB of text —
and with it left on, every document is silently clean past that point. No error,
no warning, no failing test unless one is written for it. There is one.

## 5. The taxonomy decision

The plan left this open: the student emits 55 labels, we modelled 11. The answer
is **adopt the HIPAA-mapped ones as first-class types** rather than route them
through a side channel — 14 new `EntityType` members, each with a Safe Harbor
row and a risk weight, listed in [DESIGN.md](DESIGN.md) §6. That closes HIPAA
rows 1, 2, 5, 12, 13, 16 and 18, which no rule could ever have reached.

Three groups stay out, for three different reasons:

- **`state`** — HIPAA #2 is subdivisions *smaller* than a state; a state is
  retainable under Safe Harbor. Tagging it would be wrong, not conservative.
- **Credentials** (password, api_key, cvv, pin, cookie) and **GDPR
  special-category attributes** (race/ethnicity, religious belief, political
  view, sexuality) — not HIPAA identifiers. M3 policy profiles.
- **`date` / `time` / `age`** — identifiers only when tied to an individual, or
  above 89. The model cannot draw that distinction yet, and emitting them
  unconditionally would flood every document.

**No adopted type is `phi_specific`.** A name or an address identifies a person
in any context; only an MRN or a health-plan id carries health linkage in the
identifier itself. So adopting names widens PII coverage without widening what
may be called PHI — a document still needs medical context to escalate. A test
enforces it, because getting this wrong would make every document with a name
into PHI.

One collapse deserves its own note: Nemotron tags "Jane Doe" as `first_name` +
`last_name` and "44 Elm Street, Springfield" as `street_address` + `city`.
`ner.merge_adjacent` rejoins same-type spans separated only by whitespace or a
comma, so one real-world identifier is one entity. The gap test is tight enough
that "Jane Doe and John Smith" stays two names. The external evaluation applies
**the same function to gold**, so it measures the model rather than the label
convention; `--no-merge` shows the size of the convention gap.

## 6. Calibration: making the confidence mean something

M2's deferred piece, closed. [DESIGN.md](DESIGN.md) §7 is explicit that v1
confidences are "ordinal detector certainty, not calibrated probabilities", and
§8 defers real calibration to "where scores come from a model that can be
calibrated". `training/calibrate.py` is that, and the curve ships inside the
bundle.

**Why the raw score was not a probability.** The student's span confidence is
the mean of its max-softmax over the span's tokens, and a classifier trained
with cross-entropy is pushed to put all its mass on one class. Measured on
10,000 held-out documents, the overconfidence is severe at the low end:

| raw score band | spans | mean claimed | actually exact | gap |
|---|--:|--:|--:|--:|
| 0.00–0.50 | 791 | 0.395 | **0.028** | +0.367 |
| 0.50–0.60 | 790 | 0.552 | **0.215** | +0.337 |
| 0.60–0.70 | 1,848 | 0.658 | 0.557 | +0.100 |
| 0.70–0.80 | 3,148 | 0.754 | 0.705 | +0.048 |
| 0.80–0.90 | 4,274 | 0.856 | 0.820 | +0.036 |
| 0.90–0.95 | 5,050 | 0.928 | 0.897 | +0.031 |
| 0.95–1.00 | 31,298 | 0.991 | 0.971 | +0.019 |

A span scoring 0.55 was right 22% of the time. So `min_confidence=0.75` was a
knob without units — not comparable between students, and not a statement about
anything.

**What ships.** Isotonic regression (pool-adjacent-violators, ~20 lines, no
scikit-learn dependency) from raw span confidence onto the strict target: did
this span exactly match gold on `(type, start, end)`? Fitted on
`test[20000:40000]`, checked on the disjoint `test[40000:50000]`, stored as 64
knots in `model.json` and applied with `np.interp`. **Expected calibration
error 0.0384 → 0.0126.** After it, a confidence of 0.70 means *this span has
about a 70% chance of being exactly right*, and it means that for any student.

Isotonic rather than Platt because nothing suggests the relationship is a
sigmoid; **monotone** because that is what makes it safe to add after the fact.
A monotone map cannot re-order two spans, so every precision/recall number
measured before calibration still describes the same model afterwards — a test
asserts the no-reorder property directly.

**It is not a quality win, and saying otherwise would be dishonest.** F1 at the
shipped threshold is 0.891 against 0.893 before: the same point on the same
curve, because a monotone map cannot move the curve. What changed is that the
threshold is interpretable, portable, and a defensible thing to hand to policy
config — and that the risk score in `classify.py`, which multiplies each type's
weight by its mean confidence, is now multiplying by a probability on the model
side.

### 6.1 One curve was not enough

The global curve above is nearly perfect *in aggregate* and wrong in detail,
and the aggregate number is what hid it. Pooled gap between claimed confidence
and actual exact-match rate: **+0.005**. Per type, on the same documents:

| type | spans | claims | actually right | gap |
|---|--:|--:|--:|--:|
| `URL` | 4,235 | 0.765 | 0.888 | **−0.107** under |
| `DEVICE_ID` | 323 | — | — | +0.069 over |
| `FAX_NUMBER` | 616 | — | — | +0.067 over |
| `PHONE_US` | 2,496 | — | — | +0.059 over |
| `VEHICLE_ID` | 970 | — | — | +0.058 over |
| `USER_ID` | 4,617 | — | — | +0.050 over |

The errors cancel. A single threshold then cuts every type in a different
place, which is how `URL` lost twenty points of recall to a threshold that was
correct on average.

`training/calibrate.py` now fits **one isotonic curve per type**, with a
`--min-spans` floor (default 200) below which a type keeps the global curve —
a curve fitted on forty spans is noise wearing a probability's clothes, and it
would be applied with the same authority as one fitted on thirty thousand.

| | pooled ECE | worst per-type gap |
|---|--:|--:|
| raw max-softmax | 0.0228 | — |
| one global curve | 0.0138 | 0.107 |
| **per type** | **0.0035** | **0.026** |

**And it moved the operating point, which is the part worth understanding.**
Calibration is monotone, so it cannot improve F1 *within* a type. But per-type
curves change each type's score *relative to the others*, so a fixed global
threshold lands somewhere different — and better, because the thing it is
thresholding now means the same for everyone:

| | before (one curve @0.70) | after (per type @0.50) |
|---|--:|--:|
| `l` rule-tier F1 / F2 | 0.935 / 0.919 | **0.940 / 0.927** |
| `l` model-tier F1 / F2 | 0.927 / 0.915 | **0.930 / 0.918** |
| `m` model-tier F1 / F2 | 0.904 / 0.890 | **0.914 / 0.902** |
| `PERSON_NAME` F1 | 0.915 | **0.929** |
| `URL` F1 | 0.947 | **0.964** |
| frozen accuracy / PHI recall | 1.00 / 1.00 | 1.00 / 1.00 |

Two consequences beyond the numbers. **The threshold could come down from 0.70
to 0.50**: `USER_ID` was the over-confident type whose adversarial false
positives forced the high bar, and scored honestly they fall to 0.36–0.49, so
every other type gets back the recall that bar was costing it. And **the F1 and
F2 optima now agree** on one threshold (0.30), where before they disagreed —
which is what a consistent probability scale should do.

**The residual wart, stated plainly.** The *rules* are still uncalibrated. Their
confidences are hand-set ordinal constants in the 0.70–0.95 band, so a
`PERSON_NAME` at 0.81 (an 81% chance of being exactly right) and a
`regex/mrn` at 0.85 (a constant somebody chose) are still not the same kind of
number, and the risk score still adds them. Calibrating the rules is possible —
the frozen corpus is far too small to fit on, but the Nemotron holdout is not —
and it is the obvious next step rather than a fundamental problem.

## 7. Measured

The student re-trained for this change is the `m` size (d=128, 6 layers,
6.57 M parameters, 26.3 MB fp32 ONNX), distilled for 6 epochs. Loss by epoch:
0.80, 0.52, 0.45, 0.41, 0.40, 0.39 — see §7.0 for what those last three epochs
were worth.

### 7.0 A negative result: the longer training run bought nothing

[DISTILLATION_PLAN.md](DISTILLATION_PLAN.md) section 8 closed with "the
training loss was still falling when the schedule ended, so a longer run is the
cheapest untried improvement." It was tried. Scored on the *identical* protocol
as the 3-epoch run (`training/eval_student.py`, `--revalidate`,
`--min-confidence 0.0`, exact match on the 12 rule-mapped types):

| | 3 epochs (results doc) | 6 epochs (this run) |
|---|--:|--:|
| student alone, F1 | 0.890 | **0.890** |
| checksum-first fusion, F1 | 0.901 | **0.892** |

Training loss fell from 0.45 to 0.39 across the extra three epochs and holdout
F1 **did not move** — the fusion column is if anything slightly lower, within
the noise of a 20,000-document slice against the 100,000 the results doc used.
Distillation had converged by epoch 3; the remaining loss was fitting the
teacher more closely without learning anything the holdout can see.

Recording it because it is the kind of result that quietly disappears. The
improvement in this change is **not** a better-trained model. It is the
integration: the fusion promotion rule, the confidence threshold, the taxonomy
expansion, and a decoder that stopped costing more than the model. Anyone
reaching for "train it longer" as the next lever should read this table first;
the untried knobs that remain are the ones the plan also named — the learning
rate was never swept, and alpha and temperature were never varied.

The same run also settles a fusion question the results doc left in its script
but not in its tables. `eval_student.py` implements a third policy,
`longest_wins`: *every* cue-anchored rule span is promoted unless an
overlapping same-type model span is at least as long. Measured, that is worse
than plain checksum-first (F1 0.867 vs 0.892), because it promotes rule spans
the model never disputed. The version that ships (§7.4) fires only when a
same-type model span actually exists and is *strictly shorter* — the truncation
signature — and that one costs nothing (0.932 → 0.933). The difference between
those two readings is 0.025 F1, and it is entirely in how narrowly the rule is
scoped.

### 7.1 External holdout — the shipped cascade

`eval/scripts/nemotron_deep_eval.py`, 3,000 held-out Nemotron-PII documents,
13,801 mapped gold spans, exact `(type, start, end)` match, one CPU core. This
runs `deep_pipeline()` itself — the ONNX bundle, the shipped decoder, the
shipped guards, the shipped fusion — not a cascade assembled in a script.

| configuration | rule-tier P | R | **F1** | **F2** | model-tier P | R | **F1** | **F2** |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| rules only | 0.846 | 0.752 | **0.796** | **0.769** | 0.000 | 0.000 | **0.000** | **0.000** |
| deep, `m` student | 0.957 | 0.913 | **0.935** | **0.921** | 0.934 | 0.895 | **0.914** | **0.902** |
| **deep, `l` student** (recommended) | 0.962 | 0.918 | **0.940** | **0.927** | 0.951 | 0.911 | **0.930** | **0.918** |

Both at the shipped `min_confidence=0.50` with per-type calibration (§6.1). **F2 is reported next to F1
throughout** because this system's cost matrix is not symmetric — a missed
identifier is a reportable incident, a false alarm is a reviewer-minute
([DESIGN.md](DESIGN.md) §1 and §10). Neither is *the* number; the gap between
them is what a threshold is choosing, and §7.1b makes that choice explicit.
The two students are a real ladder, not a leftover — see §7.7.

**+0.144 F1 on the types the rules already covered, and 0.930 on fourteen types
they could not touch at all.** The rules-only row (0.846 / 0.752 / 0.796)
reproduces [BASELINE_NEMOTRON.md](BASELINE_NEMOTRON.md) (0.854 / 0.732 / 0.788)
to within the difference a 3,000-document slice makes, so the comparison is
same-code and same-documents rather than a number copied across.

The span-merge costs us, and the size of it is worth stating rather than
implying. Re-run with `--no-merge`, so both gold and predictions keep
Nemotron's convention of `first_name` + `last_name` as two spans:

| | gold spans | rule-tier F1 | model-tier F1 |
|---|--:|--:|--:|
| merge on (reported above) | 13,801 | 0.933 | **0.893** |
| merge off | 15,447 | 0.933 | **0.909** |

(measured on the `m` student before word-edge snapping; the ratio is what
matters and it is unaffected by either.)

Merging is 0.016 F1 *worse* for us, because one merged span has to get two
boundaries right where two unmerged spans each had to get one. The merged
number is the one reported because it is what the product emits — "Jane Doe" as
one entity — and because reporting the more flattering of two views of the same
model is how baselines stop meaning anything. The rule tier is untouched either
way; only `PERSON_NAME` and `ADDRESS` merge.

Per type with the `l` student at the shipped threshold — the model tier is
everything the rules could never emit:

| type | gold | P | R | F1 | F2 | tier |
|---|--:|--:|--:|--:|--:|---|
| `EMAIL` | 1,221 | 0.997 | 0.998 | 0.997 | 0.997 | rule |
| `DATE_DOB` | 403 | 0.988 | 0.995 | 0.991 | 0.994 | rule |
| `MRN` | 389 | 1.000 | 0.974 | **0.987** | 0.979 | rule |
| `IP_ADDRESS` | 291 | 0.990 | 0.979 | 0.985 | 0.981 | rule |
| `MAC_ADDRESS` | 137 | 0.985 | 0.985 | 0.985 | 0.985 | model |
| `SWIFT_BIC` | 157 | 0.963 | 0.994 | 0.978 | 0.987 | model |
| `HEALTH_PLAN_ID` | 311 | 0.993 | 0.961 | **0.977** | 0.968 | rule |
| `BANK_ROUTING` | 277 | 0.993 | 0.960 | 0.976 | 0.967 | model |
| `BIOMETRIC_ID` | 358 | 0.991 | 0.961 | 0.976 | 0.967 | model |
| `ACCOUNT_NUMBER` | 510 | 0.958 | 0.941 | **0.950** | 0.945 | rule |
| `USER_ID` | 1,339 | 0.976 | 0.922 | 0.948 | 0.932 | model |
| `VEHICLE_ID` | 294 | 0.934 | 0.963 | 0.948 | 0.957 | model |
| `URL` | 1,198 | 0.979 | 0.917 | 0.947 | 0.929 | rule |
| `SSN` | 249 | 0.889 | 1.000 | 0.941 | 0.976 | rule |
| `PHONE_US` | 638 | 0.897 | 0.981 | **0.937** | 0.963 | rule |
| `ADDRESS` | 1,568 | 0.964 | 0.892 | 0.926 | 0.905 | model |
| `US_DRIVER_LICENSE` | 170 | 0.934 | 0.918 | **0.926** | 0.921 | rule |
| `DATE_TIME` | 325 | 0.930 | 0.901 | 0.916 | 0.907 | model |
| `PERSON_NAME` | 3,019 | 0.926 | 0.905 | 0.915 | 0.909 | model |
| `GEO_COORDINATE` | 233 | 0.902 | 0.910 | 0.906 | 0.908 | model |
| `DEVICE_ID` | 80 | 0.940 | 0.787 | 0.857 | 0.814 | model |
| `FAX_NUMBER` | 211 | 0.938 | 0.782 | 0.853 | 0.809 | model |
| `TAX_ID` | 43 | 0.938 | 0.349 | 0.508 | 0.399 | model |
| `CREDIT_CARD` | 380 | 0.612 | 0.108 | **0.183** | 0.129 | rule |

Where F2 exceeds F1 the detector is recall-heavy and this project's cost
matrix likes it: `SSN` 0.941 -> **0.976** (recall 1.000, so every gold SSN
is found and the cost is a few false ones), `PHONE_US` 0.937 -> 0.963,
`VEHICLE_ID` 0.948 -> 0.957. Where F2 falls below F1 the detector is
leaving identifiers on the table, which is the expensive direction:
`TAX_ID` 0.508 -> **0.399** on 43 spans, `FAX_NUMBER` 0.853 -> 0.809,
`DEVICE_ID` 0.857 -> 0.814. Those three are the recall backlog, and F1
alone would have made them look better than they are.

The bolded rows are the ones the results doc singled out as weak:
`US_DRIVER_LICENSE` was **0.001** rules-only and is 0.926 fused;
`ACCOUNT_NUMBER` 0.407 → 0.950; `HEALTH_PLAN_ID` 0.448 → 0.977;
`MRN` 0.615 → 0.987; `PHONE_US` 0.471 → 0.937.

Two rows need their explanation attached rather than buried:

- **`CREDIT_CARD` 0.183 is correct, not a regression.** 88% of Nemotron's gold
  card numbers fail the Luhn checksum, so most of that gold is unreachable by a
  system that refuses to emit card numbers no payment network would issue.
  Re-validation is what makes the number look bad and what makes the detector
  right; the alternative measured F1 0.938 by agreeing with invalid cards.
- **`TAX_ID` 0.326 recall on 43 spans** is too thin to conclude anything from.
  It is reported rather than dropped so the thinness is visible.

### 7.1a Pooled micro scores, and cost per document

The tier split above is useful for attribution and useless as a headline, since
the two tiers cover disjoint type sets and cannot be averaged. Pooled over all
24 types, micro (each type weighted by the gold it has):

| | P | R | **F1** | **F2** |
|---|--:|--:|--:|--:|
| rules only | 0.846 | 0.314 | **0.458** | **0.359** |
| **deep (`l`) @0.50** | 0.955 | 0.914 | **0.934** | **0.922** |

The rules' pooled recall of 0.314 is not a bug in the measurement: fourteen of
the twenty-four types have no regex that can emit them, so every one of their
gold spans is a miss by construction. That gap is what Stage 2 exists to close,
and the pooled row is the only place it shows up as one number.

**Cost per document, one core, `taskset -c 0`:**

| document | fast (rules) | deep `m` | deep `l` |
|---|--:|--:|--:|
| 1 KB | 0.13 ms | 1.18 ms | 2.06 ms |
| **10 KB** (the reference) | **0.69 ms** | **8.54 ms** | **15.51 ms** |
| 100 KB | 7.64 ms | 124.71 ms | 207.46 ms |
| peak RSS | 20 MB | 120 MB | 137 MB |

p95, against a 5 ms budget for `fast` and 25 ms for `deep`. On the real
Nemotron documents (854 characters mean) `l` costs **2.18 ms mean / 4.59 ms
p95** end to end, rules included — the 10 KB row is a deliberately pessimistic
reference size, not a typical one.

Throughput on one core: roughly **65 documents/second** at Nemotron's document
size, or **460 KB/s** of text at the 10 KB reference. Latency scales linearly
with length, and the model reads every token — §7.8 and the windowing study
explain why it has to.

### 7.1b The F1 and F2 optima disagree, and the frozen corpus breaks the tie

Sweeping `min_confidence` on the `l` student, 3,000 holdout documents:

| threshold | rule F1 | **rule F2** | model F1 | **model F2** | frozen doc accuracy |
|---|--:|--:|--:|--:|--:|
| 0.00 | 0.937 | **0.927** | 0.929 | **0.927** | 0.90 |
| 0.20 | 0.938 | **0.927** | 0.930 | **0.927** | 0.90 |
| 0.35 | 0.938 | **0.927** | 0.930 | 0.925 | 0.92 |
| 0.50 | 0.938 | 0.925 | 0.931 | 0.923 | 0.95 |
| **0.70** (shipped) | 0.935 | 0.919 | 0.927 | 0.915 | **1.00** |
| 0.80 | 0.935 | 0.918 | 0.918 | 0.903 | 1.00 |

**Top F2 is 0.927**, flat across 0.00–0.20. F2 always prefers a lower threshold
than F1 — lowering it only ever adds spans, so recall rises and F2 weights
recall four times as heavily. The shipped 0.70 gives up **0.008 rule-tier and
0.012 model-tier F2** against that ceiling.

That is a real cost and it is paid deliberately, for a reason the F2 column
cannot see. The last column is why: at the F2 optimum the frozen corpus falls
to document accuracy **0.90**, because the extra recall is order numbers, chart
numbers and subscriber ids being tagged as identifiers — the exact
false-positive class Track A of the improvement plan hardened the rules
against. F2 counts a recovered gold span and does not count a document that a
reviewer now has to dismiss.

Two further reasons the F2 ceiling is not the target:

- **Document-level PHI recall is 1.00 at every threshold in the table**,
  including 0.80. The recall metric this project actually ranks first is
  already saturated; lowering the threshold buys span-level recall on types
  like `USER_ID`, not one additional PHI document caught.
- The two tiers are scored over disjoint type sets, so there is no single
  combined F2 here. Averaging the two columns would not be a micro-average and
  is not reported as one.

`pii-master eval` and both external scripts now print F1 and F2 side by side,
and `nemotron_deep_eval.py` reports the F1-optimal and F2-optimal thresholds
separately with a warning when they disagree — which they always will.

### 7.2 Frozen corpus — no regression, and the gap it was built to expose

`pii-master eval --deep`. The corpus is adversarial by construction: 14 of its
39 documents are hard negatives built from near-miss identifiers, and it carries
`PERSON_NAME` / `ADDRESS` gold that rules-only scores as *undetectable*.

| | rules only | deep |
|---|--:|--:|
| document accuracy | 1.00 | **1.00** |
| PHI recall | 1.00 | **1.00** |
| false PHI | 0 | **0** |
| `PERSON_NAME` recall | undetectable | **0.50** |
| `ADDRESS` recall | undetectable | **1.00** |
| spurious spans | 0 | **0** |
| every rule type (P/R/F1) | 1.00 | **1.00** |

Deep mode is scored with `undetectable=frozenset()` — nothing is excused, a
missed name counts as a real miss. The single remaining error is one boundary:
`John Q. Patient` comes back as `John Q`.

### 7.3 What it looks like

```
Discharge summary. Patient Jane Doe, DOB: 03/14/1985, MRN: 4829471. Lives at
44 Elm Street, Springfield, MA 01103. Contact jane.doe@example.com or
(415) 555-2671. Fax 415-555-9012. Insurance beneficiary number HP-99120.
Card 4111 1111 1111 1111.
```

| deep | rules only |
|---|---|
| `PERSON_NAME` "Jane Doe" | — |
| `ADDRESS` "44 Elm Street, Springfield" | — |
| `FAX_NUMBER` "415-555-9012" | `PHONE_US` "415-555-9012" |
| `DATE_DOB`, `MRN`, `EMAIL`, `PHONE_US`, `HEALTH_PLAN_ID`, `CREDIT_CARD` | same |

Both label the document PHI. Deep mode adds the two identifiers no regex can
reach and corrects the fax/phone confusion — HIPAA lists them as separate rows
(#4 and #5) and the rules tier has no way to tell them apart.

### 7.4 Two fusion findings this change had to add

The policy the results doc measured was not sufficient on its own, and both
gaps showed up on the frozen corpus rather than the holdout — short adversarial
documents stress fusion in a way 1,000-character synthetic ones do not.

**Truncated model spans displaced correct rule spans.** `DATE_DOB "3-4-1985"`
became `"-4-"`; `PHONE_US "(415) 555-2671"` became `"415) 555-2671"`;
`HEALTH_PLAN_ID "HP-99120"` became `"-99120"`. In each case Stage 1 already had
it right and a strictly-shorter model span of the *same type* outranked it.
`pipeline.truncates` promotes the rule span in exactly that configuration, and
it is deliberately narrow — cross-type disagreements are left alone, because
that is where the model's +0.137 comes from. Cost on the holdout: **none**;
rule-tier F1 went 0.932 → 0.933.

**The confidence default is set by the hard negatives, not the holdout.** On the
holdout the model tier peaks at a threshold of 0.50 (F1 0.900) and declines
slowly; the frozen corpus is not so relaxed. The student produces four false
PII there — an order number, a chart number, a magazine subscriber id, a
confirmation number, all tagged `customer_id`/`employee_id` — and calibration
(§6) makes their scores readable:

| frozen-corpus false positive | calibrated confidence |
|---|--:|
| `USER_ID "471"` (from "The chart 4829471") | 0.111 |
| `USER_ID "71"` (from "order 4155552672") | 0.233 |
| `USER_ID "44159928170"` (an order number) | 0.497 |
| `USER_ID "A9-3321-"` (a magazine subscriber id) | **0.626** |

Only a threshold above 0.626 clears all four, so **0.70** ships. It costs 0.009
model-tier F1 against the 0.50 optimum, and re-opening the reference-number
false-positive class that Track A of the improvement plan hardened the rules
against is not worth 0.009. Synthetic-holdout precision and adversarial
near-miss precision are different measurements, and a default should be set by
the second.

Worth noting what the calibrated numbers reveal that the raw ones hid: the two
truncated fragments score 0.111 and 0.233 — the model *knows* they are junk —
while `"A9-3321-"` scores 0.626, a genuine confident error. Those are two
different failure modes that the raw scores (0.51 and 0.65) made look alike.

### 7.4b A measured "no": trimming trailing punctuation

`PERSON_NAME` triage over 1,500 holdout documents — 85.6% exact, 5.4% missed
entirely, 3.5% spurious, 3.4% short at the end, 1.1% too long — put a visible
chunk of the "too long" bucket on trailing punctuation and possessives:
`"Kathryn Carpenter."`, `"Marybeth's"`, `"191 Ardale St."`. A decode-time trim
is three lines, so it was measured before it was written.

Across all types on 2,000 documents (8,067 predicted spans, 454 not exact):

| | spans |
|---|--:|
| errors a trailing-punctuation trim would **fix** | 31 |
| errors a possessive trim would **fix** | 3 |
| currently-**exact** spans it would **break** (all `GEO_COORDINATE`) | 7 |

Net **+27 spans in 8,067**, about +0.003 F1 — and only if `GEO_COORDINATE` is
carved out by hand, on the evidence of seven examples from the same slice that
motivated the rule. That is a heuristic tuned to its own test set, and the kind
of silent boundary bug that shows up two releases later on a type nobody
checked. **Not implemented.** Recorded here so the idea does not get
re-litigated from scratch: the measurement says it is not worth it, and the
model getting better at boundaries is the fix that generalises.

### 7.5 The `xs` student fits the budget and still does not ship

`xs` (d=64, 4 layers, 3.25 M parameters, 13 MB) was trained on the same
schedule specifically to answer the question the results doc left open: can a
model tier fit the strict 5 ms contract? **It can — 4.79 ms p95 end to end —
and it fails acceptance gate 3 anyway.**

| | `xs` | `m` | `l` |
|---|--:|--:|--:|
| 10 KB p95, deep cascade | **4.79 ms** | 7.99 ms | 14.80 ms |
| holdout, rule-tier F1 | 0.921 | 0.933 | **0.935** |
| holdout, model-tier F1 | 0.818 | 0.904 | **0.926** |
| frozen corpus, doc accuracy | 0.97 | **1.00** | **1.00** |
| frozen corpus, **PHI recall** | **0.92** | **1.00** | **1.00** |

(`xs` measured before calibration and word-edge snapping, which lift both other
students; it fails on the document label, and neither of those changes what a
displaced `phi_specific` span does.)

Gate 3 of [DISTILLATION_PLAN.md](DISTILLATION_PLAN.md) is "no regression on the
frozen corpus — in particular PHI recall 1.00". `xs` scored 0.92, and the plan's
rule for that case is not ambiguous: *"If the student cannot clear the gate, do
not ship it. The cascade is additive: rules alone are already a defensible
product."*

The single document it lost is the interesting part, because it exposed a hole
in the fusion policy that `m` happened not to hit — see §7.6. So the disqualified
student paid for itself: a latency tier nobody ships, and a correctness guard
everybody gets.

The remaining case for `xs` would be a deployment that genuinely cannot afford
8 ms. If that arrives, the honest sequence is to fix the model rather than lower
the gate — its 0.818 on the model tier against `m`'s 0.904 says it is
undertrained for the label space, not too small for it, and the untried knobs
from §7.0 (learning rate, alpha, temperature) have never been swept for either
size.

### 7.6 The guard the disqualified student bought

`xs` lost `phi-011`: *"Coverage active: insurance member id 4471-2299 effective
this month."* The rules detect `HEALTH_PLAN_ID` there, which is `phi_specific`
and escalates the document on its own. The student proposed a different,
non-PHI-specific type over the same span; under tier precedence the model won;
and because "insurance" is not a medical-context term, nothing else in the
sentence could carry the label. **PHI → PII, from one displaced span.**

That is a structural hole, not an `xs` quirk. Any model span that displaces a
`phi_specific` rule span can erase the document label, and that is the one
outcome docs/DESIGN.md §1 ranks worst: *"a missed medical record number leaking
into a data lake is a reportable incident; a false alarm costs a reviewer
minutes."* `pipeline.erases_phi` closes it.

The narrowness is the whole trick, and it was measured:

| guard | holdout rule-tier F1 | MRN F1 | HEALTH_PLAN_ID F1 | frozen PHI recall (`xs`) |
|---|--:|--:|--:|--:|
| none | 0.933 | 0.979 | 0.966 | 0.92 |
| protect all `phi_specific` rule spans | 0.926 | 0.943 | 0.883 | 1.00 |
| **protect only where PHI would be erased** | **0.933** | **0.979** | **0.966** | **1.00** |

Blanket-protecting the two types costs 0.007 micro F1, concentrated exactly on
them. Yielding whenever the model *also* proposes a `phi_specific` type — so a
model `MRN` span may still refine a rule `MRN` span, since the document label
survives either way — costs **nothing at all** and closes the hole completely.
The model is the better judge of boundaries; it just does not get to decide
whether a document is PHI.

### 7.7 Two students, and why the ladder grew a rung

The `xs` result (§7.5) said something the size ladder had not been read for:
`xs` → `m` is +0.086 model-tier F1 for 3.3 M extra parameters, while three
extra training epochs bought **nothing** (§7.0). Together those say the students
are **capacity-limited, not under-optimised**. And the shipped cascade was using
8 ms of a 25 ms budget.

So the ladder grew a rung: `l`, d=192 × 8 layers, 10.0 M parameters, sized to
the measured 43 GMAC/s this core actually achieves rather than to a guess. The
dilation cycle deliberately does **not** extend past 32 — doubling it for two
more layers would give a 1,020-token receptive field, and the cue-to-value
distances PII needs are tens of characters. Repeating `(1, 2)` instead keeps the
field at 132 tokens against `m`'s 126, so `l` differs from `m` in capacity,
which is the thing being tested, and not in what it can see.

| | `m` | `l` |
|---|--:|--:|
| parameters / ONNX | 6.57 M / 26 MB | 10.0 M / 40 MB |
| 10 KB p95, whole cascade | **7.99 ms** | 14.80 ms |
| share of the 25 ms budget | 32% | 59% |
| peak RSS | 112 MB | 128 MB |
| span exact-match rate, 20k docs | 0.902 | **0.931** |
| holdout rule-tier F1 | 0.933 | **0.935** |
| holdout model-tier F1 | 0.904 | **0.927** |
| `PERSON_NAME` F1 | 0.878 | **0.915** |
| `ADDRESS` F1 | 0.891 | **0.926** |
| frozen accuracy / PHI recall | 1.00 / 1.00 | 1.00 / 1.00 |

`l` is better on every holdout metric and clears every gate, so it is the
recommended student. `m` stays supported and documented rather than deleted:
it is half the latency and two-thirds the memory for 0.022 model-tier F1, which
is the right trade for anyone whose budget is tighter than the 25 ms this
project assumed. Both are one `PII_MASTER_MODEL_DIR` apart.

The honest caveat on `l`: on the frozen corpus its `PERSON_NAME` scores 0.00
against `m`'s 0.50. That is **two gold spans**, both models miss
`John Q. Patient` the same way, and `l`'s extra error is over-covering
`Applicant Jane Doe`. Its holdout `PERSON_NAME` precision is *higher* than
`m`'s (0.924 vs 0.889) on 3,019 spans, so this is not a precision regression —
it is a two-sample corpus doing what two-sample corpora do. Worth stating
because the alternative is quietly reporting only the table that flatters.

### 7.8 One more thing the boundaries were worth

Triaging `PERSON_NAME` errors over 1,500 documents — 85.6% exact, 5.4% missed,
3.5% spurious, **3.4% short at the end**, 1.0% short at the start — turned up a
pattern that is not about names at all: `'Tommy Dijkhu'`, `'amela Navas'`,
`'ant Jane Doe'`. The span edge is *inside a word*, because that is where the
subword tokenizer happened to split.

That is an artefact of tokenization, not a judgement the model is making, and
the data says so: of 103,681 crosswalked gold spans across 20,000 documents,
**8 begin and 49 end inside an alphanumeric run** — 0.055%, concentrated in a
couple of labels that look like annotation noise. An identifier essentially
never cuts a word in half.

`ner.snap_to_word_edges` grows such edges out to the word boundary. Grow rather
than shrink, and that was measured too — on 2,000 documents growing fixed **76
spans and broke 0**, while shrinking fixed 0. Edges are clamped against
neighbouring spans so two entities inside one word cannot be grown into each
other and silently collapsed to one by overlap resolution.

Worth +0.013 model-tier F1 on `m` (0.891 → 0.904) and +0.008 on `l`, for
about fifteen lines and no new failure mode. It is the counterexample to §7.4b:
a boundary heuristic *can* be worth shipping — when the property it encodes is
measurable in the gold data rather than fitted to the errors it is trying to
fix.

### 7.9 The number that was missing: document-level detection

Everything above measures spans. The decision a user actually makes with this
tool is coarser and more consequential — **quarantine this document or not** —
and until now it had only ever been measured on the 39-document frozen corpus,
which its own documentation calls a regression test rather than a quality
claim. `eval/scripts/document_eval.py` measures it on the holdout.

Gold is derived from Nemotron's own span annotations, not from our classifier:
a document is sensitive if it carries at least one gold span crosswalking to a
type we model. 2,983 of 3,000 test documents qualify.

| | rules only | **deep (`l`)** |
|---|--:|--:|
| recall on documents containing an identifier | 0.8086 | **0.9977** |
| documents missed entirely | **571** | **7** |
| false-alarm rate on adversarial negatives | 0.0000 | **0.0000** |

**One in five documents containing PII was invisible to the rules. It is now
one in 425** — an 81× reduction in missed documents, with the false-alarm rate
on the frozen corpus's fourteen near-miss negatives still exactly zero.

That is a much larger effect than the span-level numbers suggest, and the
reason is structural: a document is caught if *any* identifier in it is caught.
The rules missed whole documents because the only identifiers present were
names and addresses, which no regex can reach. Span F1 improved 0.796 → 0.940;
document recall improved 0.809 → 0.998, because the model does not have to be
right about everything in a file, only about something.

The seven remaining misses are worth naming rather than rounding away. They are
documents whose only modelled identifier is one the cascade suppresses or
cannot see: a card number the Luhn re-validation correctly drops (§7.1), a
bare `123456` invoice number, an unpunctuated `20240615` date. The first is
working as designed; the other two are the recall backlog.

**What this does not measure.** Nemotron has no document labels and no
medical-context annotation, so the PII-vs-PHI *split* has no external gold —
any would have to be derived with the same `has_medical_context` heuristic the
classifier uses, and scoring a rule against itself measures nothing. The split
is reported as a distribution (2,402 PII / 574 PHI / 24 NONE) and the frozen
corpus remains the only place PHI escalation is scored against authored gold,
where it is 1.00 recall on 12 documents. **That is the weakest link in the
evaluation story and it is a data problem, not a modelling one:** it needs real
clinical text with document-level labels, which means n2c2 under a DUA.

### 7.10 Cross-corpus: it does not generalise as well as the headline suggests

Every number above comes from Nemotron-PII. `eval/scripts/ai4privacy_eval.py`
scores the same shipped cascade on **ai4privacy/pii-masking-300k**, a corpus
with a different author, label space, document style and locale: 7,946 English
validation documents, 34,123 crosswalked gold spans.

It is a deliberately hard shift. Nemotron is US narrative prose; ai4privacy is
structured templated data — JSON, XML, markdown key/value forms — and
predominantly UK/EU:

```
"building": "617", "street": "Holme Wood Lane", "city": "Doncaster"
- Social Number: 669 398 5477      (ten digits; a US SSN has nine)
- Phone: +16 079 662 2565          (not NANP)
```

| in-scope recall | rules only | **deep (`l`)** |
|---|--:|--:|
| strict (exact span + accepted type) | 0.187 | **0.385** |
| typed (exact span, any type we model) | 0.214 | **0.425** |
| located (overlapping span) | 0.229 | **0.575** |
| **document-level recall** | 0.560 | **0.870** |

Against Nemotron's micro F1 0.934 and document recall 0.998, that is a **large
generalisation gap, and it is the most important thing on this page.** Deep
mode roughly doubles the rules everywhere, so the cascade still earns its
place — but anyone reading 0.934 as "how well this works" would be wrong by a
wide margin on text that does not look like its training set.

**What survives the shift and what does not** is the useful part:

| ai4privacy label | gold | strict R | located R | reading |
|---|--:|--:|--:|---|
| `EMAIL` | 2,612 | **0.943** | 1.000 | format-anchored, locale-free |
| `IP` | 2,166 | **0.988** | 0.992 | same |
| `USERNAME` | 2,786 | 0.545 | 0.677 | |
| `SOCIALNUMBER` | 2,554 | 0.334 | 0.626 | typed R 0.608 — we find it, mislabel it |
| `POSTCODE` | 1,905 | 0.433 | 0.554 | |
| `CITY` / `STREET` | 3,922 | ~0.33 | ~0.72 | region found, boundaries wrong |
| `GIVENNAME1`/`LASTNAME1` | 4,409 | ~0.30 | ~0.54 | same |
| `BOD` | 2,317 | 0.161 | 0.253 | `January/88`, `17th February 1946` |
| `BUILDING` | 1,935 | **0.028** | 0.270 | a bare `617` in a JSON field |
| `GEOCOORD` | 216 | **0.000** | 0.208 | boundaries, every time |

**The format-anchored rule types transfer perfectly and the learned semantic
types do not.** `EMAIL` and `IP` are as good here as on Nemotron because an
email is an email in any corpus. Names and addresses collapse — and the
`strict` vs `located` gap says the model usually *finds* the right region and
gets the *boundaries* wrong, which is a distribution effect: it learned span
edges from prose and this corpus is JSON.

`BUILDING` at 0.028 is the clearest case of a genuine capability gap rather
than a boundary one: `"building": "617"` is a bare number that is only an
identifier because of the key next to it, and nothing in the current design
reads structural context. That is what M4's PDF/DOCX work is really about
(IMPROVEMENT_PLAN Track F), arriving earlier than expected.

Of 19,260 predicted spans, 70.7% matched crosswalked gold, 4.0% landed on gold
of a label we deliberately do not score (`TITLE`, `DATE`, `COUNTRY` — real
identifiers, unscored by choice), and **25.3% were genuinely spurious**, mostly
`ADDRESS` and `PERSON_NAME` with wrong edges.

**The crosswalk itself was the first thing this evaluation caught, and it was
mine.** The first draft mapped `SOCIALNUMBER` to `NATIONAL_ID` alone, which
scored 854 correct US-format SSN detections as false positives and reported
`NATIONAL_ID` recall as 0.002 — a number about the crosswalk, not the model.
Labels now map to *sets* of our types, and the `typed` column exists precisely
so a reader can see how much of the score depends on reconciling label spaces
at all.

**Two things this does not measure.** Non-English rows were excluded (the
corpus has 39,782 of them across five languages; the model is US/English by
design and would score near zero, which is a scope statement not a finding).
And ai4privacy has no `MRN`, `HEALTH_PLAN_ID`, `CREDIT_CARD`, `ACCOUNT_NUMBER`
or `BANK_ROUTING` labels at all, so the PHI-specific half of the taxonomy is
still unmeasured outside Nemotron. `ai4privacy/pii-masking-health-phi-preview`
would be the corpus for that; it is 50 rows with the values redacted.

