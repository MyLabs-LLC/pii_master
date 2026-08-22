# Stage 2 distillation — measured results

**Status:** the runbook in [DISTILLATION_PLAN.md](DISTILLATION_PLAN.md) section 5 was
executed end to end on 2026-08-22. Two students were trained, exported, benchmarked on
one CPU core, and scored against the Nemotron-PII holdout. **Every acceptance gate in
section 6 passes**, and the student is not integrated into the package yet — that is the
separate change section 7 describes, and section 8 of this document says what it must
carry.

Hardware: RTX 4080 Laptop (12 GB) for training, one core of an i9-13900HX for every
latency number. The plan's own figures came from a deliberately slow reference box, so
the latencies below are faster; the *ratios* it predicted all held.

---

## 1. What was run

| step | outcome |
|---|---|
| Environment, teacher, 2×100k-row Nemotron parquet | done; teacher was already in the HF cache |
| Smoke run (2,000 rows, 1 epoch) | 7 s — surfaced two blocking data bugs, section 2 |
| `xs` full run, 3 epochs | 15 min (301 s/epoch) |
| `m` full run, 3 epochs | 17 min |
| Soft-target ablation, 3 runs | section 3 |
| ONNX export + 1-core benchmark | section 4 |
| Holdout scoring, 100k documents | section 5 |
| Name-demographic slice (gate 5) | section 6 |

The plan estimated 25–45 min per epoch and 1.5–2.5 h for three. Actual: **5 min per
epoch**, ~15 min for three. The estimate was for the teacher forward pass over 100k
documents; in practice the teacher runs under `no_grad` in bf16 at batch 16 and the
student is 3 M parameters, so the GPU is nowhere near saturated.

## 2. Two data bugs the smoke run caught

Both would have trained silently to a worse model. Neither is a mistake in the plan's
arithmetic — they are properties of the teacher that only reading its output reveals.

**The teacher's head is narrower than our label space (107 vs 111).** It has no
`I-ssn`, `I-cvv`, `I-gender` or `I-employee_id` column, and `build_label_permutation`
hard-fails on exactly that, as designed. The cause is benign: the teacher tags BIO at
*word* level and those four types never span two words (0 of 24,862 train spans). Fix:
the four absent columns get a constant `ABSENT_LOGIT` (-1e4, finite — `-inf` makes KL
produce `0 * -inf = NaN`), which states the truth that the teacher assigns zero
probability to a class it cannot emit. A type missing *entirely* is still a hard failure.

**The teacher is only supervised on the first subword of each word.** Reading its
predictions token by token shows it tags `123-45-6789` as five consecutive `B-ssn` and
`4821 Maple Avenue` as `B,B,I,I` — word level, not subword level. Measured against gold
on 300 train documents:

| tokens | teacher argmax agrees with gold |
|---|--:|
| word starts | **0.990** |
| within-word continuations | 0.640 |

The continuations are not hard, they are *unlearned*: the teacher was trained with them
masked to -100. `data.py` now labels at word level to match, and the soft loss is scoped
to where the teacher is trustworthy (section 3). Without this, distillation spends most
of its weight on the teacher's unsupervised region.

## 3. Ablation: where the soft target applies

Scoping the KL term turned out to be the highest-leverage knob in the whole run —
larger than the size ladder. All three variants are the `xs` student, 3 epochs, scored
on 20,000 holdout documents.

| `--soft-scope` | mapped F1 | 55-label F1 | spurious spans |
|---|--:|--:|--:|
| `word_homogeneous` (default) | **0.804** | **0.691** | 801 |
| `word_start` | 0.792 | 0.669 | 1,471 |
| `broadcast` | 0.492 | — | — |

`word_start` puts KL only on word starts. `broadcast` copies the word-start distribution
across the whole word, on the reasoning that word-level BIO gives every subword of a word
the same gold label. That reasoning is *almost* right, and the 0.30 F1 it costs is the
gap: **a whitespace word is not a span**. In `6789,` the trailing comma is inside the word
and outside the span, so broadcasting teaches the student to swallow punctuation — and
punctuation is precisely what exact-match boundary scoring grades. `word_homogeneous`
broadcasts only onto tokens whose gold label already matches the word start, and gets the
benefit without the trap: same recall, 46% fewer spurious spans.

The failure mode it fixes is real and was measured before the fix — the single largest
error class in the first `xs` student was truncated span ends (`271210785` tagged as
`2712107`), 1,363 of 12,732 gold spans in a 1,500-document sample.

## 4. Gate 1 — latency on one core

ONNX fp32, `intra_op_num_threads=1`, `taskset -c 0`, 2,000 tokens ≈ a 10 KB document:

| size | params | MB | mean | p95 | plan's reference box (p95) |
|---|--:|--:|--:|--:|--:|
| `xs` (d=64, 4L) | 3.25 M | 13.0 | 1.12 ms | **1.14 ms** | 3.08 ms |
| `m` (d=128, 6L) | 6.57 M | 26.3 | 4.40 ms | **4.43 ms** | 10.58 ms |

**int8 is 12× slower here**, which more than reproduces the plan's finding (a):

| size | fp32 p95 | dynamic int8 p95 |
|---|--:|--:|
| `xs` | 1.14 ms | 13.48 ms |
| `m` | 4.43 ms | 43.45 ms |

Ship fp32. The 3.3 MB int8 file is a disk-space saving that costs an order of magnitude
of latency, because the model is not matmul-bound.

### The number the plan's arithmetic was missing

Section 3 of the plan budgeted `rules + student`. Serving `deep` mode also pays for
tokenization and span decoding, and on a 10 KB document those cost **more than the
student does**:

| stage | `xs` cascade | `m` cascade |
|---|--:|--:|
| rules (Stage 1) | 0.51 ms | 0.52 ms |
| **tokenize** | **2.59 ms** | **2.72 ms** |
| student (ONNX, 1 core) | 1.04 ms | 3.84 ms |
| decode spans | 1.08 ms | 1.07 ms |
| **total p95** | **5.21 ms** | **8.04 ms** |

(30 documents × 10 KB, ~1,659 tokens, one core, p95, measured with
`training/bench_e2e.py` on the same synthetic corpus `pii-master bench` uses.)

**Gate 1 passes with room: 8.04 ms against a 25 ms `deep` budget.** But the promotion
rule — "a student under 4 ms is promoted into `fast`" — does not fire on this
measurement, even for `xs` whose forward pass is 1.04 ms: the cascade is 5.21 ms, just
over the 5 ms contract, and 3.7 ms of that is tokenizer and decoder, not the model.
Two fixes, both already in the design and neither requiring a new student:

* **Run the student on candidate windows, not the whole document** (Track D.2 of
  IMPROVEMENT_PLAN.md, "windows only" in DESIGN.md section 8). Tokenization and inference
  both scale with the text handed to them; the pre-scan already knows which parts matter.
* **Vectorize `decode_spans`.** 1.08 ms is a Python loop over ~1,700 tokens.

Until one of those lands, the honest statement is: `deep` mode fits comfortably,
`fast` mode stays rules-only.

## 5. Gate 2 — beating the rules on the Nemotron holdout

100,000 held-out documents, exact `(type, start, end)` match on the 12 mapped types,
scored by `training/eval_student.py` with the same protocol as
`eval/scripts/nemotron_eval.py`. The rules row is recomputed here rather than copied
from [BASELINE_NEMOTRON.md](BASELINE_NEMOTRON.md), and it reproduces it exactly
(0.854 / 0.732 / 0.788), so the comparison is same-code and same-documents.

| system | P | R | **F1** | adj P | adj F1 | partial R |
|---|--:|--:|--:|--:|--:|--:|
| rules only (committed baseline) | 0.854 | 0.732 | 0.788 | 0.936 | 0.822 | 0.749 |
| `xs` student alone | 0.727 | 0.861 | 0.788 | 0.774 | 0.816 | 0.989 |
| `m` student alone | 0.853 | 0.930 | 0.890 | 0.877 | 0.903 | 0.992 |
| `m` + fusion (rules first) | 0.822 | 0.932 | 0.873 | 0.898 | 0.914 | 0.967 |
| `m` + fusion (checksum first) | 0.846 | 0.963 | **0.901** | 0.924 | 0.943 | 0.992 |

`adj P` excludes predictions that landed on gold of a label we deliberately do not model
— the same adjustment BASELINE_NEMOTRON.md applies to the rules.

**Gate 2 passes: 0.901 against 0.788, +0.113 F1.** `xs` alone only ties the rules, so on
this evidence `xs` is not worth shipping as a tier of its own; `m` is the student.

### The fusion policy matters more than expected

"Rules are authoritative" costs 0.028 F1 against "only *checksum-validated* rules are
authoritative". The plan's own wording is the narrow one — checksum-validated spans
outrank model spans — and the data agrees emphatically. The types where blanket rule
precedence hurts are the cue-anchored ones, where the rules were never strong:

| type | rules F1 | `m` student F1 | rules-first | checksum-first |
|---|--:|--:|--:|--:|
| `PHONE_US` | 0.471 | 0.889 | 0.651 | 0.719 |
| `ACCOUNT_NUMBER` | 0.407 | 0.803 | 0.678 | 0.806 |
| `HEALTH_PLAN_ID` | 0.448 | 0.897 | 0.797 | 0.896 |
| `MRN` | 0.615 | 0.891 | 0.851 | 0.887 |
| `US_DRIVER_LICENSE` | 0.001 | 0.829 | 0.820 | 0.831 |
| `EMAIL` | 0.995 | 0.929 | 0.980 | 0.980 |
| `URL` | 0.946 | 0.830 | 0.952 | 0.952 |
| `DATE_DOB` | 0.976 | 0.963 | 0.966 | 0.962 |

So the shipped precedence should be **checksum rules > model > cue-anchored rules**: a
Luhn-validated card or a parsed IP outranks the model, and a cue-anchored MRN guess does
not. `US_DRIVER_LICENSE` moving from 0.001 to 0.831 is the single largest win, and it is
the gap BASELINE_NEMOTRON.md predicted (Nemotron's cue is "certificate license number";
our rule only accepts driver's-licence wording).

## 6. Gates 3–5

### Gate 3 — no regression on the frozen corpus

`pytest` (239 tests, 17 new) and `pii-master eval --fail-under` are green, and
`pii-master bench --fail-over-budget` still reports 0.52 ms p95 at 10 KB. The shipped
pipeline is untouched: the student lives in `training/`, and nothing in
`src/pii_master/` imports it.

That is the letter of the gate. The spirit is more interesting, so the student was run
over the 39 frozen documents as a **rehearsal of the integration change**, and it does
not survive contact with them:

| frozen doc | what the student adds | why it matters |
|---|---|---|
| `none-007` "The form rejected 666-12-3456 as invalid, and the card 4111 1111 1111 1112 failed validation" | `MRN "666"`, `CREDIT_CARD "111"` ×3 | **a false PHI** — the exact failure Track A of IMPROVEMENT_PLAN.md closed |
| `none-011` "Your subscriber id A9-3321-77 for the magazine" | `SSN "-3321-77"` | fragment, wrong type |
| `phi-001` "MRN: 4829471" | `MRN "4829"` + `national_id "471"` | fusion would *replace* a correct rule span with a truncated one |

The frozen corpus is adversarial by construction — it is full of near-miss identifiers
that Stage 1 was hardened to reject — and these documents are one terse sentence each,
far from Nemotron's ~1,000-character documents. Two mitigations, both measured on the
holdout in section 7, remove most of it: a confidence threshold and checksum
re-validation of model spans. **The integration change must ship both, and must add
these documents to the frozen corpus as regression rows.**

### Gate 4 — checksum types must not degrade

Under checksum-first fusion, rule spans of `SSN`, `CREDIT_CARD`, `IP_ADDRESS`, `EMAIL`
and `URL` are kept verbatim and a model span may not displace them, so no
checksum-validated span is lost by construction. The other half of the plan's clause —
"model spans of checksummed types are re-validated before they may carry high
confidence" — is implemented as `--revalidate` and matters exactly as predicted:
88% of Nemotron's gold cards fail Luhn, and an un-revalidated student learns to emit
them (`CREDIT_CARD` F1 0.938 on this holdout, by agreeing with card-shaped strings no
payment network would issue). Re-validation drops that to 0.207, which is the *correct*
number against a corpus whose card gold is mostly unreachable by design — and it is what
stops the student from re-opening a false-positive class the rules were built to close.

### Gate 5 — name-demographic slice

`training/eval_names.py` joins the holdout's `last_name` gold spans to the 2010 US Census
surname file (public domain), which gives the bearer distribution by race/ethnicity per
surname. A surname joins a group when ≥60% of its bearers report that category. 52,235
spans over 100,000 documents, `m` student:

| group | spans | exact recall | partial recall |
|---|--:|--:|--:|
| White | 32,446 | 0.852 | 0.927 |
| Hispanic | 7,238 | 0.869 | 0.949 |
| Asian/Pacific Islander | 5,590 | 0.871 | 0.921 |
| Black | 1,247 | 0.851 | 0.965 |
| mixed (no ≥60% group) | 5,713 | 0.833 | 0.886 |
| Am. Indian/Alaska Native | 1 | — | too thin to report |

**Spread: 0.020 exact recall** between the best and worst groups with ≥100 spans. That is
the opposite of the pattern PRIOR_ART.md section 5f documents in published maskers, and
the reason is worth stating plainly rather than claiming as a win: Nemotron-PII is
*synthetic*, so its names are drawn from a generator rather than from the world, and a
synthetic corpus cannot show a bias that comes from real-world name frequency. The slice
is a real gate — it exists, it runs, it would catch a large disparity — but it certifies
the model on synthetic names only. Re-running it against real annotated clinical text
(n2c2, under DUA) is what would make it a claim about production behaviour.
