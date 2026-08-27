# v3 was tuned on the sealed set; re-deriving its thresholds honestly costs four gates

## Results

Three arms, one scorer, the eight sealed `data/2-eval` corpora, the two policies
this project declared before any of them existed. `v3` was scored **through its own
packaged runtime**; `v2` likewise; `v4` was built during this run.

| metric | v2 | v3 | **v4** |
| --- | ---: | ---: | ---: |
| **macro F2** *(headline ranker)* | 0.6614 | **0.6657** | 0.6641 |
| micro F1 | 0.7255 | 0.7533 | **0.7862** |
| **priority macro F0.5** *(contra-view ranker)* | 0.7434 | 0.7587 | **0.8031** |
| macro F0.5 | 0.6183 | 0.6253 | **0.6403** |
| recall, macro over catalogue | **0.6938** | 0.6887 | 0.6655 |
| precision, macro over catalogue | 0.6182 | 0.6258 | **0.6461** |
| worst measurable priority recall | **0.7221** | **0.7221** | 0.6410 |
| prediction rate | 0.8192 | 0.8181 | 0.8168 |
| document recall / precision / specificity | 0.7975 / 0.8956 / 0.8792 | *identical* | *identical* |
| one-core p95 | 3.9164 ms | 3.9164 ms | 3.9696 ms |

Document-level metrics are identical across all three to four decimal places, and
this was predicted before measuring rather than discovered: `quiet_runtime.predict`
consults the gate **before** scoring any head, and all three arms carry byte-identical
`gate_weights` and gate threshold. Only tag emission can move.

### Both declared policies, unchanged

| policy | v2 | v3 | v4 |
| --- | --- | --- | --- |
| **headline** — per-tag recall ≥ 0.90 on `ci_lower` | blocked, **29/55** | blocked, **29/55** | blocked, **25/55** |
| **precision view** — doc precision/specificity/recall + tag recall ≥ 0.75 | blocked, 45/55 | blocked, 45/55 | blocked, 44/55 |
| one-core p95 ≤ 5 ms | PASS | PASS | PASS |

`NO FEASIBLE ARM` under both, for all three. Nothing was promoted and nothing was
packaged.

### What v4 changed, and what it cost

The corrected cap moved **five** of 58 thresholds. On the calibration carve:

| tag | v2 threshold → v4 | v2 P / R / F0.5 | v4 P / R / F0.5 | v2 → v4 firings |
| --- | --- | --- | --- | --- |
| 27 date of death | −4.13 → 9.56 | 0.4145 / 0.9994 / 0.4694 | **0.9862 / 0.9703 / 0.9830** | 48,834 → 18,895 |
| 44 military ID | −3.40 → 10.03 | 0.2150 / 0.9992 / 0.2550 | **0.9364 / 0.8262 / 0.9121** | 51,997 → 9,321 |
| 47 password | −1.40 → 3.11 | 0.1094 / 0.9732 / 0.1330 | **0.8380 / 0.7772 / 0.8251** | 11,270 → 1,225 |
| 49 phone number | 13.13 → 13.18 | 0.9589 / 0.9228 / 0.9514 | 0.9602 / 0.9215 / 0.9522 | 20,266 → 20,204 |
| 57 zip code | 9.87 → 12.33 | 0.8547 / 0.9548 / 0.8730 | **0.9518 / 0.8722 / 0.9347** | 20,066 → 16,372 |

On the sealed corpora that improvement is paid for in recall, and four priority
gates flip:

| tag @ corpus | support | v2 recall (ci_low) | v4 recall (ci_low) |
| --- | ---: | --- | --- |
| military ID @ `20000_pii_holdout` | 4,327 | 1.0000 (1.0000) | 0.8343 (0.8230) |
| military ID @ `38937_openpii_eval` | 13,763 | 0.9995 (0.9991) | 0.7701 (0.7628) |
| password @ `10626_ai4privacy` | 1,245 | 0.9855 (0.9783) | 0.7173 (0.6916) |
| password @ `20000_pii_holdout` | 232 | 0.9741 (0.9526) | 0.8793 (0.8362) |

All four are PASS→FAIL. No pair moved the other way.

### The finding that matters most

`v3` set tag 44's threshold to **7.4621** by sweeping the sealed evaluation corpora.
The training-only procedure, run blind, chose **10.0263** for the same tag. Both raise
it from −3.40; the difference is where they stop:

| | tag 44 recall @ `20000_pii_holdout` | @ `38937_openpii` | gates held |
| --- | --- | --- | --- |
| v2 (−3.40) | 1.0000 (ci_low 1.0000) | 0.9995 (0.9991) | both |
| **v3 (7.4621, swept on eval)** | 0.9609 (0.955) | 0.9372 (0.933) | **both** |
| v4 (10.0263, training only) | 0.8343 (0.823) | 0.7701 (0.763) | neither |

v3 landed on the one part of the range that captures most of the precision gain
*and* stays above the sealed 0.90 gate. That is not luck and it is not skill — it is
what selecting against the sealed set buys, and it is exactly the advantage the
sealed set exists to deny. The training carve cannot see where the gate sits on
`20000_pii_holdout`, so it overshoots.

## TL;DR

- **v3's two thresholds were selected on the sealed evaluation corpora.** Its own
  `config.json` says so: a per-tag F2 sweep on "a held-in half of the eight
  `data/2-eval` corpora". The script that did it was never checked in, so the split
  cannot be reproduced.
- **v3's artifact claims check out otherwise.** `gate_weights` and `tag_weights` are
  byte-identical to v2; exactly 2 of 58 `tag_thresholds` differ, at indices 44 and 51,
  as its card says. The `weights.npz` size change is a re-save, not a content change.
- **Re-scored sealed through its own runtime, v3 is real but small**: macro F2 0.6614
  → 0.6657, micro F1 0.7255 → 0.7533, and it loses no gates. 96 sampled documents
  re-scored through the bundle's own `predict()` matched the cached path exactly.
- **The root cause was never the two tags.** `group_recall_cap` lets any source group
  with ≥ 20 positives set the cap, and at a 0.7556 floor that is a 5th order
  statistic. One 31-positive slice of `148775_pii2_train` (AUC 0.684) dictated tag
  44's threshold for 10,556 positives, against sibling caps of +10.38 and +11.57.
- **Fixing that properly moves five thresholds, not two** — including date of death,
  password and zip code, which v3 never touched, and which were firing 48,834,
  11,270 and 20,066 times on the calibration carve at precision 0.41, 0.11 and 0.85.
- **v4 wins every precision-priced aggregate and loses the headline anyway**:
  priority macro F0.5 0.7434 → 0.8031, micro F1 → 0.7862, but macro F2 stays flat
  (0.6641) because F2 weights recall four times, and four priority gates flip
  PASS→FAIL. **29/55 → 25/55.**
- **Tag 51 was not fixed**, and neither was it fixable this way: its cap-setting
  group clears the ten-tail-events bar (43 positives → 10.5 in the tail). It is also
  **not measurable on any sealed corpus**, so v3's headline claim for it (F2 0.026 →
  0.444) cannot be confirmed or refuted here at all.
- **Nothing promoted, nothing packaged.** Under the declared headline policy v4 is
  worse than the artifact it corrects. Packaging it would ship a bundle whose card
  had to explain why the honest model scores lower.

---

| | |
| --- | --- |
| Date | 2026-08-26 |
| Project | `projects/pii-head-to-head-v1` |
| Run | `26-08-26-4` |
| Request | *"Re-measure & report v3"*, then *"also re-derive thresholds reproducibly"* |
| Scope | 3 arms × 8 sealed corpora; 5 of 58 thresholds re-derived on the training carve |
| CPU budget | 1 core (the default) for every latency; training carve selection unbudgeted |
| Outcome | v3 audited and confirmed contaminated; v4 built, measured, and **not** shipped |
| Commands | `26-08-26-4_v3-audit-and-v4-thresholds.commands.txt` (29, failures included) |
| Approval | `approvals/threshold-rederivation-v4.json` |
| Tracking | `sqlite:///projects/pii-head-to-head-v1/mlflow.db`, experiment `pii-head-to-head-v1`, parent run `7cc1e84f` — 1 parent + 3 model runs + 24 model×corpus runs. No model registered and no alias moved: nothing here is promotable |
| Workbook | `26-08-26_Experiment-Log.xlsx` — 153 log rows (129 carried forward + 24), 1,135 per-tag rows |

## What was done

### 1. Auditing the v3 artifact

v3 was compared array-by-array against v2 before anything was run. `gate_weights`
(262144, f16) and `tag_weights` (58 × 262144, f16) are identical; `tag_thresholds`
(58, f32) differs in exactly two positions, 44 and 51. `model.json` is byte-identical
(sha256 `36f2c569…`). The bundle's card is accurate on every point it asserts.

Its provenance is not recoverable. The recalibration script is absent from
`training/`, from the project, and from the tree; `run.json`'s last recorded command
predates the bundle's mtime; no `mp run` event covers it. What v3 did is knowable
only from prose it wrote about itself.

### 2. Reproducing the v2 selection

Before changing a rule, the existing one was reproduced: the frozen
`cascade_balanced` weights, the same `carve_holdin` 15% calibration slice of the
training corpora (79,883 rows, 66,867 admitted by the gate), the same cascade trial
337 parameters, the same `select_per_label_robust` semantics.

Reproduction is asserted on the **operating point**, not the raw threshold. The saved
model stores `tag_weights` as float16, so a rebuilt score matrix is slightly coarser
than the full-precision one the original swept, and `sweep` only cuts between
distinct scores — a perturbed matrix can land on an adjacent realisable cut. Raw
threshold deltas reach 7.4e-02 on a scale of 14.35; the **worst shift in precision or
recall across all 57 enabled tags is 6.25e-03** (ΔP +2.19e-03, ΔR −6.25e-03), on
`sensitive_pii_immigration_and_citizenship_status`. The disabled/enabled pattern
matches element for element. The script refuses to emit anything if that check fails.

### 3. The diagnosis

All 57 enabled tags **met** the recall floor. Nothing failed; nothing fell to a
fallback branch. The two tags v3 overrode were selected exactly as the declared rule
asks, and were recorded as successes — tag 44 at precision 0.2150, tag 51 at
precision **0.0051**, firing on 51,997 and 29,069 of 66,867 gate-admitted rows.

The binding constraint is `group_recall_cap`, and both tags sit on it to four
decimals (−3.4008 against a cap of −3.4006; −3.9635 against −3.9634). Per source:

| tag | source | positives | AUC | that source's cap |
| --- | --- | ---: | ---: | ---: |
| 44 | `148775_pii2_train` | **31** | 0.684 | **−3.4006** ← sets it |
| 44 | `151708_openpii_train` | 7,864 | 0.965 | +10.3751 |
| 44 | `85593_pii_trainset` | 2,661 | 0.999 | +11.5725 |
| 51 | `148775_pii2_train` | **43** | 0.874 | **−3.9634** ← sets it |
| 51 | `85593_pii_trainset` | 99 | 0.995 | +2.4259 |

`group_recall_cap` already skips groups it considers too small — its docstring says
the point is not to set the cap "from noise" — but it counts positives anywhere on
the curve, and the quantile it needs sits in the tail. At a 0.7556 floor a
20-positive group places the cut on its 5th order statistic.

### 4. The rule change

A group may set the cap only if at least **10** of its positives fall at or below the
candidate cut — `n · (1 − floor) ≥ 10`, the ordinary ten-events-in-the-tail rule of
thumb. If no group qualifies the cap is `+inf` ("nothing could constrain it") rather
than `−inf`, which as a sentinel would make every threshold inadmissible and hand the
tag to the `argmax(recall)` fallback — the same failure by another route. No tag
takes that path today; it is closed because it is one small corpus away from opening.

`MIN_TAIL_EVENTS = 10` was fixed before any sealed corpus was scored. It is a
statistical rule of thumb, not a derived constant, and it is the one judgement call
in `training/h2h_thresholds_v4.py`.

`training/quiet_select.py` was **not** modified — the 128 existing arm results depend
on its exact semantics — so the corrected rule lives in the new module and the
uncorrected path stays available as the reproduction check.

### 5. Scoring

`training/h2h_score_bundle.py` scores an arbitrary saved cascade, or a packaged
bundle through its own `runtime/`, using the **unchanged** `h2h_eval` evaluator, so a
new arm joins the existing 128 on the same terms. `h2h_score.py`'s A/B/C registry and
`h2h_eval.py` were not touched.

Each arm re-scored 12 documents per corpus (96 total) through the model's own
`predict()` and required exact agreement with the cached feature path. All three
passed with 0 mismatches. v4's latency was measured for itself rather than carried
from arm B.

## What is still open

- **Tag 51 remains broken and unverifiable.** It fires on 29,069 of 66,867
  calibration rows at precision 0.0051, the corrected cap does not move it, and no
  sealed corpus can measure it. Either it needs gold that can judge it or it should
  be disabled outright — a decision, not a tuning step.
- **Whether the headline policy is the right instrument for this trade.** v4 is
  better on every precision-priced aggregate and worse on the gate. That is the
  policy working as declared, not a defect; but a run whose best precision result is
  blocked by a recall gate is the case the contra-view policy exists to surface, and
  both are reported here without one being folded into the other.
- **The 4 flipped gates are a threshold-placement problem, not a capability one.**
  Tag 44 holds both gates at v3's 7.4621 and neither at v4's 10.0263. A selection
  rule with access to a *training-side* proxy for the gate — rather than the sealed
  set — might land in between. Nothing here establishes that it would.
- **`mp decide --arms` is `nargs="+"`, and the previous run was bitten by it.**
  Repeating the flag overwrites rather than accumulates. The first attempt in this
  run silently decided over one arm; re-running with all three in a single flag fixed
  it. The same repeated-flag form produced `decision/headline_balanced.json` on
  2026-08-25 (`--arms arm_B.json --arms arm_cascade_balanced.json`), and that file
  contains **one** arm — `arm-B-cascade_balanced`. Whatever "balanced gate adopted"
  was based on, it was not a comparison recorded in that decision file. Worth
  re-deriving before the balanced-gate result is cited again.
