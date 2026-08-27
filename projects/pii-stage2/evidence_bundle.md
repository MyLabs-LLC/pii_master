# Evidence bundle — iteration 0

Output of the **Weakness Mining** stage. Cluster failures by failure signature —
`(terminal metric cause, per-class/per-family gap, inferred mechanism)` — order by
support × addressability, and name only what is failing and why. Do **not**
prescribe the edit here; that is the Propose stage.

- **Harness under measurement:** `h0` (Stage 1 rules engine, commit `69bf258`)
- **Evaluator:** span exact-match protocol frozen in `eval/scripts/nemotron_eval.py`, plus `model_pipeline.evaluate("tagging")` on per-doc type sets; span `macro_f2` / `severity_recall_min` via `projects/pii-stage2/scripts/gate_metrics.py` (the engine has no span-NER family)
- **Splits:** held-in `D_in` = Nemotron-PII train (100k); sealed held-out `D_ho` = Nemotron-PII test (100k); frozen_v1 (39) is a tautological regression test, not a quality claim
- **Current score (rules `h0`):** in macro_f2 = **0.614**, ho macro_f2 = **0.605**, ho severity_recall_min = **0.00033** (US_DRIVER_LICENSE), target = macro_f2 ≥ 0.92 ∧ severity_recall_min ≥ 0.90
- **Already-trained challenger (not in the package):** `cnn_m` + checksum-first fusion + `--revalidate` on `D_ho`: macro_f2 = **0.863**, severity_recall_min = **0.907**. Micro-F1 0.889 reproduces the committed distillation table (0.901 was without revalidate on invalid cards).

## Clustered failure patterns

| # | Cluster (failure signature) | Support | Representative case(s) | Inferred mechanism | Addressable? |
|---|---|---|---|---|---|
| 1 | severity_recall_min — `US_DRIVER_LICENSE` recall 0.0003 | 3,001 FN / 3,002 gold on `D_ho` | Nemotron cue is literally "certificate license number"; rules only accept driver's-licence wording | cue narrowing, HIPAA #11 | yes (student already F1 0.886; also Track C umbrella) |
| 2 | cue-anchored recall gap — ACCOUNT / MRN / HEALTH_PLAN | FN 12.4k / 6.0k / 7.2k | cue-free IDs; rules fire only next to a cue | formatless IDs without a label | yes (student recall 0.91–0.94 under fusion) |
| 3 | `CREDIT_CARD` F2 0.142 (full gold) | 11,345 FN of 12,867; only 1,525 Luhn-valid | 88% of Nemotron card gold fails Luhn; detector requires Luhn (0.998 recall on the valid subset) | dataset artifact, not a miss | no — do not loosen Luhn; do not drop the class from macro_f2 (that would edit the evaluator) |
| 4 | `PHONE_US` precision 0.58 under fusion (F2 0.85, recall 0.96) | 16,478 FP | fax numbers, biometric-shaped ids, unmodelled overlap (BASELINE_NEMOTRON triage) | taxonomy gap + NANP-shaped fax | yes (Track C `FAX_NUMBER`; stop filling cue-less rule phones into fusion) |
| 5 | package gap — measured student is not `scan_text` | n/a | `OnnxNerDetector` is a checkbox in DISTILLATION_PLAN; frozen rehearsal showed false PHI without threshold + revalidate | serving path missing | yes (integrate with checksum-first + revalidate + confidence floor) |
| 6 | tokenizer/decode dominate `deep` e2e (5.21 ms `xs` cascade) | n/a | DESIGN §8 windows never implemented | cost scales with full document | yes, later — `deep` already fits 25 ms; not the F2 gap |
| 7 | names/addresses recall 0 on frozen | 3 undetectable spans | no `PERSON_NAME`/`ADDRESS` in the 11-type catalogue | Stage 2 native 55-label upside, not the current gate | no this round (catalogue is the 11 mapped types) |

## Notes

- Micro-F1 0.788 vs macro_f2 0.605 on rules is the expected head/tail gap: EMAIL/URL/DOB/IP carry micro; LICENSE/ACCOUNT/PLAN are the tail.
- Fusion checksum-first already clears the **ship** gate (severity_recall_min 0.907 ≥ 0.90) and misses the **progress** target (macro_f2 0.863 < 0.92). CREDIT_CARD's unreachable gold is a hard drag on the mean; PHONE precision is the largest *addressable* remaining drag.
- `cnn_m` student alone: severity_recall_min 0.909, macro_f2 0.851. Fusion wins F2 (+0.012) by keeping checksum rules. Do not ship student-alone.
- `fusion_rules_first` drops severity_recall_min to 0.675 — cue-anchored rules must not outrank the model. Confirms DISTILLATION_RESULTS §5.
- Frozen rules: doc accuracy 1.00, PHI recall 1.00. Any integration that reopens `none-007` false PHI is a reject regardless of Nemotron F2.
