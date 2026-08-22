# M1 Baseline — rules-only pipeline (v0.2)

Measured results for the Stage 1 rules engine against the frozen corpus and the 5 ms/doc
performance budget (docs/DESIGN.md §5). This is the reference point every later change —
Stage 1 hardening, the Stage 2 model — must beat or explain.

**Regenerate with:**

```console
$ pii-master eval eval/corpus/frozen_v1.jsonl
$ pii-master bench
```

**Environment for the numbers below** (a shared, virtualized container — treat absolute
latencies as indicative; the budget gate runs on comparable hardware in CI):
Intel Xeon @ 2.80 GHz (virtualized, 1 core used), Python 3.11.15, Linux, package v0.2.0,
2026-08-22.

## Latency vs the 5 ms budget (`pii-master bench`, seed 7, 30 docs/bucket)

| bucket | mean | p50 | p95 | max | MB/s | allowance (p95) | verdict |
|---|---|---|---|---|---|---|---|
| 1 KB | 0.15 ms | 0.15 ms | 0.20 ms | 0.22 ms | 6.8 | 5.00 ms | PASS |
| 10 KB | 1.12 ms | 1.12 ms | **1.24 ms** | 1.30 ms | 8.9 | 5.00 ms | **PASS** |
| 100 KB | 13.18 ms | 13.00 ms | 13.94 ms | 16.66 ms | 7.6 | 50.00 ms | PASS |

Peak RSS: **18 MB** (4 GB budget). Stage 1 leaves ≈ 3.7 ms of the 5 ms typical-document
budget for the Stage 2 model layer.

**How we got here:** the first measurement of this same pipeline was **7.92 ms p95 at
10 KB — over the entire budget on its own** (≈1.4 MB/s), dominated by the birth-date
pattern (2.85 ms), phone (1.05 ms), IPv4 (0.71 ms) and card (0.55 ms) patterns scanning
every text position. The fix was pre-scan windows (docs/DESIGN.md §7): C-speed literal
hints for cue-anchored detectors, a shared cached digit-run scan for numeric types, and
a colon-pair seed for IPv6 — a 6.4× end-to-end speedup with pipeline output asserted
identical to full scanning across the frozen corpus
(`tests/test_prescan_equivalence.py`).

## Detection quality (`pii-master eval`, frozen corpus v1: 31 docs, 9 NONE / 12 PII / 10 PHI)

Span-level, exact boundaries (partial-overlap scores are identical on this corpus):

| type | gold | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|---|
| ACCOUNT_NUMBER | 2 | 2 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| ADDRESS | 1 | 0 | 0 | 1 | 0.00 | 0.00 | 0.00 |
| CREDIT_CARD | 2 | 2 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| DATE_DOB | 5 | 5 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| EMAIL | 2 | 2 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| HEALTH_PLAN_ID | 1 | 1 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| IP_ADDRESS | 3 | 3 | 2 | 0 | 0.60 | 1.00 | 0.75 |
| MRN | 3 | 3 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| PERSON_NAME | 2 | 0 | 0 | 2 | 0.00 | 0.00 | 0.00 |
| PHONE_US | 3 | 3 | 1 | 0 | 0.75 | 1.00 | 0.86 |
| SSN | 3 | 3 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| URL | 1 | 1 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| US_DRIVER_LICENSE | 1 | 1 | 0 | 0 | 1.00 | 1.00 | 1.00 |

Document-level: **accuracy 0.94 (29/31), PHI recall 1.00 (10/10)**. The two mislabels
are gold-NONE documents predicted PII.

## Reading these numbers honestly

- The corpus was authored alongside the detectors, so the perfect scores on
  format-anchored types mean "the rules do what they claim on their own hard cases" —
  not that real-world recall is 1.0. Real-document evals (ai4privacy; n2c2 under DUA)
  are the M2 yardstick.
- **The imperfect rows are the roadmap, quantified:**
  - `PERSON_NAME` and `ADDRESS` recall 0.00 — rules cannot find them; this is Stage 2's
    reason to exist (docs/DESIGN.md §8).
  - `IP_ADDRESS` precision 0.60 — four-part version strings (`Build 10.2.1.4`) are
    irreducible FPs at the rules level (corpus doc `none-003`).
  - `PHONE_US` precision 0.75 — bare 10-digit confirmation numbers collide with NANP
    shapes (corpus doc `none-006`).
  - Both FP classes need surrounding-context modeling, i.e. Stage 2.
- The corpus is append-only: these tables may only be re-generated, never pruned to
  look better.
