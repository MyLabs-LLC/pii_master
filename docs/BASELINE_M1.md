# Internal baseline — frozen corpus (v0.2.1, post-M1.5)

Measured results for the Stage 1 rules engine against the **frozen corpus** and the
5 ms/doc budget (docs/DESIGN.md §5). Regenerate — never hand-edit:

```console
$ pii-master eval eval/corpus/frozen_v1.jsonl
$ pii-master bench
```

> **These numbers are a regression test, not a quality claim.** The corpus was authored
> alongside the detectors, so perfect scores mean "the rules still do what they claim on
> their own hard cases". The honest, external number lives in
> **docs/BASELINE_NEMOTRON.md** (micro P 0.854 / R 0.732 on 100k held-out
> Nemotron-PII documents). Read that one first.

Environment (shared virtualized container; treat absolute latency as indicative):
Intel Xeon @ 2.80 GHz, 1 core used, Python 3.11.15, Linux, package v0.2.0, 2026-08-22.

## Latency vs the 5 ms budget (seed 7, 30 docs/bucket)

| bucket | mean | p50 | p95 | max | MB/s | allowance | verdict |
|---|---|---|---|---|---|---|---|
| 1 KB | 0.14 ms | 0.14 ms | **0.17 ms** | 0.18 ms | 7.3 | 5.00 ms | PASS |
| 10 KB | 0.83 ms | 0.80 ms | **0.93 ms** | 1.07 ms | 12.1 | 5.00 ms | PASS |
| 100 KB | 9.13 ms | 8.98 ms | **9.47 ms** | 11.03 ms | 11.0 | 50.00 ms | PASS |

Peak RSS: **18 MB** of the 4 GB budget. Stage 1 leaves ≈ 4.2 ms of
the 5 ms typical-document budget for the Stage 2 model layer.

**History.** The first measurement of this pipeline was **7.92 ms p95 at 10 KB** — over the
whole budget on rules alone. Pre-scan windows (§7) took it to 1.24 ms, and the Track A
switch from 27 substring scans to one compiled word-bounded regex took it to its current
figure. Every step is asserted behaviour-preserving by
`tests/test_prescan_equivalence.py`.

## Detection quality (frozen corpus: 39 docs)

| type | gold | TP | FP | FN | P | R | F1 |
|---|--:|--:|--:|--:|--:|--:|--:|
| `ACCOUNT_NUMBER` | 2 | 2 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| `ADDRESS` | 1 | 0 | 0 | 1 | 0.00 | 0.00 | 0.00 |
| `CREDIT_CARD` | 2 | 2 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| `DATE_DOB` | 5 | 5 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| `EMAIL` | 2 | 2 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| `HEALTH_PLAN_ID` | 2 | 2 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| `IP_ADDRESS` | 3 | 3 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| `MRN` | 4 | 4 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| `PERSON_NAME` | 2 | 0 | 0 | 2 | 0.00 | 0.00 | 0.00 |
| `PHONE_US` | 3 | 3 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| `SSN` | 4 | 4 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| `URL` | 1 | 1 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| `US_DRIVER_LICENSE` | 1 | 1 | 0 | 0 | 1.00 | 1.00 | 1.00 |

Document-level: **accuracy 1.00 (39/39),
PHI recall 1.00**.

### Error taxonomy

| class | count |
|---|--:|
| `undetectable` | 3 |
| `boundary` | 0 |
| `type_confusion` | 0 |
| `context_miss` | 0 |
| `spurious` | 0 |

Everything that remains is `undetectable` — gold `PERSON_NAME` and `ADDRESS` spans that no
rule can reach. That is the Stage 2 mandate, stated as a measured number rather than a
promise.

## What changed since M1 (v0.2.0 → v0.2.1)

Track A of docs/IMPROVEMENT_PLAN.md closed three reproducible **false PHI** (loose `chart`
cue, unqualified `subscriber id`, and a substring medical-context scan where `impatient`
matched `patient`) and added lexical suppressors for the two documented false-PII classes
(version strings read as IPv4, reference numbers read as phones).

Effect on this corpus, which also grew 31 → 39 documents with those cases:

| metric | M1 (v0.2.0) | now |
|---|--:|--:|
| document accuracy | 0.94 | 1.00 |
| `IP_ADDRESS` precision | 0.60 | 1.00 |
| `PHONE_US` precision | 0.75 | 1.00 |
| PHI recall | 1.00 | 1.00 |
| 10 KB p95 latency | 1.24 ms | 0.93 ms |

The corpus is append-only: these tables may be regenerated, never pruned to look better.
`eval/corpus/frozen_v1.scores.json` freezes them as a CI gate, so a future change that
lowers any cell fails the build.
