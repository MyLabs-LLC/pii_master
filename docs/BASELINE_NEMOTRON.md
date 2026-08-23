# External baseline — Nemotron-PII holdout

Measured with `eval/scripts/nemotron_eval.py` against
[`nvidia/Nemotron-PII`](https://huggingface.co/datasets/nvidia/Nemotron-PII) (CC BY 4.0), data
nobody in this repo authored. **This is the honest number**; the frozen corpus in
`eval/corpus/` is a regression test, not a quality claim.

Split: `test` · documents scored: **100,000** · source: test-00000-of-00001.parquet

> **This is the RULES-ONLY baseline, and its scope did not change at v0.3.**
> That release adopted 22 more Nemotron labels for the Stage 2 model, which
> would have widened the denominator here and dropped every number for a reason
> that has nothing to do with the rules changing — no regex emits a
> `PERSON_NAME`. `eval/scripts/nemotron_eval.py` is therefore pinned to
> `crosswalk.RULE_MAPPED`, the same 12 labels scored below, so this document
> stays comparable across releases.
>
> The deep cascade's number on the same data is in
> [STAGE2_INTEGRATION.md](STAGE2_INTEGRATION.md) §7.1: **F1 0.933 on these
> types against 0.788 here**, plus 0.893 on fourteen types this baseline
> cannot score at all.

## Scope

Only the **12 mapped labels** (`pii_master/crosswalk.py`) are scored: 203,283 gold spans.
Gold spans of the **43 unmodelled labels** (647,057 spans) are dropped, not counted as misses — scoring against categories
we deliberately do not detect would measure the crosswalk, not the detectors.
That omission is the Stage 2 / Track C backlog, sized in docs/NEMOTRON_PII_TAGS.md.

## Span-level results (mapped types only)

| type | gold | TP | FP | FN | P | R | F1 | partial R |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| `ACCOUNT_NUMBER` | 16,693 | 4,311 | 184 | 12,382 | 0.959 | 0.258 | 0.407 | 0.267 |
| `CREDIT_CARD` | 12,867 | 1,522 | 556 | 11,345 | 0.732 | 0.118 | 0.204 | 0.118 |
| `DATE_DOB` | 18,079 | 17,409 | 190 | 670 | 0.989 | 0.963 | 0.976 | 0.963 |
| `EMAIL` | 53,930 | 53,577 | 188 | 353 | 0.997 | 0.993 | 0.995 | 0.995 |
| `HEALTH_PLAN_ID` | 10,558 | 3,319 | 951 | 7,239 | 0.777 | 0.314 | 0.448 | 0.404 |
| `IP_ADDRESS` | 9,217 | 8,911 | 211 | 306 | 0.977 | 0.967 | 0.972 | 0.967 |
| `MRN` | 11,098 | 5,056 | 297 | 6,042 | 0.945 | 0.456 | 0.615 | 0.482 |
| `PHONE_US` | 23,930 | 13,599 | 20,188 | 10,331 | 0.402 | 0.568 | 0.471 | 0.615 |
| `SSN` | 6,062 | 6,040 | 1,525 | 22 | 0.798 | 0.996 | 0.886 | 0.997 |
| `URL` | 37,847 | 35,094 | 1,226 | 2,753 | 0.966 | 0.927 | 0.946 | 0.947 |
| `US_DRIVER_LICENSE` | 3,002 | 1 | 1 | 3,001 | 0.500 | 0.000 | 0.001 | 0.000 |
| **micro-average** | 203,283 | 148,839 | 25,517 | 54,444 | **0.854** | **0.732** | **0.788** | |

## What the "false positives" actually are

Strict precision counts every non-exact prediction against us. That is
misleading here: most of those predictions land on a **real identifier** whose
Nemotron label we do not model, so its gold was withheld from the denominator.
Triaging all 174,356 predictions:

| bucket | count | share | meaning |
|---|--:|--:|---|
| exact hit | 148,839 | 85.4% | correct type and boundaries |
| mapped_mismatch | 8,991 | 5.2% | overlaps mapped gold, wrong edges or type — a real error |
| unmodelled_overlap | 15,422 | 8.8% | a genuine identifier of a label we do not model — a taxonomy gap |
| **spurious** | **1,104** | **0.6%** | **overlaps no gold at all — the only true false alarms** |

**Adjusted micro-precision (excluding unmodelled_overlap): 0.936**, adjusted F1 0.822.

Where the unmodelled overlaps go — each row is a missing type, and the top
rows are exactly the Track C backlog:

| our type | actual Nemotron label | count |
|---|---|--:|
| `PHONE_US` | `biometric_identifier` | 5,548 |
| `PHONE_US` | `fax_number` | 4,018 |
| `PHONE_US` | `customer_id` | 3,632 |
| `SSN` | `national_id` | 419 |
| `SSN` | `tax_id` | 408 |
| `CREDIT_CARD` | `device_identifier` | 294 |
| `PHONE_US` | `unique_id` | 258 |
| `DATE_DOB` | `date` | 172 |
| `PHONE_US` | `national_id` | 124 |
| `URL` | `first_name` | 114 |
| `PHONE_US` | `tax_id` | 90 |
| `CREDIT_CARD` | `national_id` | 63 |

## Reading these numbers

**CREDIT_CARD recall is a dataset artifact, not a detector failure.** Only 1,525 of the 12,867 gold card spans (11.9%) satisfy the Luhn
checksum; the rest are card-shaped digit strings that no real payment network
would issue. Our detector requires Luhn — that is what makes it our
highest-precision rule — so most of this gold is unreachable by design.
**Against the Luhn-valid subset, recall is 0.998.** Keep the check.

**US_DRIVER_LICENSE recall is a genuine, cheap miss.** Nemotron's cue is
literally "certificate license number"; our detector only accepts driver's-licence
wording, so it fires on almost none of them. This is the Track C
`LICENSE_NUMBER` umbrella (HIPAA #11) arriving as a measured number.

**ACCOUNT_NUMBER / MRN / HEALTH_PLAN_ID recall (0.26-0.46) is the cue-anchoring
trade-off working as designed.** Those detectors only fire next to a cue word;
cue-free instances are exactly what Stage 2 is for. Precision on them stays
high (0.78-0.96), which is the half of the bargain we bought.

**PHONE_US** loses recall to international numbers (we validate NANP) and
precision to fax numbers, which share the shape and have no type yet.

## Document-level reach

Of 85,852 documents containing at least one mapped gold span,
we flagged **79,212** (92.3%) with at least one entity.
Nemotron carries no document label, so NONE/PII/PHI accuracy is not evaluable here.

## US vs international

| type | US recall | intl recall |
|---|--:|--:|
| `ACCOUNT_NUMBER` | 0.260 | 0.256 |
| `CREDIT_CARD` | 0.119 | 0.117 |
| `DATE_DOB` | 0.965 | 0.961 |
| `EMAIL` | 0.995 | 0.993 |
| `HEALTH_PLAN_ID` | 0.315 | 0.314 |
| `IP_ADDRESS` | 0.968 | 0.966 |
| `MRN` | 0.443 | 0.473 |
| `PHONE_US` | 0.995 | 0.163 |
| `SSN` | 0.997 | 0.995 |
| `URL` | 0.930 | 0.924 |
| `US_DRIVER_LICENSE` | 0.000 | 0.000 |

Locale gaps are expected, not bugs: `PHONE_US` validates NANP and rejects
international numbers, and `SSN` has no international analogue (Nemotron tags
those `national_id`, which we do not model).

## Reproducing

```console
$ pip install pyarrow   # analysis only, not a runtime dependency
$ python eval/scripts/nemotron_eval.py --data-dir <dir-with-parquet> \
    --out docs/BASELINE_NEMOTRON.md
```
