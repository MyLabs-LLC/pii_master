# pii-master-ner-m 0.3.1

Token tagger for **PII / PHI detection**, distilled from
`kalyan-ks/ettin-68m-nemotron-pii` on [nvidia/Nemotron-PII](https://huggingface.co/datasets/nvidia/Nemotron-PII).
Runs on **one CPU core**.

- dilated depthwise-separable CNN tagger, d=128 x 6 layers, 6.57M parameters
- 111 BIO classes over 55 Nemotron entity types, crosswalked to 25 HIPAA-mapped types
- onnx-fp32, 30.0 MB
- Confidence calibration: **isotonic, per entity type**
- Source commit: `177dd78fd24a8de589ce7ae16aad975d47083ac3`

## Intended use

Input is plain text; output is character spans typed against the 18
HIPAA Safe Harbor identifier categories. It is designed to run **behind the
Stage 1 rules tier**, which supplies checksum-validated spans it cannot beat,
and which suppresses failure classes it re-introduces on its own (see Limitations).

**Not a de-identification guarantee.** It is a detector that helps a reviewer.
Safe Harbor de-identification is a legal determination this model cannot make.

## How good is it?

### At the level you act on: does this document contain PII?

| configuration | **recall** | documents missed | false alarms |
|---|--:|--:|--:|
| deep @0.30 | 0.9983 | 5 of 2,983 | 0.143 |
| **deep @0.50** | **0.9980** | 6 of 2,983 | 0.000 |
| deep @0.70 | 0.9950 | 15 of 2,983 | 0.000 |

Measured on 3,000 held-out Nemotron documents; a document counts as
sensitive if it carries a gold span of a type we model. False alarms are measured
on 14 **adversarial** negatives -- order numbers, chart numbers,
subscriber ids -- not on easy ones.

### At the span level: is the tag right?

Exact `(type, start, end)` match, fused with the rules tier.
**F2 weights recall four times as heavily as precision**, which is closer to this
system's cost matrix than F1; both are shown.

| | recall | **F2** | F1 | precision |
|---|--:|--:|--:|--:|
| the 12 types the rules also cover | 0.913 | **0.921** | 0.935 | 0.957 |
| the 14 types only this model emits | 0.895 | **0.902** | 0.914 | 0.934 |

<details><summary>Per type</summary>

| type | gold | recall | F2 | F1 | precision |
|---|--:|--:|--:|--:|--:|
| `CREDIT_CARD` | 380 | 0.108 | 0.129 | 0.183 | 0.612 |
| `TAX_ID` | 43 | 0.326 | 0.372 | 0.475 | 0.875 |
| `DEVICE_ID` | 80 | 0.750 | 0.785 | 0.845 | 0.968 |
| `FAX_NUMBER` | 211 | 0.787 | 0.810 | 0.849 | 0.922 |
| `GEO_COORDINATE` | 233 | 0.884 | 0.883 | 0.880 | 0.877 |
| `ADDRESS` | 1,568 | 0.886 | 0.895 | 0.908 | 0.930 |
| `PERSON_NAME` | 3,019 | 0.894 | 0.895 | 0.897 | 0.900 |
| `US_DRIVER_LICENSE` | 170 | 0.894 | 0.904 | 0.918 | 0.944 |
| `USER_ID` | 1,339 | 0.892 | 0.907 | 0.930 | 0.972 |
| `DATE_TIME` | 325 | 0.892 | 0.907 | 0.931 | 0.973 |
| `ACCOUNT_NUMBER` | 510 | 0.925 | 0.934 | 0.947 | 0.969 |
| `HEALTH_PLAN_ID` | 311 | 0.945 | 0.953 | 0.966 | 0.987 |
| `PHONE_US` | 638 | 0.975 | 0.956 | 0.929 | 0.887 |
| `URL` | 1,198 | 0.957 | 0.957 | 0.955 | 0.953 |
| `BIOMETRIC_ID` | 358 | 0.953 | 0.961 | 0.974 | 0.997 |
| `VEHICLE_ID` | 294 | 0.963 | 0.963 | 0.964 | 0.966 |
| `BANK_ROUTING` | 277 | 0.964 | 0.970 | 0.978 | 0.993 |
| `MRN` | 389 | 0.967 | 0.972 | 0.979 | 0.992 |
| `MAC_ADDRESS` | 137 | 0.971 | 0.975 | 0.982 | 0.993 |
| `SSN` | 249 | 1.000 | 0.976 | 0.941 | 0.889 |
| `SWIFT_BIC` | 157 | 0.981 | 0.977 | 0.972 | 0.963 |
| `IP_ADDRESS` | 291 | 0.979 | 0.981 | 0.985 | 0.990 |
| `DATE_DOB` | 403 | 0.998 | 0.996 | 0.994 | 0.990 |
| `EMAIL` | 1,221 | 0.998 | 0.997 | 0.997 | 0.997 |

</details>

## Limitations that matter

- **Synthetic training data.** Nemotron-PII is generated, not real. These scores
  do not transfer unexamined to real clinical text. The standard benchmark
  (n2c2/i2b2 2014) requires a data use agreement and was not used.
- **The demographic slice is synthetic too.** Name recall varies by only 0.020
  across race/ethnicity groups, which sounds excellent and mostly reflects names
  drawn from a generator rather than from the world. It is a real gate that would
  catch a large disparity; it certifies synthetic names only.
- **The PII-vs-PHI split has no external gold.** Nemotron has no document labels
  and no medical-context annotation, so that boundary is only scored on a
  39-document authored corpus. It is the weakest link in the evaluation.
- **Credit card numbers are deliberately suppressed.** 88% of the training
  corpus's cards fail the Luhn checksum, so the serving path re-validates and
  drops them. Measured F1 on that type is ~0.18 against this corpus, and that is
  correct behaviour rather than a defect.
- **US / English scope.** Non-US identifier formats are out of scope.
- **Calibrated for THIS model on THIS corpus.** A threshold tuned here is not
  automatically right for another text distribution.
- **It can be confidently wrong.** Adversarial near-misses are its documented
  failure class, which is why the serving path ships a confidence floor, checksum
  re-validation and rule fusion. Do not run it bare.

## Licensing

Code MIT (see LICENSE). Trained on Nemotron-PII, **CC BY 4.0**: attribution to
NVIDIA is required when redistributing this model or its outputs. The teacher
`kalyan-ks/ettin-68m-nemotron-pii` is MIT.

## Verify before you trust it

    python training/package.py verify <this directory>

The weights ship in `model.onnx.data` beside `model.onnx`. A package whose weights
are corrupted **without changing their size** still loads and still answers: on the
real artifact a flipped kilobyte turned an `MRN` into a `USER_ID` at 0.87
confidence, which is a silent PHI miss. Only the checksum catches that.
