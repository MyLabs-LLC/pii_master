# pii-master-ner-l 0.3.1

Token tagger for **PII / PHI detection**, distilled from
`kalyan-ks/ettin-68m-nemotron-pii` on [nvidia/Nemotron-PII](https://huggingface.co/datasets/nvidia/Nemotron-PII).
Runs on **one CPU core**.

- dilated depthwise-separable CNN tagger, d=192 x 8 layers, 10.00M parameters
- 111 BIO classes over 55 Nemotron entity types, crosswalked to 25 HIPAA-mapped types
- onnx-fp32, 43.7 MB
- Confidence calibration: **isotonic, per entity type**
- Source commit: `a02891b99f59e83e76b7e65c4485b121e5e6d720`
- Trained on: `nvidia/Nemotron-PII`

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
| deep @0.30 | 0.9980 | 6 of 2,983 | 0.214 |
| **deep @0.50** | **0.9977** | 7 of 2,983 | 0.000 |
| deep @0.70 | 0.9963 | 11 of 2,983 | 0.000 |

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
| the 12 types the rules also cover | 0.918 | **0.927** | 0.940 | 0.962 |
| the 14 types only this model emits | 0.911 | **0.918** | 0.930 | 0.951 |

<details><summary>Per type</summary>

| type | gold | recall | F2 | F1 | precision |
|---|--:|--:|--:|--:|--:|
| `CREDIT_CARD` | 380 | 0.108 | 0.129 | 0.183 | 0.612 |
| `TAX_ID` | 43 | 0.349 | 0.399 | 0.508 | 0.938 |
| `FAX_NUMBER` | 211 | 0.782 | 0.808 | 0.851 | 0.932 |
| `DEVICE_ID` | 80 | 0.787 | 0.818 | 0.869 | 0.969 |
| `GEO_COORDINATE` | 233 | 0.897 | 0.902 | 0.909 | 0.921 |
| `DATE_TIME` | 325 | 0.892 | 0.907 | 0.931 | 0.973 |
| `PERSON_NAME` | 3,019 | 0.909 | 0.913 | 0.918 | 0.928 |
| `ADDRESS` | 1,568 | 0.902 | 0.913 | 0.931 | 0.961 |
| `US_DRIVER_LICENSE` | 170 | 0.912 | 0.916 | 0.923 | 0.934 |
| `USER_ID` | 1,339 | 0.923 | 0.932 | 0.947 | 0.972 |
| `ACCOUNT_NUMBER` | 510 | 0.931 | 0.941 | 0.956 | 0.981 |
| `VEHICLE_ID` | 294 | 0.959 | 0.954 | 0.946 | 0.934 |
| `PHONE_US` | 638 | 0.981 | 0.964 | 0.939 | 0.899 |
| `URL` | 1,198 | 0.968 | 0.967 | 0.964 | 0.960 |
| `BIOMETRIC_ID` | 358 | 0.961 | 0.967 | 0.976 | 0.991 |
| `HEALTH_PLAN_ID` | 311 | 0.968 | 0.972 | 0.979 | 0.990 |
| `BANK_ROUTING` | 277 | 0.968 | 0.972 | 0.980 | 0.993 |
| `SSN` | 249 | 1.000 | 0.976 | 0.941 | 0.889 |
| `MRN` | 389 | 0.974 | 0.979 | 0.986 | 0.997 |
| `IP_ADDRESS` | 291 | 0.979 | 0.981 | 0.985 | 0.990 |
| `MAC_ADDRESS` | 137 | 0.985 | 0.985 | 0.985 | 0.985 |
| `SWIFT_BIC` | 157 | 0.994 | 0.987 | 0.978 | 0.963 |
| `DATE_DOB` | 403 | 0.995 | 0.995 | 0.994 | 0.993 |
| `EMAIL` | 1,221 | 0.998 | 0.997 | 0.997 | 0.997 |

</details>

## Limitations that matter

- **These scores are for Nemotron-PII, and they do NOT generalise.** Measured on
  ai4privacy/pii-masking-300k -- a different corpus, label space, document style
  and locale -- in-scope strict span recall is **0.385** against 0.914 here, and
  document-level recall **0.870** against 0.998. Format-anchored types transfer
  intact (EMAIL 0.943, IP 0.988); learned semantic types collapse (names and
  addresses ~0.30, mostly boundary errors on structured JSON text). Deep mode
  still roughly doubles the rules on that corpus, so the cascade earns its place --
  but budget for the lower number on text unlike the training set.
- **Synthetic training data.** Nemotron-PII is generated, not real, and the
  standard clinical benchmark (n2c2/i2b2 2014) requires a data use agreement and
  was not used. No number here describes real clinical text.
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

The code is MIT (see LICENSE). **The model is a derivative work of its training
data, and inherits obligations from every corpus below.**

- `nvidia/Nemotron-PII` — CC BY 4.0; attribution to NVIDIA is required when
  redistributing this model or its outputs.
- teacher `kalyan-ks/ettin-68m-nemotron-pii` — MIT.

## Verify before you trust it

    python training/package.py verify <this directory>

The weights ship in `model.onnx.data` beside `model.onnx`. A package whose weights
are corrupted **without changing their size** still loads and still answers: on the
real artifact a flipped kilobyte turned an `MRN` into a `USER_ID` at 0.87
confidence, which is a silent PHI miss. Only the checksum catches that.
