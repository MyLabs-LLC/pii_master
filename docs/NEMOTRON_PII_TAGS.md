# Nemotron-PII — complete tag inventory

Extracted from [`nvidia/Nemotron-PII`](https://huggingface.co/datasets/nvidia/Nemotron-PII)
(CC BY 4.0, 200,000 rows = 100k train + 100k test, 1,675,796 labelled spans), by enumerating
every distinct `spans[].label` value across both splits. The dataset card advertises "55+
categories"; the data contains **exactly 55 distinct labels**, all listed below.

Regenerate with `eval/scripts/nemotron_tags.py` (see bottom).

## Why this matters here

This is the strongest openly-licensed candidate for Stage 2 training data (docs/DESIGN.md §8):
it is synthetic and commercially usable (unlike n2c2, which needs a DUA), it is span-annotated
for token classification, and it covers PHI as well as PII. Its label set is also a useful
external check on our taxonomy — see the crosswalk.

## Record shape

```
uid, domain, document_type, document_description,
document_format : structured | unstructured
locale          : us | intl
text            : the document
spans           : [{"start": int, "end": int, "text": str, "label": str}]   # Python-repr string, not JSON
text_tagged     : inline form, e.g. [Jason]first_name
```
Note the `spans` column is a *string* holding a Python literal (single quotes), so it needs
`ast.literal_eval`, not `json.loads`.

## All 55 labels

Ordered by span frequency. `docs` = how many of the 200,000 documents contain the label.
HIPAA = Safe Harbor identifier category (45 CFR 164.514(b)(2)); `—` = not a Safe Harbor
identifier (sensitive attribute or credential rather than a direct identifier).

| # | label | spans | % | docs | us / intl | HIPAA | ours | examples |
|--:|---|--:|--:|--:|---|:--:|---|---|
| 1 | `date` | 150,997 | 9.01 | 58,477 | 72,089 / 78,908 | #3 | — | `2023-08-15`, `November 15, 2023` |
| 2 | `first_name` | 143,434 | 8.56 | 46,787 | 74,427 / 69,007 | #1 | — | `Jason`, `Patricia` |
| 3 | `company_name` | 140,278 | 8.37 | 42,571 | 66,725 / 73,553 | — | — | `SwiftFlow Logistics`, `VeriTrust ID` |
| 4 | `email` | 103,864 | 6.20 | 56,477 | 41,751 / 62,113 | — | `EMAIL` | `boycec1971@gmail.com`, `raulr1968@hotmail.com` |
| 5 | `last_name` | 103,558 | 6.18 | 39,196 | 53,015 / 50,543 | #1 | — | `Boyce`, `Riggio` |
| 6 | `url` | 82,650 | 4.93 | 44,132 | 44,230 / 38,420 | — | `URL` | `https://financialreports.com` |
| 7 | `occupation` | 67,993 | 4.06 | 30,738 | 36,766 / 31,227 | — | — | `physical therapist`, `Scale Operator` |
| 8 | `country` | 51,256 | 3.06 | 27,795 | 21,697 / 29,559 | — | — | `USA`, `U.S.` |
| 9 | `phone_number` | 48,942 | 2.92 | 27,022 | 23,862 / 25,080 | — | `PHONE_US` | `707-859-9753`, `931-608-0499` |
| 10 | `time` | 45,560 | 2.72 | 25,908 | 24,128 / 21,432 | #3 | — | `10:30 AM`, `07:23 AM` |
| 11 | `customer_id` | 37,713 | 2.25 | 23,815 | 16,894 / 20,819 | #18 | — | `SFO-5813924`, `5213804976` |
| 12 | `city` | 37,660 | 2.25 | 20,089 | 17,785 / 19,875 | #2 | — | `Manchester`, `Newnan` |
| 13 | `state` | 37,385 | 2.23 | 20,653 | 21,647 / 15,738 | #2 | — | `MD`, `New Hampshire` |
| 14 | `street_address` | 33,709 | 2.01 | 20,413 | 19,638 / 14,071 | #2 | — | `87 Avenida De La Estrella`, `205 Avalon Avenue` |
| 15 | `date_of_birth` | 30,789 | 1.84 | 18,844 | 12,263 / 18,526 | — | `DATE_DOB` | `1987-05-22`, `1961-05-25` |
| 16 | `user_name` | 28,672 | 1.71 | 14,036 | 13,636 / 15,036 | #18 | — | `daxaben_coder`, `rachel.milliner` |
| 17 | `date_time` | 27,589 | 1.65 | 16,122 | 11,734 / 15,855 | #3 | — | `2025-08-01T05:24:48`, `2024-06-15T14:30:00` |
| 18 | `account_number` | 26,691 | 1.59 | 14,950 | 14,456 / 12,235 | — | `ACCOUNT_NUMBER` | `87352916`, `C37529641` |
| 19 | `biometric_identifier` | 22,567 | 1.35 | 17,143 | 11,356 / 11,211 | #16 | — | `BIO-5728946135`, `D64837291598` |
| 20 | `credit_debit_card` | 21,916 | 1.31 | 16,034 | 11,397 / 10,519 | — | `CREDIT_CARD` | `5147 6232 9417 8653`, `4916 7832 6401 9872` |
| 21 | `employee_id` | 20,945 | 1.25 | 11,267 | 11,194 / 9,751 | #18 | — | `23-MKT-562`, `MKT-3912` |
| 22 | `education_level` | 20,011 | 1.19 | 12,453 | 11,005 / 9,006 | — | — | `high school`, `bachelor's degree` |
| 23 | `employment_status` | 18,672 | 1.11 | 14,164 | 9,613 / 9,059 | — | — | `full-time`, `part-time` |
| 24 | `medical_record_number` | 18,193 | 1.09 | 9,897 | 10,881 / 7,312 | — | `MRN` | `0009271658`, `0004372819` |
| 25 | `health_plan_beneficiary_number` | 16,903 | 1.01 | 12,916 | 8,587 / 8,316 | — | `HEALTH_PLAN_ID` | `PA-0004382965`, `4F92-KL7-PT39` |
| 26 | `race_ethnicity` | 16,378 | 0.98 | 11,913 | 7,623 / 8,755 | — | — | `spanish`, `puerto rican` |
| 27 | `coordinate` | 16,097 | 0.96 | 9,545 | 8,678 / 7,419 | #2 | — | `28.6432 N, 81.2886 W`, `39.2904, -76.6122` |
| 28 | `password` | 15,270 | 0.91 | 10,791 | 7,855 / 7,415 | — | — | `G9t$fR2mXk5`, `Michael1995` |
| 29 | `language` | 15,182 | 0.91 | 10,821 | 6,634 / 8,548 | — | — | `English`, `Spanish` |
| 30 | `bank_routing_number` | 15,113 | 0.90 | 11,644 | 7,852 / 7,261 | #10 | — | `641100487`, `684106238` |
| 31 | `ipv4` | 15,032 | 0.90 | 8,156 | 7,893 / 7,139 | — | `IP_ADDRESS` | `12.34.56.78`, `186.204.24.15` |
| 32 | `county` | 14,927 | 0.89 | 11,341 | 14,581 / 346 | #2 | — | `Todd County`, `Laurens County` |
| 33 | `postcode` | 14,309 | 0.85 | 9,932 | 8,248 / 6,061 | #2 | — | `33578`, `93710` |
| 34 | `vehicle_identifier` | 13,900 | 0.83 | 6,757 | 6,646 / 7,254 | #12 | — | `VF3FJ7X14G0001234`, `5UXKL72M4KL501245` |
| 35 | `fax_number` | 13,780 | 0.82 | 11,482 | 6,767 / 7,013 | #5 | — | `854-501-5730`, `479-772-7297` |
| 36 | `pin` | 13,004 | 0.78 | 10,256 | 6,486 / 6,518 | #10 | — | `513345`, `334728` |
| 37 | `gender` | 11,949 | 0.71 | 6,855 | 6,644 / 5,305 | — | — | `male`, `female` |
| 38 | `swift_bic` | 11,380 | 0.68 | 10,073 | 5,551 / 5,829 | #10 | — | `GHTSUS4KX8L`, `QWERTUS45JHW` |
| 39 | `age` | 11,329 | 0.68 | 8,990 | 6,964 / 4,365 | #3 (>89) | — | `44`, `22` |
| 40 | `license_plate` | 11,212 | 0.67 | 6,368 | 5,999 / 5,213 | #12 | — | `FJL421`, `FTR 832` |
| 41 | `ssn` | 10,999 | 0.66 | 7,883 | 7,151 / 3,848 | — | `SSN` | `250-38-8116`, `547-44-7315` |
| 42 | `religious_belief` | 10,997 | 0.66 | 8,341 | 5,448 / 5,549 | — | — | `Christian`, `Lutheran` |
| 43 | `certificate_license_number` | 10,922 | 0.65 | 8,158 | 9,145 / 1,777 | — | `US_DRIVER_LICENSE` | `FL-72903246`, `87532917` |
| 44 | `api_key` | 10,753 | 0.64 | 5,289 | 5,186 / 5,567 | — | — | `d4a6b9c1-3e2f-4f1a-a76d-3a8c` |
| 45 | `http_cookie` | 10,678 | 0.64 | 8,848 | 5,432 / 5,246 | #18 | — | `user_session` |
| 46 | `political_view` | 10,415 | 0.62 | 7,867 | 4,828 / 5,587 | — | — | `MAGA`, `Tea Party member` |
| 47 | `mac_address` | 10,346 | 0.62 | 8,146 | 5,032 / 5,314 | #18 | — | `00:24:81:A7:B2:F9`, `79:3E:48:1A:6F:B2` |
| 48 | `blood_type` | 10,056 | 0.60 | 7,352 | 5,215 / 4,841 | — | — | `O positive`, `B+` |
| 49 | `cvv` | 8,397 | 0.50 | 6,774 | 4,411 / 3,986 | #10 | — | `775`, `144` |
| 50 | `ipv6` | 7,933 | 0.47 | 4,672 | 4,150 / 3,783 | — | `IP_ADDRESS` | `2a02:202:200e::/64`, `2001:0db8:85a3::8a2e:0370:7334` |
| 51 | `device_identifier` | 7,762 | 0.46 | 4,078 | 4,079 / 3,683 | #13 | — | `a8f8a8f8f3d2a1e2`, `WX76P7X2MZ8D` |
| 52 | `national_id` | 6,018 | 0.36 | 5,679 | 0 / 6,018 | #18 | — | `3004347698`, `82975436185` |
| 53 | `sexuality` | 5,682 | 0.34 | 4,106 | 2,912 / 2,770 | — | — | `asexual`, `straight` |
| 54 | `unique_id` | 4,164 | 0.25 | 2,713 | 2,105 / 2,059 | #18 | — | `5f9f1b9b8c6e8b7a3d2f4b7a`, `6725983140` |
| 55 | `tax_id` | 3,865 | 0.23 | 2,725 | 1,929 / 1,936 | #18 | — | `348-62-1975`, `301-23-6548` |

Locale-bound labels worth noting: `national_id` is **intl-only** (0 US spans — the US
equivalent is tagged `ssn`), and `county` is effectively US-only (14,581 us / 346 intl).

## Crosswalk to the pii_master taxonomy

**12 of 55 Nemotron labels map onto our 11 entity types,
covering 394,834 of 1,675,796 spans (23.6%).**
The remaining 43 labels (76.4% of spans) have no detector today.

### Covered

| pii_master type | Nemotron label(s) | spans |
|---|---|--:|
| `ACCOUNT_NUMBER` | `account_number` | 26,691 |
| `CREDIT_CARD` | `credit_debit_card` | 21,916 |
| `DATE_DOB` | `date_of_birth` | 30,789 |
| `EMAIL` | `email` | 103,864 |
| `HEALTH_PLAN_ID` | `health_plan_beneficiary_number` | 16,903 |
| `IP_ADDRESS` | `ipv4`, `ipv6` | 22,965 |
| `MRN` | `medical_record_number` | 18,193 |
| `PHONE_US` | `phone_number` | 48,942 |
| `SSN` | `ssn` | 10,999 |
| `URL` | `url` | 82,650 |
| `US_DRIVER_LICENSE` | `certificate_license_number` | 10,922 |

Caveats on three of these mappings:
- `certificate_license_number` → `US_DRIVER_LICENSE` is **narrower on our side**: the Nemotron
  label covers any professional/certificate licence (HIPAA #11), not just driver's licences.
  A rename to `LICENSE_NUMBER` would match the real category.
- `phone_number` → `PHONE_US` includes international numbers our NANP-validated detector rejects.
- `account_number` and `health_plan_beneficiary_number` are cue-free in much of this data,
  whereas our detectors require a cue phrase — expect low recall until Stage 2.

### Gaps, by span volume

| Nemotron label | spans | % | HIPAA | note |
|---|--:|--:|:--:|---|
| `date` | 150,997 | 9.01 | #3 | generic dates; distinct from `date_of_birth` (deferred in v1 by design) |
| `first_name` | 143,434 | 8.56 | #1 | Stage 2 target — rules cannot do names |
| `company_name` | 140,278 | 8.37 | — | organisation, not an individual identifier |
| `last_name` | 103,558 | 6.18 | #1 | Stage 2 target |
| `occupation` | 67,993 | 4.06 | — | quasi-identifier |
| `country` | 51,256 | 3.06 | — | geography above state level |
| `time` | 45,560 | 2.72 | #3 | time of day |
| `customer_id` | 37,713 | 2.25 | #18 | cue-free ids |
| `city` | 37,660 | 2.25 | #2 | HIPAA #2 geography — cheap win |
| `state` | 37,385 | 2.23 | #2 | HIPAA #2 geography |
| `street_address` | 33,709 | 2.01 | #2 | Stage 2 target |
| `user_name` | 28,672 | 1.71 | #18 | handles/logins |
| `date_time` | 27,589 | 1.65 | #3 | ISO timestamps — trivial regex add |
| `biometric_identifier` | 22,567 | 1.35 | #16 | prefixed ids (BIO-…) |
| `employee_id` | 20,945 | 1.25 | #18 | cue-free ids |
| `education_level` | 20,011 | 1.19 | — | sensitive attribute |
| `employment_status` | 18,672 | 1.11 | — | sensitive attribute |
| `race_ethnicity` | 16,378 | 0.98 | — | special-category (GDPR Art. 9) |
| `coordinate` | 16,097 | 0.96 | #2 | lat/long — regex-able |
| `password` | 15,270 | 0.91 | — | credential — high-value for secret scanning |
| `language` | 15,182 | 0.91 | — | attribute |
| `bank_routing_number` | 15,113 | 0.90 | #10 | 9-digit ABA, has a checksum |
| `county` | 14,927 | 0.89 | #2 | HIPAA #2 geography |
| `postcode` | 14,309 | 0.85 | #2 | HIPAA #2 — Safe Harbor keeps only first 3 digits |
| `vehicle_identifier` | 13,900 | 0.83 | #12 | VIN, has a checksum |
| `fax_number` | 13,780 | 0.82 | #5 | HIPAA #5 — same shape as phone, separate category |
| `pin` | 13,004 | 0.78 | #10 | credential |
| `gender` | 11,949 | 0.71 | — | attribute |
| `swift_bic` | 11,380 | 0.68 | #10 | fixed 8/11-char format — regex-able |
| `age` | 11,329 | 0.68 | #3 (>89) | Safe Harbor only for >89 |
| `license_plate` | 11,212 | 0.67 | #12 | HIPAA #12 |
| `religious_belief` | 10,997 | 0.66 | — | special-category |
| `api_key` | 10,753 | 0.64 | — | credential — secret scanning |
| `http_cookie` | 10,678 | 0.64 | #18 | session token |
| `political_view` | 10,415 | 0.62 | — | special-category |
| `mac_address` | 10,346 | 0.62 | #18 | regex-able |
| `blood_type` | 10,056 | 0.60 | — | health attribute |
| `cvv` | 8,397 | 0.50 | #10 | credential |
| `device_identifier` | 7,762 | 0.46 | #13 | HIPAA #13 |
| `national_id` | 6,018 | 0.36 | #18 | intl only |
| `sexuality` | 5,682 | 0.34 | — | special-category |
| `unique_id` | 4,164 | 0.25 | #18 | opaque hashes/ids |
| `tax_id` | 3,865 | 0.23 | #18 | EIN/ITIN — SSN-shaped |

## Findings that change our plans

1. **Our birth-cue design holds up.** 96.4% of the 18,079 `date_of_birth` spans in the test
   split have a birth cue (`DOB`, `born`, `date of birth`) within the preceding 40 characters —
   the window our `DateOfBirthDetector` uses. The 3.6% it cannot reach are almost entirely
   **table cells**, where the cue is a column header rather than adjacent text. That is a
   structured-document problem (M4), not a regex-tuning problem.
2. **Cheap Stage 1 wins are visible in the gap table**: `date_time`, `coordinate`, `mac_address`,
   `swift_bic`, `bank_routing_number` (ABA checksum) and `vehicle_identifier` (VIN checksum) are
   all format-anchored and checksum-verifiable — the same shape as detectors we already have.
3. **`fax_number` deserves its own type.** HIPAA lists fax (#5) separately from phone (#4), and
   Nemotron has 13,780 of them; today we would emit `PHONE_US` and lose the distinction.
4. **Credential types** (`password`, `api_key`, `cvv`, `pin`, `http_cookie`) are outside our
   current PII/PHI framing but are exactly what `--fail-on-detect` users want in CI.
5. **Names and addresses are 47% of the gap by volume** (`first_name`, `last_name`,
   `street_address`, `city`, `state`, `county`, `postcode`) — confirming the Stage 2 priority
   already recorded as recall 0.00 in docs/BASELINE_M1.md.
6. **Label-space mismatch is real**: 55 Nemotron labels vs our 11 types. Training a Stage 2
   model needs an explicit crosswalk (the tables above) plus a policy for the ~43 labels we do
   not model — collapse to `O`, or adopt them. Adopting the HIPAA-mapped ones is the coherent
   choice; the sensitive-attribute ones (`race_ethnicity`, `religious_belief`, `political_view`,
   `sexuality`) belong to a GDPR special-category profile (M3), not the HIPAA one.

## Reproducing

```console
$ pip install pyarrow            # analysis only — not a runtime dependency
$ python eval/scripts/nemotron_tags.py --data-dir /path/with/parquet
```
Parquet files come from
`https://huggingface.co/datasets/nvidia/Nemotron-PII/resolve/main/data/{train,test}-00000-of-00001.parquet`
(~300 MB total). Counts above were taken 2026-08-22 against the current `main` revision.
