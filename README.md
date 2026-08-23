# pii_master

Fast, CPU-friendly detection and document classification of **PII** (personally
identifiable information) and **PHI** (protected health information).

Design point: models may be trained on GPUs, but production inference must run on
**1 CPU core / 4 GB RAM within 5 ms per document (p95)**. Two serving modes fit
inside that:

| mode | tiers | dependencies | 10 KB p95 | finds |
|---|---|---|--:|---|
| `fast` (default) | Stage 1 rules | **none** (stdlib only) | 0.70 ms | format- and cue-anchored identifiers |
| `deep` (`--deep`) | rules + Stage 2 NER | `pii-master[ml]` | 7.99 / 14.80 ms | + names, addresses, cue-free identifiers |

Measured on one core with `pii-master bench [--deep]`; deep mode's budget is
25 ms and ships two students, `m` and `l`. What it buys, on 3,000 held-out
Nemotron-PII documents nobody in this repo authored:

| | rules only | deep (`m`) | **deep (`l`)** |
|---|--:|--:|--:|
| **micro F1**, all 24 types | 0.458 | 0.922 | **0.934** |
| **micro F2**, all 24 types | 0.359 | 0.910 | **0.922** |
| span F1, the 12 types rules cover | 0.796 | 0.935 | **0.940** |
| span F1, 14 types rules cannot emit | 0.000 | 0.914 | **0.930** |
| 10 KB p95, one core | 0.69 ms | 8.54 ms | 15.51 ms |
| peak RSS | 20 MB | 120 MB | 137 MB |

and at the level users actually act on — *does this document contain PII?* —
measured on the same 3,000 documents:

| | rules only | **deep (`l`)** |
|---|--:|--:|
| recall on documents containing an identifier | 0.809 | **0.998** |
| documents missed entirely | 571 | **7** |
| false alarms on adversarial near-misses | 0.000 | **0.000** |

One in five documents containing PII was invisible to the rules; it is now one
in 425. Document accuracy and PHI recall are both 1.00 on the frozen corpus in
every mode. **F2 is reported alongside F1** because this scanner's errors are not
symmetric — a missed identifier is a reportable incident, a false alarm costs a
reviewer minutes — so weighting recall four times as heavily is closer to the
real cost. Neither is *the* number: precision is what keeps the scanner switched
on. Per-type numbers, the threshold sweep where the two metrics disagree, the
fusion policy, and the students that did *not* ship:
[docs/STAGE2_INTEGRATION.md](docs/STAGE2_INTEGRATION.md).

Stage 1 is a zero-dependency rules engine: pre-scan windows narrow the text, regex
candidates are validated by checksums and format rules. Stage 2 is a distilled
dilated-CNN token tagger exported to ONNX, fused with the rules under an explicit
precedence policy. Both feed one aggregation stage that emits an explainable
document label (`NONE` / `PII` / `PHI`) and risk score.

Architecture and roadmap: **[docs/DESIGN.md](docs/DESIGN.md)**. Measured baselines:
**[docs/BASELINE_M1.md](docs/BASELINE_M1.md)** (internal, rules) and
**[docs/BASELINE_NEMOTRON.md](docs/BASELINE_NEMOTRON.md)** (external holdout, rules).
Stage 2: the **[distillation plan](docs/DISTILLATION_PLAN.md)**, its
**[measured results](docs/DISTILLATION_RESULTS.md)**, and the
**[integration results](docs/STAGE2_INTEGRATION.md)**. Also: the
**[Nemotron tag survey](docs/NEMOTRON_PII_TAGS.md)**, the
**[post-M1 review](docs/IMPROVEMENT_PLAN.md)**, and a
**[survey of prior art](docs/PRIOR_ART.md)**.

**Status:** v0.3 (milestone M2) — Stage 2 shipped behind `--deep`.

## Quickstart

```console
$ pip install -e ".[dev]"
$ pytest
$ pii-master scan document.txt --pretty
$ pii-master eval eval/corpus/frozen_v1.jsonl   # score vs the frozen gold corpus
$ pii-master bench --fail-over-budget           # latency vs the 5 ms budget (CI gate)
```

Example (`--fail-on-detect` exits 1 when anything is found, so it works as a CI gate):

```console
$ echo "Patient MRN: 4829471, DOB: 03/14/1985" | pii-master scan - --pretty
{
  "files": [
    {
      "path": "-",
      "label": "PHI",
      "risk_score": 54.0,
      "counts": { "DATE_DOB": 1, "MRN": 1 },
      "reasons": [
        "MRN detected (#8 Medical record numbers) -> PHI",
        "DATE_DOB x1 contributes 13.5 to risk",
        "MRN x1 contributes 25.5 to risk"
      ],
      "entities": [ "..." ]
    }
  ]
}
```

As a library:

```python
from pii_master import scan_text

report = scan_text("Payroll SSN 123-45-6789 on file.")
report.label        # DocLabel.PII
report.risk_score   # explainable 0-100 score
report.to_dict()    # JSON-ready report with span evidence
```

## Deep mode

The rules cannot find a person's name or a street address — no regex can, at
shippable precision. Stage 2 is a 6.6 M-parameter dilated CNN, distilled from
`kalyan-ks/ettin-68m-nemotron-pii` on the CC BY 4.0
[Nemotron-PII](https://huggingface.co/datasets/nvidia/Nemotron-PII) corpus and
exported to fp32 ONNX. It runs on one core, over the whole document.

```console
$ pip install -e ".[ml]"          # onnxruntime + tokenizers, nothing heavier
$ export PII_MASTER_MODEL_DIR=/path/to/bundle
$ pii-master scan discharge.txt --deep --pretty
$ pii-master bench --deep --fail-over-budget
```

```python
from pii_master import scan_text
report = scan_text(open("discharge.txt").read(), deep=True)
```

Deep mode never silently degrades: if the extra or the model artifact is
missing it raises `ModelUnavailable`, because a caller who asked for names and
got rules-only would read the absence of names as "this document has no names".

A **bundle** is three files that must travel together — `model.onnx` (plus its
external weights), `tokenizer.json`, and `model.json` (the label table and the
calibration curve). A model paired with the wrong tokenizer produces confident
garbage silently, so they are exported as a unit and verified against the
checkpoint at export time. To build one:

```console
$ cd training
$ python train.py --data-dir ~/nemotron --size l --epochs 3   # see the plan
$ python export.py --size l --checkpoint artifacts/student_l.pt \
      --bundle artifacts/bundle --no-int8
```

Then package it, so it can be shipped and verified:

```console
$ python package.py build --bundle artifacts/bundle \
      --name pii-master-ner-l --version 0.3.0 --out ../dist --created 2026-08-23
$ python package.py verify ../dist/pii-master-ner-l-0.3.0
OK    pii-master-ner-l 0.3.0  (4 files, 43.7 MB, commit ad83bdc8550a)
```

A package adds a `MANIFEST.json` (sha256 and size per file, source commit,
measured scores), a `MODEL_CARD.md`, and the licence — because a PII/PHI model
with unknown provenance is worse than useless. **Run `verify` on any package
you did not build.** The weights ship in `model.onnx.data` beside `model.onnx`,
and a package whose weights are corrupted *without changing their size* still
loads and still answers: on the real artifact a flipped kilobyte turned an
`MRN` into a `USER_ID` at 0.87 confidence, which is a silent PHI miss. Only the
checksum catches that.

**Two training mixtures ship.** `pii-master-ner-l-mixed` (recommended) is
trained on Nemotron **plus** ai4privacy's English rows, which raises strict
span recall on the ai4privacy holdout from 0.385 to **0.580** and document
recall from 0.870 to **0.924**, with Nemotron unchanged (micro F1 0.934 →
0.935). It costs one of fourteen adversarial frozen-corpus negatives — a
magazine subscriber id now read as `USER_ID` — without touching PHI recall,
which stays 1.00. `pii-master-ner-l` is the Nemotron-only model for anyone who
weights that differently. Details:
[docs/STAGE2_INTEGRATION.md](docs/STAGE2_INTEGRATION.md) §7.10–7.11.

**Published:** the Nemotron-only models are on the Hub —
[`MyLabs-LLC/pii-master-ner-l`](https://huggingface.co/MyLabs-LLC/pii-master-ner-l)
and [`MyLabs-LLC/pii-master-ner-m`](https://huggingface.co/MyLabs-LLC/pii-master-ner-m).

```console
$ hf download MyLabs-LLC/pii-master-ner-l --local-dir ./ner-model
$ python training/package.py verify ./ner-model     # do this before trusting it
$ PII_MASTER_MODEL_DIR=./ner-model pii-master scan report.txt --deep --pretty
```

The **mixed** model —
[`MyLabs-LLC/pii-master-ner-l-mixed`](https://huggingface.co/MyLabs-LLC/pii-master-ner-l-mixed)
— is published **gated**, as academic research output. It is a derivative work
of `ai4privacy/pii-masking-300k`, whose licence covers academic and
non-commercial use and requires written permission from AI4Privacy for
redistribution or commercial use. The Hub repo enforces an acknowledgement
before serving the weights, and the card leads with the restriction and the
`licensing@ai4privacy.com` contact. Anyone needing redistribution or commercial
rights must obtain them from AI4Privacy directly; a draft request is in
[docs/AI4PRIVACY_LICENCE_REQUEST.md](docs/AI4PRIVACY_LICENCE_REQUEST.md).

Committed packages: [`models/`](models/) carries each release's manifest and
model card. The weights themselves are build output and are distributed
separately rather than stored in git.

`artifacts/bundle` is on the default search path, so no configuration is needed
after that. The search order is `$PII_MASTER_MODEL_DIR`, then
`~/.cache/pii_master/model`, then `training/artifacts/bundle`. Full runbook:
[docs/DISTILLATION_PLAN.md](docs/DISTILLATION_PLAN.md) section 5.

Then fit the calibration curve, so `--min-confidence` has units:

```console
$ python calibrate.py --data-dir ~/nemotron --model-dir artifacts/bundle
```

It fits one isotonic curve **per entity type**, falling back to a global curve
for types with too few spans. One curve for everything is nearly perfect in
aggregate and wrong in detail — the per-type errors cancel — which is how `URL`
came out 0.107 under-confident and lost twenty points of recall to a threshold
that was correct on average.

Without it the model's confidences are raw max-softmax — a ranking signal that
is systematically overconfident (a raw 0.55 span is right about 22% of the
time). With it, a confidence of 0.70 means roughly "70% chance this exact span
is right". The curve is isotonic, so it never re-orders spans; it only changes
what the number means.

Ship **fp32**, not int8: dynamic int8 quantization makes this model an order of
magnitude *slower*, because it is normalisation- and activation-bound rather
than matmul-bound — MatMul is 4.7% of its ONNX op profile. `--no-int8` skips
producing a file you should not use.

### Fusion: which tier wins an overlap

**checksum-validated rule > model > cue-anchored rule.** A Luhn-valid card
number or a parsed IP address is a verified fact and outranks the model; a
cue-anchored MRN guess is not, and does not. Cue-anchored rules still supply
recall wherever the model is silent. Reading the policy the other way — *all*
rules outrank the model — is the obvious implementation and it measures 0.028 F1
worse. The policy is `pipeline.fusion_rank`; the measurements are in
[docs/STAGE2_INTEGRATION.md](docs/STAGE2_INTEGRATION.md).

Model spans carry two more guards: a confidence floor (`--min-confidence`,
default 0.5) and re-validation of checksummed types — 88% of Nemotron's gold
card numbers fail Luhn, so an unguarded student learns to emit card numbers no
payment network would issue.

## What it detects

**Rules tier** (both modes) — `EMAIL`, `PHONE_US`, `SSN`,
`CREDIT_CARD` (Luhn-validated), `IP_ADDRESS` (v4 + v6), `URL`,
`DATE_DOB` (birth-cue anchored), `MRN`, `ACCOUNT_NUMBER`, `HEALTH_PLAN_ID`,
`US_DRIVER_LICENSE` (cue anchored).

**Model tier** (`--deep` only) — `PERSON_NAME`, `ADDRESS`, `GEO_COORDINATE`,
`DATE_TIME`, `FAX_NUMBER`, `BANK_ROUTING`, `SWIFT_BIC`, `VEHICLE_ID`,
`DEVICE_ID`, `MAC_ADDRESS`, `NATIONAL_ID`, `TAX_ID`, `USER_ID`, `BIOMETRIC_ID`.

Every type maps to one of the 18 HIPAA Safe Harbor identifier categories
(`entities.TAXONOMY`), and reports cite the row. Two Nemotron label groups are
deliberately *not* adopted: credentials (password, api_key, cvv, pin, cookie)
and GDPR special-category attributes (race/ethnicity, religious belief,
political view, sexuality). Neither is a HIPAA identifier; both belong to the M3
policy profiles, not to the default HIPAA output.

## License

MIT — see [LICENSE](LICENSE).
