# pii-master-ner-m 0.3.1

Stage 2 token tagger for **PII / PHI detection**, distilled from
`kalyan-ks/ettin-68m-nemotron-pii` on [nvidia/Nemotron-PII](https://huggingface.co/datasets/nvidia/Nemotron-PII).
Built to run on **one CPU core**.

- Architecture: dilated depthwise-separable CNN tagger, d=128 x 6 layers
- Label space: 111 BIO classes over 55 Nemotron entity types
- Format: onnx-fp32 (30.0 MB)
- Confidence calibration: isotonic, fitted on a held-out slice
- Source commit: `d0a457f4ca9f4199a1cf4888cf6d30b90ccce3bc`

## Intended use

Input is plain text; output is character spans typed against the 18 HIPAA Safe
Harbor identifier categories. It is designed to run **behind the Stage 1 rules
tier**, not alone -- see Limitations.

**Not** intended as a de-identification guarantee. It is a detector that helps
a reviewer, and Safe Harbor de-identification is a legal determination this
model cannot make.

## Measured performance

Exact `(type, start, end)` match on held-out Nemotron-PII documents, fused with
the rules tier:

| | F1 | F2 |
|---|--:|--:|
| the 12 types the rules also cover | 0.935 | 0.921 |
| the 14 types only this model emits | 0.914 | 0.902 |

F2 is reported because the cost matrix is asymmetric: a missed identifier is a
reportable incident, a false alarm costs a reviewer minutes.

## Limitations that matter

- **Synthetic training data.** Nemotron-PII is generated, not real. Scores here
  do not transfer unexamined to real clinical text; the standard benchmark
  (n2c2/i2b2 2014) needs a data use agreement and was not used.
- **The demographic slice is synthetic too.** Name recall varies by only 0.020
  across race/ethnicity groups, which sounds excellent and mostly reflects
  names drawn from a generator rather than from the world. It is a real gate
  that would catch a large disparity, and it certifies synthetic names only.
- **Credit card numbers are deliberately suppressed.** 88% of the training
  corpus's card numbers fail the Luhn checksum, so the serving path re-validates
  and drops them. Measured F1 on that type against this corpus is ~0.18, and
  that is the correct behaviour, not a defect.
- **US / English scope.** Non-US identifier formats are out of scope.
- **Confidences are calibrated for THIS model on THIS corpus.** A threshold
  tuned here is not automatically right for another text distribution.
- **The model can be confidently wrong.** Adversarial near-misses -- order
  numbers, chart numbers, subscriber ids -- are its documented failure class,
  which is why the serving path ships a confidence floor and rule fusion.

## Licensing

Code MIT (see LICENSE). Trained on Nemotron-PII, **CC BY 4.0**: attribution to
NVIDIA is required when redistributing this model or its outputs. The teacher
`kalyan-ks/ettin-68m-nemotron-pii` is MIT.

## Verify before you trust it

    python training/package.py verify <this directory>

The weights ship in `model.onnx.data` alongside `model.onnx`. A package missing
or truncating that file still loads and returns confident garbage, so the
checksums are load-bearing rather than decorative.
