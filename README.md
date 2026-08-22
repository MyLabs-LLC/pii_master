# pii_master

Fast, CPU-friendly detection and document classification of **PII** (personally
identifiable information) and **PHI** (protected health information).

Design point: models may be trained on GPUs, but production inference must run fast on
**1 CPU core / 4 GB RAM**. Stage 1 (this version) is therefore a zero-dependency,
stdlib-only rules engine: regex candidates validated by checksums and format rules,
aggregated into an explainable document label (`NONE` / `PII` / `PHI`) and risk score.
The full staged architecture — through quantized-ONNX NER, risk policies, and PDF/OCR
ingestion — is laid out in **[docs/DESIGN.md](docs/DESIGN.md)**.

**Status:** v0.1 (milestone M0) — Stage 1 rules engine, plain-text input.

## Quickstart

```console
$ pip install -e ".[dev]"
$ pytest
$ pii-master scan document.txt --pretty
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

## What it detects (v1)

`EMAIL`, `PHONE_US`, `SSN`, `CREDIT_CARD` (Luhn-validated), `IP_ADDRESS`,
`DATE_DOB` (birth-cue anchored), `MRN` (cue anchored) — each mapped to its HIPAA
Safe Harbor identifier category. See docs/DESIGN.md section 6 for the taxonomy and the
deferred types (names, addresses, and other model-dependent entities arrive with
Stage 2).

## License

MIT — see [LICENSE](LICENSE).
