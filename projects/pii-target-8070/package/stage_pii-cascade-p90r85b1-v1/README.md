# pii-cascade-p90r85b1-v1

A CPU document tagger for sensitive data (PII / PHI / PCI) over **61 labels**.

**The generalisation-first model.** Best of seven measured models on
out-of-distribution documents, and best on every summary metric except precision.
It does **not** meet the 90% precision target — `pii-cascade-p88r90-v1` does.
Read `MODEL_CARD.md` item 2 before quoting the headline: out-of-distribution
micro F1 is **0.5453**, not 0.8564.

## Quick start

```bash
pip install -r requirements.txt
python tagger.py --text "SSN 123-45-6789, contact jane@example.com"
python tagger.py --file examples/positive.txt
```

```python
from tagger import Tagger
t = Tagger()
t.has_pii(text)   # bool
t.predict(text)   # [str]
```

## Headline numbers

Eight sealed corpora, 126,129 documents, equal-corpus aggregation, one core.

| metric | value |
| --- | ---: |
| micro F1 | 0.8564 |
| micro precision | 0.8721 |
| micro recall | 0.8415 |
| macro F0.5 | 0.7028 |
| macro F2 | 0.6865 |
| recall, macro | 0.6367 |
| precision, macro | 0.7218 |
| worst priority recall | 0.5398 |
| document recall | 0.7975 |
| document precision | 0.8956 |
| document specificity | 0.8792 |
| one-core p95 (ms) | 4.0290 |

## Out-of-distribution

1,612 synthetic PDFs no training corpus resembles, scored after selection:

| metric | value |
| --- | ---: |
| micro F1 | 0.5453 |
| micro precision | 0.7156 |
| micro recall | 0.4405 |

## What is in here

| | |
| --- | --- |
| `MODEL_CARD.md` | claims, limits, the generalisation evidence |
| `HOW_TO_BUILD.md` | how it was made, reproducibly |
| `tagger.py` | self-contained entry point |
| `models/` | weights + `metadata.json` + `verification.json` |
| `docs/` | run reports, confidence intervals, holdout results |
| `SHA256SUMS` | integrity |
