# pii-cascade-p88r90-v1

A CPU document tagger for sensitive data (PII / PHI / PCI) over **61 labels**.
Built to hit micro precision >= 0.90, micro recall >= 0.80, micro F1 >= 0.80 — it
meets all three **on the point estimate**. See `MODEL_CARD.md` item 1 before
quoting: the confidence interval does not clear the precision bar.

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
| micro F1 | 0.8470 |
| micro precision | 0.9000 |
| micro recall | 0.8020 |
| macro F0.5 | 0.7035 |
| macro F2 | 0.6480 |
| recall, macro | 0.5802 |
| precision, macro | 0.7530 |
| worst priority recall | 0.4799 |
| document recall | 0.7975 |
| document precision | 0.8956 |
| document specificity | 0.8792 |
| one-core p95 (ms) | 4.0290 |

**Do not quote a single-corpus number.** The generous rows are synthetic.

## What is in here

| | |
| --- | --- |
| `MODEL_CARD.md` | claims, limits, the confidence-interval caveat |
| `HOW_TO_BUILD.md` | how it was made, reproducibly |
| `tagger.py` | self-contained entry point |
| `models/` | weights + `metadata.json` + `verification.json` |
| `docs/` | run reports and cited evidence |
| `SHA256SUMS` | integrity |
