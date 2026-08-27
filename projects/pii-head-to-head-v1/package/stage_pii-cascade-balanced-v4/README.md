# pii-cascade-balanced-v4

A CPU document tagger for sensitive data (PII / PHI / PCI) over **58 labels**. Architecture: cascade.

**Read `MODEL_CARD.md` before using this.** It states where the model is worse than its alternatives and what it must not be used for. This file is only the quick start.

## Quick start

```bash
pip install -r requirements.txt
python tagger.py --text "SSN 123-45-6789, contact jane@example.com"
python tagger.py --file examples/positive.txt
python tagger.py --file examples/clean.txt --gate-only
```

```python
from tagger import Tagger
t = Tagger()
t.has_pii(text)   # bool  — does this document carry sensitive data
t.predict(text)   # [str] — which of the tags it carries
```

## Headline numbers

Eight sealed evaluation corpora, 126,129 documents, equal-corpus aggregation, one CPU core.

| metric | value |
| --- | ---: |
| micro F1 | 0.7862 |
| micro precision | 0.7188 |
| macro F0.5 | 0.6403 |
| macro F2 | 0.6641 |
| recall, macro over catalogue | 0.6655 |
| precision, macro over catalogue | 0.6461 |
| worst measurable priority recall | 0.6410 |
| document recall | 0.7975 |
| document precision | 0.8956 |
| document specificity | 0.8792 |
| one-core p95 (ms) | 3.9696 |

**Do not quote a single-corpus number from this suite.** The spread across corpora is large and the most flattering rows are synthetic; the equal-corpus figures above are the honest ones.

## What is in here

| | |
| --- | --- |
| `MODEL_CARD.md` | claims, limits, and what not to use it for |
| `HOW_TO_BUILD.md` | how this model was made, reproducibly |
| `tagger.py` | the entry point — self-contained |
| `models/` | weights, `metadata.json`, `verification.json` |
| `docs/` | the run reports and the evidence the card cites |
| `examples/` | inputs that work on the first try |
| `SHA256SUMS` | integrity |
