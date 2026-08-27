# pii-steady-aim-cascade-v1

A CPU document tagger for sensitive data (PII / PHI / PCI) with a document-level
gate in front of 58 per-tag heads. NumPy only, 28 MB, 4.11 ms p95 per document
on one core.

**Read `MODEL_CARD.md` before using this.** It does not clear the gate set it
was measured against, and section 1 of the card says exactly which bars it
misses and by how much.

## Quick start

```bash
pip install -r requirements.txt

python tagger.py --text "Patient MRN 4472019, SSN 123-45-6789"
python tagger.py --file report.txt --gate-only
```

```python
from tagger import Tagger

t = Tagger()
t.has_pii(text)    # bool  — one dot product; the cheap question
t.predict(text)    # list  — the tags, empty when the gate stays shut
```

## Headline numbers

Equal-corpus means over eight sealed corpora (126,129 documents). A corpus
contributes to a metric only if its gold can measure it.

| Measure | Result |
| --- | ---: |
| priority macro F0.5 (16 priority tags) | 0.7474 |
| document precision | 0.8870 |
| document specificity | 0.8798 |
| document recall | 0.7582 |
| one-core p95 / throughput | 4.11 ms / 310 docs/s |

Against the prior `pii-priority-fusion-1k-v1`: priority macro F0.5 0.2057 →
0.7474, document specificity 0.0005 → 0.8798, document recall 0.9994 → 0.7582.

**It is better at staying quiet and worse at not missing things.** If a missed
identifier matters more than a false alarm, the older model is the right choice.

## Layout

    tagger.py             self-contained entry point
    runtime/              vendored feature extraction and cascade
    models/model/         weights.npz + model.json
    models/metadata.json  architecture and provenance
    models/verification.json  post-packaging re-score through tagger.py
    docs/                 run report and evidence the card cites
    examples/             inputs that run on the first try
    SHA256SUMS            verify every member
