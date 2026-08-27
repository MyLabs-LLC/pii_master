# pii-cascade-balanced-v2

A CPU document tagger for sensitive data (PII / PHI / PCI): a document-level gate
in front of 57 per-tag heads. NumPy only, 28 MB, **3.92 ms p95** per document on
one core.

Best of **sixteen models** measured on eight sealed corpora in one head-to-head
run — two architectures re-tuned from scratch over 2,000 trials on the full
531,431-row training corpus, plus twelve operating points and a gate variant.

**Read `MODEL_CARD.md` before using this.** It does not clear the gates it was
measured against, and section 1 of the card says which, by how much, and why the
evidence says those gates are not reachable with this architecture.

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
contributes to a metric only if its gold can measure it — three of the eight
carry positive-only or partial tag gold, so precision-bearing metrics are
**NOT MEASURABLE** there and are excluded rather than counted as zero.

| Measure | Result |
| --- | ---: |
| macro F2 (declared ranker) | **0.6614** |
| macro F1 · F0.5 | 0.6398 · 0.6183 |
| macro precision · recall | 0.6182 · 0.6938 |
| micro F1 | 0.7255 |
| priority macro F0.5 (16 priority tags) | 0.7434 |
| worst measurable priority-tag recall | 0.7221 |
| document precision · specificity · recall | 0.8956 · 0.8792 · 0.7975 |
| one-core p95 / throughput | 3.92 ms / 314 docs/s |

Against `pii-steady-aim-cascade-v1`, which it supersedes: document recall
0.7532 → **0.7975**, macro recall 0.6482 → **0.6938**, worst priority-tag recall
0.6524 → **0.7221**, priority gates cleared 25 → **29** of 55. Micro F1 and
priority F0.5 regress slightly (−0.006, −0.003); if micro F1 is your headline,
the card says stay on v1.

**It is a triage model, not a redaction model.** It predicts document-level tags,
not entity spans, and it misses roughly 30% of PII-bearing real-world documents
at the gate. A "clean" verdict is not evidence of absence.

## Layout

    tagger.py                 self-contained entry point
    runtime/                  vendored feature extraction and cascade
    models/model/             weights.npz + model.json
    models/metadata.json      architecture and provenance
    models/verification.json  post-packaging re-score through tagger.py
    docs/                     the four run reports and the evidence the card cites
    examples/                 inputs that run on the first try
    SHA256SUMS                verify every member

The bundle was re-scored through its own `tagger.py` after packaging:
priority macro F0.5 on `pii_holdout_20k`, expected 0.841073, measured 0.841073,
delta 2.5e-08.
