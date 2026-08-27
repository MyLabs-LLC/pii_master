# PII priority fusion — 1k read champion

This CPU-only NumPy model emits document-level sensitive-data tags from the
first 1,000 characters. It was selected for recall and speed: all 55 measurable
priority tag–corpus gates had recall at least 90% with 95% bootstrap lower
bounds at least 90% on the eight frozen holdout corpora.

The equal-corpus macro F2 was 0.4835, so this is an internal/research candidate,
not a precision-ready redaction system. It predicts tags, not entity spans.

## Quick start

```bash
python -m pip install -r requirements.txt
python tagger.py --file examples/sample.txt
```

Python API:

```python
from tagger import Tagger

model = Tagger()
labels = model.predict("Patient record contains a medical record number.")
```

Input is a Unicode string. Output is a sorted list drawn from the frozen
61-label sensitive PII/PCI/PHI catalogue. The model deliberately truncates at
1,000 characters; use upstream chunking or a longer-read configuration when
late-document sensitive data is possible.

See `MODEL_CARD.md` and `docs/26-08-25_priority-recall-1000-run.md` before use.
