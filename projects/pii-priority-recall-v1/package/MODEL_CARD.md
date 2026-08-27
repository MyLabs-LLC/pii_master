# PII priority fusion 1k v1

## ⚠️ Things to improve

- Equal-corpus macro F2 is **0.4835** (95% corpus-bootstrap interval
  0.3501–0.5892), well below the 0.90 aspiration. The recall-first operating
  point produces many false positives and is not precision-ready.
- Only the first **1,000 characters** are read. The 1k setting preserved every
  measured priority gate on these holdouts, but it does not establish recall for
  sensitive values appearing later in long documents.
- This is document classification, not span NER. It cannot locate or redact a
  value, and a positive tag does not identify which text triggered it.
- Several sources are synthetic, silver, partial-label, or dual-judge corpora.
  Real-world drift, OCR corruption, new jurisdictions, and adversarial formatting
  require independent validation.
- Source data include AI4Privacy and NonCommercial material. Use is restricted
  to research/internal evaluation until redistribution and commercial rights are
  confirmed.

## Model description

The artifact is a 61-label CPU document tagger. It fuses four compact hashed
cue models—recall-max, F2-oriented hash, TF-IDF-derived, and low-rank
EmbeddingBag-derived heads—with a per-label Boolean strategy. All 16 requested
priority labels are locked to the recall-max component. Runtime is NumPy only.

## Measured performance

The headline is the equal-weight mean across the five label-complete holdout
corpora. The three partial-label corpora contribute only supported positive
recall gates. Every interval uses 1,000 bootstrap resamples.

| Measure | Result |
| --- | ---: |
| Equal-corpus macro F2 | 0.4835 (95% CI 0.3501–0.5892) |
| Equal-corpus micro F1 tie-break | 0.3812 (95% CI 0.2347–0.5348) |
| Priority gates | 55/55 conclusive passes |
| Worst measurable priority recall | 0.9888 |
| Lowest priority recall 95% bound | 0.9811 |
| One-core p95, 1k effective read | 2.200 ms/document |
| Throughput, same 1,000-document sample | 919.1 documents/second |

OpenPII is the most flattering complete corpus (macro F2 0.6297). Do not quote
that as general performance; quote the equal-corpus 0.4835 result. BetterDataAI
is the weakest complete corpus (0.2313).

Relative to the same fusion at 20k, the 1k model is 4.01× faster at p95 and
improves macro F2 by 0.0036, but its worst priority recall falls from 0.9966 to
0.9888. Workloads with possible late-page PII should retain longer scanning or
chunk the document even though that costs latency.

Full per-corpus, per-tag, bootstrap, leakage, and read-depth results are in
`docs/26-08-25_priority-recall-1000-run.md` and the accompanying workbook.

## Intended use and limits

Intended for internal/research triage, routing, and high-recall document-level
alerts over English-like business records. Human review or a downstream
precision/span detector is required before redaction, blocking, disclosure, or
any decision affecting a person.

Do not use it as proof that a document is free of PII, as a compliance
certification, as an access-control boundary, or for fully automated legal,
medical, employment, credit, insurance, or identity decisions. The 90% claim is
limited to measurable tag–corpus pairs with at least 30 positives in the frozen
evaluation matrix.

## Training and evaluation data

All eight approved training directories were combined with source identity and
partial-label masks retained. The audit found 554,247 indexed training rows,
501,168 unique training text hashes, and 19,668 hashes shared with evaluation;
22,816 overlapping training rows were excluded from fitting. Evaluation used
121,179 unique text hashes across eight holdout directories, scored one corpus
at a time.

## Architecture and inference

- Runtime: Python 3.10+ and NumPy.
- Catalogue: 61 document-level PII/PCI/PHI labels.
- Effective read ceiling: 1,000 characters.
- Selection order: priority gate, equal-corpus macro F2, equal-corpus micro F1,
  then one-core p95 latency.
- Integrity: verify all members with `SHA256SUMS`.

## Provenance

Run: `pii-priority-recall-v1`, 2026-08-25. Approved budget: exactly 1,000
MLflow-tracked search trials. Source commit: `d662a3a`. Licensing status:
research/internal only pending confirmation of all source-data rights.

Post-package verification re-scored all 126,129 indexed holdout rows through
the shipped entry point. Its predictions were byte-identical to the sealed
champion predictions and reproduced macro F2 exactly.
