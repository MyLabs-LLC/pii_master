# Meeting brief — Priority PII detection model

## Results

The 1,000-trial search produced a fast, recall-first document tagger that met
the critical safety requirement on the frozen evaluation matrix. The selected
model reads the first 1,000 characters and passed **55 of 55 measurable
priority tag–corpus recall gates conclusively**. Its worst observed priority
recall was **98.88%**, and its lowest 95% bootstrap lower bound was **98.11%**.

The model is fast: **2.20 ms p95 per document on one CPU core**, or **919
documents/second** on the fixed 1,000-document benchmark sample. It is 4.01×
faster at p95 than the same fusion model reading 20,000 characters.

The important trade-off is precision. Equal-corpus macro F2 is **0.4835**, well
below the 0.90 aspiration, and micro F1 is **0.3812**. The recommended decision
is therefore to approve this model for **internal high-recall triage or shadow
evaluation**, with downstream verification, but not for autonomous redaction,
blocking, compliance certification, or registry-backed production deployment.

| Executive measure | Outcome | Decision impact |
| --- | ---: | --- |
| Priority recall gates | 55/55 conclusive passes | Critical recall objective met on measurable holdout pairs |
| Worst priority recall | 98.88% | Above the requested 90% floor |
| Lowest recall 95% bound | 98.11% | Recall result is statistically conclusive on measured pairs |
| Equal-corpus macro F2 | 0.4835 | Precision/overall quality target not met |
| Micro F1 tie-break | 0.3812 | Reported, but did not override the recall-first decision |
| One-core p95 latency | 2.20 ms/document | Passes the fast operating-point requirement |
| One-core throughput | 919.1 documents/second | Suitable for high-volume first-pass triage |
| Package reproduction | Exact over 126,129 rows | Shipped loader matches the sealed model |

## TL;DR

- Exactly 1,000 MLflow-tracked trials were completed across hash, TF-IDF,
  EmbeddingBag, and fusion candidates.
- The selected 1k fusion exceeded 90% recall for every measurable priority
  tag–corpus pair, including SSN, ITIN, medical identifiers, financial account
  identifiers, identity documents, credentials, names, and addresses.
- The 1k pass is the best speed/recall operating point: 4.01× faster than 20k
  while preserving all 55 point gates.
- Macro F2 remains 0.4835. Approve only for internal triage/shadow use with
  downstream review; do not present it as a complete redaction system.
- Before production, resolve long-document coverage, precision, source-data
  licensing, registry promotion, and live monitoring.

## Meeting metadata

| Field | Value |
| --- | --- |
| Date | 2026-08-25 |
| Suggested duration | 30 minutes |
| Meeting objective | Decide the allowed use of the 1k PII model and fund the next validation/improvement phase |
| Model | `pii-priority-fusion-1k-v1` |
| Scope | 61 document-level PII/PCI/PHI tags; 16 recall-priority tags |
| Evaluation | Eight holdout corpora scored separately; 126,129 indexed rows |
| Status | Packaged research/internal candidate; not registry-promoted or deployed |
| Source commit | `d662a3a` |

## Decisions requested

1. **Approve internal use?** Approve the 1k model for high-recall triage and
   shadow evaluation only, with downstream verification and audit logging.
2. **Long-document policy?** Require upstream chunking or a longer secondary
   scan whenever sensitive content can occur after character 1,000.
3. **Quality target?** Keep macro F2 0.90 as a future target, or define a
   realistic operating threshold and per-tag precision floors for the next
   cycle.
4. **Licensing owner?** Assign review of AI4Privacy and NonCommercial source
   restrictions before redistribution or commercial use.
5. **Production path?** Decide whether to fund registry promotion, service
   integration, live traffic validation, and monitoring after the above items
   pass.

## Speed and read-depth discussion

All latency numbers below are end-to-end, one-core measurements on the same
deterministic 1,000-document sample. Quality was independently recalculated on
all eight holdouts at each read depth.

| Read pass | p95 latency | Docs/s | Speed vs 20k | Macro F2 | Priority point gates | Worst recall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **1,000** | **2.200 ms** | **919.1** | **4.01×** | **0.4835** | **55/55** | **0.9888** |
| 2,500 | 4.211 ms | 570.1 | 2.10× | 0.4797 | 55/55 | 0.9686 |
| 10,000 | 6.766 ms | 409.9 | 1.30× | 0.4799 | 55/55 | 0.9966 |
| 20,000 | 8.829 ms | 354.4 | 1.00× | 0.4798 | 55/55 | 0.9966 |

The 1k pass saves 6.63 ms/document at p95 relative to 20k and slightly improves
macro F2 by 0.0036. It reduces worst observed priority recall by 0.0079, but the
final 1,000-resample bootstrap still placed every measurable priority recall
lower bound above 90%.

The speed result must not be interpreted as late-document coverage. The selected
artifact intentionally ignores content after character 1,000. Upstream chunking
is the recommended protection when long documents are in scope.

## Priority recall coverage

The 16 requested priority categories all had measurable evidence on at least one
holdout corpus, although the number of measurable corpora differs by tag.

- Government and identity: SSN, ITIN, passport, driver's licence, military ID,
  and visa number.
- Medical: MRN, health-plan beneficiary number, and patient ID.
- Financial: bank account number, credit-card number, and IBAN.
- Credentials: password and PIN.
- Person and location: full name and address.

Across these categories, the minimum per-tag recall ranged from 98.88% to 100%
on measurable corpora. Address was the weakest measured category at 98.88%; its
lowest 95% bound was 98.11%. Patient ID was measurable in only one corpus, so
its apparent 100% recall has narrower external evidence than name or address.

## What worked

- Recall-first selection prevented higher-F2 candidates from winning when they
  missed critical tags. The standalone hash-F2, TF-IDF, and EmbeddingBag models
  had stronger micro F1 but failed between five and six priority gates.
- Locking priority labels to the recall-max head preserved the safety gate while
  per-label fusion improved non-priority quality.
- Exact overlap controls excluded 22,816 training rows sharing evaluation text;
  evaluation data were never removed or relabelled.
- Separate corpus scoring prevented a strong corpus from hiding weak results.
- The model remained compact and CPU-only, with a portable NumPy runtime.

## What did not meet target

- Equal-corpus macro F2 reached 0.4835, not 0.90. False positives remain the
  main quality problem.
- BetterDataAI was the weakest complete holdout at macro F2 0.2313. OpenPII was
  the strongest at 0.6297 and should not be quoted as general performance.
- Full 10k and 20k fusion passes exceeded the 5 ms p95 latency aspiration at
  6.77 ms and 8.83 ms respectively.
- The model emits document tags, not value locations. It cannot perform
  redaction or explain which text produced a prediction.

## Risks and safeguards

| Risk | Current safeguard | Required next step |
| --- | --- | --- |
| False positives | Recall-first model used only as a first-pass signal | Add downstream verifier/span detector and per-tag precision floors |
| PII after character 1,000 | Limitation stated in model card | Chunk documents or run a longer secondary pass |
| Dataset shift and OCR noise | Eight heterogeneous holdouts | Validate on representative live documents and OCR slices |
| Partial/silver labels | Missing labels never treated as negatives | Add human-reviewed complete-catalogue evaluation data |
| Licensing restrictions | Bundle marked research/internal only | Complete legal/data-rights review before redistribution or commercial use |
| Automated misuse | No deployment or registry promotion performed | Require human review and documented use-case approval |

## Recommended decision

Approve `pii-priority-fusion-1k-v1` as an **internal research and shadow-mode
champion candidate** for document-level, high-recall triage. Do not promote it
to a production registry alias or use it for autonomous redaction until the
following exit conditions are satisfied:

- A representative live-data evaluation confirms the priority recall gate.
- A chunking/secondary-scan design covers long documents.
- Per-tag precision/F2 requirements are agreed and met.
- Source-data usage and redistribution rights are cleared.
- Monitoring, rollback, audit logging, and human-review workflows are defined.

## Suggested 30-minute agenda

| Time | Topic | Desired outcome |
| ---: | --- | --- |
| 0–5 min | Goal and recommendation | Align on internal-only recall-first positioning |
| 5–10 min | Recall evidence | Confirm that the 55/55 gate result is sufficient for shadow use |
| 10–15 min | Speed/read-depth trade-off | Approve 1k default plus long-document fallback |
| 15–22 min | Precision, licensing, and operational risks | Agree on blocking issues for production |
| 22–27 min | Decisions requested | Record approvals, rejections, and constraints |
| 27–30 min | Owners and dates | Assign next actions |

## Decision and action log

Complete this table during the meeting.

| Item | Decision / action | Owner | Due date | Status |
| --- | --- | --- | --- | --- |
| Internal shadow/triage use |  |  |  | Open |
| Long-document chunking policy |  |  |  | Open |
| Precision/F2 acceptance criteria |  |  |  | Open |
| Live representative evaluation set |  |  |  | Open |
| Licensing and redistribution review |  |  |  | Open |
| Registry/deployment decision |  |  |  | Deferred |
| Monitoring and rollback design |  |  |  | Deferred |

## Verification

- The package re-predicted all 126,129 indexed holdout rows through its own
  `tagger.py` entry point.
- Packaged and sealed predictions were byte-identical with SHA-256
  `3bfd3f776d8b7c09a7f11daafb8b5c4d2316571e724232506857be2f0168c194`.
- Equal-corpus macro F2 reproduced exactly at 0.48345749382010117.
- Thirty-three focused tests and lint checks passed.
- ZIP integrity and all bundle member checksums passed.
- Final bundle SHA-256:
  `cd8462d346f9f3b9c095a08ffe2c7711a9ea49ec2e677d123df8bb72cd0fd192`.

## Artifacts

- Full technical report: `26-08-25_priority-recall-1000-run.md`.
- Experiment workbook: `26-08-25_Experiment-Log.xlsx`.
- Complete command archive: `26-08-25_priority-recall-1000-run.commands.txt`.
- Read-depth evidence: `../benchmarks/read_depth.json`.
- Package verification: `../dist/bundle_verification.json`.
- Delivery bundle: `../dist/pii-priority-fusion-1k-v1.zip`.
