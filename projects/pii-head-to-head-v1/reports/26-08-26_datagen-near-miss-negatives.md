# Datagen: LLM-generated near-miss negatives do not transfer to real documents

## Results

2,000 adversarial near-miss negatives were generated with `claude -p` across ten
classes, verified, added to the document gate's training set at four weights, and
measured on the sealed real-world corpora. **The result is negative and the
mechanism is now certain.**

| generated-data weight | sealed real **AUC** | datax AUC | govdocs2 AUC | sealed real recall | **held-out generated fire rate** |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 *(baseline)* | 0.8453 | 0.8504 | 0.8434 | 0.4979 | 8.2% |
| 1.0 | 0.8437 | 0.8471 | 0.8432 | 0.4786 | **0.0%** |
| 4.0 | 0.8413 | 0.8388 | 0.8446 | 0.4749 | **0.0%** |
| 12.0 | 0.8461 | 0.8461 | 0.8469 | 0.4738 | **0.0%** |

*(recall at matched specificity 0.95; AUC is threshold-free. 1,556 generated
documents added to 432,903 fit rows; 392 held out and never trained on.)*

The gate learns the generated documents **perfectly** — it stops firing on 100%
of held-out generated near-misses, which it has never seen — and the sealed
real-world number does not move at all. Recall on real documents gets slightly
worse.

### The mechanism, measured

A classifier trained to answer "is this document machine-generated?" separates the
1,948 generated negatives from 5,152 real negatives at a **held-out AUC of
1.0000**.

That is the whole explanation. Style is a free, perfectly predictive feature. A
gate can satisfy every generated row by detecting Claude's prose, so gradient
descent has no pressure whatsoever to learn the thing the corpus was built to
teach — that a form with no values, a schema with no rows, or a policy *about*
identifiers is not a document containing identifiers. The corpus taught a
shortcut, and the shortcut was cheaper than the concept.

### What the verification caught

Of 2,000 documents, **52 were rejected (2.6%)**:

| reason | n |
| --- | ---: |
| `tell:disclaimer` | 13 |
| `identifier:iban` | 13 |
| `identifier:ipv4` | 10 |
| `identifier:street_address` | 9 |
| `identifier:ssn` | 5 |
| `identifier:card` | 5 |
| `identifier:email` | 2 |

The 39 identifier hits are **mislabelled negatives** — documents asserting
`doc_has_pii = False` that contain a social security number, a card number or a
street address. Each would have taught the gate to stay shut on a document that
does carry PII, which is the precise failure this project exists to prevent. The
13 disclaimers would have handed the model a single sentence to memorise.

The surviving 1,948 were genuinely adversarial: the champion gate wrongly fires on
**23.3%** of them, against **15.5%** of real govdocs2 negatives. They were harder
than the real thing — and still useless, because they were separable.

## TL;DR

- **Generated near-miss negatives do not work.** Sealed real AUC 0.8453 → 0.8461
  across four weights: flat. Real-document recall slightly worse.
- **The gate learns them completely** — held-out generated fire rate 8.2% → 0.0%.
  Every training-side metric improves. None of it transfers.
- **Cause is measured, not inferred**: generated vs real negatives separate at
  **AUC 1.0000**. The model learns "machine-generated" instead of the concept.
- **The verification gate earned its place**: it rejected 39 documents carrying
  real identifiers under a negative label, plus 13 shared tells.
- **This does not invalidate the diagnosis.** Real-world negatives being harder is
  still the binding constraint; synthesising them is what fails.
- **The next lever is real documents, not generated ones** — active learning on an
  unlabelled real corpus (below).

---

| | |
| --- | --- |
| Date | 2026-08-26 |
| Author | Ryan Lence |
| Project | `projects/pii-head-to-head-v1` |
| Run ID | H1 (datagen follow-up) |
| Scope | document gate only; tag heads untouched |
| Outcome | negative result, mechanism identified, direction changed |

## What was built

To the datagen module's generator contract: one real file per document at a
realistic path, a stamped output directory (`2000_26082521_claude`),
`manifest.json` beside the corpus, a ten-class registry, round-robin assignment,
and per-document faker fallback (84 of 2,000 fell back; 1,916 came from the LLM).

The ten classes are all documents dense in personal-data vocabulary and empty of
personal data: unfilled intake forms, data dictionaries, privacy policies,
redacted records, aggregate demographic tables, regulatory excerpts, security
guidance, log/config extracts, vendor questionnaires and compliance training
material. Each prompt randomised industry, era, register and layout to spread the
corpus.

**The frozen training snapshot was not touched.** Generated documents live under
`projects/pii-head-to-head-v1/datagen/` and are added at fit time only, so every
earlier measurement in this run remains reproducible.

## Two false positives in my own verifier

Both found in a 40-document pilot, and both would have silently discarded the
best material:

- **`ipv6` matched timestamps.** The model's own pattern accepts two
  colon-separated groups, so `10:15:30` reads as an IPv6 address. It rejected 3 of
  40 pilot documents, all log extracts — the most adversarial class in the corpus.
  The verifier now requires four groups or a `::` elision. The model carrying that
  false positive is its own business; a verifier inheriting it throws away
  exactly the documents worth keeping.
- **Class-name echo was treated as leakage.** A data dictionary that opens "Data
  Dictionary" is a realistic document, not a leak: the class is a generation
  bucket that nothing here predicts. The oracle is `doc_has_pii = False`, and the
  phrase that would leak *that* is a disclaimer, which is still rejected.
  Downgraded to a warning (158 documents).

Pilot reject rate fell 15.0% → 2.5% once both were fixed.

## Why the negative result is worth having

The measurement was designed to distinguish success from its lookalike, and that
turned out to be the entire value of the exercise. Had I reported only what the
generated corpus achieved — "the gate now correctly stays silent on 100% of
adversarial near-miss negatives it has never seen" — it would have read as a
decisive win. It is a decisive nothing.

Any future attempt to close this gap with synthetic text needs to clear the
separability bar first. **AUC(generated vs real) ≈ 1.0 means the corpus cannot
teach anything, regardless of how good the documents look.** That check costs one
model fit and should run before generation is scaled, not after.

## Recommendation: active learning on real documents

The gap is real, the diagnosis holds, and the corpus has to be real. The cheapest
route that produces real near-misses by construction:

1. **Mine, don't synthesise.** Score a large unlabelled real-document corpus with
   the current gate and take the highest-scoring documents. Documents the gate
   fires hardest on *are* near-misses, by definition, and they carry no style
   tell because nothing generated them.
2. **Judge only those.** The dual-judge process is the expensive step, so spend it
   where it is informative — a few thousand documents from the top of the score
   distribution rather than a random sample, most of which the gate already gets
   right.
3. **Measure the same way.** Held-out slice, sealed real AUC as the headline, and
   the separability check as a precondition. If mined near-misses separate from
   ordinary real negatives at AUC ≈ 1.0, something is wrong with the mining.

The govdocs2 training pool already holds 12,148 real negatives and the eval pool
198 unseen source directories, so the raw material for step 1 exists without any
new collection.

## Limitations

- **One generator, one model.** This tests Claude prose at default settings. A
  different model, heavy paraphrase, or a style-randomisation pass might lower
  separability — but AUC 1.0000 is a long way from 0.5, and nothing here suggests
  the gap is closable by prompt engineering.
- **Volume.** 1,556 documents against 432,903 fit rows is 0.36% by count; at
  weight 12 the effective share is roughly 14%, which is where the flat AUC was
  still observed. A far larger corpus was not tried, and would not fix
  separability.
- **Negatives only.** The diagnosis showed sealed positives are also subtler.
  Generating hard *positives* was not attempted and carries a worse version of the
  same risk, since their gold would also have to be synthesised.
- **The person-name check is prompt-enforced, not verified.** Shape detectors
  catch SSNs, cards, IBANs, IPs, phones, addresses and DOB values; a bare
  fabricated personal name would pass. 158 documents carry an `echoes_class_name`
  warning and were kept.

## Artifacts

| Path | What |
| --- | --- |
| `datagen/2000_26082521_claude/` | 2,000 documents, `manifest.json`, `manifest_verified.json`, `verification.json` |
| `probe/gate_augment.json` | four gate refits: sealed AUC, recall, held-out fire rate |
| `training/h2h_datagen.py` | the generator |
| `training/h2h_datagen_verify.py` | the verification gate |
| `training/h2h_gate_augment.py` | the augmentation measurement |
| `reports/26-08-25-1_gate-diagnosis.md` | the diagnosis this follows from |
