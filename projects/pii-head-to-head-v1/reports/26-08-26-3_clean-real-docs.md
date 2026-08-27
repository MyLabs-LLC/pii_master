# 10,000 real PII-free documents: a different labelling standard, and it hurts

## Results

10,000 real govdocs documents asserted PII-free were tested against the document
gate. **They cannot be used as they stand**, for two independent reasons found
before any training, and a third found by measuring.

### Admissibility — 2,874 of 10,000 disqualified before any use

| | n | why |
| --- | ---: | --- |
| in a **sealed evaluation** corpus | **587** | training on them leaks the measurement into the model |
| **label conflict** with existing gold | **782** | existing dual-judge says PII-bearing (`sensitivity: low`); this manifest says `none` |
| overlapping any existing corpus | 2,874 | already labelled; re-adding is duplication at best |
| **usable — no existing gold at all** | **7,126** | what the rest of this report uses |

The conflict rate on the overlap is **27.2%**. That is not a rounding
difference; it is two labelling passes disagreeing about the same files —
`002/002518.doc` (Budget Proposal), `010/010044.txt` (Audit Report) and 780
others.

### They are distinguishable from the corpus they would join

| comparison | AUC |
| --- | ---: |
| **govdocs2 train vs govdocs2 eval negatives** *(control: same corpus, two halves)* | **0.5376** |
| new clean docs vs govdocs2 **eval** negatives | 0.7631 |
| new clean docs vs govdocs2 **train** negatives | 0.8087 |
| *govdocs2 vs datax negatives (control: genuinely different corpora)* | *0.9858* |

Two halves of the existing judged-clean govdocs2 documents are effectively
indistinguishable (0.54). The new documents separate from them at **0.76–0.81**,
despite all being govdocs. Something systematic differs, and the 27.2% conflict
rate names it: the bar for "clean" is not the same one.

The gate agrees. It wrongly fires on **20.1%** of the new documents against
**13.1%** of the sealed real negatives — they carry more PII-like material.

### Adding them makes the model worse

Balanced gate, 5,700 documents added to training, 1,426 held out, scored on the
sealed real corpora:

| clean-doc weight | AUC real | AUC datax | AUC govdocs2 | recall @ spec 0.95 | best recall @ bars | held-out fire |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **0.0** *(baseline)* | **0.8804** | **0.8888** | **0.8746** | **0.5662** | **0.6231** | 9.0% |
| 1.0 | 0.8764 | 0.8864 | 0.8706 | 0.5475 | 0.6020 | 5.3% |
| 4.0 | 0.8744 | 0.8858 | 0.8687 | 0.5501 | 0.5987 | 4.8% |
| 12.0 | 0.8721 | 0.8841 | 0.8664 | 0.5438 | 0.5905 | 4.8% |

**Every column degrades monotonically with weight.** This is worse than the
synthetic corpus, which was merely flat: real documents labelled to a different
standard actively teach the gate to suppress the signals the sealed gold rewards.

## TL;DR

- **587 documents are in the sealed eval set.** Excluded — training on them
  would have invalidated every number in this project.
- **782 have conflicting gold**, 27.2% of the overlap. The existing dual-judge
  process calls them PII-bearing; this manifest calls them clean.
- **The usable 7,126 are distinguishable from existing judged-clean govdocs2 at
  AUC 0.76–0.81**, against a same-corpus control of 0.54.
- **Adding them degrades the gate monotonically** — sealed real AUC 0.8804 →
  0.8721, best constrained recall 0.6231 → 0.5905.
- **Do not use them as negatives at all.** Adjudicated: the existing gold is
  right on ~97% of the 782 conflicts, so the corpus is mislabelled, not merely
  labelled to a different standard.
- The corpus itself is fine and well-formed; the problem is that its "clean"
  means something different from the corpus the model is measured against.

---

| | |
| --- | --- |
| Date | 2026-08-26 |
| Project | `projects/pii-head-to-head-v1` |
| Scope | document gate only; 7,126 admissible documents of 10,000 |
| Outcome | rejected for training; labelling-standard conflict identified |

## Adjudication of the 782 conflicts — the existing gold is right

The conflicts were checked against the documents themselves, not against either
labeller's reasoning.

**1. The identifiers the existing gold claims are actually there.**

| entity claimed by existing gold | claimed | found in the text | rate |
| --- | ---: | ---: | ---: |
| Email | 438 | 433 | **98.9%** |
| Phone Number | 492 | 477 | **97.0%** |
| ZIP Code | 183 | 183 | **100%** |

**2. They belong to named individuals, not organisations.** Adjudicating all 782
on full text:

| | n | share |
| --- | ---: | ---: |
| **named individual beside a direct contact detail** | **607** | **77.6%** |
| no email or phone in the document | 161 | 20.6% |
| role mailbox only *(clean_docs right)* | 8 | 1.0% |
| contact detail, no name nearby *(ambiguous)* | 6 | 0.8% |

Of the 161 without a phone or email, the existing gold claimed `Full Name` (86)
or `Username` (23), and **155 of 161 contain a personal-looking name**. That
check is a regex with a stop-list and is weaker evidence than the 607, which
required a name adjacent to a contact detail.

Taken together the existing gold is corroborated on roughly **97%** of the
conflicts, with strong evidence on 77.6% of them.

**3. The clean_docs reasoning is not a different definition — it is factually
wrong.** For `016/016635.pdf` its note reads *"no named individual in extracted
text"*; the document reads *"For more information, contact <NAME> at (202)
512–7215 or <EMAIL>"*. That is a named person with a direct line and address.
The emails are individual staff accounts at `.gov` and `.mil` domains, and only
6.2% of the phone numbers are toll-free.

**4. A likely mechanism.** The first contact detail sits at a median of **char
4,959**, and **86.3% appear beyond char 2,000**:

| percentile | position of first contact detail |
| --- | ---: |
| p10 | char 1,142 (1.1% through) |
| p50 | char 4,959 (8.4% through) |
| p90 | char 26,324 (92.5% through) |

Median document length is 68,240 characters. A labelling pass that saw only an
opening excerpt or a summary would miss 86% of these, and the notes read exactly
that way — accurate about the document's subject, wrong about its contact block.

### What follows

- **The existing training and evaluation labels are correct.** The worry raised
  earlier in this report — that the gate might be being taught to fire on clean
  documents — is resolved: it is not.
- **The gate diagnosis stands.** The 192 disputed documents inside the sealed
  eval corpus carry the *correct* label, so the "adversarially harder" reading of
  govdocs2-eval is not undermined by label noise.
- **`clean_docs` cannot be used as negatives at all**, not merely until
  reconciliation. Roughly 97% of the checked conflicts are mislabelled, and the
  same pass produced the labels for the other 9,218 documents.
- **The corpus could be salvaged by re-labelling on full text.** The documents
  and the extraction are fine; only the labels are wrong, and the failure mode is
  specific and fixable.

## Why this is a different failure from the synthetic one

The synthetic corpus failed because it was *separable by style* — generated and
real text split at AUC 1.0000, the gate learned "machine-written", and the sealed
number did not move. Flat, not harmful.

This fails differently. Held-out fire rate falls only to ~4.8%, not to 0%, so the
gate is not memorising a surface tell — it is genuinely moving its decision
boundary. And that movement **hurts**, monotonically, because the boundary is
being pulled toward a definition of "clean" that the sealed gold does not share.

Mixing two labelling standards is worse than adding no data at all, and it is
invisible on any training metric: the gate looks better at staying quiet on the
new documents while getting worse at the job it is measured on.

## What would make this corpus usable

1. **Reconcile the standard on the 782 known conflicts.** They are the only place
   both labelling passes have ruled on the same file, so they are the cheapest
   possible calibration set. Whichever pass is right, the answer applies to the
   other 9,218.
2. **If the new pass is right, the existing training labels are wrong** on those
   documents — which would mean the gate is currently being taught to fire on
   clean files, and is a bigger finding than this experiment.
3. **Re-run this test after reconciliation.** The mechanics are in place; it is
   one command.

It may also bear on the gate diagnosis. That report concluded govdocs2-eval is
adversarially harder than its training half. Label noise of this magnitude —
192 of the 587 eval-overlapping documents in dispute — would produce part of the
same signature, and is worth ruling in or out before the "adversarial by
construction" reading is treated as settled.

## Limitations

- The conflict rate (27.2%) is measured on the 2,874 overlapping documents. Its
  application to the 7,126 non-overlapping ones is an inference, not a
  measurement — nothing has ruled on those files twice.
- Only the document gate was tested. The tag heads were not retrained.
- The corpus was read at the `deep` profile (12,000 characters), matching the
  model under test; 0 read errors across 7,126 documents spanning
  .pdf/.doc/.html/.xls/.ppt.
