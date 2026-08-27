# A fine-tuned transformer bought +0.03 micro F1. Re-picking thresholds bought +0.11.

## Results

Four stages — fine-tune, distil, tag, fuse — measured on the eight sealed
`data/2-eval` corpora under the declared policy.

| | `cascade_scorecard61` | **`v5d-fused`** | Δ |
| --- | ---: | ---: | ---: |
| **micro F1** *(the ranker)* | 0.7299 | **0.7595** | **+0.0296** |
| **micro precision** | 0.6360 | **0.6926** | **+0.0566** |
| macro F0.5 | 0.6230 | 0.6282 | +0.0052 |
| macro F2 | **0.6651** | 0.6496 | −0.0155 |
| recall, macro over catalogue | **0.6773** | 0.6367 | −0.0406 |
| worst measurable priority recall | **0.6267** | 0.5881 | −0.0386 |
| priority macro F0.5 | 0.7980 | **0.8020** | +0.0040 |
| document recall / precision / specificity | 0.7975 / 0.8956 / 0.8792 | *identical* | — |
| **priority gates passed** | **25/55** | **22/55** | **−3** |
| **one-core p95** | **4.0290 ms** | **6.2785 ms** | **+2.25 ms** |

`NO FEASIBLE ARM`. Nothing promoted.

### The fusion tells you what the content model is worth

```
fusion chosen per tag:  {'and': 37, 'cascade': 24}
document gate fusion:   cascade
```

Four strategies were available per tag — `cascade`, `content`, `or`, `and` — each
scored by F0.5 on the calibration carve. The result is unambiguous:

- **`content` was never chosen.** Not once in 61 tags. There is no tag the content
  model answers better on its own than the cascade does.
- **`or` was never chosen.** The content model finds nothing the cascade misses;
  if it did, `or` would have won somewhere.
- **`and` won 37 times** — an intersection, which can only *remove* firings. The
  content model's entire contribution is as a **precision veto**.
- **The document gate stayed the cascade's**, so the customer-facing question —
  does this document contain PII — was not improved at all.

That is consistent with every number above: precision up, recall down, F2 down.

### The comparison that matters

The same day, on the same corpora and evaluator, re-selecting the cascade's 61
thresholds into a (precision, recall) box — no training, no new data, no extra
latency — produced:

| | micro F1 | micro precision | p95 |
| --- | ---: | ---: | ---: |
| `cascade_scorecard61` | 0.7299 | 0.6360 | 4.029 ms |
| `v5d-fused` *(this run)* | 0.7595 | 0.6926 | 6.279 ms |
| **`cascade_p80r70`** *(thresholds only)* | **0.8394** | **0.9165** | **4.029 ms** |

**The threshold re-selection beat the entire transformer chain by 0.08 micro F1
and 0.22 micro precision, at zero latency cost and zero training.** The chain cost
a GPU fine-tune over 451,548 documents, a distillation, 58 minutes of head fitting,
and 2.25 ms of every future inference, to arrive at less than a third of the gain.

## TL;DR

- **The chain works end to end and passes its serving constraint**: measured fused
  p95 **6.279 ms** against the 8 ms requirement.
- **Every stage met its checkpoint.** v5a improved 50 of 61 tags (mean +0.41);
  v5b's static table separates identifier tokens better than the span-stage table
  did (probe cosine 0.0848); v5c selected all 61 thresholds.
- **And the result is still small.** +0.0296 micro F1, −3 priority gates, +56%
  latency.
- **The content model is a precision veto and nothing else** — `and` on 37 tags,
  `content` and `or` on none, and no contribution to the document gate.
- **Re-picking thresholds was ~4× more valuable**, free, and instant.
- **The premise was reasonable and the measurement refutes it.** The hypothesis was
  that contextual signal would find identifiers a bag of hashed n-grams cannot.
  The fusion says the cascade was already finding them; what it lacked was
  knowing when to stay quiet, and a threshold answers that far more cheaply than
  a transformer.

---

| | |
| --- | --- |
| Date | 2026-08-26 |
| Project | `projects/pii-content-v5` |
| Run | `26-08-26-1` |
| Request | *"finetune the LLM kalyan-ks/ettin-68m-nemotron-pii on all 8 training datasets, model to vec it with token based not sentence based, then train that model on the same 8 training sets, then test adding that model to this model"* |
| Scope | 4 stages; 1 fused arm × 8 sealed corpora; 61 labels |
| Hardware | RTX 4080 (12 GB) for v5a/v5b; 1 CPU core for every published latency |
| Feasibility | recorded `unknown` before the budget — no ceiling existed for the fused metric |
| Outcome | chain completed, serving constraint met, **blocked** by the recall gate, nothing promoted |

## What each stage actually produced

**v5a — fine-tune.** Two stages, because only `85593_pii_trainset` carries
token-level gold (682,692 spans, 0 dropped: every `tag_id` already matched the
61-label catalogue). Span stage on 46,223 fit-side documents, then a 61-way
document head on 451,548. Checkpoint, both models on the *same* calibration rows:
50 of 61 tags improved by >0.02, 8 regressed, mean delta **+0.4057**.

The span stage alone caused severe single-corpus forgetting — CVV 0.8780 → 0.0581,
religion 0.8302 → 0.1290 — and the document stage repaired most of it (CVV back to
0.4545). That repair is the reason the two-stage design was worth the extra hour.

**v5b — distil.** 50,254 × 256, float16, **27.29 MB**, 7 seconds. Probe-token
cosine mean **0.0848**, max 0.6416 — the table did not collapse, which was the
property v5c depended on.

**v5c — the tagger.** 531,431 × 512 `[mean ‖ max]` features; fit 451,548 / calib
79,883, matching `carve_holdin` exactly. 61 heads in 3,502 s, all 61 thresholds
selected.

**v5d — fusion.** Per-tag strategy on the calibration carve, plus a separate
strategy for the document gate. Results above.

## Two corrections this run had to make about itself

**The feasibility probe was wrong about the gated tags.** It read 0.0000 on
`ssn`/`mrn`/`itin`/`cvv`/`health_plan_beneficiary_number` and that was reported as
the central risk. It was an artifact of that probe's corpus, where those tags had
2–12 gold documents. Re-measured with real support, the as-shipped model already
scored SSN 0.5730, ITIN 0.8698, CVV 0.8780. The risk was never where the probe put
it.

**The carve was wrong and would have invalidated the chain silently.**
`quiet_fit.load` derives `uid_hash` from corpus name and row ordinal, not from the
document's `uid`. The obvious implementation — hash the uid — produces a valid but
*different* split that disagrees on ~85% of calibration rows, so the encoder would
have been fine-tuned on the rows the thresholds were later selected on. Every count
would still have looked right. Caught 15 minutes into v5a, fixed, verified
element-wise against `carve_holdin` over all 531,431 rows (451,548 fit both sides),
and the run restarted.

## Reading the 6.279 ms honestly

It is a **measurement**, not the sum of two measurements. An earlier version of
this chain would have reported 4.029 + 2.533 = 6.562 ms by arithmetic; `v5_fuse`
now times the whole serving call — featurise, both arms, fuse — on one core over
the same ≥10 KB documents `h2h_bench` uses. The real number came in slightly
*below* the additive estimate.

The content path reads **6,000 characters, not the cascade's 12,000**, and that is
not a compromise: the tokenizer truncates at 1,024 tokens and 6,000 characters
already yields 1,024. At 12,000 the same 1,024 tokens cost 6.946 ms instead of
2.533 ms, and the fused arm would have missed even a 10 ms budget. Everything past
~6,000 characters is scanned and discarded.

## What is still open

- **Whether a content model helps *at all* is now answered, and the answer is
  "barely".** Before spending more here, the cheaper lever — threshold selection —
  should be exhausted, and it has just been shown to be worth four times as much.
- **The obvious follow-up is untested**: fusing the content model into
  `cascade_p80r70` rather than `cascade_scorecard61`. The two gains may be
  independent — one adds precision by vetoing, the other by moving thresholds — or
  they may be the same gain twice. This run cannot say.
- **`content` never winning is worth one more look.** It could mean the signal is
  genuinely absent, or that a 512-d `[mean ‖ max]` aggregation over static token
  vectors is too lossy to expose it. The distillation preserved token separation
  (cosine 0.0848), so the aggregation is the likelier suspect.
- **v5a is a teacher with no product.** The fine-tuned encoder is a genuinely
  better PII model than the upstream checkpoint on this catalogue (+0.41 mean F1),
  and nothing in this repo can serve it. If a GPU serving path ever exists, it is
  the most capable artifact here.
