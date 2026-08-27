# Pipeline spec — pii-steady-aim

A single-mechanism follow-up to [`pii-quiet-alarm`](../pii-quiet-alarm/README.md),
which lifted priority macro F0.5 from 0.2057 to 0.8233 and document specificity
from 0.0005 to 0.8693 but cleared **one of five hard gates** and so promoted
nothing. Everything about the objective, the gates, the ranker, the taxonomy
collapse, the suite and the data is carried over unchanged. **One thing
changes**, and this run exists to measure whether it is enough.

## The change

`pii-quiet-alarm` chose each per-label threshold where **pooled** held-in recall
met the floor. A pooled optimum is a weighted average, so a source with a harder
score distribution sits below the bar while the average sits exactly on it.
Nineteen of 55 sealed tag-corpus pairs failed the 0.75 floor and **eleven of
them landed between 0.70 and 0.77** — no margin, not a capability limit.

This run requires the floor to hold on the **worst source group**, with a
searchable safety margin:

```
cap_j = min over sources g of  quantile(scores of tag j positives in g, 1 - floor - margin)
t_j   = the F0.5-optimal threshold at or below cap_j
```

The same rule applies to the document gate, where a group must satisfy the
recall cap from above and the specificity floor from below, with the admissible
band being the intersection across groups. When that band is empty the selector
says so rather than silently favouring one floor.

`margin` is a searched parameter in `[0, 0.15]`, not a constant. The measured
trade on held-in data, discriminative heads, `deep` profile:

| Rule | priority F0.5 | priority P | priority R | worst group recall | pairs < 0.75 |
| --- | ---: | ---: | ---: | ---: | ---: |
| pooled (`pii-quiet-alarm`) | 0.9269 | 0.9534 | 0.8364 | 0.0241 | **16** |
| worst-group, margin 0.00 | 0.7279 | 0.7238 | 0.9017 | 0.7470 | 4 |
| worst-group, margin 0.03 | 0.6736 | 0.6612 | 0.9193 | 0.7791 | **0** |
| worst-group, margin 0.05 | 0.6468 | 0.6332 | 0.9256 | 0.7952 | **0** |
| worst-group, margin 0.10 | 0.6047 | 0.5871 | 0.9396 | 0.8434 | **0** |

Recorded in `probe/threshold_rule_disc.json` and its count-based twin. **The fix
works and it is expensive**: about 26 points of priority precision at the margin
that first clears every pair. That is the honest shape of the trade — the 0.75
conclusive recall floor is a hard gate, so an arm that misses it cannot win at
all, and buying the gate costs precision. The search is feasibility-first
(infeasible arms rank below every feasible one), so it will find the *cheapest*
robustness that passes rather than the safest.

**Expect a lower headline than `pii-quiet-alarm`'s 0.8233 and a champion that
actually ships.** If that trade turns out to be the wrong one for the product,
the lever is the recall floor, not the threshold rule.

## Carried over unchanged

- **Objective, gates and ranker** — `policy.yaml` is identical but for its name:
  document precision ≥ 0.90, specificity ≥ 0.85, recall ≥ 0.85, per-priority-tag
  recall ≥ 0.75, one-core p95 ≤ 5 ms, all on `ci_lower` with `min_support: 30`;
  ranked on priority macro F0.5, then macro F0.5, then latency.
- **The suite** — `suite.yaml` is identical but for its name. Eight corpora,
  three carrying genuine document-level negatives, `data_quality` assessed.
- **The data** — verified byte-identical to `pii-quiet-alarm`'s frozen snapshot
  (`training/quiet_freeze.py verify` → "data matches the frozen snapshot").
  531,431 training rows / 70,600 negatives; 126,129 evaluation rows / 10,003
  negatives. The decontamination and the directory rename that happened during
  the previous run are both already incorporated.
- **The feature cache** — the same 657,560-document, three-profile cache,
  symlinked rather than rebuilt. Reading every document again to produce
  identical features would only add a way for the two runs to diverge.
- **The catalogue and the frozen collapse** — 58 labels; name components →
  `full_name`, street → `address`.

## Candidates and budget

**1,000 trials.** Reallocated toward the family that won last time.

| Family | Trials | Why this share |
| --- | ---: | --- |
| `docgate` | 250 | document gates were already close; the change here is the group-wise cut |
| `tagdisc` | 250 | won decisively last run (held-in F0.5 0.93 vs 0.71) and is where the margin trade is cheapest |
| `tagcount` | 150 | kept as a contrasting mechanism, not expected to win |
| `cascade` | 350 | the promotion candidate, and the only family that sees both stages at once |

Wall-clock cap 12 hours. Training on all 32 cores; **every published latency
re-measured on exactly one core with nothing else running** — the previous run
recorded 8.57 ms for an artifact that measures 1.81 ms quiet, and that lesson is
carried forward as a procedure.

## Known limitation carried forward

`pii-quiet-alarm` evaluated **no `fast` or `std` read-profile cascade at all**:
its top-8 document gates were all `deep`, so profile-mismatched component pairs
were pruned. This run inherits the same pairing rule and will likely inherit the
same blind spot. It is recorded here so the next reader does not mistake absence
of evidence for evidence of absence.

## Ship

Unlike the previous run, this one is expected to package. A gate-passing arm is
promoted through `promotion_v2` and packaged under `dist/` with a
`MODEL_CARD.md`, a self-contained entry point, config, weights, evidence and
`SHA256SUMS`; the bundle must reproduce its recorded sealed metrics **through
its own code** within the 0.01 tolerance. If no arm clears every gate, nothing
is promoted and the report says what the evidence supports — the same rule that
produced no champion last time.
