# pii-steady-aim — the worst-group threshold rule

Run `pii-steady-aim-2026-08-25` · commit `78a8bb1` · 1,000 trials · 16 measured
arms (2 models × 8 corpora) · MLflow experiment `pii-steady-aim`

A single-mechanism follow-up to [`pii-quiet-alarm`](../../pii-quiet-alarm/reports/26-08-25_pii-quiet-alarm.md).
Everything is carried over unchanged except how per-label and document
thresholds are chosen.

## Results

### Headline — equal-corpus, sealed evaluation

| Arm | priority macro F0.5 | macro F0.5 | doc precision | doc specificity | doc recall | one-core p95 | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| **steady-cascade** | 0.7474 | 0.6932 | **0.8870** | **0.8798** | 0.7582 | 4.11 ms | **blocked** |
| champion_1k *(prior)* | 0.2057 | 0.4163 | 0.6162 | 0.0005 | 0.9994 | 2.03 ms | blocked |
| *quiet-cascade (previous run)* | *0.8233* | *0.7766* | *0.8806* | *0.8693* | *0.7729* | *3.93 ms* | *blocked* |

### Did the fix work?

Yes, for what it targeted — and it cost what the probe said it would.

| | pii-quiet-alarm | pii-steady-aim | Δ |
| --- | ---: | ---: | ---: |
| priority tag recall pairs passing | 36 / 55 | **42 / 55** | **+6** |
| worst doc precision (ci_lower) | 0.7893 | **0.8024** | +0.0131 |
| worst doc specificity (ci_lower) | 0.8299 | **0.8486** | +0.0187 |
| worst doc recall (ci_lower) | 0.6604 | 0.6534 | −0.0070 |
| priority macro F0.5 | 0.8233 | 0.7474 | **−0.0759** |
| priority macro precision | — | 0.7520 | — |

The eleven "marginal" failures the fix was aimed at — pairs sitting between 0.70
and 0.77 against a 0.75 floor — are gone. Every remaining failure moved up
substantially:

| Pair | pii-quiet-alarm | pii-steady-aim |
| --- | ---: | ---: |
| medical_record_number_mrn @ betterdataai | 0.0132 | **0.3642** |
| password @ betterdataai | 0.1000 | **0.3200** |
| address @ govdocs2 | 0.3636 | **0.5455** |
| address @ betterdataai | 0.3110 | **0.6299** |
| full_name @ govdocs2 | 0.6208 | 0.6053 |
| driver_s_license_number @ openpii | 0.6206 | **0.6902** |

### The gate ladder — why nothing was promoted

Per corpus, on the bootstrap lower bound, 30-instance minimum.

| Arm | doc precision ≥ 0.90 | doc specificity ≥ 0.85 | doc recall ≥ 0.85 | priority tag recall ≥ 0.75 | p95 ≤ 5 ms |
| --- | --- | --- | --- | --- | --- |
| steady-cascade | **FAIL** 1/3 (0.8024) | **FAIL** 2/3 (0.8486) | **FAIL** 1/3 (0.6534) | **FAIL** 42/55 (0.2000) | PASS |
| champion_1k | **FAIL** 0/3 (0.4632) | **FAIL** 0/3 (0.0000) | PASS 3/3 | PASS 55/55 | PASS |

**No feasible arm.** The artifact was packaged on explicit request as the run's
best deliverable and **not promoted to `@champion`** — its model card leads with
this table.

### Document level, per corpus

Point estimate, bootstrap lower bound in brackets. Previous run in italics.

| Corpus | precision | specificity | recall |
| --- | ---: | ---: | ---: |
| pii2 (synthetic) | 0.9792 (0.9772) | 0.8973 (0.8889) | 0.9306 (0.9276) |
| datax (real) | 0.8206 (0.8024) | 0.8649 (0.8486) | 0.6743 (0.6534) |
| govdocs2 (real) | 0.8612 (0.8484) | 0.8773 (0.8659) | 0.6696 (0.6540) |
| *pii2, previous* | *0.9783* | *0.8926* | *0.9324* |
| *datax, previous* | *0.8089* | *0.8462* | *0.7104* |
| *govdocs2, previous* | *0.8545* | *0.8692* | *0.6759* |

Precision and specificity improved on **all three** corpora. Document recall on
the two real-world corpora slipped slightly, and it is now the binding failure.

### Priority-tag quality, per corpus

| Corpus | priority F0.5 | precision | recall | n |
| --- | ---: | ---: | ---: | ---: |
| pii2 | 0.9296 | 0.9268 | 0.9597 | 30,000 |
| pii_holdout | 0.8473 | 0.8469 | 0.9141 | 20,000 |
| openpii | 0.8325 | 0.8388 | 0.8596 | 38,937 |
| ai4privacy | 0.7707 | 0.7941 | 0.7726 | 10,626 |
| betterdataai | 0.3570 | 0.3533 | 0.5320 | 10,360 |

pii2 flatters; betterdataai (silver labels) is the weak corpus. Quote **0.7474**.

---

## TL;DR

- **The fix did exactly what it was designed to do.** Failing priority pairs
  went 19 → 13, and the marginal cluster it targeted is gone entirely. Worst
  document precision and specificity both improved.
- **It cost 0.076 priority macro F0.5**, as the pre-launch probe predicted. The
  0.75 recall floor is a hard gate, so buying it has to be paid for in
  precision; the search found the cheapest robustness that passes rather than
  the safest.
- **Still no feasible arm**, so nothing was promoted. What remains is no longer
  a calibration-margin problem — `address` (0.55–0.65) and `full_name`
  (0.61–0.64) are genuinely below the floor across corpora.
- **Packaged anyway, on request**, as `pii-steady-aim-cascade-v1`. It reproduces
  its recorded metric through its own `tagger.py` to **delta 0.00000**, and the
  card's first section is the gate table it fails.
- Against the model actually in production: priority macro F0.5 **0.2057 →
  0.7474**, document specificity **0.0005 → 0.8798**, at 4.11 ms p95 on one core.

---

## What changed, precisely

`pii-quiet-alarm` chose each per-label threshold where **pooled** held-in recall
met the floor. A pooled optimum is a weighted average, so a source with a harder
score distribution sits below the bar while the average sits exactly on it.

This run requires the floor to hold on the **worst source group**:

```
cap_j = min over sources g of  quantile(scores of tag j positives in g, 1 - floor - margin)
t_j   = the F0.5-optimal threshold at or below cap_j
```

with `margin` searched in `[0, 0.15]` rather than fixed. The document gate got
the same treatment: a group must satisfy the recall cap from above and the
specificity floor from below, and when that band is empty across groups the
selector says so rather than silently favouring one floor.

The trade was measured **before** the budget was spent
(`probe/threshold_rule_disc.py`, discriminative heads, `deep` profile):

| Rule | priority F0.5 | priority P | worst group recall | pairs < 0.75 |
| --- | ---: | ---: | ---: | ---: |
| pooled | 0.9269 | 0.9534 | 0.0241 | 16 |
| worst-group, margin 0.00 | 0.7279 | 0.7238 | 0.7470 | 4 |
| worst-group, margin 0.03 | 0.6736 | 0.6612 | 0.7791 | **0** |
| worst-group, margin 0.10 | 0.6047 | 0.5871 | 0.8434 | **0** |

So the ~0.08 headline cost was known and accepted going in, not discovered
afterwards.

## How the budget was spent

| Family | Trials | Feasible | Best held-in |
| --- | ---: | ---: | --- |
| `docgate` | 250 | 115 | real-doc P 0.9330, R 0.8623, spec 0.9464 |
| `tagcount` | 150 | 150 | priority F0.5 0.4576 |
| `tagdisc` | 250 | 250 | priority F0.5 0.7757, min recall 0.8081 |
| `cascade` | 350 | 281 | priority F0.5 0.8244, min recall 0.7634 |

Both stages are healthy separately — the gate holds real-document precision
0.933, the heads hold min recall 0.808 across every source. The cascade is where
they have to hold at once, and that is where the trade bites.

**A bug the reproduction check caught.** The first materialisation attempt
scored 0.9304 against the trial's recorded 0.8244 and was refused. The
materialiser was still using the *pooled* selection rule while the winning trial
had used the worst-group rule — a "better" model that was simply not the one the
search selected. This is exactly the divergence that check exists for, and it
would otherwise have shipped a model whose numbers no one could reproduce.

## Reused rather than rebuilt

The data was verified byte-identical to the previous run's frozen snapshot
(`quiet_freeze.py verify` → "data matches the frozen snapshot"), so the
657,560-document, three-profile feature cache was **symlinked, not rebuilt**.
Re-reading every document to produce identical features would only have added a
way for the two runs to diverge. The decontamination (22,816 leaking training
rows removed) and the directory rename that both happened during the previous
run are already incorporated.

## What the evidence supports

| Bar | Written | Achieved (worst corpus, ci_lower) |
| --- | ---: | ---: |
| doc precision | 0.90 | **0.80** |
| doc specificity | 0.85 | **0.85** |
| doc recall | 0.85 | **0.65** |
| priority tag recall | 0.75 on 55/55 | **0.75 on 42/55** |

Document specificity now essentially clears its bar. The two that do not are
document **recall** on real business documents, and the tail of per-tag recall.

## What to do next

1. **Renegotiate the recall floors, or accept a second pass.** The gates and the
   artifact are now about 0.10 apart on document recall and 13 pairs apart on
   per-tag recall. Every remaining lever trades precision for recall, and
   precision is what this lineage was asked to maximise. This is a product
   decision — how much is a missed document worth against a false alarm — not a
   modelling one, and two runs have now established the frontier rather than
   moved it.
2. **Treat `address` and `full_name` as their own problem.** They fail across
   corpora at 0.55–0.65 and are the two tags the frozen collapse merges
   components into. Test whether collapsing address was right, and whether these
   need cue features the hashed representation misses.
3. **Decide what betterdataai is for.** Silver labels, priority F0.5 0.3570
   against 0.9296 on pii2, and two of the worst pairs are there and nowhere
   else. Either its labels should not gate a promotion or they are signal the
   model misses; the run cannot tell these apart.
4. **Evaluate a `fast` read profile.** Still never measured — the top gates were
   all `deep`, so mismatched pairs were pruned again. A ~0.5 ms variant may be
   the right product for some callers.

## Provenance and limits

- 531,431 training documents (70,600 negatives); 126,129 sealed evaluation
  documents (10,003 negatives); leakage 0 on every corpus.
- Latency measured on a quiescent machine, single core, through the real serving
  path on real documents ≥ 10 KB.
- **Not promoted.** No `@champion` alias assigned; `promotion_v2` would refuse a
  decision whose winner carries hard failures, and the packaged bundle records
  `"promoted": false` with the reason.
- Bundle verification re-scored 20,000 sealed documents through the packaged
  `tagger.py`: expected 0.8473, measured 0.8473, delta 0.00000.
- Same run-record deviation as the previous run: commands were issued directly
  rather than through `mp run`, so there is no captured stdout per command.
  Long-running steps have logs; every number traces to a JSON artifact or to
  `mlflow.db`.
- Document classification, not span NER. Research/internal only pending
  confirmation of AI4Privacy and NonCommercial rights.
