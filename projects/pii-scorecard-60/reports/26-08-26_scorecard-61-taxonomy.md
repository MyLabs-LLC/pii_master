# The scorecard's 61 tags cost nothing on the ranker and four gates on the gate

## Results

The cascade refit on the GAIA scorecard's tag taxonomy — 60 tags from
`Scorecard - GAIA Catalog(Rasool-PII-PCI-PHI).csv`, less `routing_number`, plus
`swift_code` — scored on the eight sealed `data/2-eval` corpora under the
unchanged evaluator.

| | `cascade_scorecard61` | for reference |
| --- | ---: | --- |
| labels | **61** | 58 before |
| thresholds selected | **61 of 61** | 57 of 58 before — the one disabled head was `routing_number`, now dropped |
| **micro F1** *(the ranker)* | **0.7299** | |
| micro precision | 0.6360 | |
| macro F0.5 | 0.6230 | |
| **macro F2** | **0.6651** | target 0.6641, ceiling 0.6754 |
| macro precision / recall | 0.6244 / 0.6773 | |
| worst measurable priority recall | 0.6267 | |
| prediction rate | 0.8181 | |
| document recall / precision / specificity | 0.7975 / 0.8956 / 0.8792 | |
| **one-core p95** | **4.0290 ms** | budget 5 ms — **PASS** |

**The target was met.** Macro F2 0.6651 against a target of 0.6641 and a measured
ceiling of 0.6754, so changing to the scorecard taxonomy cost the ranker nothing
detectable. The document-level numbers are identical to the 58-label lineage to
four decimals, which is expected rather than surprising: the gate is upstream of
every tag head and its refit is unchanged.

### The four restored tags, sealed

`given_name`, `family_name`, `middle_name` and `street_number_and_name` were
folded into `full_name` and `address` in the previous lineage. Restored, all four
cleared the recall floor on the training carve, and on the sealed corpora they
look like this — precision first, since that is what ranks:

| tag | corpus | support | P | R | F0.5 |
| --- | --- | ---: | ---: | ---: | ---: |
| `given_name` | `20000_pii_holdout` | 8,892 | 0.8367 | 0.9701 | 0.8603 |
| | `38937_openpii` | 29,335 | 0.7633 | 0.9974 | 0.8009 |
| | `30000_pii2_eval` | 5,453 | 0.6561 | 0.9118 | 0.6951 |
| | `10626_ai4privacy` | 3,501 | 0.3709 | 0.9797 | 0.4235 |
| | `10360_betterdataai` | 1,072 | **0.1382** | 0.7220 | 0.1649 |
| `family_name` | `20000_pii_holdout` | 8,337 | 0.7555 | 0.9745 | 0.7911 |
| | `38937_openpii` | 26,552 | 0.6901 | 0.9977 | 0.7355 |
| | `10360_betterdataai` | 247 | **0.0426** | 0.6964 | 0.0525 |
| `middle_name` | `20000_pii_holdout` | 6,039 | **0.9660** | 0.9631 | 0.9654 |
| | `38937_openpii` | 29,335 | 0.8047 | 0.9907 | 0.8361 |
| | `30000_pii2_eval` | 2,288 | 0.5099 | 0.9135 | 0.5593 |
| | `10360_betterdataai` | 14 | **0.0036** | 0.0714 | 0.0045 |
| `street_number_and_name` | `20000_pii_holdout` | 6,222 | 0.7757 | 0.9724 | 0.8084 |
| | `38937_openpii` | 15,267 | 0.6373 | 0.9954 | 0.6867 |
| | `10360_betterdataai` | 254 | **0.1251** | 0.7362 | 0.1500 |

Recall is 0.87–0.998 everywhere except betterdataai. Precision splits the corpora
sharply, and the split is the one the feasibility probe predicted from gold alone
before anything was fitted.

**`full_name` was not damaged by losing its subtypes**: 0.9739 / 0.9020 / 0.9331
precision on `pii_holdout` / `openpii` / `betterdataai`. Restoring the subtypes
took nothing away from the parent tag.

### Both declared policies

| policy | verdict |
| --- | --- |
| **headline** — recall gates, ranks micro F1 → micro precision → macro F0.5 → macro F2 | **blocked**, 25/55 priority gates, fails at `password@10360_betterdataai` |
| **contra-view** — document precision/specificity/recall + tag recall ≥ 0.75 | **blocked**, doc constraints fail on `govdocs2`, 44/55 tag gates |
| one-core p95 ≤ 5 ms | **PASS** on both |

`NO FEASIBLE ARM`. Nothing promoted, nothing packaged — the same 25/55 and the
same failing scope as the 58-label `v4`, so the taxonomy change neither created
nor fixed a gate failure.

## TL;DR

- **61 labels**: the scorecard's 60, less `sensitive_pci_routing_number` (1
  training row, 0 evaluation rows — the head that was already disabled), plus
  `sensitive_pci_swift_code` (542 / 130 rows, absent from the scorecard but a
  working detector).
- **Every scorecard tag has gold.** The coverage check found 0 tags with no gold
  anywhere, and exactly one sensitive tag in the gold that is not on the
  scorecard — `routing_number`, dropped deliberately.
- **The taxonomy change cost the ranker nothing**: macro F2 0.6651 against a
  0.6641 target and a 0.6754 ceiling; document-level metrics unchanged to four
  decimals.
- **All 61 thresholds were selectable** — none fell below the support floor,
  against 57 of 58 before.
- **The restored tags work where the gold is consistent** and fail where it is
  not, exactly as predicted before fitting. `middle_name` reaches P 0.9660 on
  `pii_holdout` and P 0.0036 on `betterdataai`, whose name rows are 85.5%
  `full_name`-only.
- **Still blocked**, 25/55 priority gates, on `password@betterdataai` — the same
  scope that blocked the previous lineage. This run did not touch that.
- **Ranking now leads with precision.** On instruction, the headline policy ranks
  micro F1 → micro precision → macro F0.5 → macro F2. Recall was not demoted; it
  is the hard gate and always was.

---

| | |
| --- | --- |
| Date | 2026-08-26 |
| Project | `projects/pii-scorecard-60` |
| Run | `26-08-26-1` |
| Request | *"rebalance this model based on these 60 pii tags"* → label space rebuilt to the scorecard; then *"micro_f1 and precision to be the focus, but still have high recall, F@3"* |
| Scope | 1 model × 8 sealed corpora; 61-label catalogue; no hyperparameter search |
| CPU budget | 1 core for the latency; training used the machine |
| Feasibility | `plausible`, ceiling 0.6754 vs target 0.6641 — `feasibility.json` |
| Outcome | target met on the new taxonomy; **blocked** by the recall gate; nothing promoted |
| Approval | `approvals/taxonomy-rebuild-61.json` |

## How the label space was built

`training/h2h_scorecard_catalogue.py` reads the 60 tags **from the scorecard CSV**
rather than transcribing them into code, so the catalogue cannot drift from the
scorecard by somebody editing one and not the other. It slugs
`PII, Driver's License Number` to `sensitive_pii_driver_s_license_number` and
checks every resulting tag against the gold before writing anything: 0 tags with
no gold, 1 gold tag off the scorecard.

**Only the labels were rebuilt.** A cache entry holds features
(`indices_deep` / `indptr_deep`) and gold (`label_cols` / `label_indptr` /
`doc_target` / `tag_complete`). Features are hashed from document text and cannot
know the catalogue exists, so re-extracting them would mean re-reading 657,560
documents to produce byte-identical arrays. They are copied through and only the
label arrays are recomputed under the new index.

`projects/pii-quiet-alarm/cache` — which the 128 published results of the previous
project depend on — was read and never written.

### The three cache roots

`quiet_fit` and `h2h_score` bind `CACHE_ROOT` at import time, so rebinding
`quiet_cache.CACHE_ROOT` alone would move `load_catalogue()` to the new catalogue
while leaving those two reading the old one. That mismatch does **not** raise —
the arrays are the same shape until the label count changes — it would silently
score a 61-label model against 58-label gold. All three names are rebound in one
place and the catalogue's label count is asserted afterwards, before any fit.

## What changed in the ranking, and why it is not a demotion of recall

The headline policy now ranks **micro F1 → micro precision → macro F0.5 → macro
F2**. It previously ranked macro F2 alone.

Recall sits where it always did: constraint 1, hard, on the bootstrap lower
bound, `min_support: 30`, and nothing ships under it. What changed is that recall
is no longer *also* the ranker. Holding it in both places pushes thresholds toward
the bottom of their curves, because F2 keeps paying for recall long after
precision has collapsed — measured in this lineage the day before, two tags sat at
precision 0.2150 and 0.0051 while *passing* their recall floors, and the macro-F2
ranking saw nothing wrong with either.

Macro F2 is still ranked on, last, and reported on every row: the macro-F0.5 /
macro-F2 gap is how much recall the precision-led ordering costs, and a report
that omits it is asserting the cost was zero.

One correction worth recording: the first version of this policy named the metric
`precision`. This evaluator emits `precision_micro` and
`precision_macro_catalogue`, so the preference matched nothing, reported `N/A`,
and silently stopped breaking ties. It is now `precision_micro`.

## What is still open

- **`password@10360_betterdataai` blocks the gate**, as it did for `v4`. Nothing
  in this run addressed it; the taxonomy change is orthogonal to it.
- **`betterdataai` cannot judge the restored tags.** 85.5% of its name rows carry
  `full_name` alone, so `middle_name` there is unlearnable as labelled (P 0.0036).
  It is one of eight corpora under equal-corpus aggregation, so it drags every
  macro average that includes it. Either its name labelling is corrected or it is
  declared unable to measure the subtype question — a gold decision, not a tuning
  one.
- **`swift_code` was kept off-scorecard.** If the scorecard is meant as an
  exhaustive contract rather than a minimum, that decision should be reversed
  deliberately rather than left as a footnote.
- **Nothing has cleared a gate in this project's history.** Four runs, sixteen-plus
  models. The recall gate at 0.90 `ci_lower` across every measurable tag×corpus
  pair may be the thing to renegotiate, and that is a policy conversation with
  evidence now attached to it.
