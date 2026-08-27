# Pipeline spec — pii-scorecard-60

Rebuild the sensitive-data document tagger onto the **GAIA scorecard's
authoritative tag taxonomy** — the 60 tags in
`/home/lence/workspace/data/scorecard/Scorecard - GAIA Catalog(Rasool-PII-PCI-PHI).csv`
(42 PII + 10 PCI + 8 PHI) — instead of the 58 collapsed labels the
`pii-head-to-head-v1` lineage was fitted on.

## Objective

| | |
| --- | --- |
| Task family | multi-label document tagging, with a document-level gate in front |
| Metric to optimize | `macro_f2` over the catalogue (equal-corpus aggregation) |
| Direction | maximize |
| Target | **≥ 0.6641** — `pii-cascade-v4-corrected-cap`'s measured number on the 58-label taxonomy. The bar is "change the taxonomy without regressing the ranker", not "beat the old model", because the two are scored over different label sets |
| Feasibility | **plausible**, ceiling ≈ 0.6754, headroom 0.011 — see `feasibility.json` |

## The label space: 61

Derived from the scorecard by two stated edits, and by nothing else:

| | n | |
| --- | ---: | --- |
| scorecard tags matching an existing model label exactly | 56 | carried over unchanged |
| scorecard tags the old lineage **collapsed away** | 4 | **restored** — see below |
| model labels absent from the scorecard | 2 | `routing_number` **dropped**, `swift_code` **kept** |
| **total** | **61** | |

**The four restored tags.** `training/quiet_data.COLLAPSE` folded
`given_name`, `family_name` and `middle_name` into `full_name`, and
`street_number_and_name` into `address`. The scorecard lists all six as separate
tags, so the collapse is reversed here. The raw gold retains the distinction with
heavy support (train: given 215,451 / family 193,810 / middle 154,388 / street
112,343), so this is a re-indexing of existing gold, not new annotation.

**`routing_number` is dropped** — 1 training instance, 0 evaluation instances. It
is the single disabled head behind the old model's `n_enabled_tags=57`, so removing
it changes nothing the model does.

**`swift_code` is kept** despite being absent from the scorecard — 542 training and
130 evaluation instances, measurable on one corpus. The scorecard is treated as a
minimum, not a maximum; deleting a working detector because a spreadsheet omits it
loses real capability.

## What un-collapsing costs, measured before fitting

The collapse existed for a reason: two independent judges agree at F1 0.99 on
whether a name is *present* and at **0.018–0.40** on *which* name tag it is
(`training/quiet_data.py:97`), and `pii-priority-recall-v1` terminated
`target_infeasible` against a judge-agreement ceiling of macro F1 0.5138.

That ceiling does not transfer to this run, and the reason is which corpora it was
measured on. It came from dual-judge agreement over datax documents. `datax` and
`govdocs2` carry **positive-only tag gold** and therefore contribute no
precision-bearing tag metric at all. The name-subtype disagreement lands almost
entirely on corpora that cannot score it.

On the five corpora with **complete** tag gold, a model that fires every subtype
whenever a name is present — the best any model without subtype signal can do —
scores:

| corpus | given | family | middle | street |
| --- | ---: | ---: | ---: | ---: |
| `38937_openpii_pii_eval` | 0.9924 | 0.9714 | 0.9924 | 1.0000 |
| `20000_pii_holdout` | 0.9456 | 0.9305 | 0.8481 | 0.9891 |
| `30000_pii2_eval` | 0.8572 | 0.8804 | 0.5975 | 0.9591 |
| `10626_ai4privacy` | 0.8989 | 0.7436 | 0.3613 | 0.9627 |
| `10360_betterdataai` | 0.4161 | 0.1289 | **0.0081** | 1.0000 |

(F2, oracle presence detection.) `betterdataai` is the outlier because 85.5% of its
name rows carry `full_name` alone; `middle_name` there is unlearnable **as
labelled**, and will remain a bad row whatever is fitted. Its macro F2 is already
the suite's lowest (0.333), so it dilutes rather than dominates.

Folding these into v4's measured per-corpus numbers puts the equal-corpus ceiling
at **0.6754** against today's 0.6641.

## Model / code

| | |
| --- | --- |
| Baseline | `pii-cascade-v4-corrected-cap` (`projects/pii-head-to-head-v1/models/cascade_v4`), macro F2 0.6641 on 58 labels — a reference point, not a gate opponent |
| Architecture | unchanged: document gate → per-tag heads, NumPy only, `score_mode="sum"`, `deep` profile, 12,000-character read window |
| Gate | refit balanced + regularised, `alpha=1e-2`, `balance=equal` — the lineage that produced `cascade_balanced` |
| Heads | 61, refit with arm B's head hyperparameters, unchanged |
| Thresholds | `training/h2h_thresholds_v4.select_per_label(corrected_cap=True)` — the group-recall cap requires `MIN_TAIL_EVENTS=10` positives in the tail |
| Search | **none.** One deterministic rebuild; no hyperparameter trials |

## Data

| | |
| --- | --- |
| Training | `/home/lence/workspace/data/1-train`, all 8 corpora, 531,431 rows |
| Evaluation | `/home/lence/workspace/data/2-eval`, all 8 corpora, 126,129 rows, sealed |
| Fit / calibration carve | `quiet_fit.carve_holdin`, 15% by stable document hash, **training corpora only** |
| Features | reused. `indices_deep` / `indptr_deep` are label-independent, so only `label_cols` / `label_indptr` are rebuilt against the new catalogue |
| Cache | written to `projects/pii-scorecard-60/cache/`. `projects/pii-quiet-alarm/cache/` is **read, never written** — 128 prior results depend on it |

## What may change

- the catalogue and the label index (that is the point of the run)
- the gate weights and threshold, and all 61 head weights, by refitting
- the 61 per-tag thresholds, by selection on the training calibration carve

Not allowed to change: `training/h2h_eval.py`, `training/quiet_select.py`,
`training/quiet_data.COLLAPSE` (read to be inverted, not edited), the sealed
`data/2-eval` corpora, and the two policy files.

## Budget & guardrails

One deterministic rebuild — no trials. Selection touches training data only; the
sealed corpora are scored once, at the end. Nothing is promoted unless it clears a
declared gate.

## Tracking

MLflow at `sqlite:///projects/pii-scorecard-60/mlflow.db`, experiment
`pii-scorecard-60`. Metrics, CIs and lineage always; the model is saved; the
reference profile is not logged (no monitoring planned yet).

## Decision policy

`policy.yaml` and `policy_precision_view.yaml`, copied unchanged from
`pii-head-to-head-v1` so that a taxonomy change is not confounded with a policy
change.

**The 16-tag priority set is deliberately unchanged**, and the four restored tags
are **not** added to it. The scorecard defines a taxonomy, not a shipping
guarantee; making `middle_name` a hard recall gate would block every arm on a
labelling convention rather than on detection quality. `full_name` and `address`
remain priority tags, so name and address detection is still gated — at the level
the gold can actually settle.

## Reporting

`.md` + `.pdf` + `.commands.txt` + the experiment-log workbook, under
`projects/pii-scorecard-60/reports/`.
