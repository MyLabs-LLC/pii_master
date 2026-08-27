# How to build this model

Everything needed to reproduce `pii-cascade-p80r70-v1` and `pii-cascade-p80r90-v1`
from source, and to build a variant at a different operating point.

These models are **threshold re-selections**. No training happens anywhere in this
document. If you are looking for how the underlying cascade was *trained*, that is
step 0 and it lives in a different project.

---

## What this model actually is

A frozen cascade plus 61 numbers.

```
document text
   │
   ├─ hash to 262,144 binary features (first 12,000 characters)
   │
   ├─ GATE:  score = X · gate_weights + intercept
   │         if score < gate_threshold  ->  emit nothing, stop        (~60% of docs)
   │
   └─ HEADS: scores = X · tag_weights.T          (61 scores, always all 61)
             emit tag j where scores[j] >= tag_thresholds[j]
```

`tag_thresholds` is the only thing these models change. `gate_weights`,
`gate_intercept`, `gate_threshold`, `tag_weights`, the feature hashing and the
read window are inherited byte-identically from `cascade_scorecard61`.

That is why the latency is identical to its parent: the head scores are computed
for all 61 tags unconditionally, so moving a threshold changes which comparisons
come out true, not how many are done.

---

## Step 0 — the inputs you need

| | where | note |
| --- | --- | --- |
| The parent cascade | `projects/pii-scorecard-60/models/cascade_scorecard61` | frozen; never refit here |
| The 61-label catalogue | `projects/pii-scorecard-60/cache/catalogue.json` | built by `training/h2h_scorecard_catalogue.py` from the GAIA scorecard CSV |
| The feature cache | `projects/pii-scorecard-60/cache/*.npz` | features + label gold, per corpus |
| Training corpora | `/home/lence/workspace/data/1-train` | 8 corpora, 531,431 rows |
| Sealed corpora | `/home/lence/workspace/data/2-eval` | 8 corpora, 126,129 rows — **selection never touches these** |

If the catalogue or cache is missing, rebuild them first:

```bash
bash "$SKILL_DIR/scripts/run.sh" training/h2h_scorecard_catalogue.py --check   # coverage, writes nothing
bash "$SKILL_DIR/scripts/run.sh" training/h2h_scorecard_catalogue.py           # writes catalogue + cache
```

Only the **label** arrays are rebuilt. Feature arrays are hashed from document
text and cannot depend on the label space, so they are copied through — otherwise
you would re-read 657,560 documents to produce byte-identical output.

---

## Step 1 — check the target is reachable before building anything

A threshold can only move a tag along its own precision-recall curve. So for any
target `(P, R)` there is a definite per-tag answer: does the curve enter the box?

```bash
bash "$SKILL_DIR/scripts/run.sh" \
  /path/to/target_feasibility.py     # sweeps each tag on the calibration carve
```

For the two targets shipped here, measured on the calibration carve:

| target | reachable by threshold alone | needs a better model or better gold |
| --- | ---: | ---: |
| P ≥ 0.80, R ≥ 0.70 | 54 of 61 | 7 |
| P ≥ 0.80, R ≥ 0.90 | 38 of 61 | 23 |

**Do this first.** It costs a minute and it separates "pick a better threshold"
from "the head is not good enough", which need completely different work.

---

## Step 2 — select the thresholds

```bash
bash "$SKILL_DIR/scripts/run.sh" training/h2h_target_box.py \
    --precision 0.80 --recall 0.90 --name cascade_p80r90
```

Per tag, on the **calibration** side of `quiet_fit.carve_holdin`:

1. sweep the precision-recall curve;
2. find points with `P ≥ p_target` **and** `R ≥ r_target`;
3. among those take the **F0.5 optimum** — precision-led, so the choice inside the
   box is not arbitrary;
4. if the curve never enters the box, keep the tag's best F0.5 point and record it
   as `unreachable` — reported, not silently parked somewhere flattering;
5. under 30 calibration positives, `not_measurable`.

Writes `projects/pii-target-8070/models/<name>/` and a per-tag record in
`probe/<name>_selection.json`.

> **The carve rule is not what you would guess.** `quiet_fit.load` derives
> `uid_hash` from the **corpus name and the row ordinal**, not from the document's
> `uid`. Hashing the uid instead produces a valid but *different* split that
> disagrees on ~85% of calibration rows — which silently selects thresholds on rows
> the model was fitted to. If you write your own loader, reproduce
> `carve_fit_mask` in `training/v5_finetune.py` exactly, and verify it
> element-wise against `carve_holdin`.

---

## Step 3 — score on the sealed corpora

```bash
bash "$SKILL_DIR/scripts/run.sh" training/h2h_score_any.py \
    --model projects/pii-target-8070/models/cascade_p80r90 \
    --name cascade_p80r90 \
    --latency projects/pii-scorecard-60/evaluations/latency_scorecard61.json \
    --out projects/pii-target-8070/evaluations/arm_cascade_p80r90.json
```

Uses `training/h2h_eval.py` — the fixed evaluator — so the result joins every
other arm in this repo on the same terms. **Do not modify that file.**

Latency is *carried* from the parent rather than re-measured, for the reason in
"What this model actually is". If you change anything other than thresholds, that
argument no longer holds and you must re-benchmark on one core.

---

## Step 4 — make precision gateable

The evaluator exports one metric per tag × corpus scope: `recall`. That is correct
— most corpora here carry positive-only tag gold, where precision is meaningless —
but it means a precision gate would be `NOT_MEASURABLE` everywhere and block every
arm while measuring none.

```bash
bash "$SKILL_DIR/scripts/run.sh" training/h2h_precision_scopes.py \
    projects/pii-target-8070/evaluations/arm_cascade_p80r90.json
```

This derives precision scopes from the `tp`/`fp` the evaluator already wrote, for
the five corpora whose `can_measure_precision` is true. It changes no existing
number and does not touch the evaluator.

**The interval differs and you should know it.** Recall scopes carry a document
*bootstrap* lower bound; these carry a **Wilson score** lower bound, because per-tag
precision has no bootstrap behind it. Two estimators in one policy.

---

## Step 5 — apply the policy

```bash
mp decide --policy projects/pii-target-8070/policy_p80r90.yaml \
          --arms projects/pii-target-8070/evaluations/arm_*.json \
          --suite projects/pii-target-8070/suite.yaml \
          --out projects/pii-target-8070/decision/p80r90.json
```

Note `--arms` is `nargs="+"`: pass every arm to **one** flag. Repeating the flag
overwrites, and a decision silently taken over a single arm looks exactly like a
decision taken over five.

---

## Step 6 — package

```bash
uv run --python 3.13 "$SKILL_DIR/scripts/package_champion.py" \
    projects/pii-target-8070/package/package_p80r90.json
```

The packager refuses to write if the model card still has template tokens, if a
declared file is missing, if there is no entry point, or if the verification does
not reproduce its claim within tolerance. **If verification fails, find what
diverged — do not widen the tolerance.**

Verification re-scores a sealed corpus **through the bundle's own `tagger.py`**,
not through the training harness. That is the entire point: the bundle ships its
own loader and thresholds, and each is a place a delivered model can quietly
differ from the one that was measured. Both shipped bundles reproduce the
evaluator to six decimal places; the residual is float16 weight storage.

---

## Building a different operating point

Change the box and re-run steps 2–6:

```bash
bash "$SKILL_DIR/scripts/run.sh" training/h2h_target_box.py \
    --precision 0.90 --recall 0.60 --name cascade_p90r60
```

Two things to expect:

- **Higher precision targets shrink measurability.** A model that fires less has
  fewer tag × corpus pairs with 30+ predictions, and a pair with too few
  predictions is `NOT_MEASURABLE` rather than passing. Compare **absolute** pass
  counts across arms, never pass rates.
- **Relaxing recall does not reliably raise micro F1.** Going from R ≥ 0.90 to
  R ≥ 0.70 brought 16 more tags into the box but *lowered* micro F1 (0.8474 →
  0.8394), because the extra tags are rare and contribute few pooled decisions
  while the relaxation lets common tags surrender recall they did not need to.

## What will not help

Measured in this repo, so you do not have to repeat it:

- **Adding a fine-tuned transformer.** A 68M ModernBERT was fine-tuned on all 8
  corpora, distilled to a static token table with model2vec, trained into a
  document tagger and fused in. Fused into the parent it added +0.03 micro F1;
  fused into `p80r70` it made things **worse** (0.8394 → 0.8373) and cost 59% more
  latency. The fusion never once chose the content model on its own, and never
  chose `or` — its only contribution is a precision veto, which thresholds already
  provide more cheaply.
- **Chasing the last precision failures with better modelling.** An audit of the
  39 failures found ~31 are gold problems, not model problems. `email` reaches
  1.0000 precision on four corpora and ceilings at 0.5779 on `betterdataai` at any
  threshold.
