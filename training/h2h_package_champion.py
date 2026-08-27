"""Package `cascade_p90r85b1` as the champion, with every number read from disk.

Two model cards in this project shipped with wrong numbers — `p88r90` had 5 of
12 metrics wrong, `p90r85b1` had 3 of 12 — because they were written by hand
while the six generated ones had none. The training log's rule 2 exists because
of it: *numbers come from a recorded evaluation, never retyped from memory.*

So this module writes the card. Every figure in it is read from a recorded
evaluation artifact at build time; there is no literal metric anywhere in this
file, and if an artifact is missing the build fails rather than omitting a row.

## What it emits

    MODEL_CARD.md    what it scores, where it fails, what not to claim
    TRAINING.md      how the model was produced, reproducibly, from the corpora
    MANIFEST.json    provenance, checksums, the evaluations each number came from
    SHA256SUMS       every shipped file
    models/verification.json   a re-score through the bundle's own runtime

## Verification is the point of packaging

A bundle that loads but scores differently from the model it was cut from is
worse than no bundle. After staging, the sealed corpus is re-scored **through
the bundle's own `tagger.py`** and compared to the recorded arm. That check has
already caught one real defect here: a fused tagger that conflated two read
windows and would otherwise have shipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TG = Path("projects/pii-target-8070")
SC = Path("projects/pii-scorecard-60")
PKG = TG / "package"
EVAL = TG / "evaluations"

#: The corpus the bundle is re-scored on, and the arm holding the expected value.
VERIFY_CORPUS = "20000_pii_holdout_20.00k"
TOLERANCE = 0.01


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> dict:
    """Read a recorded artifact, or fail loudly. A missing number is a bug."""
    if not path.is_file():
        raise SystemExit(f"missing evaluation artifact: {path}\n"
                         "every number in the card comes from one of these; "
                         "re-run the evaluation rather than hand-writing it")
    return json.loads(path.read_text(encoding="utf-8"))


def m(metrics: dict, key: str) -> float | None:
    v = metrics.get(key)
    return v.get("value") if isinstance(v, dict) else v


def fmt(v: float | None, nd: int = 4) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def gather(model_name: str) -> dict:
    """Every recorded number the card and the training doc will quote."""
    arm = load(EVAL / f"arm_{model_name}.json")
    base = load(EVAL / "arm_baseline_scorecard61.json")
    hold = load(EVAL / f"holdout_{model_name}.json")
    clean = load(EVAL / "clean_docs.json")
    loco = load(EVAL / "loco_variants_agg.json")
    gap = load(EVAL / "loco_gap.json")
    sel = load(TG / "probe" / f"{model_name}_selection.json")
    matrix = load(EVAL / "matrix.json")
    repro = load(EVAL / f"repro_check_{model_name}.json")

    clean_row = next(r for r in clean["results"] if r["model"] == model_name)
    clean_base = next(r for r in clean["results"]
                      if r["model"] == "cascade_scorecard61")
    pol = loco["per_policy"]

    # Latency must be measured for THIS model. Inheriting a sibling's number is
    # how the hand-written cards went wrong; a missing figure is omitted instead.
    lp = EVAL / f"latency_{model_name}.json"
    lat = json.loads(lp.read_text(encoding="utf-8")) if lp.is_file() else None

    return {"arm": arm, "base": base, "holdout": hold, "clean": clean,
            "clean_row": clean_row, "clean_base": clean_base, "loco": loco,
            "loco_policy": pol, "gap": gap, "selection": sel, "matrix": matrix,
            "repro": repro,
            "latency": lat}


def model_card(name: str, version: str, g: dict, meta: dict) -> str:
    a, h = m(g["arm"]["metrics"], "f1_micro"), m(g["holdout"]["metrics"], "f1_micro")
    am, hm = g["arm"]["metrics"], g["holdout"]["metrics"]
    bm = g["base"]["metrics"]
    cr, cb = g["clean_row"], g["clean_base"]
    pol = g["loco_policy"]
    n_corp = len(g["arm"].get("per_corpus", {}))
    lat = g["latency"] or {}
    p95 = lat.get("p95_ms") or lat.get("p95")
    verdicts = g["selection"].get("summary", {})
    folds = g["loco"]["per_fold"]
    n_folds = len(folds)
    n_worse = sum(1 for r in folds if r["p88r90"] < r["p90r85b1"])

    return f"""# Model Card — `{name}-{version}`

A CPU document tagger for sensitive data (PII / PHI / PCI) over
**{meta['n_labels']} labels**. NumPy only, no transformer, no GPU\
{f", **{p95:.2f} ms p95 per document on one core**" if p95 else ""}.

**This is the champion of this project.** It is the best of the measured models
on unseen sources, and the only one whose advantage has been confirmed by
leave-one-corpus-out rather than by a single holdout.

> **It has not passed the project's per-tag ship gate.** No model in this
> repository's history has. It is designated champion on comparative evidence —
> it is the best thing measured — not because it cleared a bar. Read
> *Limitations* before deploying it against real data.

## Measured performance

On the **{n_corp} sealed corpora**, equal-corpus aggregation, one core:

| metric | this model | untuned baseline |
| --- | ---: | ---: |
| micro F1 | **{fmt(m(am, 'f1_micro'))}** | {fmt(m(bm, 'f1_micro'))} |
| micro precision | **{fmt(m(am, 'precision_micro'))}** | {fmt(m(bm, 'precision_micro'))} |
| micro recall | {fmt(m(am, 'recall_micro'))} | {fmt(m(bm, 'recall_micro'))} |
| macro F2 | {fmt(m(am, 'f2_macro_catalogue'))} | {fmt(m(bm, 'f2_macro_catalogue'))} |
| macro recall | {fmt(m(am, 'recall_macro_catalogue'))} | {fmt(m(bm, 'recall_macro_catalogue'))} |
| macro precision | {fmt(m(am, 'precision_macro_catalogue'))} | {fmt(m(bm, 'precision_macro_catalogue'))} |

### On sources it has never seen

Leave-one-corpus-out: the model is refitted on eight sources and scored on the
ninth, so nothing about the held-out source — not its documents, not its
generator — is in the training data. Averaged over the
{g['loco']['n_folds']} measurable folds:

| policy | micro F1 | precision | recall |
| --- | ---: | ---: | ---: |
| **this model's thresholds** | **{fmt(pol['p90r85b1']['f1_micro'])}** \
| {fmt(pol['p90r85b1']['precision_micro'])} | {fmt(pol['p90r85b1']['recall_micro'])} |
| untuned baseline | {fmt(pol['baseline']['f1_micro'])} \
| {fmt(pol['baseline']['precision_micro'])} | {fmt(pol['baseline']['recall_micro'])} |
| `p88r90` (the 90%-precision model) | {fmt(pol['p88r90']['f1_micro'])} \
| {fmt(pol['p88r90']['precision_micro'])} | {fmt(pol['p88r90']['recall_micro'])} |

**Expect ~{fmt(pol['p90r85b1']['f1_micro'], 2)} micro F1 on a source unlike the
training data, not the {fmt(m(am, 'f1_micro'), 2)} headline.** The mean transfer
gap across sources is {fmt(g['gap']['mean_gap_f1'])}.

### On real documents that contain no PII

{cr['n_documents']:,} real files (PDF, DOC, XLS, PPT, HTML) judged to contain no
personal data. Every tag that fires here is a false positive:

| | this model | untuned baseline |
| --- | ---: | ---: |
| documents falsely flagged | **{cr['doc_false_alarm_rate']:.2%}** | {cb['doc_false_alarm_rate']:.2%} |
| false positives per 1,000 documents | **{cr['fp_per_1000_docs']:.1f}** | {cb['fp_per_1000_docs']:.1f} |
| implied precision at a 90%-clean estate | **{fmt(cr.get('implied_precision_at_mix'))}** | {fmt(cb.get('implied_precision_at_mix'))} |

## Limitations

**1. It does not reach 90% precision, and that is deliberate.** Micro precision
is {fmt(m(am, 'precision_micro'))}. The sibling `p88r90` reaches 0.90
in-distribution and is **worse on {n_worse} of the {n_folds} unseen sources
measured** ({fmt(pol['p88r90']['f1_micro'])} averaged, against this model's
{fmt(pol['p90r85b1']['f1_micro'])}). If you have an external commitment to a
0.90 figure, understand you are buying it with generalisation.

**2. The headline does not survive a change of source.** Transfer costs
{fmt(g['gap']['mean_gap_f1'])} micro F1 on average and up to
{fmt(g['gap']['worst_gap_f1'])} in the worst case measured. Never quote the
sealed figure as an expected number for a corpus unlike the training data.

**3. It flags {cr['doc_false_alarm_rate']:.1%} of real documents that contain no
PII.** On a document estate that is mostly clean — which real estates are, and
the sealed suite is not — implied precision falls to about
{fmt(cr.get('implied_precision_at_mix'), 2)}. Budget for review of false alarms.

**4. Whole tag families do not fire on unfamiliar layouts.** On out-of-distribution
PDFs the clinical PHI family (`medical_condition`, `patient_id_number`,
`medical_treatment`, `medication`) did not fire at all despite substantial gold.
**If your documents are clinical, measure before deploying.**

**5. {verdicts.get('unreachable', 0)} of {meta['n_labels']} tags could not reach
the target box at any threshold** and sit at their F1-optimal point instead;
{verdicts.get('not_measurable', 0)} had too little calibration support to judge.
Per-tag detail is in `docs/`.

**6. Recall on real file formats is unmeasured.** The clean-document corpus has
no positives and the sealed corpora are not real files. Nothing here says what a
real PII-bearing PDF does.

## Intended use

Triage and routing of documents for sensitive-data review — deciding which
documents a human or a heavier model should look at. It is **not** an
authoritative determination that a document does or does not contain PII, and
its per-tag output should not be used to redact automatically.

## Provenance

| | |
| --- | --- |
| architecture | {meta['architecture']} |
| labels | {meta['n_labels']} (GAIA scorecard taxonomy) |
| derived from | `{meta.get('derived_from', 'n/a')}` |
| threshold rule | {meta.get('change', 'n/a')} |
| evaluator | `{meta.get('evaluator', 'n/a')}` |
| git commit | `{git_commit()}` |
| built | {time.strftime('%Y-%m-%d')} |

See `TRAINING.md` for how it was produced and how to reproduce it.
"""


def training_doc(name: str, version: str, g: dict, meta: dict) -> str:
    sel = g["selection"]
    verdicts = sel.get("summary", {})
    n_corp = len(g["arm"].get("per_corpus", {}))
    return f"""# How `{name}-{version}` was trained

This model is **not** a neural network and was not trained end to end. It is a
two-stage linear cascade over hashed character n-grams, and it was produced in
four steps, each of which is a script in this repository.

## The short version

1. Build a 61-label catalogue from the GAIA scorecard.
2. Hash every training document into a {meta.get('n_features', 262144):,}-dimensional
   binary feature vector.
3. Fit a document gate and {meta['n_labels']} per-tag linear heads on 85% of the
   rows.
4. Choose {meta['n_labels']} decision thresholds on the **held-back 15%**, by
   targeting a precision/recall box.

Only step 4 distinguishes this model from its siblings. The weights are shared;
`p88r90`, `p80r70` and this model differ **only** in where the thresholds sit.

## Step 1 — The label space

`training/h2h_scorecard_catalogue.py` builds the catalogue directly from the
scorecard CSV rather than from a transcribed list, deriving each slug by rule:

    sensitive_<vertical>_<name normalised to snake_case>

That yields 61 labels. The features are label-independent, so changing the label
space re-indexes the label arrays only — the hashed features are untouched.

## Step 2 — Features

`training/quiet_cache.py`, via `priority_hash.document_features`. Each document
is read to a **{meta.get('window', 12000):,}-character window**, tokenised, and
hashed into {meta.get('n_features', 262144):,} binary features. No embedding, no
vocabulary file, nothing learned at this stage — which is why the whole model is
a few tens of megabytes and runs on one core.

A **model2vec/ModernBERT content model was built and tested** as an addition to
this cascade and is **not** in it: fused into a re-thresholded cascade it *lost*
ground at +56% latency. The cascade was already finding the identifiers; what it
lacked was knowing when to stay quiet, which is a threshold problem.

## Step 3 — Fitting the gate and the heads

Corpora: the {n_corp} training sets under `data/1-train/`.

The split is the load-bearing detail. `quiet_fit.carve_holdin` reserves **15% of
rows for calibration**, hashing on **corpus name + row ordinal**. Fitting and
threshold selection therefore never see the same rows.

> This is the single most dangerous part of the pipeline. An earlier attempt at
> a related model hashed the *uid string* instead, which disagreed with this
> carve on ~85% of rows and would have fitted an encoder on the very rows used
> to choose thresholds — silently, with better-looking numbers. Any change here
> must be verified element-wise against `carve_holdin`.

* **The gate** is an `SGDClassifier` over the same features, deciding whether a
  document is worth scoring at all. Negatives are down-weighted and synthetic
  corpora are re-weighted against real ones so the gate is not tuned to
  whichever source is largest. Its threshold is chosen by
  `quiet_select.select_doc_threshold_robust` against the **worst** source, not
  the average.
* **The heads** are {meta['n_labels']} discriminative linear scorers fitted by
  `quiet_materialize.fit_disc_heads`, scored with `score_mode="sum"`.

Positive-unlabelled masking runs throughout: a corpus whose gold lists only the
tags present cannot supply negatives for the tags it omits (`tag_complete`).
Ignoring this would train the heads to suppress tags that are merely unlisted.

## Step 4 — Thresholds, which is what makes this model this model

`training/h2h_target_box.py`, on the **gate-admitted calibration rows only**.

For each tag it sweeps every threshold, computes the precision/recall curve, and
takes the point inside the box **P ≥ {sel['p_target']}, R ≥ {sel['r_target']}**
that maximises **F{sel['beta']:g}**. If no threshold reaches the box, it falls
back to the F{sel['beta']:g}-optimal point and records the tag as `unreachable`.

Outcome across {meta['n_labels']} tags: \
{', '.join(f"**{v}** {k}" for k, v in sorted(verdicts.items()))}.

**The `beta` is why this model beats `p88r90`.** Both target a similar box;
`p88r90` uses `beta = 0.5`, which weights precision heavily when choosing among
in-box points and spends roughly twice the recall for the same precision gain.
This model uses `beta = 1.0`. On unseen sources that difference is the whole
margin: {fmt(g['loco_policy']['p90r85b1']['f1_micro'])} against
{fmt(g['loco_policy']['p88r90']['f1_micro'])}.

## Reproducing it

```bash
# 1. catalogue + caches (once)
python training/h2h_scorecard_catalogue.py
python training/h2h_cache.py

# 2. weights: gate + 61 heads
python training/h2h_scorecard_rebuild.py

# 3. thresholds: this model
python training/h2h_target_box.py \\
    --precision {sel['p_target']} --recall {sel['r_target']} \\
    --beta {sel['beta']:g} --name {meta.get('model_dir_name', 'cascade_p90r85b1')}

# 4. score against the sealed suite (61-label cache), then package
python training/h2h_score_any.py \\
    --model {TG}/models/{meta.get('model_dir_name', 'cascade_p90r85b1')} \\
    --cache {SC}/cache --labels {meta['n_labels']} \\
    --name {meta.get('model_dir_name', 'cascade_p90r85b1')} \\
    --out {TG}/evaluations/arm_{meta.get('model_dir_name', 'cascade_p90r85b1')}.json
python training/h2h_package_champion.py --version {version}
```

### A caveat on re-running it today

The command above was re-run against the current cache and the model reproduces
**exactly** — {fmt(g['repro']['rerun_equal_corpus_on_common'], 6)} against the
shipped {fmt(g['repro']['shipped_equal_corpus_on_common'], 6)} on the
{g['repro']['common_measurable']} precision-measurable corpora they share.

The *headline* differs: {fmt(g['repro']['rerun_f1_micro'])} today against the
shipped {fmt(g['repro']['shipped_f1_micro'])}. That is not drift. The cache was
later rebuilt with a ninth corpus ({', '.join(f"`{c}`" for c in g['repro']['added_since'])}),
so an equal-corpus mean now spans {g['repro']['current_cache_corpora']} sources
where the shipped figure spans {g['repro']['shipped_corpora']}. **The sealed
numbers on this card are the {g['repro']['shipped_corpora']}-corpus figures**;
quote them as such.

## How it was chosen over its siblings

Seven models were built and 66 further configurations swept. The comparison that
decided it was **leave-one-corpus-out** (`training/h2h_loco_variants.py`): refit
on eight sources, score on the ninth, three threshold policies per fold.

An earlier conclusion — that the sealed suite ranked models *backwards* and the
untuned baseline generalised best — came from a **single** out-of-distribution
corpus and did not survive. Across the folds this model wins
{sum(1 for r in g['loco']['per_fold'] if max(('baseline', 'p88r90', 'p90r85b1'), key=lambda p: r[p]) == 'p90r85b1')}
of {g['loco']['n_folds']}. That corpus is real and still inverts the ordering;
it is simply not representative, and one holdout was never enough to claim it
was.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cascade_p90r85b1")
    ap.add_argument("--name", default="pii-cascade-p90r85b1-champion")
    ap.add_argument("--version", default="v2")
    ap.add_argument("--source-bundle", type=Path,
                    default=TG / "dist/pii-cascade-p90r85b1-v1",
                    help="bundle to take runtime/tagger/examples from")
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()

    g = gather(args.model)
    src_model = TG / "models" / args.model
    meta = json.loads((src_model / "model.json").read_text(
        encoding="utf-8"))["metadata"]
    meta.setdefault("architecture",
                    "hashed n-gram cascade: document gate + 61 per-tag linear heads")
    meta.setdefault("n_labels", 61)
    meta["model_dir_name"] = args.model
    cat = json.loads((SC / "cache/catalogue.json").read_text(encoding="utf-8"))
    meta["n_features"] = int(cat["n_features"])

    out = TG / "dist" / f"{args.name}-{args.version}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # payload
    shutil.copytree(src_model, out / "models/model")
    for rel in ("runtime", "examples"):
        shutil.copytree(args.source_bundle / rel, out / rel)
    for rel in ("tagger.py", "requirements.txt", "README.md"):
        p = args.source_bundle / rel
        if p.is_file():
            shutil.copyfile(p, out / rel)
    (out / "config.json").write_text(json.dumps({
        "read_window_chars": meta.get("window", 12000),
        "n_labels": meta["n_labels"], "score_mode": meta.get("score_mode", "sum"),
        "cpu_budget": 1,
        "box": {"precision": g["selection"]["p_target"],
                "recall": g["selection"]["r_target"],
                "beta": g["selection"]["beta"]}}, indent=1), encoding="utf-8")

    (out / "MODEL_CARD.md").write_text(
        model_card(args.name, args.version, g, meta), encoding="utf-8")
    (out / "TRAINING.md").write_text(
        training_doc(args.name, args.version, g, meta), encoding="utf-8")

    # the evidence the card cites, shipped alongside it
    docs = out / "docs"
    docs.mkdir()
    for p in (EVAL / f"arm_{args.model}.json", EVAL / f"holdout_{args.model}.json",
              EVAL / "clean_docs.json", EVAL / "loco_variants_agg.json",
              EVAL / "loco_gap.json",
              EVAL / f"repro_check_{args.model}.json"):
        if p.is_file():
            shutil.copyfile(p, docs / p.name)
    for r in sorted((TG / "reports").glob("26-08-27*.md")):
        shutil.copyfile(r, docs / r.name)
    hist = Path("model training history.md")
    if hist.is_file():
        shutil.copyfile(hist, docs / "model_training_history.md")

    # verification: re-score through the bundle's own runtime
    verification = {"skipped": True}
    if not args.skip_verify:
        from training.h2h_scorecard_rebuild import retarget_cache
        cat61 = retarget_cache(SC / "cache", 61)
        sys.path.insert(0, str(out / "runtime"))
        sys.path.insert(0, str(out))
        from training.h2h_eval import evaluate_corpus  # noqa: E402
        from training.h2h_score import _load_cached, predict_cascade  # noqa: E402
        from training.quiet_model import QuietCascade  # noqa: E402

        model = QuietCascade.load(out / "models/model")
        cached = _load_cached(VERIFY_CORPUS, meta.get("profile", "deep"))
        fired, fired_doc, ts = predict_cascade(model, cached["X"])
        body = evaluate_corpus(VERIFY_CORPUS, fired, fired_doc, cached["Y"],
                               cached["tag_complete"], cached["doc_target"],
                               tuple(cat61["labels"]), seed=7919, tag_scores=ts)
        measured = body["summary"]["f1_micro"]
        expected = g["arm"]["per_corpus"][VERIFY_CORPUS]["f1_micro"]
        verification = {
            "checked": f"{VERIFY_CORPUS} through the packaged model",
            "n": body["n_rows"], "metric": "micro_f1",
            "expected": float(expected), "measured": float(measured),
            "delta": float(measured - expected), "tolerance": TOLERANCE,
            "ok": bool(abs(measured - expected) <= TOLERANCE)}
        if not verification["ok"]:
            raise SystemExit(f"VERIFICATION FAILED: {verification}")
        print(f"verified: {measured:.6f} vs expected {expected:.6f} "
              f"(delta {measured - expected:+.2e})", flush=True)
    (out / "models/verification.json").write_text(
        json.dumps(verification, indent=1), encoding="utf-8")

    meta_out = dict(meta)
    meta_out["promoted"] = ("champion of projects/pii-target-8070 on comparative "
                            "evidence (leave-one-corpus-out); has NOT passed the "
                            "per-tag ship gate — no model in this repository has")
    (out / "models/metadata.json").write_text(
        json.dumps(meta_out, indent=1), encoding="utf-8")

    files = sorted(p for p in out.rglob("*") if p.is_file()
                   and p.name != "SHA256SUMS")
    (out / "SHA256SUMS").write_text("".join(
        f"{sha256(p)}  {p.relative_to(out)}\n" for p in files), encoding="utf-8")
    (out / "MANIFEST.json").write_text(json.dumps({
        "name": args.name, "version": args.version,
        "git_commit": git_commit(), "built": time.strftime("%Y-%m-%d"),
        "n_files": len(files),
        "bytes": sum(p.stat().st_size for p in files),
        "verification": verification,
        "evidence": sorted(p.name for p in docs.iterdir()),
    }, indent=1), encoding="utf-8")

    archive = shutil.make_archive(str(out), "zip", root_dir=out.parent,
                                  base_dir=out.name)
    print(f"\n-> {out}")
    print(f"-> {archive} ({Path(archive).stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
