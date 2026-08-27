"""Fold the three scored arms into the run record, the decisions and the log rows.

Two shapes come out of this, and they are deliberately different.

**Three model-level arms** feed `mp decide`. A policy reasons about a model: its
headline metrics and its scoped ones (`doc@corpus`, `tag@corpus`), because that
is what a gate like "every measurable tag x corpus pair clears 0.90" reads.

**Twenty-four exploded rows** feed `run.json` and the Experiment Log -- one per
*model x dataset*, which is the log's unit. Collapsing a head-to-head into three
rows would put three models' scores in columns that hold one model's score;
collapsing it into one row would put all of them in a comma-joined name with no
numbers at all. Eight corpora times three arms is twenty-four results and
twenty-four rows.

Both decisions declared before the run are applied and both are recorded. The
headline is named in the spec, not chosen here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from model_pipeline import load_arms, load_policy  # noqa: E402
from model_pipeline.runlog import assemble  # noqa: E402

from training.h2h_eval import BETA_NAMES, BETAS  # noqa: E402
from training.h2h_score import ARMS  # noqa: E402

PROJECT = Path("/home/lence/workspace/pii_master/projects/pii-head-to-head-v1")
RUN_ID = "h2h-2026-08-25"
#: Short, stable names -- these become the Experiment Log's `model` column, so
#: they have to stay recognisable across runs and fit in a cell.
MODEL_NAMES = {"A": "fusion-1k", "B": "steady-cascade", "C": "fusion-12k"}


def _per_tag_rows(per_tag: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """The workbook's per-tag schema. Tags absent from a corpus's gold are dropped."""
    rows = []
    for tag, t in sorted(per_tag.items(), key=lambda kv: (kv[1]["support"] == 0,
                                                          kv[1].get("f2") or 0.0)):
        if not t["support"] and not t["predicted"]:
            continue
        rows.append({
            "tag": tag,
            "gold_seen": t["support"],
            "predicted": t["predicted"],
            "true_positive_found": t["tp"],
            "missed": t["fn"],
            "false_positive": t["fp"],
            "precision": t["precision"],
            "recall": t["recall"],
            "f1": t["f1"],
            "f2": t["f2"],
            # Carried so the tab can be read at every beta the run was asked for.
            "f05": t["f05"],
            "f3": t["f3"],
        })
    return rows


def _corpus_metrics(s: dict[str, Any]) -> dict[str, Any]:
    """Per-corpus metrics, named so the log's fixed columns find them."""
    m: dict[str, Any] = {
        # The log matches these four by name for its Macro F1 / Micro F1 /
        # Precision / Recall columns.
        "f1_macro": s.get("f1_macro_catalogue"),
        "f1_micro": s.get("f1_micro"),
        "precision_macro": s.get("precision_macro_catalogue"),
        "recall_macro": s.get("recall_macro_catalogue"),
        # The declared headline and the contra-view's ranker.
        "macro_f2": s.get("f2_macro_catalogue"),
        "priority_macro_f05": s.get("priority_macro_f05"),
    }
    for beta in BETAS:
        n = BETA_NAMES[beta]
        for scope in ("catalogue", "support30"):
            m[f"{n}_macro_{scope}"] = s.get(f"{n}_macro_{scope}")
        m[f"{n}_micro"] = s.get(f"{n}_micro")
        m[f"priority_macro_{n}"] = s.get(f"priority_macro_{n}")
    ci = s.get("f2_macro_catalogue_ci")
    if ci:
        m["macro_f2"] = {"value": s.get("f2_macro_catalogue"),
                         "ci_low": ci[0], "ci_high": ci[1]}
    ci = s.get("f1_micro_ci")
    if ci:
        m["f1_micro"] = {"value": s.get("f1_micro"), "ci_low": ci[0], "ci_high": ci[1]}
    # The top-k ladder and the severity mean are per-corpus too: the log's three
    # leading columns and its Severity Recall Mean column read them from HERE,
    # not from the arm-level roll-up, so omitting them from this list leaves
    # four columns blank on all 128 rows while the arm-level numbers look fine.
    for k in (1, 3, 5):
        for stem in ("f1", "precision", "recall"):
            m[f"{stem}@{k}"] = s.get(f"{stem}@{k}")
    for key in ("precision_micro", "recall_micro", "priority_macro_precision",
                "priority_macro_recall", "severity_recall_min",
                "severity_recall_mean", "prediction_rate",
                "f2_min", "f2_median", "n_tags_f2_zero", "n_tags_f2_below_10pct",
                "tags_predicted_zero_times", "doc_fire_rate",
                "n_tags_catalogue", "n_tags_support30", "n_priority_measurable"):
        m[key] = s.get(key)
    return {k: v for k, v in m.items() if v is not None}


def _doc_scope(arm: dict[str, Any], corpus: str) -> dict[str, Any]:
    scope = arm["scopes"].get(f"doc@{corpus}")
    if not scope:
        return {}
    return {f"{k}": v["value"] for k, v in scope.items() if v.get("value") is not None}


#: Arms measured beyond the original three. Every operating point and every gate
#: variant scored on the sealed set is its own Experiment Log row -- the log's
#: unit is one model x one dataset, and a run that measured seventeen models and
#: logged three has thrown away the history it exists to keep.
EXTRA_ARMS = {
    "cascade_balanced": ("steady-cascade-bal", 12_000, "cascade"),
}


def _extra_spec(name: str) -> tuple[str, int, str]:
    if name in EXTRA_ARMS:
        return EXTRA_ARMS[name]
    if name.startswith("f") and "b" in name:          # cascade_f70b50 -> floor/beta
        floor, beta = name[1:].split("b")
        return (f"steady-cascade-f{floor}b{beta}", 12_000, "cascade")
    return (name, 12_000, "cascade")


def explode(arms: dict[str, dict[str, Any]], latency: dict[str, dict[str, Any]]
            ) -> list[dict[str, Any]]:
    gold = gold_modes()
    rows: list[dict[str, Any]] = []
    for key in sorted(arms):
        arm = arms[key]
        if key in ARMS:
            spec = ARMS[key]
            lat = latency.get(key, {})
        else:
            model_name, window, kind = _extra_spec(key)
            spec = {"label": arm.get("label", key), "window": window, "kind": kind}
            lat = latency.get("B", {})   # same architecture, same measured cost
        for corpus, summary in arm["per_corpus"].items():
            doc = _doc_scope(arm, corpus)
            metrics = _corpus_metrics(summary) | doc
            measurable = summary.get("can_measure_precision")
            rows.append({
                "model": MODEL_NAMES.get(key) or _extra_spec(key)[0],
                "dataset": corpus,
                "n_samples": summary.get("n_rows"),
                "tier": f"{spec['window'] // 1000}k",
                "latency_ms_per_doc": lat.get("mean_ms"),
                "effective_chars": spec["window"],
                "primary_metric": "macro_f2",
                "task_type": "tagging",
                # Every derived arm is compared against the model it derives
                # from, on the same corpus, so the Delta column answers the only
                # question worth asking of a variant: did the change help HERE.
                # Arm B is the reference and carries "" -- an explicit baseline,
                # not a missing value.
                "compare_vs": ("" if MODEL_NAMES.get(key) == "steady-cascade"
                               else "steady-cascade"),
                "what_changed": (f"{spec['label']}; re-tuned on full 1-train "
                                 f"(531,431 rows) under the corrected loader"),
                "verdict": ("" if measurable
                            else _blank_reason(gold.get(corpus, "unknown"), doc)),
                "metrics": metrics,
                "per_tag": _per_tag_rows(arm["per_tag"][corpus]),
                "params": {"arm": key, "read_window_chars": spec["window"],
                           "kind": spec["kind"]},
            })
    return rows


def gold_modes() -> dict[str, str]:
    """Each corpus's gold mode, so a blank cell can say WHY it is blank."""
    suite = yaml.safe_load((PROJECT / "suite.yaml").read_text(encoding="utf-8"))
    return {c["name"]: c["gold"] for c in suite["corpora"]}


def _blank_reason(gold: str, doc: dict[str, Any]) -> str:
    """What a reader needs when the quality columns are empty.

    A blank cell reads as "not measured", which on these corpora is wrong in two
    different ways: the reason differs by gold mode, and the document-level
    question IS measured on the two that carry judge-asserted negatives. Naming
    the reason and the numbers that do exist is the difference between a row
    that looks like a gap and one that points at the right tab.
    """
    why = {
        "positive_only": ("tag gold is positive-only: an unlisted tag is unknown, "
                          "not absent, so no precision-bearing metric is computable"),
        "partial": ("tag gold is partial/coarse: the loader recovers no sensitive-tag "
                    "positives, so neither precision nor recall is computable"),
    }.get(gold, f"gold mode {gold!r} cannot measure precision")
    if doc:
        got = " · ".join(f"{k.removeprefix('doc_')} {v:.4f}"
                         for k, v in sorted(doc.items()) if isinstance(v, float))
        return (f"NOT MEASURABLE — {why}. Document-level IS measured "
                f"(see Run Results): {got}")
    return f"NOT MEASURABLE — {why}"


def data_quality_rows() -> list[dict[str, Any]]:
    suite = yaml.safe_load((PROJECT / "suite.yaml").read_text(encoding="utf-8"))
    out = []
    for c in suite["corpora"]:
        dq = dict(c.get("data_quality") or {})
        out.append({"dataset": c["name"], "role": c["role"], "gold": c["gold"],
                    "n": c["n"], **dq})
    return out


def decide(arms_paths: list[Path]) -> dict[str, Any]:
    decisions: dict[str, Any] = {}
    for label, path in (("headline", PROJECT / "policy.yaml"),
                        ("precision_view", PROJECT / "policy_precision_view.yaml")):
        policy = load_policy(str(path))
        result = policy.decide(load_arms([str(p) for p in arms_paths]))
        payload = result.to_dict() if hasattr(result, "to_dict") else json.loads(json.dumps(result, default=str))
        (PROJECT / "decision" / f"{label}.json").write_text(
            json.dumps(payload, indent=1, default=str) + "\n", encoding="utf-8")
        decisions[label] = payload
        winner = payload.get("winner") or payload.get("selected")
        print(f"  {label:15s} policy={policy.name:38s} winner={winner}")
    return decisions


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=PROJECT / "run.json")
    args = ap.parse_args()

    paths = [PROJECT / "evaluations" / f"arm_{k}.json" for k in sorted(ARMS)]
    arms = {k: json.loads(p.read_text(encoding="utf-8"))
            for k, p in zip(sorted(ARMS), paths)}
    # Everything else scored on the sealed set this run: the twelve
    # floor x beta operating points and the balanced-gate cascade.
    # `cascade_floor*.json` is deliberately NOT globbed: that ladder was computed
    # with the head trial's margin/min_support instead of the cascade trial's, so
    # its rungs are a superseded configuration. Two rows both labelled "floor
    # 0.75" with different numbers and nothing to tell them apart is worse than
    # one row. The corrected `f<floor>b<beta>` ladder is what is logged.
    for extra in sorted((PROJECT / "evaluations").glob("cascade_f*b*.json")):
        arms[extra.stem] = json.loads(extra.read_text(encoding="utf-8"))
    for extra in sorted((PROJECT / "evaluations").glob("arm_cascade_*.json")):
        arms[extra.stem.removeprefix("arm_")] = json.loads(
            extra.read_text(encoding="utf-8"))
    latency = {}
    for k in sorted(ARMS):
        p = PROJECT / "evaluations" / f"latency_{k}.json"
        if p.is_file():
            latency[k] = json.loads(p.read_text(encoding="utf-8"))

    print("decisions (both declared before the numbers):")
    decisions = decide(paths)

    rows = explode(arms, latency)
    print(f"exploded {len(arms)} arms x 8 corpora -> {len(rows)} Experiment Log rows")

    run = assemble(
        str(PROJECT / "runs" / RUN_ID),
        out_path=str(args.out),
        date="2026-08-25",
        project="pii-head-to-head-v1",
        run_id="H1",
        track="sensitive-data",
        what_changed=("head-to-head: both lineages re-tuned from scratch on the full "
                      "1-train (531,431 rows) under one loader, one 58-label catalogue, "
                      "one fit/calibration carve and one fixed evaluator"),
        arms=rows,
        data_quality=data_quality_rows(),
        decision=decisions["headline"],
        decision_precision_view=decisions["precision_view"],
    )
    print(f"run.json -> {args.out} ({args.out.stat().st_size:,} bytes, "
          f"{len(run.get('commands', []))} commands)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
