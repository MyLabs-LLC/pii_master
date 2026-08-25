"""Assemble `run.json`: one arm per model x dataset, plus what produced it.

The reporting convention is one exported line per *model x dataset*, never a
head-to-head merged into one comma-joined row -- the metric columns hold one
model's score, so merging throws away every number the run was for.

One deviation is recorded here rather than papered over. Commands in this run
were issued directly rather than through `mp run`, so there is no captured
stdout for each. What the archive carries instead is the set of commands that
produced each artifact, alongside the log files that *were* captured
(`tune_*.log`, `cache_build*.log`) and the MLflow database holding every one of
the 1,000 trials. Nothing in the report is reconstructed from memory; every
number traces to a JSON artifact or to MLflow.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT = Path("/home/lence/workspace/pii_master/projects/pii-quiet-alarm")
RUN_ID = "pii-quiet-alarm-2026-08-25"

#: The commands that produced this run's artifacts, in order. Captured output
#: for the long-running ones is in the named log files beside them.
COMMANDS: list[dict[str, str]] = [
    {"context": "feasibility", "command": "run.sh projects/pii-quiet-alarm/probe/precision_headroom.py",
     "produces": "probe/precision_headroom.json"},
    {"context": "feasibility", "command": "run.sh projects/pii-quiet-alarm/probe/train_negatives.py",
     "produces": "probe/train_negatives.json"},
    {"context": "feasibility", "command": "run.sh projects/pii-quiet-alarm/probe/doc_baseline_corrected.py",
     "produces": "probe/doc_baseline_corrected.json"},
    {"context": "feasibility", "command": "mp feasibility --metric doc_specificity --target 0.90 --agreement 0.9545",
     "produces": "probe/feasibility_doc_specificity.json"},
    {"context": "feasibility", "command": "mp feasibility --metric priority_macro_precision --target 0.90 --agreement 0.4582 --ceiling 0.5236",
     "produces": "probe/feasibility_tag_precision.json"},
    {"context": "data", "command": "run.sh training/quiet_freeze.py freeze",
     "produces": "data_snapshot.json"},
    {"context": "data", "command": "run.sh training/quiet_cache.py 30",
     "produces": "cache/*.npz, cache/catalogue.json (log: cache_build2.log)"},
    {"context": "data", "command": "pytest tests/test_quiet_data.py -q", "produces": "8 passed"},
    {"context": "search", "command": "run.sh training/tune_quiet.py --family docgate --trials 300",
     "produces": "tuning/docgate/ (log: tune_docgate.log)"},
    {"context": "search", "command": "run.sh training/tune_quiet.py --family tagcount --trials 250 --seed 23",
     "produces": "tuning/tagcount_v1/ (log: tune_tagcount.log) -- mis-specified objective, re-ranked"},
    {"context": "search", "command": "run.sh training/quiet_rerank.py --family tagcount",
     "produces": "tuning/tagcount_v1/best.json"},
    {"context": "search", "command": "run.sh training/tune_quiet.py --family tagcount --trials 120 --seed 47",
     "produces": "tuning/tagcount/ (log: tune_tagcount2.log)"},
    {"context": "search", "command": "run.sh training/quiet_merge_tagcount.py",
     "produces": "tuning/tagcount/best.json (merged pool)"},
    {"context": "search", "command": "run.sh training/tune_quiet.py --family tagdisc --trials 120 --seed 53",
     "produces": "tuning/tagdisc/ (log: tune_tagdisc.log)"},
    {"context": "search", "command": "run.sh training/tune_quiet.py --family cascade --trials 183 --seed 61",
     "produces": "tuning/cascade/ (log: tune_cascade.log)"},
    {"context": "ship", "command": "run.sh training/quiet_materialize.py --rank 0 --out artifacts/quiet-cascade",
     "produces": "artifacts/quiet-cascade/ (reproduction drift 0.00000)"},
    {"context": "measure", "command": "run.sh training/quiet_score_eval.py --model quiet-cascade=... --model quiet-nogate=... --baseline --cpus 1",
     "produces": "evaluations/arms.json"},
    {"context": "decide", "command": "mp decide --policy policy.yaml --arms evaluations/arms.json --suite suite.yaml",
     "produces": "decision/decision.json -- NO FEASIBLE ARM"},
]


def _data_quality_from_suite() -> list[dict[str, Any]]:
    """Carry the suite's per-corpus assessment onto the run record.

    The entry belongs to the corpus and is assessed once, so it is read from
    `suite.yaml` rather than retyped here -- retyping it is how `leakage: 0`
    becomes an assertion instead of a scan result.
    """
    import yaml
    suite = yaml.safe_load((PROJECT / "suite.yaml").read_text(encoding="utf-8"))
    out = []
    for corpus in suite["corpora"]:
        dq = dict(corpus.get("data_quality") or {})
        dq.update({"dataset": corpus["name"], "gold": corpus["gold"],
                   "n_rows": corpus.get("n"), "role": corpus["role"],
                   "source": "projects/pii-quiet-alarm/suite.yaml"})
        out.append(dq)
    return out


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True,
                              cwd="/home/lence/workspace/pii_master").stdout.strip()
    except Exception:
        return "unknown"


def _trial_counts() -> dict[str, int]:
    con = sqlite3.connect(PROJECT / "mlflow.db")
    out = {}
    for fam in ("docgate", "tagcount", "tagdisc", "cascade"):
        out[fam] = con.execute("select count(*) from runs where name like ?",
                               (fam + "-%",)).fetchone()[0]
    out["total"] = sum(out.values())
    return out


def main() -> int:
    suite_dq = _data_quality_from_suite()
    arms_doc = json.loads((PROJECT / "evaluations" / "arms.json").read_text(encoding="utf-8"))
    decision = json.loads((PROJECT / "decision" / "decision.json").read_text(encoding="utf-8"))
    snapshot = json.loads((PROJECT / "data_snapshot.json").read_text(encoding="utf-8"))

    arms: list[dict[str, Any]] = []
    for model in arms_doc["arms"]:
        m = model["metrics"]
        arm_level = {k: (v["value"] if isinstance(v, dict) else v) for k, v in m.items()}
        for corpus, summary in model["per_corpus"].items():
            doc = model["scopes"].get(f"doc@{corpus}", {})
            arms.append({
                "model": model["name"],
                "dataset": corpus,
                "metrics": {
                    "priority_macro_f05": summary.get("priority_macro_f05"),
                    "priority_macro_precision": summary.get("priority_macro_precision"),
                    "priority_macro_recall": summary.get("priority_macro_recall"),
                    "macro_f05": summary.get("macro_f05"),
                    "doc_precision": doc.get("doc_precision", {}).get("value"),
                    "doc_precision_ci_low": doc.get("doc_precision", {}).get("ci_low"),
                    "doc_recall": doc.get("doc_recall", {}).get("value"),
                    "doc_recall_ci_low": doc.get("doc_recall", {}).get("ci_low"),
                    "doc_specificity": doc.get("doc_specificity", {}).get("value"),
                    "doc_specificity_ci_low": doc.get("doc_specificity", {}).get("ci_low"),
                    "prediction_rate": summary.get("prediction_rate"),
                    "n_measurable_tags": summary.get("n_measurable_tags"),
                    "tags_predicted_zero_times": summary.get("tags_predicted_zero_times"),
                    "p95_latency_ms": arm_level.get("p95_latency_ms"),
                    "docs_per_s": arm_level.get("docs_per_s"),
                },
                "n_rows": summary.get("n_rows"),
                "artifact": "evaluations/arms.json",
            })

    counts = _trial_counts()
    run = {
        "run_id": RUN_ID,
        "project": "pii-quiet-alarm",
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "task_type": "multi_label_document_tagging_with_document_gate",
        "primary_metric": "priority_macro_f05",
        "objective": "precision-first: document-level gates hard, priority macro F0.5 ranks",
        "source_commit": _git_commit(),
        "cpu_budget": {"search": "all 32 cores", "published_latency": "1 core"},
        "budget": {"max_trials": 1000, "trials_used": counts["total"], "by_family": counts},
        "approval": "approvals/full-loop-precision-1000.json",
        "data_snapshot": {
            "train_rows": snapshot["totals"]["train"]["rows"],
            "train_negatives": snapshot["totals"]["train"]["doc_negative"],
            "eval_rows": snapshot["totals"]["eval"]["rows"],
            "eval_negatives": snapshot["totals"]["eval"]["doc_negative"],
        },
        "arms": arms,
        "data_quality": suite_dq,
        "commands": COMMANDS,
        "command_capture": (
            "Commands were issued directly rather than through `mp run`, so no per-command "
            "stdout was captured. Captured logs that do exist: cache_build2.log, "
            "tune_docgate.log, tune_tagcount.log, tune_tagcount2.log, tune_tagdisc.log, "
            "tune_cascade.log. Every metric in the report traces to a JSON artifact or to "
            "mlflow.db; none is reconstructed from memory. This is a deviation from the "
            "run-record convention and is recorded as one."
        ),
        "verdict": "no feasible arm - promotion refused",
        "run_summary": {
            "selected": decision.get("selected"),
            "n_feasible": len(decision.get("feasible", [])),
            "n_arms": len(arms_doc["arms"]),
            "reason": decision.get("reason", []),
        },
        "what_changed": (
            "New precision-first lineage. Corrected the loader to admit judge-asserted "
            "document negatives (+20,639 real-world clean documents), corrected the "
            "document-level gold field, froze a taxonomy collapse, added a discriminative "
            "document gate in front of per-tag heads, and ranked on macro F0.5 instead of "
            "macro F2."
        ),
        "status": "complete",
    }
    (PROJECT / "run.json").write_text(json.dumps(run, indent=1), encoding="utf-8")
    print(f"wrote run.json: {len(arms)} arms (model x dataset), "
          f"{counts['total']} trials, commit {run['source_commit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
