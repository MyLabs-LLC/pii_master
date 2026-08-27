"""Attach the frozen champion, bootstrap, and read-depth evidence to run.json."""

from __future__ import annotations

import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
RUN_PATH = PROJECT / "run.json"
BOOTSTRAP = PROJECT / "evaluations" / "champion_1k" / "bootstrap"


def metric(value: float, **extra: object) -> dict[str, object]:
    return {"value": value, "greater_is_better": True, **extra}


def main() -> None:
    run = json.loads(RUN_PATH.read_text(encoding="utf-8"))
    summary = json.loads((BOOTSTRAP / "summary.json").read_text(encoding="utf-8"))
    aggregate = summary["aggregate"]
    benchmark = json.loads(
        (PROJECT / "benchmarks" / "read_depth.json").read_text(encoding="utf-8")
    )
    existing_artifacts = run.get("artifacts", {})
    if isinstance(existing_artifacts, list):
        run["artifact_index"] = existing_artifacts
        existing_artifacts = {}

    per_tag: list[dict[str, object]] = []
    for path in sorted(BOOTSTRAP.glob("*.json")):
        if path.name == "summary.json":
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        for tag, priority in result["priority"].items():
            if priority["support"] < 30:
                continue
            counts = result["per_tag"][tag]
            precision = counts.get("precision")
            recall = counts.get("recall")
            f1 = None
            if precision is not None and recall is not None and precision + recall:
                f1 = 2.0 * precision * recall / (precision + recall)
            per_tag.append(
                {
                    "tag": f"{result['dataset']} / {tag}",
                    "gold_seen": counts["support"],
                    "predicted": counts["predicted"],
                    "true_positive_found": counts["tp"],
                    "missed": counts["fn"],
                    "false_positive": counts["fp"],
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "f2": counts.get("f2"),
                    "recall_ci_low": priority["ci_low"],
                    "recall_ci_high": priority["ci_high"],
                    "gate_status": priority["status"],
                }
            )

    # This script is deliberately idempotent so reports can be regenerated.
    kept = [
        arm
        for arm in run.get("arms", [])
        if not str(arm.get("model", "")).endswith("_readpass")
        and arm.get("model") != "champion_1k_bootstrap"
    ]

    macro_ci = aggregate["equal_corpus_macro_f2_ci"]
    micro_ci = aggregate["equal_corpus_micro_f1_ci"]
    kept.append(
        {
            "run_id": "R7.65",
            "track": "sensitive",
            "model": "champion_1k_bootstrap",
            "dataset": "8 holdouts (5 complete for F2)",
            "tier": "T1",
            "n_samples": 126129,
            "c_read": 1000,
            "model_ceiling": 1000,
            "latency_ms_per_doc": 2.2000839,
            "task_type": "tagging",
            "primary_metric": "macro_f2",
            "what_changed": "Final 1k per-label fusion; 1,000 bootstrap resamples",
            "verdict": (
                "55/55 priority gates conclusively pass; latency passes; "
                "macro-F2 target 0.90 misses"
            ),
            "metrics": {
                "macro_f2": metric(
                    aggregate["equal_corpus_macro_f2"],
                    ci_low=macro_ci["ci_low"],
                    ci_high=macro_ci["ci_high"],
                    confidence=macro_ci["confidence"],
                    n_bootstrap=macro_ci["n_resamples"],
                ),
                "micro_f1": metric(
                    aggregate["equal_corpus_micro_f1"],
                    ci_low=micro_ci["ci_low"],
                    ci_high=micro_ci["ci_high"],
                    confidence=micro_ci["confidence"],
                    n_bootstrap=micro_ci["n_resamples"],
                ),
                "worst_priority_recall": metric(
                    aggregate["worst_priority_recall"]
                ),
                "priority_conclusive_passes": metric(
                    aggregate["priority_conclusive_passes"]
                ),
                "docs_per_s": metric(919.1197682993916),
                "latency_p95_ms": {
                    "value": 2.2000839,
                    "greater_is_better": False,
                },
            },
            "per_tag": per_tag,
        }
    )

    for row in benchmark["rows"]:
        kept.append(
            {
                "track": "sensitive",
                "model": f"{row['family']}_readpass",
                "dataset": "8 holdouts / fixed 1,000-doc latency sample",
                "tier": "T1" if row["p95_ms"] <= 5.0 else "T2",
                "n_samples": row["n_documents"],
                "c_read": row["read_depth_chars"],
                "latency_ms_per_doc": row["p95_ms"],
                "task_type": "tagging",
                "primary_metric": "macro_f2",
                "what_changed": f"Read-pass sweep at {row['read_depth_chars']} characters",
                "verdict": (
                    f"{row['priority_point_passes']}/{row['measurable_priority_gates']} "
                    "priority point gates pass"
                ),
                "metrics": {
                    "macro_f2": metric(row["equal_corpus_macro_f2"]),
                    "micro_f1": metric(row["equal_corpus_micro_f1"]),
                    "worst_priority_recall": metric(row["worst_priority_recall"]),
                    "priority_point_passes": metric(row["priority_point_passes"]),
                    "docs_per_s": metric(row["docs_per_s"]),
                    "latency_p50_ms": {
                        "value": row["p50_ms"],
                        "greater_is_better": False,
                    },
                    "latency_p95_ms": {
                        "value": row["p95_ms"],
                        "greater_is_better": False,
                    },
                    "latency_p99_ms": {
                        "value": row["p99_ms"],
                        "greater_is_better": False,
                    },
                    "peak_rss_mb": {
                        "value": row["peak_rss_mb"],
                        "greater_is_better": False,
                    },
                },
            }
        )

    run.update(
        {
            "run_id": "R7",
            "date": "2026-08-25",
            "track": "sensitive",
            "model": "champion_1k",
            "dataset": "8 frozen holdouts",
            "tier": "T1",
            "task_type": "tagging",
            "primary_metric": "macro_f2",
            "read_depth_chars": 1000,
            "model_ceiling": 1000,
            "latency_ms_per_doc": 2.2000839,
            "what_changed": "1,000-trial multi-family search and read-depth sweep",
            "verdict": (
                "Priority and effective-latency gates pass; macro-F2 target misses; "
                "package for research/internal use"
            ),
            "status": "completed_with_macro_f2_target_gap",
            "arms": kept,
            "artifacts": {
                **existing_artifacts,
                "mlflow_run_id": "00c00641e3864afc86cbc4476ea8cf8c",
                "model_uri": "projects/pii-priority-recall-v1/models/champion_1k",
                "report_md": "reports/26-08-25_priority-recall-1000-run.md",
                "bootstrap": "evaluations/champion_1k/bootstrap/summary.json",
                "read_depth_benchmark": "benchmarks/read_depth.json",
                "package_manifest": "dist/package.json",
                "model_bundle": "dist/pii-priority-fusion-1k-v1.zip",
                "package_verification": "dist/bundle_verification.json",
            },
        }
    )
    if isinstance(run.get("budget"), dict):
        run["budget"]["trials_used"] = 1000
    RUN_PATH.write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
