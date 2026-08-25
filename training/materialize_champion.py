"""Materialize the evidence-selected 1k fusion champion."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from training.priority_fusion import FusionPriorityModel
from training.priority_hash import load_priority_model
from training.tune_priority_hash import _save_json

FAMILY = "champion_1k"


def run(project: Path) -> dict:
    import mlflow

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    source = load_priority_model(project / "models" / "hybrid_priority")
    if not isinstance(source, FusionPriorityModel):
        raise TypeError("selected source is not a fusion model")
    champion = FusionPriorityModel(
        labels=source.labels,
        components=source.components,
        strategies=source.strategies,
        read_window_override=1_000,
    )
    model_dir = project / "models" / FAMILY
    benchmark = json.loads(
        (project / "benchmarks" / "read_depth.json").read_text(encoding="utf-8")
    )
    evidence = next(
        row
        for row in benchmark["rows"]
        if row["family"] == "hybrid_priority" and row["read_depth_chars"] == 1_000
    )
    champion.save(
        model_dir,
        metadata={
            "family": FAMILY,
            "source_family": "hybrid_priority",
            "source_trial": 63,
            "selection": "priority gate -> macro F2 -> micro F1 -> one-core p95",
            "read_depth_evidence": evidence,
        },
    )
    mlflow.set_tracking_uri((project / "mlruns").resolve().as_uri())
    mlflow.set_experiment("pii-priority-recall-v1")
    with mlflow.start_run(run_name=f"{FAMILY}_selection") as active:
        mlflow.set_tags(
            {
                "model": FAMILY,
                "source_model": "hybrid_priority",
                "phase": "selection",
                "dataset": "all_eval_read_depth_ladder",
            }
        )
        mlflow.log_params(
            {
                "read_window_chars": 1_000,
                "source_trial": 63,
                "priority_gates": evidence["measurable_priority_gates"],
            }
        )
        mlflow.log_metrics(
            {
                "equal_corpus_macro_f2": evidence["equal_corpus_macro_f2"],
                "equal_corpus_micro_f1": evidence["equal_corpus_micro_f1"],
                "worst_priority_recall": evidence["worst_priority_recall"],
                "priority_point_passes": evidence["priority_point_passes"],
                "p95_ms_one_core": evidence["p95_ms"],
                "docs_per_s_one_core": evidence["docs_per_s"],
            }
        )
        run_id = active.info.run_id
    result = {
        "family": FAMILY,
        "source_family": "hybrid_priority",
        "source_trial": 63,
        "read_window_chars": 1_000,
        "selection_mlflow_run_id": run_id,
        "evidence": evidence,
        "model_dir": str(model_dir.relative_to(project)),
    }
    _save_json(project / "tuning" / FAMILY / "resolved_config.json", result)
    run_path = project / "run.json"
    run_record = json.loads(run_path.read_text(encoding="utf-8"))
    run_record.setdefault("run_summary", {})[FAMILY] = result
    for artifact in (
        f"tuning/{FAMILY}/resolved_config.json",
        f"models/{FAMILY}/model.json",
    ):
        if artifact not in run_record.setdefault("artifacts", []):
            run_record["artifacts"].append(artifact)
    _save_json(run_path, run_record)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.project.resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
