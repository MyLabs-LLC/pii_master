"""Build the first approved priority/generic hybrid without hold-out tuning."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from training.priority_data import PRIORITY_TAGS
from training.priority_hash import HashCueModel, HybridPriorityModel
from training.tune_priority_hash import _save_json

FAMILY = "hybrid_priority_001"


def run(project: Path) -> dict:
    import mlflow

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    recall_model = HashCueModel.load(project / "models" / "hash_sgd")
    f2_model = HashCueModel.load(project / "models" / "hash_sgd_f2")
    hybrid = HybridPriorityModel(
        priority_model=recall_model,
        generic_model=f2_model,
        priority_tags=frozenset(PRIORITY_TAGS),
    )
    model_dir = project / "models" / FAMILY
    hybrid.save(
        model_dir,
        metadata={
            "family": FAMILY,
            "priority_component": "hash_sgd trial 188 recall-max",
            "generic_component": "hash_sgd trial 32 feasible-region F2-max",
            "selection_data": "combined internal validation only",
        },
    )
    mlflow.set_tracking_uri((project / "mlruns").resolve().as_uri())
    mlflow.set_experiment("pii-priority-recall-v1")
    with mlflow.start_run(run_name=f"{FAMILY}_build") as active:
        mlflow.set_tags(
            {
                "model_family": "hybrid_priority",
                "model": FAMILY,
                "phase": "build",
                "dataset": "combined_training",
            }
        )
        mlflow.log_params(
            {
                "priority_component": "hash_sgd",
                "generic_component": "hash_sgd_f2",
                "priority_tags": len(PRIORITY_TAGS),
                "read_window_chars": hybrid.read_window_chars,
            }
        )
        run_id = active.info.run_id
    result = {
        "family": FAMILY,
        "mlflow_run_id": run_id,
        "priority_component": "hash_sgd",
        "generic_component": "hash_sgd_f2",
        "priority_tags": list(PRIORITY_TAGS),
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
