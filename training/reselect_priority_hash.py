"""Materialize the approved feasible-region winner from completed hash trials."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from training.priority_data import READ_WINDOW
from training.priority_eval import aggregate_arms
from training.priority_hash import HashCounts, HashCueModel, build_weights
from training.tune_priority_hash import (
    FAMILY,
    _exact_best_arms,
    _load_excluded,
    _load_jsonl,
    _objective,
    _save_json,
    _thresholds_for_trial,
    fast_metrics,
    score_validation,
    threshold_bank,
)

OUTPUT_FAMILY = "hash_sgd_f2"


def _load_trials(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def run(project: Path) -> dict[str, Any]:
    import mlflow

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    data_dir = project / "data"
    rows = _load_jsonl(data_dir / "train_index.jsonl")
    excluded = _load_excluded(data_dir / "train_exclusions.json")
    frozen = json.loads(
        (data_dir / "evaluation_catalogue.json").read_text(encoding="utf-8")
    )
    quality = json.loads((data_dir / "data_quality.json").read_text(encoding="utf-8"))
    labels = tuple(frozen["full_catalogue"])
    counts = HashCounts.load(project / "cache" / FAMILY / "counts.npz")
    weights = build_weights(
        counts,
        alpha=1.0,
        partial_weight=0.75,
        min_document_frequency=3,
    )
    validation = score_validation(
        rows, labels=labels, excluded=excluded, weights=weights
    )
    bank = threshold_bank(
        validation["scores"], validation["y_true"], validation["datasets"], labels
    )
    trials = _load_trials(project / "tuning" / FAMILY / "trials.jsonl")
    best = max(trials, key=lambda trial: _objective(trial["metrics"]))
    config = best["config"]
    thresholds = _thresholds_for_trial(config, bank, labels)
    predicted = validation["scores"][config["score_mode"]] >= thresholds
    metrics = fast_metrics(
        predicted,
        validation["y_true"],
        validation["datasets"],
        validation["complete"],
        labels,
        quality["train"],
    )
    if metrics["priority_point_passes"] != metrics["measurable_priority_gates"]:
        raise RuntimeError(
            "reselected trial is outside the priority-recall feasible region"
        )

    model_dir = project / "models" / OUTPUT_FAMILY
    model = HashCueModel(
        labels=labels,
        weights=weights,
        thresholds=thresholds,
        score_mode=config["score_mode"],
        read_window_chars=READ_WINDOW,
    )
    model.save(
        model_dir,
        metadata={
            "family": OUTPUT_FAMILY,
            "source_family": FAMILY,
            "source_trials": len(trials),
            "source_trial": best["trial"],
            "selection": "priority gate, macro F2, micro F1",
            "validation_metrics": metrics,
        },
    )
    exact_arms = _exact_best_arms(validation, labels, predicted, quality["train"])
    exact_summary = aggregate_arms(exact_arms)
    mlflow.set_tracking_uri((project / "mlruns").resolve().as_uri())
    mlflow.set_experiment("pii-priority-recall-v1")
    with mlflow.start_run(run_name=f"{OUTPUT_FAMILY}_selection") as active:
        mlflow.set_tags(
            {
                "model_family": OUTPUT_FAMILY,
                "source_family": FAMILY,
                "phase": "selection",
                "dataset": "combined_internal_validation",
                "split": "validation",
            }
        )
        mlflow.log_params(
            {**config, "source_trial": best["trial"], "source_trials": len(trials)}
        )
        mlflow.log_metrics(
            {key: value for key, value in metrics.items() if key != "per_corpus"}
        )
        selection_run_id = active.info.run_id
    result = {
        "family": OUTPUT_FAMILY,
        "source_family": FAMILY,
        "source_trials": len(trials),
        "source_trial": best["trial"],
        "source_trial_mlflow_run_id": best["mlflow_run_id"],
        "selection_mlflow_run_id": selection_run_id,
        "best_config": config,
        "fast_metrics": metrics,
        "exact_metrics": exact_summary,
        "model_dir": str(model_dir.relative_to(project)),
    }
    output_dir = project / "tuning" / OUTPUT_FAMILY
    _save_json(output_dir / "resolved_config.json", result)
    _save_json(output_dir / "validation_arms.json", exact_arms)
    run_path = project / "run.json"
    run_record = json.loads(run_path.read_text(encoding="utf-8"))
    run_record.setdefault("run_summary", {})[OUTPUT_FAMILY] = {
        "source_trials": len(trials),
        "source_trial": best["trial"],
        "metrics": metrics,
        "model": str(model_dir.relative_to(project)),
    }
    for artifact in (
        f"tuning/{OUTPUT_FAMILY}/resolved_config.json",
        f"tuning/{OUTPUT_FAMILY}/validation_arms.json",
        f"models/{OUTPUT_FAMILY}/model.json",
        f"models/{OUTPUT_FAMILY}/model.npz",
    ):
        if artifact not in run_record.setdefault("artifacts", []):
            run_record["artifacts"].append(artifact)
    _save_json(run_path, run_record)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.project.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
