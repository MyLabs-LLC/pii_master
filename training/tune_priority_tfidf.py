"""Tune the approved TF-IDF linear family on the fixed internal split."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from training.priority_data import READ_WINDOW
from training.priority_eval import aggregate_arms
from training.priority_hash import HashCounts, HashCueModel, build_weights
from training.tune_priority_hash import (
    N_TRIALS,
    TARGETS,
    _exact_best_arms,
    _load_excluded,
    _load_jsonl,
    _objective,
    _save_json,
    _thresholds_for_trial,
    fast_metrics,
    score_validation,
    threshold_bank,
    trial_configs,
)

FAMILY = "tfidf_linear"


def build_tfidf_weights(
    counts: HashCounts,
    *,
    idf_power: float = 0.65,
    alpha: float = 1.0,
    partial_weight: float = 0.75,
    min_document_frequency: int = 3,
) -> np.ndarray:
    """Combine supervised log-odds coefficients with unsupervised TF-IDF."""
    weights = build_weights(
        counts,
        alpha=alpha,
        partial_weight=partial_weight,
        min_document_frequency=min_document_frequency,
    )
    idf = np.log((counts.n_all + 1.0) / (counts.all_df.astype(np.float32) + 1.0)) + 1.0
    np.clip(idf, 1.0, 6.0, out=idf)
    weights *= np.power(idf, idf_power)[np.newaxis, :]
    return weights


def _run_trials(
    *,
    project: Path,
    validation: dict[str, Any],
    quality: dict[str, Any],
    labels: tuple[str, ...],
    bank: dict[str, np.ndarray],
    n_trials: int,
) -> dict[str, Any]:
    import mlflow

    configs = trial_configs(n_trials=n_trials, seed=20260826)
    output_dir = project / "tuning" / FAMILY
    output_dir.mkdir(parents=True, exist_ok=True)
    best: dict[str, Any] | None = None
    with (output_dir / "trials.jsonl").open("w", encoding="utf-8") as stream:
        for trial_number, config in enumerate(configs):
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
            with mlflow.start_run(
                run_name=f"{FAMILY}_trial_{trial_number:03d}"
            ) as active:
                mlflow.set_tags(
                    {
                        "model_family": FAMILY,
                        "dataset": "combined_internal_validation",
                        "split": "validation",
                        "phase": "tune",
                        "trial_number": trial_number,
                    }
                )
                mlflow.log_params(
                    {
                        **config,
                        "priority_target": TARGETS[config["priority_target_index"]],
                        "generic_target": TARGETS[config["generic_target_index"]],
                        "idf_power": 0.65,
                        "alpha": 1.0,
                        "partial_weight": 0.75,
                        "min_document_frequency": 3,
                        "read_window_chars": READ_WINDOW,
                    }
                )
                mlflow.log_metrics(
                    {
                        key: value
                        for key, value in metrics.items()
                        if key != "per_corpus"
                    }
                )
                run_id = active.info.run_id
            trial = {
                "trial": trial_number,
                "mlflow_run_id": run_id,
                "config": config,
                "metrics": metrics,
            }
            stream.write(json.dumps(trial, sort_keys=True) + "\n")
            if best is None or _objective(metrics) > _objective(best["metrics"]):
                best = {**trial, "thresholds": thresholds, "predicted": predicted}
            if (trial_number + 1) % 25 == 0:
                assert best is not None
                print(
                    json.dumps(
                        {
                            "phase": "tune",
                            "family": FAMILY,
                            "trials": trial_number + 1,
                            "best_trial": best["trial"],
                            "best": {
                                key: value
                                for key, value in best["metrics"].items()
                                if key != "per_corpus"
                            },
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    assert best is not None
    return best


def run(project: Path, *, n_trials: int) -> dict[str, Any]:
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
    counts = HashCounts.load(project / "cache" / "hash_sgd" / "counts.npz")
    weights = build_tfidf_weights(counts)
    validation = score_validation(
        rows, labels=labels, excluded=excluded, weights=weights
    )
    bank = threshold_bank(
        validation["scores"], validation["y_true"], validation["datasets"], labels
    )
    mlflow.set_tracking_uri((project / "mlruns").resolve().as_uri())
    mlflow.set_experiment("pii-priority-recall-v1")
    best = _run_trials(
        project=project,
        validation=validation,
        quality=quality,
        labels=labels,
        bank=bank,
        n_trials=n_trials,
    )
    model_dir = project / "models" / FAMILY
    model = HashCueModel(
        labels=labels,
        weights=weights,
        thresholds=best["thresholds"],
        score_mode=best["config"]["score_mode"],
        read_window_chars=READ_WINDOW,
    )
    model.save(
        model_dir,
        metadata={
            "family": FAMILY,
            "trial": best["trial"],
            "mlflow_run_id": best["mlflow_run_id"],
            "idf_power": 0.65,
            "source_counts": "hash_sgd leakage-safe combined fit partition",
            "validation_metrics": best["metrics"],
        },
    )
    exact_arms = _exact_best_arms(
        validation, labels, best["predicted"], quality["train"]
    )
    exact_summary = aggregate_arms(exact_arms)
    resolved = {
        "family": FAMILY,
        "n_trials": n_trials,
        "best_trial": best["trial"],
        "best_mlflow_run_id": best["mlflow_run_id"],
        "best_config": best["config"],
        "idf_power": 0.65,
        "fast_metrics": best["metrics"],
        "exact_metrics": exact_summary,
        "validation_rows": len(validation["rows"]),
        "validation_read_errors": validation["read_errors"],
        "model_dir": str(model_dir.relative_to(project)),
    }
    output_dir = project / "tuning" / FAMILY
    _save_json(output_dir / "resolved_config.json", resolved)
    _save_json(output_dir / "validation_arms.json", exact_arms)
    run_path = project / "run.json"
    run_record = json.loads(run_path.read_text(encoding="utf-8"))
    run_record.setdefault("run_summary", {})[FAMILY] = {
        "trials": n_trials,
        "best_trial": best["trial"],
        "metrics": best["metrics"],
        "model": str(model_dir.relative_to(project)),
    }
    for artifact in (
        f"tuning/{FAMILY}/trials.jsonl",
        f"tuning/{FAMILY}/resolved_config.json",
        f"tuning/{FAMILY}/validation_arms.json",
        f"models/{FAMILY}/model.json",
        f"models/{FAMILY}/model.npz",
    ):
        if artifact not in run_record.setdefault("artifacts", []):
            run_record["artifacts"].append(artifact)
    _save_json(run_path, run_record)
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.project.resolve(), n_trials=args.trials), indent=2, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
