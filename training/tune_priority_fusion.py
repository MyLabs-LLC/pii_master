"""Run the final 100 per-label fusion trials on internal validation."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from training.priority_data import PRIORITY_TAGS, READ_WINDOW, read_document
from training.priority_eval import EvaluationRow, aggregate_arms, evaluate_corpus
from training.priority_fusion import FusionPriorityModel
from training.priority_hash import load_priority_model
from training.tune_priority_hash import (
    _is_validation,
    _load_excluded,
    _load_jsonl,
    _objective,
    _save_json,
    fast_metrics,
)

FAMILY = "hybrid_priority"
N_TRIALS = 100
COMPONENT_DIRS = {
    "recall": "hash_sgd",
    "hash": "hash_sgd_f2",
    "tfidf": "tfidf_linear",
    "embedding": "embeddingbag_asl",
}
GENERIC_COMPONENTS = ("hash", "tfidf", "embedding")
OPTIONS = (
    "source:hash",
    "source:tfidf",
    "source:embedding",
    "or:hash,tfidf",
    "or:hash,embedding",
    "or:tfidf,embedding",
    "and:hash,tfidf",
    "and:hash,embedding",
    "and:tfidf,embedding",
    "majority:hash,tfidf,embedding",
    "or:hash,tfidf,embedding",
    "and:hash,tfidf,embedding",
)


def _is_calibration(dataset: str, uid: str) -> bool:
    return zlib.crc32(f"fusion::{dataset}::{uid}".encode()) % 5 == 0


def score_components(
    rows: list[dict[str, Any]],
    *,
    labels: tuple[str, ...],
    excluded: set[tuple[str, str]],
    models: dict[str, Any],
) -> dict[str, Any]:
    label_index = {label: index for index, label in enumerate(labels)}
    selected = [
        row
        for row in rows
        if _is_validation(row)
        and (row["dataset"], row["uid"]) not in excluded
        and not row.get("read_error")
    ]
    n_rows = len(selected)
    y_true = np.zeros((n_rows, len(labels)), dtype=np.bool_)
    predictions = {
        name: np.zeros((n_rows, len(labels)), dtype=np.bool_) for name in models
    }
    datasets: list[str] = []
    uids: list[str] = []
    complete = np.zeros(n_rows, dtype=np.bool_)
    calibration = np.zeros(n_rows, dtype=np.bool_)
    started = time.perf_counter()
    read_errors = 0
    for index, row in enumerate(selected):
        dataset, uid = row["dataset"], row["uid"]
        datasets.append(dataset)
        uids.append(uid)
        complete[index] = bool(row["label_complete"])
        calibration[index] = _is_calibration(dataset, uid)
        for label in row["labels"]:
            if label in label_index:
                y_true[index, label_index[label]] = True
        try:
            text = read_document(Path(row["path"]), limit=READ_WINDOW)
            for name, model in models.items():
                for label in model.predict(text):
                    predictions[name][index, label_index[label]] = True
        except (OSError, ValueError):
            read_errors += 1
        if (index + 1) % 10_000 == 0:
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "phase": "fusion_score",
                        "seen": index + 1,
                        "docs_per_s": round((index + 1) / elapsed, 1),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return {
        "rows": selected,
        "datasets": np.asarray(datasets),
        "uids": uids,
        "complete": complete,
        "calibration": calibration,
        "y_true": y_true,
        "predictions": predictions,
        "read_errors": read_errors,
    }


def _option_prediction(
    option: str, components: dict[str, np.ndarray], label_index: int
) -> np.ndarray:
    operation, raw_names = option.split(":", 1)
    names = raw_names.split(",")
    votes = np.stack([components[name][:, label_index] for name in names], axis=1)
    if operation == "source":
        return votes[:, 0]
    if operation == "or":
        return votes.any(axis=1)
    if operation == "and":
        return votes.all(axis=1)
    if operation == "majority":
        return votes.sum(axis=1) >= (votes.shape[1] // 2 + 1)
    raise ValueError(operation)


def rank_options(
    validation: dict[str, Any], labels: tuple[str, ...]
) -> dict[str, list[str]]:
    calibration = validation["calibration"] & validation["complete"]
    ranked: dict[str, list[str]] = {}
    for label_index, label in enumerate(labels):
        gold = validation["y_true"][calibration, label_index]
        scored: list[tuple[float, float, float, str]] = []
        for option in OPTIONS:
            pred = _option_prediction(
                option,
                {
                    name: values[calibration]
                    for name, values in validation["predictions"].items()
                },
                label_index,
            )
            tp = int((gold & pred).sum())
            fp = int((~gold & pred).sum())
            fn = int((gold & ~pred).sum())
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            denominator = 4.0 * precision + recall
            f2 = 5.0 * precision * recall / denominator if denominator else 0.0
            scored.append((f2, precision, recall, option))
        ranked[label] = [entry[3] for entry in sorted(scored, reverse=True)]
    return ranked


def strategy_configs(
    ranked: dict[str, list[str]], labels: tuple[str, ...], n_trials: int = N_TRIALS
) -> list[dict[str, str]]:
    priority = set(PRIORITY_TAGS)
    configs: list[dict[str, str]] = []
    # Systematic ranks first, then reproducible per-label top-three mixtures.
    for option_rank in range(len(OPTIONS)):
        configs.append(
            {
                label: "source:recall"
                if label in priority
                else ranked[label][option_rank]
                for label in labels
            }
        )
    rng = random.Random(20260828)
    attempts = 0
    while len(configs) < n_trials:
        attempts += 1
        if attempts > n_trials * 1_000:
            raise ValueError(
                "requested more unique fusion configs than the label space supports"
            )
        config = {}
        for label in labels:
            if label in priority:
                config[label] = "source:recall"
            else:
                draw = rng.random()
                choice = 0 if draw < 0.65 else (1 if draw < 0.9 else 2)
                config[label] = ranked[label][choice]
        if config not in configs:
            configs.append(config)
    return configs


def fuse_predictions(
    config: dict[str, str],
    labels: tuple[str, ...],
    components: dict[str, np.ndarray],
) -> np.ndarray:
    output = np.zeros_like(next(iter(components.values())))
    for label_index, label in enumerate(labels):
        output[:, label_index] = _option_prediction(
            config[label], components, label_index
        )
    return output


def _exact_arms(
    validation: dict[str, Any],
    selection: np.ndarray,
    labels: tuple[str, ...],
    predicted: np.ndarray,
    quality: dict[str, Any],
) -> list[dict[str, Any]]:
    arms = []
    labels_array = np.asarray(labels)
    for dataset in sorted(np.unique(validation["datasets"][selection])):
        indices = np.flatnonzero(selection & (validation["datasets"] == dataset))
        rows = [
            EvaluationRow(
                dataset=dataset,
                uid=validation["uids"][index],
                gold=frozenset(labels_array[validation["y_true"][index]]),
                predicted=frozenset(labels_array[predicted[index]]),
                label_complete=bool(validation["complete"][index]),
            )
            for index in indices
        ]
        arms.append(
            evaluate_corpus(
                rows,
                catalogue=sorted(quality[dataset]["tag_counts"]),
                bootstrap=False,
            )
        )
    return arms


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
    models = {
        name: load_priority_model(project / "models" / directory)
        for name, directory in COMPONENT_DIRS.items()
    }
    validation = score_components(rows, labels=labels, excluded=excluded, models=models)
    ranked = rank_options(validation, labels)
    configs = strategy_configs(ranked, labels, n_trials=n_trials)
    selection = ~validation["calibration"]
    selection_components = {
        name: values[selection] for name, values in validation["predictions"].items()
    }
    mlflow.set_tracking_uri((project / "mlruns").resolve().as_uri())
    mlflow.set_experiment("pii-priority-recall-v1")
    output_dir = project / "tuning" / FAMILY
    output_dir.mkdir(parents=True, exist_ok=True)
    best: dict[str, Any] | None = None
    with (output_dir / "trials.jsonl").open("w", encoding="utf-8") as stream:
        for trial_number, config in enumerate(configs):
            selected_predicted = fuse_predictions(config, labels, selection_components)
            metrics = fast_metrics(
                selected_predicted,
                validation["y_true"][selection],
                validation["datasets"][selection],
                validation["complete"][selection],
                labels,
                quality["train"],
            )
            with mlflow.start_run(
                run_name=f"{FAMILY}_trial_{trial_number:03d}"
            ) as active:
                mlflow.set_tags(
                    {
                        "model_family": FAMILY,
                        "dataset": "combined_internal_selection",
                        "split": "validation_selection",
                        "phase": "tune",
                        "trial_number": trial_number,
                    }
                )
                counts = {
                    option: list(config.values()).count(option) for option in OPTIONS
                }
                mlflow.log_params(
                    {
                        "priority_strategy": "source:recall",
                        "strategy_signature": zlib.crc32(
                            json.dumps(config, sort_keys=True).encode()
                        ),
                        **{
                            f"n_{key.replace(':', '_').replace(',', '_')}": value
                            for key, value in counts.items()
                        },
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
                "strategies": config,
                "metrics": metrics,
            }
            stream.write(json.dumps(trial, sort_keys=True) + "\n")
            if best is None or _objective(metrics) > _objective(best["metrics"]):
                full_predicted = fuse_predictions(
                    config, labels, validation["predictions"]
                )
                best = {**trial, "predicted": full_predicted}
            if (trial_number + 1) % 10 == 0:
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
    model_dir = project / "models" / FAMILY
    model = FusionPriorityModel(
        labels=labels, components=models, strategies=best["strategies"]
    )
    model.save(
        model_dir,
        metadata={
            "family": FAMILY,
            "trial": best["trial"],
            "mlflow_run_id": best["mlflow_run_id"],
            "validation_metrics": best["metrics"],
        },
    )
    exact_arms = _exact_arms(
        validation, selection, labels, best["predicted"], quality["train"]
    )
    exact_summary = aggregate_arms(exact_arms)
    resolved = {
        "family": FAMILY,
        "n_trials": n_trials,
        "best_trial": best["trial"],
        "best_mlflow_run_id": best["mlflow_run_id"],
        "best_strategies": best["strategies"],
        "fast_metrics": best["metrics"],
        "exact_metrics": exact_summary,
        "calibration_rows": int(validation["calibration"].sum()),
        "selection_rows": int(selection.sum()),
        "validation_read_errors": validation["read_errors"],
        "model_dir": str(model_dir.relative_to(project)),
    }
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
    result = run(args.project.resolve(), n_trials=args.trials)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
