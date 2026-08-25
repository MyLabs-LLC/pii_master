"""Evaluate a saved priority model on each untouched hold-out corpus."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from training.priority_data import read_document
from training.priority_eval import record_artifacts as _record_artifacts
from training.priority_eval import (
    aggregate_arms,
    evaluate_corpus,
    rows_from_predictions,
)
from training.priority_hash import (
    HashCueModel,
    HybridPriorityModel,
    load_priority_model,
)

_WORKER_MODEL: HashCueModel | HybridPriorityModel | None = None


def _load_worker(model_dir: str) -> None:
    global _WORKER_MODEL
    _WORKER_MODEL = load_priority_model(Path(model_dir))


def _predict_one(payload: tuple[str, str, str]) -> tuple[str, str, list[str], str]:
    dataset, uid, raw_path = payload
    assert _WORKER_MODEL is not None
    try:
        text = read_document(Path(raw_path), limit=_WORKER_MODEL.read_window_chars)
        return dataset, uid, _WORKER_MODEL.predict(text), ""
    except Exception as exc:  # noqa: BLE001 - per-document evidence, not a silent drop
        return dataset, uid, [], f"{type(exc).__name__}: {exc}"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def run(project: Path, *, family: str, workers: int) -> dict[str, Any]:
    import mlflow

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    model_dir = project / "models" / family
    model = load_priority_model(model_dir)
    index_rows = _load_jsonl(project / "data" / "eval_index.jsonl")
    frozen = json.loads(
        (project / "data" / "evaluation_catalogue.json").read_text(encoding="utf-8")
    )
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in index_rows:
        by_dataset[row["dataset"]].append(row)

    prediction_map: dict[tuple[str, str], list[str]] = {}
    errors: dict[str, list[str]] = defaultdict(list)
    speed: dict[str, dict[str, float]] = {}
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_load_worker,
        initargs=(str(model_dir),),
    ) as executor:
        for dataset in sorted(by_dataset):
            corpus = by_dataset[dataset]
            tasks = [(dataset, row["uid"], row["path"]) for row in corpus]
            started = time.perf_counter()
            for result_dataset, uid, labels, error in executor.map(
                _predict_one, tasks, chunksize=64
            ):
                prediction_map[(result_dataset, uid)] = labels
                if error:
                    errors[result_dataset].append(f"{uid}: {error}")
            elapsed = time.perf_counter() - started
            speed[dataset] = {
                "elapsed_s": elapsed,
                "docs_per_s": len(corpus) / elapsed if elapsed else 0.0,
            }
            print(
                json.dumps(
                    {
                        "dataset": dataset,
                        "rows": len(corpus),
                        "docs_per_s": round(speed[dataset]["docs_per_s"], 2),
                        "errors": len(errors[dataset]),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    output_dir = project / "evaluations" / family
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped = rows_from_predictions(index_rows, prediction_map)
    mlflow.set_tracking_uri((project / "mlruns").resolve().as_uri())
    mlflow.set_experiment("pii-priority-recall-v1")
    arm_results: list[dict[str, Any]] = []
    arm_records: list[dict[str, Any]] = []
    for dataset in sorted(grouped):
        result = evaluate_corpus(
            grouped[dataset],
            catalogue=frozen["corpora"][dataset]["catalogue"],
            bootstrap=False,
        )
        result["speed"] = speed[dataset]
        result["read_errors"] = errors[dataset][:100]
        result_path = output_dir / f"{dataset}.json"
        _save_json(result_path, result)
        with mlflow.start_run(run_name=f"{family}__{dataset}") as active:
            mlflow.set_tags(
                {
                    "model": family,
                    "dataset": dataset,
                    "split": "eval",
                    "phase": "holdout_validation",
                    "arm_key": f"{family}::{dataset}",
                }
            )
            mlflow.log_params(
                {
                    "read_window_chars": model.read_window_chars,
                    "workers": workers,
                    "score_mode": model.score_mode,
                    "bootstrap": False,
                    "label_complete": result["label_complete"],
                }
            )
            metrics = {
                "docs_per_s": speed[dataset]["docs_per_s"],
                "priority_measurable": result["priority_summary"]["measurable_tags"],
                "priority_point_passes": result["priority_summary"]["point_passes"],
                "priority_worst_recall": result["priority_summary"]["worst_recall"]
                or 0.0,
            }
            if result["macro_f2"] is not None:
                metrics["macro_f2"] = result["macro_f2"]
                metrics["micro_f1"] = result["micro_f1"]
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(result_path), artifact_path="evaluation")
            run_id = active.info.run_id
        arm_records.append(
            {
                "model": family,
                "dataset": dataset,
                "mlflow_run_id": run_id,
                "metrics": {
                    "macro_f2": result["macro_f2"],
                    "micro_f1": result["micro_f1"],
                    "worst_priority_recall": result["priority_summary"]["worst_recall"],
                    "priority_point_passes": result["priority_summary"]["point_passes"],
                    "priority_measurable": result["priority_summary"][
                        "measurable_tags"
                    ],
                    "docs_per_s": speed[dataset]["docs_per_s"],
                },
                "artifact": str(result_path.relative_to(project)),
            }
        )
        arm_results.append(result)

    predictions_path = output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as stream:
        for (dataset, uid), labels in sorted(prediction_map.items()):
            stream.write(
                json.dumps(
                    {"dataset": dataset, "uid": uid, "labels": labels}, sort_keys=True
                )
                + "\n"
            )
    summary = {
        "model": family,
        "aggregate": aggregate_arms(arm_results),
        "arms": arm_records,
        "read_window_chars": model.read_window_chars,
        "workers": workers,
    }
    _save_json(output_dir / "summary.json", summary)

    run_path = project / "run.json"
    run_record = json.loads(run_path.read_text(encoding="utf-8"))
    run_record["arms"] = [
        arm for arm in run_record.get("arms", []) if arm.get("model") != family
    ]
    run_record["arms"].extend(arm_records)
    run_record.setdefault("run_summary", {}).setdefault(family, {})["holdout"] = (
        summary["aggregate"]
    )
    _record_artifacts(
        run_record,
        {
            f"holdout_summary::{family}": f"evaluations/{family}/summary.json",
            f"holdout_predictions::{family}": f"evaluations/{family}/predictions.jsonl",
        },
    )
    _save_json(run_path, run_record)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--family", required=True)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    args = parser.parse_args()
    result = run(args.project.resolve(), family=args.family, workers=args.workers)
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
