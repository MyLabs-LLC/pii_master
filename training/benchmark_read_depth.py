"""Quality and one-core speed ladder for finalist document read depths."""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from training.priority_data import read_document
from training.priority_eval import (
    aggregate_arms,
    evaluate_corpus,
    rows_from_predictions,
)
from training.priority_hash import load_priority_model
from training.tune_priority_hash import _load_jsonl, _save_json

FINALISTS = ("hash_sgd", "hybrid_priority_001", "hybrid_priority")
READ_DEPTHS = (1_000, 2_500, 10_000, 20_000)
_WORKER_MODEL: Any = None
_WORKER_DEPTH = 20_000


def _load_worker(model_dir: str, read_depth: int) -> None:
    global _WORKER_MODEL, _WORKER_DEPTH
    _WORKER_MODEL = load_priority_model(Path(model_dir))
    _WORKER_DEPTH = read_depth


def _predict_one(payload: tuple[str, str, str]) -> tuple[str, str, list[str], str]:
    dataset, uid, raw_path = payload
    try:
        text = read_document(Path(raw_path), limit=_WORKER_DEPTH)
        return dataset, uid, _WORKER_MODEL.predict(text), ""
    except Exception as exc:  # noqa: BLE001 - persist document-level failures
        return dataset, uid, [], f"{type(exc).__name__}: {exc}"


def stratified_sample(
    rows: list[dict[str, Any]], *, per_dataset: int = 125
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row.get("read_error"):
            grouped[row["dataset"]].append(row)
    sample: list[dict[str, Any]] = []
    for dataset in sorted(grouped):
        corpus = grouped[dataset]
        if len(corpus) <= per_dataset:
            sample.extend(corpus)
            continue
        positions = np.linspace(0, len(corpus) - 1, per_dataset, dtype=np.int64)
        sample.extend(corpus[int(position)] for position in positions)
    return sample


def quality_arm(
    project: Path,
    *,
    family: str,
    read_depth: int,
    workers: int,
    index_rows: list[dict[str, Any]],
    frozen: dict[str, Any],
) -> dict[str, Any]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in index_rows:
        by_dataset[row["dataset"]].append(row)
    predictions: dict[tuple[str, str], list[str]] = {}
    errors: dict[str, int] = defaultdict(int)
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_load_worker,
        initargs=(str(project / "models" / family), read_depth),
    ) as executor:
        for dataset in sorted(by_dataset):
            tasks = [(dataset, row["uid"], row["path"]) for row in by_dataset[dataset]]
            for result_dataset, uid, labels, error in executor.map(
                _predict_one, tasks, chunksize=64
            ):
                predictions[(result_dataset, uid)] = labels
                errors[result_dataset] += bool(error)
    grouped = rows_from_predictions(index_rows, predictions)
    output_dir = project / "benchmarks" / "read_depth" / family / str(read_depth)
    arms = []
    for dataset in sorted(grouped):
        result = evaluate_corpus(
            grouped[dataset],
            catalogue=frozen["corpora"][dataset]["catalogue"],
            bootstrap=False,
        )
        result["read_errors"] = errors[dataset]
        _save_json(output_dir / f"{dataset}.json", result)
        arms.append(result)
    aggregate = aggregate_arms(arms)
    _save_json(
        output_dir / "summary.json",
        {"family": family, "read_depth_chars": read_depth, "aggregate": aggregate},
    )
    print(
        json.dumps(
            {
                "phase": "quality",
                "family": family,
                "read_depth": read_depth,
                "macro_f2": aggregate["equal_corpus_macro_f2"],
                "priority_passes": aggregate["priority_point_passes"],
                "worst_recall": aggregate["worst_priority_recall"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return aggregate


def latency_arm(
    project: Path,
    *,
    family: str,
    read_depth: int,
    sample: list[dict[str, Any]],
) -> dict[str, Any]:
    model = load_priority_model(project / "models" / family)
    for row in sample[:20]:
        model.predict(read_document(Path(row["path"]), limit=read_depth))
    latencies_ms: list[float] = []
    gc.disable()
    try:
        for row in sample:
            started = time.perf_counter_ns()
            text = read_document(Path(row["path"]), limit=read_depth)
            model.predict(text)
            latencies_ms.append((time.perf_counter_ns() - started) / 1_000_000)
    finally:
        gc.enable()
    values = np.asarray(latencies_ms)
    elapsed_s = float(values.sum() / 1_000.0)
    return {
        "n_documents": len(sample),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "mean_ms": float(values.mean()),
        "docs_per_s": len(sample) / elapsed_s if elapsed_s else 0.0,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }


def run(project: Path, *, workers: int, sample_per_dataset: int) -> dict[str, Any]:
    import mlflow
    from threadpoolctl import threadpool_limits

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    index_rows = _load_jsonl(project / "data" / "eval_index.jsonl")
    frozen = json.loads(
        (project / "data" / "evaluation_catalogue.json").read_text(encoding="utf-8")
    )
    quality: dict[str, dict[int, dict[str, Any]]] = {family: {} for family in FINALISTS}
    for family in FINALISTS:
        for read_depth in READ_DEPTHS:
            quality[family][read_depth] = quality_arm(
                project,
                family=family,
                read_depth=read_depth,
                workers=workers,
                index_rows=index_rows,
                frozen=frozen,
            )

    sample = stratified_sample(index_rows, per_dataset=sample_per_dataset)
    original_affinity = os.sched_getaffinity(0)
    benchmark_core = min(original_affinity)
    os.sched_setaffinity(0, {benchmark_core})
    latency: dict[str, dict[int, dict[str, Any]]] = {family: {} for family in FINALISTS}
    try:
        with threadpool_limits(limits=1):
            for family in FINALISTS:
                for read_depth in READ_DEPTHS:
                    latency[family][read_depth] = latency_arm(
                        project,
                        family=family,
                        read_depth=read_depth,
                        sample=sample,
                    )
                    print(
                        json.dumps(
                            {
                                "phase": "one_core_latency",
                                "family": family,
                                "read_depth": read_depth,
                                **latency[family][read_depth],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
    finally:
        os.sched_setaffinity(0, original_affinity)

    rows = []
    for family in FINALISTS:
        baseline_quality = quality[family][20_000]
        baseline_latency = latency[family][20_000]
        for read_depth in READ_DEPTHS:
            q = quality[family][read_depth]
            speed = latency[family][read_depth]
            row = {
                "family": family,
                "read_depth_chars": read_depth,
                **speed,
                "p95_speedup_vs_20k": baseline_latency["p95_ms"] / speed["p95_ms"],
                "p95_delta_ms_vs_20k": speed["p95_ms"] - baseline_latency["p95_ms"],
                "equal_corpus_macro_f2": q["equal_corpus_macro_f2"],
                "macro_f2_delta_vs_20k": q["equal_corpus_macro_f2"]
                - baseline_quality["equal_corpus_macro_f2"],
                "equal_corpus_micro_f1": q["equal_corpus_micro_f1"],
                "priority_point_passes": q["priority_point_passes"],
                "measurable_priority_gates": q["measurable_priority_gates"],
                "worst_priority_recall": q["worst_priority_recall"],
                "worst_recall_delta_vs_20k": q["worst_priority_recall"]
                - baseline_quality["worst_priority_recall"],
            }
            rows.append(row)
    result = {
        "cpu_affinity_core": benchmark_core,
        "workers_for_quality": workers,
        "sample_documents": len(sample),
        "read_depths_chars": list(READ_DEPTHS),
        "rows": rows,
    }
    output_path = project / "benchmarks" / "read_depth.json"
    _save_json(output_path, result)
    mlflow.set_tracking_uri((project / "mlruns").resolve().as_uri())
    mlflow.set_experiment("pii-priority-recall-v1")
    for row in rows:
        with mlflow.start_run(
            run_name=f"read_depth__{row['family']}__{row['read_depth_chars']}"
        ):
            mlflow.set_tags(
                {
                    "phase": "read_depth_benchmark",
                    "model": row["family"],
                    "dataset": "stratified_eval_latency_and_all_eval_quality",
                    "split": "eval",
                }
            )
            mlflow.log_param("read_depth_chars", row["read_depth_chars"])
            mlflow.log_param("one_core", True)
            mlflow.log_metrics(
                {
                    key: value
                    for key, value in row.items()
                    if key not in {"family", "read_depth_chars"}
                }
            )
    run_path = project / "run.json"
    run_record = json.loads(run_path.read_text(encoding="utf-8"))
    run_record.setdefault("run_summary", {})["read_depth_benchmark"] = {
        "arms": len(rows),
        "artifact": "benchmarks/read_depth.json",
    }
    if "benchmarks/read_depth.json" not in run_record.setdefault("artifacts", []):
        run_record["artifacts"].append("benchmarks/read_depth.json")
    _save_json(run_path, run_record)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--sample-per-dataset", type=int, default=125)
    args = parser.parse_args()
    result = run(
        args.project.resolve(),
        workers=args.workers,
        sample_per_dataset=args.sample_per_dataset,
    )
    print(json.dumps({"rows": len(result["rows"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
