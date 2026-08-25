"""Fit and tune the approved hash_sgd family on an internal hash split."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from training.priority_data import PRIORITY_TAGS, READ_WINDOW, read_document
from training.priority_eval import EvaluationRow, aggregate_arms, evaluate_corpus
from training.priority_hash import (
    HashCounts,
    HashCueModel,
    build_weights,
    document_features,
    score_modes,
)

FAMILY = "hash_sgd"
N_TRIALS = 300
TARGETS = np.asarray(
    [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.92, 0.94, 0.96, 0.98, 0.99, 0.995, 0.999]
)
PRIORITY_TARGET_INDICES = tuple(range(5, len(TARGETS)))
GENERIC_TARGET_INDICES = tuple(range(10))
SCORE_MODES = ("top1", "top3", "top6")


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


def _is_validation(row: dict[str, Any]) -> bool:
    return int(row["text_sha256"][:8], 16) % 10 == 0


def _load_excluded(path: Path) -> set[tuple[str, str]]:
    exclusions = json.loads(path.read_text(encoding="utf-8"))
    return {(str(row["dataset"]), str(row["uid"])) for row in exclusions}


def fit_counts(
    rows: list[dict[str, Any]],
    *,
    labels: tuple[str, ...],
    excluded: set[tuple[str, str]],
    output: Path,
) -> tuple[HashCounts, dict[str, Any]]:
    counts = HashCounts.empty(labels)
    source_rows: dict[str, int] = defaultdict(int)
    started = time.perf_counter()
    skipped_excluded = skipped_validation = skipped_error = 0
    for seen, row in enumerate(rows, 1):
        key = (row["dataset"], row["uid"])
        if key in excluded:
            skipped_excluded += 1
            continue
        if _is_validation(row):
            skipped_validation += 1
            continue
        if row.get("read_error"):
            skipped_error += 1
            continue
        try:
            text = read_document(Path(row["path"]), limit=READ_WINDOW)
        except (OSError, ValueError):
            skipped_error += 1
            continue
        features = document_features(text)
        counts.update(
            features, set(row["labels"]), label_complete=bool(row["label_complete"])
        )
        source_rows[row["dataset"]] += 1
        if seen % 25_000 == 0:
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "phase": "fit_counts",
                        "seen": seen,
                        "used": counts.n_all,
                        "docs_per_s": round(counts.n_all / elapsed, 1),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    counts.save(output)
    elapsed = time.perf_counter() - started
    stats = {
        "fit_rows": counts.n_all,
        "complete_rows": counts.n_complete,
        "partial_rows": counts.n_all - counts.n_complete,
        "skipped_eval_leakage": skipped_excluded,
        "internal_validation_rows": skipped_validation,
        "read_errors": skipped_error,
        "source_rows": dict(sorted(source_rows.items())),
        "elapsed_s": elapsed,
        "docs_per_s": counts.n_all / elapsed if elapsed else 0.0,
    }
    return counts, stats


def score_validation(
    rows: list[dict[str, Any]],
    *,
    labels: tuple[str, ...],
    excluded: set[tuple[str, str]],
    weights: np.ndarray,
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
    scores = {
        mode: np.zeros((n_rows, len(labels)), dtype=np.float32) for mode in SCORE_MODES
    }
    datasets: list[str] = []
    uids: list[str] = []
    complete = np.zeros(n_rows, dtype=np.bool_)
    started = time.perf_counter()
    read_errors = 0
    for index, row in enumerate(selected):
        datasets.append(row["dataset"])
        uids.append(row["uid"])
        complete[index] = bool(row["label_complete"])
        for label in row["labels"]:
            if label in label_index:
                y_true[index, label_index[label]] = True
        try:
            text = read_document(Path(row["path"]), limit=READ_WINDOW)
            feature_ids = document_features(text)
            values = score_modes(weights, feature_ids)
            for mode in SCORE_MODES:
                scores[mode][index] = values[mode]
        except (OSError, ValueError):
            read_errors += 1
        if (index + 1) % 10_000 == 0:
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "phase": "score_validation",
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
        "y_true": y_true,
        "scores": scores,
        "read_errors": read_errors,
    }


def threshold_bank(
    scores: dict[str, np.ndarray],
    y_true: np.ndarray,
    datasets: np.ndarray,
    labels: tuple[str, ...],
) -> dict[str, np.ndarray]:
    priority = set(PRIORITY_TAGS)
    bank: dict[str, np.ndarray] = {}
    for mode, mode_scores in scores.items():
        values = np.full((len(labels), len(TARGETS)), np.inf, dtype=np.float32)
        for label_index, label in enumerate(labels):
            positive = y_true[:, label_index]
            if not positive.any():
                continue
            for target_index, target in enumerate(TARGETS):
                if label in priority:
                    per_source: list[float] = []
                    for dataset in np.unique(datasets):
                        source_positive = positive & (datasets == dataset)
                        if source_positive.sum() < 30:
                            continue
                        per_source.append(
                            float(
                                np.quantile(
                                    mode_scores[source_positive, label_index],
                                    1.0 - target,
                                    method="lower",
                                )
                            )
                        )
                    if per_source:
                        values[label_index, target_index] = min(per_source)
                        continue
                values[label_index, target_index] = float(
                    np.quantile(
                        mode_scores[positive, label_index], 1.0 - target, method="lower"
                    )
                )
        bank[mode] = values
    return bank


def trial_configs(
    n_trials: int = N_TRIALS, seed: int = 20260825
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    candidates: list[dict[str, Any]] = []
    for mode in SCORE_MODES:
        for priority_index in PRIORITY_TARGET_INDICES:
            for generic_index in GENERIC_TARGET_INDICES:
                for multiplier in (0.85, 0.95, 1.0, 1.05):
                    candidates.append(
                        {
                            "score_mode": mode,
                            "priority_target_index": priority_index,
                            "generic_target_index": generic_index,
                            "threshold_multiplier": multiplier,
                        }
                    )
    rng.shuffle(candidates)
    # Anchor trials make the critical boundary reproducible even if the random
    # subset changes in a later, explicitly approved round.
    anchors = [
        {
            "score_mode": mode,
            "priority_target_index": priority_index,
            "generic_target_index": generic_index,
            "threshold_multiplier": 1.0,
        }
        for mode in SCORE_MODES
        for priority_index in (5, 8, 10, 12)
        for generic_index in (3, 5, 7, 9)
    ]
    ordered: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for candidate in anchors + candidates:
        key = tuple(candidate.values())
        if key not in seen:
            ordered.append(candidate)
            seen.add(key)
        if len(ordered) == n_trials:
            break
    return ordered


def _thresholds_for_trial(
    config: dict[str, Any], bank: dict[str, np.ndarray], labels: tuple[str, ...]
) -> np.ndarray:
    table = bank[config["score_mode"]]
    thresholds = np.empty(len(labels), dtype=np.float32)
    priority = set(PRIORITY_TAGS)
    for index, label in enumerate(labels):
        target_index = (
            config["priority_target_index"]
            if label in priority
            else config["generic_target_index"]
        )
        thresholds[index] = table[index, target_index]
    finite = np.isfinite(thresholds)
    thresholds[finite] *= config["threshold_multiplier"]
    return thresholds


def fast_metrics(
    predicted: np.ndarray,
    y_true: np.ndarray,
    datasets: np.ndarray,
    complete: np.ndarray,
    labels: tuple[str, ...],
    data_quality: dict[str, Any],
) -> dict[str, Any]:
    label_index = {label: index for index, label in enumerate(labels)}
    priority_indices = [
        label_index[label] for label in PRIORITY_TAGS if label in label_index
    ]
    arm_results: list[dict[str, Any]] = []
    all_gate_recalls: list[float] = []
    point_passes = measurable_gates = 0
    for dataset in sorted(np.unique(datasets)):
        mask = datasets == dataset
        gold = y_true[mask]
        pred = predicted[mask]
        support = gold.sum(axis=0)
        true_positive = (gold & pred).sum(axis=0)
        recalls = np.divide(
            true_positive,
            support,
            out=np.zeros(len(labels), dtype=np.float64),
            where=support > 0,
        )
        for index in priority_indices:
            if support[index] >= 30:
                recall = float(recalls[index])
                all_gate_recalls.append(recall)
                measurable_gates += 1
                point_passes += recall >= 0.90
        is_complete = bool(complete[mask].all())
        macro_f2 = micro_f1 = None
        if is_complete:
            false_positive = (pred & ~gold).sum(axis=0)
            catalog_indices = [
                label_index[label]
                for label in data_quality[dataset]["tag_counts"]
                if label in label_index
            ]
            tp = true_positive[catalog_indices].astype(np.float64)
            fp = false_positive[catalog_indices].astype(np.float64)
            fn = (support[catalog_indices] - true_positive[catalog_indices]).astype(
                np.float64
            )
            precision = np.divide(
                tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0
            )
            recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
            denominator = 4.0 * precision + recall
            f2 = np.divide(
                5.0 * precision * recall,
                denominator,
                out=np.zeros_like(tp),
                where=denominator > 0,
            )
            macro_f2 = float(f2.mean()) if len(f2) else None
            total_tp, total_fp, total_fn = tp.sum(), fp.sum(), fn.sum()
            micro_f1 = (
                float(2.0 * total_tp / (2.0 * total_tp + total_fp + total_fn))
                if total_tp
                else 0.0
            )
        arm_results.append(
            {"dataset": dataset, "macro_f2": macro_f2, "micro_f1": micro_f1}
        )
    f2_values = [arm["macro_f2"] for arm in arm_results if arm["macro_f2"] is not None]
    f1_values = [arm["micro_f1"] for arm in arm_results if arm["micro_f1"] is not None]
    return {
        "measurable_priority_gates": measurable_gates,
        "priority_point_passes": point_passes,
        "worst_priority_recall": min(all_gate_recalls) if all_gate_recalls else 0.0,
        "equal_corpus_macro_f2": float(np.mean(f2_values)) if f2_values else 0.0,
        "equal_corpus_micro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "per_corpus": arm_results,
    }


def _objective(metrics: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(metrics["priority_point_passes"]),
        float(metrics["worst_priority_recall"]),
        float(metrics["equal_corpus_macro_f2"]),
        float(metrics["equal_corpus_micro_f1"]),
    )


def _exact_best_arms(
    validation: dict[str, Any],
    labels: tuple[str, ...],
    predicted: np.ndarray,
    data_quality: dict[str, Any],
) -> list[dict[str, Any]]:
    arms: list[dict[str, Any]] = []
    datasets = validation["datasets"]
    for dataset in sorted(np.unique(datasets)):
        indices = np.flatnonzero(datasets == dataset)
        rows = [
            EvaluationRow(
                dataset=dataset,
                uid=validation["uids"][index],
                gold=frozenset(np.asarray(labels)[validation["y_true"][index]]),
                predicted=frozenset(np.asarray(labels)[predicted[index]]),
                label_complete=bool(validation["complete"][index]),
            )
            for index in indices
        ]
        arms.append(
            evaluate_corpus(
                rows,
                catalogue=sorted(data_quality[dataset]["tag_counts"]),
                bootstrap=False,
            )
        )
    return arms


def run(project: Path, *, n_trials: int) -> dict[str, Any]:
    import mlflow

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    data_dir = project / "data"
    cache_dir = project / "cache" / FAMILY
    model_dir = project / "models" / FAMILY
    output_dir = project / "tuning" / FAMILY
    rows = _load_jsonl(data_dir / "train_index.jsonl")
    excluded = _load_excluded(data_dir / "train_exclusions.json")
    frozen = json.loads(
        (data_dir / "evaluation_catalogue.json").read_text(encoding="utf-8")
    )
    quality = json.loads((data_dir / "data_quality.json").read_text(encoding="utf-8"))
    labels = tuple(frozen["full_catalogue"])

    counts_path = cache_dir / "counts.npz"
    if counts_path.exists():
        counts = HashCounts.load(counts_path)
        fit_stats = json.loads(
            (cache_dir / "fit_stats.json").read_text(encoding="utf-8")
        )
    else:
        counts, fit_stats = fit_counts(
            rows,
            labels=labels,
            excluded=excluded,
            output=counts_path,
        )
        _save_json(cache_dir / "fit_stats.json", fit_stats)
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

    mlflow.set_tracking_uri((project / "mlruns").resolve().as_uri())
    mlflow.set_experiment("pii-priority-recall-v1")
    configs = trial_configs(n_trials=n_trials)
    output_dir.mkdir(parents=True, exist_ok=True)
    trials_path = output_dir / "trials.jsonl"
    best: dict[str, Any] | None = None
    with trials_path.open("w", encoding="utf-8") as stream:
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
            "fit_stats": fit_stats,
            "validation_metrics": best["metrics"],
        },
    )
    exact_arms = _exact_best_arms(
        validation, labels, best["predicted"], quality["train"]
    )
    exact_summary = aggregate_arms(exact_arms)
    resolved = {
        "family": FAMILY,
        "n_trials": len(configs),
        "best_trial": best["trial"],
        "best_mlflow_run_id": best["mlflow_run_id"],
        "best_config": best["config"],
        "fast_metrics": best["metrics"],
        "exact_metrics": exact_summary,
        "fit_stats": fit_stats,
        "validation_rows": len(validation["rows"]),
        "validation_read_errors": validation["read_errors"],
        "model_dir": str(model_dir.relative_to(project)),
    }
    _save_json(output_dir / "resolved_config.json", resolved)
    _save_json(output_dir / "validation_arms.json", exact_arms)

    run_path = project / "run.json"
    run_record = json.loads(run_path.read_text(encoding="utf-8"))
    run_record.setdefault("run_summary", {})[FAMILY] = {
        "trials": len(configs),
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
    result = run(args.project.resolve(), n_trials=args.trials)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
