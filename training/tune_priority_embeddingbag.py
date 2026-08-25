"""Tune low-rank EmbeddingBag heads with asymmetric-loss calibration."""

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

from training.priority_data import READ_WINDOW, read_document
from training.priority_embeddingbag import LowRankEmbeddingBagModel
from training.priority_eval import EvaluationRow, aggregate_arms, evaluate_corpus
from training.priority_hash import HashCounts, document_features
from training.tune_priority_hash import (
    N_TRIALS,
    TARGETS,
    _load_excluded,
    _load_jsonl,
    _objective,
    _save_json,
    fast_metrics,
    threshold_bank,
)
from training.tune_priority_tfidf import build_tfidf_weights

FAMILY = "embeddingbag_asl"
RANKS = (8, 16, 24, 32)
GAMMA_POS = 0.0
GAMMA_NEG = 4.0
NEGATIVE_CLIP = 0.05


def _is_calibration(dataset: str, uid: str) -> bool:
    return zlib.crc32(f"{dataset}::{uid}".encode()) % 5 == 0


def factorize_ranks(weights: np.ndarray) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    positive = np.maximum(weights.astype(np.float32), 0.0)
    left, singular, right = np.linalg.svd(positive, full_matrices=False)
    factors: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for rank in RANKS:
        root = np.sqrt(singular[:rank])
        embeddings = right[:rank].T * root[np.newaxis, :]
        head = left[:, :rank] * root[np.newaxis, :]
        factors[rank] = (embeddings.astype(np.float32), head.astype(np.float32))
    return factors


def score_validation(
    rows: list[dict[str, Any]],
    *,
    labels: tuple[str, ...],
    excluded: set[tuple[str, str]],
    factors: dict[int, tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    label_index = {label: index for index, label in enumerate(labels)}
    selected = [
        row
        for row in rows
        if row.get("text_sha256")
        and int(row["text_sha256"][:8], 16) % 10 == 0
        and (row["dataset"], row["uid"]) not in excluded
        and not row.get("read_error")
    ]
    n_rows = len(selected)
    y_true = np.zeros((n_rows, len(labels)), dtype=np.bool_)
    raw_scores = {
        f"rank{rank}": np.zeros((n_rows, len(labels)), dtype=np.float32)
        for rank in RANKS
    }
    datasets: list[str] = []
    uids: list[str] = []
    complete = np.zeros(n_rows, dtype=np.bool_)
    calibration = np.zeros(n_rows, dtype=np.bool_)
    max_embeddings = factors[max(RANKS)][0]
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
            features = document_features(text)
            full_bag = (
                max_embeddings[features].mean(axis=0)
                if len(features)
                else np.zeros(max(RANKS), dtype=np.float32)
            )
            for rank in RANKS:
                head = factors[rank][1]
                raw_scores[f"rank{rank}"][index] = head @ full_bag[:rank]
        except (OSError, ValueError):
            read_errors += 1
        if (index + 1) % 10_000 == 0:
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "phase": "embeddingbag_score",
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
        "raw_scores": raw_scores,
        "read_errors": read_errors,
    }


def fit_asl_calibration(
    raw_scores: np.ndarray,
    y_true: np.ndarray,
    observed: np.ndarray,
    *,
    epochs: int = 160,
    learning_rate: float = 0.03,
) -> tuple[np.ndarray, np.ndarray, float]:
    import torch

    torch.manual_seed(20260825)
    torch.set_num_threads(min(24, os.cpu_count() or 1))
    raw = torch.from_numpy(raw_scores.astype(np.float32))
    target = torch.from_numpy(y_true.astype(np.float32))
    mask = torch.from_numpy(observed.astype(np.float32))
    n_labels = raw.shape[1]
    calibration = torch.nn.Parameter(torch.eye(n_labels))
    bias = torch.nn.Parameter(torch.zeros(n_labels))
    optimizer = torch.optim.AdamW(
        [calibration, bias], lr=learning_rate, weight_decay=1e-4
    )
    identity = torch.eye(n_labels)
    final_loss = 0.0
    for _ in range(epochs):
        logits = raw @ calibration.T + bias
        probability = torch.sigmoid(logits)
        positive_loss = torch.log(probability.clamp_min(1e-8))
        if GAMMA_POS:
            positive_loss *= (1.0 - probability).pow(GAMMA_POS)
        clipped_negative = (probability - NEGATIVE_CLIP).clamp(0.0, 1.0)
        negative_loss = torch.log((1.0 - clipped_negative).clamp_min(1e-8))
        negative_loss *= clipped_negative.pow(GAMMA_NEG)
        per_entry = target * positive_loss + (1.0 - target) * negative_loss
        loss = -(per_entry * mask).sum() / mask.sum().clamp_min(1.0)
        loss += 1e-4 * (calibration - identity).pow(2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
    return (
        calibration.detach().numpy().astype(np.float32),
        bias.detach().numpy().astype(np.float32),
        final_loss,
    )


def trial_configs(
    n_trials: int = N_TRIALS, seed: int = 20260827
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    candidates: list[dict[str, Any]] = []
    for rank in RANKS:
        for priority_target_index in range(5, len(TARGETS)):
            for generic_target_index in range(10):
                for multiplier in (0.85, 0.95, 1.0, 1.05):
                    candidates.append(
                        {
                            "score_mode": f"rank{rank}",
                            "priority_target_index": priority_target_index,
                            "generic_target_index": generic_target_index,
                            "threshold_multiplier": multiplier,
                        }
                    )
    rng.shuffle(candidates)
    anchors = [
        {
            "score_mode": f"rank{rank}",
            "priority_target_index": priority_target,
            "generic_target_index": generic_target,
            "threshold_multiplier": 1.0,
        }
        for rank in RANKS
        for priority_target in (5, 8, 10, 12)
        for generic_target in (3, 5, 7, 9)
    ]
    ordered: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for config in anchors + candidates:
        key = tuple(config.values())
        if key not in seen:
            ordered.append(config)
            seen.add(key)
        if len(ordered) == n_trials:
            break
    return ordered


def _thresholds(
    config: dict[str, Any], bank: dict[str, np.ndarray], labels: tuple[str, ...]
) -> np.ndarray:
    from training.priority_data import PRIORITY_TAGS

    table = bank[config["score_mode"]]
    priority = set(PRIORITY_TAGS)
    values = np.empty(len(labels), dtype=np.float32)
    for index, label in enumerate(labels):
        target_index = (
            config["priority_target_index"]
            if label in priority
            else config["generic_target_index"]
        )
        values[index] = table[index, target_index]
    finite = np.isfinite(values)
    values[finite] *= config["threshold_multiplier"]
    return values


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
    counts = HashCounts.load(project / "cache" / "hash_sgd" / "counts.npz")
    source_weights = build_tfidf_weights(counts, idf_power=0.35)
    factors = factorize_ranks(source_weights)
    validation = score_validation(
        rows, labels=labels, excluded=excluded, factors=factors
    )
    calibration_mask = validation["calibration"]
    selection_mask = ~calibration_mask
    observed = validation["complete"][:, np.newaxis] | validation["y_true"]
    calibrated_scores: dict[str, np.ndarray] = {}
    calibrations: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    losses: dict[int, float] = {}
    for rank in RANKS:
        mode = f"rank{rank}"
        calibration, bias, loss = fit_asl_calibration(
            validation["raw_scores"][mode][calibration_mask],
            validation["y_true"][calibration_mask],
            observed[calibration_mask],
        )
        calibrations[rank] = (calibration, bias)
        losses[rank] = loss
        calibrated_scores[mode] = validation["raw_scores"][mode] @ calibration.T + bias
        print(json.dumps({"phase": "asl", "rank": rank, "loss": loss}), flush=True)
    selection_scores = {
        mode: scores[selection_mask] for mode, scores in calibrated_scores.items()
    }
    bank = threshold_bank(
        selection_scores,
        validation["y_true"][selection_mask],
        validation["datasets"][selection_mask],
        labels,
    )
    mlflow.set_tracking_uri((project / "mlruns").resolve().as_uri())
    mlflow.set_experiment("pii-priority-recall-v1")
    configs = trial_configs(n_trials=n_trials)
    output_dir = project / "tuning" / FAMILY
    output_dir.mkdir(parents=True, exist_ok=True)
    best: dict[str, Any] | None = None
    with (output_dir / "trials.jsonl").open("w", encoding="utf-8") as stream:
        for trial_number, config in enumerate(configs):
            thresholds = _thresholds(config, bank, labels)
            selected_scores = selection_scores[config["score_mode"]]
            selected_predicted = selected_scores >= thresholds
            metrics = fast_metrics(
                selected_predicted,
                validation["y_true"][selection_mask],
                validation["datasets"][selection_mask],
                validation["complete"][selection_mask],
                labels,
                quality["train"],
            )
            rank = int(config["score_mode"].removeprefix("rank"))
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
                mlflow.log_params(
                    {
                        **config,
                        "rank": rank,
                        "priority_target": TARGETS[config["priority_target_index"]],
                        "generic_target": TARGETS[config["generic_target_index"]],
                        "gamma_pos": GAMMA_POS,
                        "gamma_neg": GAMMA_NEG,
                        "negative_clip": NEGATIVE_CLIP,
                        "read_window_chars": READ_WINDOW,
                    }
                )
                mlflow.log_metrics(
                    {
                        **{
                            key: value
                            for key, value in metrics.items()
                            if key != "per_corpus"
                        },
                        "asl_calibration_loss": losses[rank],
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
                full_predicted = calibrated_scores[config["score_mode"]] >= thresholds
                best = {
                    **trial,
                    "thresholds": thresholds,
                    "predicted": full_predicted,
                    "rank": rank,
                }
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
    rank = best["rank"]
    calibration, bias = calibrations[rank]
    embeddings, head = factors[rank]
    model_dir = project / "models" / FAMILY
    model = LowRankEmbeddingBagModel(
        labels=labels,
        embeddings=embeddings,
        head=head,
        calibration=calibration,
        bias=bias,
        thresholds=best["thresholds"],
        read_window_chars=READ_WINDOW,
    )
    model.save(
        model_dir,
        metadata={
            "family": FAMILY,
            "trial": best["trial"],
            "mlflow_run_id": best["mlflow_run_id"],
            "rank": rank,
            "gamma_pos": GAMMA_POS,
            "gamma_neg": GAMMA_NEG,
            "negative_clip": NEGATIVE_CLIP,
            "asl_calibration_loss": losses[rank],
            "validation_metrics": best["metrics"],
        },
    )
    exact_arms = _exact_arms(
        validation,
        selection_mask,
        labels,
        best["predicted"],
        quality["train"],
    )
    exact_summary = aggregate_arms(exact_arms)
    resolved = {
        "family": FAMILY,
        "n_trials": n_trials,
        "best_trial": best["trial"],
        "best_mlflow_run_id": best["mlflow_run_id"],
        "best_config": best["config"],
        "rank": rank,
        "asl_losses": losses,
        "fast_metrics": best["metrics"],
        "exact_metrics": exact_summary,
        "calibration_rows": int(calibration_mask.sum()),
        "selection_rows": int(selection_mask.sum()),
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
