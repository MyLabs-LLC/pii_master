"""Final independent bootstrap evaluation for the selected champion."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from training.priority_eval import (
    aggregate_arms,
    evaluate_corpus,
    rows_from_predictions,
)
from training.tune_priority_hash import _load_jsonl, _save_json
from training.priority_eval import record_artifacts as _record_artifacts

N_RESAMPLES = 1_000
CONFIDENCE = 0.95


def cluster_bootstrap_ci(
    values: list[float],
    *,
    n_resamples: int = N_RESAMPLES,
    confidence: float = CONFIDENCE,
    seed: int = 20260825,
) -> dict[str, float]:
    if not values:
        raise ValueError("cluster bootstrap needs at least one corpus")
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=np.float64)
    sampled = rng.choice(array, size=(n_resamples, len(array)), replace=True).mean(
        axis=1
    )
    alpha = (1.0 - confidence) / 2.0
    return {
        "value": float(array.mean()),
        "ci_low": float(np.quantile(sampled, alpha)),
        "ci_high": float(np.quantile(sampled, 1.0 - alpha)),
        "confidence": confidence,
        "n_resamples": n_resamples,
        "unit": "holdout_corpus",
    }


def _load_predictions(path: Path) -> dict[tuple[str, str], list[str]]:
    predictions = {}
    for row in _load_jsonl(path):
        predictions[(row["dataset"], row["uid"])] = list(map(str, row["labels"]))
    return predictions


def run(project: Path, *, family: str) -> dict[str, Any]:
    import mlflow

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    index_rows = _load_jsonl(project / "data" / "eval_index.jsonl")
    predictions = _load_predictions(
        project / "evaluations" / family / "predictions.jsonl"
    )
    frozen = json.loads(
        (project / "data" / "evaluation_catalogue.json").read_text(encoding="utf-8")
    )
    grouped = rows_from_predictions(index_rows, predictions)
    output_dir = project / "evaluations" / family / "bootstrap"
    arms = []
    mlflow.set_tracking_uri((project / "mlruns").resolve().as_uri())
    mlflow.set_experiment("pii-priority-recall-v1")
    for dataset in sorted(grouped):
        result = evaluate_corpus(
            grouped[dataset],
            catalogue=frozen["corpora"][dataset]["catalogue"],
            bootstrap=True,
            n_resamples=N_RESAMPLES,
            confidence=CONFIDENCE,
        )
        result_path = output_dir / f"{dataset}.json"
        _save_json(result_path, result)
        with mlflow.start_run(run_name=f"{family}__{dataset}__bootstrap"):
            mlflow.set_tags(
                {
                    "model": family,
                    "dataset": dataset,
                    "split": "eval",
                    "phase": "final_bootstrap",
                    "arm_key": f"{family}::{dataset}",
                }
            )
            mlflow.log_params(
                {
                    "bootstrap_resamples": N_RESAMPLES,
                    "confidence": CONFIDENCE,
                    "read_window_chars": 1_000,
                }
            )
            metrics = {
                "priority_measurable": result["priority_summary"]["measurable_tags"],
                "priority_conclusive_passes": result["priority_summary"][
                    "conclusive_passes"
                ],
                "priority_failures": result["priority_summary"]["failures"],
                "priority_inconclusive": result["priority_summary"]["inconclusive"],
                "priority_worst_recall": result["priority_summary"]["worst_recall"]
                or 0.0,
            }
            if result["macro_f2"] is not None:
                metrics["macro_f2"] = result["macro_f2"]
                metrics["micro_f1"] = result["micro_f1"]
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(result_path), artifact_path="final_bootstrap")
        arms.append(result)
    aggregate = aggregate_arms(arms)
    complete = [arm for arm in arms if arm["macro_f2"] is not None]
    aggregate["equal_corpus_macro_f2_ci"] = cluster_bootstrap_ci(
        [float(arm["macro_f2"]) for arm in complete], seed=20260825
    )
    aggregate["equal_corpus_micro_f1_ci"] = cluster_bootstrap_ci(
        [float(arm["micro_f1"]) for arm in complete], seed=20260826
    )
    measurable = [
        entry
        for arm in arms
        for entry in arm["priority"].values()
        if entry["support"] >= 30
    ]
    aggregate["priority_conclusive_passes"] = sum(
        entry["status"] == "PASS" for entry in measurable
    )
    aggregate["priority_inconclusive"] = sum(
        entry["status"] == "INCONCLUSIVE" for entry in measurable
    )
    aggregate["priority_gate_pass"] = aggregate["priority_conclusive_passes"] == len(
        measurable
    )
    aggregate["macro_f2_target"] = 0.90
    aggregate["macro_f2_target_pass"] = aggregate["equal_corpus_macro_f2"] >= 0.90
    summary = {
        "family": family,
        "bootstrap_resamples": N_RESAMPLES,
        "confidence": CONFIDENCE,
        "aggregate": aggregate,
    }
    _save_json(output_dir / "summary.json", summary)
    run_path = project / "run.json"
    run_record = json.loads(run_path.read_text(encoding="utf-8"))
    run_record.setdefault("run_summary", {}).setdefault(family, {})[
        "final_bootstrap"
    ] = aggregate
    _record_artifacts(
        run_record,
        {f"final_bootstrap::{family}": f"evaluations/{family}/bootstrap/summary.json"},
    )
    _save_json(run_path, run_record)

    # Standing taxonomy diagnostic, written beside the gate numbers on every
    # run. NOT a gate: the contract is the current taxonomy. It records what
    # the same predictions score once the name and street tags are folded, so
    # the share of the score that is reproducing this corpus's labelling
    # convention stays visible instead of resting in one report. See
    # reports/26-08-25_taxonomy-collapse-scope.md.
    from training.simulate_taxonomy_collapse import emit_diagnostic, record_in_run

    diagnostic = emit_diagnostic(project, family, bootstrap=False)
    record_in_run(project, diagnostic)
    summary["collapsed_taxonomy"] = diagnostic["delta"]
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--family", default="champion_1k")
    args = parser.parse_args()
    result = run(args.project.resolve(), family=args.family)
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
