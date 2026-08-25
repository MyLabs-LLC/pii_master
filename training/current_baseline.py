"""Evaluate the currently shipped rules model on every held-out corpus."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from pii_master.classify import scan_text
from training.priority_data import READ_WINDOW, read_document
from training.priority_eval import (
    aggregate_arms,
    evaluate_corpus,
    rows_from_predictions,
)

MODEL_NAME = "current_rules"
ENTITY_TAGS = {
    "EMAIL": ("sensitive_pii_email",),
    "PHONE_US": ("sensitive_pii_phone_number",),
    "SSN": ("sensitive_pii_social_security_number",),
    "CREDIT_CARD": ("sensitive_pci_credit_card", "sensitive_pci_credit_card_number"),
    "DATE_DOB": ("sensitive_pii_date_of_birth_dob",),
    "MRN": ("sensitive_phi_medical_record_number_mrn",),
    "ACCOUNT_NUMBER": ("sensitive_pci_bank_account_number",),
    "HEALTH_PLAN_ID": ("sensitive_phi_health_plan_beneficiary_number",),
    "US_DRIVER_LICENSE": ("sensitive_pii_driver_s_license_number",),
    "PERSON_NAME": ("sensitive_pii_full_name",),
    "ADDRESS": ("sensitive_pii_address", "sensitive_pii_street_number_and_name"),
    "GEO_COORDINATE": ("sensitive_pii_geolocation",),
    "BANK_ROUTING": ("sensitive_pci_routing_number",),
    "SWIFT_BIC": ("sensitive_pci_swift_code",),
    "VEHICLE_ID": ("sensitive_pii_vehicle_identification_number_vin",),
    "MAC_ADDRESS": ("sensitive_pii_mac_address",),
    "TAX_ID": ("sensitive_pci_individual_taxpayer_identification_number_itin",),
    "USER_ID": ("sensitive_pii_username",),
}


def entity_tags(entity_type: str, text: str = "") -> set[str]:
    if entity_type == "IP_ADDRESS":
        return {"sensitive_pii_ipv6" if ":" in text else "sensitive_pii_ipv4"}
    return set(ENTITY_TAGS.get(entity_type, ()))


def _scan_one(payload: tuple[str, str, str]) -> tuple[str, str, list[str], str]:
    dataset, uid, raw_path = payload
    try:
        text = read_document(Path(raw_path), limit=READ_WINDOW)
        report = scan_text(text)
        tags: set[str] = set()
        for entity in report.entities:
            tags.update(entity_tags(entity.type.value, entity.text))
        return dataset, uid, sorted(tags), ""
    except Exception as exc:  # noqa: BLE001 - preserve per-document failures as evidence
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


def _sync_run_record(
    project: Path, quality: dict[str, Any], arms: list[dict[str, Any]]
) -> None:
    run_path = project / "run.json"
    record = json.loads(run_path.read_text(encoding="utf-8"))
    record["data_quality"] = [
        {"dataset": dataset, **details}
        for split in ("train", "eval")
        for dataset, details in sorted(quality[split].items())
    ]
    existing = [arm for arm in record.get("arms", []) if arm.get("model") != MODEL_NAME]
    record["arms"] = existing + arms
    artifacts = record.setdefault("artifacts", [])
    for relative in (
        "data/data_quality.json",
        "data/evaluation_catalogue.json",
        f"evaluations/{MODEL_NAME}/summary.json",
        f"evaluations/{MODEL_NAME}/predictions.jsonl",
    ):
        if relative not in artifacts:
            artifacts.append(relative)
    _save_json(run_path, record)


def run(project: Path, *, workers: int) -> dict[str, Any]:
    import mlflow

    data_dir = project / "data"
    index_rows = _load_jsonl(data_dir / "eval_index.jsonl")
    quality = json.loads((data_dir / "data_quality.json").read_text(encoding="utf-8"))
    frozen = json.loads(
        (data_dir / "evaluation_catalogue.json").read_text(encoding="utf-8")
    )
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in index_rows:
        by_dataset[row["dataset"]].append(row)

    output_dir = project / "evaluations" / MODEL_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_map: dict[tuple[str, str], list[str]] = {}
    errors: dict[str, list[str]] = defaultdict(list)
    speed: dict[str, dict[str, float]] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for dataset in sorted(by_dataset):
            corpus = by_dataset[dataset]
            tasks = [(dataset, row["uid"], row["path"]) for row in corpus]
            started = time.perf_counter()
            results = executor.map(_scan_one, tasks, chunksize=64)
            for result_dataset, uid, tags, error in results:
                prediction_map[(result_dataset, uid)] = tags
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

    grouped = rows_from_predictions(index_rows, prediction_map)
    tracking_dir = project / "mlruns"
    mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())
    mlflow.set_experiment("pii-priority-recall-v1")
    arms: list[dict[str, Any]] = []
    arm_results: list[dict[str, Any]] = []
    for dataset in sorted(grouped):
        catalogue = frozen["corpora"][dataset]["catalogue"]
        result = evaluate_corpus(grouped[dataset], catalogue=catalogue, bootstrap=False)
        result["speed"] = speed[dataset]
        result["read_errors"] = errors[dataset][:100]
        result_path = output_dir / f"{dataset}.json"
        _save_json(result_path, result)
        with mlflow.start_run(run_name=f"{MODEL_NAME}__{dataset}") as active:
            mlflow.set_tags(
                {
                    "model": MODEL_NAME,
                    "dataset": dataset,
                    "split": "eval",
                    "phase": "baseline",
                    "arm_key": f"{MODEL_NAME}::{dataset}",
                }
            )
            mlflow.log_params(
                {
                    "read_window_chars": READ_WINDOW,
                    "workers": workers,
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
        arm = {
            "model": MODEL_NAME,
            "dataset": dataset,
            "mlflow_run_id": run_id,
            "metrics": {
                "macro_f2": result["macro_f2"],
                "micro_f1": result["micro_f1"],
                "worst_priority_recall": result["priority_summary"]["worst_recall"],
                "priority_point_passes": result["priority_summary"]["point_passes"],
                "priority_measurable": result["priority_summary"]["measurable_tags"],
                "docs_per_s": speed[dataset]["docs_per_s"],
            },
            "artifact": str(result_path.relative_to(project)),
        }
        arms.append(arm)
        arm_results.append(result)

    predictions_path = output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as stream:
        for (dataset, uid), tags in sorted(prediction_map.items()):
            stream.write(
                json.dumps(
                    {"dataset": dataset, "uid": uid, "labels": tags}, sort_keys=True
                )
                + "\n"
            )
    summary = {
        "model": MODEL_NAME,
        "aggregate": aggregate_arms(arm_results),
        "arms": arms,
        "read_window_chars": READ_WINDOW,
    }
    _save_json(output_dir / "summary.json", summary)
    _sync_run_record(project, quality, arms)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    args = parser.parse_args()
    summary = run(args.project.resolve(), workers=args.workers)
    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
