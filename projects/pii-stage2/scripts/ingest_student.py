"""Turn training/eval_student.py --json-out into span-gate arms + MLflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlflow  # noqa: E402
from gate_metrics import micro_prf, span_gate  # noqa: E402
from runjson import upsert_arm  # noqa: E402

from model_pipeline import tracking  # noqa: E402
from model_pipeline.result import EvaluationResult, MetricValue  # noqa: E402

CATALOGUE = (
    "ACCOUNT_NUMBER", "CREDIT_CARD", "DATE_DOB", "EMAIL", "HEALTH_PLAN_ID",
    "IP_ADDRESS", "MRN", "PHONE_US", "SSN", "URL", "US_DRIVER_LICENSE",
)


def _result(metrics: dict, *, model: str, dataset: str, n: int) -> EvaluationResult:
    res = EvaluationResult(
        task_type="tagging",
        primary_metric="macro_f2",
        n_samples=n,
        dataset={"name": dataset},
        params={"model": model},
    )
    for name, value in metrics.items():
        if value is None:
            continue
        res.add(MetricValue(
            name=name, value=float(value),
            greater_is_better=not str(name).startswith("n_tags"),
        ))
    return res


def ingest(path: Path, *, model: str, dataset: str, system: str) -> dict:
    payload = json.loads(path.read_text())
    exact = payload["mapped"][system]
    gate = span_gate(exact, catalogue=CATALOGUE)
    p, r, f = micro_prf(gate)
    n = payload.get("documents", 0)
    metrics = {
        "macro_f2": gate["macro_f2"],
        "severity_recall_min": gate["severity_recall_min"],
        "severity_recall_mean": gate["severity_recall_mean"],
        "f2_min": gate["f2_min"],
        "f2_median": gate["f2_median"],
        "n_tags_f2_zero": gate["n_tags_f2_zero"],
        "span_precision_micro": p,
        "span_recall_micro": r,
        "span_f1_micro": f,
    }
    native = payload.get("mapped_micro", {}).get(system, {})
    res = _result(metrics, model=model, dataset=dataset, n=n)
    res.tags["per_tag_f2_worst_first"] = gate["per_tag"]
    res.tags["severity_tags_measured"] = gate["severity_tags_measured"]
    out = {
        "model": model,
        "dataset": dataset,
        "system": system,
        "n_samples": n,
        "gate": gate,
        "mapped_micro": native,
        "evaluation": res.to_dict(),
    }
    dest = path.with_name(path.stem + f"_{system}_gate.json")
    dest.write_text(json.dumps(out, indent=2, default=str) + "\n")
    with tracking.evaluation_run(
        f"{model}-{dataset}-{system}", experiment="pii-stage2",
        tags={"model": model, "dataset": dataset, "system": system},
    ):
        tracking.log_evaluation(res)
        mlflow.log_artifact(str(dest), artifact_path="evaluation")
    upsert_arm({
        "model": f"{model}/{system}",
        "dataset": dataset,
        "n_samples": n,
        "verdict": "challenger" if "fusion_checksum" in system else "measured",
        "metrics": {k: {"value": v} for k, v in metrics.items() if v is not None},
        "evaluation": res.to_dict(),
    })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("json_out")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--systems", default="rules,student,fusion_checksum_first")
    args = ap.parse_args()
    path = Path(args.json_out)
    for system in args.systems.split(","):
        system = system.strip()
        out = ingest(path, model=args.model, dataset=args.dataset, system=system)
        print(json.dumps({
            "system": system,
            "macro_f2": out["gate"]["macro_f2"],
            "severity_recall_min": out["gate"]["severity_recall_min"],
            "f2_min": out["gate"]["f2_min"],
        }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
