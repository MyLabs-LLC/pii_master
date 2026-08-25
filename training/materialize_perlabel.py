"""Materialize the per-label gate-boundary threshold candidate.

Copies the champion fusion model and replaces **only** the recall component's
per-label ``thresholds`` entries for the priority tags. Weights, features, read
window, strategies and the generic head are byte-identical to the champion, so
the arm differs from it in exactly one config surface (``per-tag thresholds``,
declared editable in the project spec) and its latency is unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np

from training.priority_data import PRIORITY_TAGS
from training.priority_fusion import FusionPriorityModel
from training.priority_hash import load_priority_model


def run(project: Path, *, source: str, family: str, selection: str) -> dict:
    import mlflow

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    model = load_priority_model(project / "models" / source)
    if not isinstance(model, FusionPriorityModel):
        raise TypeError(f"{source} is not a fusion model")

    chosen = json.loads(
        (project / "tuning" / selection / "selected_thresholds.json").read_text(
            encoding="utf-8"
        )
    )

    recall = model.components["recall"]
    labels = list(recall.labels)
    thresholds = np.array(recall.thresholds, dtype=np.float32, copy=True)

    changed: dict[str, dict] = {}
    for tag in PRIORITY_TAGS:
        if tag not in labels:
            continue
        pick = chosen.get(tag) or {}
        if pick.get("threshold") is None:
            continue
        index = labels.index(tag)
        before = float(thresholds[index])
        after = float(pick["threshold"])
        thresholds[index] = after
        changed[tag] = {
            "threshold_before": before,
            "threshold_after": after,
            "target": pick.get("target"),
            "held_in_precision": pick.get("held_in_precision"),
            "held_in_recall": pick.get("held_in_recall"),
            "worst_corpus_ci_lower": pick.get("worst_corpus_ci_lower"),
        }

    patched = replace(recall, thresholds=thresholds)
    candidate = FusionPriorityModel(
        labels=model.labels,
        components={**model.components, "recall": patched},
        strategies=model.strategies,
        read_window_override=model.read_window_override,
    )
    model_dir = project / "models" / family
    candidate.save(
        model_dir,
        metadata={
            "family": family,
            "source_family": source,
            "selection": "per-label gate-boundary threshold (ci_lower >= 0.90 + margin on held-in)",
            "changed_tags": changed,
        },
    )

    mlflow.set_tracking_uri((project / "mlruns").resolve().as_uri())
    mlflow.set_experiment("pii-priority-recall-v1")
    with mlflow.start_run(run_name=f"materialize__{family}"):
        mlflow.set_tags({"model": family, "phase": "materialize", "source": source})
        mlflow.log_params(
            {"source_family": source, "n_tags_retuned": len(changed)}
        )
    summary = {"family": family, "model_dir": str(model_dir), "changed": changed}
    (project / "tuning" / selection / "materialized.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=Path, default=Path("projects/pii-priority-recall-v1"))
    ap.add_argument("--source", default="champion_1k")
    ap.add_argument("--family", default="perlabel_1k")
    ap.add_argument("--selection", default="perlabel_1k")
    args = ap.parse_args()
    summary = run(
        args.project, source=args.source, family=args.family, selection=args.selection
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
