"""Frozen metrics for document-level sensitive-tag evaluation.

The lifecycle engine remains the independent general tagging scorer.  This
module adds the approved domain contract that the installed engine version
does not provide: catalogue-locked macro F2 and support-aware priority recall
gates.  Partial-label corpora contribute known-positive recall only; they never
manufacture false positives from missing annotations.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.priority_data import PRIORITY_TAGS

F2_BETA = 2.0
MIN_PRIORITY_SUPPORT = 30
PRIORITY_RECALL_GATE = 0.90
READ_DEPTHS = (1_000, 2_500, 10_000, 20_000)


@dataclass(frozen=True)
class EvaluationRow:
    dataset: str
    uid: str
    gold: frozenset[str]
    predicted: frozenset[str]
    label_complete: bool


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _fbeta(
    precision: float | None, recall: float | None, beta: float = F2_BETA
) -> float | None:
    if (precision is None and recall == 0.0) or (recall is None and precision == 0.0):
        return 0.0
    if precision is None or recall is None:
        return None
    beta2 = beta * beta
    denominator = beta2 * precision + recall
    return (1.0 + beta2) * precision * recall / denominator if denominator else 0.0


def _percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def _bootstrap_recall_ci(
    rows: Sequence[EvaluationRow],
    tag: str,
    *,
    n_resamples: int,
    confidence: float,
    seed: int,
) -> tuple[float | None, float | None]:
    import numpy as np

    n_rows = len(rows)
    true_positive = sum(tag in row.gold and tag in row.predicted for row in rows)
    false_negative = sum(tag in row.gold and tag not in row.predicted for row in rows)
    other = n_rows - true_positive - false_negative
    if true_positive + false_negative == 0:
        return None, None
    # A document bootstrap collapsed into TP/FN/other categories is exactly a
    # multinomial draw, avoiding materializing n_resamples x n_rows indices.
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(
        n_rows,
        np.asarray([true_positive, false_negative, other], dtype=np.float64) / n_rows,
        size=n_resamples,
    )
    support = draws[:, 0] + draws[:, 1]
    samples = (draws[support > 0, 0] / support[support > 0]).tolist()
    alpha = (1.0 - confidence) / 2.0
    return _percentile(samples, alpha), _percentile(samples, 1.0 - alpha)


def _engine_metrics(
    rows: Sequence[EvaluationRow], *, bootstrap: bool, n_resamples: int
) -> dict[str, Any]:
    """Call the canonical engine without making this module depend on its internals."""
    if not rows:
        return {"assessable": False, "reason": "no complete-label rows"}
    from model_pipeline import evaluate

    result = evaluate(
        "tagging",
        [sorted(row.gold) for row in rows],
        [sorted(row.predicted) for row in rows],
        primary_metric="f1_micro",
        bootstrap=bootstrap,
        n_resamples=n_resamples,
        seed=1729,
        dataset={"label_completeness": "complete"},
    )
    return {"assessable": True, "result": result.to_dict()}


def evaluate_corpus(
    rows: Sequence[EvaluationRow],
    *,
    catalogue: Sequence[str],
    bootstrap: bool = False,
    n_resamples: int = 1_000,
    confidence: float = 0.95,
    min_support: int = MIN_PRIORITY_SUPPORT,
    recall_gate: float = PRIORITY_RECALL_GATE,
) -> dict[str, Any]:
    """Evaluate one model x corpus arm under the immutable metric contract."""
    if not rows:
        raise ValueError("cannot evaluate an empty corpus")
    datasets = {row.dataset for row in rows}
    if len(datasets) != 1:
        raise ValueError(f"one arm must contain one dataset, got {sorted(datasets)}")
    uids = [row.uid for row in rows]
    if len(uids) != len(set(uids)):
        raise ValueError("duplicate uid in evaluation arm")
    locked_catalogue = tuple(dict.fromkeys(catalogue))
    complete_rows = [row for row in rows if row.label_complete]
    fully_complete = len(complete_rows) == len(rows)

    per_tag: dict[str, dict[str, Any]] = {}
    priority: dict[str, dict[str, Any]] = {}
    total_tp = total_fp = total_fn = 0
    for tag in locked_catalogue:
        support = sum(tag in row.gold for row in rows)
        predicted = sum(tag in row.predicted for row in rows)
        true_positive = sum(tag in row.gold and tag in row.predicted for row in rows)
        false_negative = support - true_positive
        false_positive = predicted - true_positive if fully_complete else None
        recall = _ratio(true_positive, support)
        precision = _ratio(true_positive, predicted) if fully_complete else None
        f2 = _fbeta(precision, recall)
        per_tag[tag] = {
            "support": support,
            "predicted": predicted,
            "tp": true_positive,
            "fp": false_positive,
            "fn": false_negative,
            "precision": precision,
            "recall": recall,
            "f2": f2,
        }
        if fully_complete:
            total_tp += true_positive
            total_fp += int(false_positive or 0)
            total_fn += false_negative

    for offset, tag in enumerate(PRIORITY_TAGS):
        metric = per_tag.get(tag)
        if metric is None:
            metric = {
                "support": sum(tag in row.gold for row in rows),
                "predicted": sum(tag in row.predicted for row in rows),
                "tp": sum(tag in row.gold and tag in row.predicted for row in rows),
            }
            metric["fn"] = metric["support"] - metric["tp"]
            metric["recall"] = _ratio(metric["tp"], metric["support"])
        support = int(metric["support"])
        recall = metric["recall"]
        ci_low = ci_high = None
        if bootstrap and support >= min_support:
            ci_low, ci_high = _bootstrap_recall_ci(
                rows,
                tag,
                n_resamples=n_resamples,
                confidence=confidence,
                seed=97_531 + offset,
            )
        if support < min_support:
            status = "NOT_ASSESSABLE"
        elif recall is None or recall < recall_gate:
            status = "FAIL"
        elif not bootstrap:
            status = "POINT_PASS_UNVERIFIED"
        elif ci_low is not None and ci_low >= recall_gate:
            status = "PASS"
        else:
            status = "INCONCLUSIVE"
        priority[tag] = {
            "support": support,
            "tp": int(metric["tp"]),
            "fn": int(metric["fn"]),
            "recall": recall,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "confidence": confidence if bootstrap else None,
            "n_resamples": n_resamples if bootstrap else None,
            "status": status,
        }

    if fully_complete:
        # The denominator is the frozen corpus catalogue; absent/undetected
        # labels score zero instead of silently disappearing from macro F2.
        macro_f2 = (
            sum(float(per_tag[tag]["f2"] or 0.0) for tag in locked_catalogue)
            / len(locked_catalogue)
            if locked_catalogue
            else None
        )
        micro_precision = _ratio(total_tp, total_tp + total_fp)
        micro_recall = _ratio(total_tp, total_tp + total_fn)
        micro_f1 = _fbeta(micro_precision, micro_recall, beta=1.0)
    else:
        macro_f2 = micro_f1 = micro_precision = micro_recall = None

    measurable = [
        entry for entry in priority.values() if entry["support"] >= min_support
    ]
    recalls = [
        float(entry["recall"]) for entry in measurable if entry["recall"] is not None
    ]
    return {
        "dataset": next(iter(datasets)),
        "n_rows": len(rows),
        "label_complete": fully_complete,
        "catalogue": list(locked_catalogue),
        "macro_f2": macro_f2,
        "micro_f1": micro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "priority": priority,
        "priority_summary": {
            "measurable_tags": len(measurable),
            "point_passes": sum(
                entry["recall"] is not None and entry["recall"] >= recall_gate
                for entry in measurable
            ),
            "conclusive_passes": sum(entry["status"] == "PASS" for entry in measurable),
            "failures": sum(entry["status"] == "FAIL" for entry in measurable),
            "inconclusive": sum(
                entry["status"] == "INCONCLUSIVE" for entry in measurable
            ),
            "worst_recall": min(recalls) if recalls else None,
        },
        "per_tag": per_tag,
        "engine": _engine_metrics(
            complete_rows, bootstrap=bootstrap, n_resamples=n_resamples
        ),
        "metric_contract": {
            "beta": F2_BETA,
            "min_priority_support": min_support,
            "priority_recall_gate": recall_gate,
            "confidence": confidence,
            "partial_labels": "known-positive recall only",
        },
    }


def aggregate_arms(arms: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Equal-weight complete corpora; micro F1 breaks macro-F2 ties."""
    f2_values = [
        float(arm["macro_f2"]) for arm in arms if arm.get("macro_f2") is not None
    ]
    f1_values = [
        float(arm["micro_f1"]) for arm in arms if arm.get("micro_f1") is not None
    ]
    measurable = [
        entry
        for arm in arms
        for entry in arm["priority"].values()
        if entry["support"] >= MIN_PRIORITY_SUPPORT
    ]
    recalls = [
        float(entry["recall"]) for entry in measurable if entry["recall"] is not None
    ]
    return {
        "n_arms": len(arms),
        "n_complete_arms": len(f2_values),
        "equal_corpus_macro_f2": sum(f2_values) / len(f2_values) if f2_values else None,
        "equal_corpus_micro_f1": sum(f1_values) / len(f1_values) if f1_values else None,
        "measurable_priority_gates": len(measurable),
        "priority_point_passes": sum(
            entry["recall"] >= PRIORITY_RECALL_GATE for entry in measurable
        ),
        "priority_conclusive_passes": sum(
            entry["status"] == "PASS" for entry in measurable
        ),
        "priority_failures": sum(entry["status"] == "FAIL" for entry in measurable),
        "worst_priority_recall": min(recalls) if recalls else None,
    }


def freeze_catalogue(data_quality_path: Path, output_path: Path) -> dict[str, Any]:
    """Persist the observed eval catalogues before model fitting begins."""
    payload = data_quality_path.read_bytes()
    data_quality = json.loads(payload)
    corpora = {}
    union: set[str] = set()
    for dataset, quality in sorted(data_quality["eval"].items()):
        tags = sorted(quality["tag_counts"])
        union.update(tags)
        corpora[dataset] = {
            "catalogue": tags,
            "label_complete": quality["complete_label_rows"] == quality["n_rows"],
            "n_rows": quality["n_rows"],
        }
    frozen = {
        "version": 1,
        "source_data_quality_sha256": hashlib.sha256(payload).hexdigest(),
        "full_catalogue": sorted(union),
        "priority_tags": list(PRIORITY_TAGS),
        "metric_contract": {
            "beta": F2_BETA,
            "min_priority_support": MIN_PRIORITY_SUPPORT,
            "priority_recall_gate": PRIORITY_RECALL_GATE,
            "read_depths": list(READ_DEPTHS),
            "selection": [
                "priority_gates",
                "equal_corpus_macro_f2",
                "equal_corpus_micro_f1",
                "one_core_p95_ms",
            ],
        },
        "corpora": corpora,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return frozen


def rows_from_predictions(
    index_rows: Iterable[Mapping[str, Any]],
    predictions: Mapping[tuple[str, str], Iterable[str]],
) -> dict[str, list[EvaluationRow]]:
    grouped: dict[str, list[EvaluationRow]] = defaultdict(list)
    for row in index_rows:
        dataset, uid = str(row["dataset"]), str(row["uid"])
        grouped[dataset].append(
            EvaluationRow(
                dataset=dataset,
                uid=uid,
                gold=frozenset(map(str, row.get("labels", []))),
                predicted=frozenset(map(str, predictions.get((dataset, uid), ()))),
                label_complete=bool(row["label_complete"]),
            )
        )
    return dict(grouped)
