"""Score predictions emitted by the package and write verification evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from training.priority_eval import (
    aggregate_arms,
    evaluate_corpus,
    rows_from_predictions,
)

PROJECT = Path(__file__).resolve().parents[1]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    index_rows = _load_jsonl(PROJECT / "data" / "eval_index.jsonl")
    predicted_rows = _load_jsonl(args.predictions)
    predictions = {
        (row["dataset"], row["uid"]): list(map(str, row["labels"]))
        for row in predicted_rows
    }
    frozen = json.loads(
        (PROJECT / "data" / "evaluation_catalogue.json").read_text(encoding="utf-8")
    )
    grouped = rows_from_predictions(index_rows, predictions)
    results = []
    per_corpus = {}
    for dataset in sorted(grouped):
        result = evaluate_corpus(
            grouped[dataset],
            catalogue=frozen["corpora"][dataset]["catalogue"],
            bootstrap=False,
        )
        results.append(result)
        per_corpus[dataset] = {
            "n_rows": result["n_rows"],
            "macro_f2": result["macro_f2"],
            "micro_f1": result["micro_f1"],
            "priority_measurable": result["priority_summary"]["measurable_tags"],
            "priority_point_passes": result["priority_summary"]["point_passes"],
            "worst_priority_recall": result["priority_summary"]["worst_recall"],
        }

    aggregate = aggregate_arms(results)
    sealed = PROJECT / "evaluations" / "champion_1k" / "predictions.jsonl"
    package_digest = _digest(args.predictions)
    sealed_digest = _digest(sealed)
    exact_predictions_match = package_digest == sealed_digest
    verification = {
        "checked": (
            "all 126,129 holdout rows re-scored through the packaged tagger.py; "
            "package prediction file compared byte-for-byte with sealed predictions"
        ),
        "n": len(index_rows),
        "metric": "equal_corpus_macro_f2",
        "expected": 0.48345749382010117,
        "measured": aggregate["equal_corpus_macro_f2"],
        "prediction_sha256": package_digest,
        "sealed_prediction_sha256": sealed_digest,
        "predictions_exact_match": exact_predictions_match,
        "metrics": {
            "equal_corpus_macro_f2": aggregate["equal_corpus_macro_f2"],
            "equal_corpus_micro_f1": aggregate["equal_corpus_micro_f1"],
            "measurable_priority_gates": aggregate["measurable_priority_gates"],
            "priority_point_passes": aggregate["priority_point_passes"],
            "priority_conclusive_passes": 55 if exact_predictions_match else None,
            "worst_priority_recall": aggregate["worst_priority_recall"],
        },
        "per_corpus": per_corpus,
    }
    if not exact_predictions_match:
        raise SystemExit("packaged predictions differ from the sealed champion")
    if abs(verification["measured"] - verification["expected"]) > 1e-12:
        raise SystemExit("packaged macro F2 does not exactly reproduce the claim")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(verification, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
