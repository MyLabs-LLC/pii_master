"""Generate holdout predictions exclusively through the packaged entry point."""

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
_DATA_SPEC = importlib.util.spec_from_file_location(
    "bundle_verification_document_reader",
    PROJECT.parents[1] / "training" / "priority_data.py",
)
assert _DATA_SPEC is not None and _DATA_SPEC.loader is not None
_DATA_MODULE = importlib.util.module_from_spec(_DATA_SPEC)
sys.modules[_DATA_SPEC.name] = _DATA_MODULE
_DATA_SPEC.loader.exec_module(_DATA_MODULE)
read_document = _DATA_MODULE.read_document

_TAGGER: Any = None


def _load_worker(bundle: str) -> None:
    global _TAGGER
    import sys

    sys.path.insert(0, bundle)
    from tagger import Tagger

    _TAGGER = Tagger()


def _predict_one(payload: tuple[str, str, str]) -> tuple[str, str, list[str], str]:
    dataset, uid, raw_path = payload
    assert _TAGGER is not None
    try:
        text = read_document(Path(raw_path), limit=_TAGGER.read_window_chars)
        return dataset, uid, _TAGGER.predict(text), ""
    except Exception as exc:  # noqa: BLE001 - preserve per-document evidence
        return dataset, uid, [], f"{type(exc).__name__}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in (PROJECT / "data" / "eval_index.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    tasks = [(row["dataset"], row["uid"], row["path"]) for row in rows]
    predictions: dict[tuple[str, str], list[str]] = {}
    errors: dict[str, list[str]] = defaultdict(list)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
        initializer=_load_worker,
        initargs=(str(args.bundle.resolve()),),
    ) as executor:
        for dataset, uid, labels, error in executor.map(
            _predict_one, tasks, chunksize=64
        ):
            predictions[(dataset, uid)] = labels
            if error:
                errors[dataset].append(f"{uid}: {error}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as stream:
        for (dataset, uid), labels in sorted(predictions.items()):
            stream.write(
                json.dumps(
                    {"dataset": dataset, "uid": uid, "labels": labels},
                    sort_keys=True,
                )
                + "\n"
            )
    print(
        json.dumps(
            {
                "bundle": str(args.bundle.resolve()),
                "predictions": len(predictions),
                "read_errors": {key: len(value) for key, value in errors.items()},
                "out": str(args.out),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
