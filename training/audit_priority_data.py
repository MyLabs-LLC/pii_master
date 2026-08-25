"""Build the frozen dataset index and the required data-quality/leakage record."""

from __future__ import annotations

import argparse
import json
import subprocess
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from priority_data import (
    PRIORITY_TAGS,
    CorpusRow,
    iter_corpus,
    list_dataset_dirs,
    normalized_text_digest,
    read_document,
)


def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def inspect_row(row: CorpusRow) -> tuple[CorpusRow, str, str]:
    path = Path(row.path)
    if not path.is_file():
        return row, "", "missing"
    try:
        text = read_document(path)
    except (
        OSError,
        RuntimeError,
        subprocess.SubprocessError,
        ValueError,
        zipfile.BadZipFile,
    ) as exc:  # recorded per corpus; one bad binary cannot hide the rest
        return row, "", f"{type(exc).__name__}: {exc}"
    if not text:
        return row, "", "empty"
    return row, normalized_text_digest(text), ""


def audit_split(root: Path, split: str, out: Path, workers: int) -> tuple[dict, set[str], Counter[str]]:
    split_hashes: set[str] = set()
    all_tag_counts: Counter[str] = Counter()
    quality: dict[str, Any] = {}
    index_path = out / f"{split}_index.jsonl"
    with index_path.open("w", encoding="utf-8") as index_stream:
        for dataset_dir in list_dataset_dirs(root):
            print(f"audit {split}: {dataset_dir.name}", flush=True)
            rows = list(iter_corpus(dataset_dir))
            tag_counts: Counter[str] = Counter()
            native_counts: Counter[str] = Counter()
            provenance_counts: Counter[str] = Counter()
            errors: Counter[str] = Counter()
            local_hashes: set[str] = set()
            duplicates = 0
            complete = 0
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for row, digest, error in pool.map(inspect_row, rows, chunksize=64):
                    tag_counts.update(row.labels)
                    native_counts.update(row.native_labels)
                    provenance_counts[row.provenance] += 1
                    complete += int(row.label_complete)
                    if error:
                        errors[error] += 1
                    if digest:
                        if digest in local_hashes:
                            duplicates += 1
                        local_hashes.add(digest)
                        split_hashes.add(digest)
                    payload = row.to_dict()
                    payload["text_sha256"] = digest
                    payload["read_error"] = error
                    index_stream.write(json.dumps(payload, sort_keys=True) + "\n")
            all_tag_counts.update(tag_counts)
            quality[dataset_dir.name] = {
                "split": split,
                "n_rows": len(rows),
                "n_readable": len(rows) - sum(errors.values()),
                "n_missing_or_unreadable": sum(errors.values()),
                "read_errors": dict(errors.most_common()),
                "duplicates_within": duplicates,
                "duplicate_rate": duplicates / len(rows) if rows else 0.0,
                "complete_label_rows": complete,
                "partial_label_rows": len(rows) - complete,
                "tag_counts": dict(sorted(tag_counts.items())),
                "priority_support": {tag: tag_counts[tag] for tag in PRIORITY_TAGS},
                "native_label_count": len(native_counts),
                "provenance": dict(provenance_counts.most_common()),
                "missing_rate": sum(errors.values()) / len(rows) if rows else 0.0,
                "label_noise": "not directly measurable; provenance retained for slicing",
                "contamination_risk": "machine/synthetic labels and shared upstream sources; exact cross-split hash audit follows",
                "leakage": 0,
            }
            print(
                f"done {split}: {dataset_dir.name} rows={len(rows)} readable={len(rows) - sum(errors.values())} "
                f"duplicates={duplicates} labels={len(tag_counts)}",
                flush=True,
            )
    return quality, split_hashes, all_tag_counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    train_quality, train_hashes, train_tags = audit_split(args.train_root, "train", args.out, args.workers)
    eval_quality, eval_hashes, eval_tags = audit_split(args.eval_root, "eval", args.out, args.workers)
    overlaps = train_hashes & eval_hashes
    # Mark every corpus as assessed even when its leakage is zero. Per-corpus
    # overlap counts are filled by a second streaming pass over the frozen indexes.
    overlap_by_dataset: Counter[str] = Counter()
    exclusions: list[dict[str, str]] = []
    with (args.out / "train_index.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            digest = row.get("text_sha256", "")
            if digest and digest in overlaps:
                overlap_by_dataset[row["dataset"]] += 1
                exclusions.append({"dataset": row["dataset"], "uid": row["uid"], "text_sha256": digest})
    for dataset, record in train_quality.items():
        record["leakage"] = overlap_by_dataset[dataset]
    result = {
        "read_window_chars": 20_000,
        "train": train_quality,
        "eval": eval_quality,
        "global": {
            "n_train_unique_text_hashes": len(train_hashes),
            "n_eval_unique_text_hashes": len(eval_hashes),
            "n_cross_split_exact_hashes": len(overlaps),
            "n_training_rows_excluded_for_leakage": len(exclusions),
            "train_tag_counts": dict(sorted(train_tags.items())),
            "eval_tag_counts": dict(sorted(eval_tags.items())),
            "priority_train_support": {tag: train_tags[tag] for tag in PRIORITY_TAGS},
            "priority_eval_support": {tag: eval_tags[tag] for tag in PRIORITY_TAGS},
        },
    }
    atomic_json(args.out / "data_quality.json", result)
    atomic_json(args.out / "train_exclusions.json", exclusions)
    atomic_json(args.out / "catalog.json", {"priority_tags": list(PRIORITY_TAGS)})
    print(json.dumps(result["global"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
