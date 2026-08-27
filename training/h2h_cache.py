"""The priority lineage's feature cache, so its search stops being I/O-bound.

`quiet_cache` did this for the steady-aim lineage: read every document once into
hashed feature indices, then make every trial linear algebra. The priority
lineage never got the same treatment -- `tune_priority_hash.fit_counts` and
`score_validation` re-read documents on every family run, which is why a
1,000-trial budget there cost hours of disk rather than minutes of arithmetic.

This builds the equivalent cache at the priority lineage's own feature width
(2**17) and its own read profiles. Nothing about the features changes: the
extraction is the same `priority_hash.document_features` call with the same
arguments the tuners pass it, so a count accumulated from this cache is the same
count accumulated from the documents. Only the number of times the disk is
touched changes.

Three profiles, from one read per document:

``train20k``  (20,000, 768, 512) -- what `fit_counts` and `score_validation` see.
``serve1k``   ( 1,000, 768, 512) -- arm A, the shipped `read_window_override`.
``serve12k``  (12,000, 768, 512) -- arm C, matched to the steady-aim read window.

The train/serve split is not symmetry for its own sake. The fusion recipe
calibrates component thresholds on 20,000-character scores and then serves at
1,000, and reproducing that faithfully means the cache has to carry both.

The catalogue is **not** re-derived here. It is loaded from the steady-aim
lineage's frozen `catalogue.json` so that both lineages index the same 58
collapsed labels in the same order -- without which the two arms' score matrices
could not be compared column by column.
"""

from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.priority_hash import document_features  # noqa: E402
from training.quiet_cache import load_catalogue  # noqa: E402
from training.quiet_data import (  # noqa: E402
    EVAL_ROOT,
    TRAIN_ROOT,
    iter_quiet_corpus,
    list_dataset_dirs,
    read_document,
)

CACHE_ROOT = Path(
    "/home/lence/workspace/pii_master/projects/pii-head-to-head-v1/cache")
#: The priority lineage's width, not the steady-aim lineage's 2**18. Each model
#: keeps its own recipe; only the data and the catalogue are shared.
N_FEATURES = 1 << 17
#: name -> (read window in characters, max tokens, max hashed features per doc)
PROFILES: dict[str, tuple[int, int, int]] = {
    "train20k": (20_000, 768, 512),
    "serve1k": (1_000, 768, 512),
    "serve12k": (12_000, 768, 512),
}
CHUNK = 2_000
READ_LIMIT = max(w for w, _, _ in PROFILES.values())


# --------------------------------------------------------------------- workers
_WORKER: dict[str, Any] = {}


def _init(dataset_dir: str, label_index: dict[str, int]) -> None:
    _WORKER["rows"] = list(iter_quiet_corpus(Path(dataset_dir)))
    _WORKER["label_index"] = label_index


def _do_chunk(bounds: tuple[int, int]) -> dict[str, Any]:
    lo, hi = bounds
    rows = _WORKER["rows"][lo:hi]
    index = _WORKER["label_index"]
    per_profile: dict[str, list[np.ndarray]] = {name: [] for name in PROFILES}
    label_cols: list[np.ndarray] = []
    doc_target = np.empty(len(rows), dtype=np.int8)
    tag_complete = np.empty(len(rows), dtype=bool)
    n_chars = np.zeros(len(rows), dtype=np.int32)
    read_errors = 0
    for i, qr in enumerate(rows):
        try:
            text = read_document(Path(qr.path), limit=READ_LIMIT)
        except (FileNotFoundError, OSError):
            text = ""
            read_errors += 1
        n_chars[i] = len(text)
        for name, (window, max_tokens, max_features) in PROFILES.items():
            per_profile[name].append(
                document_features(
                    text[:window], n_features=N_FEATURES,
                    max_tokens=max_tokens, max_features=max_features,
                )
            )
        cols = [index[t] for t in qr.collapsed_labels if t in index]
        label_cols.append(np.asarray(sorted(cols), dtype=np.int32))
        doc_target[i] = 1 if qr.doc_has_pii is True else (0 if qr.doc_has_pii is False else -1)
        tag_complete[i] = qr.tag_labels_complete
    return {
        "lo": lo,
        "features": {name: per_profile[name] for name in PROFILES},
        "label_cols": label_cols,
        "doc_target": doc_target,
        "tag_complete": tag_complete,
        "n_chars": n_chars,
        "read_errors": read_errors,
    }


def _pack(list_of_arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.fromiter((len(a) for a in list_of_arrays), dtype=np.int64,
                          count=len(list_of_arrays))
    indptr = np.zeros(len(list_of_arrays) + 1, dtype=np.int64)
    np.cumsum(lengths, out=indptr[1:])
    data = (np.concatenate(list_of_arrays) if list_of_arrays
            else np.empty(0, dtype=np.int32))
    return indptr, data.astype(np.int32, copy=False)


def build_corpus(dataset_dir: Path, label_index: dict[str, int], workers: int) -> dict[str, Any]:
    n = sum(1 for _ in iter_quiet_corpus(dataset_dir))
    bounds = [(lo, min(lo + CHUNK, n)) for lo in range(0, n, CHUNK)]
    results: list[dict[str, Any]] = []
    with Pool(workers, initializer=_init, initargs=(str(dataset_dir), label_index)) as pool:
        for done, res in enumerate(pool.imap_unordered(_do_chunk, bounds), 1):
            results.append(res)
            if done % 10 == 0 or done == len(bounds):
                print(f"    {dataset_dir.name}: {done}/{len(bounds)} chunks",
                      file=sys.stderr, flush=True)
    results.sort(key=lambda r: r["lo"])

    payload: dict[str, Any] = {}
    for name in PROFILES:
        flat = [a for r in results for a in r["features"][name]]
        indptr, data = _pack(flat)
        payload[f"indptr_{name}"] = indptr
        payload[f"indices_{name}"] = data
    lab_indptr, lab_data = _pack([a for r in results for a in r["label_cols"]])
    payload["label_indptr"] = lab_indptr
    payload["label_cols"] = lab_data
    payload["doc_target"] = np.concatenate([r["doc_target"] for r in results])
    payload["tag_complete"] = np.concatenate([r["tag_complete"] for r in results])
    payload["n_chars"] = np.concatenate([r["n_chars"] for r in results])

    out = CACHE_ROOT / f"{dataset_dir.name}.npz"
    np.savez_compressed(out, **payload)
    return {
        "dataset": dataset_dir.name,
        "n_rows": int(len(payload["doc_target"])),
        "read_errors": sum(r["read_errors"] for r in results),
        **{f"nnz_{name}": int(len(payload[f"indices_{name}"])) for name in PROFILES},
        "bytes": out.stat().st_size,
    }


def main() -> int:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    # Borrowed, never re-derived: the two lineages must index the same labels in
    # the same order or their score matrices are not comparable.
    cat = load_catalogue()
    label_index = {t: i for i, t in enumerate(cat["labels"])}
    print(f"catalogue: {cat['n_labels']} collapsed labels (borrowed from quiet_cache)",
          file=sys.stderr)
    (CACHE_ROOT / "catalogue.json").write_text(json.dumps({
        "n_features": N_FEATURES,
        "profiles": {k: list(v) for k, v in PROFILES.items()},
        "labels": list(cat["labels"]),
        "n_labels": cat["n_labels"],
        "source": "training/quiet_cache.py catalogue.json (identical label order)",
    }, indent=1), encoding="utf-8")

    report = []
    for root in (TRAIN_ROOT, EVAL_ROOT):
        for d in list_dataset_dirs(root):
            print(f"  caching {d.name} ...", file=sys.stderr, flush=True)
            report.append(build_corpus(d, label_index, workers))
            r = report[-1]
            print(f"    -> {r['n_rows']:,} rows, {r['nnz_train20k']:,} nnz@train20k, "
                  f"{r['nnz_serve1k']:,} nnz@serve1k, "
                  f"{r['read_errors']} read errors, {r['bytes']/1e6:.0f} MB",
                  file=sys.stderr, flush=True)
    (CACHE_ROOT / "cache_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"total rows: {sum(r['n_rows'] for r in report):,}")
    print(f"total read errors: {sum(r['read_errors'] for r in report):,}")
    print(f"cache size: {sum(r['bytes'] for r in report)/1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
