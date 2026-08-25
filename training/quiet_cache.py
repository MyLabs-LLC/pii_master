"""Read every document once; make every trial pure NumPy.

A 1,000-trial budget cannot re-read 657,560 documents per trial -- many of them
zipped Office XML or HTML that has to be parsed before a single feature exists.
So the read happens once, into hashed, value-redacted feature indices, and each
trial is then linear algebra over a cached CSR matrix.

Three things are frozen here because a trial must not be able to move them:

* **The catalogue** -- the collapsed sensitive-tag vocabulary, derived from the
  manifests alone and written to ``catalogue.json`` before any document is read.
* **The hash width** (2**18). The prior lineage searched this; it is fixed here
  so the budget goes to the levers that matter for precision -- negative
  weighting, loss shape, per-label operating points, calibration -- rather than
  to a parameter worth a fraction of a point. The narrowing is deliberate and
  is reported.
* **The read profiles.** Not just a character window: ``document_features``
  caps at ``max_tokens`` regardless of how much text it is handed, so a 4,000
  character window already sits at that cap and widening it alone buys nothing.
  A profile therefore pins ``(window, max_tokens, max_features)`` together.
  Three are cached from a single read, because real business documents are long
  -- govdocs2 averages 149,000 characters -- and how much of one the model is
  allowed to see is the strongest lever found on real-world recall, while one
  core has the latency headroom to pay for it (0.45 ms p95 at ``fast``, 1.44 ms
  at ``std``, against a 5 ms budget).

Each row also carries what its gold can say: ``doc_target`` is 1/0/-1
(positive / negative / unknown) and ``tag_complete`` gates whether the row may
contribute a negative to a per-tag loss.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.priority_data import SENSITIVE_PREFIXES  # noqa: E402
from training.priority_hash import document_features  # noqa: E402
from training.quiet_data import (  # noqa: E402
    EVAL_ROOT,
    TRAIN_ROOT,
    collapse_tags,
    iter_quiet_corpus,
    list_dataset_dirs,
    read_document,
)

CACHE_ROOT = Path("/home/lence/workspace/pii_master/projects/pii-quiet-alarm/cache")
N_FEATURES = 1 << 18
#: name -> (read window in characters, max tokens, max hashed features per doc)
PROFILES: dict[str, tuple[int, int, int]] = {
    "fast": (1_000, 768, 512),
    "std": (4_000, 768, 512),
    "deep": (12_000, 2_048, 1_024),
}
CHUNK = 2_000


# --------------------------------------------------------------------------- catalogue
def freeze_catalogue() -> dict[str, Any]:
    """The collapsed tag vocabulary, from manifests only -- no document reads."""
    counts: Counter[str] = Counter()
    per_corpus: dict[str, Counter] = {}
    for root in (TRAIN_ROOT, EVAL_ROOT):
        for d in list_dataset_dirs(root):
            local: Counter[str] = Counter()
            for qr in iter_quiet_corpus(d):
                local.update(qr.collapsed_labels)
            per_corpus[d.name] = local
            counts.update(local)
    labels = tuple(sorted(counts))
    payload = {
        "n_features": N_FEATURES,
        "profiles": {k: list(v) for k, v in PROFILES.items()},
        "labels": list(labels),
        "n_labels": len(labels),
        "total_counts": dict(counts),
        "per_corpus_counts": {k: dict(v) for k, v in per_corpus.items()},
    }
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    (CACHE_ROOT / "catalogue.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return payload


def load_catalogue() -> dict[str, Any]:
    return json.loads((CACHE_ROOT / "catalogue.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- workers
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
            text = read_document(Path(qr.path), limit=max(w for w, _, _ in PROFILES.values()))
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
    print("freezing catalogue ...", file=sys.stderr)
    cat = freeze_catalogue()
    label_index = {t: i for i, t in enumerate(cat["labels"])}
    print(f"catalogue: {cat['n_labels']} collapsed labels", file=sys.stderr)

    report = []
    for root in (TRAIN_ROOT, EVAL_ROOT):
        for d in list_dataset_dirs(root):
            print(f"  caching {d.name} ...", file=sys.stderr, flush=True)
            report.append(build_corpus(d, label_index, workers))
            r = report[-1]
            print(f"    -> {r['n_rows']:,} rows, {r['nnz_fast']:,} nnz@fast, "
                  f"{r['nnz_deep']:,} nnz@deep, "
                  f"{r['read_errors']} read errors, {r['bytes']/1e6:.0f} MB",
                  file=sys.stderr, flush=True)
    (CACHE_ROOT / "cache_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"total rows: {sum(r['n_rows'] for r in report):,}")
    print(f"total read errors: {sum(r['read_errors'] for r in report):,}")
    print(f"cache size: {sum(r['bytes'] for r in report)/1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
