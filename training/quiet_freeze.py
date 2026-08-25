"""Freeze and verify the data state a run was fitted on.

An external decontamination pass rewrote all eight training manifests at
2026-08-25 11:32 -- during this session, between two censuses -- removing the
22,816 rows that leaked into evaluation.  That was the right change, and it is
exactly why a run must pin what it read: a 1,000-trial budget spent against a
moving corpus produces numbers nothing can reproduce.

``freeze`` records the manifest digests and row counts; ``verify`` re-reads them
and fails loudly if anything moved.  ``verify`` runs before the first fit and
again before promotion, so a mid-run change is caught rather than averaged in.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.quiet_data import (  # noqa: E402
    EVAL_ROOT,
    TRAIN_ROOT,
    iter_quiet_corpus,
    list_dataset_dirs,
)

SNAPSHOT = Path(__file__).resolve().parents[1] / "projects/pii-quiet-alarm/data_snapshot.json"


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _corpus_state(dataset_dir: Path) -> dict[str, Any]:
    files = {}
    for name in ("manifest.json", "labels.jsonl"):
        p = dataset_dir / name
        if p.is_file():
            files[name] = {"sha256": _digest(p), "bytes": p.stat().st_size,
                           "mtime": p.stat().st_mtime}
    n = pos = neg = unk = 0
    for qr in iter_quiet_corpus(dataset_dir):
        n += 1
        pos += qr.doc_has_pii is True
        neg += qr.doc_has_pii is False
        unk += qr.doc_has_pii is None
    return {"files": files, "n_rows": n, "doc_positive": pos,
            "doc_negative": neg, "doc_unknown": unk}


def freeze() -> dict[str, Any]:
    snap: dict[str, Any] = {"train": {}, "eval": {}}
    for root, key in ((TRAIN_ROOT, "train"), (EVAL_ROOT, "eval")):
        for d in list_dataset_dirs(root):
            snap[key][d.name] = _corpus_state(d)
            print(f"  froze {d.name}", file=sys.stderr, flush=True)
    snap["totals"] = {
        split: {
            "rows": sum(v["n_rows"] for v in snap[split].values()),
            "doc_positive": sum(v["doc_positive"] for v in snap[split].values()),
            "doc_negative": sum(v["doc_negative"] for v in snap[split].values()),
        }
        for split in ("train", "eval")
    }
    return snap


def verify(snapshot: dict[str, Any]) -> list[str]:
    drift: list[str] = []
    for root, key in ((TRAIN_ROOT, "train"), (EVAL_ROOT, "eval")):
        recorded = snapshot[key]
        present = {d.name for d in list_dataset_dirs(root)}
        for name in sorted(set(recorded) ^ present):
            drift.append(f"{key}/{name}: corpus appeared or disappeared")
        for name in sorted(set(recorded) & present):
            want = recorded[name]
            for fname, meta in want["files"].items():
                p = root / name / fname
                if not p.is_file():
                    drift.append(f"{key}/{name}/{fname}: missing")
                elif _digest(p) != meta["sha256"]:
                    drift.append(f"{key}/{name}/{fname}: content changed")
    return drift


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "freeze"
    if mode == "freeze":
        snap = freeze()
        SNAPSHOT.write_text(json.dumps(snap, indent=1), encoding="utf-8")
        t = snap["totals"]
        print(f"train: {t['train']['rows']:,} rows  "
              f"{t['train']['doc_positive']:,} positive  {t['train']['doc_negative']:,} negative")
        print(f"eval:  {t['eval']['rows']:,} rows  "
              f"{t['eval']['doc_positive']:,} positive  {t['eval']['doc_negative']:,} negative")
        print(f"wrote {SNAPSHOT}")
        return 0
    snap = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    drift = verify(snap)
    if drift:
        print("DATA DRIFT since the snapshot:", file=sys.stderr)
        for d in drift:
            print("  " + d, file=sys.stderr)
        return 1
    print("data matches the frozen snapshot")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
