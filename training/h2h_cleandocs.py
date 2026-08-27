"""Test 10,000 real PII-free documents against the gate — the honest way.

These are what the synthetic corpus failed to be: real files, of the same kind
the model is weakest on, asserted PII-free. But they are **govdocs**, and so are
two of the corpora this project trains and measures on, so the first job is not
scoring — it is establishing which of them may be touched at all.

Two disqualifications, both measured rather than assumed:

* **587 of the 10,000 are in a SEALED evaluation corpus.** Training on those
  leaks the measurement into the model, and every number in this project would
  be quietly wrong afterwards.
* **782 carry conflicting gold.** The existing dual-judge process labelled them
  PII-bearing (`sensitivity: low`); this manifest says `none`. That is 27.2% of
  the overlap, and it is not a rounding difference — it is two labelling passes
  disagreeing about the same files. Admitting them as negatives would teach the
  gate to stay silent on documents the sealed set counts as positives.

So only the **7,126 documents with no existing gold at all** are used. The rest
are excluded by name and counted, because "we could not use this" and "this
helped" are different claims.

The three questions, in the order that makes the answer trustworthy:

1. **Are they adversarial?** Score them with the champion gate. Documents it
   already gets right teach it nothing.
2. **Are they separable?** The synthetic corpus failed because generated and
   real text separate at AUC 1.0000 — the gate learned the style and nothing
   else. Real documents should sit near 0.5 against other real documents. This
   check is the precondition; if it fails, nothing downstream is worth running.
3. **Do they transfer?** Add them to gate training, hold 20% out, and read the
   sealed real AUC. Held-out fire rate falling while sealed AUC stays flat is
   the same lookalike-success the synthetic run produced.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.h2h_priority import PROJECT  # noqa: E402
from training.priority_hash import document_features  # noqa: E402
from training.quiet_cache import PROFILES, load_catalogue  # noqa: E402
from training.quiet_data import read_document  # noqa: E402

CORPUS = Path("/home/lence/workspace/data/clean_docs_10000_no_pii_phi_pfi")
CACHE = PROJECT / "cache" / "clean_docs.npz"
#: Corpora whose gold already covers a document: overlapping it means the
#: document is either sealed (leakage) or already labelled (possibly contrary).
EXISTING = (
    ("train", "/home/lence/workspace/data/1-train/23693_govdocs2-dualjudge-train80-12.86k"),
    ("train", "/home/lence/workspace/data/1-train/15986_datax-dualjudge-trainset-5.36k"),
    ("eval", "/home/lence/workspace/data/2-eval/6589_govdocs2-dualjudge-eval20-3.53k"),
    ("eval", "/home/lence/workspace/data/2-eval/4000_datax-dualjudge-evalset-1.32k"),
)


def admissible() -> tuple[list[str], dict[str, int]]:
    """The document ids that may be used, and why the others may not."""
    rows = list(csv.DictReader((CORPUS / "manifest.csv").open()))
    clean = {r["doc_id"] for r in rows}
    in_eval: set[str] = set()
    conflict: set[str] = set()
    overlap: set[str] = set()
    for split, d in EXISTING:
        man = json.loads((Path(d) / "manifest.json").read_text(encoding="utf-8"))
        for r in (man if isinstance(man, list) else man.get("rows", [])):
            did = r.get("doc_id")
            if did in clean:
                overlap.add(did)
                if split == "eval":
                    in_eval.add(did)
                if r.get("pii_entities"):
                    conflict.add(did)
    usable = sorted(clean - overlap)
    return usable, {"total": len(clean), "overlap": len(overlap),
                    "in_sealed_eval": len(in_eval), "label_conflict": len(conflict),
                    "usable": len(usable)}


_W: dict[str, Any] = {}


def _init(ids: list[str]) -> None:
    _W["ids"] = ids


def _one(bounds: tuple[int, int]) -> dict[str, Any]:
    lo, hi = bounds
    window, max_tokens, max_feats = PROFILES["deep"]
    n_features = int(load_catalogue()["n_features"])
    feats, chars, errs = [], [], 0
    for did in _W["ids"][lo:hi]:
        try:
            text = read_document(CORPUS / did, limit=window)
        except (OSError, ValueError):
            text, errs = "", errs + 1
        chars.append(len(text))
        feats.append(document_features(text[:window], n_features=n_features,
                                       max_tokens=max_tokens, max_features=max_feats))
    return {"lo": lo, "feats": feats, "chars": chars, "errs": errs}


def featurise(ids: list[str], workers: int) -> tuple[sp.csr_matrix, np.ndarray, int]:
    n_features = int(load_catalogue()["n_features"])
    chunk = 200
    bounds = [(lo, min(lo + chunk, len(ids))) for lo in range(0, len(ids), chunk)]
    out: list[dict[str, Any]] = []
    with Pool(workers, initializer=_init, initargs=(ids,)) as pool:
        for done, r in enumerate(pool.imap_unordered(_one, bounds), 1):
            out.append(r)
            if done % 10 == 0 or done == len(bounds):
                print(f"    {done}/{len(bounds)} chunks", file=sys.stderr, flush=True)
    out.sort(key=lambda r: r["lo"])
    feats = [a for r in out for a in r["feats"]]
    chars = np.concatenate([np.asarray(r["chars"], dtype=np.int32) for r in out])
    indptr = np.zeros(len(feats) + 1, dtype=np.int64)
    np.cumsum([len(a) for a in feats], out=indptr[1:])
    data = np.concatenate(feats) if feats else np.empty(0, np.int32)
    X = sp.csr_matrix((np.ones(len(data), np.float32), data, indptr),
                      shape=(len(feats), n_features))
    return X, chars, sum(r["errs"] for r in out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    ids, counts = admissible()
    print("admissibility:")
    for k, v in counts.items():
        print(f"    {k:18s} {v:>7,}")
    print(f"\nfeaturising {len(ids):,} usable documents at the `deep` profile ...",
          flush=True)
    X, chars, errs = featurise(ids, args.workers)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, indptr=X.indptr, indices=X.indices,
                        n_chars=chars, ids=np.asarray(ids))
    nnz = np.diff(X.indptr)
    print(f"  {X.shape[0]:,} documents, {errs} read errors")
    print(f"  chars   median {np.median(chars):,.0f}  mean {chars.mean():,.0f}")
    print(f"  features/doc median {np.median(nnz):.0f}  mean {nnz.mean():.0f}")
    print(f"  empty (no features): {int((nnz == 0).sum()):,}")
    print(f"-> {CACHE}")
    (PROJECT / "probe" / "cleandocs_admissibility.json").write_text(
        json.dumps(counts, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
