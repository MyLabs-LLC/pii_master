"""What every model fires on 7,126 real documents that contain no PII.

Every corpus in the suite is synthetic text or extracted plain text. These are
real files in the formats a deployment actually meets — 3,296 PDF, 2,657 DOC,
1,559 HTML, 771 XLS, 587 PPT, plus CSV/XML/RTF — and each was judged to contain
no personal data at all.

## What an all-negative corpus can and cannot measure

It **cannot** measure precision. Precision is TP/(TP+FP), and a corpus with no
positives supplies no TP; the ratio is undefined, not zero. An earlier note in
this project claimed otherwise and was wrong.

What it measures exactly, with no annotation ambiguity anywhere, is the
**false-positive side**: every tag that fires here is a false positive, because
there is nothing here to correctly detect. That yields

* the **document false-alarm rate** — how often the cascade flags a clean file;
* **false positives per 1,000 documents, per tag** — which tags cause it;

and those compose with the sealed suite's true-positive counts into an
**implied precision** at any assumed prevalence of clean documents (`--mix`).
That last number is the one a deployment cares about, and it is the reason this
corpus is worth scoring: the suite's precision figures assume the suite's own
mixture, which is roughly half PII-bearing. Real document estates are not.

## Admissibility is inherited, not re-derived

`h2h_cleandocs.admissible()` already established which documents may be touched:
of 10,000, some overlap `govdocs2`/`datax` corpora this project trains and
measures on — including documents in a **sealed** eval split — and some carry
gold that contradicts this manifest. Only the documents with no existing gold at
all are used. That function is imported rather than reimplemented so the two
runs cannot drift apart.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.h2h_scorecard_rebuild import retarget_cache  # noqa: E402

SC = Path("projects/pii-scorecard-60")
OUT = Path("projects/pii-target-8070")
CACHE = OUT / "cache/clean_docs_61.npz"

#: Every 61-label cascade in the repository, as (name, directory).
MODELS = [
    ("cascade_scorecard61", SC / "models/cascade_scorecard61"),
    ("cascade_v2_9corp", SC / "models/cascade_v2_9corp"),
] + [(n, OUT / "models" / n) for n in (
    "cascade_p80r70", "cascade_p80r90", "cascade_p88r90", "cascade_p88r90b1",
    "cascade_p90r85b1", "cascade_p90r90", "v2_p80r90", "v2_p88r90",
    "v2_p90r85b1")]


def build_cache(workers: int) -> dict:
    """Featurise the admissible documents once, in the 61-label feature space."""
    from training.h2h_cleandocs import admissible, featurise

    ids, why = admissible()
    print(f"admissible: {why}", flush=True)
    t0 = time.perf_counter()
    X, chars, errs = featurise(ids, workers)
    print(f"featurised {X.shape[0]:,} documents in "
          f"{time.perf_counter() - t0:.0f}s ({errs} unreadable)", flush=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(CACHE.with_suffix(".X.npz"), X)
    CACHE.with_suffix(".meta.json").write_text(json.dumps(
        {"ids": ids, "why": why, "chars": chars.tolist(), "errors": errs},
        indent=1), encoding="utf-8")
    return {"X": X, "ids": ids, "why": why, "chars": chars, "errors": errs}


def load_cache() -> dict | None:
    xp, mp = CACHE.with_suffix(".X.npz"), CACHE.with_suffix(".meta.json")
    if not (xp.is_file() and mp.is_file()):
        return None
    meta = json.loads(mp.read_text(encoding="utf-8"))
    return {"X": sp.load_npz(xp), "ids": meta["ids"], "why": meta["why"],
            "chars": np.asarray(meta["chars"]), "errors": meta["errors"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8,
                    help="featurisation only; scoring is single-core")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--mix", type=float, default=0.90,
                    help="assumed fraction of clean documents in a real estate, "
                         "for the implied-precision column")
    ap.add_argument("--out", type=Path, default=OUT / "evaluations/clean_docs.json")
    args = ap.parse_args()

    cat = retarget_cache(SC / "cache", 61)
    labels = list(cat["labels"])

    from training.h2h_score import predict_cascade  # noqa: E402
    from training.quiet_model import QuietCascade  # noqa: E402

    data = None if args.rebuild else load_cache()
    if data is None:
        data = build_cache(args.workers)
    X = data["X"]
    n = X.shape[0]
    print(f"\n{n:,} clean documents, {X.shape[1]:,} features\n", flush=True)

    # Sealed-suite true positives, to compose an implied precision with.
    sealed = {}
    mat = OUT / "evaluations/matrix.json"
    if mat.is_file():
        m = json.loads(mat.read_text(encoding="utf-8"))
        keep = set(m["sealed"])
        for r in m["results"]:
            if r["corpus"] in keep:
                agg = sealed.setdefault(r["model"], {"tp": 0, "fp": 0})
                for t in r["per_tag"].values():
                    agg["tp"] += t["tp"]
                    agg["fp"] += t["fp"]

    rows = []
    for name, d in MODELS:
        if not (d / "model.json").is_file():
            print(f"  {name:<24} MISSING at {d}", flush=True)
            continue
        model = QuietCascade.load(d)
        t0 = time.perf_counter()
        fired, fired_doc, _ = predict_cascade(model, X)
        fp_per_tag = np.asarray(fired.sum(axis=0)).ravel().astype(int)
        any_tag = np.asarray(fired.sum(axis=1)).ravel() > 0
        row = {
            "model": name,
            "n_documents": int(n),
            "gate_open_rate": float(fired_doc.mean()),
            "doc_false_alarm_rate": float(any_tag.mean()),
            "false_positives": int(fp_per_tag.sum()),
            "fp_per_1000_docs": float(1000.0 * fp_per_tag.sum() / n),
            "enabled_tags": int(np.isfinite(model.tag_thresholds).sum()),
            "seconds": round(time.perf_counter() - t0, 1),
            "per_tag_fp": {labels[j]: int(v)
                           for j, v in enumerate(fp_per_tag) if v},
        }
        s = sealed.get(name)
        if s and s["tp"]:
            # Scale the clean corpus up to `mix` of a hypothetical estate that
            # otherwise looks like the sealed suite, then recompute precision.
            n_sealed_docs = sum(r["n_rows"] for r in
                                json.loads(mat.read_text(encoding="utf-8"))["results"]
                                if r["model"] == name and r["corpus"] in keep)
            scale = ((args.mix / (1 - args.mix)) * n_sealed_docs / n) if args.mix < 1 else 0
            implied_fp = s["fp"] + scale * fp_per_tag.sum()
            row["sealed_tp"] = s["tp"]
            row["sealed_fp"] = s["fp"]
            row["sealed_precision"] = s["tp"] / max(s["tp"] + s["fp"], 1)
            row["implied_precision_at_mix"] = float(
                s["tp"] / max(s["tp"] + implied_fp, 1e-9))
            row["mix"] = args.mix
        rows.append(row)
        print(f"  {name:<24} doc-alarm {row['doc_false_alarm_rate']:>7.4f}  "
              f"FP {row['false_positives']:>7,}  "
              f"FP/1k {row['fp_per_1000_docs']:>8.1f}"
              + (f"  implied-P {row['implied_precision_at_mix']:.4f}"
                 if "implied_precision_at_mix" in row else ""), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"corpus": "clean_docs_10000_no_pii_phi_pfi",
         "admissibility": data["why"], "n_scored": int(n),
         "unreadable": data["errors"], "mix": args.mix,
         "note": ("all-negative corpus: measures the false-positive side only; "
                  "precision is undefined here and is composed with sealed-suite "
                  "true positives for the implied-precision column"),
         "results": rows}, indent=1), encoding="utf-8")
    print(f"\n{len(rows)} models -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
