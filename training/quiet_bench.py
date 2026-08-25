"""One-core latency for the serving path, measured the way the policy reads it.

The gate is p95 on a 10 KB document with a single core available. Feature
extraction dominates: the linear algebra is 58 dot products over a few hundred
non-zeros, while tokenising the read window is real work proportional to its
length. So the read window is a latency decision before it is a quality one.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_pipeline import set_cpus  # noqa: E402
from training.priority_hash import document_features  # noqa: E402
from training.quiet_data import EVAL_ROOT, iter_quiet_corpus, read_document  # noqa: E402

N_FEATURES = 1 << 18


def sample_documents(n: int, min_chars: int) -> list[str]:
    """Real documents of at least the target size, from the real-world corpora."""
    out: list[str] = []
    for corpus in ("6589_govdocs2-dualjudge-eval20-3.53k", "4000_datax-dualjudge-evalset-1.32k"):
        for qr in iter_quiet_corpus(EVAL_ROOT / corpus):
            try:
                text = read_document(Path(qr.path), limit=min_chars * 2)
            except (FileNotFoundError, OSError):
                continue
            if len(text) >= min_chars:
                out.append(text)
            if len(out) >= n:
                return out
    return out


def bench(windows: tuple[int, ...], n_labels: int, docs: list[str], repeats: int) -> dict:
    rng = np.random.default_rng(0)
    W = rng.normal(0, 0.1, size=(n_labels, N_FEATURES)).astype(np.float32)
    gate = rng.normal(0, 0.1, size=N_FEATURES).astype(np.float32)
    result = {}
    for window in windows:
        timings: list[float] = []
        for _ in range(repeats):
            for text in docs:
                t0 = time.perf_counter()
                idx = document_features(text[:window], n_features=N_FEATURES, max_features=512)
                # stage 1: the document gate
                g = float(gate[idx].sum())
                # stage 2: the per-tag heads, consulted only when the gate fires
                if g > -1e9:
                    _ = W[:, idx].sum(axis=1)
                timings.append((time.perf_counter() - t0) * 1000.0)
        a = np.asarray(timings)
        result[window] = {
            "n": int(a.size),
            "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)),
            "p99_ms": float(np.percentile(a, 99)),
            "mean_ms": float(a.mean()),
            "docs_per_s": float(1000.0 / a.mean()),
        }
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpus", default="1")
    ap.add_argument("--chars", type=int, default=10_000)
    ap.add_argument("--docs", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--labels", type=int, default=58)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    set_cpus(args.cpus if args.cpus == "all" else int(args.cpus))

    docs = sample_documents(args.docs, args.chars)
    print(f"{len(docs)} documents of >= {args.chars:,} characters", file=sys.stderr)
    res = bench((1_000, 2_000, 4_000, 8_000), args.labels, docs, args.repeats)
    payload = {"cpu_budget": args.cpus, "doc_chars_min": args.chars,
               "n_documents": len(docs), "repeats": args.repeats, "windows": res}
    if args.out:
        args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"{'window':>8}{'p50':>9}{'p95':>9}{'p99':>9}{'docs/s':>10}   budget 5ms")
    for w, r in res.items():
        flag = "PASS" if r["p95_ms"] <= 5.0 else "FAIL"
        print(f"{w:>8}{r['p50_ms']:>9.3f}{r['p95_ms']:>9.3f}{r['p99_ms']:>9.3f}"
              f"{r['docs_per_s']:>10.0f}   {flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
