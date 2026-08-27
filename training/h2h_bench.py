"""One-core latency, measured through each arm's real serving path.

`quiet_bench` timed the *shape* of the serving path -- feature extraction plus
one gate dot product plus 58 head dot products, against randomly generated
weights. That is the right instrument for asking what a read window costs, and
the wrong one for a head-to-head: the fusion arm consults four component heads
where the cascade consults one gate and then, only sometimes, its tags. A
random-weight proxy cannot see either difference, and both are real serving cost.

So this times `model.predict(text)` on the actual trained artifact, for every
arm, through one harness.

Two things the numbers depend on and which are therefore fixed and reported:

* **One core, nothing else running.** `set_cpus(1)` caps BLAS and OpenMP. A
  latency measured while a search is using 32 cores is not comparable with one
  measured on a quiet machine, and the Experiment Log has no column for "but it
  was busy".
* **Real documents from the real-world corpora**, at the same >=10 KB target the
  prior lineage benchmarked at, so a number here can be read against the 4.11 ms
  the shipped cascade recorded.

The cascade short-circuits on a document its gate rejects, so its cost depends on
the document mix. That is a property worth seeing rather than averaging away, so
the split by gate decision is reported beside the overall figure.
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
from training.h2h_score import ARMS, load_arm  # noqa: E402
from training.quiet_data import EVAL_ROOT, iter_quiet_corpus, read_document  # noqa: E402
from training.quiet_model import QuietCascade  # noqa: E402

#: The real-world half of the suite: heterogeneous business and web documents,
#: which is where a read window actually costs something.
REAL_CORPORA = ("6589_govdocs2-dualjudge-eval20-3.53k",
                "4000_datax-dualjudge-evalset-1.32k")


def sample_documents(n: int, min_chars: int) -> list[str]:
    out: list[str] = []
    for corpus in REAL_CORPORA:
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


def bench_arm(arm: str, docs: list[str], repeats: int) -> dict:
    model = load_arm(arm)
    is_cascade = isinstance(model, QuietCascade)
    timings: list[float] = []
    gate_open: list[bool] = []
    for _ in range(repeats):
        for text in docs:
            t0 = time.perf_counter()
            tags = model.predict(text)
            timings.append((time.perf_counter() - t0) * 1000.0)
            gate_open.append(bool(tags) if not is_cascade else model.has_pii(text))
    a = np.asarray(timings)
    payload = {
        "arm": arm, "label": ARMS[arm]["label"],
        "read_window_chars": ARMS[arm]["window"],
        "n": int(a.size), "repeats": repeats,
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "mean_ms": float(a.mean()),
        "docs_per_s": float(1000.0 / a.mean()),
    }
    g = np.asarray(gate_open)
    if g.any() and (~g).any():
        payload["p95_ms_firing"] = float(np.percentile(a[g], 95))
        payload["p95_ms_silent"] = float(np.percentile(a[~g], 95))
        payload["fire_rate"] = float(g.mean())
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--cpus", default="1")
    ap.add_argument("--chars", type=int, default=10_000)
    ap.add_argument("--docs", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    set_cpus(args.cpus if args.cpus == "all" else int(args.cpus))

    docs = sample_documents(args.docs, args.chars)
    print(f"{len(docs)} documents of >= {args.chars:,} characters, cpu_budget={args.cpus}",
          file=sys.stderr)
    payload = bench_arm(args.arm, docs, args.repeats)
    payload |= {"cpu_budget": args.cpus, "doc_chars_min": args.chars,
                "n_documents": len(docs), "corpora": list(REAL_CORPORA)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    flag = "PASS" if payload["p95_ms"] <= 5.0 else "FAIL"
    print(f"arm {args.arm}: p50={payload['p50_ms']:.3f} p95={payload['p95_ms']:.3f} "
          f"p99={payload['p99_ms']:.3f} ms  {payload['docs_per_s']:.0f} docs/s   "
          f"5ms budget {flag}")
    if "p95_ms_silent" in payload:
        print(f"  firing p95={payload['p95_ms_firing']:.3f}  "
              f"silent p95={payload['p95_ms_silent']:.3f}  "
              f"fire rate={payload['fire_rate']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
