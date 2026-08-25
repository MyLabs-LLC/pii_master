"""One-core latency comparison for the per-label threshold candidate.

Reuses ``benchmark_read_depth``'s sample, warm-up and pinning so the numbers are
directly comparable with ``benchmarks/read_depth.json``: the same deterministic
stratified sample, the same 20-document warm-up, and the process pinned to one
core for the whole measurement.

The candidate differs from its source only in ``HashCueModel.thresholds``, so
the expectation is that latency is unchanged; this measures it rather than
asserting it, because the 5 ms p95 budget is a hard gate constraint and an
unmeasured constraint is not a passed one.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from training.benchmark_read_depth import stratified_sample
from training.priority_data import read_document
from training.priority_hash import load_priority_model


def measure(model, sample: list[dict[str, Any]]) -> dict[str, float]:
    window = model.read_window_chars
    for row in sample[:20]:                      # warm-up, excluded from timings
        model.predict(read_document(Path(row["path"]), limit=window))

    latencies_ms: list[float] = []
    started_all = time.perf_counter()
    for row in sample:
        text = read_document(Path(row["path"]), limit=window)
        started = time.perf_counter_ns()
        model.predict(text)
        latencies_ms.append((time.perf_counter_ns() - started) / 1_000_000)
    elapsed_s = time.perf_counter() - started_all

    values = np.asarray(latencies_ms)
    return {
        "n_documents": len(sample),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "mean_ms": float(values.mean()),
        "docs_per_s": len(sample) / elapsed_s if elapsed_s else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=Path, default=Path("projects/pii-priority-recall-v1"))
    ap.add_argument("--families", nargs="+", default=["champion_1k", "perlabel_v4"])
    ap.add_argument("--sample-per-dataset", type=int, default=125)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    project = args.project.resolve()
    index_rows = [
        json.loads(line)
        for line in (project / "data" / "eval_index.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    sample = stratified_sample(index_rows, per_dataset=args.sample_per_dataset)

    original = os.sched_getaffinity(0)
    core = min(original)
    os.sched_setaffinity(0, {core})
    try:
        rows = []
        for family in args.families:
            model = load_priority_model(project / "models" / family)
            speed = measure(model, sample)
            speed["family"] = family
            speed["read_window_chars"] = model.read_window_chars
            rows.append(speed)
            print(json.dumps(speed, sort_keys=True), flush=True)
    finally:
        os.sched_setaffinity(0, original)

    payload = {"cpu_affinity_core": core, "cpu_budget": 1, "rows": rows}
    out = args.out or (project / "benchmarks" / "perlabel_latency.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
