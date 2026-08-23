"""Reproducible single-core benchmark for the current pipeline.

Generates a deterministic synthetic corpus (seeded PRNG; prose chunks with
entity snippets sprinkled roughly every ~400 characters), then times
:func:`pii_master.classify.scan_text` per document and reports latency
percentiles per size bucket against the 5 ms / 10 KB budget from
docs/DESIGN.md section 5. Stdlib only, like everything else in Stage 1.

The budget rule: a typical document (<= TYPICAL_BYTES of text) must scan in
<= budget_ms at p95; larger buckets get a pro-rated allowance
(budget_ms per TYPICAL_BYTES).

Two serving modes have two budgets, and both are gateable here:

    fast (default)   rules only, stdlib only        <= 5 ms / 10 KB
    deep (--deep)    rules + the Stage 2 student    <= 25 ms / 10 KB

`--deep` is what turns docs/DISTILLATION_PLAN.md's gate 1 into something the
shipped CLI can check on the shipped pipeline, rather than a number produced by
a script in training/ that measures a cascade assembled by hand.
"""

from __future__ import annotations

import random
import statistics
import time
from dataclasses import dataclass, field

from .classify import scan_text
from .validators import luhn_ok

TYPICAL_BYTES = 10_000
DEFAULT_SIZES = (1_000, 10_000, 100_000)

#: Per-mode p95 budget in ms per TYPICAL_BYTES. The deep figure is the one
#: docs/DISTILLATION_PLAN.md section 6 set for the opt-in tier; the fast figure
#: is the standing production contract from docs/DESIGN.md section 5.
MODE_BUDGET_MS = {"fast": 5.0, "deep": 25.0}

_WORDS = (
    "the quarterly report covers revenue growth and operating margins across "
    "each region while the planning team reviews staffing vendor contracts "
    "timeline updates budget notes facility access training records shipping "
    "schedules customer feedback product roadmap release criteria meeting "
    "minutes action items followup owners status summary draft final"
).split()


def _make_card(rng: random.Random) -> str:
    digits = "4" + "".join(str(rng.randrange(10)) for _ in range(14))
    for check in "0123456789":
        if luhn_ok(digits + check):
            return digits + check
    raise AssertionError("unreachable: some check digit always satisfies Luhn")


def _make_snippet(rng: random.Random, kind: int) -> str:
    if kind == 0:
        return f"contact user{rng.randrange(1000)}@example.com for details."
    if kind == 1:
        return f"call ({rng.randrange(200, 990)}) 555-{rng.randrange(10000):04d} today."
    if kind == 2:
        area = rng.choice([n for n in range(100, 900) if n != 666])
        return f"SSN {area:03d}-{rng.randrange(1, 100):02d}-{rng.randrange(1, 10000):04d} noted."
    if kind == 3:
        return f"card {_make_card(rng)} charged."
    if kind == 4:
        return f"host 10.{rng.randrange(256)}.{rng.randrange(256)}.{rng.randrange(1, 255)} responded."
    if kind == 5:
        return (
            f"DOB: {rng.randrange(1, 13):02d}/{rng.randrange(1, 29):02d}/"
            f"{rng.randrange(1940, 2010)} on file."
        )
    if kind == 6:
        return f"MRN: {rng.randrange(1_000_000, 10_000_000)} attached."
    return f"see https://example.com/r/{rng.randrange(100_000)} for the record."


def make_doc(rng: random.Random, target_bytes: int) -> str:
    """Prose chunks (~80 chars) with an entity snippet every 5th chunk."""
    parts: list[str] = []
    size = 0
    chunk_index = 0
    while size < target_bytes:
        if chunk_index % 5 == 4:
            chunk = _make_snippet(rng, chunk_index // 5 % 8)
        else:
            chunk = " ".join(rng.choice(_WORDS) for _ in range(12)) + "."
        parts.append(chunk)
        size += len(chunk) + 1
        chunk_index += 1
    return " ".join(parts)


def generate_docs(
    seed: int, sizes: tuple[int, ...], docs_per_size: int
) -> dict[int, list[str]]:
    rng = random.Random(seed)
    return {
        size: [make_doc(rng, size) for _ in range(docs_per_size)] for size in sizes
    }


@dataclass
class BucketResult:
    target_bytes: int
    docs: int
    total_bytes: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: float
    mb_per_s: float
    allowed_ms: float

    @property
    def ok(self) -> bool:
        return self.p95_ms <= self.allowed_ms

    def to_dict(self) -> dict:
        return {
            "target_bytes": self.target_bytes,
            "docs": self.docs,
            "mean_ms": round(self.mean_ms, 3),
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "mb_per_s": round(self.mb_per_s, 2),
            "allowed_ms": round(self.allowed_ms, 3),
            "ok": self.ok,
        }


@dataclass
class BenchReport:
    seed: int
    budget_ms: float
    buckets: list[BucketResult] = field(default_factory=list)
    peak_rss_mb: float | None = None
    mode: str = "fast"

    @property
    def ok(self) -> bool:
        return all(b.ok for b in self.buckets)

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "mode": self.mode,
            "budget_ms_per_10kb": self.budget_ms,
            "buckets": [b.to_dict() for b in self.buckets],
            "peak_rss_mb": self.peak_rss_mb,
            "ok": self.ok,
        }

    def render(self) -> str:
        lines = [f"Single-core benchmark, {self.mode} mode (seed {self.seed})"]
        lines.append(
            f"  {'bucket':>8} {'docs':>5} {'mean':>8} {'p50':>8} {'p95':>8}"
            f" {'max':>8} {'MB/s':>7} {'allowed':>8}  verdict"
        )
        for b in self.buckets:
            lines.append(
                f"  {b.target_bytes // 1000:>6}KB {b.docs:>5}"
                f" {b.mean_ms:>6.2f}ms {b.p50_ms:>6.2f}ms {b.p95_ms:>6.2f}ms"
                f" {b.max_ms:>6.2f}ms {b.mb_per_s:>7.1f} {b.allowed_ms:>6.2f}ms"
                f"  {'PASS' if b.ok else 'FAIL'}"
            )
        if self.peak_rss_mb is not None:
            lines.append(f"  peak RSS: {self.peak_rss_mb:.0f} MB")
        lines.append(
            f"  budget: {self.budget_ms:.1f} ms per {TYPICAL_BYTES // 1000} KB document"
            f" (p95) -> {'PASS' if self.ok else 'FAIL'}"
        )
        return "\n".join(lines)


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    index = max(0, round(0.95 * len(ordered)) - 1)
    return ordered[index]


def _peak_rss_mb() -> float | None:
    try:
        import resource
    except ImportError:  # non-POSIX platform
        return None
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def run(
    seed: int = 7,
    sizes: tuple[int, ...] = DEFAULT_SIZES,
    docs_per_size: int = 30,
    budget_ms: float | None = None,
    mode: str = "fast",
    scan=None,
) -> BenchReport:
    """Time one document at a time on this core.

    `mode` selects the pipeline and, unless `budget_ms` overrides it, the
    budget: see MODE_BUDGET_MS. `scan` overrides the callable outright, which
    is how a caller benchmarks a non-default Stage 2 configuration.
    """
    if budget_ms is None:
        budget_ms = MODE_BUDGET_MS[mode]
    if scan is None:
        if mode == "deep":
            from .pipeline import deep_pipeline

            pipeline = deep_pipeline()
            def scan(text):
                return scan_text(text, pipeline)
        else:
            scan = scan_text

    corpus = generate_docs(seed, sizes, docs_per_size)
    report = BenchReport(seed=seed, budget_ms=budget_ms, mode=mode)

    # Warm up regex caches, the ONNX session and the allocator before timing.
    # In deep mode this is not a nicety: session creation and the first run's
    # arena allocation cost tens of milliseconds, which would land entirely in
    # the first document's latency.
    for docs in corpus.values():
        scan(docs[0])
        break

    for size, docs in corpus.items():
        latencies: list[float] = []
        total_bytes = 0
        for doc in docs:
            start = time.perf_counter()
            scan(doc)
            latencies.append((time.perf_counter() - start) * 1000.0)
            total_bytes += len(doc)
        total_s = sum(latencies) / 1000.0
        report.buckets.append(
            BucketResult(
                target_bytes=size,
                docs=len(docs),
                total_bytes=total_bytes,
                mean_ms=statistics.fmean(latencies),
                p50_ms=statistics.median(latencies),
                p95_ms=_p95(latencies),
                max_ms=max(latencies),
                mb_per_s=(total_bytes / 1_000_000) / total_s if total_s else 0.0,
                allowed_ms=budget_ms * max(1.0, size / TYPICAL_BYTES),
            )
        )
    report.peak_rss_mb = _peak_rss_mb()
    return report
