"""Command-line interface: scan files, score the frozen corpus, benchmark.

`scan --fail-on-detect` and `bench --fail-over-budget` are CI gates in the
style of a secret scanner and a performance-regression check respectively.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .classify import scan_text


def _cmd_scan(args: argparse.Namespace) -> int:
    results: list[dict] = []
    detected = False
    for path in args.paths:
        if path == "-":
            text = sys.stdin.read()
        else:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        report = scan_text(text)
        detected = detected or bool(report.entities)
        results.append({"path": path, **report.to_dict()})

    print(json.dumps({"files": results}, indent=2 if args.pretty else None))
    return 1 if (args.fail_on_detect and detected) else 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from .evaluation import evaluate, load_corpus

    report = evaluate(load_corpus(args.paths))
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.render())
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    from .bench import run

    sizes = tuple(int(s) for s in args.sizes.split(","))
    report = run(
        seed=args.seed,
        sizes=sizes,
        docs_per_size=args.docs_per_size,
        budget_ms=args.budget_ms,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.render())
    return 1 if (args.fail_over_budget and not report.ok) else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pii-master",
        description="Fast, CPU-friendly PII/PHI detection and document classification.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser("scan", help="scan text files for PII/PHI")
    scan.add_argument(
        "paths", nargs="+", metavar="PATH", help="files to scan, or - for stdin"
    )
    scan.add_argument("--pretty", action="store_true", help="indented JSON output")
    scan.add_argument(
        "--fail-on-detect",
        action="store_true",
        help="exit with status 1 if any entity is detected",
    )
    scan.set_defaults(func=_cmd_scan)

    ev = subparsers.add_parser(
        "eval", help="score the pipeline against a frozen gold corpus"
    )
    ev.add_argument(
        "paths", nargs="+", metavar="CORPUS", help="corpus .jsonl files"
    )
    ev.add_argument("--json", action="store_true", help="JSON instead of tables")
    ev.set_defaults(func=_cmd_eval)

    bench = subparsers.add_parser(
        "bench", help="single-core latency/throughput benchmark vs the 5 ms budget"
    )
    bench.add_argument("--seed", type=int, default=7)
    bench.add_argument("--docs-per-size", type=int, default=30)
    bench.add_argument(
        "--sizes",
        default="1000,10000,100000",
        help="comma-separated document sizes in bytes",
    )
    bench.add_argument(
        "--budget-ms",
        type=float,
        default=5.0,
        help="p95 budget in ms per 10 KB document",
    )
    bench.add_argument("--json", action="store_true", help="JSON instead of tables")
    bench.add_argument(
        "--fail-over-budget",
        action="store_true",
        help="exit with status 1 if any bucket exceeds its allowance",
    )
    bench.set_defaults(func=_cmd_bench)

    args = parser.parse_args(argv)
    return args.func(args)
