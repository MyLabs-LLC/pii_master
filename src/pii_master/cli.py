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
from .ner import MODEL_DIR_ENV, ModelUnavailable


def _cmd_scan(args: argparse.Namespace) -> int:
    from .classify import default_pipeline
    from .pipeline import deep_pipeline

    if args.deep:
        # Built once for the whole invocation: an ONNX session costs tens of
        # milliseconds to create, so per-file construction would dominate a
        # multi-file scan. ModelUnavailable is deliberately not caught -- see
        # pipeline.deep_pipeline for why deep mode must not fall back to rules.
        pipeline = deep_pipeline(args.model_dir,
                                 min_confidence=args.min_confidence)
    else:
        pipeline = default_pipeline()

    results: list[dict] = []
    detected = False
    for path in args.paths:
        if path == "-":
            text = sys.stdin.read()
        else:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        report = scan_text(text, pipeline)
        detected = detected or bool(report.entities)
        results.append({"path": path, **report.to_dict()})

    print(json.dumps({"files": results}, indent=2 if args.pretty else None))
    return 1 if (args.fail_on_detect and detected) else 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from .evaluation import FUTURE_TYPES, compare_scores, evaluate, load_corpus

    if args.deep:
        from .pipeline import deep_pipeline

        pipeline = deep_pipeline(args.model_dir,
                                 min_confidence=args.min_confidence)
        # In deep mode nothing is "undetectable": the corpus's PERSON_NAME and
        # ADDRESS gold is exactly what the student is for, so a miss must be
        # triaged as a real recall failure and not excused.
        report = evaluate(load_corpus(args.paths),
                          scan=lambda text: scan_text(text, pipeline),
                          undetectable=frozenset())
    else:
        report = evaluate(load_corpus(args.paths), undetectable=FUTURE_TYPES)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.render())

    if args.save_scores:
        Path(args.save_scores).write_text(
            json.dumps(report.scores(), indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nwrote scores baseline: {args.save_scores}", file=sys.stderr)

    if args.fail_under:
        baseline = json.loads(Path(args.fail_under).read_text(encoding="utf-8"))
        drops = compare_scores(report.scores(), baseline)
        if drops:
            print("\nQUALITY REGRESSION vs " + args.fail_under, file=sys.stderr)
            for line in drops:
                print(f"  {line}", file=sys.stderr)
            return 1
        print(f"\nno regression vs {args.fail_under}", file=sys.stderr)
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    from .bench import run

    sizes = tuple(int(s) for s in args.sizes.split(","))
    report = run(
        seed=args.seed,
        sizes=sizes,
        docs_per_size=args.docs_per_size,
        budget_ms=args.budget_ms,
        mode="deep" if args.deep else "fast",
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
    scan.add_argument(
        "--deep",
        action="store_true",
        help="also run the Stage 2 NER student: finds names, addresses and "
             "cue-free identifiers that no rule can. Needs pii-master[ml] and "
             "a model artifact; slower than the default rules-only mode.",
    )
    scan.add_argument(
        "--model-dir",
        metavar="DIR",
        help=f"Stage 2 model directory (default: ${MODEL_DIR_ENV}, then the "
             "user cache, then training/artifacts)",
    )
    scan.add_argument(
        "--min-confidence",
        type=float,
        default=0.70,
        metavar="P",
        help="drop Stage 2 spans below this mean per-token probability "
             "(default: 0.70 — calibrated, so this is roughly the minimum "
             "probability that a span is exactly right); 0 disables the filter",
    )
    scan.set_defaults(func=_cmd_scan)

    ev = subparsers.add_parser(
        "eval", help="score the pipeline against a frozen gold corpus"
    )
    ev.add_argument(
        "paths", nargs="+", metavar="CORPUS", help="corpus .jsonl files"
    )
    ev.add_argument("--json", action="store_true", help="JSON instead of tables")
    ev.add_argument(
        "--fail-under",
        metavar="SCORES.json",
        help="exit 1 if any metric dropped below this committed baseline",
    )
    ev.add_argument(
        "--save-scores",
        metavar="SCORES.json",
        help="write the current scores as a new baseline (a deliberate act)",
    )
    ev.add_argument(
        "--deep",
        action="store_true",
        help="score the rules + Stage 2 cascade instead of rules only",
    )
    ev.add_argument("--model-dir", metavar="DIR", help="Stage 2 model directory")
    ev.add_argument("--min-confidence", type=float, default=0.70, metavar="P",
                    help="drop Stage 2 spans below this probability")
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
        "--deep",
        action="store_true",
        help="benchmark rules + the Stage 2 student against the 25 ms deep "
             "budget instead of rules-only against 5 ms",
    )
    bench.add_argument(
        "--budget-ms",
        type=float,
        default=None,
        help="p95 budget in ms per 10 KB document (default: 5 fast, 25 deep)",
    )
    bench.add_argument("--json", action="store_true", help="JSON instead of tables")
    bench.add_argument(
        "--fail-over-budget",
        action="store_true",
        help="exit with status 1 if any bucket exceeds its allowance",
    )
    bench.set_defaults(func=_cmd_bench)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ModelUnavailable as exc:
        # A traceback here would be noise: the cause is always a missing extra
        # or a missing artifact, and both have a one-line fix. Exit 2 rather
        # than 1 so a CI job can tell "deep mode is not set up" apart from
        # "PII was detected", which is what exit 1 means for scan.
        print(f"pii-master: {exc}", file=sys.stderr)
        return 2
