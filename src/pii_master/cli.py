"""Command-line interface: scan files (or stdin) and print a JSON report.

Usable as a CI gate in the style of a secret scanner:
    pii-master scan build/output.txt --fail-on-detect
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .classify import scan_text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pii-master",
        description="Fast, CPU-friendly PII/PHI detection and document classification.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    scan = subparsers.add_parser("scan", help="scan text files for PII/PHI")
    scan.add_argument(
        "paths",
        nargs="+",
        metavar="PATH",
        help="files to scan, or - for stdin",
    )
    scan.add_argument(
        "--pretty", action="store_true", help="indented JSON output"
    )
    scan.add_argument(
        "--fail-on-detect",
        action="store_true",
        help="exit with status 1 if any entity is detected",
    )
    args = parser.parse_args(argv)

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
