from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from datetime import datetime, timezone


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_run(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_run(path: Path, run: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(run, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one command and append its full output to run.json")
    parser.add_argument("--run-json", default=str(Path(__file__).with_name("run.json")))
    parser.add_argument("--cwd", default=None)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    command = [sys.executable if token == "{PYTHON}" else token for token in command]
    run_path = Path(args.run_json).resolve()
    run = load_run(run_path)
    for stale in run.get("commands", []):
        if stale.get("status") == "running":
            stale["status"] = "aborted"
            stale["exit_code"] = None
            stale["finished_at"] = now()
            stale["output"] = stale.get("output", "") + "\nInterrupted before the command wrapper finalized this entry."
    entry = {
        "command": subprocess.list2cmdline(command),
        "started_at": now(),
        "status": "running",
        "exit_code": None,
        "output": "",
    }
    run.setdefault("commands", []).append(entry)
    save_run(run_path, run)
    process = subprocess.Popen(
        command,
        cwd=args.cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    output: list[str] = []
    assert process.stdout is not None
    try:
        for line in process.stdout:
            output.append(line)
            sys.stdout.write(line)
            sys.stdout.flush()
        returncode = process.wait()
    except KeyboardInterrupt:
        process.send_signal(signal.SIGINT)
        returncode = process.wait()
        latest = load_run(run_path)
        latest_entry = next(
            item for item in latest["commands"] if item["started_at"] == entry["started_at"]
        )
        latest_entry["status"] = "aborted"
        latest_entry["exit_code"] = returncode
        latest_entry["finished_at"] = now()
        latest_entry["output"] = "".join(output) + "\nInterrupted by user/operator."
        save_run(run_path, latest)
        raise
    # Reload before finalizing so child commands may safely append arms,
    # metrics, and artifacts to the same lifecycle run record.
    latest = load_run(run_path)
    latest_entry = next(
        item for item in latest["commands"] if item["started_at"] == entry["started_at"]
    )
    latest_entry["status"] = "success" if returncode == 0 else "failed"
    latest_entry["exit_code"] = returncode
    latest_entry["finished_at"] = now()
    latest_entry["output"] = "".join(output)
    save_run(run_path, latest)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
