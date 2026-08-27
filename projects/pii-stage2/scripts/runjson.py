"""Keep run.json open from the first command. Output cannot be reconstructed later."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUN_JSON = ROOT / "run.json"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load() -> dict[str, Any]:
    return json.loads(RUN_JSON.read_text(encoding="utf-8"))


def save(data: dict[str, Any]) -> None:
    RUN_JSON.parent.mkdir(parents=True, exist_ok=True)
    RUN_JSON.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")


def init_run() -> dict[str, Any]:
    data = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "project": "pii-stage2",
        "track": "PII/PHI",
        "tier": "n/a",
        "what_changed": "measure existing rules + trained CNN students; then loop",
        "verdict": "in progress",
        "commands": [],
        "arms": [],
        "data_quality": [
            {
                "dataset": "nvidia/Nemotron-PII@test",
                "leakage": 0,
                "notes": "synthetic, CC BY 4.0; 100k docs nobody here authored; sealed D_ho",
                "real_synth_mix": "100% synthetic",
            },
            {
                "dataset": "nvidia/Nemotron-PII@train",
                "leakage": 0,
                "notes": "D_in; proposer-visible; not the gate",
                "real_synth_mix": "100% synthetic",
            },
            {
                "dataset": "eval/corpus/frozen_v1.jsonl",
                "leakage": 1,
                "notes": "authored alongside detectors — tautological regression test, not a quality claim",
                "n_docs": 39,
            },
        ],
        "artifacts": {},
    }
    save(data)
    return data


def record_command(
    command: str,
    output: str = "",
    exit_code: int = 0,
    context: str = "",
    cwd: str = "",
    duration_s: float | None = None,
) -> None:
    data = load()
    data.setdefault("commands", []).append(
        {
            "command": command,
            "output": output[-200_000:] if output else "",
            "exit_code": exit_code,
            "timestamp": _now(),
            "context": context,
            "cwd": cwd,
            "duration_s": duration_s,
        }
    )
    save(data)


def upsert_arm(arm: dict[str, Any]) -> None:
    data = load()
    arms = data.setdefault("arms", [])
    key = (arm.get("model"), arm.get("dataset"))
    for i, existing in enumerate(arms):
        if (existing.get("model"), existing.get("dataset")) == key:
            arms[i] = {**existing, **arm}
            save(data)
            return
    arms.append(arm)
    save(data)
