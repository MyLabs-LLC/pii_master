"""Round-1 arms: the original eight plus the per-label threshold candidate.

`decision/` stays exactly as the 1,000-trial run left it -- `verify.py` there is
the acceptance test that the policy object reproduces that run's published
decision over those eight arms, and adding a ninth would change the question it
asks. This writes a *new* arms file next to a new decision, against the same
frozen `policy.yaml` and `suite.json`.

The candidate's latency comes from `benchmarks/perlabel_latency.json` -- the
one-core run that timed it head-to-head with the champion on the same sample --
rather than from `read_depth.json`, which predates it.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent

spec = importlib.util.spec_from_file_location(
    "build_decision", PROJECT / "decision" / "build_decision.py"
)
bd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bd)

#: The ninth arm. Same fusion, same weights, same 1,000-character read window --
#: only the recall head's per-tag thresholds differ.
CANDIDATE = ("perlabel_v4", "Per-label gate-boundary thresholds, 1k", 1_000)


_ORIGINAL_LATENCY_ROWS = bd.latency_rows


def latency_rows_with_candidate() -> dict[tuple[str, int], dict]:
    rows = _ORIGINAL_LATENCY_ROWS()
    measured = json.loads(
        (PROJECT / "benchmarks" / "perlabel_latency.json").read_text(encoding="utf-8")
    )
    for row in measured["rows"]:
        rows[(row["family"], row["read_window_chars"])] = {
            "p95_ms": row["p95_ms"],
            "docs_per_s": row["docs_per_s"],
        }
    return rows


def main() -> int:
    bd.ARMS = list(bd.ARMS) + [CANDIDATE]
    bd.latency_rows = latency_rows_with_candidate

    suite = bd.build_suite()
    arms = bd.build_arms(suite)
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "arms.json").write_text(
        json.dumps(arms, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for arm in arms:
        lat = arm["metrics"].get("p95_latency_ms")
        print(
            f"{arm['name']:16} macro_f2={arm['metrics']['macro_f2']['value']:.4f} "
            f"p95={(lat or {}).get('value', float('nan')):.3f}ms scopes={len(arm['scopes'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
