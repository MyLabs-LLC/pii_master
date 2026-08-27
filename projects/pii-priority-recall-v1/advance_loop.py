"""Advance this project's durable full-loop state with final run evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from model_pipeline.loop_state import LoopRoute, LoopState


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("ship", "observe"))
    parser.add_argument(
        "route", choices=("continue_loop", "budget_exhausted")
    )
    args = parser.parse_args()

    path = Path(__file__).resolve().parent / "loop_state.json"
    state = LoopState.load(path)
    state.trials_used = 1000
    state.advance(
        stage=args.stage,
        route=LoopRoute(args.route),
        priority_gate_pass=True,
        latency_gate_pass=True,
        macro_f2_target=0.9,
        macro_f2=0.48345749382010117,
        macro_f2_target_pass=False,
        budget_trials=1000,
    )
    if args.route == "budget_exhausted":
        state.stopped_reason = (
            "Approved 1,000-trial budget exhausted; priority and latency gates "
            "pass, but macro-F2 target 0.90 was not reached."
        )
    state.save(path)


if __name__ == "__main__":
    main()
