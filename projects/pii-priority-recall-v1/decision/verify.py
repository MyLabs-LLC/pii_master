"""Does the decision object reproduce the run's published result?

The 1,000-trial run's report states six numbers. They were produced by project
code and a hand-maintained ladder. This asks the same question of the policy
object and checks the answers match exactly — including *which artifact wins*.

A mismatch is a finding about the object, not a tolerance to widen.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from model_pipeline.decision import DecisionSpec, Verdict, load_arms
from model_pipeline.suite import load_suite

HERE = Path(__file__).resolve().parent

#: Straight from reports/26-08-25_priority-recall-1000-run.md.
PUBLISHED = {
    "selected": "champion_1k",
    "measurable_gates": 55,
    "passed_gates": 55,
    "worst_priority_recall": 0.9887766554433222,
    "lowest_ci_bound": 0.9811,          # the report quotes 4 decimal places
    "equal_corpus_macro_f2": 0.48345749382010117,
    "equal_corpus_micro_f1": 0.38118396080534256,
    "p95_latency_ms": 2.200,
}


def close(a: float | None, b: float, tol: float) -> bool:
    return a is not None and abs(a - b) <= tol


def main() -> int:
    suite = load_suite(HERE / "suite.json")
    policy = DecisionSpec.from_dict(_policy_dict())
    arms = load_arms([HERE / "arms.json"])

    result = policy.decide(arms)
    winner = result.selected_decision
    gates = winner.constraints[0] if winner else None
    scopes = gates.scopes if gates else ()
    measurable = [s for s in scopes if s.verdict is not Verdict.NOT_MEASURABLE]
    passed = [s for s in scopes if s.verdict is Verdict.PASS]

    checks: list[tuple[str, bool, str]] = [
        ("selected artifact", result.selected == PUBLISHED["selected"],
         f"{result.selected} (published: {PUBLISHED['selected']})"),
        ("measurable priority gates", len(measurable) == PUBLISHED["measurable_gates"],
         f"{len(measurable)} of {len(scopes)} candidate tag×corpus pairs"),
        ("gates passed", len(passed) == PUBLISHED["passed_gates"],
         f"{len(passed)}/{len(measurable)}"),
        ("worst priority recall",
         close(min((s.value for s in measurable), default=None),
               PUBLISHED["worst_priority_recall"], 1e-9),
         f"{min((s.value for s in measurable), default=float('nan')):.4f}"),
        ("lowest 95% bound",
         close(min((s.compared for s in measurable), default=None),
               PUBLISHED["lowest_ci_bound"], 5e-5),
         f"{min((s.compared for s in measurable), default=float('nan')):.4f}"),
        ("equal-corpus macro F2",
         close(result.metric("macro_f2"), PUBLISHED["equal_corpus_macro_f2"], 1e-12),
         f"{result.metric('macro_f2'):.4f}"),
        ("equal-corpus micro F1",
         close(result.metric("micro_f1"), PUBLISHED["equal_corpus_micro_f1"], 1e-12),
         f"{result.metric('micro_f1'):.4f}"),
        ("one-core p95",
         close(result.metric("p95_latency_ms"), PUBLISHED["p95_latency_ms"], 5e-4),
         f"{result.metric('p95_latency_ms'):.3f} ms"),
    ]

    width = max(len(name) for name, _, _ in checks)
    print(f"\nReproducing reports/26-08-25_priority-recall-1000-run.md\n")
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")

    print("\nHow the policy got there:")
    for line in result.reason:
        print(f"  - {line}")

    print("\nPer-arm feasibility:")
    for a in result.arms:
        hard = a.constraints[0]
        lat = a.constraints[1]
        note = []
        if hard.verdict is Verdict.FAIL:
            note.append(f"{hard.n_pass}/{hard.n_measurable} priority gates")
        elif hard.verdict is Verdict.NOT_MEASURABLE:
            note.append("priority gates unverified (no bootstrap)")
        if lat.verdict is Verdict.FAIL:
            note.append(f"p95 {lat.scopes[0].value:.3f} ms over budget")
        elif lat.verdict is Verdict.NOT_MEASURABLE:
            note.append("p95 not measured")
        print(f"  {'ok  ' if a.feasible else 'NO  '} {a.arm:<22} "
              + ("; ".join(note) if note else f"{hard.n_pass}/{hard.n_measurable} gates, "
                 f"p95 {lat.scopes[0].value:.3f} ms"))

    print()
    print(suite.to_markdown())
    print()
    print(result.to_markdown())

    failed = [n for n, ok, _ in checks if not ok]
    if failed:
        print(f"\n{len(failed)} check(s) did NOT reproduce: {', '.join(failed)}")
        return 1
    print(f"\nAll {len(checks)} published numbers reproduced from the policy object.")
    return 0


def _policy_dict() -> dict:
    p = HERE / "policy.yaml"
    if p.exists():
        import yaml

        return yaml.safe_load(p.read_text())
    return json.loads((HERE / "policy.json").read_text())


if __name__ == "__main__":
    sys.exit(main())
