"""Rebuild this project's shipping decision from the evidence it already wrote.

The 1,000-trial run selected its champion with a ladder that existed only as
project code and a paragraph of prose:

    55 hard per-tag recall gates (point AND 95% bootstrap lower bound >= 0.90)
      -> equal-corpus macro F2
      -> micro F1
      -> one-core p95 latency under the 5 ms aspiration

This script expresses that ladder as a `model_pipeline.DecisionSpec` over a
`model_pipeline.EvalSuite`, reads the same per-corpus evidence files the run
produced, and asks the policy to pick a winner. If the object is right, it
picks the same artifact and reproduces the same published numbers — that is
the acceptance test, and `verify.py` beside this file asserts it.

Writes, next to this file: `suite.json`, `policy.yaml`, `arms.json`.
"""

from __future__ import annotations

import json
from pathlib import Path

from model_pipeline.decision import Constraint, DecisionSpec, Preference
from model_pipeline.suite import Corpus, DataQuality, EvalSuite

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
EVALS = PROJECT / "evaluations"
READ_DEPTH = PROJECT / "benchmarks" / "read_depth.json"

#: The eight measured arms, as the run's report names them, and where each
#: one's evidence lives. `depth` picks the latency row out of read_depth.json.
ARMS = [
    ("current_rules", "Current rules baseline", 20_000),
    ("hash_sgd", "Hash recall-max, 20k", 20_000),
    ("hash_sgd_f2", "Hash F2, 20k", 20_000),
    ("tfidf_linear", "TF-IDF linear, 20k", 20_000),
    ("embeddingbag_asl", "EmbeddingBag ASL, 20k", 20_000),
    ("hybrid_priority_001", "Two-head priority hybrid, 20k", 20_000),
    ("hybrid_priority", "Full per-label fusion, 20k", 20_000),
    ("champion_1k", "Selected full fusion, 1k", 1_000),
]

#: champion_1k is the 1,000-character read of the hybrid_priority fusion, so
#: its latency row lives under that family's name.
LATENCY_FAMILY = {"champion_1k": "hybrid_priority"}

#: The p95 aspiration this scanner is built around: one core, 5 ms per document.
LATENCY_BUDGET_MS = 5.0
PRIORITY_RECALL_GATE = 0.90
MIN_PRIORITY_SUPPORT = 30


def corpus_files(family: str) -> list[Path]:
    """Prefer the bootstrapped copies — they are the ones carrying CIs."""
    boot = EVALS / family / "bootstrap"
    root = boot if boot.is_dir() else EVALS / family
    return sorted(p for p in root.glob("*.json") if p.name != "summary.json")


def build_suite() -> EvalSuite:
    """One corpus per evaluation file, with its gold mode read from the evidence.

    `label_complete: false` in these files means *known-positive recall only* —
    the corpus can measure recall and cannot measure precision. That is
    `gold="positive_only"`, and it is why three of the eight rows in the
    published report carry `N/A` where a macro F2 would be.
    """
    corpora = []
    for path in corpus_files("champion_1k"):
        d = json.loads(path.read_text())
        name = d["dataset"]
        corpora.append(
            Corpus(
                name=name,
                path=str(path.relative_to(PROJECT)),
                role=_role_for(name),
                gold="complete" if d["label_complete"] else "positive_only",
                n=d["n_rows"],
                data_quality=DataQuality(
                    leakage=0,
                    leakage_scanned=True,
                    real_synth_mix=_mix_for(name),
                    note="carried forward from the 1,000-trial run's corpus review",
                ),
            )
        )
    return EvalSuite(name="pii-priority-recall", corpora=tuple(corpora),
                     aggregation="equal_corpus")


def _role_for(name: str) -> str:
    if name.startswith("pii_holdout"):
        return "sealed"
    if "dualjudge" in name:
        return "adversarial"
    return "external"


def _mix_for(name: str) -> str:
    if "govdocs2" in name:
        return "real business documents"
    if "nemotron" in name or "ai4privacy" in name or "betterdataai" in name:
        return "synthetic"
    return "mixed"


def latency_rows() -> dict[tuple[str, int], dict]:
    rows = json.loads(READ_DEPTH.read_text())["rows"]
    return {(r["family"], r["read_depth_chars"]): r for r in rows}


def build_arms(suite: EvalSuite) -> list[dict]:
    """One arm per measured model, carrying arm-level and per-tag measurements.

    Arm level: the equal-corpus headline numbers and the one-core p95.
    Scoped:    `<tag>@<corpus>` recall with its support and bootstrap bound —
               the 128 candidate gates of which 55 turn out measurable.
    """
    lat = latency_rows()
    arms = []
    for family, label, depth in ARMS:
        summary = json.loads((EVALS / family / "summary.json").read_text())["aggregate"]
        metrics: dict[str, object] = {
            "macro_f2": {
                "value": summary["equal_corpus_macro_f2"],
                "ci_low": summary.get("equal_corpus_macro_f2_ci", {}).get("ci_low"),
                "ci_high": summary.get("equal_corpus_macro_f2_ci", {}).get("ci_high"),
            },
            "micro_f1": {
                "value": summary["equal_corpus_micro_f1"],
                "ci_low": summary.get("equal_corpus_micro_f1_ci", {}).get("ci_low"),
                "ci_high": summary.get("equal_corpus_micro_f1_ci", {}).get("ci_high"),
            },
        }
        row = lat.get((LATENCY_FAMILY.get(family, family), depth))
        if row is not None:
            metrics["p95_latency_ms"] = {
                "value": row["p95_ms"], "greater_is_better": False,
            }
            metrics["docs_per_s"] = row["docs_per_s"]

        scopes: dict[str, dict] = {}
        for path in corpus_files(family):
            d = json.loads(path.read_text())
            corpus = d["dataset"]
            for tag, p in d["priority"].items():
                scopes[f"{tag}@{corpus}"] = {
                    "recall": {
                        "value": p["recall"],
                        "ci_low": p.get("ci_low"),
                        "ci_high": p.get("ci_high"),
                        "support": p["support"],
                    }
                }
        arms.append({
            "name": family,
            "label": label,
            "read_depth_chars": depth,
            "metrics": metrics,
            "scopes": scopes,
        })
    return arms


def build_policy() -> DecisionSpec:
    """The published selection ladder, as constraints plus a preference order.

    Two hard constraints and two preferences replace a paragraph:

    * **priority recall, conclusively.** `basis="ci_lower"` is what "and every
      95% lower bound was at least 0.90" means. An arm that has the point
      estimate but never ran the bootstrap is `NOT_MEASURABLE` here — which is
      exactly the `POINT_PASS_UNVERIFIED` status this project had already
      invented for itself.
    * **the latency budget, at selection time.** The report had to warn in
      prose that the 10k and 20k passes "must not be described as meeting"
      the 5 ms aspiration. As a constraint they simply do not win.
    """
    return DecisionSpec(
        name="pii-priority-recall-v1",
        constraints=(
            Constraint(
                name="priority tag recall ≥ 0.90 (conclusive)",
                metric="recall", op=">=", threshold=PRIORITY_RECALL_GATE,
                scope="*@*", basis="ci_lower", min_support=MIN_PRIORITY_SUPPORT,
                severity="hard",
            ),
            Constraint(
                name="one-core p95 ≤ 5 ms",
                metric="p95_latency_ms", op="<=", threshold=LATENCY_BUDGET_MS,
                severity="hard",
            ),
        ),
        preferences=(
            Preference(metric="macro_f2", min_relative_change=0.0),
            Preference(metric="micro_f1"),
        ),
    )


def main() -> None:
    suite = build_suite()
    policy = build_policy()
    arms = build_arms(suite)

    suite.save(HERE / "suite.json")
    (HERE / "arms.json").write_text(json.dumps({"arms": arms}, indent=2) + "\n")
    try:
        import yaml

        (HERE / "policy.yaml").write_text(
            yaml.safe_dump(policy.to_dict(), sort_keys=False)
        )
    except ModuleNotFoundError:  # pragma: no cover
        (HERE / "policy.json").write_text(json.dumps(policy.to_dict(), indent=2) + "\n")

    print(f"suite:  {len(suite)} corpora "
          f"({len(suite.measurable('macro_f2'))} can measure macro F2)")
    print(f"policy: {len(policy.constraints)} constraints, "
          f"{len(policy.preferences)} preferences")
    print(f"arms:   {len(arms)}")
    print(f"written to {HERE}")


if __name__ == "__main__":
    main()
