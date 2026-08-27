"""Measure the Stage 1 rules engine on D_ho, D_in, and the frozen corpus."""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "eval" / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlflow  # noqa: E402
import nemotron_eval as ne  # noqa: E402
from gate_metrics import micro_prf, span_gate  # noqa: E402
from runjson import record_command, upsert_arm  # noqa: E402

from model_pipeline import evaluate as mp_evaluate  # noqa: E402
from model_pipeline import set_cpus, tracking  # noqa: E402
from model_pipeline.result import MetricValue  # noqa: E402
from pii_master.classify import DocumentClassifier, scan_text  # noqa: E402
from pii_master.pipeline import Pipeline  # noqa: E402
from pii_master.crosswalk import to_entity_type  # noqa: E402
from pii_master.evaluation import TypeScore  # noqa: E402
from pii_master.evaluation import evaluate as eval_frozen  # noqa: E402
from pii_master.evaluation import load_corpus  # noqa: E402
from pii_master.validators import luhn_ok  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "measure"
CATALOGUE = (
    "ACCOUNT_NUMBER", "CREDIT_CARD", "DATE_DOB", "EMAIL", "HEALTH_PLAN_ID",
    "IP_ADDRESS", "MRN", "PHONE_US", "SSN", "URL", "US_DRIVER_LICENSE",
)


def _load_parquet(data_dir: Path, split: str, limit: int | None):
    import pyarrow.parquet as pq

    files = sorted(data_dir.glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {split} parquet in {data_dir}")
    rows = []
    for path in files:
        table = pq.read_table(path, columns=["text", "spans", "locale"])
        rows.extend(
            zip(
                table.column("text").to_pylist(),
                table.column("spans").to_pylist(),
                table.column("locale").to_pylist(),
            )
        )
    if limit:
        rows = rows[:limit]
    return rows, [p.name for p in files]


def score_rows(rows):
    """One scan per doc: span TypeScores (nemotron protocol) + doc tag sets."""
    exact: dict[str, TypeScore] = defaultdict(TypeScore)
    y_true, y_pred = [], []
    card_gold = card_luhn = card_luhn_hit = 0
    docs_with_gold = flagged = 0
    pipeline = Pipeline()
    classifier = DocumentClassifier()
    started = time.perf_counter()
    for text, raw, _locale in rows:
        gold: dict[str, list[tuple[int, int]]] = defaultdict(list)
        luhn_spans: list[tuple[int, int]] = []
        for span in ne.parse_spans(raw):
            label = span["label"]
            mapped = to_entity_type(label)
            if label == "credit_debit_card":
                card_gold += 1
                digits = "".join(c for c in str(span.get("text", "")) if c.isdigit())
                if 13 <= len(digits) <= 19 and luhn_ok(digits):
                    card_luhn += 1
                    luhn_spans.append((span["start"], span["end"]))
            if mapped is not None:
                gold[mapped.value].append((span["start"], span["end"]))
        report = classifier.classify(text, pipeline.run(text))
        pred: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for entity in report.entities:
            pred[entity.type.value].append((entity.start, entity.end))
        if gold:
            docs_with_gold += 1
            if report.entities:
                flagged += 1
        found_cards = set(pred.get("CREDIT_CARD", []))
        card_luhn_hit += sum(1 for span in luhn_spans if span in found_cards)
        for entity_type in set(gold) | set(pred):
            g = set(gold.get(entity_type, []))
            p = set(pred.get(entity_type, []))
            sc = exact[entity_type]
            sc.gold += len(g)
            sc.tp += len(g & p)
            sc.fp += len(p - g)
            sc.fn += len(g - p)
        y_true.append(set(gold))
        y_pred.append([entity.type.value for entity in report.entities])
    elapsed = time.perf_counter() - started
    return {
        "exact": dict(exact),
        "y_true": y_true,
        "y_pred": y_pred,
        "card_gold": card_gold,
        "card_luhn": card_luhn,
        "card_luhn_hit": card_luhn_hit,
        "documents": len(rows),
        "documents_with_mapped_gold": docs_with_gold,
        "documents_flagged": flagged,
        "ms_per_doc": (elapsed / max(len(rows), 1)) * 1000.0,
        "elapsed_s": elapsed,
    }


def _exact_dicts(exact) -> dict:
    return {k: v.to_dict() for k, v in exact.items()}


def attach_span_gate(res, gate: dict) -> None:
    p, r, f = micro_prf(gate)
    extras = {
        "macro_f2": gate["macro_f2"],
        "severity_recall_min": gate["severity_recall_min"]
        if gate["severity_recall_min"] is not None else float("nan"),
        "severity_recall_mean": gate["severity_recall_mean"]
        if gate["severity_recall_mean"] is not None else float("nan"),
        "f2_min": gate["f2_min"],
        "f2_median": gate["f2_median"],
        "n_tags_f2_zero": float(gate["n_tags_f2_zero"]),
        "n_tags_f2_below_10pct": float(gate["n_tags_f2_below_10pct"]),
        "span_precision_micro": p,
        "span_recall_micro": r,
        "span_f1_micro": f,
    }
    for name, value in extras.items():
        res.add(MetricValue(
            name=name, value=float(value),
            greater_is_better=not name.startswith("n_tags"),
        ))
    res.primary_metric = "macro_f2"
    res.tags["severity_tags_measured"] = gate["severity_tags_measured"]
    res.tags["severity_tags_excluded"] = gate["severity_tags_excluded"]
    res.tags["per_tag_f2_worst_first"] = gate["per_tag"][:15]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def measure_nemotron(split: str, data_dir: Path, limit: int | None, run_name: str):
    rows, files = _load_parquet(data_dir, split, limit)
    result = score_rows(rows)
    exact = _exact_dicts(result["exact"])
    gate = span_gate(
        exact,
        card_luhn=result["card_luhn"],
        card_luhn_hit=result["card_luhn_hit"],
        catalogue=CATALOGUE,
    )
    tagging = mp_evaluate(
        "tagging",
        y_true=result["y_true"],
        y_pred=result["y_pred"],
        k_values=[1, 3, 5],
        require="all",
        primary_k=5,
        bootstrap=True,
        n_resamples=200,
        seed=0,
        dataset={"name": f"nemotron-pii/{split}", "source": str(data_dir), "files": files},
        params={"model": "rules_stage1", "split": split, "n": str(len(rows))},
    )
    attach_span_gate(tagging, gate)
    out_path = OUT / f"rules_{split}.json"
    payload = {
        "model": "rules_stage1",
        "dataset": f"nemotron-{split}",
        "n_samples": len(rows),
        "ms_per_doc": result["ms_per_doc"],
        "gate": gate,
        "exact": exact,
        "card_gold": result["card_gold"],
        "card_luhn": result["card_luhn"],
        "card_luhn_hit": result["card_luhn_hit"],
        "documents_flagged": result["documents_flagged"],
        "documents_with_mapped_gold": result["documents_with_mapped_gold"],
        "evaluation": tagging.to_dict(),
    }
    # Drop per-doc vectors from the on-disk dump — 100k sets is huge and unused.
    write_json(out_path, payload)
    with tracking.evaluation_run(
        run_name, experiment="pii-stage2",
        tags={"model": "rules_stage1", "split": split},
    ):
        tracking.log_evaluation(tagging)
        mlflow.log_artifact(str(out_path), artifact_path="evaluation")
    p, r, f = micro_prf(gate)
    upsert_arm({
        "model": "rules_stage1",
        "dataset": f"nemotron-{split}",
        "n_samples": len(rows),
        "latency_ms_per_doc": round(result["ms_per_doc"], 4),
        "verdict": "baseline",
        "metrics": {
            "macro_f2": {"value": gate["macro_f2"]},
            "severity_recall_min": {"value": gate["severity_recall_min"]},
            "f1_micro": {"value": f},
            "precision_micro": {"value": p},
            "recall_micro": {"value": r},
            "f2_min": {"value": gate["f2_min"]},
        },
        "evaluation": tagging.to_dict(),
    })
    return payload


def measure_frozen():
    docs = load_corpus([REPO / "eval" / "corpus" / "frozen_v1.jsonl"])
    report = eval_frozen(docs)
    y_true = [doc.label for doc in docs]
    y_pred = [scan_text(doc.text).label.name for doc in docs]
    clf = mp_evaluate(
        "classification",
        y_true=y_true,
        y_pred=y_pred,
        average="macro",
        bootstrap=True,
        n_resamples=200,
        seed=0,
        dataset={"name": "frozen_v1"},
        params={"model": "rules_stage1"},
    )
    payload = report.to_dict()
    payload["evaluation"] = clf.to_dict()
    write_json(OUT / "rules_frozen.json", payload)
    with tracking.evaluation_run(
        "rules-frozen", experiment="pii-stage2",
        tags={"model": "rules_stage1", "split": "frozen"},
    ):
        tracking.log_evaluation(clf)
        mlflow.log_artifact(str(OUT / "rules_frozen.json"), artifact_path="evaluation")
    upsert_arm({
        "model": "rules_stage1",
        "dataset": "frozen_v1",
        "n_samples": report.doc_count,
        "verdict": "regression test — tautological",
        "metrics": {
            "accuracy": {"value": report.doc_accuracy},
            "phi_recall": {"value": report.phi_recall},
        },
        "evaluation": clf.to_dict(),
    })
    return payload


def main() -> int:
    set_cpus(1)
    OUT.mkdir(parents=True, exist_ok=True)
    data_dir = Path("/home/lence/nemotron")
    t0 = time.perf_counter()
    ho = measure_nemotron("test", data_dir, None, "rules-D_ho")
    t1 = time.perf_counter()
    record_command(
        "measure_rules.py D_ho (Nemotron test, 100k)",
        output=json.dumps({
            "macro_f2": ho["gate"]["macro_f2"],
            "severity_recall_min": ho["gate"]["severity_recall_min"],
            "span_f1_micro": micro_prf(ho["gate"])[2],
            "n": ho["n_samples"],
            "ms_per_doc": ho["ms_per_doc"],
        }),
        context="measure D_ho rules",
        cwd=str(REPO),
        duration_s=t1 - t0,
    )
    inn = measure_nemotron("train", data_dir, None, "rules-D_in")
    t2 = time.perf_counter()
    record_command(
        "measure_rules.py D_in (Nemotron train, 100k)",
        output=json.dumps({
            "macro_f2": inn["gate"]["macro_f2"],
            "severity_recall_min": inn["gate"]["severity_recall_min"],
            "n": inn["n_samples"],
        }),
        context="measure D_in rules",
        cwd=str(REPO),
        duration_s=t2 - t1,
    )
    frozen = measure_frozen()
    t3 = time.perf_counter()
    record_command(
        "measure_rules.py frozen_v1",
        output=json.dumps({
            "doc_accuracy": frozen.get("doc_accuracy"),
            "phi_recall": frozen.get("phi_recall"),
        }),
        context="measure frozen rules",
        cwd=str(REPO),
        duration_s=t3 - t2,
    )
    print(json.dumps({
        "D_ho": {
            "macro_f2": ho["gate"]["macro_f2"],
            "severity_recall_min": ho["gate"]["severity_recall_min"],
            "n": ho["n_samples"],
        },
        "D_in": {
            "macro_f2": inn["gate"]["macro_f2"],
            "severity_recall_min": inn["gate"]["severity_recall_min"],
            "n": inn["n_samples"],
        },
        "frozen": {
            "doc_accuracy": frozen.get("doc_accuracy"),
            "phi_recall": frozen.get("phi_recall"),
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
