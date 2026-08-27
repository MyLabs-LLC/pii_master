"""Score a Nemotron sample through the packaged deep cascade (scan_text)."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "eval" / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import nemotron_eval as ne  # noqa: E402
from gate_metrics import span_gate  # noqa: E402
from pii_master.classify import scan_text  # noqa: E402
from pii_master.crosswalk import to_entity_type  # noqa: E402
from pii_master.evaluation import TypeScore  # noqa: E402

CATALOGUE = (
    "ACCOUNT_NUMBER", "CREDIT_CARD", "DATE_DOB", "EMAIL", "HEALTH_PLAN_ID",
    "IP_ADDRESS", "MRN", "PHONE_US", "SSN", "URL", "US_DRIVER_LICENSE",
)


def main() -> int:
    import argparse
    import pyarrow.parquet as pq

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/home/lence/nemotron")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    files = sorted(Path(args.data_dir).glob(f"{args.split}-*.parquet"))
    table = pq.read_table(files[0], columns=["text", "spans", "locale"])
    rows = list(zip(
        table.column("text").to_pylist(),
        table.column("spans").to_pylist(),
        table.column("locale").to_pylist(),
    ))[: args.limit]
    exact: dict[str, TypeScore] = defaultdict(TypeScore)
    for text, raw, _locale in rows:
        gold: dict[str, set] = defaultdict(set)
        for span in ne.parse_spans(raw):
            mapped = to_entity_type(span["label"])
            if mapped is not None:
                gold[mapped.value].add((span["start"], span["end"]))
        report = scan_text(text, deep=True)
        pred: dict[str, set] = defaultdict(set)
        for entity in report.entities:
            pred[entity.type.value].add((entity.start, entity.end))
        for name in set(gold) | set(pred):
            g, p = gold.get(name, set()), pred.get(name, set())
            sc = exact[name]
            sc.gold += len(g)
            sc.tp += len(g & p)
            sc.fp += len(p - g)
            sc.fn += len(g - p)
    exact_d = {k: v.to_dict() for k, v in exact.items()}
    gate = span_gate(exact_d, catalogue=CATALOGUE)
    payload = {"n": len(rows), "gate": gate, "exact": exact_d}
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "n": len(rows),
        "macro_f2": gate["macro_f2"],
        "severity_recall_min": gate["severity_recall_min"],
        "f2_min": gate["f2_min"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
