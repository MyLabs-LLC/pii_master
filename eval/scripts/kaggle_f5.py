"""Official-style micro F5 on the Kaggle 2024 PII Detection public train set.

Competition: The Learning Agency Lab - PII Data Detection
Metric: micro-averaged F-beta (beta=5) over (document, token, BIO-label)
triples, excluding O. 1st place private LB was ~0.974 with a DeBERTa-large
ensemble; this scores *this* CNN student on the public 6,807-essay train
split (hidden test is unavailable). A 20% document holdout (seed 42) is
reported so we do not quote a train-set number as a leaderboard score.

Backends:
  student  native 55-label CNN via training/eval_student.predict (GPU)
  deep     packaged scan_text(deep=True) HIPAA types mapped to Kaggle
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "training"))
sys.path.insert(0, str(REPO / "src"))

from pii_master.fusion import snap_to_word_bounds  # noqa: E402

NATIVE_TO_KAGGLE = {
    "first_name": "NAME_STUDENT",
    "last_name": "NAME_STUDENT",
    "email": "EMAIL",
    "user_name": "USERNAME",
    "phone_number": "PHONE_NUM",
    "fax_number": "PHONE_NUM",
    "url": "URL_PERSONAL",
    "street_address": "STREET_ADDRESS",
    "city": "STREET_ADDRESS",
    "county": "STREET_ADDRESS",
    "postcode": "STREET_ADDRESS",
    "ssn": "ID_NUM",
    "unique_id": "ID_NUM",
    "customer_id": "ID_NUM",
    "employee_id": "ID_NUM",
    "account_number": "ID_NUM",
    "medical_record_number": "ID_NUM",
    "national_id": "ID_NUM",
    "certificate_license_number": "ID_NUM",
    "tax_id": "ID_NUM",
}

HIPAA_TO_KAGGLE = {
    "PERSON_NAME": "NAME_STUDENT",
    "EMAIL": "EMAIL",
    "USERNAME": "USERNAME",
    "PHONE_US": "PHONE_NUM",
    "FAX_NUMBER": "PHONE_NUM",
    "URL": "URL_PERSONAL",
    "ADDRESS": "STREET_ADDRESS",
    "SSN": "ID_NUM",
    "ACCOUNT_NUMBER": "ID_NUM",
    "MRN": "ID_NUM",
    "US_DRIVER_LICENSE": "ID_NUM",
    "HEALTH_PLAN_ID": "ID_NUM",
}

KAGGLE_TYPES = (
    "NAME_STUDENT", "EMAIL", "USERNAME", "ID_NUM",
    "PHONE_NUM", "URL_PERSONAL", "STREET_ADDRESS",
)
_GAP = frozenset(" \t\n\r,;.")


def token_offsets(tokens: list[str], trailing_whitespace: list[bool]) -> list[tuple[int, int]]:
    offsets = []
    pos = 0
    for tok, space in zip(tokens, trailing_whitespace):
        offsets.append((pos, pos + len(tok)))
        pos += len(tok) + (1 if space else 0)
    return offsets


def gold_triples(doc: dict) -> set[tuple[int, int, str]]:
    out = set()
    for i, label in enumerate(doc["labels"]):
        if label != "O":
            out.add((int(doc["document"]), i, label))
    return out


def merge_spans(
    spans: list[tuple[str, int, int]],
    text: str,
    max_gap: int = 3,
) -> list[tuple[str, int, int]]:
    if len(spans) < 2:
        return list(spans)
    ordered = sorted(spans, key=lambda s: (s[1], s[2], s[0]))
    merged = [ordered[0]]
    for typ, start, end in ordered[1:]:
        ptyp, pstart, pend = merged[-1]
        gap = start - pend
        if (
            typ == ptyp
            and 0 <= gap <= max_gap
            and (gap == 0 or all(ch in _GAP for ch in text[pend:start]))
        ):
            merged[-1] = (ptyp, pstart, max(pend, end))
        else:
            merged.append((typ, start, end))
    snapped: list[tuple[str, int, int]] = []
    for typ, start, end in merged:
        if typ in {"NAME_STUDENT", "STREET_ADDRESS", "USERNAME"}:
            start, end = snap_to_word_bounds(text, start, end)
            if start >= end:
                continue
        snapped.append((typ, start, end))
    return snapped


def spans_to_triples(
    document: int,
    offsets: list[tuple[int, int]],
    spans: list[tuple[str, int, int]],
) -> set[tuple[int, int, str]]:
    labels = ["O"] * len(offsets)
    for typ, start, end in spans:
        first = True
        for i, (a, b) in enumerate(offsets):
            if a < end and start < b:
                labels[i] = ("B-" if first else "I-") + typ
                first = False
    return {(document, i, lab) for i, lab in enumerate(labels) if lab != "O"}


def fbeta(tp: int, fp: int, fn: int, beta: float = 5.0) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    b2 = beta * beta
    denom = b2 * precision + recall
    f = ((1 + b2) * precision * recall / denom) if denom else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f5": f,
    }


def score_sets(pred: set, gold: set, beta: float = 5.0) -> dict:
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    out = fbeta(tp, fp, fn, beta)
    by_type: dict[str, dict] = {}
    for typ in KAGGLE_TYPES:
        g = {t for t in gold if t[2].endswith(typ)}
        p = {t for t in pred if t[2].endswith(typ)}
        by_type[typ] = fbeta(len(p & g), len(p - g), len(g - p), beta)
        by_type[typ]["gold"] = len(g)
        by_type[typ]["pred"] = len(p)
    out["by_type"] = by_type
    return out


def split_indices(n: int, seed: int = 42, holdout: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_ho = max(1, int(round(holdout * n)))
    return np.sort(idx[n_ho:]), np.sort(idx[:n_ho])


def native_spans(preds: list[tuple[str, int, int]], text: str) -> list[tuple[str, int, int]]:
    mapped = []
    for label, start, end in preds:
        kaggle = NATIVE_TO_KAGGLE.get(label)
        if kaggle is None:
            continue
        mapped.append((kaggle, start, end))
    return merge_spans(mapped, text)


def hipaa_spans(entities, text: str) -> list[tuple[str, int, int]]:
    mapped = []
    for entity in entities:
        kaggle = HIPAA_TO_KAGGLE.get(entity.type.value)
        if kaggle is None:
            continue
        mapped.append((kaggle, entity.start, entity.end))
    return merge_spans(mapped, text)


def main() -> int:
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="/home/lence/pii-stage2-runs/kaggle/train.json")
    ap.add_argument("--checkpoint", default=str(REPO / "training" / "artifacts" / "student_m.pt"))
    ap.add_argument("--size", default="m")
    ap.add_argument("--backend", choices=("student", "deep"), default="student")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--o-scale", type=float, default=1.0,
                    help="multiply O-class softmax before argmax (Kaggle F5 recall bias)")
    args = ap.parse_args()

    docs = json.loads(Path(args.data).read_text(encoding="utf-8"))
    if args.limit:
        docs = docs[: args.limit]
    n = len(docs)
    in_idx, ho_idx = split_indices(n, seed=args.seed)
    texts = [d["full_text"] for d in docs]
    print(f"kaggle public train: {n} essays; D_in {len(in_idx)} D_ho {len(ho_idx)} "
          f"backend={args.backend} device={args.device} o_scale={args.o_scale}", flush=True)

    pred_spans: list[list[tuple[str, int, int]]] = [[] for _ in docs]
    if args.backend == "student":
        from eval_student import TEACHER_ID, load_student, predict
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(TEACHER_ID)
        student = load_student(Path(args.checkpoint), args.size)
        raw = predict(
            student, tokenizer, texts, torch.device(args.device), args.batch_size,
            o_scale=args.o_scale,
        )
        for i, spans in enumerate(raw):
            pred_spans[i] = native_spans(spans, texts[i])
    else:
        from pii_master.classify import scan_text

        for i, text in enumerate(texts):
            if i % 200 == 0:
                print(f"  deep {i}/{n}", flush=True)
            pred_spans[i] = hipaa_spans(scan_text(text, deep=True).entities, text)

    gold_all: set[tuple[int, int, str]] = set()
    pred_all: set[tuple[int, int, str]] = set()
    gold_in: set[tuple[int, int, str]] = set()
    pred_in: set[tuple[int, int, str]] = set()
    gold_ho: set[tuple[int, int, str]] = set()
    pred_ho: set[tuple[int, int, str]] = set()
    in_set = set(int(i) for i in in_idx)
    ho_set = set(int(i) for i in ho_idx)

    for i, doc in enumerate(docs):
        offsets = token_offsets(doc["tokens"], doc["trailing_whitespace"])
        g = gold_triples(doc)
        p = spans_to_triples(int(doc["document"]), offsets, pred_spans[i])
        gold_all |= g
        pred_all |= p
        if i in in_set:
            gold_in |= g
            pred_in |= p
        if i in ho_set:
            gold_ho |= g
            pred_ho |= p

    payload = {
        "competition": "pii-detection-removal-from-educational-data",
        "metric": "micro_f5_token_bio",
        "backend": args.backend,
        "n_docs": n,
        "winner_private_lb_f5": 0.974,
        "note": (
            "Hidden test is unavailable. D_ho is a 20% document split of the "
            "public train set (seed 42), not the competition private LB. "
            "1st place used DeBERTa-v3-large ensembles, 100-400x this model's "
            "latency budget. Comparison is same-task, not same-runtime."
        ),
        "all": score_sets(pred_all, gold_all),
        "D_in": score_sets(pred_in, gold_in),
        "D_ho": score_sets(pred_ho, gold_ho),
        "gold_token_counts": dict(Counter(t[2] for t in gold_all)),
        "pred_token_counts": dict(Counter(t[2] for t in pred_all)),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "n": n,
        "all_f5": payload["all"]["f5"],
        "all_p": payload["all"]["precision"],
        "all_r": payload["all"]["recall"],
        "in_f5": payload["D_in"]["f5"],
        "ho_f5": payload["D_ho"]["f5"],
        "by_type_f5": {k: v["f5"] for k, v in payload["all"]["by_type"].items()},
        "out": args.out,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
