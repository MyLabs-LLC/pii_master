"""Score a trained student against the Nemotron-PII holdout — acceptance gate 2.

Gate 2 of docs/DISTILLATION_PLAN.md: span F1 on the holdout must beat the
committed rules-only baseline (docs/BASELINE_NEMOTRON.md: micro P 0.854 /
R 0.732 / F1 0.788) **on the same 12 mapped types**. So this scores with the
identical protocol as eval/scripts/nemotron_eval.py -- exact (type, start, end)
match, gold of unmodelled labels dropped, strict precision -- and reports:

  rules    the committed baseline, recomputed here so the comparison is
           same-code, same-documents rather than a number copied from a doc
  student  the student's 55 labels pushed through pii_master.crosswalk
  fusion   the production cascade, in the two readings of the plan's fusion
           policy (docs/DISTILLATION_PLAN.md section 7):
             rules_first      every rule span outranks the model
             checksum_first   only rule spans with a checksum or hard format
                              validator outrank it; on the cue-anchored types
                              the model wins, and rules still fill in where the
                              model is silent
             longest_wins     checksum_first, except that a cue-anchored rule
                              span beats an overlapping model span of the same
                              type unless the model's is at least as long

It also reports the student's own 55-label span F1, which is the number to
watch across training runs; the 12-type view is deliberately narrow.

Documents are scored WHOLE, not truncated to the training window: the student
is a CNN with no position embeddings, so its receptive field is the same at
any length.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import ID2LABEL, NUM_LABELS, parse_spans, read_split  # noqa: E402
from decode import decode_spans  # noqa: E402
from model import LADDER, StudentConfig, StudentTagger  # noqa: E402

from pii_master.classify import scan_text  # noqa: E402
from pii_master.crosswalk import to_entity_type  # noqa: E402
from pii_master.evaluation import TypeScore  # noqa: E402
from pii_master.validators import ipv4_ok, ipv6_ok, luhn_ok, ssn_ok  # noqa: E402

TEACHER_ID = "kalyan-ks/ettin-68m-nemotron-pii"
# Rule types whose validator is a checksum or a hard format parse: Luhn, the
# SSN range rules, ipaddress.ip_address, RFC-ish email and URL structure. These
# are the spans docs/DISTILLATION_PLAN.md gate 4 protects. The remaining rule
# types (ACCOUNT_NUMBER, MRN, HEALTH_PLAN_ID, DATE_DOB, PHONE_US,
# US_DRIVER_LICENSE) are cue-anchored guesses, not validated facts.
CHECKSUMMED = {"SSN", "CREDIT_CARD", "EMAIL", "IP_ADDRESS", "URL"}
FUSIONS = ("fusion_rules_first", "fusion_checksum_first", "fusion_longest_wins")


def revalidate(entity_type: str, text: str) -> bool:
    """Gate 4's second clause: a model span of a checksummed type must pass it.

    The student learns whatever the corpus labels, and 88% of Nemotron's gold
    cards fail Luhn -- so without this it happily emits card numbers no payment
    network would issue. Types with no validator pass through.
    """
    digits = "".join(c for c in text if c.isdigit())
    if entity_type == "CREDIT_CARD":
        return 13 <= len(digits) <= 19 and luhn_ok(digits)
    if entity_type == "SSN":
        return len(digits) == 9 and ssn_ok(digits[:3], digits[3:5], digits[5:])
    if entity_type == "IP_ADDRESS":
        stripped = text.strip()
        return ipv4_ok(stripped) or ipv6_ok(stripped)
    if entity_type == "EMAIL":
        return "@" in text and "." in text.split("@")[-1]
    if entity_type == "URL":
        return "." in text or ":" in text
    return True


def load_student(checkpoint: Path, size: str | None) -> StudentTagger:
    meta_path = checkpoint.with_suffix(".json")
    if meta_path.exists():
        cfg = StudentConfig(**json.loads(meta_path.read_text())["config"])
    else:
        cfg = LADDER[size or "xs"]
    model = StudentTagger(cfg)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model.eval()


def predict(model, tokenizer, texts, device, batch_size=32, max_length=4096,
            min_confidence=0.0):
    """-> [[(nemotron_label, start, end)]] for each document.

    `min_confidence` drops spans whose mean per-token probability is below the
    threshold, which is the knob DESIGN.md section 12 leaves as policy.
    """
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out: list[list[tuple[str, int, int]]] = [[] for _ in texts]
    model = model.to(device)
    for chunk in range(0, len(order), batch_size):
        rows = order[chunk:chunk + batch_size]
        enc = tokenizer([texts[i] for i in rows], truncation=True, padding=True,
                        max_length=max_length, return_offsets_mapping=True,
                        return_tensors="np")
        ids = torch.as_tensor(enc["input_ids"], dtype=torch.long).to(device)
        mask = torch.as_tensor(enc["attention_mask"], dtype=torch.long).to(device)
        with torch.no_grad():
            logits = model(ids, mask)
            if min_confidence > 0.0:
                probability = logits.softmax(-1).max(-1).values.cpu().numpy()
            pred = logits.argmax(-1).cpu().numpy()
        for position, row in enumerate(rows):
            offsets = enc["offset_mapping"][position]
            spans = decode_spans(texts[row], offsets, pred[position], ID2LABEL)
            if min_confidence > 0.0:
                spans = [span for span in spans
                         if span_confidence(span, offsets, probability[position])
                         >= min_confidence]
            out[row] = spans
    return out


def span_confidence(span, offsets, probability) -> float:
    _, start, end = span
    scores = [probability[i] for i, (a, b) in enumerate(offsets)
              if b > a and a < end and start < b]
    return float(sum(scores) / len(scores)) if scores else 0.0


def overlaps(span, others) -> bool:
    _, start, end = span
    return any(start < b and a < end for _, a, b in others)


def fuse(rule_spans, student_spans, policy: str):
    """Merge the two tiers under one of the precedence readings."""
    if policy == "fusion_rules_first":
        authoritative, deferred = list(rule_spans), []
    else:
        authoritative = [r for r in rule_spans if r[0] in CHECKSUMMED]
        deferred = [r for r in rule_spans if r[0] not in CHECKSUMMED]
    if policy == "fusion_longest_wins":
        # Same reading as checksum-first, plus: when a model span and a
        # cue-anchored rule span of the SAME type overlap, keep whichever is
        # longer. The model's failure mode on short documents is truncation
        # ("84-J99-1220" -> "84-J99-12"), and a truncated span of the right
        # type is worse than the rule it replaced.
        authoritative = list(authoritative)
        for rule in deferred:
            rival = [s for s in student_spans
                     if s[0] == rule[0] and overlaps(s, [rule])]
            if rival and max(s[2] - s[1] for s in rival) >= rule[2] - rule[1]:
                continue
            authoritative.append(rule)
        deferred = []
    kept = list(authoritative)
    kept += [s for s in student_spans if not overlaps(s, authoritative)]
    # Cue-anchored rule spans still contribute recall where the model is silent.
    kept += [r for r in deferred if not overlaps(r, kept)]
    return kept


def micro(scores: dict[str, TypeScore]) -> tuple[float, float, float]:
    tp = sum(s.tp for s in scores.values())
    fp = sum(s.fp for s in scores.values())
    fn = sum(s.fn for s in scores.values())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def triage(gold_native, gold_mapped, pred_spans, buckets) -> None:
    """Split non-exact predictions the way eval/scripts/nemotron_eval.py does.

    Strict precision counts every prediction that is not an exact mapped hit,
    which is misleading when the prediction landed on a real identifier whose
    label we do not model -- that gold was withheld from the denominator.
    """
    exact = set(gold_mapped)
    for span in pred_spans:
        if span in exact:
            continue
        _, start, end = span
        if any(start < b and a < end for _, a, b in gold_mapped):
            buckets["mapped_mismatch"] += 1
        elif any(start < b and a < end for _, a, b in gold_native):
            buckets["unmodelled_overlap"] += 1
        else:
            buckets["spurious"] += 1


def tally(scores: dict[str, TypeScore], gold_spans, pred_spans,
          partial: dict[str, TypeScore] | None = None) -> None:
    gold: dict[str, set] = defaultdict(set)
    pred: dict[str, set] = defaultdict(set)
    for t, a, b in gold_spans:
        gold[t].add((a, b))
    for t, a, b in pred_spans:
        pred[t].add((a, b))
    for t in set(gold) | set(pred):
        g, p = gold.get(t, set()), pred.get(t, set())
        sc = scores[t]
        sc.gold += len(g)
        sc.tp += len(g & p)
        sc.fp += len(p - g)
        sc.fn += len(g - p)
        if partial is None:
            continue
        # Overlap match, greedy and one-to-one: separates a boundary error
        # (partial hit) from a genuine miss.
        remaining, hits = sorted(p), 0
        for gs, ge in sorted(g):
            for index, (ps, pe) in enumerate(remaining):
                if gs < pe and ps < ge:
                    hits += 1
                    del remaining[index]
                    break
        ps_ = partial[t]
        ps_.gold += len(g)
        ps_.tp += hits
        ps_.fp += len(p) - hits
        ps_.fn += len(g) - hits


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--size", default=None, choices=[*LADDER])
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--min-confidence", type=float, default=0.0,
                    help="drop student spans below this mean token probability")
    ap.add_argument("--revalidate", action="store_true",
                    help="gate 4: re-run our validators on model spans of "
                         "checksummed types and drop the failures")
    ap.add_argument("--skip-rules", action="store_true",
                    help="student only; skips the rules and fusion columns")
    ap.add_argument("--json-out", help="write the scores here")
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TEACHER_ID)
    student = load_student(Path(args.checkpoint), args.size)
    texts, raw_spans = read_split(args.data_dir, args.split, limit=args.limit)
    if args.limit:
        texts, raw_spans = texts[:args.limit], raw_spans[:args.limit]
    print(f"scoring {len(texts):,} {args.split} documents on {args.device}", flush=True)

    predictions = predict(student, tokenizer, texts, torch.device(args.device),
                          args.batch_size, min_confidence=args.min_confidence)

    native: dict[str, TypeScore] = defaultdict(TypeScore)      # all 55 labels
    systems = ("rules", "student", *FUSIONS)
    mapped = {name: defaultdict(TypeScore) for name in systems}
    loose = {name: defaultdict(TypeScore) for name in systems}
    buckets = {name: Counter() for name in systems}

    for text, raw, student_labels in zip(texts, raw_spans, predictions):
        gold_native, gold_mapped = [], []
        for span in parse_spans(raw):
            gold_native.append((span["label"], span["start"], span["end"]))
            entity = to_entity_type(span["label"])
            if entity is not None:
                gold_mapped.append((entity.value, span["start"], span["end"]))
        tally(native, gold_native, student_labels)

        student_mapped = []
        for label, start, end in student_labels:
            entity = to_entity_type(label)
            if entity is None:
                continue
            if args.revalidate and not revalidate(entity.value, text[start:end]):
                continue
            student_mapped.append((entity.value, start, end))
        tally(mapped["student"], gold_mapped, student_mapped, loose["student"])
        triage(gold_native, gold_mapped, student_mapped, buckets["student"])

        if not args.skip_rules:
            rule_spans = [(e.type.value, e.start, e.end) for e in scan_text(text).entities]
            tally(mapped["rules"], gold_mapped, rule_spans, loose["rules"])
            triage(gold_native, gold_mapped, rule_spans, buckets["rules"])
            for policy in FUSIONS:
                fused = fuse(rule_spans, student_mapped, policy)
                tally(mapped[policy], gold_mapped, fused, loose[policy])
                triage(gold_native, gold_mapped, fused, buckets[policy])

    print(f"\nStudent, native 55-label taxonomy: "
          f"P {micro(native)[0]:.3f} R {micro(native)[1]:.3f} F1 {micro(native)[2]:.3f}")
    print("\nGate 2 — the 12 mapped types, exact span match.")
    print("adj P excludes predictions landing on gold of an unmodelled label,")
    print("the same adjustment docs/BASELINE_NEMOTRON.md reports for the rules.")
    print(f"{'system':>8} {'P':>7} {'R':>7} {'F1':>7} {'adj P':>7} {'adj F1':>7} "
          f"{'partial R':>10} {'spurious':>9}")
    for name in systems:
        if not mapped[name]:
            continue
        p, r, f = micro(mapped[name])
        tp = sum(s.tp for s in mapped[name].values())
        b = buckets[name]
        honest = tp + b["mapped_mismatch"] + b["spurious"]
        adj_p = tp / honest if honest else 0.0
        adj_f = 2 * adj_p * r / (adj_p + r) if adj_p + r else 0.0
        print(f"{name:>8} {p:>7.3f} {r:>7.3f} {f:>7.3f} {adj_p:>7.3f} {adj_f:>7.3f} "
              f"{micro(loose[name])[1]:>10.3f} {b['spurious']:>9,}")

    print("\nPer-type (gold / P / R / F1):")
    print(f"{'type':>20} {'gold':>8} {'rules F1':>9} {'stud F1':>9} "
          f"{'rules-1st':>10} {'cksum-1st':>10} {'longest':>8}")
    for t in sorted(mapped["student"]):
        row = [mapped[n][t] for n in ("rules", "student", *FUSIONS)]
        print(f"{t:>20} {row[1].gold:>8,} {row[0].f1:>9.3f} {row[1].f1:>9.3f} "
              f"{row[2].f1:>10.3f} {row[3].f1:>10.3f} {row[4].f1:>8.3f}")

    if args.json_out:
        payload = {
            "documents": len(texts),
            "buckets": {n: dict(buckets[n]) for n in systems if mapped[n]},
            "partial_micro": {n: dict(zip("prf", micro(loose[n])))
                              for n in systems if mapped[n]},
            "native_micro": dict(zip("prf", micro(native))),
            "native": {k: v.to_dict() for k, v in native.items()},
            "mapped_micro": {n: dict(zip("prf", micro(mapped[n])))
                             for n in mapped if mapped[n]},
            "mapped": {n: {k: v.to_dict() for k, v in mapped[n].items()}
                       for n in mapped if mapped[n]},
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
