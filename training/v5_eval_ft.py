"""v5a checkpoint -- did fine-tuning move the tags the ship gate depends on?

The feasibility probe measured `kalyan-ks/ettin-68m-nemotron-pii` **as shipped**,
hand-mapped onto 32 of our 61 tags, and found a clean split: F1 0.71-0.95 on
`email`, `zip_code`, `phone_number`, `given_name`, `family_name` and
`street_number_and_name`, and **0.0000** on `ssn`, `mrn`, `itin`, `cvv` and
`health_plan_beneficiary_number` -- the five identifiers the priority gate is
about. Whether fine-tuning fixes those five is the question that decides if this
chain is worth continuing, so it is measured before v5b is funded.

## Both models, the same rows

The probe ran on an arbitrary 1,200-document sample, some of which are now
training rows. Comparing the fine-tuned model against those numbers would compare
two models on two different samples, one of which the new model has memorised.

So this scores **both** models on the same documents, drawn from the **calibration**
side of `quiet_fit.carve_holdin` -- rows `v5_finetune` was forbidden to train on.
The as-shipped column is therefore re-measured rather than quoted, and the delta
between the columns is a real one.

The sealed `data/2-eval` corpora are not opened here. This is a checkpoint on
held-in data; the sealed measurement happens once, in `v5_fuse`.

## Two thresholds, both reported

`@0.5` is the sigmoid default -- what the head says without tuning. `best-F1` is
the per-tag optimum over the sweep, which is the ceiling a calibrated threshold
could reach. A tag that is 0.0000 at both has genuinely not been learned; a tag
that is 0.0000 at 0.5 and strong at its optimum is a calibration problem, and
those need opposite responses.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.quiet_data import (  # noqa: E402
    TRAIN_ROOT, iter_quiet_corpus, list_dataset_dirs, read_document,
)
from training.v5_finetune import BASE, MAX_TOKENS, READ_CHARS, carve_fit_mask  # noqa: E402

PROJECT = Path("projects/pii-content-v5")
CATALOGUE = Path("projects/pii-scorecard-60/cache/catalogue.json")
#: the same hand mapping the feasibility probe used, so the as-shipped column here
#: and the one in `probe/ettin_asis.json` mean the same thing
MAP = {k: v for k, v in
       json.loads(Path(__file__).with_name("v5_entity_map.json").read_text()).items()
       if not k.startswith("_")}
GATED = ["sensitive_pii_social_security_number",
         "sensitive_phi_medical_record_number_mrn",
         "sensitive_pci_individual_taxpayer_identification_number_itin",
         "sensitive_pci_card_verification_value_cvv",
         "sensitive_phi_health_plan_beneficiary_number"]


def prf(pred: np.ndarray, gold: np.ndarray) -> tuple[float, float, float]:
    tp = int((pred & gold).sum())
    fp = int((pred & ~gold).sum())
    fn = int((~pred & gold).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return p, r, (0.0 if p + r == 0 else 2 * p * r / (p + r))


def best_f1(score: np.ndarray, gold: np.ndarray) -> tuple[float, float]:
    """Best achievable F1 over the sweep, and the threshold that reaches it."""
    if not gold.any():
        return 0.0, float("nan")
    order = np.argsort(-score)
    g = gold[order]
    tp = np.cumsum(g)
    fp = np.cumsum(~g)
    fn = int(gold.sum()) - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    k = int(np.argmax(f1))
    return float(f1[k]), float(score[order][k])


def calib_sample(n: int, seed: int = 11, only: str | None = None):
    """Documents from the calibration side, spread across all eight corpora."""
    rng = np.random.default_rng(seed)
    out = []
    dirs = [d for d in list_dataset_dirs(TRAIN_ROOT)
            if only is None or d.name == only]
    per = max(n // max(len(dirs), 1), 1)
    for d in dirs:
        rows = list(iter_quiet_corpus(d))
        calib = ~carve_fit_mask(d.name, len(rows))
        idx = np.flatnonzero(calib)
        if not len(idx):
            continue
        take = rng.choice(idx, size=min(per, len(idx)), replace=False)
        out.extend(rows[i] for i in take)
    return out


def score_span_head(head: Path, texts: list[str], labels, index, device, batch: int):
    """Document-level tag decisions from the BIO token head: a tag fires if any
    token in the document was tagged with it.

    Available as soon as the span stage finishes, hours before the document stage
    does, and it answers the same checkpoint question on the tags the span corpus
    actually annotates. It is a *floor* on what the finished v5a can do -- the
    document stage has eight corpora and this has one -- so a tag that is already
    strong here is settled, and a weak one is not yet evidence of absence.
    """
    bio = json.loads((head / "bio_labels.json").read_text())
    inv = {v: k for k, v in bio.items()}
    tok = AutoTokenizer.from_pretrained(head)
    model = AutoModelForTokenClassification.from_pretrained(head).to(device).eval()
    fired = np.zeros((len(texts), len(labels)), dtype=bool)
    with torch.inference_mode():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], return_tensors="pt", truncation=True,
                      max_length=MAX_TOKENS, padding=True).to(device)
            out = model(**enc).logits.argmax(-1).cpu().numpy()
            mask = enc["attention_mask"].cpu().numpy().astype(bool)
            for k, (r, m) in enumerate(zip(out, mask)):
                for lab_id in {int(x) for x in r[m]}:
                    name = inv.get(lab_id, "O")
                    if name == "O":
                        continue
                    tag = name.split("-", 1)[1]
                    if tag in index:
                        fired[i + k, index[tag]] = True
    del model
    torch.cuda.empty_cache()
    return fired


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft", type=Path,
                    default=PROJECT / "models/v5a-ettin-ft/doc_head")
    ap.add_argument("--span-head", type=Path, default=None,
                    help="evaluate the BIO token head instead of the document head")
    ap.add_argument("--n", type=int, default=2400)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--corpus", default=None,
                    help="restrict the calibration sample to one corpus")
    ap.add_argument("--out", type=Path, default=PROJECT / "probe/v5a_checkpoint.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    labels = tuple(json.loads(CATALOGUE.read_text())["labels"])
    index = {t: i for i, t in enumerate(labels)}

    rows = calib_sample(args.n, only=args.corpus)
    texts, gold = [], np.zeros((len(rows), len(labels)), dtype=bool)
    for i, qr in enumerate(rows):
        try:
            texts.append(read_document(Path(qr.path), limit=READ_CHARS))
        except (FileNotFoundError, OSError):
            texts.append("")
        for t in qr.row.labels:
            j = index.get(t)
            if j is not None:
                gold[i, j] = True
    print(f"{len(texts):,} calibration documents across "
          f"{len(list_dataset_dirs(TRAIN_ROOT))} corpora", flush=True)

    # ------------------------------------------------------------ fine-tuned
    t0 = time.perf_counter()
    span_mode = args.span_head is not None
    if span_mode:
        fired = score_span_head(args.span_head, texts, labels, index, device,
                                args.batch)
        scores = fired.astype(np.float32)      # a hard decision, not a probability
    else:
        tok = AutoTokenizer.from_pretrained(args.ft)
        ft = AutoModelForSequenceClassification.from_pretrained(
            args.ft).to(device).eval()
        scores = np.zeros((len(texts), len(labels)), dtype=np.float32)
        with torch.inference_mode():
            for i in range(0, len(texts), args.batch):
                enc = tok(texts[i:i + args.batch], return_tensors="pt",
                          truncation=True, max_length=MAX_TOKENS,
                          padding=True).to(device)
                scores[i:i + args.batch] = torch.sigmoid(
                    ft(**enc).logits.float()).cpu().numpy()
        del ft
        torch.cuda.empty_cache()
    print(f"fine-tuned inference {time.perf_counter() - t0:.0f}s "
          f"({'span head' if span_mode else 'document head'})", flush=True)

    # ------------------------------------------------------------- as shipped
    asis = np.zeros((len(texts), len(labels)), dtype=bool)
    if MAP:
        tok0 = AutoTokenizer.from_pretrained(BASE)
        base = AutoModelForTokenClassification.from_pretrained(BASE).to(device).eval()
        id2label = base.config.id2label
        with torch.inference_mode():
            for i in range(0, len(texts), args.batch):
                enc = tok0(texts[i:i + args.batch], return_tensors="pt",
                           truncation=True, max_length=MAX_TOKENS,
                           padding=True).to(device)
                out = base(**enc).logits.argmax(-1).cpu().numpy()
                mask = enc["attention_mask"].cpu().numpy().astype(bool)
                for k, (r, m) in enumerate(zip(out, mask)):
                    for lab in {id2label[int(x)] for x in r[m]}:
                        if lab == "O":
                            continue
                        tag = MAP.get(lab.split("-", 1)[1])
                        if tag in index:
                            asis[i + k, index[tag]] = True
        del base
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------- report
    table = {}
    for j, tag in enumerate(labels):
        g = gold[:, j]
        if not g.any():
            continue
        p, r, f = prf(scores[:, j] >= 0.5, g)
        bf, bt = best_f1(scores[:, j], g)
        row = {"support": int(g.sum()), "ft_p@0.5": p, "ft_r@0.5": r, "ft_f1@0.5": f,
               "ft_best_f1": bf, "ft_best_threshold": bt}
        if MAP:
            ap_, ar_, af_ = prf(asis[:, j], g)
            row |= {"asis_p": ap_, "asis_r": ar_, "asis_f1": af_,
                    "delta_f1": bf - af_}
        table[tag] = row

    print(f"\n{'tag':<52}{'sup':>6}{'as-is F1':>10}{'ft F1@.5':>10}"
          f"{'ft best':>9}{'delta':>8}")
    for tag, row in sorted(table.items(), key=lambda kv: -kv[1].get("delta_f1", 0)):
        print(f"{tag:<52}{row['support']:>6}"
              f"{row.get('asis_f1', float('nan')):>10.4f}"
              f"{row['ft_f1@0.5']:>10.4f}{row['ft_best_f1']:>9.4f}"
              f"{row.get('delta_f1', float('nan')):>8.4f}")

    print("\n--- the five tags the gate depends on ---")
    verdict = {}
    for tag in GATED:
        row = table.get(tag)
        if row is None:
            print(f"  {tag:<52} no gold in this sample")
            continue
        verdict[tag] = row["ft_best_f1"]
        print(f"  {tag:<52} support={row['support']:<6} "
              f"as-is {row.get('asis_f1', float('nan')):.4f} -> "
              f"ft best {row['ft_best_f1']:.4f}")
    moved = [t for t, f in verdict.items() if f >= 0.30]
    print(f"\n{len(moved)} of {len(verdict)} measurable gated tags cleared F1 0.30 "
          f"after fine-tuning.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"model": str(args.ft), "n_documents": len(texts), "split": "calibration",
         "max_tokens": MAX_TOKENS, "read_chars": READ_CHARS,
         "gated_tags_cleared_0.30": moved, "per_tag": table}, indent=1),
        encoding="utf-8")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
