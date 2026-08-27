"""v5a -- fine-tune `kalyan-ks/ettin-68m-nemotron-pii` on this repo's corpora.

Two stages, because only one of eight training corpora carries token-level gold:

1. **span stage** -- `85593_pii_trainset_85.59k/spans.jsonl` has character offsets
   whose `tag_id`s are already this repo's slugs. The token head is re-headed from
   the upstream 55 Nemotron entity types onto our 61 tags in BIO form (123 labels)
   and trained on those documents.
2. **document stage** -- a 61-way sigmoid head over the pooled encoder, trained on
   all 8 corpora, 531,431 rows, against the document-level `gold` lists that every
   corpus has.

Both stages update the **encoder**, and the encoder is the only thing `v5_distill`
consumes. The heads are saved for diagnosis and are not themselves shippable: a
68M transformer cannot serve inside this project's 5 ms / one-core budget, which
is the entire reason the chain continues into model2vec.

## Why 1,024 tokens is not the compromise it looks like

The cascade reads 12,000 characters; 1,024 tokens is roughly 4,000. For a *served*
transformer that truncation would cap recall on long documents, and the
feasibility probe measured exactly that.

It matters far less here, because the artifact this run produces is a
**vocabulary-wide static embedding table**, not a document encoder. Fine-tuning at
1,024 tokens shapes the token representations; `v5_distill` then exports one
vector per vocabulary entry, and `v5_tagger` applies that table across the full
12,000-character window at negligible cost. The truncation bounds what the encoder
*saw while learning*, not what the shipped model can read.

## Held-out discipline

Training uses the `fit` side of `quiet_fit.carve_holdin` only. The `calib` side is
reserved for threshold selection in `v5_tagger`, exactly as every other model in
this repo does it, so the operating points are not chosen on rows the encoder was
fitted to. The sealed `data/2-eval` corpora are not opened by this module at all.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.quiet_data import (  # noqa: E402
    TRAIN_ROOT, iter_quiet_corpus, list_dataset_dirs, read_document,
)

BASE = "kalyan-ks/ettin-68m-nemotron-pii"
CATALOGUE = Path("projects/pii-scorecard-60/cache/catalogue.json")
SPAN_CORPUS = "85593_pii_trainset_85.59k"
PROJECT = Path("projects/pii-content-v5")
READ_CHARS = 12_000
MAX_TOKENS = 1_024


# --------------------------------------------------------------------- carve
_U64 = 0xFFFF_FFFF_FFFF_FFFF
_GOLDEN = 0x9E37_79B9_7F4A_7C15


def carve_fit_mask(corpus_name: str, n_rows: int, calib_frac: float = 0.15):
    """`quiet_fit.carve_holdin`'s split, reproduced exactly, for a streamed corpus.

    **This must be bit-identical to `quiet_fit`, and it is not obvious why.**
    `quiet_fit.load` does not hash the document's `uid` string. It derives
    `uid_hash` from the corpus NAME and the row ORDINAL:

        seed = blake2b(name, digest_size=8);  base = int(seed, "little")
        uid_hash[i] = (i * 0x9E3779B97F4A7C15) ^ base        # uint64, wrapping

    and `carve_holdin` then takes `uid_hash % 10_000 < calib_frac * 10_000` as the
    calibration side.

    Hashing the uid string instead — the obvious thing to write — produces a
    split that is perfectly valid and *completely different*. The two disagree on
    about 85% of the calibration rows, so an encoder fitted under one rule and
    thresholds selected under the other are chosen on rows the encoder trained on,
    which is the exact failure the carve exists to prevent. It is silent: every
    count still looks right.
    """
    import hashlib
    base = int.from_bytes(
        hashlib.blake2b(corpus_name.encode(), digest_size=8).digest(), "little")
    idx = np.arange(n_rows, dtype=object)
    h = np.array([((int(i) * _GOLDEN) & _U64) ^ base for i in idx], dtype=np.uint64)
    return (h % np.uint64(10_000)) >= np.uint64(int(calib_frac * 10_000))


# ----------------------------------------------------------------- datasets
class SpanDataset(Dataset):
    """Documents with character-offset gold, as BIO token labels."""

    def __init__(self, rows, spans, tok, labels):
        self.rows, self.spans, self.tok = rows, spans, tok
        self.bio = {"O": 0}
        for t in labels:
            self.bio[f"B-{t}"] = len(self.bio)
            self.bio[f"I-{t}"] = len(self.bio)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i):
        qr = self.rows[i]
        try:
            text = read_document(Path(qr.path), limit=READ_CHARS)
        except (FileNotFoundError, OSError):
            text = ""
        enc = self.tok(text, truncation=True, max_length=MAX_TOKENS,
                       return_offsets_mapping=True)
        labels = [0] * len(enc["input_ids"])
        for start, end, tag in self.spans.get(qr.uid, ()):
            first = True
            for j, (s, e) in enumerate(enc["offset_mapping"]):
                if s == e:                      # special token
                    continue
                if s >= end or e <= start:      # no overlap
                    continue
                key = f"{'B' if first else 'I'}-{tag}"
                if key in self.bio:
                    labels[j] = self.bio[key]
                first = False
        for j, (s, e) in enumerate(enc["offset_mapping"]):
            if s == e:
                labels[j] = -100                # ignored by the loss
        return {"input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"], "labels": labels}


class DocDataset(Dataset):
    """Documents with document-level multi-label gold."""

    def __init__(self, rows, tok, index):
        self.rows, self.tok, self.index = rows, tok, index

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i):
        qr = self.rows[i]
        try:
            text = read_document(Path(qr.path), limit=READ_CHARS)
        except (FileNotFoundError, OSError):
            text = ""
        enc = self.tok(text, truncation=True, max_length=MAX_TOKENS)
        y = np.zeros(len(self.index), dtype=np.float32)
        for t in qr.row.labels:
            j = self.index.get(t)
            if j is not None:
                y[j] = 1.0
        return {"input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"], "labels": y}


def collate(batch, pad_id: int, token_level: bool):
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), n), dtype=torch.long)
    if token_level:
        lab = torch.full((len(batch), n), -100, dtype=torch.long)
    else:
        lab = torch.zeros((len(batch), len(batch[0]["labels"])), dtype=torch.float)
    for i, b in enumerate(batch):
        k = len(b["input_ids"])
        ids[i, :k] = torch.tensor(b["input_ids"], dtype=torch.long)
        mask[i, :k] = torch.tensor(b["attention_mask"], dtype=torch.long)
        if token_level:
            lab[i, :k] = torch.tensor(b["labels"], dtype=torch.long)
        else:
            lab[i] = torch.tensor(b["labels"], dtype=torch.float)
    return {"input_ids": ids, "attention_mask": mask, "labels": lab}


# ------------------------------------------------------------------- loading
def load_spans(corpus_dir: Path, labels: set[str]):
    spans = defaultdict(list)
    path = corpus_dir / "spans.jsonl"
    kept = dropped = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            tag = rec.get("tag_id")
            if tag in labels:
                spans[rec["uid"]].append((int(rec["start"]), int(rec["end"]), tag))
                kept += 1
            else:
                dropped += 1
    print(f"spans: {kept:,} kept, {dropped:,} dropped (tag not in the 61-label "
          f"catalogue), over {len(spans):,} documents", flush=True)
    return spans


def fit_rows(only: str | None = None):
    """Fit-side rows only, in `list_dataset_dirs` order — the order `quiet_fit`
    concatenates in, which is what makes the ordinal-based carve line up."""
    for d in list_dataset_dirs(TRAIN_ROOT):
        if only and d.name != only:
            continue
        rows = list(iter_quiet_corpus(d))
        keep = carve_fit_mask(d.name, len(rows))
        for qr, k in zip(rows, keep):
            if k:
                yield qr


# --------------------------------------------------------------------- loops
def run_epoch(model, loader, opt, sched, device, scaler_dtype, tag: str,
              log_every: int = 200):
    model.train()
    total, seen, t0 = 0.0, 0, time.perf_counter()
    for step, batch in enumerate(loader, start=1):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=scaler_dtype):
            out = model(**batch)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        total += float(loss.detach())
        seen += 1
        if step % log_every == 0:
            rate = step * loader.batch_size / (time.perf_counter() - t0)
            print(f"  [{tag}] step {step:>6,}/{len(loader):,}  "
                  f"loss {total / seen:.4f}  {rate:,.0f} docs/s", flush=True)
    return total / max(seen, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["span", "doc", "both"], default="both")
    ap.add_argument("--epochs-span", type=int, default=1)
    ap.add_argument("--epochs-doc", type=int, default=1)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit-doc", type=int, default=0,
                    help="cap document-stage rows (smoke tests only)")
    ap.add_argument("--out", type=Path, default=PROJECT / "models" / "v5a-ettin-ft")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("no CUDA device: fine-tuning a 68M encoder on CPU is not "
                         "a reasonable use of this budget. Fix the GPU or say so.")
    labels = tuple(json.loads(CATALOGUE.read_text())["labels"])
    index = {t: i for i, t in enumerate(labels)}
    tok = AutoTokenizer.from_pretrained(BASE)
    args.out.mkdir(parents=True, exist_ok=True)
    history: dict[str, object] = {"base": BASE, "n_labels": len(labels),
                                  "max_tokens": MAX_TOKENS, "read_chars": READ_CHARS,
                                  "device": torch.cuda.get_device_name(0)}

    # ------------------------------------------------------------ span stage
    if args.stage in ("span", "both"):
        spans = load_spans(TRAIN_ROOT / SPAN_CORPUS, set(labels))
        rows = [qr for qr in fit_rows(SPAN_CORPUS) if qr.uid in spans]
        print(f"span stage: {len(rows):,} fit-side documents with spans", flush=True)
        ds = SpanDataset(rows, spans, tok, labels)
        model = AutoModelForTokenClassification.from_pretrained(
            BASE, num_labels=len(ds.bio), ignore_mismatched_sizes=True).to(device)
        loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                            num_workers=args.workers, pin_memory=True,
                            collate_fn=lambda b: collate(b, tok.pad_token_id, True))
        steps = len(loader) * args.epochs_span
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=args.lr, total_steps=max(steps, 1), pct_start=0.1)
        losses = [run_epoch(model, loader, opt, sched, device, torch.bfloat16,
                            f"span {e + 1}/{args.epochs_span}")
                  for e in range(args.epochs_span)]
        history["span_stage"] = {"documents": len(rows), "bio_labels": len(ds.bio),
                                 "epoch_loss": losses}
        model.save_pretrained(args.out / "span_head")
        tok.save_pretrained(args.out / "span_head")
        (args.out / "span_head" / "bio_labels.json").write_text(
            json.dumps(ds.bio, indent=1), encoding="utf-8")
        print(f"-> {args.out / 'span_head'}", flush=True)
        encoder_src = str(args.out / "span_head")
    else:
        encoder_src = BASE

    # -------------------------------------------------------- document stage
    if args.stage in ("doc", "both"):
        rows = [qr for qr in fit_rows()]
        if args.limit_doc:
            rows = rows[:args.limit_doc]
        print(f"document stage: {len(rows):,} fit-side documents "
              f"across all training corpora", flush=True)
        ds = DocDataset(rows, tok, index)
        model = AutoModelForSequenceClassification.from_pretrained(
            encoder_src, num_labels=len(labels),
            problem_type="multi_label_classification",
            ignore_mismatched_sizes=True).to(device)
        loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                            num_workers=args.workers, pin_memory=True,
                            collate_fn=lambda b: collate(b, tok.pad_token_id, False))
        steps = len(loader) * args.epochs_doc
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=args.lr, total_steps=max(steps, 1), pct_start=0.1)
        losses = [run_epoch(model, loader, opt, sched, device, torch.bfloat16,
                            f"doc {e + 1}/{args.epochs_doc}")
                  for e in range(args.epochs_doc)]
        history["doc_stage"] = {"documents": len(rows), "labels": len(labels),
                                "epoch_loss": losses}
        model.save_pretrained(args.out / "doc_head")
        tok.save_pretrained(args.out / "doc_head")
        (args.out / "doc_head" / "labels.json").write_text(
            json.dumps(list(labels), indent=1), encoding="utf-8")
        print(f"-> {args.out / 'doc_head'}", flush=True)

    (args.out / "history.json").write_text(json.dumps(history, indent=1),
                                           encoding="utf-8")
    print(f"\n-> {args.out / 'history.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
