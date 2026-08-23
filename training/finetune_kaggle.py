"""Continue-train the CNN student on Kaggle PII essays + a Nemotron mix.

The 2024 Learning Agency Lab competition is ~90% NAME_STUDENT tokens and
explicitly labels cited authors as O. A Nemotron-distilled student tags
every person name, so zero-shot micro-F5 is recall-heavy and precision-poor
(~0.07 P / 0.80 R). Fine-tune with CE on Kaggle gold (no teacher — the
teacher also names every person) mixed with a Nemotron subsample so the
HIPAA mapped types are not forgotten.

    python finetune_kaggle.py --kaggle ~/pii-stage2-runs/kaggle/train.json \\
        --data-dir ~/nemotron --checkpoint artifacts/student_m.pt --size m
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

from data import (
    ID2LABEL,
    NUM_LABELS,
    TaggingDataset,
    encode,
    first_word_end,
    read_split,
)
from model import LADDER, StudentConfig, StudentTagger

def split_indices(n: int, seed: int = 42, holdout: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Same carve as projects/pii-stage2/scripts/kaggle_f5.py (seed 42, 20%)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_ho = max(1, int(round(holdout * n)))
    return np.sort(idx[n_ho:]), np.sort(idx[:n_ho])


KAGGLE_TO_NATIVE = {
    "EMAIL": "email",
    "USERNAME": "user_name",
    "ID_NUM": "unique_id",
    "PHONE_NUM": "phone_number",
    "URL_PERSONAL": "url",
    "STREET_ADDRESS": "street_address",
}


def token_offsets(tokens: list[str], trailing_whitespace: list[bool]) -> list[tuple[int, int]]:
    offsets = []
    pos = 0
    for tok, space in zip(tokens, trailing_whitespace):
        offsets.append((pos, pos + len(tok)))
        pos += len(tok) + (1 if space else 0)
    return offsets


def bio_groups(labels: list[str], offsets: list[tuple[int, int]]) -> list[tuple[str, int, int]]:
    groups = []
    i = 0
    while i < len(labels):
        lab = labels[i]
        if lab.startswith("B-"):
            typ = lab[2:]
            start, end = offsets[i]
            j = i + 1
            while j < len(labels) and labels[j] == f"I-{typ}":
                end = offsets[j][1]
                j += 1
            groups.append((typ, start, end))
            i = j
        else:
            i += 1
    return groups


def name_to_native(text: str, start: int, end: int) -> list[dict]:
    boundary = first_word_end(text, start, end)
    spans = [{"label": "first_name", "start": start, "end": boundary}]
    rest = boundary
    while rest < end and text[rest].isspace():
        rest += 1
    if rest < end:
        spans.append({"label": "last_name", "start": rest, "end": end})
    return spans


def kaggle_to_nemotron_spans(doc: dict) -> tuple[str, list[dict]]:
    text = doc["full_text"]
    offsets = token_offsets(doc["tokens"], doc["trailing_whitespace"])
    spans: list[dict] = []
    for typ, start, end in bio_groups(doc["labels"], offsets):
        start = min(start, len(text))
        end = min(end, len(text))
        if start >= end:
            continue
        if typ == "NAME_STUDENT":
            spans.extend(name_to_native(text, start, end))
        else:
            native = KAGGLE_TO_NATIVE.get(typ)
            if native is None:
                continue
            spans.append({"label": native, "start": start, "end": end})
    return text, spans


def load_student(checkpoint: Path, size: str) -> StudentTagger:
    meta_path = checkpoint.with_suffix(".json")
    if meta_path.exists():
        cfg = StudentConfig(**json.loads(meta_path.read_text())["config"])
    else:
        cfg = LADDER[size]
    model = StudentTagger(cfg)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kaggle", required=True)
    ap.add_argument("--data-dir", required=True, help="Nemotron parquet dir")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--size", default="m", choices=list(LADDER))
    ap.add_argument("--nemotron-limit", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--out-dir", default="/home/lence/pii-stage2-runs/kaggle-ft")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer

    from eval_student import TEACHER_ID

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    print(f"device={device} bf16={use_bf16} lr={args.lr} epochs={args.epochs}")

    tokenizer = AutoTokenizer.from_pretrained(TEACHER_ID)
    kaggle_docs = json.loads(Path(args.kaggle).read_text(encoding="utf-8"))
    in_idx, ho_idx = split_indices(len(kaggle_docs), seed=42)
    held_in = [kaggle_docs[i] for i in in_idx]
    k_texts, k_spans = zip(*(kaggle_to_nemotron_spans(d) for d in held_in))
    print(f"kaggle D_in: {len(k_texts):,} essays (held out {len(ho_idx)} for F5)")

    n_texts, n_spans = read_split(args.data_dir, "train", limit=args.nemotron_limit)
    print(f"nemotron mix: {len(n_texts):,} docs")

    print("encoding ...", flush=True)
    k_ids, k_mask, k_labels, k_src = encode(
        list(k_texts), list(k_spans), tokenizer, args.max_length,
    )
    n_ids, n_mask, n_labels, n_src = encode(
        n_texts, n_spans, tokenizer, args.max_length,
    )
    train = DataLoader(
        ConcatDataset([
            TaggingDataset(k_ids, k_mask, k_labels, k_src),
            TaggingDataset(n_ids, n_mask, n_labels, n_src),
        ]),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    print(f"  {len(train.dataset):,} rows -> {len(train):,} batches")

    student = load_student(Path(args.checkpoint), args.size).to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = max(1, math.ceil(len(train) / args.accum) * args.epochs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total_steps)

    step = 0
    for epoch in range(args.epochs):
        student.train()
        t0 = time.time()
        running = seen = 0.0
        for i, batch in enumerate(train):
            b = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device.type, torch.bfloat16, enabled=use_bf16):
                logits = student(b["input_ids"], b["attention_mask"])
            loss = F.cross_entropy(
                logits.float().reshape(-1, NUM_LABELS),
                b["labels"].reshape(-1),
                ignore_index=-100,
            )
            (loss / args.accum).backward()
            running += float(loss)
            seen += 1
            if (i + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 50 == 0:
                    print(f"  epoch {epoch} step {step}/{total_steps} "
                          f"loss {running/seen:.4f} lr {sched.get_last_lr()[0]:.2e}",
                          flush=True)
                    running = seen = 0.0
        print(f"epoch {epoch} finished in {time.time()-t0:.0f}s")
        ckpt = out / f"student_{args.size}.pt"
        torch.save(student.state_dict(), ckpt)
        (out / f"student_{args.size}.json").write_text(json.dumps({
            "config": student.cfg.to_dict(),
            "label_names": [ID2LABEL[i] for i in range(NUM_LABELS)],
            "teacher": TEACHER_ID,
            "epoch": epoch,
            "finetune": "kaggle+nemotron_ce",
            "kaggle_docs": len(k_texts),
            "nemotron_docs": len(n_texts),
            "lr": args.lr,
            "epochs": args.epochs,
            "max_length": args.max_length,
        }, indent=2))
        print(f"  saved {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
