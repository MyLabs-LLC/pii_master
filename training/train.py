"""Distil kalyan-ks/ettin-68m-nemotron-pii into the CNN student.

Loss = alpha * KL(student || teacher soft targets, temperature T)
     + (1 - alpha) * CE(student, gold BIO labels)

Nemotron-PII ships gold spans, so this is distillation *plus* supervision, not
pure distillation -- the teacher supplies dark knowledge (which wrong labels are
plausible) while the gold labels keep it anchored.

THE ONE THING THAT WILL SILENTLY RUIN THIS RUN: the teacher's label ids are not
our label ids. `build_label_permutation` remaps teacher logits into our order
using the teacher's own config.id2label and hard-fails on any mismatch. Never
assume the orders agree.

Sized for a 12 GB laptop GPU: bf16 autocast, batch 16 at seq 512, gradient
accumulation for a larger effective batch. The teacher is only 68M, so it fits
alongside the student comfortably.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import ID2LABEL, NUM_LABELS, TaggingDataset, encode, read_split
from model import LADDER, StudentTagger

TEACHER_ID = "kalyan-ks/ettin-68m-nemotron-pii"


def build_label_permutation(teacher_id2label: dict) -> torch.Tensor:
    """Index tensor mapping teacher logit columns into our label order.

    teacher_logits[..., perm] is then in our order. Hard-fails rather than
    silently mis-aligning, which would train the student on scrambled targets.
    """
    normalised = {}
    for idx, name in teacher_id2label.items():
        key = str(name).strip()
        for variant in (key, key.upper(), key.lower(), key.replace("_", "-")):
            normalised.setdefault(variant, int(idx))
    perm, missing = [], []
    for i in range(NUM_LABELS):
        ours = ID2LABEL[i]
        for candidate in (ours, ours.upper(), ours.lower(), ours.replace("_", "-")):
            if candidate in normalised:
                perm.append(normalised[candidate])
                break
        else:
            missing.append(ours)
    if missing:
        raise SystemExit(
            f"teacher label space misses {len(missing)} of our labels, e.g. {missing[:8]}.\n"
            f"teacher labels: {sorted(set(teacher_id2label.values()))[:12]} ...\n"
            "Fix build_label_permutation before training -- do not train through this."
        )
    return torch.tensor(perm, dtype=torch.long)


def distillation_loss(student_logits, teacher_logits, labels, alpha, temperature):
    active = labels.view(-1) != -100
    s = student_logits.reshape(-1, student_logits.shape[-1])[active]
    t = teacher_logits.reshape(-1, teacher_logits.shape[-1])[active]
    y = labels.view(-1)[active]
    if s.numel() == 0:
        return student_logits.sum() * 0.0, 0.0, 0.0
    hard = F.cross_entropy(s, y)
    soft = F.kl_div(
        F.log_softmax(s / temperature, dim=-1),
        F.log_softmax(t / temperature, dim=-1),
        reduction="batchmean", log_target=True,
    ) * (temperature ** 2)
    return alpha * soft + (1 - alpha) * hard, soft.detach().item(), hard.detach().item()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, help="dir holding Nemotron parquet")
    ap.add_argument("--size", default="xs", choices=list(LADDER))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--temperature", type=float, default=3.0)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None, help="rows, for a smoke run")
    ap.add_argument("--out-dir", default="artifacts")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    from transformers import AutoModelForTokenClassification, AutoTokenizer

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    print(f"device={device}  bf16={use_bf16}  student={args.size}")

    tokenizer = AutoTokenizer.from_pretrained(TEACHER_ID)
    teacher = AutoModelForTokenClassification.from_pretrained(TEACHER_ID).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    perm = build_label_permutation(teacher.config.id2label).to(device)
    print(f"label permutation OK: {NUM_LABELS} labels aligned to the teacher")

    print("loading + encoding train split ...")
    texts, spans = read_split(args.data_dir, "train", limit=args.limit)
    ids, mask, labels = encode(texts, spans, tokenizer, args.max_length)
    train = DataLoader(TaggingDataset(ids, mask, labels), batch_size=args.batch_size,
                       shuffle=True, num_workers=2, pin_memory=device.type == "cuda")
    print(f"  {len(texts):,} documents -> {len(train):,} batches")

    cfg = LADDER[args.size]
    cfg.vocab_size = len(tokenizer)
    cfg.num_labels = NUM_LABELS
    cfg.pad_id = tokenizer.pad_token_id or 0
    student = StudentTagger(cfg).to(device)
    print(f"  student {student.num_parameters()/1e6:.2f}M params, "
          f"{cfg.macs_per_token()/1000:.1f}k MACs/token")

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = max(1, math.ceil(len(train) / args.accum) * args.epochs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total_steps)

    step = 0
    for epoch in range(args.epochs):
        student.train(); t0 = time.time()
        running = soft_sum = hard_sum = seen = 0.0
        for i, batch in enumerate(train):
            b = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.no_grad(), torch.autocast(device.type, torch.bfloat16, enabled=use_bf16):
                t_logits = teacher(input_ids=b["input_ids"],
                                   attention_mask=b["attention_mask"]).logits
            t_logits = t_logits.float()[..., perm]
            with torch.autocast(device.type, torch.bfloat16, enabled=use_bf16):
                s_logits = student(b["input_ids"], b["attention_mask"])
            loss, soft, hard = distillation_loss(
                s_logits.float(), t_logits, b["labels"], args.alpha, args.temperature)
            (loss / args.accum).backward()
            running += float(loss); soft_sum += soft; hard_sum += hard; seen += 1

            if (i + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 50 == 0:
                    print(f"  epoch {epoch} step {step}/{total_steps} "
                          f"loss {running/seen:.4f} (soft {soft_sum/seen:.4f} "
                          f"hard {hard_sum/seen:.4f}) lr {sched.get_last_lr()[0]:.2e}")
                    running = soft_sum = hard_sum = seen = 0.0
        print(f"epoch {epoch} finished in {time.time()-t0:.0f}s")

        torch.save(student.state_dict(), out / f"student_{args.size}.pt")
        (out / f"student_{args.size}.json").write_text(json.dumps({
            "config": cfg.to_dict(),
            "label_names": [ID2LABEL[i] for i in range(NUM_LABELS)],
            "teacher": TEACHER_ID, "epoch": epoch,
        }, indent=2))
        print(f"  saved {out}/student_{args.size}.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
