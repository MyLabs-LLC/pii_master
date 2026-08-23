"""Distil kalyan-ks/ettin-68m-nemotron-pii into the CNN student.

Loss = alpha * KL(student || teacher soft targets, temperature T)   [word starts]
     + (1 - alpha) * CE(student, gold BIO labels)                    [every token]

Nemotron-PII ships gold spans, so this is distillation *plus* supervision, not
pure distillation -- the teacher supplies dark knowledge (which wrong labels are
plausible) while the gold labels keep it anchored.

The teacher is only trustworthy on the first subword of each word: on 300 train
documents its argmax matches gold on 99.0% of word-start tokens and 64% of
within-word continuations, because it was trained with the continuations masked
out (-100) and never learned them. But dropping the soft term on continuations
leaves them supervised by gold CE alone, and a student trained that way stops
labelling a span partway through a word -- measured, the single largest error
class was truncated span ends ("271210785" tagged as "2712107").

`--soft-scope` selects what to do about that. All three were trained and
scored on the 100k holdout (docs/DISTILLATION_RESULTS.md):

  word_start        KL only on word starts, the teacher's supervised region.
  broadcast         copy the word-start distribution over the whole word.
                    Much worse, and the reason is instructive: a whitespace
                    word is not a span. "6789," is one word whose last token is
                    outside the span, so broadcasting teaches the student to
                    swallow trailing punctuation -- exactly the boundary the
                    exact-match metric grades.
  word_homogeneous  (default) broadcast, but only onto tokens whose gold label
                    already matches the word start: the teacher's word-level
                    judgement applied only where the word really is one label.

Measured on the xs student, 3 epochs, 20k holdout documents, mapped-type F1:

    word_homogeneous  0.804     (55-label F1 0.691,  801 spurious spans)
    word_start        0.792     (55-label F1 0.669, 1471 spurious spans)
    broadcast         0.492     (100k holdout)

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


# Logit written into teacher columns the teacher does not have. Large and
# negative, but finite: -inf would make KL produce 0 * -inf = NaN.
ABSENT_LOGIT = -1e4


def build_label_permutation(teacher_id2label: dict) -> tuple[torch.Tensor, int]:
    """Index tensor mapping teacher logit columns into our label order.

    `remap_teacher_logits(logits, perm)` is then in our order. Hard-fails
    rather than silently mis-aligning, which would train the student on
    scrambled targets.

    One legitimate mismatch is tolerated, loudly. ettin-68m has 107 labels to
    our 111: it tags BIO at word level, so the four types that never span two
    words (`ssn`, `cvv`, `gender`, `employee_id`) have a B- column and no I-
    column. Those get a constant `ABSENT_LOGIT` column -- the teacher assigns
    a class it cannot emit zero probability, which is exactly true. A type
    missing *entirely* is still a hard failure: that is real misalignment.
    """
    normalised = {}
    for idx, name in teacher_id2label.items():
        key = str(name).strip()
        for variant in (key, key.upper(), key.lower(), key.replace("_", "-")):
            normalised.setdefault(variant, int(idx))
    n_teacher = max(int(i) for i in teacher_id2label) + 1
    perm, missing, padded = [], [], []
    for i in range(NUM_LABELS):
        ours = ID2LABEL[i]
        for candidate in (ours, ours.upper(), ours.lower(), ours.replace("_", "-")):
            if candidate in normalised:
                perm.append(normalised[candidate])
                break
        else:
            prefix, _, type_name = ours.partition("-")
            sibling = f"{'I' if prefix == 'B' else 'B'}-{type_name}"
            if type_name and sibling in normalised:
                perm.append(n_teacher)      # the appended ABSENT_LOGIT column
                padded.append(ours)
            else:
                missing.append(ours)
    if missing:
        raise SystemExit(
            f"teacher label space misses {len(missing)} of our labels, e.g. {missing[:8]}.\n"
            f"teacher labels: {sorted(set(teacher_id2label.values()))[:12]} ...\n"
            "Fix build_label_permutation before training -- do not train through this."
        )
    if padded:
        print(f"  teacher has no column for {len(padded)} labels "
              f"({', '.join(padded)}); soft target pinned to zero probability")
    return torch.tensor(perm, dtype=torch.long), n_teacher


def remap_teacher_logits(logits: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Teacher logits (..., n_teacher) -> (..., NUM_LABELS) in our label order."""
    pad = logits.new_full(logits.shape[:-1] + (1,), ABSENT_LOGIT)
    return torch.cat([logits, pad], dim=-1)[..., perm]


def broadcast_to_words(teacher_logits, word_src):
    """Copy each word-start token's distribution over the rest of its word."""
    index = word_src.unsqueeze(-1).expand(-1, -1, teacher_logits.shape[-1])
    return teacher_logits.gather(1, index)


def distillation_loss(student_logits, teacher_logits, labels, alpha, temperature,
                      soft_mask=None):
    """Hard CE on every labelled token, soft KL where the teacher is trusted.

    `soft_mask` None means the teacher logits are already valid everywhere
    (the broadcast scope); otherwise KL is limited to the masked positions.
    """
    flat_labels = labels.view(-1)
    active = flat_labels != -100
    s_all = student_logits.reshape(-1, student_logits.shape[-1])
    s = s_all[active]
    y = flat_labels[active]
    if s.numel() == 0:
        return student_logits.sum() * 0.0, 0.0, 0.0
    hard = F.cross_entropy(s, y)

    trusted = active if soft_mask is None else (active & soft_mask.reshape(-1))
    if not bool(trusted.any()):
        return (1 - alpha) * hard, 0.0, hard.detach().item()
    st = s_all[trusted]
    tt = teacher_logits.reshape(-1, teacher_logits.shape[-1])[trusted]
    soft = F.kl_div(
        F.log_softmax(st / temperature, dim=-1),
        F.log_softmax(tt / temperature, dim=-1),
        reduction="batchmean", log_target=True,
    ) * (temperature ** 2)
    return alpha * soft + (1 - alpha) * hard, soft.detach().item(), hard.detach().item()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, help="dir holding Nemotron parquet")
    ap.add_argument("--mix-dir",
                    help="dir holding ai4privacy parquet; its English rows are "
                         "mapped into Nemotron's label space and mixed in. See "
                         "ai4privacy.py and docs/STAGE2_INTEGRATION.md 7.10")
    ap.add_argument("--mix-limit", type=int, default=None,
                    help="cap the mixed-in rows (default: all English rows)")
    ap.add_argument("--size", default="xs", choices=list(LADDER))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--temperature", type=float, default=3.0)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None, help="rows, for a smoke run")
    ap.add_argument("--soft-scope", default="word_homogeneous",
                    choices=["word_start", "broadcast", "word_homogeneous"],
                    help="where the KL term applies; see the module docstring")
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
    perm, n_teacher = build_label_permutation(teacher.config.id2label)
    perm = perm.to(device)
    print(f"label permutation OK: {NUM_LABELS} labels aligned to the teacher")

    print("loading + encoding train split ...")
    texts, spans = read_split(args.data_dir, "train", limit=args.limit)
    print(f"  Nemotron: {len(texts):,} documents")
    if args.mix_dir:
        import ai4privacy

        mix_texts, mix_spans = ai4privacy.read_split(
            args.mix_dir, "train", limit=args.mix_limit)
        print(f"  ai4privacy (English, mapped): {len(mix_texts):,} documents "
              f"-> {len(mix_texts) / (len(texts) + len(mix_texts)):.0%} of the "
              f"mixture")
        texts, spans = texts + mix_texts, spans + mix_spans
    ids, mask, labels, word_src = encode(texts, spans, tokenizer, args.max_length)
    train = DataLoader(TaggingDataset(ids, mask, labels, word_src),
                       batch_size=args.batch_size,
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
            assert t_logits.shape[-1] == n_teacher
            t_logits = remap_teacher_logits(t_logits.float(), perm)
            word_src = b["word_src"]
            if args.soft_scope == "word_start":
                soft_mask = word_src == torch.arange(
                    word_src.shape[1], device=word_src.device)
            else:
                t_logits = broadcast_to_words(t_logits, word_src)
                soft_mask = None
                if args.soft_scope == "word_homogeneous":
                    soft_mask = b["labels"] == b["labels"].gather(1, word_src)
            with torch.autocast(device.type, torch.bfloat16, enabled=use_bf16):
                s_logits = student(b["input_ids"], b["attention_mask"])
            loss, soft, hard = distillation_loss(
                s_logits.float(), t_logits, b["labels"], args.alpha, args.temperature,
                soft_mask)
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
            "soft_scope": args.soft_scope, "alpha": args.alpha,
            "mixed_corpora": ["nvidia/Nemotron-PII"] + (
                ["ai4privacy/pii-masking-300k (English, label-mapped)"]
                if args.mix_dir else []),
            "temperature": args.temperature, "lr": args.lr,
            "epochs": args.epochs, "max_length": args.max_length,
        }, indent=2))
        print(f"  saved {out}/student_{args.size}.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
