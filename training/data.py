"""Nemotron-PII -> BIO token-classification tensors, aligned to the teacher tokenizer.

The dataset ships character spans, not token labels, so alignment happens here
via `return_offsets_mapping`. Two rules that matter:

  * BIO is assigned at WORD level, not subword level: every subword of the
    span's first whitespace-delimited word gets B-, every subword of a later
    word gets I-. That is the teacher's own convention, verified by reading
    its predictions -- it tags "123-45-6789" as five consecutive B-ssn tokens
    and "4821 Maple Avenue" as B,B,I,I. Matching it matters because the soft
    targets and the gold labels have to agree token by token; a per-subword
    B/I scheme disagrees on every continuation token and makes the two halves
    of the distillation loss pull against each other. It is also why the
    teacher has no I-ssn/I-cvv/I-gender/I-employee_id column: those types
    never span two words (0 of 24,862 spans in train).
  * Partial overlaps count -- tokenizers split identifiers ("4829471" ->
    "482", "9471") and dropping those would teach the student to ignore
    exactly the spans we care about.
  * Special tokens and padding get label -100 so they are ignored by the loss.
  * `encode` also returns a word-start index: for every token, the position of
    the first subword of its word. Under word-level BIO every subword of a word
    carries the SAME gold label, so the teacher's distribution at the word start
    is a valid target for the whole word -- train.py broadcasts it there, which
    is how continuation tokens get a trustworthy soft target instead of none.
    The boolean mask of word starts is derived from it (`index == position`).

    Why this matters: the FIRST SUBWORD OF EACH WORD is the only place
    the teacher's logits are trustworthy. Measured
    against gold on 300 train documents it agrees 99.0% on word-start tokens
    and 64% on within-word continuations, because it was trained with the
    continuations masked out (-100) and never learned them. Distilling KL on
    the noisy positions would spend most of the soft loss on the teacher's
    unsupervised region.

`spans` in the parquet is a Python-repr string, not JSON (see
docs/NEMOTRON_PII_TAGS.md), hence ast.literal_eval.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# The 55 labels, frozen in sorted order so label ids are stable across runs.
# Sourced from docs/NEMOTRON_PII_TAGS.md; asserted against the data at load.
NEMOTRON_LABELS = sorted([
    "account_number", "age", "api_key", "bank_routing_number", "biometric_identifier",
    "blood_type", "certificate_license_number", "city", "company_name", "coordinate",
    "country", "county", "credit_debit_card", "customer_id", "cvv", "date",
    "date_of_birth", "date_time", "device_identifier", "education_level", "email",
    "employee_id", "employment_status", "fax_number", "first_name", "gender",
    "health_plan_beneficiary_number", "http_cookie", "ipv4", "ipv6", "language",
    "last_name", "license_plate", "mac_address", "medical_record_number",
    "national_id", "occupation", "password", "phone_number", "pin", "political_view",
    "postcode", "race_ethnicity", "religious_belief", "sexuality", "ssn", "state",
    "street_address", "swift_bic", "tax_id", "time",
    "unique_id", "url", "user_name", "vehicle_identifier",
])

LABEL_NAMES = ["O"] + [f"{p}-{l}" for l in NEMOTRON_LABELS for p in ("B", "I")]
LABEL2ID = {name: i for i, name in enumerate(LABEL_NAMES)}
ID2LABEL = {i: name for name, i in LABEL2ID.items()}
NUM_LABELS = len(LABEL_NAMES)


def parse_spans(raw):
    if raw is None:
        return []
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return ast.literal_eval(raw)


def read_split(data_dir: str | Path, split: str, limit: int | None = None):
    import pyarrow.parquet as pq

    files = sorted(Path(data_dir).glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {split} parquet under {data_dir}")
    texts, spans = [], []
    for path in files:
        table = pq.read_table(path, columns=["text", "spans"])
        texts.extend(table.column("text").to_pylist())
        spans.extend(table.column("spans").to_pylist())
        if limit and len(texts) >= limit:
            break
    if limit:
        texts, spans = texts[:limit], spans[:limit]
    return texts, spans


def first_word_end(text: str, start: int, end: int) -> int:
    """Char offset where the span's first whitespace-delimited word ends.

    Tokens starting before this are B-, tokens after it are I-. Leading
    whitespace inside the span is skipped, otherwise a span that begins with a
    space would have no B- token at all.
    """
    segment = text[start:end]
    lead = len(segment) - len(segment.lstrip())
    for offset, char in enumerate(segment[lead:]):
        if char.isspace():
            return start + lead + offset
    return end


def word_start_mask(text: str, offsets, real) -> np.ndarray:
    """True where a token is the first subword of a whitespace-delimited word.

    Tokenizer offsets include the leading space (" Jane" -> [7,12]), so a token
    starts a word when its first character is whitespace, or it sits at the
    very start of the document.
    """
    mask = np.zeros(offsets.shape[0], dtype=bool)
    for index in np.where(real)[0]:
        start = int(offsets[index][0])
        mask[index] = start == 0 or text[start].isspace() or text[start - 1].isspace()
    return mask


def word_start_index(mask: np.ndarray) -> np.ndarray:
    """For each position, the index of the word start it belongs to.

    Running maximum over word-start positions. Tokens before the first word
    start (specials) point at themselves; their labels are -100 anyway.
    """
    positions = np.arange(mask.shape[0])
    source = np.maximum.accumulate(np.where(mask, positions, -1))
    return np.where(source < 0, positions, source).astype(np.int64)


def encode(texts, raw_spans, tokenizer, max_length: int = 512):
    """Tokenize and project character spans onto token labels."""
    enc = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors="np",
    )
    labels = np.full(enc["input_ids"].shape, -100, dtype=np.int64)
    word_src = np.zeros(enc["input_ids"].shape, dtype=np.int64)

    for row, raw in enumerate(raw_spans):
        offsets = enc["offset_mapping"][row]
        mask = enc["attention_mask"][row]
        # Real tokens (offset end > start) are labelled O by default.
        real = (mask == 1) & (offsets[:, 1] > offsets[:, 0])
        labels[row][real] = LABEL2ID["O"]
        word_src[row] = word_start_index(word_start_mask(texts[row], offsets, real))

        for span in parse_spans(raw):
            label = span["label"]
            if label not in LABEL2ID and f"B-{label}" not in LABEL2ID:
                raise KeyError(f"unknown label {label!r}: update NEMOTRON_LABELS")
            start, end = span["start"], span["end"]
            hit = np.where(real & (offsets[:, 0] < end) & (offsets[:, 1] > start))[0]
            boundary = first_word_end(texts[row], start, end)
            for position, token_index in enumerate(hit):
                # Word level, matching the teacher: B- until the first word of
                # the span ends, I- after it. Position 0 is forced to B- so
                # every span still decodes even if a token straddles the edge.
                inside_first_word = offsets[token_index][0] < boundary
                prefix = "B" if position == 0 or inside_first_word else "I"
                labels[row][token_index] = LABEL2ID[f"{prefix}-{label}"]

    return enc["input_ids"], enc["attention_mask"], labels, word_src


class TaggingDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels, word_src=None):
        self.input_ids = torch.as_tensor(input_ids, dtype=torch.long)
        self.attention_mask = torch.as_tensor(attention_mask, dtype=torch.long)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        if word_src is None:
            word_src = np.arange(self.labels.shape[1])[None, :].repeat(
                self.labels.shape[0], axis=0)
        self.word_src = torch.as_tensor(word_src, dtype=torch.long)

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, i):
        return {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_mask[i],
            "labels": self.labels[i],
            "word_src": self.word_src[i],
        }
