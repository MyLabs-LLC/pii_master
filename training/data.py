"""Nemotron-PII -> BIO token-classification tensors, aligned to the teacher tokenizer.

The dataset ships character spans, not token labels, so alignment happens here
via `return_offsets_mapping`. Two rules that matter:

  * A token is labelled B- if it is the first token overlapping a gold span,
    I- if it overlaps a later part, O otherwise. Partial overlaps count --
    tokenizers split identifiers ("4829471" -> "482", "9471") and dropping
    those would teach the student to ignore exactly the spans we care about.
  * Special tokens and padding get label -100 so they are ignored by the loss.

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

    for row, raw in enumerate(raw_spans):
        offsets = enc["offset_mapping"][row]
        mask = enc["attention_mask"][row]
        # Real tokens (offset end > start) are labelled O by default.
        real = (mask == 1) & (offsets[:, 1] > offsets[:, 0])
        labels[row][real] = LABEL2ID["O"]

        for span in parse_spans(raw):
            label = span["label"]
            if label not in LABEL2ID and f"B-{label}" not in LABEL2ID:
                raise KeyError(f"unknown label {label!r}: update NEMOTRON_LABELS")
            start, end = span["start"], span["end"]
            hit = np.where(real & (offsets[:, 0] < end) & (offsets[:, 1] > start))[0]
            for position, token_index in enumerate(hit):
                prefix = "B" if position == 0 else "I"
                labels[row][token_index] = LABEL2ID[f"{prefix}-{label}"]

    return enc["input_ids"], enc["attention_mask"], labels


class TaggingDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = torch.as_tensor(input_ids, dtype=torch.long)
        self.attention_mask = torch.as_tensor(attention_mask, dtype=torch.long)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, i):
        return {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_mask[i],
            "labels": self.labels[i],
        }
