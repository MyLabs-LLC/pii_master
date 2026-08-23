"""ai4privacy/pii-masking-300k as additional training data, in Nemotron's label space.

Why mix at all: the shipped student scores micro F1 0.934 on Nemotron and
**0.385 strict span recall on ai4privacy** (docs/STAGE2_INTEGRATION.md section
7.10). The gap is not mostly a capability gap -- located recall is 0.575, so
the model usually finds the right region and gets the boundaries wrong. It
learned span edges from US narrative prose, and ai4privacy is structured
JSON/XML key-value forms. That is a data-distribution problem, and the direct
fix is to train on both.

**The labels are mapped onto NEMOTRON's space, not merged into a bigger one.**
Extending the head would leave the new classes unsupervised by the teacher --
it emits 107 columns over Nemotron's inventory and knows nothing else -- so the
soft-target half of the distillation loss would go silent exactly where the new
data is meant to teach something. Mapping keeps the 111-class head, keeps the
teacher useful, and costs only the labels below that have no Nemotron
equivalent.

Scoped to English. The corpus is six languages, and the design is explicitly
US/English (DESIGN.md section 3); training on French addresses to fix English
JSON boundaries would be changing two variables at once.
"""

from __future__ import annotations

import re
from pathlib import Path

# ai4privacy label -> Nemotron label. Their taxonomy is largely a subset of
# Nemotron's, so most of this is renaming.
AI4P_TO_NEMOTRON: dict[str, str | None] = {
    "EMAIL": "email",
    "GIVENNAME1": "first_name",
    "GIVENNAME2": "first_name",
    "LASTNAME1": "last_name",
    "LASTNAME2": "last_name",
    "LASTNAME3": "last_name",
    "STREET": "street_address",
    # A bare house number in its own field ("building": "617"). It is part of a
    # street address and Nemotron has no separate label for it.
    "BUILDING": "street_address",
    "SECADDRESS": "street_address",
    "CITY": "city",
    "POSTCODE": "postcode",
    "STATE": "state",
    "COUNTRY": "country",
    "GEOCOORD": "coordinate",
    "USERNAME": "user_name",
    "TEL": "phone_number",
    "BOD": "date_of_birth",
    "DATE": "date",
    "TIME": "time",
    "SEX": "gender",
    "PASS": "password",
    "IDCARD": "national_id",
    # A passport IS a national identity document, and Nemotron has no passport
    # label. Mapping to national_id is better than dropping to O, which would
    # actively teach the student that passport numbers are not identifiers.
    "PASSPORT": "national_id",
    "DRIVERLICENSE": "certificate_license_number",
    "CARDISSUER": "company_name",
    # A salutation. Nemotron has no equivalent and it is not an identifier on
    # its own, so it becomes background -- deliberately, not by omission.
    "TITLE": None,
}

# SOCIALNUMBER covers both US-format SSNs and non-US national numbers under one
# label. Nemotron distinguishes them, and so do our detectors, so the format
# decides: nine digits in 3-2-4 grouping is an SSN, anything else is a national
# id. Collapsing both to one label would teach the student to call UK numbers
# SSNs, which is what the evaluation crosswalk had to be fixed for.
_US_SSN = re.compile(r"^\d{3}[- ]?\d{2}[- ]?\d{4}$")

# ipv4 vs ipv6 is likewise a format question, not an annotation one.
_IPV6 = re.compile(r"^[0-9A-Fa-f:]*:[0-9A-Fa-f:]*$")


def to_nemotron_label(label: str, value: str) -> str | None:
    """-> Nemotron label, or None for background. Raises on an unknown label."""
    if label == "SOCIALNUMBER":
        return "ssn" if _US_SSN.match(value.strip()) else "national_id"
    if label == "IP":
        return "ipv6" if _IPV6.match(value.strip()) else "ipv4"
    if label in AI4P_TO_NEMOTRON:
        return AI4P_TO_NEMOTRON[label]
    raise KeyError(f"unknown ai4privacy label {label!r}: update ai4privacy.py")


def read_split(data_dir, split: str = "train", limit: int | None = None,
               language: str = "English"):
    """-> (texts, spans) shaped exactly like data.read_split's Nemotron output.

    `spans` entries are dicts with 'label', 'start', 'end' so the existing
    encode() path needs no special case.
    """
    import pyarrow.parquet as pq

    files = sorted(Path(data_dir).glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {split} parquet under {data_dir}")
    texts, spans = [], []
    for path in files:
        table = pq.read_table(path,
                              columns=["source_text", "privacy_mask", "language"])
        for text, mask, lang in zip(table.column("source_text").to_pylist(),
                                    table.column("privacy_mask").to_pylist(),
                                    table.column("language").to_pylist()):
            if language not in ("*", lang):
                continue
            mapped = []
            for span in (mask or []):
                label = to_nemotron_label(span["label"], span.get("value", ""))
                if label is None:
                    continue
                mapped.append({"label": label, "start": span["start"],
                               "end": span["end"]})
            texts.append(text)
            spans.append(mapped)
            if limit and len(texts) >= limit:
                return texts, spans
    return texts, spans
