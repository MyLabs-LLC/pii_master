"""Unified, leakage-aware loaders for the priority PII tagging experiments.

The source corpora do not share one manifest schema and several carry only
partial/coarse labels.  This module normalizes provenance and paths without
inventing negatives: ``label_complete`` is true only when the source explicitly
claims a complete sensitive-tag catalogue for that document.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import zipfile
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from html import unescape
from pathlib import Path
from typing import Any

READ_WINDOW = 20_000
SENSITIVE_PREFIXES = ("sensitive_pii_", "sensitive_pci_", "sensitive_phi_")

PRIORITY_TAGS = (
    "sensitive_pii_social_security_number",
    "sensitive_pci_individual_taxpayer_identification_number_itin",
    "sensitive_phi_medical_record_number_mrn",
    "sensitive_phi_health_plan_beneficiary_number",
    "sensitive_phi_patient_id_number",
    "sensitive_pci_bank_account_number",
    "sensitive_pci_credit_card_number",
    "sensitive_pci_iban",
    "sensitive_pii_passport_number",
    "sensitive_pii_driver_s_license_number",
    "sensitive_pii_military_identification_number",
    "sensitive_pii_visa_number",
    "sensitive_pii_password",
    "sensitive_pii_personal_identification_number_pin",
    "sensitive_pii_full_name",
    "sensitive_pii_address",
)

COARSE_ENTITY_MAP = {
    "address": "sensitive_pii_address",
    "bank account": "sensitive_pci_bank_account_number",
    "bank account number": "sensitive_pci_bank_account_number",
    "credit card": "sensitive_pci_credit_card_number",
    "credit card number": "sensitive_pci_credit_card_number",
    "driver license": "sensitive_pii_driver_s_license_number",
    "driver's license": "sensitive_pii_driver_s_license_number",
    "drivers license": "sensitive_pii_driver_s_license_number",
    "email": "sensitive_pii_email",
    "email address": "sensitive_pii_email",
    "full name": "sensitive_pii_full_name",
    "health plan beneficiary number": "sensitive_phi_health_plan_beneficiary_number",
    "iban": "sensitive_pci_iban",
    "itin": "sensitive_pci_individual_taxpayer_identification_number_itin",
    "medical record number": "sensitive_phi_medical_record_number_mrn",
    "military id": "sensitive_pii_military_identification_number",
    "mrn": "sensitive_phi_medical_record_number_mrn",
    "passport": "sensitive_pii_passport_number",
    "passport number": "sensitive_pii_passport_number",
    "password": "sensitive_pii_password",
    "patient id": "sensitive_phi_patient_id_number",
    "phone": "sensitive_pii_phone_number",
    "phone number": "sensitive_pii_phone_number",
    "pin": "sensitive_pii_personal_identification_number_pin",
    "social security number": "sensitive_pii_social_security_number",
    "ssn": "sensitive_pii_social_security_number",
    "state": "sensitive_pii_state",
    "street address": "sensitive_pii_street_number_and_name",
    "taxpayer identification number": "sensitive_pci_individual_taxpayer_identification_number_itin",
    "visa": "sensitive_pii_visa_number",
    "visa number": "sensitive_pii_visa_number",
    "zip": "sensitive_pii_zip_code",
    "zip code": "sensitive_pii_zip_code",
}

_TEXT_EXTENSIONS = {
    "",
    ".csv",
    ".htm",
    ".html",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".rtf",
    ".text",
    ".tsv",
    ".txt",
    ".xml",
}
_ZIP_XML_EXTENSIONS = {".docx", ".pptx", ".xlsx"}
_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class CorpusRow:
    dataset: str
    split: str
    uid: str
    path: str
    labels: tuple[str, ...]
    native_labels: tuple[str, ...]
    label_complete: bool
    provenance: str
    source_corpus: str
    supplied_hash: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["labels"] = list(self.labels)
        data["native_labels"] = list(self.native_labels)
        return data


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc


def iter_raw_rows(dataset_dir: Path) -> Iterator[dict[str, Any]]:
    """Yield a dataset's rows, preferring streaming labels.jsonl when present."""
    labels_path = dataset_dir / "labels.jsonl"
    if labels_path.is_file():
        yield from _iter_jsonl(labels_path)
        return
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest if isinstance(manifest, list) else manifest.get("rows", [])
    if not isinstance(rows, list):
        raise TypeError(f"manifest rows are not a list: {manifest_path}")
    yield from rows


def _safe_relative(candidate: str) -> str:
    candidate = candidate.strip()
    if not candidate:
        return ""
    path = Path(candidate)
    if path.is_absolute() or ".." in path.parts:
        return ""
    return candidate


def resolve_document_path(dataset_dir: Path, row: dict[str, Any]) -> Path:
    """Resolve a manifest row to its preferred extracted text or source file."""
    candidates = (
        row.get("text_path"),
        row.get("file"),
        row.get("path"),
        row.get("zip_path"),
        row.get("src"),
    )
    for raw in candidates:
        if not isinstance(raw, str) or not raw.strip():
            continue
        raw_path = Path(raw)
        if raw_path.is_absolute() and raw_path.is_file():
            return raw_path
        relative = _safe_relative(raw)
        if relative:
            path = dataset_dir / relative
            if path.is_file():
                return path
    # Return the most useful expected path for an actionable missing-file record.
    for raw in candidates:
        if isinstance(raw, str) and _safe_relative(raw):
            return dataset_dir / raw
    return dataset_dir / "__missing_document_path__"


def _sensitive(labels: Iterable[Any]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                str(label)
                for label in labels
                if str(label).startswith(SENSITIVE_PREFIXES)
            }
        )
    )


def normalize_row(dataset_dir: Path, raw: dict[str, Any], index: int) -> CorpusRow:
    gold = [str(value) for value in raw.get("gold", [])]
    labels = set(_sensitive(gold))
    native = {value for value in gold if not value.startswith(SENSITIVE_PREFIXES)}
    coarse_found = False
    for entity in raw.get("pii_entities", []) or []:
        normalized = _SPACE_RE.sub(" ", str(entity).strip().lower())
        mapped = COARSE_ENTITY_MAP.get(normalized)
        if mapped:
            labels.add(mapped)
            coarse_found = True
        elif normalized:
            native.add(f"coarse_pii::{normalized}")
    native_label = raw.get("document_type") or raw.get("label")
    if native_label:
        native.add(f"native::{native_label}")
    path = resolve_document_path(dataset_dir, raw)
    uid = str(raw.get("uid") or raw.get("doc_id") or raw.get("n") or f"row-{index}")
    # A source that emits catalog gold is complete for its declared catalogue.
    # Coarse entity lists are positive-only observations and therefore masked.
    label_complete = "gold" in raw and (
        bool(labels) or dataset_dir.name.startswith("pii")
    )
    if coarse_found and "gold" not in raw:
        label_complete = False
    return CorpusRow(
        dataset=dataset_dir.name,
        split=str(
            raw.get("split") or ("eval" if "eval" in dataset_dir.name else "train")
        ),
        uid=uid,
        path=str(path),
        labels=tuple(sorted(labels)),
        native_labels=tuple(sorted(native)),
        label_complete=label_complete,
        provenance=str(
            raw.get("provenance") or raw.get("label_provenance") or "unknown"
        ),
        source_corpus=str(
            raw.get("source_corpus") or raw.get("corpus") or dataset_dir.name
        ),
        supplied_hash=str(raw.get("sha256_read_window") or raw.get("sha256") or ""),
    )


def iter_corpus(dataset_dir: Path) -> Iterator[CorpusRow]:
    for index, raw in enumerate(iter_raw_rows(dataset_dir)):
        yield normalize_row(dataset_dir, raw, index)


def _xml_archive_text(path: Path, limit: int) -> str:
    chunks: list[str] = []
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith(".xml"):
                continue
            if not name.startswith(("word/", "ppt/slides/", "xl/sharedStrings")):
                continue
            with archive.open(name) as stream:
                payload = stream.read(limit * 20).decode("utf-8", errors="ignore")
            chunks.append(_TAG_RE.sub(" ", payload))
            if sum(map(len, chunks)) >= limit:
                break
    return " ".join(chunks)


def read_document(path: Path, *, limit: int = READ_WINDOW) -> str:
    """Extract a deterministic read window without adding heavyweight parsers."""
    suffix = path.suffix.lower()
    if suffix in _TEXT_EXTENSIONS:
        with path.open("rb") as stream:
            text = stream.read(limit * 20).decode("utf-8", errors="ignore")
        if suffix in {".htm", ".html", ".xml"}:
            text = _TAG_RE.sub(" ", text)
    elif suffix in _ZIP_XML_EXTENSIONS:
        text = _xml_archive_text(path, limit)
    elif suffix == ".pdf":
        completed = subprocess.run(
            ["pdftotext", "-layout", str(path), "-"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=20,
        )
        text = completed.stdout
    else:
        completed = subprocess.run(
            ["strings", "-n", "4", str(path)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=20,
        )
        text = completed.stdout
    return _SPACE_RE.sub(" ", unescape(text)).strip()[:limit]


def normalized_text_digest(text: str) -> str:
    normalized = _SPACE_RE.sub(" ", text).strip().casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def list_dataset_dirs(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "manifest.json").is_file()
    )


def summarize_tags(rows: Iterable[CorpusRow]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(row.labels)
    return counts
