"""Format-anchored detectors for HIPAA rows that used to be model-only.

SHIELD (arXiv 2605.03301, 2026) measured that *structured* PHI categories
transfer across institutions while institution-specific identifiers do not.
RECAP and the Track C leftover in docs/IMPROVEMENT_PLAN.md say the same
thing from the other direction: a checksum the model cannot verify is a
fusion-tier the model cannot earn.

These detectors are that missing rules path. Each one is the same shape as
Stage 1 — broad candidate, strict validator — and the types they emit join
``CHECKSUMMED_TYPES`` so a Luhn-class fact outranks the student on overlap
and a model span of the same type is re-validated before it may ship.
"""

from __future__ import annotations

import re

from ..entities import EntityType
from ..models import Entity
from ..validators import aba_ok, ein_ok, mac_ok, nanp_ok, swift_ok, vin_ok
from .base import RegexDetector
from .contact import _REFERENCE_CUE


class BankRoutingDetector(RegexDetector):
    """9-digit ABA routing number, cue-anchored, checksum-validated."""

    name = "regex/bank_routing"
    entity_type = EntityType.BANK_ROUTING
    pattern = re.compile(
        r"(?i)\b(?:routing(?:\s+(?:number|no|#))?|aba(?:\s+(?:number|no|#))?|"
        r"rtn|transit(?:\s+(?:number|no|#))?)\s*[:#]?\s*"
        r"(\d{9})\b"
    )
    base_confidence = 0.90
    capture_group = 1
    hints = ("routing", "aba", "rtn", "transit")
    hint_lead = 0
    hint_window = 40
    overshoot = 16

    def validate(self, match: re.Match[str]) -> float | None:
        if not aba_ok(match.group(1)):
            return None
        return self.base_confidence


class VehicleIdDetector(RegexDetector):
    """17-character VIN with an ISO 3779 check digit."""

    name = "regex/vehicle_id"
    entity_type = EntityType.VEHICLE_ID
    pattern = re.compile(
        r"(?i)(?<![A-Za-z0-9])([A-HJ-NPR-Z0-9]{17})(?![A-Za-z0-9])"
    )
    base_confidence = 0.90
    window_pattern = re.compile(r"[A-HJ-NPR-Za-hj-npr-z0-9]{17}")
    overshoot = 4

    def validate(self, match: re.Match[str]) -> float | None:
        if not vin_ok(match.group(1)):
            return None
        return self.base_confidence


class SwiftBicDetector(RegexDetector):
    """ISO 9362 BIC. Cue-anchored, or 11-char / digit-bearing 8-char free.

    All-alpha 8-character English words (SOFTWARE) are structurally valid
    BICs, so a cue or a digit is required before we will emit one.
    """

    name = "regex/swift_bic"
    entity_type = EntityType.SWIFT_BIC
    pattern = re.compile(
        r"(?i)\b(?:swift(?:\s*(?:code|bic))?|bic(?:\s*code)?)\s*[:#]?\s*"
        r"([A-Za-z]{4}[A-Za-z]{2}[A-Za-z0-9]{2}(?:[A-Za-z0-9]{3})?)\b"
    )
    base_confidence = 0.85
    capture_group = 1
    hints = ("swift", "bic")
    hint_lead = 0
    hint_window = 24
    overshoot = 16
    # Also pick up digit-bearing standalone BICs far from a cue.
    _standalone = re.compile(
        r"\b([A-Za-z]{4}[A-Za-z]{2}[A-Za-z0-9]{2}(?:[A-Za-z0-9]{3})?)\b"
    )

    def detect(self, text: str) -> list:
        # Parent handles the cue-windowed matches. Then scan once more for
        # standalone BICs that contain a digit or are 11 characters — those
        # do not look like English words.
        entities = super().detect(text)
        seen = {(e.start, e.end) for e in entities}
        for match in self._standalone.finditer(text):
            code = match.group(1)
            span = match.span(1)
            if span in seen:
                continue
            if not swift_ok(code):
                continue
            # Cue-less matches must contain a digit. All-alpha 8- and
            # 11-character English words (SOFTWARE, withholding,
            # Beneficiary, transferred) are structurally valid BICs.
            if not any(ch.isdigit() for ch in code):
                continue
            entities.append(Entity(
                type=self.entity_type,
                start=span[0],
                end=span[1],
                text=code,
                confidence=self.base_confidence,
                detector=self.name,
            ))
            seen.add(span)
        return entities

    def validate(self, match: re.Match[str]) -> float | None:
        code = match.group(1)
        if not swift_ok(code):
            return None
        return self.base_confidence


class MacAddressDetector(RegexDetector):
    """Canonical hex-colon or hex-dash MAC address."""

    name = "regex/mac_address"
    entity_type = EntityType.MAC_ADDRESS
    pattern = re.compile(
        r"(?<![\w:])(?:[0-9A-Fa-f]{2}([:-]))(?:[0-9A-Fa-f]{2}\1){4}[0-9A-Fa-f]{2}(?![\w:])"
    )
    base_confidence = 0.90
    window_pattern = re.compile(r"[0-9A-Fa-f]{2}[:-][0-9A-Fa-f]{2}")
    overshoot = 20

    def validate(self, match: re.Match[str]) -> float | None:
        if not mac_ok(match.group(0)):
            return None
        return self.base_confidence


class FaxNumberDetector(RegexDetector):
    """NANP number whose left window carries a fax cue.

    HIPAA lists fax separately from telephone (#5 vs #4). Without the cue
    we keep emitting PHONE_US; with it, this detector wins the overlap on
    confidence so the report cites the right Safe Harbor row.
    """

    name = "regex/fax"
    entity_type = EntityType.FAX_NUMBER
    pattern = re.compile(
        r"(?<!\d)(?<!\d\.)"
        r"(?:\+?1[-. ]?)?"
        r"(?:\((\d{3})\)|(\d{3}))"
        r"[-. ]?(\d{3})[-. ]?(\d{4})"
        r"(?!\d)(?!\.\d)"
    )
    base_confidence = 0.90
    hints = ("fax", "facsimile", "telefax")
    hint_lead = 8
    hint_window = 40
    overshoot = 20
    cue_window = 40
    _fax_cue = re.compile(r"(?i)\b(?:fax|facsimile|telefax)\b")

    def validate(self, match: re.Match[str]) -> float | None:
        area = match.group(1) or match.group(2)
        exchange = match.group(3)
        if not nanp_ok(area, exchange):
            return None
        start = match.start()
        left = match.string[max(0, start - self.cue_window):start]
        if not self._fax_cue.search(left):
            return None
        raw = match.group(0)
        if raw.isdigit() and _REFERENCE_CUE.search(left):
            return None
        return self.base_confidence


class TaxIdDetector(RegexDetector):
    """US EIN (employer tax id), cue-anchored, prefix-validated."""

    name = "regex/tax_id"
    entity_type = EntityType.TAX_ID
    pattern = re.compile(
        r"(?i)\b(?:ein|employer\s+id(?:entification)?(?:\s+number)?|"
        r"tax(?:payer)?\s+id(?:entification)?(?:\s+number)?|tin)"
        r"\s*[:#]?\s*"
        r"(\d{2}-?\d{7})\b"
    )
    base_confidence = 0.85
    capture_group = 1
    hints = ("ein", "employer", "tax id", "tax identification", "tin", "taxpayer")
    hint_lead = 0
    hint_window = 48
    overshoot = 16

    def validate(self, match: re.Match[str]) -> float | None:
        digits = match.group(1).replace("-", "")
        if not ein_ok(digits):
            return None
        return self.base_confidence
