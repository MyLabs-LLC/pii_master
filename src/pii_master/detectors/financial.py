"""Financial identifier detectors: payment card numbers."""

from __future__ import annotations

import re

from ..entities import EntityType
from ..validators import card_iin_known, luhn_ok
from .base import RegexDetector


class CreditCardDetector(RegexDetector):
    # Broad candidate net; the Luhn checksum is the hard reject that makes
    # this the highest-precision detector. Known IIN prefixes boost
    # confidence but never reject — IIN ranges churn.
    name = "regex/credit_card"
    entity_type = EntityType.CREDIT_CARD
    pattern = re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])")
    base_confidence = 0.80
    iin_confidence = 0.95

    def validate(self, match: re.Match[str]) -> float | None:
        raw = match.group(0)
        if " " in raw and "-" in raw:
            return None
        digits = raw.replace(" ", "").replace("-", "")
        if not 13 <= len(digits) <= 19:
            return None
        if not luhn_ok(digits):
            return None
        return self.iin_confidence if card_iin_known(digits) else self.base_confidence
