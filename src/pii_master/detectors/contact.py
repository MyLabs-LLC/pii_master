"""Contact-information detectors: email addresses and US phone numbers."""

from __future__ import annotations

import re

from ..entities import EntityType
from ..validators import nanp_ok
from .base import RegexDetector

# A bare 10-digit run preceded by one of these is a reference number, not a
# phone. Formatted numbers -- (415) 555-2671, 415.555.2671 -- keep their
# shape and are never suppressed.
_REFERENCE_CUE = re.compile(
    r"(?i)\b(?:confirmation|conf|order|tracking|invoice|reference|ref|receipt|"
    r"ticket|case|claim|transaction|txn|po|account|acct)\b\W{0,12}$"
)


class EmailDetector(RegexDetector):
    # Pragmatic pattern, not RFC 5322: quoted-string local parts and
    # comment syntax are legal but unseen in real documents.
    name = "regex/email"
    entity_type = EntityType.EMAIL
    pattern = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    base_confidence = 0.95
    # Every email contains "@"; the RFC caps local parts at 64 chars
    # (lead 128 is belt-and-braces), domains fit inside the overshoot.
    hints = ("@",)
    hint_lead = 128
    hint_window = 2
    overshoot = 320

    def validate(self, match: re.Match[str]) -> float | None:
        local = match.group(0).split("@", 1)[0]
        if local.startswith(".") or local.endswith("."):
            return None
        return self.base_confidence


class UsPhoneDetector(RegexDetector):
    # 10 bare digits colliding with order/tracking numbers are the known
    # false-positive class, hence base confidence below the format types.
    name = "regex/phone_us"
    entity_type = EntityType.PHONE_US
    # Boundary lookarounds reject candidates embedded in longer digit runs
    # or dotted version-like strings, while still allowing a sentence-ending
    # period right after the number.
    pattern = re.compile(
        r"(?<!\d)(?<!\d\.)"
        r"(?:\+?1[-. ]?)?"
        r"(?:\((\d{3})\)|(\d{3}))"
        r"[-. ]?(\d{3})[-. ]?(\d{4})"
        r"(?!\d)(?!\.\d)"
    )
    base_confidence = 0.85
    use_digit_runs = True
    overshoot = 8

    reference_window = 32

    def validate(self, match: re.Match[str]) -> float | None:
        area = match.group(1) or match.group(2)
        exchange = match.group(3)
        if not nanp_ok(area, exchange):
            return None
        raw = match.group(0)
        if raw.isdigit():  # bare run: no parens, separators or +1 prefix
            start = match.start()
            left = match.string[max(0, start - self.reference_window):start]
            if _REFERENCE_CUE.search(left):
                return None
        return self.base_confidence
