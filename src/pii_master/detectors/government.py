"""Government-issued identifier detectors: Social Security numbers."""

from __future__ import annotations

import re

from ..entities import EntityType
from ..validators import ssn_ok
from .base import RegexDetector


class SsnDetector(RegexDetector):
    # Backreferenced separator: consistently hyphenated or spaced, never
    # mixed. Bare 9-digit runs are deliberately not matched in v1 — the
    # false-positive flood outweighs the recall (docs/DESIGN.md section 7).
    name = "regex/ssn"
    entity_type = EntityType.SSN
    pattern = re.compile(r"(?<!\d)(\d{3})([- ])(\d{2})\2(\d{4})(?!\d)")
    base_confidence = 0.90
    space_confidence = 0.70

    def validate(self, match: re.Match[str]) -> float | None:
        if not ssn_ok(match.group(1), match.group(3), match.group(4)):
            return None
        if match.group(2) == " ":
            return self.space_confidence
        return self.base_confidence
