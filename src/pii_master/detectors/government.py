"""Government-issued identifier detectors: SSNs and driver's licenses."""

from __future__ import annotations

import re

from ..entities import EntityType
from ..validators import ssn_ok
from .base import CueAnchoredIdDetector, RegexDetector


class SsnDetector(RegexDetector):
    # Backreferenced separator: consistently hyphenated or spaced, never
    # mixed. Bare 9-digit runs are deliberately not matched in v1 — the
    # false-positive flood outweighs the recall (docs/DESIGN.md section 7).
    name = "regex/ssn"
    entity_type = EntityType.SSN
    pattern = re.compile(r"(?<!\d)(\d{3})([- ])(\d{2})\2(\d{4})(?!\d)")
    base_confidence = 0.90
    space_confidence = 0.70
    use_digit_runs = True
    overshoot = 8

    def validate(self, match: re.Match[str]) -> float | None:
        if not ssn_ok(match.group(1), match.group(3), match.group(4)):
            return None
        if match.group(2) == " ":
            return self.space_confidence
        return self.base_confidence


class UsDriverLicenseDetector(CueAnchoredIdDetector):
    # Cue-anchored only: per-state format validation is deferred. The bare
    # "DL" cue requires a ":" or "#" separator to avoid matching prose.
    name = "regex/us_driver_license"
    entity_type = EntityType.US_DRIVER_LICENSE
    pattern = re.compile(
        r"(?i)\b(?:driver'?s?\s+licen[sc]e(?:\s+(?:no|num(?:ber)?|#))?\s*[:#]?"
        r"|DL\s*[:#])"
        r"\s*([A-Za-z0-9][A-Za-z0-9-]{3,12})\b"
    )
    base_confidence = 0.80
    hints = ("driver", "dl")
