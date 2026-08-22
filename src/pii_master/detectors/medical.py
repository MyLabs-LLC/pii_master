"""PHI-flavored detectors: birth dates and medical record numbers.

Both are context-anchored: a bare date or a bare numeric ID is far too
ambiguous for rules, so v1 only fires when an explicit cue is present.
Cue-free detection is a Stage 2 (learned NER) objective.
"""

from __future__ import annotations

import re

from ..entities import EntityType
from ..validators import plausible_dob
from .base import RegexDetector

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

_BIRTH_CUE = re.compile(
    r"(?i)\b(?:dob|date\s+of\s+birth|birth\s*date|born(?:\s+on)?)\b"
)


class DateOfBirthDetector(RegexDetector):
    # Bare dates are quasi-identifiers with a huge false-positive surface,
    # so a date only becomes DATE_DOB when a birth cue appears in the
    # window of text just before it (docs/DESIGN.md sections 6-7).
    name = "regex/date_dob"
    entity_type = EntityType.DATE_DOB
    pattern = re.compile(
        r"(?<!\d)(?:"
        r"(?P<m1>\d{1,2})(?P<sep>[/-])(?P<d1>\d{1,2})(?P=sep)(?P<y1>\d{4})"
        r"|(?P<y2>\d{4})-(?P<m2>\d{2})-(?P<d2>\d{2})"
        r"|(?P<mon>Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
        r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|"
        r"Nov(?:ember)?|Dec(?:ember)?)\s+(?P<d3>\d{1,2}),?\s+(?P<y3>\d{4})"
        r")(?!\d)",
        re.IGNORECASE,
    )
    base_confidence = 0.90
    cue_window = 40

    def validate(self, match: re.Match[str]) -> float | None:
        g = match.groupdict()
        if g["y1"]:
            year, month, day = int(g["y1"]), int(g["m1"]), int(g["d1"])
        elif g["y2"]:
            year, month, day = int(g["y2"]), int(g["m2"]), int(g["d2"])
        else:
            year = int(g["y3"])
            month = _MONTHS[g["mon"][:3].lower()]
            day = int(g["d3"])
        if not plausible_dob(year, month, day):
            return None
        window_start = max(0, match.start() - self.cue_window)
        if not _BIRTH_CUE.search(match.string[window_start:match.start()]):
            return None
        return self.base_confidence


class MrnDetector(RegexDetector):
    # MRNs have no universal format — every hospital system mints its own —
    # so v1 detects them by their labels. The emitted span covers only the
    # captured ID, not the cue.
    name = "regex/mrn"
    entity_type = EntityType.MRN
    pattern = re.compile(
        r"(?i)\b(?:MRN|medical\s+record\s+(?:no|num(?:ber)?|#)|chart\s*#?)"
        r"\s*[:#]?\s*([A-Za-z0-9][A-Za-z0-9-]{4,11})\b"
    )
    base_confidence = 0.85
    capture_group = 1

    def validate(self, match: re.Match[str]) -> float | None:
        if sum(ch.isdigit() for ch in match.group(1)) < 3:
            return None
        return self.base_confidence
