"""Detector contract and the regex base class.

Detector is a structural Protocol so any detection strategy — the Stage 2
ONNX NER detector included — plugs into the pipeline without inheriting
anything from this package.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from ..entities import EntityType
from ..models import Entity


@runtime_checkable
class Detector(Protocol):
    name: str

    def detect(self, text: str) -> list[Entity]: ...


class RegexDetector:
    """Compiled pattern -> candidates -> validate hook -> Entities.

    Subclasses set the class attributes and override :meth:`validate`, which
    both filters false positives (return None to reject) and assigns the
    entity's confidence. capture_group narrows the emitted span to one group
    (e.g. the MRN id after its cue) instead of the whole match.
    """

    name: str
    entity_type: EntityType
    pattern: re.Pattern[str]
    base_confidence: float
    capture_group: int = 0

    def detect(self, text: str) -> list[Entity]:
        entities: list[Entity] = []
        for match in self.pattern.finditer(text):
            confidence = self.validate(match)
            if confidence is None:
                continue
            start, end = match.span(self.capture_group)
            entities.append(
                Entity(
                    type=self.entity_type,
                    start=start,
                    end=end,
                    text=text[start:end],
                    confidence=confidence,
                    detector=self.name,
                )
            )
        return entities

    def validate(self, match: re.Match[str]) -> float | None:
        return self.base_confidence


class CueAnchoredIdDetector(RegexDetector):
    """Cue phrase followed by an identifier; the span covers only the ID.

    Formatless identifiers (MRNs, account numbers, plan IDs, license
    numbers) have no universal shape, so v1 detects them by their labels.
    The captured ID must contain at least min_digits digits; cue-free
    detection is Stage 2's job.
    """

    capture_group = 1
    min_digits = 3

    def validate(self, match: re.Match[str]) -> float | None:
        digits = sum(ch.isdigit() for ch in match.group(self.capture_group))
        if digits < self.min_digits:
            return None
        return self.base_confidence
