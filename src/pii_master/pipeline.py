"""Run detectors and resolve overlapping spans into a flat entity list."""

from __future__ import annotations

from typing import Sequence

from .detectors import Detector, default_detectors
from .models import Entity


class Pipeline:
    """Collects candidates from every detector and resolves overlaps.

    Resolution is greedy weighted-interval selection: candidates sorted by
    (-confidence, -length, start, detector name), each accepted iff it
    overlaps no already-accepted span. Deterministic and explainable; the
    sort dominates at O(n log n) for typical span counts.

    v2 replacement point: allowing nested spans of *different* types (e.g.
    a phone number inside an email's display name) means swapping only this
    resolution step — detectors and the Entity contract are unaffected.
    """

    def __init__(self, detectors: Sequence[Detector] | None = None):
        self.detectors: list[Detector] = (
            list(detectors) if detectors is not None else default_detectors()
        )

    def run(self, text: str) -> list[Entity]:
        candidates: list[Entity] = []
        for detector in self.detectors:
            candidates.extend(detector.detect(text))
        candidates.sort(
            key=lambda e: (-e.confidence, -(e.end - e.start), e.start, e.detector)
        )
        accepted: list[Entity] = []
        for candidate in candidates:
            if any(candidate.overlaps(kept) for kept in accepted):
                continue
            accepted.append(candidate)
        accepted.sort(key=lambda e: (e.start, e.end))
        return accepted
