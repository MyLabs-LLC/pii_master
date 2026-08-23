"""Run detectors and resolve overlapping spans into a flat entity list."""

from __future__ import annotations

from typing import Sequence

from .detectors import Detector, default_detectors
from .fusion import fuse_checksum_first, resolve_greedy
from .models import Entity


class Pipeline:
    """Collects candidates from every detector and resolves overlaps.

    Fast path (``ner is None``): greedy weighted-interval selection — candidates
    sorted by (-confidence, -length, start, detector name), each accepted iff
    it overlaps no already-accepted span. Deep path: checksum-first fusion
    (docs/DISTILLATION_RESULTS.md §5) between rule detectors and an optional
    Stage 2 NER detector.
    """

    def __init__(
        self,
        detectors: Sequence[Detector] | None = None,
        ner: Detector | None = None,
    ):
        self.detectors: list[Detector] = (
            list(detectors) if detectors is not None else default_detectors()
        )
        self.ner = ner

    def run(self, text: str) -> list[Entity]:
        candidates: list[Entity] = []
        for detector in self.detectors:
            candidates.extend(detector.detect(text))
        if self.ner is None:
            return resolve_greedy(candidates)
        model = self.ner.detect(text)
        return fuse_checksum_first(candidates, model)
