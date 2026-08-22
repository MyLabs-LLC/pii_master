"""Detector contract, the regex base class, and its pre-scan machinery.

Detector is a structural Protocol so any detection strategy — the Stage 2
ONNX NER detector included — plugs into the pipeline without inheriting
anything from this package.

RegexDetector supports three *pre-scan* modes that narrow where the full
pattern runs, because scanning every position of every document with a
dozen back-tracking patterns blows the 5 ms/doc budget (docs/DESIGN.md
section 5; measured in docs/BASELINE_M1.md):

- ``hints``: lowercase literal substrings located with C-speed ``str.find``
  (a cue word, "@", "http"). The pattern runs only in a window around each
  hit. Every possible match must contain a hint, and ``hint_lead`` /
  ``hint_window`` must cover the farthest a match can start before/after
  its hit.
- ``use_digit_runs``: for numeric types (phone, SSN, card, IPv4). One
  shared regex finds runs of digits-and-separators (cached per document
  text), and the pattern runs only inside those runs. Every candidate these
  patterns can match is made of run characters, so runs cover all of them.
- ``window_pattern``: a cheap custom regex whose matches seed windows
  (IPv6 uses a colon-pair finder).

With no mode set, the pattern scans the full text (the correct default for
detectors without a reliable anchor). Windows are scanned with
``pattern.finditer(text, start, end)``, so lookbehinds still see the real
preceding text; ``overshoot`` extends the scan region past the nominal
window so a match *starting* inside the window is never truncated, and
matches starting past the nominal end are skipped (a later window owns
them). Equivalence with full scanning is asserted in
tests/test_prescan_equivalence.py.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Protocol, runtime_checkable

from ..entities import EntityType
from ..models import Entity


@runtime_checkable
class Detector(Protocol):
    name: str

    def detect(self, text: str) -> list[Entity]: ...


_DIGIT_RUN = re.compile(r"[\d(+][\d(). +-]{5,}\d")


@lru_cache(maxsize=8)
def _digit_run_windows(text: str) -> tuple[tuple[int, int], ...]:
    return tuple(m.span() for m in _DIGIT_RUN.finditer(text))


def _merge(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Coalesce ordered, possibly-overlapping windows."""
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


class RegexDetector:
    """Compiled pattern -> pre-scan windows -> candidates -> validate hook.

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

    # Pre-scan configuration — see the module docstring.
    hints: tuple[str, ...] = ()
    hint_lead: int = 0
    hint_window: int = 64
    use_digit_runs: bool = False
    window_pattern: re.Pattern[str] | None = None
    overshoot: int = 32

    def detect(self, text: str) -> list[Entity]:
        windows = self._candidate_windows(text)
        if windows is None:
            windows = [(0, len(text))]
        entities: list[Entity] = []
        seen: set[tuple[int, int]] = set()
        n = len(text)
        for w_start, w_end in windows:
            scan_end = min(n, w_end + self.overshoot)
            for match in self.pattern.finditer(text, w_start, scan_end):
                if match.start() >= w_end:
                    continue
                span = match.span(self.capture_group)
                if span in seen:
                    continue
                seen.add(span)
                confidence = self.validate(match)
                if confidence is None:
                    continue
                start, end = span
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

    def _candidate_windows(self, text: str) -> list[tuple[int, int]] | None:
        """Regions worth scanning, or None for a full scan."""
        if self.use_digit_runs:
            return _merge(list(_digit_run_windows(text)))
        if self.hints:
            lower = text.lower()
            hits: list[int] = []
            for hint in self.hints:
                i = lower.find(hint)
                while i != -1:
                    hits.append(i)
                    i = lower.find(hint, i + 1)
            if not hits:
                return []
            hits.sort()
            return _merge(
                [(max(0, h - self.hint_lead), h + self.hint_window) for h in hits]
            )
        if self.window_pattern is not None:
            return _merge([m.span() for m in self.window_pattern.finditer(text)])
        return None

    def validate(self, match: re.Match[str]) -> float | None:
        return self.base_confidence


class CueAnchoredIdDetector(RegexDetector):
    """Cue phrase followed by an identifier; the span covers only the ID.

    Formatless identifiers (MRNs, account numbers, plan IDs, license
    numbers) have no universal shape, so v1 detects them by their labels.
    The captured ID must contain at least min_digits digits; cue-free
    detection is Stage 2's job. Subclasses set ``hints`` to their cue
    words so documents without the cue never pay for the regex.
    """

    capture_group = 1
    min_digits = 3
    overshoot = 24

    def validate(self, match: re.Match[str]) -> float | None:
        digits = sum(ch.isdigit() for ch in match.group(self.capture_group))
        if digits < self.min_digits:
            return None
        return self.base_confidence
