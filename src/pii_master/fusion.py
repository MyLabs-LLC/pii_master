"""Named fusion policies for mixing rule spans with model spans.

Checksum-validated rule spans outrank the model; cue-anchored rule spans
do not. Measured in docs/DISTILLATION_RESULTS.md section 5: blanket
rules-first costs 0.028 F1 against this policy, mostly on PHONE / ACCOUNT /
MRN / PLAN / LICENSE.
"""

from __future__ import annotations

from .entities import EntityType
from .models import Entity

# Types whose validator is a checksum or a hard format parse. Gate 4 of
# docs/DISTILLATION_PLAN.md protects these from being displaced by the model.
CHECKSUMMED: frozenset[EntityType] = frozenset({
    EntityType.SSN,
    EntityType.CREDIT_CARD,
    EntityType.EMAIL,
    EntityType.IP_ADDRESS,
    EntityType.URL,
})


_GAP_CHARS = frozenset(" \t\n\r,;.")
_SNAP_TYPES = frozenset({
    EntityType.PERSON_NAME,
    EntityType.ADDRESS,
    EntityType.USERNAME,
})


def snap_to_word_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """If a span starts or ends inside a neighbouring word, drop the fragment.

    The student sometimes tags the last subword of the previous word
    (``Applicant Jane Doe`` -> ``ant Jane Doe``). Exact-match gold and
    Kaggle token F5 both treat that as a miss plus a false positive.
    """
    if start >= end:
        return start, end
    if start > 0 and text[start - 1].isalnum() and text[start].isalnum():
        while start < end and not text[start].isspace():
            start += 1
        while start < end and text[start].isspace():
            start += 1
    if end < len(text) and end > 0 and text[end - 1].isalnum() and text[end].isalnum():
        while end > start and not text[end - 1].isspace():
            end -= 1
        while end > start and text[end - 1].isspace():
            end -= 1
    return start, end


def _overlaps(entity: Entity, others: list[Entity]) -> bool:
    return any(entity.overlaps(other) for other in others)


def merge_adjacent_same_type(
    entities: list[Entity],
    text: str,
    max_gap: int = 3,
) -> list[Entity]:
    """Join same-type spans separated only by a short punctuation/space gap.

    The student emits ``first_name`` and ``last_name`` (and street/city)
    as separate native labels that collapse to one HIPAA type. Frozen gold
    and the Kaggle NAME_STUDENT / STREET_ADDRESS labels are full phrases
    (``Jane Doe``, ``44 Elm Street, Springfield``), so serving has to
    reconstruct the phrase or every exact-match metric treats a correct
    name as two false positives plus a miss.
    """
    if not entities:
        return []
    ordered = sorted(entities, key=lambda entity: (entity.start, entity.end, entity.detector))
    merged = [ordered[0]]
    for span in ordered[1:]:
        prev = merged[-1]
        gap = span.start - prev.end
        if (
            span.type == prev.type
            and 0 <= gap <= max_gap
            and (gap == 0 or all(ch in _GAP_CHARS for ch in text[prev.end:span.start]))
        ):
            end = max(prev.end, span.end)
            merged[-1] = Entity(
                type=prev.type,
                start=prev.start,
                end=end,
                text=text[prev.start:end],
                confidence=max(prev.confidence, span.confidence),
                detector=prev.detector,
            )
        else:
            merged.append(span)
    snapped: list[Entity] = []
    for span in merged:
        if span.type not in _SNAP_TYPES:
            snapped.append(span)
            continue
        start, end = snap_to_word_bounds(text, span.start, span.end)
        if start >= end:
            continue
        snapped.append(
            Entity(
                type=span.type,
                start=start,
                end=end,
                text=text[start:end],
                confidence=span.confidence,
                detector=span.detector,
            )
        )
    return snapped


def fuse_checksum_first(rule_spans: list[Entity], model_spans: list[Entity]) -> list[Entity]:
    """Checksum rules > model > cue-anchored rules, with one serving-path refinement.

    When a model span overlaps a *same-type* cue-anchored rule span and the
    rule is longer, keep the rule. The student's measured failure on the
    frozen corpus is truncation (``84-J99-1220`` -> ``84-J99-12``); Nemotron
    documents are long enough that this almost never fires, while the
    adversarial one-liners are made of it.
    """
    authoritative = [span for span in rule_spans if span.type in CHECKSUMMED]
    deferred = [span for span in rule_spans if span.type not in CHECKSUMMED]
    kept = list(authoritative)
    for span in model_spans:
        if _overlaps(span, kept):
            continue
        rivals = [rule for rule in deferred if rule.type == span.type and span.overlaps(rule)]
        if rivals and max(rule.end - rule.start for rule in rivals) > (span.end - span.start):
            continue
        kept.append(span)
    kept += [span for span in deferred if not _overlaps(span, kept)]
    kept.sort(key=lambda entity: (entity.start, entity.end, entity.detector))
    return kept


def resolve_greedy(candidates: list[Entity]) -> list[Entity]:
    """v1 overlap resolution: sort by (-conf, -length, start, detector)."""
    ordered = sorted(
        candidates,
        key=lambda entity: (
            -entity.confidence,
            -(entity.end - entity.start),
            entity.start,
            entity.detector,
        ),
    )
    accepted: list[Entity] = []
    for candidate in ordered:
        if _overlaps(candidate, accepted):
            continue
        accepted.append(candidate)
    accepted.sort(key=lambda entity: (entity.start, entity.end))
    return accepted
