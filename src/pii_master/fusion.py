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


def _overlaps(entity: Entity, others: list[Entity]) -> bool:
    return any(entity.overlaps(other) for other in others)


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
