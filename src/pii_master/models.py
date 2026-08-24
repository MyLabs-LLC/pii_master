"""Core data records exchanged between pipeline stages.

Entity is the contract every detection strategy (regex today, ONNX NER at M2)
must emit; DocumentReport is what the classifier and CLI hand to callers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .entities import DocLabel, EntityType


@dataclass(frozen=True, slots=True)
class Entity:
    """One detected span.

    confidence is ordinal detector certainty, not a calibrated probability
    (see docs/DESIGN.md section 7). detector records provenance, e.g.
    "regex/ssn" now, "onnx/ner-v1" at Stage 2.
    """

    type: EntityType
    start: int  # char offset, inclusive
    end: int  # char offset, exclusive
    text: str
    confidence: float
    detector: str

    def overlaps(self, other: "Entity") -> bool:
        return self.start < other.end and other.start < self.end

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "detector": self.detector,
        }


@dataclass
class DocumentReport:
    """Document-level classification with span evidence and explanation."""

    label: DocLabel
    risk_score: float
    entities: list[Entity] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    # Papadopoulou et al. 2023 privacy-risk indicators. Combinations of
    # identifiers, not just the sum of their weights — Golle (2006) /
    # HIPAA Safe Harbor's reason for treating {DOB, geo, name} as jointly
    # identifying even when no single field is.
    direct_count: int = 0
    quasi_count: int = 0
    reidentification_combos: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "label": self.label.name,
            "risk_score": round(self.risk_score, 2),
            "counts": self.counts,
            "direct_count": self.direct_count,
            "quasi_count": self.quasi_count,
            "reidentification_combos": self.reidentification_combos,
            "reasons": self.reasons,
            "entities": [e.to_dict() for e in self.entities],
        }
