"""Document-level classification: entities -> NONE/PII/PHI label + risk score."""

from __future__ import annotations

from collections import defaultdict

from .entities import TAXONOMY, DocLabel, EntityType
from .models import DocumentReport, Entity
from .pipeline import Pipeline

# v1 stand-in for real health-context modeling (replaced at Stage 2/3):
# a cheap lowercase substring scan over ~20 clinical terms.
MEDICAL_CONTEXT_TERMS: frozenset[str] = frozenset({
    "patient",
    "diagnosis",
    "physician",
    "prescription",
    "medical record",
    "hospital",
    "clinic",
    "icd-10",
    "treatment",
    "discharge",
    "lab results",
    "hipaa",
    "provider",
    "dosage",
    "symptoms",
    "admission",
    "medication",
    "health plan",
    "rx",
})


def has_medical_context(text: str) -> bool:
    lower = text.lower()
    return any(term in lower for term in MEDICAL_CONTEXT_TERMS)


class DocumentClassifier:
    """Maps detected entities to a document label and an explainable score.

    Label rules (each firing rule appends a line to report.reasons):
      1. no entities                                   -> NONE
      2. any PHI-specific entity (v1: MRN)             -> PHI
      3. any entity AND medical context in the text    -> PHI
      4. otherwise                                     -> PII

    Risk score:
        score = clamp( sum over types of
                           min(count_t, PER_TYPE_CAP) * weight_t * mean_conf_t
                       + PHI_BONUS if label is PHI,
                       0, 100 )
    The per-type cap keeps a CSV of 500 emails from outscoring one SSN;
    weights live in entities.TAXONOMY; all values are hand-set v1 heuristics —
    the durable part is the structure (additive, capped, attributable per type).
    """

    PER_TYPE_CAP = 3
    PHI_BONUS = 15.0

    def classify(self, text: str, entities: list[Entity]) -> DocumentReport:
        by_type: dict[EntityType, list[Entity]] = defaultdict(list)
        for entity in entities:
            by_type[entity.type].append(entity)

        reasons: list[str] = []
        if not entities:
            label = DocLabel.NONE
            reasons.append("no entities detected")
        else:
            phi_specific = sorted(
                {e.type for e in entities if TAXONOMY[e.type].phi_specific}
            )
            if phi_specific:
                label = DocLabel.PHI
                for t in phi_specific:
                    reasons.append(
                        f"{t.value} detected ({TAXONOMY[t].hipaa_category}) -> PHI"
                    )
            elif has_medical_context(text):
                label = DocLabel.PHI
                reasons.append(
                    "identifying entities present in a medical context -> PHI"
                )
            else:
                label = DocLabel.PII
                reasons.append("identifying entities present, no medical context -> PII")

        score = 0.0
        for entity_type, found in sorted(by_type.items()):
            mean_conf = sum(e.confidence for e in found) / len(found)
            contribution = (
                min(len(found), self.PER_TYPE_CAP)
                * TAXONOMY[entity_type].weight
                * mean_conf
            )
            score += contribution
            reasons.append(
                f"{entity_type.value} x{len(found)} contributes "
                f"{round(contribution, 1)} to risk"
            )
        if label is DocLabel.PHI:
            score += self.PHI_BONUS
        score = max(0.0, min(100.0, score))

        return DocumentReport(
            label=label,
            risk_score=score,
            entities=entities,
            counts={t.value: len(found) for t, found in sorted(by_type.items())},
            reasons=reasons,
        )


def scan_text(text: str) -> DocumentReport:
    """One-call public API: default pipeline + default classifier."""
    pipeline = Pipeline()
    return DocumentClassifier().classify(text, pipeline.run(text))
