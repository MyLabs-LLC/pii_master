"""Document-level classification: entities -> NONE/PII/PHI label + risk score."""

from __future__ import annotations

import re
from collections import defaultdict

from .entities import TAXONOMY, DocLabel, EntityType
from .models import DocumentReport, Entity
from .pipeline import Pipeline

# v1 stand-in for real health-context modeling (replaced at Stage 2/3).
#
# Two rules govern what may appear here, because any term firing on a
# document that already contains an identifier escalates it to PHI:
#   1. Terms are matched on word boundaries, never as substrings. A raw
#      substring scan made "impatient" match "patient".
#   2. A term must be unambiguously health-flavored on its own. Bare
#      "provider" is not: "cloud provider" and "service provider" are the
#      common readings, so only the qualified forms are listed.
MEDICAL_CONTEXT_TERMS: frozenset[str] = frozenset({
    "patient",
    "inpatient",
    "outpatient",
    "diagnosis",
    "diagnosed",
    "physician",
    "prescription",
    "medical record",
    "medical history",
    "hospital",
    "clinic",
    "icd-10",
    "icd-9",
    "treatment",
    "discharge",
    "lab results",
    "hipaa",
    "healthcare provider",
    "health care provider",
    "medical provider",
    "care provider",
    "dosage",
    "symptoms",
    "admission",
    "medication",
    "health plan",
    "rx",
})

# (?<!\w) / (?!\w) rather than \b so terms ending in punctuation ("icd-10")
# behave, and so "rx" cannot match inside a longer word. Longest-first so
# "medical record" wins over a shorter prefix term.
_MEDICAL_CONTEXT_RX = re.compile(
    r"(?<!\w)(?:"
    + "|".join(re.escape(t) for t in sorted(MEDICAL_CONTEXT_TERMS, key=len, reverse=True))
    + r")(?!\w)",
    re.IGNORECASE,
)


def has_medical_context(text: str) -> bool:
    """True if any medical-context term appears as a whole word.

    Known residual weakness: "patient" also occurs as an adjective
    ("please be patient"), and "treatment"/"discharge"/"admission" have
    non-clinical readings. These are accepted for recall; real context
    modeling is a Stage 2/3 job, not a longer deny-list.
    """
    return _MEDICAL_CONTEXT_RX.search(text) is not None


class DocumentClassifier:
    """Maps detected entities to a document label and an explainable score.

    Label rules (each firing rule appends a line to report.reasons):
      1. no entities                                       -> NONE
      2. any PHI-specific entity (MRN, HEALTH_PLAN_ID)     -> PHI
      3. any entity AND medical context in the text        -> PHI
      4. otherwise                                         -> PII

    Rule 2 reads `phi_specific` from entities.TAXONOMY rather than naming
    types here, so adding a PHI-specific type needs no change in this file.

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


_FAST_PIPELINE: Pipeline | None = None
_DEEP_PIPELINE: Pipeline | None = None
_CLASSIFIER: DocumentClassifier | None = None


def _fast_pipeline() -> Pipeline:
    global _FAST_PIPELINE
    if _FAST_PIPELINE is None:
        _FAST_PIPELINE = Pipeline()
    return _FAST_PIPELINE


def _deep_pipeline() -> Pipeline:
    global _DEEP_PIPELINE
    if _DEEP_PIPELINE is None:
        from .onnx_ner import OnnxNerDetector

        _DEEP_PIPELINE = Pipeline(ner=OnnxNerDetector())
    return _DEEP_PIPELINE


def _classifier() -> DocumentClassifier:
    global _CLASSIFIER
    if _CLASSIFIER is None:
        _CLASSIFIER = DocumentClassifier()
    return _CLASSIFIER


def scan_text(text: str, *, deep: bool = False, pipeline: Pipeline | None = None) -> DocumentReport:
    """One-call public API. ``deep=True`` adds the Stage 2 ONNX student.

    Fast mode (default) stays stdlib-only. Deep mode needs
    ``pip install 'pii-master[ml]'`` and the exported student artifact.
    """
    if pipeline is None:
        pipeline = _deep_pipeline() if deep else _fast_pipeline()
    return _classifier().classify(text, pipeline.run(text))
