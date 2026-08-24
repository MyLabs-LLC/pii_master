"""Document-level classification: entities -> NONE/PII/PHI label + risk score."""

from __future__ import annotations

import re
from collections import defaultdict

from .entities import TAXONOMY, DocLabel, EntityType, IdentifierKind
from .models import DocumentReport, Entity
from .pipeline import Pipeline, deep_pipeline

# Combinations the sanitization literature treats as jointly identifying
# even when no single field is. Papadopoulou et al. 2023 call this the
# "span classification / combination" privacy-risk indicator; the classic
# measurement is Golle (2006): gender + DOB + ZIP IDs 63–78% of the US.
# We do not have gender or ZIP as first-class types, so the nearest
# shippable triples use PERSON_NAME / DATE_DOB / ADDRESS / PHONE_US.
_REID_COMBOS: tuple[tuple[str, frozenset[EntityType], float], ...] = (
    ("name+dob+address", frozenset({
        EntityType.PERSON_NAME, EntityType.DATE_DOB, EntityType.ADDRESS,
    }), 20.0),
    ("name+dob", frozenset({EntityType.PERSON_NAME, EntityType.DATE_DOB}), 12.0),
    ("name+address", frozenset({EntityType.PERSON_NAME, EntityType.ADDRESS}), 12.0),
    ("dob+address", frozenset({EntityType.DATE_DOB, EntityType.ADDRESS}), 12.0),
    ("dob+phone", frozenset({EntityType.DATE_DOB, EntityType.PHONE_US}), 8.0),
)

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

        present = set(by_type)
        combos: list[str] = []
        # Longest (highest-bonus) combo first; skip a pair that is already
        # covered by a triple so the report does not double-count Golle.
        covered: set[frozenset[EntityType]] = set()
        for name, needed, bonus in _REID_COMBOS:
            if not needed <= present:
                continue
            if any(needed <= prev for prev in covered):
                continue
            combos.append(name)
            covered.add(needed)
            score += bonus
            reasons.append(
                f"re-identification combination {name} contributes "
                f"{bonus:.0f} to risk (Papadopoulou et al. 2023 / Golle 2006)"
            )

        if label is DocLabel.PHI:
            score += self.PHI_BONUS
        score = max(0.0, min(100.0, score))

        direct = sum(
            1 for e in entities
            if TAXONOMY[e.type].identifier_kind is IdentifierKind.DIRECT
        )
        quasi = sum(
            1 for e in entities
            if TAXONOMY[e.type].identifier_kind is IdentifierKind.QUASI
        )

        return DocumentReport(
            label=label,
            risk_score=score,
            entities=entities,
            counts={t.value: len(found) for t, found in sorted(by_type.items())},
            reasons=reasons,
            direct_count=direct,
            quasi_count=quasi,
            reidentification_combos=combos,
        )


# Rules-only, built once. Detectors are stateless (compiled patterns and pure
# validators), so one instance serves every call and every thread. v0.2 built a
# fresh Pipeline -- twelve detectors, twelve compiled patterns -- on every
# scan_text call, which eval, bench and the CLI all go through
# (docs/IMPROVEMENT_PLAN.md section 3.5). Deep mode makes that setup cost
# matter: an ONNX session takes tens of milliseconds to create, thousands of
# documents' worth of budget.
_DEFAULT_PIPELINE = Pipeline()
_DEEP_PIPELINE: Pipeline | None = None


def default_pipeline(deep: bool = False) -> Pipeline:
    """The shared pipeline for a serving mode.

    ``fast`` (default) is rules only and stdlib only. ``deep`` adds the Stage 2
    student and needs the optional ML extra; it is built lazily and cached, and
    raises :class:`pii_master.ner.ModelUnavailable` rather than degrading to
    rules if the model is missing.
    """
    global _DEEP_PIPELINE
    if not deep:
        return _DEFAULT_PIPELINE
    if _DEEP_PIPELINE is None:
        _DEEP_PIPELINE = deep_pipeline()
    return _DEEP_PIPELINE


def scan_text(
    text: str,
    pipeline: Pipeline | None = None,
    *,
    deep: bool = False,
) -> DocumentReport:
    """One-call public API: a pipeline + the default classifier.

    Args:
        pipeline: an explicit pipeline; overrides ``deep``. Pass one to use a
            custom detector set or non-default Stage 2 thresholds.
        deep: run the Stage 2 student alongside the rules (docs/DESIGN.md
            section 8). Costs the ML extra and more latency; finds names,
            addresses and cue-free identifiers that no rule can.
    """
    chosen = pipeline if pipeline is not None else default_pipeline(deep)
    return DocumentClassifier().classify(text, chosen.run(text))
