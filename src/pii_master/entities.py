"""Entity taxonomy: what we detect and how it maps to PII/PHI.

Each :class:`EntityType` maps to one of the 18 HIPAA Safe Harbor identifiers
(45 CFR 164.514(b)(2)) via :data:`TAXONOMY`; see docs/DESIGN.md section 6 for
the full crosswalk.

Two tiers of type share this enum, and the split is the reason Stage 2 exists
rather than a taxonomy accident:

  * **Rule types** (EMAIL ... US_DRIVER_LICENSE) are found by Stage 1 regex +
    validators. They are format-anchored or cue-anchored, and they ship in the
    zero-dependency default install.
  * **Model types** (PERSON_NAME ... BIOMETRIC_ID) have no reliable regex --
    that is exactly why docs/DESIGN.md section 8 specifies a learned tagger.
    They are only ever populated by the Stage 2 detector, so a rules-only
    install simply never emits them. :data:`MODEL_ONLY_TYPES` names them so
    evaluation can report "not detectable in this configuration" rather than
    silently scoring recall 0.

Adding a model type is a deliberate act: it needs a HIPAA row here, a risk
weight, and a crosswalk entry in crosswalk.py. The four GDPR special-category
Nemotron labels (race_ethnicity, religious_belief, political_view, sexuality)
and the five credential labels (password, api_key, cvv, pin, http_cookie) are
deliberately NOT here -- they are not HIPAA identifiers, and they belong to the
M3 policy profiles described in docs/IMPROVEMENT_PLAN.md Track E.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum


class EntityType(str, Enum):
    """v1 entity types. str mixin makes JSON serialization free."""

    EMAIL = "EMAIL"
    PHONE_US = "PHONE_US"
    SSN = "SSN"
    CREDIT_CARD = "CREDIT_CARD"
    IP_ADDRESS = "IP_ADDRESS"
    DATE_DOB = "DATE_DOB"
    MRN = "MRN"
    URL = "URL"
    ACCOUNT_NUMBER = "ACCOUNT_NUMBER"
    HEALTH_PLAN_ID = "HEALTH_PLAN_ID"
    US_DRIVER_LICENSE = "US_DRIVER_LICENSE"

    # Stage 2 (model-only) types. No regex can find these with shippable
    # precision; see MODEL_ONLY_TYPES below and docs/DESIGN.md section 8.
    PERSON_NAME = "PERSON_NAME"
    ADDRESS = "ADDRESS"
    GEO_COORDINATE = "GEO_COORDINATE"
    DATE_TIME = "DATE_TIME"
    FAX_NUMBER = "FAX_NUMBER"
    BANK_ROUTING = "BANK_ROUTING"
    SWIFT_BIC = "SWIFT_BIC"
    VEHICLE_ID = "VEHICLE_ID"
    DEVICE_ID = "DEVICE_ID"
    MAC_ADDRESS = "MAC_ADDRESS"
    NATIONAL_ID = "NATIONAL_ID"
    TAX_ID = "TAX_ID"
    USER_ID = "USER_ID"
    BIOMETRIC_ID = "BIOMETRIC_ID"


class DocLabel(IntEnum):
    """Document-level label, ordered by sensitivity so max() picks the worst.

    PHI implies PII is present (PHI is a superset of PII once health context
    exists — see docs/DESIGN.md section 2).
    """

    NONE = 0
    PII = 1
    PHI = 2


@dataclass(frozen=True)
class EntityInfo:
    """Static classification metadata for one entity type.

    phi_specific: the entity alone establishes health-context linkage (an MRN
    has no non-medical reading), so its presence makes a document PHI outright.
    weight: risk-score contribution per occurrence (see classify.py).
    """

    is_pii: bool
    phi_specific: bool
    hipaa_category: str | None
    weight: float


TAXONOMY: dict[EntityType, EntityInfo] = {
    EntityType.EMAIL: EntityInfo(
        is_pii=True, phi_specific=False,
        hipaa_category="#6 Email addresses", weight=10.0,
    ),
    EntityType.PHONE_US: EntityInfo(
        is_pii=True, phi_specific=False,
        hipaa_category="#4 Telephone numbers", weight=10.0,
    ),
    EntityType.SSN: EntityInfo(
        is_pii=True, phi_specific=False,
        hipaa_category="#7 Social Security numbers", weight=30.0,
    ),
    EntityType.CREDIT_CARD: EntityInfo(
        is_pii=True, phi_specific=False,
        hipaa_category="#10 Account numbers", weight=30.0,
    ),
    EntityType.IP_ADDRESS: EntityInfo(
        is_pii=True, phi_specific=False,
        hipaa_category="#15 IP addresses", weight=5.0,
    ),
    EntityType.DATE_DOB: EntityInfo(
        is_pii=True, phi_specific=False,
        hipaa_category="#3 Dates related to an individual", weight=15.0,
    ),
    EntityType.MRN: EntityInfo(
        is_pii=True, phi_specific=True,
        hipaa_category="#8 Medical record numbers", weight=30.0,
    ),
    EntityType.URL: EntityInfo(
        is_pii=True, phi_specific=False,
        hipaa_category="#14 Web URLs", weight=5.0,
    ),
    EntityType.ACCOUNT_NUMBER: EntityInfo(
        is_pii=True, phi_specific=False,
        hipaa_category="#10 Account numbers", weight=20.0,
    ),
    # phi_specific holds because the detector only fires on unambiguously
    # health-flavored cues (health plan / beneficiary / subscriber), never
    # generic ones like "member id" or "policy number".
    EntityType.HEALTH_PLAN_ID: EntityInfo(
        is_pii=True, phi_specific=True,
        hipaa_category="#9 Health plan beneficiary numbers", weight=30.0,
    ),
    EntityType.US_DRIVER_LICENSE: EntityInfo(
        is_pii=True, phi_specific=False,
        hipaa_category="#11 Certificate/license numbers", weight=20.0,
    ),

    # --- Stage 2 model-only types (see MODEL_ONLY_TYPES) ---------------------
    # None of these is phi_specific. A name, an address or a fax number is an
    # identifier in any context; only an MRN or a health-plan id carries its
    # health linkage in the identifier itself. Adding a name to the taxonomy
    # therefore widens PII coverage without widening what can be called PHI --
    # a document still needs medical context (classify.py rule 3) to escalate.
    EntityType.PERSON_NAME: EntityInfo(
        is_pii=True, phi_specific=False,
        hipaa_category="#1 Names", weight=15.0,
    ),
    # street_address / city / county / postcode all fold in here; adjacent
    # spans are merged by the detector so "44 Elm Street, Springfield" is one
    # entity. HIPAA #2 is subdivisions SMALLER than a state, so Nemotron's
    # `state` label is deliberately left unmodelled in crosswalk.py.
    EntityType.ADDRESS: EntityInfo(
        is_pii=True, phi_specific=False,
        hipaa_category="#2 Geographic subdivisions smaller than a state",
        weight=20.0,
    ),
    EntityType.GEO_COORDINATE: EntityInfo(
        is_pii=True, phi_specific=False,
        hipaa_category="#2 Geographic subdivisions smaller than a state",
        weight=20.0,
    ),
    # A timestamp is only a HIPAA #3 identifier when it is tied to an
    # individual (admission, discharge, death). Weighted low precisely because
    # the detector cannot yet tell those from a log line's clock.
    EntityType.DATE_TIME: EntityInfo(
        is_pii=True, phi_specific=False,
        hipaa_category="#3 Dates related to an individual", weight=5.0,
    ),
    EntityType.FAX_NUMBER: EntityInfo(
        is_pii=True, phi_specific=False,
        hipaa_category="#5 Fax numbers", weight=10.0,
    ),
    # Routing numbers and BICs identify an institution, not a person -- they
    # are HIPAA #10 but weak on their own, hence the low weights.
    EntityType.BANK_ROUTING: EntityInfo(
        is_pii=True, phi_specific=False,
        hipaa_category="#10 Account numbers", weight=10.0,
    ),
    EntityType.SWIFT_BIC: EntityInfo(
        is_pii=True, phi_specific=False,
        hipaa_category="#10 Account numbers", weight=5.0,
    ),
    EntityType.VEHICLE_ID: EntityInfo(
        is_pii=True, phi_specific=False,
        hipaa_category="#12 Vehicle identifiers and license plates", weight=15.0,
    ),
    EntityType.DEVICE_ID: EntityInfo(
        is_pii=True, phi_specific=False,
        hipaa_category="#13 Device identifiers and serial numbers", weight=10.0,
    ),
    EntityType.MAC_ADDRESS: EntityInfo(
        is_pii=True, phi_specific=False,
        hipaa_category="#18 Other unique identifying number or code", weight=10.0,
    ),
    # A national id is another country's SSN, so it carries SSN's weight.
    EntityType.NATIONAL_ID: EntityInfo(
        is_pii=True, phi_specific=False,
        hipaa_category="#18 Other unique identifying number or code", weight=30.0,
    ),
    EntityType.TAX_ID: EntityInfo(
        is_pii=True, phi_specific=False,
        hipaa_category="#18 Other unique identifying number or code", weight=20.0,
    ),
    EntityType.USER_ID: EntityInfo(
        is_pii=True, phi_specific=False,
        hipaa_category="#18 Other unique identifying number or code", weight=10.0,
    ),
    EntityType.BIOMETRIC_ID: EntityInfo(
        is_pii=True, phi_specific=False,
        hipaa_category="#16 Biometric identifiers", weight=25.0,
    ),
}

# Types no rule can produce. A rules-only install never emits them, so
# evaluation reports them as "undetectable in this configuration" instead of
# scoring them as recall 0 (see evaluation.FUTURE_TYPES, which reads this).
MODEL_ONLY_TYPES: frozenset[EntityType] = frozenset({
    EntityType.PERSON_NAME,
    EntityType.ADDRESS,
    EntityType.GEO_COORDINATE,
    EntityType.DATE_TIME,
    EntityType.FAX_NUMBER,
    EntityType.BANK_ROUTING,
    EntityType.SWIFT_BIC,
    EntityType.VEHICLE_ID,
    EntityType.DEVICE_ID,
    EntityType.MAC_ADDRESS,
    EntityType.NATIONAL_ID,
    EntityType.TAX_ID,
    EntityType.USER_ID,
    EntityType.BIOMETRIC_ID,
})

assert MODEL_ONLY_TYPES <= set(TAXONOMY), "every model type needs a HIPAA row"

#: Types whose Stage 1 validator is a checksum or a hard format parse: the Luhn
#: mod-10 check, the never-issued SSN ranges, stdlib ``ipaddress``, and the
#: structural email/URL tests. A span of one of these that a rule accepted is a
#: verified *fact*; a cue-anchored MRN or account number is a *guess* that
#: happened to sit next to the right word.
#:
#: That distinction is load-bearing in two places, which is why it lives here
#: rather than in either of them:
#:   * ``pipeline.py`` ranks these above the Stage 2 model on overlap, and ranks
#:     the model above cue-anchored rule spans. Reading the plan's fusion clause
#:     the narrow way (only *checksummed* rules are authoritative) rather than
#:     the broad way (all rules are) is worth +0.028 F1 -- measured, per type,
#:     in docs/DISTILLATION_RESULTS.md section 5.
#:   * ``ner.py`` re-runs these validators on model-proposed spans of the same
#:     types before they may be emitted at all (gate 4).
CHECKSUMMED_TYPES: frozenset[EntityType] = frozenset({
    EntityType.SSN,
    EntityType.CREDIT_CARD,
    EntityType.EMAIL,
    EntityType.IP_ADDRESS,
    EntityType.URL,
})
