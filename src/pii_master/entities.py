"""Entity taxonomy: what we detect and how it maps to PII/PHI.

Each :class:`EntityType` maps to one of the 18 HIPAA Safe Harbor identifiers
(45 CFR 164.514(b)(2)) via :data:`TAXONOMY`; see docs/DESIGN.md section 6 for
the full crosswalk and the list of deferred types (PERSON_NAME, ADDRESS,
US_DRIVER_LICENSE, PASSPORT, HEALTH_PLAN_ID, IPv6, URL, ...) with the reasons
each is deferred.
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
}
