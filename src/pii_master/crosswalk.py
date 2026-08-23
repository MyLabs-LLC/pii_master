"""Nemotron-PII label space -> our EntityType.

The authoritative crosswalk, as code rather than a markdown table, so the
external evaluation (eval/scripts/nemotron_eval.py) and the Stage 2 data
prep cannot drift apart. The narrative version with span counts lives in
docs/NEMOTRON_PII_TAGS.md.

`None` means "we do not model this label": collapse it to `O` when building
training data. A label absent from NEMOTRON_TO_ENTITY entirely is an error
(a new dataset revision), not an implicit `None` -- UNMODELLED lists every
one explicitly so the two sets always cover the dataset.
"""

from __future__ import annotations

from .entities import EntityType

# Labels we detect today. Caveats, all deliberate and measured:
#   - ipv4 and ipv6 both fold into IP_ADDRESS (we do not split by version).
#   - certificate_license_number -> US_DRIVER_LICENSE is a NARROWING: the
#     Nemotron label covers any HIPAA #11 certificate or licence, so recall
#     against it understates nothing but precision may look better than the
#     category deserves. A LICENSE_NUMBER umbrella type is the fix (Track C).
#   - first_name + last_name collapse to PERSON_NAME; street/city/county/
#     postcode collapse to ADDRESS. Serving merges adjacent same-type spans
#     so a full name or street+city phrase is one entity.
NEMOTRON_TO_ENTITY: dict[str, EntityType | None] = {
    "email": EntityType.EMAIL,
    "phone_number": EntityType.PHONE_US,
    "ssn": EntityType.SSN,
    "credit_debit_card": EntityType.CREDIT_CARD,
    "ipv4": EntityType.IP_ADDRESS,
    "ipv6": EntityType.IP_ADDRESS,
    "date_of_birth": EntityType.DATE_DOB,
    "medical_record_number": EntityType.MRN,
    "url": EntityType.URL,
    "account_number": EntityType.ACCOUNT_NUMBER,
    "health_plan_beneficiary_number": EntityType.HEALTH_PLAN_ID,
    "certificate_license_number": EntityType.US_DRIVER_LICENSE,
    "first_name": EntityType.PERSON_NAME,
    "last_name": EntityType.PERSON_NAME,
    "street_address": EntityType.ADDRESS,
    "city": EntityType.ADDRESS,
    "county": EntityType.ADDRESS,
    "postcode": EntityType.ADDRESS,
    "user_name": EntityType.USERNAME,
}

# Everything else in the dataset, grouped by why we do not model it.
# Promoting a label out of here is a deliberate act with a detector and a
# taxonomy row attached.
UNMODELLED: dict[str, tuple[str, ...]] = {
    # Rules cannot do these; Stage 2's reason to exist.
    "stage2": ("state", "coordinate", "customer_id", "employee_id",
               "unique_id", "biometric_identifier"),
    # Format-anchored: candidates for Stage 1 (Track C), not the model.
    "track_c": ("fax_number", "bank_routing_number", "mac_address", "swift_bic",
                "vehicle_identifier", "license_plate", "date_time",
                "device_identifier", "national_id", "tax_id"),
    # Credentials: real, valuable, but outside the PII/PHI framing. A
    # "secrets" policy profile (M3) is where these belong.
    "secrets": ("password", "api_key", "cvv", "pin", "http_cookie"),
    # GDPR special-category attributes: not HIPAA identifiers. They belong to
    # a gdpr_special profile (M3), never to the HIPAA one.
    "gdpr_special": ("race_ethnicity", "religious_belief", "political_view",
                     "sexuality"),
    # Quasi-identifiers and non-identifying context.
    "attributes": ("date", "time", "age", "gender", "language", "occupation",
                   "education_level", "employment_status", "blood_type",
                   "company_name", "country"),
}

ALL_UNMODELLED: frozenset[str] = frozenset(
    label for group in UNMODELLED.values() for label in group
)


def to_entity_type(label: str) -> EntityType | None:
    """Map one Nemotron label, or None if we deliberately do not model it.

    Raises KeyError for an unknown label so a dataset revision that adds a
    category fails loudly instead of silently becoming background.
    """
    if label in NEMOTRON_TO_ENTITY:
        return NEMOTRON_TO_ENTITY[label]
    if label in ALL_UNMODELLED:
        return None
    raise KeyError(f"unknown Nemotron label {label!r}: update crosswalk.py")
