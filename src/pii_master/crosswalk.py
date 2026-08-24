"""Nemotron-PII label space -> our EntityType.

The authoritative crosswalk, as code rather than a markdown table, so the
external evaluation (eval/scripts/nemotron_eval.py), the Stage 2 data prep and
the Stage 2 serving detector (ner.py) cannot drift apart. The narrative version
with span counts lives in docs/NEMOTRON_PII_TAGS.md.

`None` means "we do not model this label": collapse it to `O` when building
training data, and drop it when decoding model output. A label absent from
NEMOTRON_TO_ENTITY entirely is an error (a new dataset revision), not an
implicit `None` -- UNMODELLED lists every one explicitly so the two sets always
cover the dataset.

**Two generations of mapping live here.** RULE_MAPPED_LABELS is the original
12-label set, the one the rules tier can actually detect and the one
docs/BASELINE_NEMOTRON.md scored. It is kept as a named constant, not as
history: gate 2 of docs/DISTILLATION_PLAN.md compares the student against the
committed rules baseline *on those same 12 types*, and that comparison would
quietly become meaningless if adopting PERSON_NAME silently widened the
denominator. Everything beyond those 12 is Stage 2 upside and is reported
separately.
"""

from __future__ import annotations

from .entities import STAGE2_TYPES, EntityType

# The 12 labels the Stage 1 rules tier targets. Two caveats, both deliberate
# and measured:
#   - ipv4 and ipv6 both fold into IP_ADDRESS (we do not split by version).
#   - certificate_license_number -> US_DRIVER_LICENSE is a NARROWING: the
#     Nemotron label covers any HIPAA #11 certificate or licence, so recall
#     against it understates nothing but precision may look better than the
#     category deserves. Measured cost: the rules score F1 0.001 on this type
#     (docs/DISTILLATION_RESULTS.md section 5) because Nemotron's cue is
#     "certificate license number" and the rule only accepts driver's-licence
#     wording. The student scores 0.829 on it, which is the single largest
#     win in the whole Stage 2 change.
RULE_MAPPED: dict[str, EntityType] = {
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
}

RULE_MAPPED_LABELS: frozenset[str] = frozenset(RULE_MAPPED)

# Labels adopted with the Stage 2 detector. Each one has a HIPAA row and a
# risk weight in entities.TAXONOMY; none is phi_specific.
#
# Several Nemotron labels collapse onto one of ours on purpose:
#   - first_name + last_name -> PERSON_NAME. Nemotron tags "Jane Doe" as two
#     adjacent spans; ner.py merges adjacent same-type spans, so we emit one
#     entity and match the frozen corpus's [[PERSON_NAME:Jane Doe]] gold.
#   - street_address / city / county / postcode -> ADDRESS, merged the same
#     way, which is how "44 Elm Street, Springfield" becomes one entity.
#   - vehicle_identifier + license_plate -> VEHICLE_ID (HIPAA #12 lists them
#     in one row).
#   - user_name / customer_id / employee_id / unique_id -> USER_ID, HIPAA #18.
MODEL_MAPPED: dict[str, EntityType] = {
    "first_name": EntityType.PERSON_NAME,
    "last_name": EntityType.PERSON_NAME,
    "street_address": EntityType.ADDRESS,
    "city": EntityType.ADDRESS,
    "county": EntityType.ADDRESS,
    "postcode": EntityType.ADDRESS,
    "coordinate": EntityType.GEO_COORDINATE,
    "date_time": EntityType.DATE_TIME,
    "fax_number": EntityType.FAX_NUMBER,
    "bank_routing_number": EntityType.BANK_ROUTING,
    "swift_bic": EntityType.SWIFT_BIC,
    "vehicle_identifier": EntityType.VEHICLE_ID,
    "license_plate": EntityType.VEHICLE_ID,
    "device_identifier": EntityType.DEVICE_ID,
    "mac_address": EntityType.MAC_ADDRESS,
    "national_id": EntityType.NATIONAL_ID,
    "tax_id": EntityType.TAX_ID,
    "user_name": EntityType.USER_ID,
    "customer_id": EntityType.USER_ID,
    "employee_id": EntityType.USER_ID,
    "unique_id": EntityType.USER_ID,
    "biometric_identifier": EntityType.BIOMETRIC_ID,
}

NEMOTRON_TO_ENTITY: dict[str, EntityType] = {**RULE_MAPPED, **MODEL_MAPPED}

# Everything else in the dataset, grouped by why we do not model it.
# Promoting a label out of here is a deliberate act with a detector or a
# policy profile and a taxonomy row attached.
UNMODELLED: dict[str, tuple[str, ...]] = {
    # HIPAA #2 is geographic subdivisions SMALLER than a state. A state is
    # explicitly retainable under Safe Harbor, so tagging it as an identifier
    # would be wrong, not merely conservative.
    "not_an_identifier": ("state",),
    # Credentials: real, valuable, but outside the PII/PHI framing. A
    # "secrets" policy profile (M3) is where these belong -- and they are
    # exactly what a `--fail-on-detect` CI user wants, so this is a product
    # gap, not a permanent exclusion.
    "secrets": ("password", "api_key", "cvv", "pin", "http_cookie"),
    # GDPR special-category attributes: not HIPAA identifiers. They belong to
    # a gdpr_special profile (M3), never to the HIPAA one.
    "gdpr_special": ("race_ethnicity", "religious_belief", "political_view",
                     "sexuality"),
    # Quasi-identifiers and non-identifying context. `age` is a HIPAA #3
    # identifier only above 89, and `date`/`time` only when tied to an
    # individual -- distinctions the model cannot draw yet, and emitting them
    # unconditionally would flood every document. M3 co-occurrence scoring
    # (docs/IMPROVEMENT_PLAN.md Track E) is where these earn their place.
    "attributes": ("date", "time", "age", "gender", "language", "occupation",
                   "education_level", "employment_status", "blood_type",
                   "company_name", "country"),
}

ALL_UNMODELLED: frozenset[str] = frozenset(
    label for group in UNMODELLED.values() for label in group
)

assert not (set(NEMOTRON_TO_ENTITY) & ALL_UNMODELLED), "a label cannot be both"
assert set(MODEL_MAPPED.values()) <= STAGE2_TYPES, (
    "a label adopted at Stage 2 must be declared in STAGE2_TYPES"
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
