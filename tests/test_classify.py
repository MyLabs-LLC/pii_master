import json

import pytest

from pii_master.classify import scan_text
from pii_master.entities import DocLabel


def test_clean_prose_is_none_with_zero_risk():
    report = scan_text("The quarterly meeting moved to the large room.")
    assert report.label is DocLabel.NONE
    assert report.risk_score == 0.0
    assert report.entities == []


def test_email_only_is_pii():
    report = scan_text("Reach me at jane.doe@example.com about the invoice.")
    assert report.label is DocLabel.PII


def test_mrn_alone_is_phi_without_medical_keywords():
    report = scan_text("Ref MRN: 4829471 for the transfer.")
    assert report.label is DocLabel.PHI
    assert any("MRN" in r for r in report.reasons)


def test_health_plan_id_alone_is_phi():
    report = scan_text("Beneficiary number 84-J99-1220 confirmed.")
    assert report.label is DocLabel.PHI


@pytest.mark.parametrize(
    "text",
    [
        # Word-boundary regressions: "patient" inside "impatient".
        "The impatient customer called about shipping.",
        "Our cloud provider raised prices.",
        "The matrix proxy configuration is fine.",
        "The therapists office is closed",  # "therapist" is not a listed term
    ],
)
def test_medical_context_is_word_bounded(text):
    from pii_master.classify import has_medical_context

    assert has_medical_context(text) is False


@pytest.mark.parametrize(
    "text",
    [
        "Patient seen in clinic today.",
        "healthcare provider assigned at intake",
        "Rx refill approved.",
        "Reviewed the medical record before discharge.",
    ],
)
def test_medical_context_still_fires_on_real_cues(text):
    from pii_master.classify import has_medical_context

    assert has_medical_context(text) is True


@pytest.mark.parametrize(
    "text",
    [
        # Every one of these produced a false PHI before the Track A fixes.
        "The chart 4829471 is in the appendix.",
        "subscriber id A9-3321-77 for the magazine",
        "SSN 123-45-6789 from the cloud provider.",
    ],
)
def test_no_false_phi(text):
    assert scan_text(text).label is not DocLabel.PHI


def test_phi_specific_rule_is_taxonomy_driven():
    """Rule 2 must read TAXONOMY, not a hardcoded list of types."""
    from pii_master.entities import TAXONOMY, EntityType

    phi_specific = {t for t, info in TAXONOMY.items() if info.phi_specific}
    assert phi_specific == {EntityType.MRN, EntityType.HEALTH_PLAN_ID}


def test_taxonomy_covers_every_entity_type():
    from pii_master.entities import TAXONOMY, EntityType

    assert set(TAXONOMY) == set(EntityType)


def test_ssn_in_medical_context_is_phi():
    report = scan_text(
        "The patient (SSN 123-45-6789) received a diagnosis on admission."
    )
    assert report.label is DocLabel.PHI


def test_ssn_alone_is_pii():
    report = scan_text("Payroll form lists SSN 123-45-6789 for withholding.")
    assert report.label is DocLabel.PII


def test_risk_monotonic_under_added_entities():
    smaller = scan_text("Contact jane@example.com.")
    larger = scan_text("Contact jane@example.com. SSN 123-45-6789.")
    assert smaller.risk_score <= larger.risk_score


def test_per_type_cap_limits_contribution():
    three = scan_text("a@x.com b@x.com c@x.com")
    four = scan_text("a@x.com b@x.com c@x.com d@x.com")
    assert three.risk_score == pytest.approx(four.risk_score)
    assert four.counts["EMAIL"] == 4  # counts still report the true number


def test_risk_clamped_to_100():
    report = scan_text(
        "SSN 123-45-6789 and 987-65-4321 and 456-78-9012, "
        "card 4111 1111 1111 1111, MRN: 4829471"
    )
    assert report.risk_score <= 100.0


def test_report_round_trips_through_json():
    report = scan_text("Patient DOB: 03/14/1985, MRN: 4829471.")
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["label"] == "PHI"
    assert payload["counts"]["MRN"] == 1
    types = {e["type"] for e in payload["entities"]}
    assert types == {"DATE_DOB", "MRN"}
