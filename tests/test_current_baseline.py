from __future__ import annotations

from training.current_baseline import entity_tags


def test_priority_rule_entity_crosswalk() -> None:
    assert entity_tags("SSN") == {"sensitive_pii_social_security_number"}
    assert entity_tags("CREDIT_CARD") == {
        "sensitive_pci_credit_card",
        "sensitive_pci_credit_card_number",
    }
    assert entity_tags("MRN") == {"sensitive_phi_medical_record_number_mrn"}


def test_ip_version_is_not_double_counted() -> None:
    assert entity_tags("IP_ADDRESS", "127.0.0.1") == {"sensitive_pii_ipv4"}
    assert entity_tags("IP_ADDRESS", "2001:db8::1") == {"sensitive_pii_ipv6"}


def test_unknown_entity_is_ignored() -> None:
    assert entity_tags("NOT_A_REAL_ENTITY") == set()
