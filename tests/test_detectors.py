"""Table-driven true-positive / false-positive cases, one table per detector.

Each row is (text, [(EntityType, matched_text), ...]); an empty expected list
is a false-positive case the detector must NOT fire on. These tables are the
seed of the frozen hard-case corpus (docs/DESIGN.md section 10): add cases,
never remove them.
"""

import pytest

from pii_master.detectors import (
    CreditCardDetector,
    DateOfBirthDetector,
    EmailDetector,
    IpAddressDetector,
    MrnDetector,
    SsnDetector,
    UsPhoneDetector,
)
from pii_master.entities import EntityType


def check(detector, text, expected):
    found = [(e.type, e.text) for e in detector.detect(text)]
    assert found == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "Contact jane.doe@example.com today.",
            [(EntityType.EMAIL, "jane.doe@example.com")],
        ),
        (
            "reply to bob+tag@sub.example.co.uk.",
            [(EntityType.EMAIL, "bob+tag@sub.example.co.uk")],
        ),
        ("no emails here @ all", []),
        (".lead@example.com is malformed local part", []),
    ],
)
def test_email(text, expected):
    check(EmailDetector(), text, expected)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Call (415) 555-2671 now", [(EntityType.PHONE_US, "(415) 555-2671")]),
        ("Call +1 415-555-2671 now", [(EntityType.PHONE_US, "+1 415-555-2671")]),
        ("Call 415.555.2671 now", [(EntityType.PHONE_US, "415.555.2671")]),
        # Sentence-ending period must not suppress the match.
        ("Call (415) 555-2671.", [(EntityType.PHONE_US, "(415) 555-2671")]),
        ("123-456-7890 is not valid NANP", []),  # area code starts with 1
        ("tracking 415555267190 embedded digits", []),  # inside a longer run
        # "1." parses as a country-code prefix: dotted-with-country-code is a
        # real phone format, so this version-string collision is a documented
        # irreducible FP at the rules level (Stage 2 context disambiguates).
        (
            "release 1.415.555.2671 is a version string",
            [(EntityType.PHONE_US, "1.415.555.2671")],
        ),
    ],
)
def test_phone(text, expected):
    check(UsPhoneDetector(), text, expected)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("SSN: 123-45-6789.", [(EntityType.SSN, "123-45-6789")]),
        ("SSN 078-05-1120 (test number)", [(EntityType.SSN, "078-05-1120")]),
        ("SSN 123 45 6789 spaced", [(EntityType.SSN, "123 45 6789")]),
        ("bad area 666-45-6789", []),
        ("bad area 900-45-6789", []),
        ("mixed separators 123-45 6789", []),
        ("account 9123-45-67890 embedded", []),  # inside a longer digit run
        ("bare 123456789 not matched in v1", []),
    ],
)
def test_ssn(text, expected):
    check(SsnDetector(), text, expected)


def test_ssn_confidence_hyphen_higher_than_space():
    det = SsnDetector()
    hyphen = det.detect("123-45-6789")[0]
    spaced = det.detect("123 45 6789")[0]
    assert hyphen.confidence > spaced.confidence


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "card 4111 1111 1111 1111 on file",
            [(EntityType.CREDIT_CARD, "4111 1111 1111 1111")],
        ),
        (
            "amex 378282246310005 ok",
            [(EntityType.CREDIT_CARD, "378282246310005")],
        ),
        (
            "card 4111-1111-1111-1111 hyphenated",
            [(EntityType.CREDIT_CARD, "4111-1111-1111-1111")],
        ),
        ("card 4111 1111 1111 1112 fails luhn", []),
        ("mixed 4111-1111 1111-1111 separators", []),
    ],
)
def test_credit_card(text, expected):
    check(CreditCardDetector(), text, expected)


def test_credit_card_iin_boosts_confidence():
    det = CreditCardDetector()
    # Both Luhn-valid; the first has a known Visa IIN, the second does not.
    known = det.detect("4111111111111111")[0]
    unknown = det.detect("9999999999999995")[0]
    assert known.confidence > unknown.confidence


@pytest.mark.parametrize(
    "text,expected",
    [
        ("server at 192.168.0.1 responded", [(EntityType.IP_ADDRESS, "192.168.0.1")]),
        # Sentence-ending period must not suppress the match.
        ("The host is 192.168.0.1.", [(EntityType.IP_ADDRESS, "192.168.0.1")]),
        ("bad octet 999.1.1.1 ignored", []),
        ("not an ip 1.2.3.4.5 (five octets)", []),
    ],
)
def test_ip_address(text, expected):
    check(IpAddressDetector(), text, expected)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("DOB: 03/14/1985", [(EntityType.DATE_DOB, "03/14/1985")]),
        ("Date of birth 3-4-1985", [(EntityType.DATE_DOB, "3-4-1985")]),
        ("born on March 14, 1985", [(EntityType.DATE_DOB, "March 14, 1985")]),
        ("Birthdate: 1985-03-14", [(EntityType.DATE_DOB, "1985-03-14")]),
        ("Meeting on 03/14/1985", []),  # no birth cue -> not DOB
        ("DOB: 02/30/1985", []),        # not a real calendar date
        ("DOB: 03/14/1850", []),        # implausible year
    ],
)
def test_date_of_birth(text, expected):
    check(DateOfBirthDetector(), text, expected)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("MRN: 4829471", [(EntityType.MRN, "4829471")]),
        ("Medical Record Number 82-94-71A", [(EntityType.MRN, "82-94-71A")]),
        ("Chart #4829471 attached", [(EntityType.MRN, "4829471")]),
        ("chart topper of the year", []),  # cue followed by non-ID prose
        ("MRN: ABCDEF has too few digits", []),
    ],
)
def test_mrn(text, expected):
    check(MrnDetector(), text, expected)


def test_mrn_span_covers_only_the_id():
    text = "Patient MRN: 4829471 admitted"
    entity = MrnDetector().detect(text)[0]
    assert text[entity.start : entity.end] == "4829471"
