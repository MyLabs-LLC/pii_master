"""Table-driven true-positive / false-positive cases, one table per detector.

Each row is (text, [(EntityType, matched_text), ...]); an empty expected list
is a false-positive case the detector must NOT fire on. These tables are the
seed of the frozen hard-case corpus (docs/DESIGN.md section 10): add cases,
never remove them.
"""

import pytest

from pii_master.detectors import (
    AccountNumberDetector,
    CreditCardDetector,
    DateOfBirthDetector,
    EmailDetector,
    HealthPlanIdDetector,
    IpAddressDetector,
    Ipv6AddressDetector,
    MrnDetector,
    SsnDetector,
    UrlDetector,
    UsDriverLicenseDetector,
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
        ("born 14 March 1985 in Ohio", [(EntityType.DATE_DOB, "14 March 1985")]),
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


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "gateway 2001:0db8:85a3:0000:0000:8a2e:0370:7334 up",
            [(EntityType.IP_ADDRESS, "2001:0db8:85a3:0000:0000:8a2e:0370:7334")],
        ),
        ("loopback ::1 works", [(EntityType.IP_ADDRESS, "::1")]),
        # Sentence-ending period must not suppress the match.
        (
            "The address is 2001:db8::8a2e:370:7334.",
            [(EntityType.IP_ADDRESS, "2001:db8::8a2e:370:7334")],
        ),
        ("std::vector<int> is C++ code", []),
        ("meeting ran 12:30:45 exactly", []),
        ("the scope :: resolution operator", []),
    ],
)
def test_ipv6(text, expected):
    check(Ipv6AddressDetector(), text, expected)


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "Read https://example.com/profile?id=7 now",
            [(EntityType.URL, "https://example.com/profile?id=7")],
        ),
        # Sentence-ending period must not be swallowed into the URL.
        ("Docs at www.example.org/help.", [(EntityType.URL, "www.example.org/help")]),
        ("(see https://a.io/x)", [(EntityType.URL, "https://a.io/x")]),
        ("plain example.com is not matched in v1", []),
        ("broken https:// scheme only", []),
    ],
)
def test_url(text, expected):
    check(UrlDetector(), text, expected)


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "Account number: 8272-1189-90 was closed",
            [(EntityType.ACCOUNT_NUMBER, "8272-1189-90")],
        ),
        ("acct #99881234 flagged", [(EntityType.ACCOUNT_NUMBER, "99881234")]),
        ("wire to account 123456789 today", [(EntityType.ACCOUNT_NUMBER, "123456789")]),
        ("the account balance is high", []),
        ("account 42 is too short", []),
    ],
)
def test_account_number(text, expected):
    check(AccountNumberDetector(), text, expected)


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "Health plan ID: XG-448812 on file",
            [(EntityType.HEALTH_PLAN_ID, "XG-448812")],
        ),
        (
            "Beneficiary number 84-J99-1220 verified",
            [(EntityType.HEALTH_PLAN_ID, "84-J99-1220")],
        ),
        (
            "subscriber id A9-3321-77 active",
            [(EntityType.HEALTH_PLAN_ID, "A9-3321-77")],
        ),
        # Generic cues are deliberately excluded (gym/loyalty collisions).
        ("member id 99881234 renewed", []),
        ("policy number 8842113 issued", []),
    ],
)
def test_health_plan_id(text, expected):
    check(HealthPlanIdDetector(), text, expected)


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "Driver's License No: D1234567",
            [(EntityType.US_DRIVER_LICENSE, "D1234567")],
        ),
        (
            "drivers license 8829-1123-44 suspended",
            [(EntityType.US_DRIVER_LICENSE, "8829-1123-44")],
        ),
        ("DL# 99-112-83 on record", [(EntityType.US_DRIVER_LICENSE, "99-112-83")]),
        ("DL 4432198 needs a separator after bare DL", []),
        ("license to operate heavy machinery", []),
    ],
)
def test_us_driver_license(text, expected):
    check(UsDriverLicenseDetector(), text, expected)
