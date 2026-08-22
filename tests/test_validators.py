import pytest

from pii_master.validators import (
    card_iin_known,
    ipv4_ok,
    luhn_ok,
    nanp_ok,
    plausible_dob,
    ssn_ok,
)


@pytest.mark.parametrize(
    "digits,expected",
    [
        ("4111111111111111", True),   # Visa test number
        ("378282246310005", True),    # Amex test number
        ("6011111111111117", True),   # Discover test number
        ("4111111111111112", False),  # last digit off by one
        ("1234567890123456", False),
        ("", False),
        ("4111-1111", False),         # non-digits rejected outright
    ],
)
def test_luhn(digits, expected):
    assert luhn_ok(digits) is expected


@pytest.mark.parametrize(
    "digits,expected",
    [
        ("4111111111111111", True),   # Visa
        ("5111111111111118", True),   # Mastercard 51-55
        ("2221000000000009", True),   # Mastercard 2-series
        ("378282246310005", True),    # Amex
        ("6011111111111117", True),   # Discover
        ("9999999999999999", False),
    ],
)
def test_card_iin_known(digits, expected):
    assert card_iin_known(digits) is expected


@pytest.mark.parametrize(
    "area,group,serial,expected",
    [
        ("078", "05", "1120", True),   # famous test SSN: structurally valid
        ("123", "45", "6789", True),
        ("000", "45", "6789", False),
        ("666", "45", "6789", False),
        ("900", "45", "6789", False),
        ("999", "45", "6789", False),
        ("123", "00", "6789", False),
        ("123", "45", "0000", False),
    ],
)
def test_ssn_ok(area, group, serial, expected):
    assert ssn_ok(area, group, serial) is expected


@pytest.mark.parametrize(
    "area,exchange,expected",
    [
        ("415", "555", True),
        ("123", "555", False),  # area starting 1
        ("055", "555", False),  # area starting 0
        ("415", "155", False),  # exchange starting 1
        ("415", "055", False),  # exchange starting 0
    ],
)
def test_nanp_ok(area, exchange, expected):
    assert nanp_ok(area, exchange) is expected


@pytest.mark.parametrize(
    "candidate,expected",
    [
        ("192.168.0.1", True),
        ("255.255.255.255", True),
        ("256.1.1.1", False),
        ("999.1.1.1", False),
        ("1.2.3", False),
    ],
)
def test_ipv4_ok(candidate, expected):
    assert ipv4_ok(candidate) is expected


@pytest.mark.parametrize(
    "year,month,day,expected",
    [
        (1985, 3, 14, True),
        (1900, 1, 1, True),
        (1985, 2, 30, False),  # not a real calendar date
        (1850, 3, 14, False),  # implausibly old
        (2999, 3, 14, False),  # future
        (1985, 13, 1, False),
    ],
)
def test_plausible_dob(year, month, day, expected):
    assert plausible_dob(year, month, day) is expected
