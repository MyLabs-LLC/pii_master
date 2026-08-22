"""Pure validation functions applied to regex candidates.

Design principle (docs/DESIGN.md section 7): the regex over-generates, the
validator rejects or scores. Everything here is stdlib-only and side-effect
free so it can be unit-tested in isolation and reused to re-verify
model-proposed spans at Stage 2.
"""

from __future__ import annotations

import datetime
import ipaddress


def luhn_ok(digits: str) -> bool:
    """Standard mod-10 checksum over a string of ASCII digits."""
    if not digits.isdigit():
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = ord(ch) - 48
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def card_iin_known(digits: str) -> bool:
    """True if the number starts with a major-network IIN prefix.

    Used only to boost confidence, never to reject — IIN ranges churn.
    Visa 4; Mastercard 51-55 and 2221-2720; Amex 34/37; Discover 6011/65.
    """
    if digits.startswith("4"):
        return True
    if len(digits) >= 2 and 51 <= int(digits[:2]) <= 55:
        return True
    if len(digits) >= 4 and 2221 <= int(digits[:4]) <= 2720:
        return True
    if digits[:2] in ("34", "37", "65"):
        return True
    if digits.startswith("6011"):
        return True
    return False


def ssn_ok(area: str, group: str, serial: str) -> bool:
    """Reject SSNs in never-issued ranges.

    Never issued: area 000, 666, 900-999; group 00; serial 0000. Post-2011
    randomization means the area no longer encodes geography, so no
    geographic checks. Famous test SSNs like 078-05-1120 are structurally
    valid and are deliberately accepted.
    """
    if area == "000" or area == "666" or area >= "900":
        return False
    if group == "00" or serial == "0000":
        return False
    return True


def nanp_ok(area: str, exchange: str) -> bool:
    """NANP rule: area code and exchange must start with digit 2-9."""
    return area[0] in "23456789" and exchange[0] in "23456789"


def ipv4_ok(candidate: str) -> bool:
    try:
        ipaddress.IPv4Address(candidate)
    except ValueError:
        return False
    return True


def plausible_dob(year: int, month: int, day: int) -> bool:
    """Real calendar date with a year plausible for a living person's birth."""
    try:
        datetime.date(year, month, day)
    except ValueError:
        return False
    return 1900 <= year <= datetime.date.today().year
