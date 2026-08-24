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


def ipv6_ok(candidate: str) -> bool:
    try:
        ipaddress.IPv6Address(candidate)
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


def aba_ok(digits: str) -> bool:
    """ABA routing number checksum (9 digits).

    ``3(d1+d4+d7) + 7(d2+d5+d8) + (d3+d6+d9) ≡ 0 (mod 10)``. This is the
    Luhn of the banking family: a random 9-digit run survives 1-in-10, and
    a rule span that passed it is a verified fact, so it earns checksum
    fusion precedence the same way a Luhn-valid card does.
    """
    if len(digits) != 9 or not digits.isdigit():
        return False
    weights = (3, 7, 1, 3, 7, 1, 3, 7, 1)
    return sum(int(d) * w for d, w in zip(digits, weights)) % 10 == 0


# ISO 3779 VIN: letters I, O, Q are not used. Transliteration and weights
# are the standard SAE J853 / 49 CFR 565.15 tables.
_VIN_TRANS = {
    **{str(i): i for i in range(10)},
    **dict(zip("ABCDEFGHJKLMNPRSTUVWXYZ",
               (1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 7, 9, 2, 3, 4, 5, 6, 7, 8, 9))),
}
_VIN_WEIGHTS = (8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2)


def vin_ok(vin: str) -> bool:
    """ISO 3779 VIN check digit (position 9)."""
    vin = vin.upper()
    if len(vin) != 17:
        return False
    if any(ch not in _VIN_TRANS for ch in vin):
        return False
    total = sum(_VIN_TRANS[ch] * w for ch, w in zip(vin, _VIN_WEIGHTS))
    check = total % 11
    expected = "X" if check == 10 else str(check)
    return vin[8] == expected


def swift_ok(code: str) -> bool:
    """ISO 9362 BIC: 8 or 11 alphanumeric characters, bank+country+location.

    Structure only — we do not consult the SWIFT directory. All-alpha 8-char
    strings collide with English words (SOFTWARE parses as SOFT+WA+RE), so
    the detector, not this function, decides whether a cue or a digit is
    required before emitting.
    """
    code = code.replace(" ", "").replace("-", "").upper()
    if len(code) not in (8, 11):
        return False
    if not (code[:4].isalpha() and code[4:6].isalpha() and code[6:8].isalnum()):
        return False
    if len(code) == 11 and not code[8:11].isalnum():
        return False
    return True


def mac_ok(addr: str) -> bool:
    """Six hex octets separated by ``:`` or ``-``, consistently."""
    sep = ":" if ":" in addr else "-" if "-" in addr else ""
    if not sep or (":" in addr and "-" in addr):
        return False
    parts = addr.split(sep)
    if len(parts) != 6:
        return False
    return all(len(p) == 2 and all(c in "0123456789abcdefABCDEF" for c in p)
               for p in parts)


# IRS EIN prefix ranges that have actually been issued. A 2+7 digit run
# with a never-issued prefix is not an EIN; one with a real prefix still
# needs a cue (the detector's job) because the format alone is common.
_EIN_PREFIXES = frozenset({
    *range(1, 7), *range(10, 17), *range(20, 28), *range(30, 49),
    *range(50, 69), *range(71, 78), *range(80, 89), *range(90, 96),
    98, 99,
})


def ein_ok(digits: str) -> bool:
    """US Employer Identification Number: 9 digits and an issued prefix."""
    if len(digits) != 9 or not digits.isdigit():
        return False
    return int(digits[:2]) in _EIN_PREFIXES
