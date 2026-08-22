"""Network and web identifier detectors: IPv4/IPv6 addresses and URLs."""

from __future__ import annotations

import re

from ..entities import EntityType
from ..validators import ipv4_ok, ipv6_ok
from .base import RegexDetector

# A build/release/version token just before a dotted quad means it is a
# version string, not an address ("Build 10.2.1.4"). High-precision reject;
# residual ambiguity is Stage 2's problem.
_VERSION_CUE = re.compile(
    r"(?i)(?:\b(?:build|release|version|ver|rev|revision|patch|upgrade|"
    r"shipped|firmware|schema)\b\W{0,3}|\bv)$"
)


class IpAddressDetector(RegexDetector):
    # Version strings like "10.2.1.4" are irreducible false positives at
    # the rules level, hence the low base confidence; disambiguating them
    # needs Stage 2 context.
    name = "regex/ip_address"
    entity_type = EntityType.IP_ADDRESS
    # Lookarounds reject dotted runs longer than four octets while allowing
    # a sentence-ending period right after the address.
    pattern = re.compile(r"(?<!\d)(?<!\d\.)(?:\d{1,3}\.){3}\d{1,3}(?!\d)(?!\.\d)")
    base_confidence = 0.70
    use_digit_runs = True
    overshoot = 8

    version_window = 24

    def validate(self, match: re.Match[str]) -> float | None:
        if not ipv4_ok(match.group(0)):
            return None
        start = match.start()
        left = match.string[max(0, start - self.version_window):start]
        if _VERSION_CUE.search(left):
            return None
        return self.base_confidence


class Ipv6AddressDetector(RegexDetector):
    # Candidate: hex groups joined by >= 2 colons; ipaddress does the real
    # validation. Leading lookarounds keep code tokens like std::vector out.
    # The IPv4-mapped textual form (::ffff:192.0.2.1) is not covered in v1.
    name = "regex/ipv6"
    entity_type = EntityType.IP_ADDRESS
    pattern = re.compile(
        r"(?<![\w:.])[0-9A-Fa-f]{0,4}(?::[0-9A-Fa-f]{0,4}){2,7}(?![\w:])"
    )
    base_confidence = 0.70
    # A cheap colon-pair finder seeds the windows; the leading lookbehind
    # keeps mid-address windows from producing suffix matches.
    window_pattern = re.compile(r"[0-9A-Fa-f]{0,4}:[0-9A-Fa-f]{0,4}:")
    overshoot = 48

    def validate(self, match: re.Match[str]) -> float | None:
        candidate = match.group(0)
        # Reject bare "::" and other hexless runs (scope-resolution prose).
        if not any(ch in "0123456789abcdefABCDEF" for ch in candidate):
            return None
        if not ipv6_ok(candidate):
            return None
        return self.base_confidence


class UrlDetector(RegexDetector):
    # The final character class excludes closing punctuation so a
    # sentence-ending period or bracket isn't swallowed into the URL.
    name = "regex/url"
    entity_type = EntityType.URL
    pattern = re.compile(
        r"\b(?:https?://|www\.)[^\s<>\"']*[^\s<>\"'.,;:!?)\]]",
        re.IGNORECASE,
    )
    base_confidence = 0.85
    # URLs longer than the overshoot are clipped, not missed.
    hints = ("http", "www.")
    hint_window = 6
    overshoot = 2048
