"""Network identifier detectors: IPv4 addresses. IPv6 is deferred to M1."""

from __future__ import annotations

import re

from ..entities import EntityType
from ..validators import ipv4_ok
from .base import RegexDetector


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

    def validate(self, match: re.Match[str]) -> float | None:
        if not ipv4_ok(match.group(0)):
            return None
        return self.base_confidence
