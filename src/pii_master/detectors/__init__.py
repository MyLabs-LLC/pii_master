"""Detector implementations and the default registry."""

from __future__ import annotations

from .base import CueAnchoredIdDetector, Detector, RegexDetector
from .contact import EmailDetector, UsPhoneDetector
from .financial import AccountNumberDetector, CreditCardDetector
from .government import SsnDetector, UsDriverLicenseDetector
from .medical import DateOfBirthDetector, HealthPlanIdDetector, MrnDetector
from .network import IpAddressDetector, Ipv6AddressDetector, UrlDetector

__all__ = [
    "Detector",
    "RegexDetector",
    "CueAnchoredIdDetector",
    "EmailDetector",
    "UsPhoneDetector",
    "SsnDetector",
    "UsDriverLicenseDetector",
    "CreditCardDetector",
    "AccountNumberDetector",
    "IpAddressDetector",
    "Ipv6AddressDetector",
    "UrlDetector",
    "DateOfBirthDetector",
    "HealthPlanIdDetector",
    "MrnDetector",
    "default_detectors",
]


def default_detectors() -> list[Detector]:
    """Fresh instances of all v1 detectors.

    Order is fixed: detector name participates in overlap-resolution
    tie-breaking, so a stable order keeps the pipeline deterministic.
    """
    return [
        EmailDetector(),
        UsPhoneDetector(),
        SsnDetector(),
        UsDriverLicenseDetector(),
        CreditCardDetector(),
        AccountNumberDetector(),
        IpAddressDetector(),
        Ipv6AddressDetector(),
        UrlDetector(),
        DateOfBirthDetector(),
        HealthPlanIdDetector(),
        MrnDetector(),
    ]
