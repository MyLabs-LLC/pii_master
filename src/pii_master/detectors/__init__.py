"""Detector implementations and the default registry."""

from __future__ import annotations

from .base import Detector, RegexDetector
from .contact import EmailDetector, UsPhoneDetector
from .financial import CreditCardDetector
from .government import SsnDetector
from .medical import DateOfBirthDetector, MrnDetector
from .network import IpAddressDetector

__all__ = [
    "Detector",
    "RegexDetector",
    "EmailDetector",
    "UsPhoneDetector",
    "SsnDetector",
    "CreditCardDetector",
    "IpAddressDetector",
    "DateOfBirthDetector",
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
        CreditCardDetector(),
        IpAddressDetector(),
        DateOfBirthDetector(),
        MrnDetector(),
    ]
