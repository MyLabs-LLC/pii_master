"""pii_master: fast, CPU-friendly PII/PHI detection and document classification.

See docs/DESIGN.md for the architecture and roadmap. Quick start:

    from pii_master import scan_text
    report = scan_text("Patient MRN: 4829471, DOB: 03/14/1985")
    report.label          # DocLabel.PHI
    report.to_dict()      # JSON-ready report
"""

from .classify import DocumentClassifier, scan_text
from .entities import DocLabel, EntityType
from .models import DocumentReport, Entity
from .pipeline import Pipeline

__version__ = "0.2.0"

__all__ = [
    "DocLabel",
    "DocumentClassifier",
    "DocumentReport",
    "Entity",
    "EntityType",
    "Pipeline",
    "scan_text",
    "__version__",
]
