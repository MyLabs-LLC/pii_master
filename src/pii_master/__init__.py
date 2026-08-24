"""pii_master: fast, CPU-friendly PII/PHI detection and document classification.

See docs/DESIGN.md for the architecture and roadmap. Quick start:

    from pii_master import scan_text
    report = scan_text("Patient MRN: 4829471, DOB: 03/14/1985")
    report.label          # DocLabel.PHI
    report.to_dict()      # JSON-ready report
"""

from .classify import DocumentClassifier, default_pipeline, scan_text
from .entities import DocLabel, EntityType, IdentifierKind
from .models import DocumentReport, Entity
from .pipeline import Pipeline, deep_pipeline, fusion_rank

__version__ = "0.4.0"

__all__ = [
    "DocLabel",
    "DocumentClassifier",
    "DocumentReport",
    "Entity",
    "EntityType",
    "IdentifierKind",
    "Pipeline",
    "deep_pipeline",
    "default_pipeline",
    "fusion_rank",
    "scan_text",
    "__version__",
]
