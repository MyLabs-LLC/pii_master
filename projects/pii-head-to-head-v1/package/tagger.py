"""Command-line and Python entry point for the packaged pii-steady-aim cascade.

    from tagger import Tagger
    t = Tagger()
    t.has_pii(text)     # -> bool   : does this document contain sensitive PII
    t.predict(text)     # -> [str]  : which tags, empty when the gate stays shut

    python tagger.py --file report.docx.txt
    python tagger.py --text "SSN 123-45-6789"

`has_pii` is the cheap question and the one this model was built for: it costs
one dot product and never scores the 58 tag heads. `predict` runs the full
cascade.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_BUNDLE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_BUNDLE_ROOT / "runtime"))

from quiet_runtime import QuietCascade  # noqa: E402


class Tagger:
    """Load the frozen cascade and answer both questions about a document."""

    def __init__(self, model_dir: str | Path | None = None) -> None:
        self.model = QuietCascade.load(
            Path(model_dir) if model_dir else _BUNDLE_ROOT / "models" / "model")

    @property
    def read_window_chars(self) -> int:
        return int(self.model.window)

    @property
    def labels(self) -> tuple[str, ...]:
        return self.model.labels

    def has_pii(self, text: str) -> bool:
        return self.model.has_pii(text)

    def predict(self, text: str) -> list[str]:
        return self.model.predict(text)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tag one document for sensitive data (PII / PHI / PCI)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="document text")
    source.add_argument("--file", type=Path, help="UTF-8 text file")
    parser.add_argument("--gate-only", action="store_true",
                        help="answer only 'does this contain sensitive PII'")
    args = parser.parse_args()

    text = (args.text if args.text is not None
            else args.file.read_text(encoding="utf-8", errors="ignore"))
    tagger = Tagger()
    payload = {"has_pii": tagger.has_pii(text),
               "read_window_chars": tagger.read_window_chars}
    if not args.gate_only:
        payload["labels"] = tagger.predict(text)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
