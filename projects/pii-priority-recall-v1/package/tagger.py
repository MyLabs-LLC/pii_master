"""Self-contained command-line and Python entry point for the packaged tagger."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_BUNDLE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_BUNDLE_ROOT / "runtime"))

from training.priority_hash import load_priority_model


class Tagger:
    """Load the frozen fusion model and return document-level sensitive tags."""

    def __init__(self, model_dir: str | Path | None = None) -> None:
        path = Path(model_dir) if model_dir else _BUNDLE_ROOT / "models" / "model"
        self.model = load_priority_model(path)

    @property
    def read_window_chars(self) -> int:
        return int(self.model.read_window_chars)

    def predict(self, text: str) -> list[str]:
        return list(self.model.predict(text[: self.read_window_chars]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Tag one document for sensitive data")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="document text")
    source.add_argument("--file", type=Path, help="UTF-8 text file")
    args = parser.parse_args()

    text = args.text if args.text is not None else args.file.read_text(
        encoding="utf-8", errors="ignore"
    )
    tagger = Tagger()
    print(
        json.dumps(
            {
                "labels": tagger.predict(text),
                "read_window_chars": tagger.read_window_chars,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
