"""Self-contained inference for the pii-steady-aim cascade. NumPy only.

Everything the model needs to run is in this file: the feature extractor, the
cascade, and the loader. It deliberately imports nothing from the project that
trained it -- a bundle whose loader reaches back into a training tree is not a
delivery, it is a pointer to the machine the packaging exists to escape.

The feature extractor is vendored verbatim from the training code rather than
reimplemented. A "cleaner" rewrite here would be a second implementation of the
one thing that must be bit-identical between fitting and serving.
"""

from __future__ import annotations

import json
import math
import re
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'’-]{0,39}|\d+(?:[-./:]\d+){0,5}")
_DIGIT_RE = re.compile(r"\d")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

#: Value-redacted shape cues. These fire on the *form* of an identifier, never
#: on its content, so no observed value is retained in a feature name.
_SHAPE_PATTERNS = {
    "ssn": re.compile(r"(?<!\d)\d{3}[- ]?\d{2}[- ]?\d{4}(?!\d)"),
    "itin": re.compile(r"(?<!\d)9\d{2}[- ]?(?:7\d|8[0-8])[- ]?\d{4}(?!\d)"),
    "card": re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)"),
    "iban": re.compile(r"(?i)(?<![A-Z0-9])[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]){11,30}(?![A-Z0-9])"),
    "email": re.compile(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "ipv4": re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)"),
    "ipv6": re.compile(r"(?i)(?<![0-9a-f])(?:[0-9a-f]{1,4}:){2,7}[0-9a-f]{0,4}(?![0-9a-f])"),
}


def _normalize_token(token: str) -> str:
    return _DIGIT_RE.sub("0", token.casefold().replace("’", "'"))


def _hash_feature(feature: str, n_features: int) -> int:
    return zlib.crc32(feature.encode("utf-8")) & (n_features - 1)


def document_features(text: str, *, n_features: int, max_tokens: int,
                      max_features: int) -> np.ndarray:
    """Deterministic, value-redacted hashed word/cue features."""
    if n_features <= 0 or n_features & (n_features - 1):
        raise ValueError("n_features must be a power of two")
    tokens = [_normalize_token(m.group()) for m in _TOKEN_RE.finditer(text)][:max_tokens]
    names: set[str] = set()
    previous = ""
    for token in tokens:
        if not token:
            continue
        names.add(f"u:{token}")
        compact = _NON_ALNUM_RE.sub("", token)
        if len(compact) >= 6 and compact.isalpha():
            names.add(f"p:{compact[:5]}")
            names.add(f"s:{compact[-5:]}")
        if previous:
            names.add(f"b:{previous}|{token}")
        previous = token
    for name, pattern in _SHAPE_PATTERNS.items():
        if pattern.search(text):
            names.add(f"r:{name}")
    names.add(f"m:length:{min(20, int(math.log2(max(1, len(text)))))}")
    indices = sorted({_hash_feature(n, n_features) for n in names})
    if len(indices) > max_features:
        step = len(indices) / max_features
        indices = [indices[int(i * step)] for i in range(max_features)]
    return np.asarray(indices, dtype=np.int32)


@dataclass(frozen=True)
class QuietCascade:
    """A document gate in front of 58 per-tag heads.

    A clean document is decided by one dot product and never reaches the heads,
    which is both the precision mechanism and the reason a negative costs a
    fraction of a positive.
    """

    labels: tuple[str, ...]
    gate_weights: np.ndarray
    gate_intercept: float
    gate_threshold: float
    tag_weights: np.ndarray
    tag_thresholds: np.ndarray
    score_mode: str
    window: int
    max_tokens: int
    max_features: int
    n_features: int

    def features(self, text: str) -> np.ndarray:
        return document_features(text[: self.window], n_features=self.n_features,
                                 max_tokens=self.max_tokens, max_features=self.max_features)

    def gate_score(self, idx: np.ndarray) -> float:
        return float(self.gate_weights[idx].sum() + self.gate_intercept)

    def tag_scores(self, idx: np.ndarray) -> np.ndarray:
        cols = self.tag_weights[:, idx]
        if self.score_mode == "sum":
            return cols.sum(axis=1)
        if self.score_mode == "mean":
            return (cols.mean(axis=1) if cols.shape[1]
                    else np.zeros(len(self.labels), dtype=np.float32))
        if self.score_mode.startswith("top"):
            k = int(self.score_mode[3:])
            values = np.maximum(cols, 0.0)
            if not values.shape[1]:
                return np.zeros(len(self.labels), dtype=np.float32)
            kk = min(k, values.shape[1])
            return np.partition(values, -kk, axis=1)[:, -kk:].mean(axis=1)
        raise ValueError(f"unknown score mode: {self.score_mode}")

    def has_pii(self, text: str) -> bool:
        """The document question alone, without scoring 58 heads."""
        idx = self.features(text)
        return bool(len(idx)) and self.gate_score(idx) >= self.gate_threshold

    def predict(self, text: str) -> list[str]:
        """The document's sensitive tags; empty when the gate stays shut."""
        idx = self.features(text)
        if not len(idx) or self.gate_score(idx) < self.gate_threshold:
            return []
        scores = self.tag_scores(idx)
        return [lab for lab, keep in zip(self.labels, scores >= self.tag_thresholds) if keep]

    @classmethod
    def load(cls, directory: str | Path) -> QuietCascade:
        directory = Path(directory)
        cfg = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        with np.load(directory / "weights.npz", allow_pickle=False) as z:
            return cls(
                labels=tuple(cfg["labels"]),
                gate_weights=z["gate_weights"].astype(np.float32),
                gate_intercept=float(cfg["gate_intercept"]),
                gate_threshold=float(cfg["gate_threshold"]),
                tag_weights=z["tag_weights"].astype(np.float32),
                tag_thresholds=z["tag_thresholds"].astype(np.float32),
                score_mode=cfg["score_mode"],
                window=int(cfg["window"]),
                max_tokens=int(cfg["max_tokens"]),
                max_features=int(cfg["max_features"]),
                n_features=int(cfg["n_features"]),
            )

    @property
    def config(self) -> dict[str, Any]:
        return {"n_labels": len(self.labels), "score_mode": self.score_mode,
                "read_window_chars": self.window, "max_tokens": self.max_tokens,
                "max_features": self.max_features, "n_features": self.n_features,
                "gate_threshold": self.gate_threshold,
                "n_enabled_tags": int(np.isfinite(self.tag_thresholds).sum())}


__all__ = ["QuietCascade", "document_features"]
