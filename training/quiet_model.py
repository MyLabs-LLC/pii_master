"""The serving artifact: a document gate in front of per-tag heads.

Inference is two dot products over the same hashed feature indices, in the order
that makes the document decision cheap:

1. extract features once for the configured read profile;
2. score the **gate** -- one weight vector -- and stop if the document is clean;
3. only then score the 58 **tag heads** and apply their per-label thresholds.

Stopping at step 2 is the whole design. It is what makes "this document contains
no PII" a property the artifact asserts rather than something a downstream
filter has to infer from an empty tag list, and it is why a clean document costs
roughly a fifth of what a positive one does.

NumPy is the only runtime dependency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from training.priority_hash import document_features


@dataclass(frozen=True)
class QuietCascade:
    labels: tuple[str, ...]
    gate_weights: np.ndarray      # (n_features,)
    gate_intercept: float
    gate_threshold: float
    tag_weights: np.ndarray       # (n_labels, n_features)
    tag_thresholds: np.ndarray    # (n_labels,)
    score_mode: str
    window: int
    max_tokens: int
    max_features: int
    n_features: int

    # ------------------------------------------------------------------ score
    def features(self, text: str) -> np.ndarray:
        return document_features(
            text[: self.window], n_features=self.n_features,
            max_tokens=self.max_tokens, max_features=self.max_features,
        )

    def gate_score(self, idx: np.ndarray) -> float:
        return float(self.gate_weights[idx].sum() + self.gate_intercept)

    def tag_scores(self, idx: np.ndarray) -> np.ndarray:
        cols = self.tag_weights[:, idx]
        if self.score_mode == "sum":
            return cols.sum(axis=1)
        if self.score_mode == "mean":
            return cols.mean(axis=1) if cols.shape[1] else np.zeros(len(self.labels), np.float32)
        if self.score_mode.startswith("top"):
            k = int(self.score_mode[3:])
            values = np.maximum(cols, 0.0)
            if not values.shape[1]:
                return np.zeros(len(self.labels), dtype=np.float32)
            kk = min(k, values.shape[1])
            return np.partition(values, -kk, axis=1)[:, -kk:].mean(axis=1)
        raise ValueError(f"unknown score mode: {self.score_mode}")

    # ------------------------------------------------------------------ predict
    def predict(self, text: str) -> list[str]:
        """The document's sensitive tags -- empty when the gate stays shut."""
        idx = self.features(text)
        if not len(idx) or self.gate_score(idx) < self.gate_threshold:
            return []
        scores = self.tag_scores(idx)
        return [label for label, keep in zip(self.labels, scores >= self.tag_thresholds) if keep]

    def has_pii(self, text: str) -> bool:
        """The document-level question on its own, without scoring 58 heads."""
        idx = self.features(text)
        return bool(len(idx)) and self.gate_score(idx) >= self.gate_threshold

    # ------------------------------------------------------------------ io
    def save(self, directory: Path, metadata: dict[str, Any] | None = None) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        # float16 halves a 61 MB weight matrix at a cost far below the
        # threshold granularity; the thresholds are re-derived from the stored
        # weights at materialisation time, so nothing drifts.
        np.savez_compressed(
            directory / "weights.npz",
            gate_weights=self.gate_weights.astype(np.float16),
            tag_weights=self.tag_weights.astype(np.float16),
            tag_thresholds=self.tag_thresholds.astype(np.float32),
        )
        (directory / "model.json").write_text(json.dumps({
            "labels": list(self.labels),
            "gate_intercept": self.gate_intercept,
            "gate_threshold": self.gate_threshold,
            "score_mode": self.score_mode,
            "window": self.window,
            "max_tokens": self.max_tokens,
            "max_features": self.max_features,
            "n_features": self.n_features,
            "metadata": metadata or {},
        }, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> QuietCascade:
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
        return {
            "n_labels": len(self.labels), "score_mode": self.score_mode,
            "window": self.window, "max_tokens": self.max_tokens,
            "max_features": self.max_features, "n_features": self.n_features,
            "gate_threshold": self.gate_threshold,
            "n_enabled_tags": int(np.isfinite(self.tag_thresholds).sum()),
        }


__all__ = ["QuietCascade"]
