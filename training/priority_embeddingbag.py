"""NumPy runtime for a compact low-rank EmbeddingBag tagger."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from training.priority_hash import (
    MAX_DOCUMENT_FEATURES,
    MAX_TOKENS,
    N_FEATURES,
    document_features,
)


def factorize_positive_weights(
    weights: np.ndarray, rank: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return feature embeddings and output head for positive linear weights."""
    if rank <= 0 or rank > min(weights.shape):
        raise ValueError("rank must be in [1, min(weight dimensions)]")
    positive = np.maximum(weights.astype(np.float32), 0.0)
    left, singular, right = np.linalg.svd(positive, full_matrices=False)
    root = np.sqrt(singular[:rank])
    embeddings = right[:rank].T * root[np.newaxis, :]
    head = left[:, :rank] * root[np.newaxis, :]
    return embeddings.astype(np.float32), head.astype(np.float32)


def embeddingbag_scores(
    embeddings: np.ndarray,
    head: np.ndarray,
    features: np.ndarray,
) -> np.ndarray:
    if len(features):
        bag = embeddings[features].mean(axis=0)
    else:
        bag = np.zeros(embeddings.shape[1], dtype=np.float32)
    return head @ bag


@dataclass(frozen=True)
class LowRankEmbeddingBagModel:
    labels: tuple[str, ...]
    embeddings: np.ndarray
    head: np.ndarray
    calibration: np.ndarray
    bias: np.ndarray
    thresholds: np.ndarray
    read_window_chars: int
    max_tokens: int = MAX_TOKENS
    max_document_features: int = MAX_DOCUMENT_FEATURES

    @property
    def score_mode(self) -> str:
        return f"embeddingbag_rank_{self.embeddings.shape[1]}"

    def predict_scores(self, text: str) -> np.ndarray:
        features = document_features(
            text[: self.read_window_chars],
            n_features=self.embeddings.shape[0],
            max_tokens=self.max_tokens,
            max_features=self.max_document_features,
        )
        return self.predict_scores_from_features(features)

    def predict_scores_from_features(self, features: np.ndarray) -> np.ndarray:
        raw = embeddingbag_scores(self.embeddings, self.head, features)
        return self.calibration @ raw + self.bias

    def predict_from_features(self, features: np.ndarray) -> list[str]:
        scores = self.predict_scores_from_features(features)
        return [
            label for label, keep in zip(self.labels, scores >= self.thresholds) if keep
        ]

    def predict(self, text: str) -> list[str]:
        scores = self.predict_scores(text)
        return [
            label for label, keep in zip(self.labels, scores >= self.thresholds) if keep
        ]

    def save(self, directory: Path, metadata: dict[str, Any] | None = None) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / "model.npz",
            embeddings=self.embeddings.astype(np.float16),
            head=self.head.astype(np.float16),
            calibration=self.calibration.astype(np.float32),
            bias=self.bias.astype(np.float32),
            thresholds=self.thresholds.astype(np.float32),
        )
        manifest = {
            "format": "pii-priority-embeddingbag-v1",
            "labels": list(self.labels),
            "rank": self.embeddings.shape[1],
            "n_features": self.embeddings.shape[0],
            "read_window_chars": self.read_window_chars,
            "max_tokens": self.max_tokens,
            "max_document_features": self.max_document_features,
            "metadata": metadata or {},
        }
        (directory / "model.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> LowRankEmbeddingBagModel:
        manifest = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        with np.load(directory / "model.npz", allow_pickle=False) as stored:
            return cls(
                labels=tuple(manifest["labels"]),
                embeddings=stored["embeddings"].astype(np.float32),
                head=stored["head"].astype(np.float32),
                calibration=stored["calibration"].astype(np.float32),
                bias=stored["bias"].astype(np.float32),
                thresholds=stored["thresholds"].astype(np.float32),
                read_window_chars=int(manifest["read_window_chars"]),
                max_tokens=int(manifest["max_tokens"]),
                max_document_features=int(manifest["max_document_features"]),
            )


def identity_calibration(n_labels: int) -> tuple[np.ndarray, np.ndarray]:
    return np.eye(n_labels, dtype=np.float32), np.zeros(n_labels, dtype=np.float32)


def default_model_shapes(rank: int) -> tuple[tuple[int, int], tuple[int, int]]:
    return (N_FEATURES, rank), (0, rank)
