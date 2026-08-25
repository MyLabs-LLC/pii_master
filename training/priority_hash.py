"""Compact hashed cue-and-shape model for sensitive document tags.

Training is positive/unlabelled aware. Complete catalogue rows contribute
positive and negative evidence; partial rows contribute positives only; every
remaining document contributes unsupervised document-frequency statistics.
This lets all approved corpora influence the model without turning missing
annotations into false negatives.
"""

from __future__ import annotations

import json
import math
import re
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

N_FEATURES = 1 << 17
MAX_TOKENS = 768
MAX_DOCUMENT_FEATURES = 512
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'’-]{0,39}|\d+(?:[-./:]\d+){0,5}")
_DIGIT_RE = re.compile(r"\d")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

_SHAPE_PATTERNS = {
    "ssn": re.compile(r"(?<!\d)\d{3}[- ]?\d{2}[- ]?\d{4}(?!\d)"),
    "itin": re.compile(r"(?<!\d)9\d{2}[- ]?(?:7\d|8[0-8])[- ]?\d{4}(?!\d)"),
    "card": re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)"),
    "iban": re.compile(
        r"(?i)(?<![A-Z0-9])[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]){11,30}(?![A-Z0-9])"
    ),
    "email": re.compile(
        r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    ),
    "ipv4": re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)"),
    "ipv6": re.compile(
        r"(?i)(?<![0-9a-f])(?:[0-9a-f]{1,4}:){2,7}[0-9a-f]{0,4}(?![0-9a-f])"
    ),
}


def _normalize_token(token: str) -> str:
    lowered = token.casefold().replace("’", "'")
    return _DIGIT_RE.sub("0", lowered)


def _hash_feature(feature: str, n_features: int) -> int:
    return zlib.crc32(feature.encode("utf-8")) & (n_features - 1)


def document_features(
    text: str,
    *,
    n_features: int = N_FEATURES,
    max_tokens: int = MAX_TOKENS,
    max_features: int = MAX_DOCUMENT_FEATURES,
) -> np.ndarray:
    """Return deterministic, value-redacted hashed word/cue features."""
    if n_features <= 0 or n_features & (n_features - 1):
        raise ValueError("n_features must be a power of two")
    tokens = [_normalize_token(match.group()) for match in _TOKEN_RE.finditer(text)]
    tokens = tokens[:max_tokens]
    feature_names: set[str] = set()
    previous = ""
    for token in tokens:
        if not token:
            continue
        feature_names.add(f"u:{token}")
        compact = _NON_ALNUM_RE.sub("", token)
        if len(compact) >= 6 and compact.isalpha():
            feature_names.add(f"p:{compact[:5]}")
            feature_names.add(f"s:{compact[-5:]}")
        if previous:
            feature_names.add(f"b:{previous}|{token}")
        previous = token
    for name, pattern in _SHAPE_PATTERNS.items():
        if pattern.search(text):
            feature_names.add(f"r:{name}")
    # Length buckets let the model distinguish one-line records from long
    # forms without retaining or logging content.
    length_bucket = min(20, int(math.log2(max(1, len(text)))))
    feature_names.add(f"m:length:{length_bucket}")
    indices = sorted({_hash_feature(name, n_features) for name in feature_names})
    if len(indices) > max_features:
        # Stable sub-sampling avoids a long boilerplate document dominating
        # both fit time and serving latency.
        step = len(indices) / max_features
        indices = [indices[int(position * step)] for position in range(max_features)]
    return np.asarray(indices, dtype=np.int32)


@dataclass
class HashCounts:
    labels: tuple[str, ...]
    n_features: int
    all_df: np.ndarray
    complete_df: np.ndarray
    positive_complete_df: np.ndarray
    positive_partial_df: np.ndarray
    n_all: int
    n_complete: int
    n_positive_complete: np.ndarray
    n_positive_partial: np.ndarray

    @classmethod
    def empty(cls, labels: tuple[str, ...], n_features: int = N_FEATURES) -> HashCounts:
        shape = (len(labels), n_features)
        return cls(
            labels=labels,
            n_features=n_features,
            all_df=np.zeros(n_features, dtype=np.uint32),
            complete_df=np.zeros(n_features, dtype=np.uint32),
            positive_complete_df=np.zeros(shape, dtype=np.uint32),
            positive_partial_df=np.zeros(shape, dtype=np.uint32),
            n_all=0,
            n_complete=0,
            n_positive_complete=np.zeros(len(labels), dtype=np.uint32),
            n_positive_partial=np.zeros(len(labels), dtype=np.uint32),
        )

    def update(
        self, features: np.ndarray, gold: set[str], *, label_complete: bool
    ) -> None:
        self.n_all += 1
        self.all_df[features] += 1
        label_index = {label: index for index, label in enumerate(self.labels)}
        positives = [label_index[label] for label in gold if label in label_index]
        if label_complete:
            self.n_complete += 1
            self.complete_df[features] += 1
            for index in positives:
                self.n_positive_complete[index] += 1
                self.positive_complete_df[index, features] += 1
        else:
            for index in positives:
                self.n_positive_partial[index] += 1
                self.positive_partial_df[index, features] += 1

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            labels=np.asarray(self.labels),
            n_features=np.asarray([self.n_features]),
            all_df=self.all_df,
            complete_df=self.complete_df,
            positive_complete_df=self.positive_complete_df,
            positive_partial_df=self.positive_partial_df,
            n_all=np.asarray([self.n_all]),
            n_complete=np.asarray([self.n_complete]),
            n_positive_complete=self.n_positive_complete,
            n_positive_partial=self.n_positive_partial,
        )

    @classmethod
    def load(cls, path: Path) -> HashCounts:
        with np.load(path, allow_pickle=False) as stored:
            return cls(
                labels=tuple(map(str, stored["labels"])),
                n_features=int(stored["n_features"][0]),
                all_df=stored["all_df"],
                complete_df=stored["complete_df"],
                positive_complete_df=stored["positive_complete_df"],
                positive_partial_df=stored["positive_partial_df"],
                n_all=int(stored["n_all"][0]),
                n_complete=int(stored["n_complete"][0]),
                n_positive_complete=stored["n_positive_complete"],
                n_positive_partial=stored["n_positive_partial"],
            )


def build_weights(
    counts: HashCounts,
    *,
    alpha: float = 1.0,
    partial_weight: float = 0.5,
    min_document_frequency: int = 2,
    clip: float = 8.0,
) -> np.ndarray:
    """Build positive-vs-negative Bernoulli log-odds weights."""
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    positive = counts.positive_complete_df.astype(np.float32)
    positive += partial_weight * counts.positive_partial_df
    negative = np.broadcast_to(
        counts.complete_df[np.newaxis, :], counts.positive_complete_df.shape
    ).astype(np.float32, copy=True)
    negative -= counts.positive_complete_df
    np.maximum(negative, 0.0, out=negative)
    n_positive = counts.n_positive_complete.astype(np.float32)
    n_positive += partial_weight * counts.n_positive_partial
    n_negative = counts.n_complete - counts.n_positive_complete.astype(np.float32)
    p_positive = (positive + alpha) / (n_positive[:, np.newaxis] + 2.0 * alpha)
    p_negative = (negative + alpha) / (n_negative[:, np.newaxis] + 2.0 * alpha)
    np.clip(p_positive, 1e-6, 1.0 - 1e-6, out=p_positive)
    np.clip(p_negative, 1e-6, 1.0 - 1e-6, out=p_negative)
    weights = np.log(p_positive / (1.0 - p_positive))
    weights -= np.log(p_negative / (1.0 - p_negative))
    weights[:, counts.all_df < min_document_frequency] = 0.0
    np.clip(weights, -clip, clip, out=weights)
    return weights.astype(np.float32)


def score_modes(weights: np.ndarray, features: np.ndarray) -> dict[str, np.ndarray]:
    """Compute three cheap robust aggregations used by threshold trials."""
    n_labels = weights.shape[0]
    if not len(features):
        zeros = np.zeros(n_labels, dtype=np.float32)
        return {"top1": zeros, "top3": zeros.copy(), "top6": zeros.copy()}
    values = weights[:, features]
    values = np.maximum(values, 0.0)
    outputs: dict[str, np.ndarray] = {"top1": values.max(axis=1)}
    for top_k in (3, 6):
        actual = min(top_k, values.shape[1])
        selected = np.partition(values, -actual, axis=1)[:, -actual:]
        outputs[f"top{top_k}"] = selected.mean(axis=1)
    return outputs


@dataclass(frozen=True)
class HashCueModel:
    labels: tuple[str, ...]
    weights: np.ndarray
    thresholds: np.ndarray
    score_mode: str
    read_window_chars: int
    n_features: int = N_FEATURES
    max_tokens: int = MAX_TOKENS
    max_document_features: int = MAX_DOCUMENT_FEATURES

    def predict_scores(self, text: str) -> np.ndarray:
        features = document_features(
            text[: self.read_window_chars],
            n_features=self.n_features,
            max_tokens=self.max_tokens,
            max_features=self.max_document_features,
        )
        return self.predict_scores_from_features(features)

    def predict_scores_from_features(self, features: np.ndarray) -> np.ndarray:
        return score_modes(self.weights, features)[self.score_mode]

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
            weights=self.weights.astype(np.float16),
            thresholds=self.thresholds.astype(np.float32),
        )
        manifest = {
            "format": "pii-priority-hash-cue-v1",
            "labels": list(self.labels),
            "score_mode": self.score_mode,
            "read_window_chars": self.read_window_chars,
            "n_features": self.n_features,
            "max_tokens": self.max_tokens,
            "max_document_features": self.max_document_features,
            "metadata": metadata or {},
        }
        (directory / "model.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> HashCueModel:
        manifest = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        with np.load(directory / "model.npz", allow_pickle=False) as stored:
            weights = stored["weights"].astype(np.float32)
            thresholds = stored["thresholds"].astype(np.float32)
        fields = {
            "labels": tuple(manifest["labels"]),
            "weights": weights,
            "thresholds": thresholds,
            "score_mode": manifest["score_mode"],
            "read_window_chars": int(manifest["read_window_chars"]),
            "n_features": int(manifest["n_features"]),
            "max_tokens": int(manifest["max_tokens"]),
            "max_document_features": int(manifest["max_document_features"]),
        }
        return cls(**fields)

    def config(self) -> dict[str, Any]:
        config = asdict(self)
        config["labels"] = list(self.labels)
        config["weights"] = f"array{self.weights.shape}"
        config["thresholds"] = self.thresholds.tolist()
        return config


@dataclass(frozen=True)
class HybridPriorityModel:
    """Use a recall-max head for priority tags and an F2 head elsewhere."""

    priority_model: HashCueModel
    generic_model: HashCueModel
    priority_tags: frozenset[str]

    def __post_init__(self) -> None:
        if self.priority_model.labels != self.generic_model.labels:
            raise ValueError("hybrid component label catalogues differ")

    @property
    def labels(self) -> tuple[str, ...]:
        return self.priority_model.labels

    @property
    def read_window_chars(self) -> int:
        return max(
            self.priority_model.read_window_chars,
            self.generic_model.read_window_chars,
        )

    @property
    def score_mode(self) -> str:
        return f"priority={self.priority_model.score_mode};generic={self.generic_model.score_mode}"

    def predict(self, text: str) -> list[str]:
        features = document_features(
            text[: self.read_window_chars],
            n_features=self.priority_model.n_features,
            max_tokens=max(
                self.priority_model.max_tokens, self.generic_model.max_tokens
            ),
            max_features=max(
                self.priority_model.max_document_features,
                self.generic_model.max_document_features,
            ),
        )
        priority_predictions = set(self.priority_model.predict_from_features(features))
        generic_predictions = set(self.generic_model.predict_from_features(features))
        return [
            label
            for label in self.labels
            if (label in self.priority_tags and label in priority_predictions)
            or (label not in self.priority_tags and label in generic_predictions)
        ]

    def save(self, directory: Path, metadata: dict[str, Any] | None = None) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.priority_model.save(directory / "priority")
        self.generic_model.save(directory / "generic")
        manifest = {
            "format": "pii-priority-hybrid-v1",
            "priority_tags": sorted(self.priority_tags),
            "priority_model": "priority",
            "generic_model": "generic",
            "metadata": metadata or {},
        }
        (directory / "model.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> HybridPriorityModel:
        manifest = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        return cls(
            priority_model=HashCueModel.load(directory / manifest["priority_model"]),
            generic_model=HashCueModel.load(directory / manifest["generic_model"]),
            priority_tags=frozenset(manifest["priority_tags"]),
        )


def load_priority_model(directory: Path) -> HashCueModel | HybridPriorityModel:
    manifest = json.loads((directory / "model.json").read_text(encoding="utf-8"))
    if manifest["format"] == "pii-priority-hash-cue-v1":
        return HashCueModel.load(directory)
    if manifest["format"] == "pii-priority-hybrid-v1":
        return HybridPriorityModel.load(directory)
    if manifest["format"] == "pii-priority-embeddingbag-v1":
        from training.priority_embeddingbag import LowRankEmbeddingBagModel

        return LowRankEmbeddingBagModel.load(directory)
    if manifest["format"] == "pii-priority-fusion-v1":
        from training.priority_fusion import FusionPriorityModel

        return FusionPriorityModel.load(directory)
    raise ValueError(f"unsupported priority model format: {manifest['format']}")
