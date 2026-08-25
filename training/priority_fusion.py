"""Per-label boolean fusion runtime for priority document taggers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.priority_hash import document_features


def fuse_strategy(strategy: str, predicted: dict[str, set[str]], label: str) -> bool:
    operation, raw_names = strategy.split(":", 1)
    names = raw_names.split(",")
    votes = [label in predicted[name] for name in names]
    if operation == "source":
        if len(votes) != 1:
            raise ValueError("source strategy needs exactly one component")
        return votes[0]
    if operation == "or":
        return any(votes)
    if operation == "and":
        return all(votes)
    if operation == "majority":
        return sum(votes) >= (len(votes) // 2 + 1)
    raise ValueError(f"unknown fusion operation: {operation}")


@dataclass(frozen=True)
class FusionPriorityModel:
    labels: tuple[str, ...]
    components: dict[str, Any]
    strategies: dict[str, str]

    def __post_init__(self) -> None:
        if set(self.strategies) != set(self.labels):
            raise ValueError("every label needs exactly one fusion strategy")
        for model in self.components.values():
            if tuple(model.labels) != self.labels:
                raise ValueError("fusion component catalogues differ")

    @property
    def read_window_chars(self) -> int:
        return max(model.read_window_chars for model in self.components.values())

    @property
    def score_mode(self) -> str:
        return "per_label_boolean_fusion"

    def predict(self, text: str) -> list[str]:
        first = next(iter(self.components.values()))
        n_features = (
            first.n_features
            if hasattr(first, "n_features")
            else first.embeddings.shape[0]
        )
        features = document_features(
            text[: self.read_window_chars],
            n_features=n_features,
            max_tokens=max(model.max_tokens for model in self.components.values()),
            max_features=max(
                model.max_document_features for model in self.components.values()
            ),
        )
        predicted = {
            name: set(model.predict_from_features(features))
            for name, model in self.components.items()
        }
        return [
            label
            for label in self.labels
            if fuse_strategy(self.strategies[label], predicted, label)
        ]

    def save(self, directory: Path, metadata: dict[str, Any] | None = None) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        component_paths: dict[str, str] = {}
        for name, model in self.components.items():
            relative = f"component_{name}"
            model.save(directory / relative)
            component_paths[name] = relative
        manifest = {
            "format": "pii-priority-fusion-v1",
            "labels": list(self.labels),
            "components": component_paths,
            "strategies": self.strategies,
            "metadata": metadata or {},
        }
        (directory / "model.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> FusionPriorityModel:
        from training.priority_hash import load_priority_model

        manifest = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        components = {
            name: load_priority_model(directory / relative)
            for name, relative in manifest["components"].items()
        }
        return cls(
            labels=tuple(manifest["labels"]),
            components=components,
            strategies={
                str(key): str(value) for key, value in manifest["strategies"].items()
            },
        )
