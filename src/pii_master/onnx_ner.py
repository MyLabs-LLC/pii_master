"""Stage 2 ONNX student detector. Optional extra: ``pip install pii-master[ml]``.

The default install stays stdlib-only. This module is imported only when
``scan_text(..., deep=True)`` (or ``pii-master scan --deep``) is used.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .bio import decode_spans
from .crosswalk import to_entity_type
from .entities import EntityType
from .fusion import merge_adjacent_same_type
from .models import Entity
from .validators import ipv4_ok, ipv6_ok, luhn_ok, ssn_ok

TEACHER_ID = "kalyan-ks/ettin-68m-nemotron-pii"
DEFAULT_MIN_CONFIDENCE = 0.40
# Identifier-like types: a 1–2 digit model span is a truncation fragment
# (docs/DISTILLATION_RESULTS.md frozen rehearsal), not an entity.
_ID_TYPES = frozenset({
    EntityType.MRN,
    EntityType.ACCOUNT_NUMBER,
    EntityType.HEALTH_PLAN_ID,
    EntityType.US_DRIVER_LICENSE,
    EntityType.PHONE_US,
    EntityType.SSN,
    EntityType.CREDIT_CARD,
})
MIN_ID_DIGITS = 3


def default_artifact_dir() -> Path:
    env = os.environ.get("PII_MASTER_ONNX_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "training" / "artifacts"


def revalidate(entity_type: EntityType, text: str) -> bool:
    """Drop model spans of checksummed types that fail the rule validator."""
    digits = "".join(ch for ch in text if ch.isdigit())
    if entity_type is EntityType.CREDIT_CARD:
        return 13 <= len(digits) <= 19 and luhn_ok(digits)
    if entity_type is EntityType.SSN:
        return len(digits) == 9 and ssn_ok(digits[:3], digits[3:5], digits[5:])
    if entity_type is EntityType.IP_ADDRESS:
        stripped = text.strip()
        return ipv4_ok(stripped) or ipv6_ok(stripped)
    if entity_type is EntityType.EMAIL:
        return "@" in text and "." in text.split("@")[-1]
    if entity_type is EntityType.URL:
        return "." in text or ":" in text
    return True


class OnnxNerDetector:
    """``Detector`` protocol: ``detect(text) -> list[Entity]``."""

    name = "onnx/student-m"

    def __init__(
        self,
        onnx_path: Path | None = None,
        meta_path: Path | None = None,
        *,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        max_length: int = 4096,
    ):
        try:
            import numpy as np
            import onnxruntime as ort
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "deep mode needs the [ml] extra: pip install 'pii-master[ml]'"
            ) from exc

        artifacts = default_artifact_dir()
        onnx_path = Path(onnx_path or artifacts / "student_m.onnx")
        meta_path = Path(meta_path or artifacts / "student_m.json")
        if not onnx_path.is_file():
            raise FileNotFoundError(
                f"student ONNX not found at {onnx_path}; train/export first "
                "or set PII_MASTER_ONNX_DIR"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self._id2label = {i: name for i, name in enumerate(meta["label_names"])}
        self._min_confidence = min_confidence
        self._max_length = max_length
        self._np = np
        self._tokenizer = AutoTokenizer.from_pretrained(TEACHER_ID)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(onnx_path), opts, providers=["CPUExecutionProvider"]
        )

    def detect(self, text: str) -> list[Entity]:
        if not text:
            return []
        np = self._np
        enc = self._tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=self._max_length,
            return_offsets_mapping=True,
            return_tensors="np",
        )
        logits = self._session.run(
            None,
            {
                "input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64),
            },
        )[0][0]
        # logits shape (seq, labels)
        pred = logits.argmax(-1)
        if self._min_confidence > 0:
            shifted = logits - logits.max(axis=-1, keepdims=True)
            exp = np.exp(shifted)
            prob = (exp / exp.sum(axis=-1, keepdims=True)).max(axis=-1)
        else:
            prob = None
        offsets = enc["offset_mapping"][0]
        raw = decode_spans(text, offsets, pred, self._id2label)
        entities: list[Entity] = []
        for label, start, end in raw:
            mapped = to_entity_type(label)
            if mapped is None:
                continue
            span_text = text[start:end]
            if mapped in _ID_TYPES:
                digits = sum(ch.isdigit() for ch in span_text)
                if digits < MIN_ID_DIGITS:
                    continue
            if not revalidate(mapped, span_text):
                continue
            confidence = 0.70
            if prob is not None:
                token_scores = [
                    float(prob[i])
                    for i, (a, b) in enumerate(offsets)
                    if int(b) > int(a) and int(a) < end and start < int(b)
                ]
                if token_scores:
                    confidence = sum(token_scores) / len(token_scores)
                if confidence < self._min_confidence:
                    continue
            entities.append(
                Entity(
                    type=mapped,
                    start=start,
                    end=end,
                    text=span_text,
                    confidence=float(confidence),
                    detector=self.name,
                )
            )
        return merge_adjacent_same_type(entities, text)
