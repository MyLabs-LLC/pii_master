"""Command-line and Python entry point for a packaged FUSED sensitive-data tagger.

    from tagger import Tagger
    t = Tagger()
    t.has_pii(text)     # -> bool   : does this document contain sensitive PII
    t.predict(text)     # -> [str]  : which tags

    python tagger.py --file report.docx.txt
    python tagger.py --text "SSN 123-45-6789"

Self-contained: it imports nothing from the project that trained it.

## What "fused" means here

Two independent models answer the same 61-label question and their answers are
combined **per tag**:

* the **cascade** — a hashed character/word n-gram linear model with a document
  gate in front of 61 per-tag heads;
* the **content tagger** — a linear model over static token embeddings distilled
  from a fine-tuned transformer, aggregated as `[mean ‖ max]` over the document's
  token vectors.

`fusion.json` records, for each tag, which of four rules applies:

| rule | fires when |
| --- | --- |
| `cascade` | the cascade fires (the content model is ignored for this tag) |
| `content` | the content tagger fires |
| `or` | either fires |
| `and` | both fire |

Those rules were chosen on a training calibration split, never on evaluation data.

**Read the model card before drawing conclusions from `and`.** In both shipped
fusions the content model was never selected on its own and `or` was never
selected at all, so its entire contribution is to *suppress* cascade firings.
This bundle is therefore the cascade plus a precision veto, not two detectors
pooling their finds.

## Cost

Two models means two feature extractions per document. The measured one-core p95
is in the model card and is roughly 55% above the cascade alone, almost all of it
tokenizer time for the content path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_BUNDLE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_BUNDLE_ROOT / "runtime"))

from quiet_runtime import QuietCascade  # noqa: E402


class Tagger:
    """The fused model: cascade + content tagger, combined per tag."""

    def __init__(self, bundle_dir: str | Path | None = None) -> None:
        root = Path(bundle_dir) if bundle_dir else _BUNDLE_ROOT
        self.cascade = QuietCascade.load(root / "models" / "cascade")
        fusion = json.loads((root / "models" / "fusion.json").read_text(encoding="utf-8"))
        self.picks = fusion["per_tag"]
        self.doc_rule = fusion["document"]
        self.labels = tuple(self.cascade.labels)

        head = np.load(root / "models" / "content" / "head.npz")
        self._W = head["W"].astype(np.float32)
        self._b = head["b"].astype(np.float32)
        self._thr = head["thresholds"].astype(np.float32)
        self._mu = head["mu"].astype(np.float32)
        self._sd = head["sd"].astype(np.float32)
        # TWO windows, and conflating them is a silent correctness bug. The
        # cascade was fitted and measured reading 12,000 characters; the content
        # arm reads 6,000, because its tokenizer truncates at 1,024 tokens and
        # 6,000 characters already produces 1,024 — reading further costs 2.7x the
        # time for no extra tokens. Feeding the cascade the content arm's shorter
        # window changes the model that was measured.
        self.read_window_chars = int(self.cascade.window)
        self._content_chars = int(fusion["content_read_chars"])

        from model2vec import StaticModel
        self._m2v = StaticModel.from_pretrained(str(root / "models" / "m2v"))
        if getattr(self._m2v, "token_mapping", None) is not None:
            raise RuntimeError(
                "this static table was vocabulary-quantized; token ids must be "
                "remapped through `token_mapping` before indexing the embedding "
                "matrix. This bundle's loader does not do that.")
        self._emb = np.asarray(self._m2v.embedding, dtype=np.float32)

    # ------------------------------------------------------------- content arm
    def _content_fire(self, text: str) -> np.ndarray:
        ids = self._m2v.tokenizer.encode(
            text[:self._content_chars], add_special_tokens=False).ids
        if not ids:
            return np.zeros(len(self.labels), dtype=bool)
        v = self._emb[ids]
        feat = np.concatenate([v.mean(axis=0), v.max(axis=0)])
        return (((feat - self._mu) / self._sd) @ self._W.T + self._b) >= self._thr

    @staticmethod
    def _apply(rule: str, casc: bool, cont: bool) -> bool:
        if rule == "cascade":
            return casc
        if rule == "content":
            return cont
        if rule == "or":
            return casc or cont
        return casc and cont

    # ------------------------------------------------------------------ public
    def has_pii(self, text: str) -> bool:
        casc = self.cascade.has_pii(text)
        if self.doc_rule == "cascade":
            return casc                      # the content arm is not consulted
        cont = bool(self._content_fire(text).any())
        return self._apply(self.doc_rule, casc, cont)

    def predict(self, text: str) -> list[str]:
        casc_tags = set(self.cascade.predict(text))
        cont = self._content_fire(text)
        out = []
        for j, tag in enumerate(self.labels):
            if self._apply(self.picks.get(tag, "cascade"),
                           tag in casc_tags, bool(cont[j])):
                out.append(tag)
        return out


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
               "read_window_chars": tagger.read_window_chars,
               "content_read_chars": tagger._content_chars}
    if not args.gate_only:
        payload["labels"] = tagger.predict(text)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
