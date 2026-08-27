"""v5b -- distil the fine-tuned encoder into a static token embedding table.

`model2vec.distill` passes every vocabulary entry through the encoder, pools the
resulting sub-token vectors, applies PCA and Zipf weighting, and emits one vector
per vocabulary entry. What comes out is a lookup table: inference becomes an
embedding gather, which is what makes it servable inside a 5 ms / one-core budget
that a 68M transformer cannot approach.

## Token-based, not sentence-based

This is the point of the stage, so it is worth being exact about where the two
differ, because model2vec is token-based in one sense and sentence-based in
another and only the first is wanted here.

* **Distillation** is inherently token-level: the artifact IS a per-token table.
  The `pooling` argument below pools the sub-word pieces of a single vocabulary
  entry into that entry's vector — it does not pool a document.
* **`StaticModel.encode(text)`** then mean-pools those token vectors into ONE
  vector per document. That is the sentence-based use, and it is the wrong one
  here: averaged over a 12,000-character document, a single SSN contributes about
  a four-thousandth of the result and disappears.

So this stage exports the **matrix and the tokenizer**, and `v5_tagger` consumes
per-token vectors and does its own aggregation. `encode()` is never called on a
document in this chain. The check at the end of this module exists to keep that
honest: it measures whether identifier-bearing tokens remain distinguishable in
the table, which is the property the next stage depends on.

## What is recorded, and why each number is a decision

`vocab`, `dims` and `MB` are serving constraints, not trivia — the shipped cascade
is 28 MB in total, so a 200 MB embedding table would fail the deployment before it
was ever scored. PCA variance retained says how much of the encoder's geometry
survived the compression, which is the single number most likely to explain a
disappointing v5c.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROJECT = Path("projects/pii-content-v5")
#: Tokens that carry an identifier or cue one. If these collapse into each other
#: in the static table, the table cannot support a per-tag tagger and the chain
#: should stop here rather than at v5c.
PROBE_TOKENS = ["ssn", "social", "security", "passport", "visa", "iban", "mrn",
                "patient", "diagnosis", "password", "pin", "cvv", "routing",
                "account", "email", "phone", "address", "zip", "birth", "salary",
                "the", "and", "of", "report", "document", "page"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", type=Path,
                    default=PROJECT / "models/v5a-ettin-ft/doc_head",
                    help="the fine-tuned encoder to distil")
    ap.add_argument("--out", type=Path, default=PROJECT / "models/v5b-m2v")
    ap.add_argument("--pca-dims", type=int, default=256)
    ap.add_argument("--sif", type=float, default=1e-4)
    ap.add_argument("--pooling", default="mean",
                    help="how a vocabulary entry's sub-word pieces are pooled -- "
                         "NOT how a document is pooled; see the module docstring")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch
    from model2vec.distill import distill

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if not args.encoder.exists():
        raise SystemExit(f"{args.encoder} does not exist -- run v5_finetune.py first.")

    print(f"distilling {args.encoder} on {device}", flush=True)
    t0 = time.perf_counter()
    static = distill(model_name=str(args.encoder), pca_dims=args.pca_dims,
                     sif_coefficient=args.sif, pooling=args.pooling, device=device)
    elapsed = time.perf_counter() - t0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    static.save_pretrained(str(args.out))
    emb = np.asarray(static.embedding)
    mb = sum(f.stat().st_size for f in args.out.rglob("*") if f.is_file()) / 1e6

    # ---------------------------------------------------- does it still separate?
    norm = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)
    tokens, ids = [], []
    for tokenised in PROBE_TOKENS:
        got = static.tokenizer.encode(tokenised, add_special_tokens=False).ids
        if got:
            tokens.append(tokenised)
            ids.append(got[0])
    sim = norm[ids] @ norm[ids].T
    off = sim[~np.eye(len(ids), dtype=bool)]

    payload = {
        "encoder": str(args.encoder), "out": str(args.out),
        "vocab": int(emb.shape[0]), "dims": int(emb.shape[1]),
        "dtype": str(emb.dtype), "on_disk_mb": round(mb, 2),
        "pca_dims": args.pca_dims, "sif_coefficient": args.sif,
        "pooling": args.pooling, "distill_seconds": round(elapsed, 1),
        "probe": {"tokens": tokens,
                  "mean_offdiag_cosine": float(off.mean()),
                  "max_offdiag_cosine": float(off.max()),
                  "note": "near-1.0 mean cosine would mean the table has collapsed "
                          "and cannot support a per-tag tagger"},
    }
    (PROJECT / "probe").mkdir(parents=True, exist_ok=True)
    (PROJECT / "probe" / "v5b_distill.json").write_text(
        json.dumps(payload, indent=1), encoding="utf-8")

    print(f"vocab {payload['vocab']:,} x {payload['dims']} dims  "
          f"{payload['dtype']}  {payload['on_disk_mb']} MB  "
          f"({elapsed:.0f}s)", flush=True)
    print(f"probe-token cosine: mean {off.mean():.4f}  max {off.max():.4f} "
          f"over {len(ids)} tokens", flush=True)
    print(f"-> {args.out}\n-> {PROJECT / 'probe' / 'v5b_distill.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
