"""v5c -- a document tagger over the static token embeddings from v5b.

This is the first artifact in the chain that could actually ship, and the stage
with a hard stop in it: if one-core p95 exceeds 5 ms the chain ends here whatever
the quality numbers say, because a model too slow for its SLA has failed
regardless of its score.

## Aggregation: mean AND max, not mean

`StaticModel.encode()` mean-pools a document into one vector. Over a 12,000-
character document that is the wrong operation for this task: a document is
labelled `sensitive_pii_social_security_number` because of ONE span in it, and an
average over ~3,000 tokens attenuates that span to roughly a three-thousandth of
the result. Mean pooling asks "what is this document about"; the tagging question
is "does this document contain, anywhere, a thing".

So each document becomes `[mean(d) ‖ max(d)]`:

* the **max** channel is the one that matters here — an element-wise maximum over
  token vectors preserves the strongest activation on any single token, which is
  exactly the rare-identifier signal the mean destroys;
* the **mean** channel keeps the topical context that tells a medical record from
  an invoice, which is what disambiguates a bare 9-digit number.

Both are recorded separately in the feature cache so a later run can ablate one
without recomputing the other.

## Held-out discipline

Features are built for every training row, but the **fit** side of
`quiet_fit.carve_holdin` trains the heads and the **calib** side selects the
thresholds — the same split the encoder was fine-tuned under in `v5_finetune`, so
no threshold is chosen on a row any part of this chain was fitted to. The sealed
corpora are scored once, by `v5_fuse`, and never opened here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.quiet_data import (  # noqa: E402
    EVAL_ROOT, TRAIN_ROOT, iter_quiet_corpus, list_dataset_dirs, read_document,
)
from training.v5_finetune import carve_fit_mask  # noqa: E402

PROJECT = Path("projects/pii-content-v5")
CATALOGUE = Path("projects/pii-scorecard-60/cache/catalogue.json")

#: 6,000, where the cascade reads 12,000 — and this is not a quality compromise.
#:
#: The tokenizer truncates at 1,024 tokens, and 6,000 characters already produces
#: 1,024. Measured on one core with the real static table:
#:
#:     12,000 chars -> 1,024 tokens, 6.946 ms p95
#:      8,000 chars -> 1,024 tokens, 4.265 ms
#:      6,000 chars -> 1,024 tokens, 2.533 ms
#:      4,000 chars ->   975 tokens, 1.699 ms
#:
#: Everything past ~6,000 characters is scanned by the tokenizer and then thrown
#: away, so reading 12,000 delivers the model no extra information and costs 2.7x
#: the time. At 12,000 the fused arm misses even a 10 ms budget (10.975 ms); at
#: 6,000 it lands at 6.562 ms.
#:
#: It also matches what the encoder was fine-tuned on: `v5_finetune` truncates at
#: `MAX_TOKENS=1024`, and truncation keeps the FIRST 1,024 tokens — the same ones
#: 6,000 characters yields. Training and serving see identical input.
READ_CHARS = 6_000


class TokenFeaturiser:
    """Static token table -> one [mean ‖ max] vector per document."""

    def __init__(self, model_dir: Path):
        from model2vec import StaticModel
        self.model = StaticModel.from_pretrained(str(model_dir))
        # A vocabulary-quantized model stores a `token_mapping` from token id to
        # row in `embedding`; indexing `embedding` by raw token id would then read
        # the wrong vectors and return a plausible, wrong answer. v5_distill does
        # not quantize, so this should never fire — which is exactly why it is
        # worth a check rather than a comment.
        if getattr(self.model, "token_mapping", None) is not None:
            raise SystemExit(
                "this static model was vocabulary-quantized: token ids must be "
                "remapped through `token_mapping` before indexing `embedding`. "
                "Either distil without quantization or add the remap here.")
        self.emb = np.asarray(self.model.embedding, dtype=np.float32)
        self.dims = self.emb.shape[1]

    def one(self, text: str) -> np.ndarray:
        ids = self.model.tokenizer.encode(text, add_special_tokens=False).ids
        if not ids:
            return np.zeros(self.dims * 2, dtype=np.float32)
        vecs = self.emb[ids]
        return np.concatenate([vecs.mean(axis=0), vecs.max(axis=0)])

    def many(self, texts: list[str]) -> np.ndarray:
        encs = self.model.tokenizer.encode_batch(texts, add_special_tokens=False)
        out = np.zeros((len(texts), self.dims * 2), dtype=np.float32)
        for i, enc in enumerate(encs):
            if not enc.ids:
                continue
            vecs = self.emb[enc.ids]
            out[i, :self.dims] = vecs.mean(axis=0)
            out[i, self.dims:] = vecs.max(axis=0)
        return out


def build_features(feat: TokenFeaturiser, root: Path, index: dict[str, int],
                   out: Path, batch: int = 256) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for d in list_dataset_dirs(root):
        dest = out / f"{d.name}.npz"
        if dest.exists():
            print(f"  {d.name:<48} cached", flush=True)
            continue
        t0 = time.perf_counter()
        rows = list(iter_quiet_corpus(d))
        X = np.zeros((len(rows), feat.dims * 2), dtype=np.float32)
        Y = np.zeros((len(rows), len(index)), dtype=bool)
        # The same ordinal-based carve `quiet_fit` uses, so this fit mask and the
        # cascade's are the same mask. See `carve_fit_mask`'s docstring for why
        # hashing the uid instead is silently wrong.
        fit = carve_fit_mask(d.name, len(rows))
        texts: list[str] = []
        at = 0
        for i, qr in enumerate(rows):
            try:
                texts.append(read_document(Path(qr.path), limit=READ_CHARS))
            except (FileNotFoundError, OSError):
                texts.append("")
            for t in qr.row.labels:
                j = index.get(t)
                if j is not None:
                    Y[i, j] = True
            if len(texts) >= batch:
                X[at:at + len(texts)] = feat.many(texts)
                at += len(texts)
                texts = []
        if texts:
            X[at:at + len(texts)] = feat.many(texts)
        # `uids` is what makes the join to the cascade's features checkable rather
        # than assumed: v5_fuse pairs the two side by side and a silent
        # misalignment pairs unrelated documents.
        np.savez_compressed(dest, X=X, Y=Y, fit=fit,
                            uids=np.array([qr.uid for qr in rows], dtype=np.str_))
        print(f"  {d.name:<48} {len(rows):>7,} docs  "
              f"{time.perf_counter() - t0:6.1f}s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2v", type=Path, default=PROJECT / "models/v5b-m2v")
    ap.add_argument("--features", type=Path, default=PROJECT / "cache/features")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--eval-too", action="store_true",
                    help="also featurise the sealed corpora (v5_fuse needs them)")
    ap.add_argument("--out", type=Path, default=PROJECT / "models/v5c-tagger")
    args = ap.parse_args()

    labels = tuple(json.loads(CATALOGUE.read_text())["labels"])
    index = {t: i for i, t in enumerate(labels)}
    feat = TokenFeaturiser(args.m2v)
    print(f"static table: {feat.emb.shape[0]:,} x {feat.dims}  "
          f"-> {feat.dims * 2}-d document features", flush=True)

    print("featurising training corpora ...", flush=True)
    build_features(feat, TRAIN_ROOT, index, args.features / "train")
    if args.eval_too:
        print("featurising sealed corpora ...", flush=True)
        build_features(feat, EVAL_ROOT, index, args.features / "eval")
    if args.build_only:
        return 0

    from sklearn.linear_model import LogisticRegression

    from training.h2h_thresholds_v4 import select_per_label

    Xs, Ys, fits, groups = [], [], [], []
    for g, f in enumerate(sorted((args.features / "train").glob("*.npz"))):
        with np.load(f) as z:
            Xs.append(z["X"]); Ys.append(z["Y"]); fits.append(z["fit"])
            groups.append(np.full(len(z["X"]), g, dtype=np.int32))
    X = np.concatenate(Xs); Y = np.concatenate(Ys)
    fit = np.concatenate(fits); group = np.concatenate(groups)
    del Xs, Ys, fits, groups
    print(f"features {X.shape}, fit {int(fit.sum()):,} / calib {int((~fit).sum()):,}",
          flush=True)

    mu = X[fit].mean(axis=0)
    sd = np.maximum(X[fit].std(axis=0), 1e-6)
    Xn = (X - mu) / sd

    W = np.zeros((len(labels), X.shape[1]), dtype=np.float32)
    b = np.zeros(len(labels), dtype=np.float32)
    t0 = time.perf_counter()
    for j, tag in enumerate(labels):
        y = Y[fit, j]
        if y.sum() < 24 or (~y).sum() < 24:
            W[j] = 0.0
            continue
        clf = LogisticRegression(max_iter=300, C=1.0, solver="liblinear",
                                 class_weight="balanced")
        clf.fit(Xn[fit], y)
        W[j] = clf.coef_.ravel()
        b[j] = float(clf.intercept_[0])
        if (j + 1) % 10 == 0:
            print(f"  fitted {j + 1}/{len(labels)} heads  "
                  f"{time.perf_counter() - t0:.0f}s", flush=True)

    S = (Xn @ W.T + b).astype(np.float32)
    thr, report = select_per_label(
        S[~fit], Y[~fit], np.ones(int((~fit).sum()), dtype=bool), group[~fit],
        beta=0.5, recall_floor=0.75, margin=0.0056, min_support=24,
        corrected_cap=True)
    enabled = int(np.isfinite(thr).sum())
    print(f"selected {enabled} of {len(labels)} thresholds", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "head.npz", W=W, b=b, thresholds=thr, mu=mu, sd=sd)
    (args.out / "model.json").write_text(json.dumps({
        "labels": list(labels), "m2v": str(args.m2v), "dims": feat.dims,
        "aggregation": "mean||max over static token vectors",
        "read_chars": READ_CHARS, "n_enabled": enabled,
        "threshold_rule": "h2h_thresholds_v4.select_per_label(corrected_cap=True)",
    }, indent=1), encoding="utf-8")
    (PROJECT / "probe").mkdir(parents=True, exist_ok=True)
    (PROJECT / "probe" / "v5c_thresholds.json").write_text(
        json.dumps({labels[r["label"]]: r for r in report}, indent=1), encoding="utf-8")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
