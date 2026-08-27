"""Does adding generated near-miss negatives move the sealed real-world number?

The honest test, and the one it is easy to fake.

Adding negatives to a binary problem shifts the boundary toward silence. So a
gate trained on 2,000 extra negatives will *always* look better on specificity
and worse on recall, and quoting either alone would be meaningless. Two
measurements avoid that:

* **ROC-AUC on the sealed real corpora** -- threshold-free, so it answers "did
  the model's ranking of real documents actually improve" rather than "did the
  boundary move". This is the headline.
* **Recall at matched specificity (0.95)** -- the same operating point on both
  gates, so the two recalls are comparable.

And one that separates success from its lookalike:

* **Fire rate on held-out generated documents.** 20% of the corpus is never
  trained on. If the gate stops firing on those but the sealed real number is
  flat, it learned the generator's prose style, not the concept -- a result that
  looks like progress on every training metric and is worth nothing. That
  distinction is the reason this module exists rather than a one-line refit.

The sealed corpora were never involved in generating, verifying or selecting
anything here, so they remain a clean measurement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.linear_model import SGDClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from training.h2h_gate_diag import _cut_for_specificity, _recall_at, is_real  # noqa: E402
from training.h2h_priority import PROJECT  # noqa: E402
from training.h2h_score import _load_cached  # noqa: E402
from training.priority_hash import document_features  # noqa: E402
from training.quiet_cache import PROFILES, load_catalogue  # noqa: E402
from training.quiet_fit import carve_holdin, load, train_corpora  # noqa: E402

SEALED_REAL = ("4000_datax-dualjudge-evalset-1.32k",
               "6589_govdocs2-dualjudge-eval20-3.53k")
SEALED_SYNTH = ("30000_pii2_eval_25.15k",)


def featurise(corpus: Path, records: list[dict[str, Any]], n_features: int
              ) -> sp.csr_matrix:
    """The generated documents in the same feature space as the `deep` profile."""
    window, max_tokens, max_feats = PROFILES["deep"]
    rows, indptr = [], [0]
    for rec in records:
        text = (corpus / rec["path"]).read_text(encoding="utf-8")
        idx = document_features(text[:window], n_features=n_features,
                                max_tokens=max_tokens, max_features=max_feats)
        rows.append(idx)
        indptr.append(indptr[-1] + len(idx))
    data = np.concatenate(rows) if rows else np.empty(0, np.int32)
    return sp.csr_matrix((np.ones(len(data), np.float32), data,
                          np.asarray(indptr, np.int64)),
                         shape=(len(records), n_features))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--held-frac", type=float, default=0.20)
    ap.add_argument("--weight", type=float, nargs="+", default=[0.0, 1.0, 4.0],
                    help="per-document weight on generated rows; 0.0 = baseline")
    ap.add_argument("--balance", default="none", choices=["none", "equal"])
    ap.add_argument("--spec-target", type=float, default=0.95)
    args = ap.parse_args()

    meta = json.loads((PROJECT / "models" / "cascade" / "model.json").read_text(
        encoding="utf-8"))["metadata"]
    gp = meta["gate_params"]
    alpha = args.alpha if args.alpha is not None else gp["alpha"]
    n_features = int(load_catalogue()["n_features"])

    recs = json.loads((args.corpus / "manifest_verified.json").read_text(encoding="utf-8"))
    print(f"generated corpus: {len(recs):,} verified documents", flush=True)
    Xg = featurise(args.corpus, recs, n_features)
    rng = np.random.default_rng(11)
    held = rng.random(len(recs)) < args.held_frac
    print(f"    {int((~held).sum()):,} added to training, {int(held.sum()):,} held out",
          flush=True)

    ds = load(train_corpora(), profile="deep")
    fit_mask, _ = carve_holdin(ds)
    known = ds.doc_target >= 0
    names = np.asarray(ds.corpus_names)
    real_row = np.asarray([is_real(n) for n in names])[ds.corpus]
    rows = fit_mask & known
    y_base = ds.doc_target[rows].astype(bool)
    X_base = ds.X[rows]
    rf = real_row[rows]
    nr, nsy = int(rf.sum()), int((~rf).sum())

    sealed = {}
    for corpus in SEALED_REAL + SEALED_SYNTH:
        d = _load_cached(corpus, "deep")
        m = d["doc_target"] >= 0
        sealed[corpus] = (d["X"][m], d["doc_target"][m].astype(bool))

    print(f"\nall recalls at specificity {args.spec_target:.2f}; AUC is threshold-free")
    hdr = (f"{'gen weight':>10} | {'AUC real':>9} {'AUC datax':>10} {'AUC govdocs2':>13} "
           f"{'AUC synth':>10} | {'R real':>7} | {'held-out gen fire':>18}")
    print(hdr); print("-" * len(hdr))
    out: list[dict[str, Any]] = []
    for wgen in args.weight:
        if wgen > 0:
            X = sp.vstack([X_base, Xg[~held]], format="csr")
            y = np.concatenate([y_base, np.zeros(int((~held).sum()), bool)])
            w = np.where(y_base, 1.0, gp["neg_weight"])
            if args.balance == "equal":
                w = w * np.where(rf, nsy / max(nr, 1), 1.0)
            w = np.concatenate([w, np.full(int((~held).sum()), wgen * gp["neg_weight"])])
        else:
            X, y = X_base, y_base
            w = np.where(y_base, 1.0, gp["neg_weight"])
            if args.balance == "equal":
                w = w * np.where(rf, nsy / max(nr, 1), 1.0)

        clf = SGDClassifier(loss=gp["loss"], alpha=alpha, max_iter=gp["max_iter"],
                            tol=None, random_state=7)
        clf.fit(X, y.astype(np.int8), sample_weight=w)
        c, b = clf.coef_.ravel().astype(np.float32), float(clf.intercept_[0])

        def sc(M):
            return (M @ c + b).astype(np.float32)

        aucs = {k: roc_auc_score(yy, sc(XX)) for k, (XX, yy) in sealed.items()}
        Xr = sp.vstack([sealed[k][0] for k in SEALED_REAL], format="csr")
        yr = np.concatenate([sealed[k][1] for k in SEALED_REAL])
        s_r = sc(Xr)
        auc_real = roc_auc_score(yr, s_r)
        cut = _cut_for_specificity(s_r, yr, args.spec_target)
        r_real = _recall_at(s_r, yr, cut)
        fire = float((sc(Xg[held]) >= cut).mean()) if held.any() else float("nan")

        print(f"{wgen:>10.1f} | {auc_real:>9.4f} {aucs[SEALED_REAL[0]]:>10.4f} "
              f"{aucs[SEALED_REAL[1]]:>13.4f} {aucs[SEALED_SYNTH[0]]:>10.4f} | "
              f"{r_real:>7.4f} | {fire:>17.1%}", flush=True)
        out.append({"gen_weight": wgen, "alpha": alpha, "balance": args.balance,
                    "auc_sealed_real": auc_real,
                    "auc_per_corpus": {k: float(v) for k, v in aucs.items()},
                    "recall_sealed_real": r_real, "held_out_gen_fire_rate": fire,
                    "n_generated_in_fit": int((~held).sum())})

    path = PROJECT / "probe" / "gate_augment.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"\n-> {path}")
    print("\nreading it: 'AUC real' rising is the result. 'held-out gen fire' falling "
          "while 'AUC real' stays flat means the gate learned the generator's style, "
          "not the concept — no gain, and it would look like one on any training metric.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
