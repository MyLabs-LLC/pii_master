"""Make the search's own estimate honest: carve real-world corpora by source group.

The diagnosis, in one line: **the held-in estimate is inflated because the split
that produces it is the wrong shape for the data.**

`govdocs2` is organised into 991 source directories, and its train/eval split is
**grouped** -- 793 directories on the training side, 198 on the evaluation side,
**zero shared**. `carve_holdin`, however, splits the training rows *by document*.
So the held-in calibration slice draws from the same 793 directories the gate was
fitted on, and a directory's shared boilerplate, templates and site chrome leak
straight across the split.

The measured consequence, for arm B's gate:

    govdocs2   held-in AUC 0.9894  ->  sealed AUC 0.8434   (-0.146)
    datax      held-in AUC 0.8977  ->  sealed AUC 0.8504   (-0.047)

An AUC of 0.9894 on heterogeneous real-world web and office documents is not a
capability, it is a memorised directory. And because the search selects on that
number, it cannot see the failure: every one of the 250 docgate trials was scored
against an estimate inflated by the same leak, so the search reliably preferred
gates that memorise.

Regularisation does not fix this -- sweeping `alpha` across four orders of
magnitude moved the gap by 0.04 -- because it is not variance. It is the wrong
split.

This module carves whole **groups** to one side or the other for the grouped
corpora, leaving per-document carving for the synthetic corpora that have no
group structure. If the fix is right, the held-in estimate should fall to meet
the sealed one: an honest number, not a better one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.linear_model import SGDClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from training.h2h_gate_diag import _cut_for_specificity, _recall_at, is_real  # noqa: E402
from training.h2h_priority import PROJECT  # noqa: E402
from training.h2h_score import _load_cached  # noqa: E402
from training.quiet_data import canonical_stem, iter_quiet_corpus, resolve_dataset  # noqa: E402
from training.quiet_fit import Dataset, carve_holdin, load, train_corpora  # noqa: E402

#: Corpora whose documents cluster by a source directory that carries shared
#: boilerplate. Keyed by stem so a rename cannot detach them.
GROUPED_STEMS = frozenset({"govdocs2-dualjudge-train80", "datax-dualjudge-trainset"})


def group_ids(ds: Dataset) -> tuple[np.ndarray, dict[str, int]]:
    """A stable group id per training row, in the cache's own row order.

    Rows of a corpus with no group structure get their own id, so they are
    carved per document exactly as before -- the change is confined to the
    corpora that actually have groups.
    """
    ids = np.zeros(len(ds), dtype=np.int64)
    per_corpus: dict[str, int] = {}
    cursor = 0
    for ci, name in enumerate(ds.corpus_names):
        n = int((ds.corpus == ci).sum())
        grouped = canonical_stem(name) in GROUPED_STEMS
        if not grouped:
            # Unique per row: the existing per-document behaviour.
            ids[cursor:cursor + n] = np.arange(cursor, cursor + n) + 1_000_000_000
            per_corpus[name] = 0
            cursor += n
            continue
        seen: dict[str, int] = {}
        local = np.empty(n, dtype=np.int64)
        for i, qr in enumerate(iter_quiet_corpus(resolve_dataset(name))):
            parts = Path(qr.uid).parts
            key = "/".join(parts[:-1]) if len(parts) > 1 else qr.uid
            if key not in seen:
                seen[key] = len(seen)
            local[i] = seen[key]
        # Offset so two corpora's group 0 are not the same group.
        base = int.from_bytes(hashlib.blake2b(name.encode(), digest_size=4).digest(),
                              "little") % 1_000_000
        ids[cursor:cursor + n] = local + base * 1000
        per_corpus[name] = len(seen)
        cursor += n
    return ids, per_corpus


def carve_grouped(ds: Dataset, gids: np.ndarray, calib_frac: float = 0.15
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Whole groups to one side. Same 15% target, honest boundary."""
    h = (gids.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(11)
    bucket = (h % np.uint64(10_000)).astype(np.int64)
    calib = bucket < int(calib_frac * 10_000)
    return ~calib, calib


def _fit(X, y, w, alpha, loss, max_iter):
    clf = SGDClassifier(loss=loss, alpha=alpha, max_iter=max_iter, tol=None,
                        random_state=7)
    clf.fit(X, y.astype(np.int8), sample_weight=w)
    return clf.coef_.ravel().astype(np.float32), float(clf.intercept_[0])


PAIRS = (("govdocs2", "23693_govdocs2-dualjudge-train80-12.86k",
          "6589_govdocs2-dualjudge-eval20-3.53k"),
         ("datax", "15986_datax-dualjudge-trainset-5.36k",
          "4000_datax-dualjudge-evalset-1.32k"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=None, help="default: arm B's")
    ap.add_argument("--balance", default="none", choices=["none", "equal"])
    ap.add_argument("--spec-target", type=float, default=0.95)
    args = ap.parse_args()

    meta = json.loads((PROJECT / "models" / "cascade" / "model.json").read_text(
        encoding="utf-8"))["metadata"]
    gp = meta["gate_params"]
    alpha = args.alpha if args.alpha is not None else gp["alpha"]

    ds = load(train_corpora(), profile=meta["profile"])
    gids, per_corpus = group_ids(ds)
    print("source groups found per corpus:")
    for name, n in per_corpus.items():
        if n:
            print(f"    {name[:44]:<44} {n:>4} groups")

    known = ds.doc_target >= 0
    names = np.asarray(ds.corpus_names)
    real_row = np.asarray([is_real(n) for n in names])[ds.corpus]

    results: dict[str, Any] = {}
    for mode in ("per-document (current)", "grouped (fixed)"):
        fit_mask, calib_mask = (carve_holdin(ds) if mode.startswith("per-document")
                                else carve_grouped(ds, gids))
        rows = fit_mask & known
        y = ds.doc_target[rows].astype(bool)
        rf = real_row[rows]
        nr, nsy = int(rf.sum()), int((~rf).sum())
        w = np.where(y, 1.0, gp["neg_weight"])
        if args.balance == "equal":
            w = w * np.where(rf, nsy / max(nr, 1), 1.0)
        coef, b = _fit(ds.X[rows], y, w, alpha, gp["loss"], gp["max_iter"])

        print(f"\n=== held-in carve: {mode}   (alpha={alpha:.3g}, balance={args.balance})")
        print(f"    fit {int(rows.sum()):,}   calibration {int(calib_mask.sum()):,}")
        print(f"    {'corpus':<10} {'AUC held-in':>11} {'AUC sealed':>10} {'dAUC':>8} "
              f"{'R held-in':>9} {'R sealed':>9}")
        block = {}
        for label, tr, ev in PAIRS:
            m = calib_mask & known & (ds.corpus == list(names).index(tr))
            s_in = (ds.X[m] @ coef + b).astype(np.float32)
            y_in = ds.doc_target[m].astype(bool)
            d = _load_cached(ev, meta["profile"])
            msk = d["doc_target"] >= 0
            s_ev = (d["X"][msk] @ coef + b).astype(np.float32)
            y_ev = d["doc_target"][msk].astype(bool)
            a_in = roc_auc_score(y_in, s_in)
            a_ev = roc_auc_score(y_ev, s_ev)
            cut = _cut_for_specificity(s_in, y_in, args.spec_target)
            r_in, r_ev = _recall_at(s_in, y_in, cut), _recall_at(s_ev, y_ev, cut)
            print(f"    {label:<10} {a_in:>11.4f} {a_ev:>10.4f} {a_ev - a_in:>8.4f} "
                  f"{r_in:>9.4f} {r_ev:>9.4f}   (n_in={int(m.sum()):,})")
            block[label] = {"auc_held_in": a_in, "auc_sealed": a_ev,
                            "d_auc": a_ev - a_in, "recall_held_in": r_in,
                            "recall_sealed": r_ev, "n_held_in": int(m.sum())}
        results[mode] = block

    path = PROJECT / "probe" / "grouped_split.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"alpha": alpha, "balance": args.balance,
                                "spec_target": args.spec_target,
                                "groups": per_corpus, "results": results},
                               indent=1) + "\n", encoding="utf-8")
    print(f"\n-> {path}")
    print("\nthe fix works if the held-in AUC FALLS to meet the sealed one. A lower "
          "held-in number is the point: it is the honest one, and it is what lets a "
          "search stop preferring gates that memorise a directory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
