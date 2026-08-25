"""Sufficient statistics once; every trial after that is linear algebra.

The budget is 1,000 trials over 531,431 training documents. Refitting a
discriminative model per trial per label would spend the whole wall-clock on
57 one-vs-rest solvers, so the work is split by what actually varies:

* **Counted once** -- the label x feature co-occurrence matrix, obtained as a
  single sparse product ``Y.T @ X`` rather than a Python loop over 96 million
  non-zeros. Every count-based trial then differs only in how those counts are
  turned into weights, which is a handful of array operations over a
  58 x 262,144 float matrix.
* **Fitted per trial** -- the document gate, and only the document gate. It is
  one binary target rather than 58, it is the mechanism the whole run exists to
  add, and it is cheap enough to fit properly.

The held-in / sealed discipline is enforced here rather than by convention:
:func:`carve_holdin` splits the *training* corpora by a hash of the document's
identity, so a trial selects on data it did not fit, and the eight evaluation
directories are never loaded by anything in this module.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.quiet_cache import CACHE_ROOT, load_catalogue  # noqa: E402
from training.quiet_data import EVAL_ROOT, PRIORITY_TAGS, TRAIN_ROOT, list_dataset_dirs  # noqa: E402


@dataclass
class Dataset:
    """A cached corpus (or several) as sparse matrices plus what gold can say."""

    X: sp.csr_matrix          # documents x hashed features, binary
    Y: sp.csr_matrix          # documents x labels, binary
    doc_target: np.ndarray    # 1 positive / 0 negative / -1 unknown
    tag_complete: np.ndarray  # may this row contribute a per-tag negative
    corpus: np.ndarray        # index into `corpus_names`
    corpus_names: tuple[str, ...]
    uid_hash: np.ndarray      # stable per-document hash, for splitting
    labels: tuple[str, ...]

    def __len__(self) -> int:
        return self.X.shape[0]

    def subset(self, mask: np.ndarray) -> Dataset:
        return Dataset(
            X=self.X[mask], Y=self.Y[mask], doc_target=self.doc_target[mask],
            tag_complete=self.tag_complete[mask], corpus=self.corpus[mask],
            corpus_names=self.corpus_names, uid_hash=self.uid_hash[mask],
            labels=self.labels,
        )

    def corpus_mask(self, name: str) -> np.ndarray:
        return self.corpus == self.corpus_names.index(name)


def _csr_from_packed(indptr: np.ndarray, indices: np.ndarray, n_cols: int) -> sp.csr_matrix:
    data = np.ones(len(indices), dtype=np.float32)
    return sp.csr_matrix((data, indices, indptr), shape=(len(indptr) - 1, n_cols))


def load(names: Iterable[str], profile: str = "std") -> Dataset:
    cat = load_catalogue()
    labels = tuple(cat["labels"])
    n_features, n_labels = int(cat["n_features"]), len(labels)
    Xs, Ys, tgt, comp, corp, uids = [], [], [], [], [], []
    names = list(names)
    for ci, name in enumerate(names):
        with np.load(CACHE_ROOT / f"{name}.npz", allow_pickle=False) as z:
            Xs.append(_csr_from_packed(z[f"indptr_{profile}"], z[f"indices_{profile}"], n_features))
            Ys.append(_csr_from_packed(z["label_indptr"], z["label_cols"], n_labels))
            tgt.append(z["doc_target"])
            comp.append(z["tag_complete"])
            n = len(z["doc_target"])
            corp.append(np.full(n, ci, dtype=np.int16))
            # A stable identity for splitting: corpus name plus row ordinal.
            seed = hashlib.blake2b(name.encode(), digest_size=8).digest()
            base = int.from_bytes(seed, "little")
            uids.append(((np.arange(n, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15))
                         ^ np.uint64(base)))
    return Dataset(
        X=sp.vstack(Xs, format="csr"), Y=sp.vstack(Ys, format="csr"),
        doc_target=np.concatenate(tgt), tag_complete=np.concatenate(comp),
        corpus=np.concatenate(corp), corpus_names=tuple(names),
        uid_hash=np.concatenate(uids), labels=labels,
    )


def train_corpora() -> list[str]:
    return [d.name for d in list_dataset_dirs(TRAIN_ROOT)]


def eval_corpora() -> list[str]:
    return [d.name for d in list_dataset_dirs(EVAL_ROOT)]


def carve_holdin(ds: Dataset, calib_frac: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """Split training into fit / calibration by a stable document hash.

    Selection happens on `calib`; weights are fitted on `fit`. Both come from
    the training corpora only -- the sealed evaluation directories are not
    reachable from this module.
    """
    bucket = (ds.uid_hash % np.uint64(10_000)).astype(np.int64)
    calib = bucket < int(calib_frac * 10_000)
    return ~calib, calib


# --------------------------------------------------------------- counted once
@dataclass
class Counts:
    """Label x feature co-occurrence, split by whether gold can supply negatives."""

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

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, labels=np.asarray(self.labels), n_features=np.asarray([self.n_features]),
            all_df=self.all_df, complete_df=self.complete_df,
            positive_complete_df=self.positive_complete_df,
            positive_partial_df=self.positive_partial_df,
            n_all=np.asarray([self.n_all]), n_complete=np.asarray([self.n_complete]),
            n_positive_complete=self.n_positive_complete,
            n_positive_partial=self.n_positive_partial,
        )

    @classmethod
    def load(cls, path: Path) -> Counts:
        with np.load(path, allow_pickle=False) as z:
            return cls(
                labels=tuple(map(str, z["labels"])), n_features=int(z["n_features"][0]),
                all_df=z["all_df"], complete_df=z["complete_df"],
                positive_complete_df=z["positive_complete_df"],
                positive_partial_df=z["positive_partial_df"],
                n_all=int(z["n_all"][0]), n_complete=int(z["n_complete"][0]),
                n_positive_complete=z["n_positive_complete"],
                n_positive_partial=z["n_positive_partial"],
            )


def accumulate(ds: Dataset, mask: np.ndarray) -> Counts:
    """One sparse product per statistic; no Python loop over non-zeros."""
    X, Y = ds.X[mask], ds.Y[mask]
    complete = ds.tag_complete[mask]
    Xc, Yc = X[complete], Y[complete]
    Xp, Yp = X[~complete], Y[~complete]
    to_dense = lambda M: np.asarray(M.todense(), dtype=np.float32)  # noqa: E731
    return Counts(
        labels=ds.labels,
        n_features=X.shape[1],
        all_df=np.asarray(X.sum(axis=0)).ravel().astype(np.float32),
        complete_df=np.asarray(Xc.sum(axis=0)).ravel().astype(np.float32),
        positive_complete_df=to_dense(Yc.T @ Xc),
        positive_partial_df=to_dense(Yp.T @ Xp),
        n_all=int(X.shape[0]),
        n_complete=int(Xc.shape[0]),
        n_positive_complete=np.asarray(Yc.sum(axis=0)).ravel().astype(np.float32),
        n_positive_partial=np.asarray(Yp.sum(axis=0)).ravel().astype(np.float32),
    )


def build_weights(
    counts: Counts, *, alpha: float = 1.0, partial_weight: float = 0.5,
    min_df: int = 2, clip: float = 8.0, idf_power: float = 0.0,
) -> np.ndarray:
    """Bernoulli log-odds weights, optionally tempered by inverse document frequency.

    ``idf_power`` is the one addition over the prior lineage's builder: a
    feature that fires on nearly every document carries almost no evidence
    about *which* document is sensitive, and down-weighting it is the cheapest
    precision lever available to a counting model.
    """
    positive = counts.positive_complete_df.astype(np.float32, copy=True)
    positive += partial_weight * counts.positive_partial_df
    negative = np.broadcast_to(counts.complete_df[None, :], positive.shape).astype(np.float32)
    negative = np.maximum(negative - counts.positive_complete_df, 0.0)
    n_pos = counts.n_positive_complete + partial_weight * counts.n_positive_partial
    n_neg = np.maximum(counts.n_complete - counts.n_positive_complete, 0.0)
    p_pos = (positive + alpha) / (n_pos[:, None] + 2.0 * alpha)
    p_neg = (negative + alpha) / (n_neg[:, None] + 2.0 * alpha)
    np.clip(p_pos, 1e-6, 1 - 1e-6, out=p_pos)
    np.clip(p_neg, 1e-6, 1 - 1e-6, out=p_neg)
    W = np.log(p_pos / (1 - p_pos)) - np.log(p_neg / (1 - p_neg))
    if idf_power:
        df = np.maximum(counts.all_df, 1.0)
        idf = np.log(counts.n_all / df) ** idf_power
        W *= idf[None, :].astype(np.float32)
    W[:, counts.all_df < min_df] = 0.0
    np.clip(W, -clip, clip, out=W)
    return np.ascontiguousarray(W, dtype=np.float32)


# ------------------------------------------------------------------- scoring
def score(X: sp.csr_matrix, W: np.ndarray, mode: str = "sum") -> np.ndarray:
    """Document x label scores. `sum` is the true linear score."""
    if mode == "sum":
        return (X @ W.T).astype(np.float32)
    if mode == "mean":
        n = np.maximum(np.diff(X.indptr), 1)[:, None]
        return ((X @ W.T) / n).astype(np.float32)
    if mode.startswith("top"):
        k = int(mode[3:])
        out = np.empty((X.shape[0], W.shape[0]), dtype=np.float32)
        Wt = np.ascontiguousarray(W.T)
        for i in range(X.shape[0]):
            cols = X.indices[X.indptr[i]:X.indptr[i + 1]]
            if not len(cols):
                out[i] = 0.0
                continue
            vals = np.maximum(Wt[cols], 0.0)
            kk = min(k, vals.shape[0])
            out[i] = np.partition(vals, -kk, axis=0)[-kk:].mean(axis=0)
        return out
    raise ValueError(f"unknown score mode: {mode}")


def priority_indices(labels: tuple[str, ...]) -> np.ndarray:
    return np.asarray([labels.index(t) for t in PRIORITY_TAGS if t in labels], dtype=np.int64)


__all__ = [
    "Counts", "Dataset", "accumulate", "build_weights", "carve_holdin",
    "eval_corpora", "load", "priority_indices", "score", "train_corpora",
]
