"""End-to-end smoke test of the fitting path, on held-in data only."""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/lence/workspace/pii_master")
from training.quiet_fit import (Dataset, accumulate, build_weights, carve_holdin,
                                load, priority_indices, score, train_corpora)
from training.quiet_select import doc_metrics, select_doc_threshold, select_per_label

t0 = time.time()
ds = load(train_corpora(), window=1000)
print(f"loaded {len(ds):,} rows, {ds.X.nnz:,} nnz, {len(ds.labels)} labels  [{time.time()-t0:.1f}s]")
fit_mask, calib_mask = carve_holdin(ds)
print(f"fit {fit_mask.sum():,}  calib {calib_mask.sum():,}")
for name, m in (("fit", fit_mask), ("calib", calib_mask)):
    t = ds.doc_target[m]
    print(f"  {name}: doc+ {int((t==1).sum()):,}  doc- {int((t==0).sum()):,}  unknown {int((t==-1).sum()):,}")

t0 = time.time()
counts = accumulate(ds, fit_mask)
print(f"counts accumulated [{time.time()-t0:.1f}s]  n_all={counts.n_all:,} n_complete={counts.n_complete:,}")
counts.save(Path("/home/lence/workspace/pii_master/projects/pii-quiet-alarm/cache/counts_w1000.npz"))

t0 = time.time()
W = build_weights(counts, alpha=1.0, partial_weight=0.5, min_df=3, clip=8.0, idf_power=0.0)
print(f"weights built {W.shape} [{time.time()-t0:.1f}s]")

calib = ds.subset(calib_mask)
t0 = time.time()
S = score(calib.X, W, mode="sum")
print(f"scored calib {S.shape} [{time.time()-t0:.1f}s]")

Yd = np.asarray(calib.Y.todense()).astype(bool)
pri = priority_indices(calib.labels)

for floor in (0.0, 0.75):
    thr, rep = select_per_label(S, Yd, calib.tag_complete, beta=0.5, recall_floor=floor)
    ok = [r for r in rep if not r["disabled"]]
    prir = [rep[j] for j in pri if not rep[j]["disabled"]]
    print(f"\nrecall_floor={floor}: {len(ok)}/{len(rep)} labels enabled")
    print(f"  ALL      macroP={np.mean([r['precision'] for r in ok]):.4f} "
          f"macroR={np.mean([r['recall'] for r in ok]):.4f} "
          f"macroF0.5={np.mean([r['f'] for r in ok]):.4f}")
    print(f"  PRIORITY macroP={np.mean([r['precision'] for r in prir]):.4f} "
          f"macroR={np.mean([r['recall'] for r in prir]):.4f} "
          f"macroF0.5={np.mean([r['f'] for r in prir]):.4f} "
          f"minR={min(r['recall'] for r in prir):.4f} "
          f"floors_met={sum(r['floor_met'] for r in prir)}/{len(prir)}")
    fired = (S >= thr).any(axis=1)
    dm = doc_metrics(fired, calib.doc_target)
    print(f"  DOC      P={dm['precision']:.4f} R={dm['recall']:.4f} spec={dm['specificity']:.4f} "
          f"(n={dm['n']:,}, prev={dm['prevalence']:.3f})")
print(f"\ntotal {time.time()-t0:.1f}s")
