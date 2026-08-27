"""Real-world document recall is the binding problem. Two candidate mechanisms:

1. a longer read window -- govdocs2 documents average 149,000 characters and we
   are reading the first 1,000 of them;
2. sample weights that stop 460,000 synthetic documents from deciding what the
   gate learns about the 40,000 real ones.
"""
from __future__ import annotations
import sys, time
import numpy as np
sys.path.insert(0, "/home/lence/workspace/pii_master")
from sklearn.linear_model import SGDClassifier
from training.quiet_fit import carve_holdin, load, train_corpora
from training.quiet_select import select_doc_threshold, doc_metrics

REAL = ("16000_datax-dualjudge-trainset-5.37k", "26095_govdocs2-dualjudge-train80-14.25k")

for window in (1000, 4000):
    ds = load(train_corpora(), window=window)
    fit_mask, calib_mask = carve_holdin(ds)
    known = ds.doc_target >= 0
    fit, cal = fit_mask & known, calib_mask & known
    y = ds.doc_target.astype(np.int8)
    real_mask = np.zeros(len(ds), dtype=bool)
    for n in REAL:
        real_mask |= ds.corpus_mask(n)
    print(f"\n=== window {window} ===  (nnz {ds.X.nnz:,})")
    for real_w in (1.0, 5.0, 20.0):
        w = np.where(real_mask, real_w, 1.0)[fit]
        t0 = time.time()
        clf = SGDClassifier(loss="log_loss", alpha=1e-6, max_iter=12, tol=None,
                            random_state=7, n_jobs=-1)
        clf.fit(ds.X[fit], y[fit], sample_weight=w)
        s = clf.decision_function(ds.X)
        # Choose the cut on REAL calibration data: it is the only slice that
        # predicts the corpus the gate is actually for.
        rc, sc = cal & real_mask, cal & ~real_mask
        cut, _ = select_doc_threshold(s[rc], ds.doc_target[rc],
                                      recall_floor=0.88, specificity_floor=0.88)
        fired = s >= cut
        real = doc_metrics(fired[rc], ds.doc_target[rc])
        synth = doc_metrics(fired[sc], ds.doc_target[sc])
        print(f"  real_w={real_w:<5} [{time.time()-t0:5.1f}s]  "
              f"REAL P={real['precision']:.4f} R={real['recall']:.4f} sp={real['specificity']:.4f} | "
              f"SYNTH P={synth['precision']:.4f} R={synth['recall']:.4f} sp={synth['specificity']:.4f}")
