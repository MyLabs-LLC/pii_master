"""Can a dedicated document gate reach the specificity the policy demands?

Scored separately on synthetic and real-world negatives, because those are
different questions and only the real one predicts the customer's corpus.
"""
from __future__ import annotations
import sys, time
import numpy as np
sys.path.insert(0, "/home/lence/workspace/pii_master")
from sklearn.linear_model import SGDClassifier
from training.quiet_fit import carve_holdin, load, train_corpora
from training.quiet_select import select_doc_threshold, doc_metrics

REAL = ("16000_datax-dualjudge-trainset-5.37k", "26095_govdocs2-dualjudge-train80-14.25k")

ds = load(train_corpora(), window=1000)
fit_mask, calib_mask = carve_holdin(ds)
known = ds.doc_target >= 0
fit = fit_mask & known
cal = calib_mask & known
y = ds.doc_target.astype(np.int8)
print(f"fit {fit.sum():,} (pos {int((y[fit]==1).sum()):,} / neg {int((y[fit]==0).sum()):,})")

real_mask = np.zeros(len(ds), dtype=bool)
for n in REAL:
    real_mask |= ds.corpus_mask(n)

for cw in (None, "balanced", {0: 6.0, 1: 1.0}):
    for alpha in (1e-6, 1e-5):
        t0 = time.time()
        clf = SGDClassifier(loss="log_loss", alpha=alpha, class_weight=cw,
                            max_iter=12, tol=None, random_state=7, n_jobs=-1)
        clf.fit(ds.X[fit], y[fit])
        s = clf.decision_function(ds.X)
        cut, _ = select_doc_threshold(s[cal], ds.doc_target[cal],
                                      recall_floor=0.90, specificity_floor=0.85)
        fired = s >= cut
        overall = doc_metrics(fired[cal], ds.doc_target[cal])
        rc = cal & real_mask
        real = doc_metrics(fired[rc], ds.doc_target[rc])
        sc = cal & ~real_mask
        synth = doc_metrics(fired[sc], ds.doc_target[sc])
        cwn = "none" if cw is None else ("balanced" if cw == "balanced" else "6:1")
        print(f"cw={cwn:<9} a={alpha:<7} [{time.time()-t0:5.1f}s]  "
              f"ALL P={overall['precision']:.4f} R={overall['recall']:.4f} sp={overall['specificity']:.4f} | "
              f"REAL P={real['precision']:.4f} R={real['recall']:.4f} sp={real['specificity']:.4f} (n={real['n']:,}) | "
              f"SYNTH sp={synth['specificity']:.4f}")
