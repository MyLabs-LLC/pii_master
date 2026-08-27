"""Closed-form checks on the evaluator, run before it produces any real number.

The evaluator is the instrument: if it is wrong, every table in the report is
wrong in the same direction and nothing downstream will notice. So it is checked
against values worked out by hand on a ten-document fixture, chosen so precision
and recall differ (0.5 against 0.6) -- with P == R every F-beta collapses to the
same number and a beta bug is invisible.

The three properties that matter most are the ones that would flatter a model:

* every precision-bearing metric is `None` on positive-only gold, never `0.0`;
* a corpus with no document-level negatives emits no document scope at all,
  rather than a perfect one;
* a priority tag below `min_support` is scoped with `None` and its support, so
  "could not measure" stays distinct from "passed".
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.h2h_eval import evaluate_corpus, fbeta  # noqa: E402

TOL = 1e-6
LABELS = ("sensitive_pii_social_security_number", "sensitive_pii_email", "sensitive_pii_age")


def fixture():
    """Ten documents. tag0: gold 0-4, fires 0,1,2 (TP) and 5,6,7 (FP) -> P .5, R .6."""
    Y = np.zeros((10, 3), bool)
    Y[0:5, 0] = True
    Y[0:2, 1] = True
    fired = np.zeros((10, 3), bool)
    fired[[0, 1, 2, 5, 6, 7], 0] = True
    fired[0, 1] = True
    doc_target = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], np.int8)
    return Y, fired, doc_target


def main() -> int:
    Y, fired, doc_t = fixture()
    complete = np.ones(10, bool)
    r = evaluate_corpus("30000_pii2_eval_25.15k", fired, fired.any(axis=1), Y,
                        complete, doc_t, LABELS, seed=1)

    t0, t1, t2 = (r["per_tag"][x] for x in LABELS)
    assert (t0["tp"], t0["fp"], t0["fn"], t0["support"]) == (3, 3, 2, 5), t0
    for beta, name in ((0.5, "f05"), (1.0, "f1"), (2.0, "f2"), (3.0, "f3")):
        want = float(fbeta(np.array([0.5]), np.array([0.6]), beta)[0])
        assert abs(t0[name] - want) < TOL, (name, t0[name], want)
    assert t0["f05"] < t0["f1"] < t0["f2"] < t0["f3"], "beta ordering wrong when R > P"
    print(f"per-tag F0.5/F1/F2/F3 = {t0['f05']:.6f}/{t0['f1']:.6f}/"
          f"{t0['f2']:.6f}/{t0['f3']:.6f}  match closed form, ordered  OK")

    assert (t1["tp"], t1["fp"], t1["fn"]) == (1, 0, 1), t1
    assert t2["support"] == 0 and t2["recall"] is None
    s = r["summary"]
    assert s["n_tags_catalogue"] == 2, "a tag with no gold must not join the macro"
    assert abs(s["f2_macro_catalogue"] - (t0["f2"] + t1["f2"]) / 2) < TOL
    assert abs(s["precision_micro"] - 4 / 7) < TOL
    assert abs(s["recall_micro"] - 4 / 7) < TOL
    print(f"macro F2 {s['f2_macro_catalogue']:.6f} = mean over gold-bearing tags; "
          f"micro P/R {s['precision_micro']:.4f}/{s['recall_micro']:.4f} = pooled 4/7  OK")

    d = r["scopes"]["doc@30000_pii2_eval_25.15k"]
    assert abs(d["doc_precision"]["value"] - 0.5) < TOL
    assert abs(d["doc_recall"]["value"] - 0.6) < TOL
    assert abs(d["doc_specificity"]["value"] - 0.4) < TOL
    for key, value in d.items():
        assert value["ci_low"] <= value["value"] <= value["ci_high"], key
    lo, hi = s["f2_macro_catalogue_ci"]
    assert lo <= s["f2_macro_catalogue"] <= hi
    print("document P/R/spec = 0.5/0.6/0.4 (TP3 FP3 FN2 TN2); every CI brackets "
          "its point estimate  OK")

    assert r["scopes"][f"{LABELS[0]}@30000_pii2_eval_25.15k"]["recall"]["value"] is None
    print("priority tag at support 5 < 30: scoped with value None and its support  OK")

    # ------------------------------------------------- positive-only tag gold
    r2 = evaluate_corpus("6589_govdocs2-dualjudge-eval20-3.53k", fired, fired.any(axis=1),
                         Y, np.zeros(10, bool), doc_t, LABELS, seed=2)
    s2 = r2["summary"]
    for key in ("precision_micro", "f1_micro", "f2_macro_catalogue", "f05_macro_catalogue",
                "f3_macro_catalogue", "precision_macro_catalogue", "priority_macro_f05"):
        assert s2[key] is None, (key, s2[key])
    assert s2["recall_macro_catalogue"] is not None
    assert r2["per_tag"][LABELS[0]]["precision"] is None
    assert abs(r2["per_tag"][LABELS[0]]["recall"] - 0.6) < TOL
    assert "doc@6589_govdocs2-dualjudge-eval20-3.53k" in r2["scopes"]
    print("positive-only gold: precision-bearing metrics None (not 0.0); recall "
          "still measured; document scope still present  OK")

    # ------------------------------------------------------- no doc negatives
    r3 = evaluate_corpus("20000_pii_holdout_20.00k", fired, fired.any(axis=1), Y,
                         complete, np.ones(10, np.int8), LABELS, seed=3)
    assert not any(k.startswith("doc@") for k in r3["scopes"])
    print("prevalence-1.0 corpus emits no document scope (absence, not a free pass)  OK")

    print("\nALL EVALUATOR CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
