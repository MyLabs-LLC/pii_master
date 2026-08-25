"""Turn a winning cascade trial back into a servable artifact.

A trial records *which* configuration won, not the weights it won with -- the
search keeps only calibration scores, because keeping 1,000 weight matrices
would cost more disk than the search costs compute. So the winner is refitted
here, deterministically, from the same fit split and the same seeds, and the
result is checked against the trial's recorded calibration numbers before it is
allowed out. A refit that does not reproduce its own trial is a bug in this
module, and it fails rather than shipping.

The model is fitted on the **fit** split only and its thresholds are chosen on
the **calibration** split, exactly as during search. Refitting on both would
invalidate every threshold that made the winner win.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import SGDClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.quiet_cache import PROFILES, load_catalogue  # noqa: E402
from training.quiet_fit import (  # noqa: E402
    accumulate, build_weights, carve_holdin, load, priority_indices, score, train_corpora,
)
from training.quiet_model import QuietCascade  # noqa: E402
from training.quiet_objective import evaluate, group_masks  # noqa: E402
from training.quiet_select import select_per_label_robust  # noqa: E402

PROJECT = Path(os.environ.get(
    "QUIET_PROJECT", "/home/lence/workspace/pii_master/projects/pii-quiet-alarm"))
TUNING = PROJECT / "tuning"
HEAD_JOBS = 12
TOLERANCE = 0.02


def _best(family: str) -> list[dict]:
    return json.loads((TUNING / family / "best.json").read_text(encoding="utf-8"))


def fit_gate(ds, fit_mask, params) -> tuple[np.ndarray, float]:
    known = ds.doc_target >= 0
    rows = fit_mask & known
    y = ds.doc_target[rows].astype(np.int8)
    w = np.where(y == 0, params["neg_weight"], 1.0)
    clf = SGDClassifier(loss=params["loss"], alpha=params["alpha"],
                        max_iter=params["max_iter"], tol=None, random_state=7)
    clf.fit(ds.X[rows], y, sample_weight=w)
    return clf.coef_.ravel().astype(np.float32), float(clf.intercept_[0])


def fit_disc_heads(ds, fit_mask, params) -> np.ndarray:
    X, Y = ds.X[fit_mask], ds.Y[fit_mask]
    complete = ds.tag_complete[fit_mask]
    Yd = np.asarray(Y.todense()).astype(bool)
    n_features = X.shape[1]

    def one(j: int) -> np.ndarray:
        pos = Yd[:, j]
        eligible = pos | complete
        if int(pos.sum()) < 20 or not (~pos & eligible).any():
            return np.zeros(n_features, dtype=np.float32)
        w = np.where(eligible, np.where(pos, params["pos_weight"], 1.0), 0.0)
        clf = SGDClassifier(loss="log_loss", alpha=params["alpha"],
                            max_iter=params["max_iter"], tol=None, random_state=11)
        clf.fit(X, pos.astype(np.int8), sample_weight=w)
        return clf.coef_.ravel().astype(np.float32)

    rows = Parallel(n_jobs=HEAD_JOBS, backend="loky", batch_size=1)(
        delayed(one)(j) for j in range(Yd.shape[1]))
    return np.stack(rows, axis=0)


def materialize(cascade_trial: dict, out_dir: Path) -> tuple[QuietCascade, dict[str, Any]]:
    gates, heads = _best("docgate"), _best("tagcount") + _best("tagdisc")
    gate_cfg = next(g for g in gates if g["number"] == cascade_trial["extra"]["gate_source"])
    head_cfg = next(h for h in heads
                    if h["number"] == cascade_trial["extra"]["head_source"]
                    and h["family"] == cascade_trial["extra"]["head_family"])
    profile = cascade_trial["extra"]["profile"]
    window, max_tokens, max_features = PROFILES[profile]

    ds = load(train_corpora(), profile=profile)
    fit_mask, calib_mask = carve_holdin(ds)
    calib = ds.subset(calib_mask)

    gate_w, gate_b = fit_gate(ds, fit_mask, gate_cfg["params"])
    hp = head_cfg["params"]
    if head_cfg["family"] == "tagcount":
        counts = accumulate(ds, fit_mask)
        W = build_weights(counts, alpha=hp["alpha"], partial_weight=hp["partial_weight"],
                          min_df=hp["min_df"], clip=hp["clip"], idf_power=hp["idf_power"])
        score_mode = hp["score_mode"]
    else:
        W = fit_disc_heads(ds, fit_mask, hp)
        score_mode = "sum"

    # Re-derive the operating points exactly as the trial did: gate first, then
    # tag cuts on the documents the gate lets through.
    g_cal = (calib.X @ gate_w + gate_b).astype(np.float32)
    cut = float(cascade_trial["extra"]["gate_threshold"])
    S_cal = score(calib.X, W, mode=score_mode)
    Ycal = np.asarray(calib.Y.todense()).astype(bool)
    open_doc = g_cal >= cut
    # Must be the same selection rule the trial used, including its margin and
    # its source grouping. Using the pooled rule here produced thresholds that
    # scored 0.9304 against the trial's recorded 0.8244 -- a "better" model that
    # was simply not the one the search selected, and the drift check below is
    # what caught it.
    thr, _ = select_per_label_robust(
        S_cal[open_doc], Ycal[open_doc], calib.tag_complete[open_doc],
        calib.corpus[open_doc],
        beta=0.5, recall_floor=cascade_trial["params"]["recall_floor"],
        margin=cascade_trial["params"].get("margin", 0.0),
        min_support=cascade_trial["params"]["min_support_fit"])

    reproduced = evaluate(S_cal, thr, Ycal, calib.tag_complete, calib.doc_target,
                          group_masks(calib.corpus, calib.corpus_names),
                          priority_indices(calib.labels), gate=g_cal, gate_threshold=cut)

    recorded = cascade_trial["metrics"]["priority_macro_f05"]
    drift = abs(reproduced.priority_macro_f05 - recorded)
    if drift > TOLERANCE:
        raise SystemExit(
            f"refit did not reproduce its own trial: priority macro F0.5 "
            f"{reproduced.priority_macro_f05:.4f} vs recorded {recorded:.4f} "
            f"(drift {drift:.4f} > {TOLERANCE})")

    model = QuietCascade(
        labels=tuple(ds.labels), gate_weights=gate_w, gate_intercept=gate_b,
        gate_threshold=cut, tag_weights=W.astype(np.float32), tag_thresholds=thr,
        score_mode=score_mode, window=window, max_tokens=max_tokens,
        max_features=max_features, n_features=int(load_catalogue()["n_features"]),
    )
    provenance = {
        "cascade_trial": cascade_trial["number"],
        "gate_trial": gate_cfg["number"], "gate_params": gate_cfg["params"],
        "head_family": head_cfg["family"], "head_trial": head_cfg["number"],
        "head_params": hp, "profile": profile,
        "calibration_reproduced": reproduced.as_metrics(),
        "calibration_recorded": cascade_trial["metrics"],
        "reproduction_drift": drift,
        "fit_rows": int(fit_mask.sum()), "calibration_rows": int(calib_mask.sum()),
    }
    model.save(out_dir, metadata=provenance)
    return model, provenance


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=0, help="which cascade finalist")
    ap.add_argument("--trial", type=int, default=None,
                    help="a specific cascade trial number, for a contrasting arm "
                         "the top of the ranking does not cover")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.trial is not None:
        pool = json.loads((TUNING / "cascade" / "trials.json").read_text(encoding="utf-8"))
        trial = next((t for t in pool if t["number"] == args.trial), None)
        if trial is None:
            raise SystemExit(f"cascade trial {args.trial} not found")
    else:
        trial = _best("cascade")[args.rank]
    model, prov = materialize(trial, args.out)
    print(f"materialised cascade trial {trial['number']} -> {args.out}")
    print(f"  profile={prov['profile']} heads={prov['head_family']} "
          f"score_mode={model.score_mode} enabled_tags={model.config['n_enabled_tags']}")
    print(f"  reproduction drift {prov['reproduction_drift']:.5f} (tolerance {TOLERANCE})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
