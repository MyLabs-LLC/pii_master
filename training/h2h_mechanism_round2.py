"""Round 2: cross the one axis round 1 found with the box parameters.

Round 1 swept 21 mechanisms and only one beat the shipped baseline on calibration
micro F1 — choosing the point inside the box by F1 rather than F0.5 (+0.0124). It
did so by trading precision away (0.9244 -> 0.8984), which the target forbids.

So `beta` is a real axis and the box is a real axis, and round 1 only moved one at
a time. This crosses them, still on the **calibration carve**, and screens
candidates through the measured calibration->sealed gap before any of them is
allowed near the sealed corpora.

## The gap correction, and why it is not a fudge

Two arms have been measured on both splits, so the gap is observed rather than
assumed:

    cascade_p80r70   calib P 0.9558 -> sealed 0.9165   (-0.039)
    cascade_p88r90   calib P 0.9244 -> sealed 0.9000   (-0.024)
    cascade_p80r70   calib R 0.8712 -> sealed 0.7777   (-0.094)
    cascade_p88r90   calib R 0.8352 -> sealed 0.8020   (-0.033)

The gap is not constant, so the screen uses the **worst** observed drop as the
margin a candidate must clear. A candidate that only passes under the friendlier
gap is not promoted to a sealed measurement; that would be choosing the
correction to suit the answer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.h2h_mechanism_sweep import (  # noqa: E402
    box_thresholds, equal_corpus_micro, fire_from_thr,
)
from training.h2h_scorecard_rebuild import retarget_cache  # noqa: E402

SCORECARD = Path("projects/pii-scorecard-60")
#: worst observed calibration -> sealed drop, from the two arms measured on both
GAP_P, GAP_R = 0.039, 0.094
TARGET_P, TARGET_R, TARGET_F1 = 0.90, 0.80, 0.80


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path("projects/pii-target-8070/probe/round2_sweep.json"))
    args = ap.parse_args()

    cat = retarget_cache(SCORECARD / "cache", 61)
    labels = tuple(cat["labels"])
    from training.quiet_fit import carve_holdin, load, score, train_corpora  # noqa
    from training.quiet_model import QuietCascade  # noqa

    model = QuietCascade.load(SCORECARD / "models" / "cascade_scorecard61")
    ds = load(train_corpora(), profile="deep")
    _, calib_mask = carve_holdin(ds)
    calib = ds.subset(calib_mask)
    g = (calib.X @ model.gate_weights + model.gate_intercept).astype(np.float32)
    open_doc = g >= model.gate_threshold
    S = score(calib.X[open_doc], model.tag_weights, mode="sum")
    Y = np.asarray(calib.Y[open_doc].todense()).astype(bool)
    TC = calib.tag_complete[open_doc]
    corpus = calib.corpus[open_doc]
    print(f"{S.shape[0]:,} calibration rows; screening with the WORST observed gap "
          f"(P -{GAP_P}, R -{GAP_R})\n", flush=True)

    rows = []
    print(f"{'box P':>6}{'box R':>6}{'beta':>6}{'calib P':>9}{'calib R':>9}"
          f"{'calib F1':>10}{'pred P':>8}{'pred R':>8}   screen")
    for p_t in (0.86, 0.88, 0.90, 0.92, 0.94):
        for r_t in (0.85, 0.90, 0.92):
            for beta in (0.5, 0.75, 1.0):
                thr = box_thresholds(S, Y, TC, p_t, r_t, beta=beta)
                P, R, F = equal_corpus_micro(fire_from_thr(S, thr), Y, TC, corpus)
                pp, pr = P - GAP_P, R - GAP_R
                ok = pp >= TARGET_P and pr >= TARGET_R and F >= TARGET_F1
                rows.append({"p_target": p_t, "r_target": r_t, "beta": beta,
                             "calib_precision": P, "calib_recall": R, "calib_f1": F,
                             "predicted_sealed_precision": pp,
                             "predicted_sealed_recall": pr, "passes_screen": bool(ok)})
                print(f"{p_t:>6.2f}{r_t:>6.2f}{beta:>6.2f}{P:>9.4f}{R:>9.4f}"
                      f"{F:>10.4f}{pp:>8.4f}{pr:>8.4f}   {'PASS' if ok else ''}",
                      flush=True)

    ok = [r for r in rows if r["passes_screen"]]
    rows.sort(key=lambda r: -r["calib_f1"])
    print(f"\n{len(ok)} of {len(rows)} candidates clear the screen")
    if ok:
        ok.sort(key=lambda r: -r["calib_f1"])
        for r in ok[:3]:
            print(f"  box P>={r['p_target']} R>={r['r_target']} beta={r['beta']}  "
                  f"calib F1 {r['calib_f1']:.4f}  predicted sealed "
                  f"P {r['predicted_sealed_precision']:.4f} "
                  f"R {r['predicted_sealed_recall']:.4f}")
    else:
        print("  none — the shipped model is at the frontier for this target under "
              "the worst-case gap. Best calibration F1 overall:")
        for r in rows[:3]:
            print(f"  box P>={r['p_target']} R>={r['r_target']} beta={r['beta']}  "
                  f"calib F1 {r['calib_f1']:.4f}  predicted sealed "
                  f"P {r['predicted_sealed_precision']:.4f} "
                  f"R {r['predicted_sealed_recall']:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"split": "training calibration carve", "gap": {"precision": GAP_P,
                                                        "recall": GAP_R},
         "target": {"precision": TARGET_P, "recall": TARGET_R, "micro_f1": TARGET_F1},
         "n_passing_screen": len(ok), "results": rows}, indent=1), encoding="utf-8")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
