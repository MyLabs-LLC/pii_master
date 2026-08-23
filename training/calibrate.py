"""Fit a calibration curve for a serving bundle — the deferred piece of M2.

docs/DESIGN.md section 7 is explicit that detector confidences are "ordinal
detector certainty, not calibrated probabilities", and section 8 defers real
calibration to "where scores come from a model that can be calibrated". This is
that script.

**Why the raw score is not a probability.** The student's per-span confidence is
the mean of its max-softmax over the span's tokens. A token classifier trained
with cross-entropy is pushed to put all the mass on one class, so that number is
systematically overconfident and has no units: `min_confidence=0.75` means "in
the top band of this particular model's self-assessment" and nothing more. It
cannot be compared between students, and it cannot be reasoned about.

**What this fits.** Isotonic regression (PAVA) from raw span confidence onto the
strict target: did this span exactly match a gold span on `(type, start, end)`?
Afterwards a confidence of 0.75 means *this span has roughly a 75% chance of
being exactly right*, for any student, and the risk score in classify.py is
multiplying weights by something that means something.

Isotonic rather than Platt because the relationship is not assumed sigmoid, and
**monotone** because that guarantees calibration can never re-order two spans:
it changes what the number means without changing which spans outrank which, so
no ranking measured before calibration is invalidated by it.

**Fit and evaluation slices must be disjoint,** and neither may be the training
split -- the student's confidence on documents it was trained on is not the
confidence it will have in production. Both default to slices of `test`.

    python training/calibrate.py --data-dir ~/nemotron \\
        --model-dir training/artifacts/bundle_m --fit-slice 20000:40000
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pii_master.crosswalk import to_entity_type  # noqa: E402
from pii_master.ner import OnnxNerDetector, load_bundle, merge_adjacent  # noqa: E402


def parse_spans(raw):
    if raw is None:
        return []
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return ast.literal_eval(raw)


def read_slice(data_dir, split, start, stop):
    import pyarrow.parquet as pq

    files = sorted(Path(data_dir).glob(f"{split}-*.parquet"))
    if not files:
        raise SystemExit(f"no {split} parquet under {data_dir}")
    table = pq.read_table(files[0], columns=["text", "spans"])
    texts = table.column("text").to_pylist()[start:stop]
    spans = table.column("spans").to_pylist()[start:stop]
    return texts, spans


def gold_for(text, raw, kinds):
    """Gold spans, crosswalked and merged exactly as the detector merges."""
    spans = []
    for span in parse_spans(raw):
        entity = to_entity_type(span["label"])
        if entity is not None:
            spans.append((entity, span["start"], span["end"]))
    index = {k: i for i, k in enumerate(kinds)}
    packed = sorted(((index[t], s, e, 1.0) for t, s, e in spans if t in index),
                    key=lambda x: (x[1], x[2]))
    return {(kinds[k], s, e) for k, s, e, _ in merge_adjacent(packed, text, kinds)}


def collect(detector, texts, raw_spans, kinds):
    """-> (raw confidences, hit/miss) for every span the detector emits."""
    scores, correct = [], []
    for index, (text, raw) in enumerate(zip(texts, raw_spans)):
        gold = gold_for(text, raw, kinds)
        for entity in detector.detect(text):
            scores.append(entity.confidence)
            correct.append((entity.type, entity.start, entity.end) in gold)
        if index and index % 2000 == 0:
            print(f"  {index:,} documents, {len(scores):,} spans", flush=True)
    return np.asarray(scores), np.asarray(correct, dtype=np.float64)


def isotonic(x, y, weights=None):
    """Pool-adjacent-violators. -> (knot_x, knot_y), knot_x strictly ascending.

    Twenty lines instead of a scikit-learn dependency, and the serving side
    only needs np.interp over the result -- so calibration costs the runtime
    two arrays and no new package.

    Equal x values are pooled BEFORE the sweep. Tens of thousands of spans
    share a handful of distinct scores, and np.interp requires strictly
    increasing x: leaving ties in would produce two knots at the same score
    with different values, where the interpolation silently picks one.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=np.float64)

    unique, inverse = np.unique(x, return_inverse=True)
    totals = np.bincount(inverse, weights=y * w, minlength=unique.size)
    counts = np.bincount(inverse, weights=w, minlength=unique.size)

    # Each block is (sum of w*y, sum of w) over a contiguous run of x. Append
    # left to right, merging backwards while the running means decrease.
    sums: list[float] = []
    mass: list[float] = []
    left: list[float] = []
    right: list[float] = []
    for position, total, count in zip(unique, totals, counts):
        sums.append(float(total))
        mass.append(float(count))
        left.append(float(position))
        right.append(float(position))
        while len(sums) > 1 and sums[-2] / mass[-2] > sums[-1] / mass[-1]:
            merged_sum, merged_mass = sums.pop(), mass.pop()
            left.pop()
            merged_right = right.pop()
            sums[-1] += merged_sum
            mass[-1] += merged_mass
            right[-1] = merged_right       # the merged block spans both

    # Two knots per block, at its left and right edge, so np.interp is FLAT
    # inside a block and ramps only between them. Emitting one knot per block
    # would make it ramp across the block instead, which is not the function
    # isotonic regression fitted -- on the textbook case [4,5,1,6,8,7] that
    # error turns the correct 3.33,3.33,3.33 into 3.33,4.22,5.11.
    knot_x: list[float] = []
    knot_y: list[float] = []
    for block_sum, block_mass, low, high in zip(sums, mass, left, right):
        value = block_sum / block_mass
        knot_x.append(low)
        knot_y.append(value)
        if high > low:
            knot_x.append(high)
            knot_y.append(value)
    return np.asarray(knot_x), np.asarray(knot_y)


def thin(x, y, max_knots=64):
    """Keep at most max_knots, evenly spaced along x, always both ends.

    The raw fit has one knot per distinct score -- tens of thousands. The curve
    is monotone and smooth, so an evenly spaced subset reproduces it to well
    inside the noise of the fit, and the bundle stays small and readable.
    """
    if x.size <= max_knots:
        return x, y
    picks = np.unique(np.linspace(0, x.size - 1, max_knots).round().astype(int))
    return x[picks], y[picks]


def reliability(scores, correct, edges=(0.0, .5, .6, .7, .8, .9, .95, 1.01)):
    rows = []
    for low, high in zip(edges, edges[1:]):
        mask = (scores >= low) & (scores < high)
        if mask.sum():
            rows.append((low, high, int(mask.sum()),
                         float(scores[mask].mean()), float(correct[mask].mean())))
    return rows


def show(title, rows):
    print(f"\n{title}")
    print(f"  {'band':>12} {'spans':>8} {'mean score':>11} {'actual':>8} {'gap':>8}")
    for low, high, n, mean, actual in rows:
        print(f"  {low:>5.2f}-{high:<6.2f} {n:>8,} {mean:>11.3f} {actual:>8.3f} "
              f"{mean - actual:>+8.3f}")


def expected_error(rows, total):
    """Expected calibration error: mean |claimed - actual|, weighted by mass."""
    return sum(n * abs(mean - actual) for _, _, n, mean, actual in rows) / total


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--fit-slice", default="20000:40000",
                    help="start:stop rows to FIT on")
    ap.add_argument("--check-slice", default="40000:50000",
                    help="start:stop rows to CHECK on; must not overlap the fit")
    ap.add_argument("--max-knots", type=int, default=64)
    ap.add_argument("--dry-run", action="store_true",
                    help="report the curve without writing it to the bundle")
    args = ap.parse_args(argv)

    fit_lo, fit_hi = (int(v) for v in args.fit_slice.split(":"))
    chk_lo, chk_hi = (int(v) for v in args.check_slice.split(":"))
    if max(fit_lo, chk_lo) < min(fit_hi, chk_hi):
        raise SystemExit("--fit-slice and --check-slice overlap; they must not")

    bundle = load_bundle(args.model_dir)
    if bundle.calibration:
        print("note: this bundle is ALREADY calibrated. Refitting on top of an "
              "existing curve would compose two of them; fitting against raw "
              "scores instead.")
    kinds = bundle.kinds

    # min_confidence=0 so the fit sees the whole score range, not just the
    # part that survives the shipped threshold -- a curve fitted only above
    # 0.75 cannot tell you what 0.4 means.
    detector = OnnxNerDetector(args.model_dir, min_confidence=0.0)
    detector._bundle = ModelBundleWithoutCalibration(bundle)

    print(f"fitting on {args.split}[{fit_lo}:{fit_hi}]", flush=True)
    texts, raws = read_slice(args.data_dir, args.split, fit_lo, fit_hi)
    scores, correct = collect(detector, texts, raws, kinds)
    print(f"  {len(scores):,} spans, {correct.mean():.3f} exact-match rate")

    knot_x, knot_y = thin(*isotonic(scores, correct), args.max_knots)
    print(f"  isotonic fit -> {knot_x.size} knots, "
          f"range {knot_y[0]:.3f}..{knot_y[-1]:.3f}")

    print(f"\nchecking on {args.split}[{chk_lo}:{chk_hi}] (disjoint)", flush=True)
    texts, raws = read_slice(args.data_dir, args.split, chk_lo, chk_hi)
    scores, correct = collect(detector, texts, raws, kinds)
    adjusted = np.interp(scores, knot_x, knot_y)

    before = reliability(scores, correct)
    after = reliability(adjusted, correct)
    show("BEFORE — raw max-softmax vs actual exact-match rate", before)
    show("AFTER — calibrated vs actual", after)
    print(f"\nexpected calibration error: "
          f"{expected_error(before, len(scores)):.4f} -> "
          f"{expected_error(after, len(scores)):.4f}")

    if args.dry_run:
        print("\n--dry-run: bundle not modified")
        return 0

    meta_path = Path(args.model_dir) / "model.json"
    meta = json.loads(meta_path.read_text())
    meta["calibration"] = {
        "method": "isotonic",
        "target": "exact (type, start, end) match against crosswalked gold",
        "fit_split": f"{args.split}[{fit_lo}:{fit_hi}]",
        "check_split": f"{args.split}[{chk_lo}:{chk_hi}]",
        "x": [round(float(v), 6) for v in knot_x],
        "y": [round(float(v), 6) for v in knot_y],
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\nwrote calibration into {meta_path}")
    return 0


class ModelBundleWithoutCalibration:
    """Proxy that hides an existing curve so a refit sees raw scores.

    Refitting isotonic on already-calibrated output would compose two curves
    and quietly make the second one wrong -- and re-running this script after a
    successful fit is the most natural thing in the world to do.
    """

    def __init__(self, bundle):
        self._bundle = bundle

    def __getattr__(self, name):
        if name == "calibration":
            return ()
        return getattr(self._bundle, name)


if __name__ == "__main__":
    raise SystemExit(main())
