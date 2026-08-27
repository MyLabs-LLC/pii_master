"""Verify a generated negative corpus before any of it reaches training.

Two failure modes make generated negatives worse than no data at all, and neither
is visible by reading a few samples.

**A mislabelled negative.** The corpus asserts `doc_has_pii = False` for every
document. If an LLM asked for a blank intake form helpfully filled in a name and a
social security number, that document is a positive labelled negative, and it
teaches the gate to stay shut on exactly the documents it exists to catch. Every
document is therefore scanned with the same shape detectors the model itself uses
(`priority_hash._SHAPE_PATTERNS`) plus phone, date-of-birth and street-address
patterns, and any hit is **rejected, not repaired** -- a corpus is cheaper to
regenerate than to trust.

**A shared tell.** If every generated negative carries "this document contains no
real data", the gate learns that sentence, the generated set becomes trivially
separable, and the sealed score does not move -- while the training metric
improves and looks like progress. Disclaimer phrasing is rejected outright. A
document echoing its own class name is only *warned* about: the class is a
generation bucket that nothing here predicts, so "Data Dictionary" at the top of
a data dictionary is a realistic document, not a leak.

The last section is not a gate but the point of the exercise: it scores the
surviving documents with the current champion gate and compares them against real
negatives. Documents that score *low* are ordinary clean documents, of which
training already has plenty. Only documents the gate currently gets **wrong** are
worth adding.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.h2h_datagen import CLASSES  # noqa: E402
from training.h2h_priority import PROJECT  # noqa: E402
from training.priority_hash import _SHAPE_PATTERNS, document_features  # noqa: E402
from training.quiet_cache import CACHE_ROOT as QUIET_CACHE  # noqa: E402
from training.quiet_model import QuietCascade  # noqa: E402

#: Identifier shapes that make a document a positive. The model's own detectors,
#: plus three the model does not carry but the judges would certainly tag.
#:
#: `ipv6` is overridden rather than inherited. The model's own pattern accepts
#: two colon-separated groups, so it matches the timestamp `10:15:30` -- which
#: cost 3 of the pilot's 40 documents, all of them log extracts, the single most
#: adversarial class in the corpus. The model carrying that false positive is
#: its business; a *verifier* that inherits it silently discards exactly the
#: documents worth keeping. This one needs four groups or a `::` elision.
REJECT_PATTERNS: dict[str, re.Pattern[str]] = dict(_SHAPE_PATTERNS) | {
    "ipv6": re.compile(
        r"(?<![0-9A-Fa-f:])(?:(?:[0-9A-Fa-f]{1,4}:){3,7}[0-9A-Fa-f]{1,4}"
        r"|(?:[0-9A-Fa-f]{1,4})?::(?:[0-9A-Fa-f]{1,4}:){0,6}[0-9A-Fa-f]{1,4})"
        r"(?![0-9A-Fa-f:])"),
    "phone": re.compile(r"(?<!\d)\(?\d{3}\)?[-. ]\d{3}[-. ]\d{4}(?!\d)"),
    "street_address": re.compile(
        r"(?i)\b\d{1,5}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+"
        r"(street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr|court|ct)\b"),
    "dob_value": re.compile(
        r"(?i)(date of birth|dob|birth ?date)\s*[:=]\s*\d"),
}

#: Phrases that would give the whole corpus one tell to memorise.
TELLS = tuple(re.compile(p, re.I) for p in (
    r"\bno (real|actual|genuine|personal) (data|information|identifiers)\b",
    r"\bsample (document|only)\b", r"\bfor illustrative purposes\b",
    r"\bthis is a (template|sample|test|mock|example) (document|record|file)\b",
    r"\bfictitious\b", r"\bdummy data\b", r"\bplaceholder data\b",
    r"\bdoes not contain (any )?(real|actual) \w+\b",
))


def scan(text: str, slug: str) -> tuple[list[str], list[str]]:
    """(reasons to reject, warnings). An empty reject list means usable."""
    reasons, warnings = [], []
    for name, pat in REJECT_PATTERNS.items():
        if pat.search(text):
            reasons.append(f"identifier:{name}")
    for pat in TELLS:
        if pat.search(text):
            reasons.append("tell:disclaimer")
            break
    # A document of class `data_dictionary` naturally opens "Data Dictionary",
    # and that is not leakage: the class is a generation bucket, not something
    # any model here predicts. The oracle is `doc_has_pii = False`, and the
    # phrase that would leak *that* is a disclaimer, which is rejected above.
    # Recorded as a warning so a genuine slug echo is still visible.
    if slug and (slug in text.lower() or slug.replace("_", " ") in text.lower()):
        warnings.append("echoes_class_name")
    return reasons, warnings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--gate", type=Path, default=PROJECT / "models" / "cascade")
    ap.add_argument("--sample", type=int, default=400,
                    help="real negatives to compare the score distribution against")
    args = ap.parse_args()

    manifest = json.loads((args.corpus / "manifest.json").read_text(encoding="utf-8"))
    slugs = {c.name for c in CLASSES}
    print(f"scanning {len(manifest):,} documents from {args.corpus.name}", flush=True)

    kept: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    warns: Counter[str] = Counter()
    per_class_rej: Counter[str] = Counter()
    for rec in manifest:
        text = (args.corpus / rec["path"]).read_text(encoding="utf-8")
        bad, warn = scan(text, rec["class"] if rec["class"] in slugs else "")
        for w in warn:
            warns[w] += 1
        if bad:
            for b in bad:
                reasons[b] += 1
            per_class_rej[rec["class"]] += 1
            rec["rejected_for"] = bad
        else:
            kept.append(rec)

    n_rej = len(manifest) - len(kept)
    print(f"\nrejected {n_rej:,} of {len(manifest):,} ({n_rej / max(len(manifest), 1):.1%})")
    for r, n in reasons.most_common():
        print(f"    {r:<28} {n:>5}")
    if per_class_rej:
        print("  worst classes:", dict(per_class_rej.most_common(4)))
    if warns:
        print("  warnings (kept):", dict(warns))
    if not kept:
        raise SystemExit("every document was rejected; fix the generator and regenerate")

    # ------------------------------------------------- is any of this adversarial?
    model = QuietCascade.load(args.gate)
    def gate_score(text: str) -> float:
        idx = model.features(text)
        return model.gate_score(idx) if len(idx) else float("-inf")

    gen = np.asarray([gate_score((args.corpus / r["path"]).read_text(encoding="utf-8"))
                      for r in kept], dtype=np.float64)

    # Real negatives from the sealed real-world corpus, for scale.
    with np.load(QUIET_CACHE / "6589_govdocs2-dualjudge-eval20-3.53k.npz",
                 allow_pickle=False) as z:
        indptr, indices, tgt = z["indptr_deep"], z["indices_deep"], z["doc_target"]
    neg = np.flatnonzero(tgt == 0)[: args.sample]
    real = np.asarray([model.gate_weights[indices[indptr[i]:indptr[i + 1]]].sum()
                       + model.gate_intercept for i in neg], dtype=np.float64)

    q = (10, 25, 50, 75, 90)
    print(f"\ngate scores (threshold {model.gate_threshold:.1f} — above it the gate FIRES):")
    print(f"{'':<26}{'n':>7}" + "".join(f"{f'p{x}':>10}" for x in q) + f"{'% firing':>10}")
    for lbl, v in (("generated negatives", gen), ("real negatives (govdocs2)", real)):
        print(f"{lbl:<26}{len(v):>7,}" + "".join(f"{np.percentile(v, x):>10.1f}" for x in q)
              + f"{(v >= model.gate_threshold).mean():>9.1%}")

    fires = float((gen >= model.gate_threshold).mean())
    print(f"\nthe current gate wrongly fires on {fires:.1%} of the generated negatives.")
    print("    Those are the documents worth adding: the ones it already gets right "
          "teach it nothing.")

    out = args.corpus / "manifest_verified.json"
    out.write_text(json.dumps(kept, indent=1) + "\n", encoding="utf-8")
    (args.corpus / "verification.json").write_text(json.dumps({
        "n_input": len(manifest), "n_kept": len(kept), "n_rejected": n_rej,
        "reasons": dict(reasons), "per_class_rejected": dict(per_class_rej),
        "warnings": dict(warns),
        "gate_threshold": model.gate_threshold,
        "generated_fire_rate": fires,
        "generated_score_percentiles": {f"p{x}": float(np.percentile(gen, x)) for x in q},
        "real_negative_score_percentiles": {f"p{x}": float(np.percentile(real, x)) for x in q},
    }, indent=1) + "\n", encoding="utf-8")
    print(f"\n-> {out}  ({len(kept):,} usable documents)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
