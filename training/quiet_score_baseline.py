"""The prior champion as a comparison arm, under this lineage's rules.

`pii-priority-fusion-1k-v1` is the thing this run is trying to beat, so it has
to be measured the same way: the same collapsed catalogue, the same corpora, the
same document-level gold. Two details matter.

Its **predictions are reused**, not recomputed. The prior lineage recorded all
126,129 sealed predictions, the evaluation manifests have not changed since, and
re-running inference to obtain identical output would only add a way for the
baseline to drift. Its labels are mapped through the same frozen collapse that
the new models' gold is mapped through, so neither side is scored against a
catalogue the other does not use.

Its **latency is remeasured** here, on one core, through its own packaged entry
point. A number carried over from another run's report is not comparable to one
measured today on this machine, and the p95 column is a gate.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.quiet_cache import load_catalogue  # noqa: E402
from training.quiet_data import (  # noqa: E402
    EVAL_ROOT, canonical_stem, collapse_tags, iter_quiet_corpus, read_document,
    resolve_dataset,
)

BUNDLE = Path("/home/lence/workspace/pii_master/projects/pii-priority-recall-v1/"
              "dist/pii-priority-fusion-1k-v1")
PREDICTIONS = (Path("/home/lence/workspace/pii_master/projects/pii-priority-recall-v1/"
                    "evaluations/champion_1k/predictions.jsonl"))


def _recorded() -> dict[str, dict[str, list[str]]]:
    """Keyed by canonical stem, not by the dataset name the prior run wrote.

    The predictions were recorded before the 2026-08-25 directory rename, so
    they name `pii2_eval_30k` where the corpus is now `30000_pii2_eval_25.15k`.
    Keying on the literal name silently matched nothing on six of eight corpora
    and scored the baseline as predicting no tags at all -- which would have
    flattered the new champion enormously on precision.
    """
    by_stem: dict[str, dict[str, list[str]]] = {}
    with PREDICTIONS.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            by_stem.setdefault(canonical_stem(r["dataset"]), {})[r["uid"]] = r["labels"]
    return by_stem


def _latency(n_docs: int = 150, min_chars: int = 10_000) -> tuple[float, float]:
    sys.path.insert(0, str(BUNDLE / "runtime"))
    from training.priority_hash import load_priority_model

    model = load_priority_model(BUNDLE / "models" / "model")
    window = int(model.read_window_chars)
    texts: list[str] = []
    for corpus in ("6589_govdocs2-dualjudge-eval20-3.53k", "4000_datax-dualjudge-evalset-1.32k"):
        for qr in iter_quiet_corpus(resolve_dataset(corpus)):
            try:
                text = read_document(Path(qr.path), limit=min_chars * 2)
            except (FileNotFoundError, OSError):
                continue
            if len(text) >= min_chars:
                texts.append(text)
            if len(texts) >= n_docs:
                break
        if len(texts) >= n_docs:
            break
    timings = []
    for _ in range(3):
        for text in texts:
            t0 = time.perf_counter()
            model.predict(text[:window])
            timings.append((time.perf_counter() - t0) * 1000.0)
    a = np.asarray(timings)
    return float(np.percentile(a, 95)), float(1000.0 / a.mean())


def baseline_predictions() -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], float, float]:
    """Per-corpus (fired_tags, fired_doc) aligned to this lineage's cache order."""
    catalogue = tuple(load_catalogue()["labels"])
    index = {label: i for i, label in enumerate(catalogue)}
    recorded = _recorded()
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for corpus_dir in sorted(d for d in EVAL_ROOT.iterdir() if d.is_dir()):
        name = corpus_dir.name
        rows = list(iter_quiet_corpus(corpus_dir))
        preds = recorded.get(canonical_stem(name), {})
        fired = np.zeros((len(rows), len(catalogue)), dtype=bool)
        missing = 0
        for i, qr in enumerate(rows):
            labels = preds.get(qr.uid)
            if labels is None:
                missing += 1
                continue
            for label in collapse_tags(labels):
                j = index.get(label)
                if j is not None:
                    fired[i, j] = True
        if missing:
            print(f"    note: {missing} of {len(rows)} {name} rows had no recorded "
                  f"prediction and are scored as no-tags", file=sys.stderr)
        out[name] = (fired, fired.any(axis=1))
    p95, dps = _latency()
    return out, p95, dps


__all__ = ["baseline_predictions"]
