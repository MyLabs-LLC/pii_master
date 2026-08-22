"""The pre-scan windows are an optimization, not a behavior change.

For every corpus document and a set of generated benchmark documents, the
pipeline must produce identical entities whether detectors use their
pre-scan windows or scan the full text.
"""

import random
from pathlib import Path

import pytest

from pii_master.bench import make_doc
from pii_master.evaluation import load_corpus
from pii_master.pipeline import Pipeline
from pii_master.detectors import default_detectors

CORPUS_DIR = Path(__file__).resolve().parent.parent / "eval" / "corpus"


def full_scan_pipeline() -> Pipeline:
    detectors = default_detectors()
    for d in detectors:
        d.hints = ()
        d.use_digit_runs = False
        d.window_pattern = None
    return Pipeline(detectors)


def sample_texts() -> list[str]:
    texts = [doc.text for doc in load_corpus(sorted(CORPUS_DIR.glob("*.jsonl")))]
    rng = random.Random(42)
    texts += [make_doc(rng, size) for size in (1_000, 5_000, 20_000)]
    return texts


@pytest.mark.parametrize("index", range(len(sample_texts())))
def test_windowed_equals_full_scan(index):
    text = sample_texts()[index]
    assert Pipeline().run(text) == full_scan_pipeline().run(text)
