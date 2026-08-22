import random

from pii_master.bench import _make_card, generate_docs, make_doc, run
from pii_master.validators import luhn_ok


def test_generation_is_deterministic():
    a = generate_docs(seed=11, sizes=(500, 1500), docs_per_size=3)
    b = generate_docs(seed=11, sizes=(500, 1500), docs_per_size=3)
    assert a == b


def test_generated_cards_pass_luhn():
    rng = random.Random(3)
    for _ in range(20):
        assert luhn_ok(_make_card(rng))


def test_doc_size_close_to_target():
    doc = make_doc(random.Random(5), 2_000)
    assert 2_000 <= len(doc) <= 2_300


def test_run_reports_buckets_and_verdict():
    report = run(seed=1, sizes=(500, 1500), docs_per_size=3, budget_ms=5.0)
    assert len(report.buckets) == 2
    for bucket in report.buckets:
        assert bucket.docs == 3
        assert bucket.p95_ms > 0
        assert bucket.allowed_ms == 5.0  # both buckets are under 10 KB
    payload = report.to_dict()
    assert payload["budget_ms_per_10kb"] == 5.0
    assert isinstance(payload["ok"], bool)
    assert "verdict" in report.render() or "PASS" in report.render() or "FAIL" in report.render()
