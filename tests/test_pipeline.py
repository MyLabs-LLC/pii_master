from pii_master.detectors import Detector
from pii_master.entities import EntityType
from pii_master.models import Entity
from pii_master.pipeline import Pipeline


class FakeDetector:
    """Plain class, no inheritance: proves the structural Protocol seam that
    the Stage 2 ONNX detector will rely on."""

    def __init__(self, name: str, entities: list[Entity]):
        self.name = name
        self._entities = entities

    def detect(self, text: str) -> list[Entity]:
        return list(self._entities)


def make(type_, start, end, conf, detector="fake"):
    return Entity(
        type=type_, start=start, end=end,
        text="x" * (end - start), confidence=conf, detector=detector,
    )


def test_fake_detector_satisfies_protocol():
    assert isinstance(FakeDetector("fake", []), Detector)


def test_overlap_keeps_higher_confidence():
    low = make(EntityType.PHONE_US, 0, 10, 0.85, "fake/phone")
    high = make(EntityType.SSN, 5, 16, 0.90, "fake/ssn")
    result = Pipeline([FakeDetector("a", [low]), FakeDetector("b", [high])]).run("x" * 20)
    assert result == [high]


def test_non_overlapping_spans_all_kept_and_sorted():
    a = make(EntityType.EMAIL, 20, 30, 0.95)
    b = make(EntityType.SSN, 0, 11, 0.90)
    result = Pipeline([FakeDetector("a", [a, b])]).run("x" * 40)
    assert result == [b, a]  # sorted by start


def test_exact_duplicates_collapse():
    a = make(EntityType.EMAIL, 0, 10, 0.95, "fake/1")
    b = make(EntityType.EMAIL, 0, 10, 0.95, "fake/2")
    result = Pipeline([FakeDetector("a", [a]), FakeDetector("b", [b])]).run("x" * 10)
    assert len(result) == 1


def test_deterministic_on_real_detectors():
    text = "Email jane@example.com, SSN 123-45-6789, card 4111 1111 1111 1111."
    pipeline = Pipeline()
    assert pipeline.run(text) == pipeline.run(text)


def test_empty_text():
    assert Pipeline().run("") == []
