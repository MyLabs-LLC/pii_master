from pii_master.entities import EntityType
from pii_master.fusion import (
    CHECKSUMMED,
    fuse_checksum_first,
    merge_adjacent_same_type,
    resolve_greedy,
)
from pii_master.models import Entity
from pii_master.onnx_ner import revalidate
from pii_master.pipeline import Pipeline
from pii_master.validators import luhn_ok


def _ent(type_, start, end, conf=0.8, detector="regex/x"):
    return Entity(
        type=type_, start=start, end=end,
        text="x" * (end - start), confidence=conf, detector=detector,
    )


def test_checksum_rule_outranks_overlapping_model():
    rule = _ent(EntityType.CREDIT_CARD, 0, 16, 0.95, "regex/card")
    model = _ent(EntityType.CREDIT_CARD, 0, 16, 0.99, "onnx/student-m")
    assert fuse_checksum_first([rule], [model]) == [rule]


def test_longer_same_type_rule_beats_truncated_model():
    rule = _ent(EntityType.HEALTH_PLAN_ID, 0, 11, 0.80, "regex/plan")
    truncated = _ent(EntityType.HEALTH_PLAN_ID, 0, 8, 0.90, "onnx/student-m")
    assert fuse_checksum_first([rule], [truncated]) == [rule]


def test_model_kept_when_as_long_as_the_rule():
    rule = _ent(EntityType.MRN, 0, 7, 0.85, "regex/mrn")
    model = _ent(EntityType.MRN, 0, 7, 0.70, "onnx/student-m")
    assert fuse_checksum_first([rule], [model]) == [model]


def test_cue_anchored_rule_fills_silence():
    rule = _ent(EntityType.MRN, 0, 7, 0.85, "regex/mrn")
    assert fuse_checksum_first([rule], []) == [rule]


def test_non_overlapping_model_kept_beside_checksum_rule():
    card = _ent(EntityType.CREDIT_CARD, 0, 16, 0.95, "regex/card")
    name_as_mrn = _ent(EntityType.MRN, 20, 27, 0.80, "onnx/student-m")
    kept = fuse_checksum_first([card], [name_as_mrn])
    assert kept == [card, name_as_mrn]


def test_checksummed_set_matches_plan():
    assert EntityType.SSN in CHECKSUMMED
    assert EntityType.MRN not in CHECKSUMMED
    assert EntityType.PHONE_US not in CHECKSUMMED


def test_pipeline_without_ner_still_greedy():
    low = _ent(EntityType.PHONE_US, 0, 10, 0.85, "regex/phone")
    high = _ent(EntityType.SSN, 5, 16, 0.90, "regex/ssn")

    class Fake:
        name = "fake"

        def __init__(self, entities):
            self._entities = entities

        def detect(self, text):
            return list(self._entities)

    result = Pipeline([Fake([low, high])]).run("x" * 20)
    assert result == [high]
    assert result == resolve_greedy([low, high])


def test_revalidate_rejects_invalid_card():
    assert luhn_ok("4111111111111111")
    assert not revalidate(EntityType.CREDIT_CARD, "4111 1111 1111 1112")
    assert revalidate(EntityType.CREDIT_CARD, "4111 1111 1111 1111")
    assert revalidate(EntityType.MRN, "4829471")


def test_merge_adjacent_rebuilds_full_name():
    text = "Applicant Jane Doe applied."
    first = Entity(EntityType.PERSON_NAME, 10, 14, "Jane", 0.9, "onnx/student-m")
    last = Entity(EntityType.PERSON_NAME, 15, 18, "Doe", 0.8, "onnx/student-m")
    merged = merge_adjacent_same_type([first, last], text)
    assert len(merged) == 1
    assert merged[0].text == "Jane Doe"
    assert (merged[0].start, merged[0].end) == (10, 18)


def test_merge_adjacent_joins_street_and_city():
    text = "mail to 44 Elm Street, Springfield today"
    street = Entity(EntityType.ADDRESS, 8, 21, "44 Elm Street", 0.9, "onnx/student-m")
    city = Entity(EntityType.ADDRESS, 23, 34, "Springfield", 0.7, "onnx/student-m")
    merged = merge_adjacent_same_type([street, city], text)
    assert text[8:21] == "44 Elm Street"
    assert text[23:34] == "Springfield"
    assert len(merged) == 1
    assert merged[0].text == "44 Elm Street, Springfield"


def test_merge_adjacent_does_not_join_different_types():
    text = "Jane 123-45-6789"
    name = Entity(EntityType.PERSON_NAME, 0, 4, "Jane", 0.9, "onnx/student-m")
    ssn = Entity(EntityType.SSN, 5, 16, "123-45-6789", 0.9, "regex/ssn")
    merged = merge_adjacent_same_type([name, ssn], text)
    assert len(merged) == 2


def test_merge_snaps_leading_subword_fragment():
    text = "Applicant Jane Doe applied."
    # Student tagged the last subword of "Applicant" as part of the name.
    blob = Entity(EntityType.PERSON_NAME, 6, 18, "ant Jane Doe", 0.9, "onnx/student-m")
    assert text[6:18] == "ant Jane Doe"
    merged = merge_adjacent_same_type([blob], text)
    assert len(merged) == 1
    assert merged[0].text == "Jane Doe"
    assert (merged[0].start, merged[0].end) == (10, 18)
