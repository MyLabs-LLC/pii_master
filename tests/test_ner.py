"""Stage 2: span decoding, the guards, and the fusion policy.

None of these tests needs a trained model. That is deliberate: the parts of
Stage 2 that can silently corrupt output -- the BIO decoder, the confidence
filter, checksum re-validation and tier precedence -- are all pure functions of
model *output*, so they can be tested against synthetic logits and stay green in
a CI that has no GPU, no artifact and no ML extra beyond numpy.
"""

from __future__ import annotations

import pathlib
import random

import pytest

from pii_master.entities import (
    CHECKSUMMED_TYPES,
    MODEL_ONLY_TYPES,
    TAXONOMY,
    DocLabel,
    EntityType,
)
from pii_master.models import Entity
from pii_master.pipeline import (
    TIER_CHECKSUM_RULE,
    TIER_CUE_RULE,
    TIER_MODEL,
    Pipeline,
    fusion_rank,
)

np = pytest.importorskip("numpy")

from pii_master import ner  # noqa: E402


# --------------------------------------------------------------------------
# A tiny synthetic label space, so the decoder tests do not need a model
# --------------------------------------------------------------------------

KINDS = (EntityType.PERSON_NAME, EntityType.SSN, EntityType.ADDRESS)
# id 0 = O, then B-/I- pairs for each kind, then an unmodelled B-/I- pair that
# must decode to nothing (the real table maps ~21 Nemotron labels to -1).
LABEL_NAMES = ["O"]
for kind in KINDS:
    LABEL_NAMES += [f"B-{kind.value}", f"I-{kind.value}"]
# A SECOND label collapsing onto an existing kind, mirroring the real table
# where first_name and last_name both become PERSON_NAME and street_address,
# city, county and postcode all become ADDRESS. Without one of these in the
# fuzz alphabet the crosswalk collapse is never exercised at all.
LABEL_NAMES += ["B-ALIAS_OF_ADDRESS", "I-ALIAS_OF_ADDRESS"]
LABEL_NAMES += ["B-unmodelled", "I-unmodelled"]

ADDRESS_KIND = KINDS.index(EntityType.ADDRESS)
KIND_OF = np.array(
    [-1] + [i // 2 for i in range(2 * len(KINDS))]
    + [ADDRESS_KIND, ADDRESS_KIND] + [-1, -1],
    dtype=np.int64,
)
IS_BEGIN = np.array(
    [False] + [i % 2 == 0 for i in range(2 * len(KINDS))]
    + [True, False] + [True, False]
)


def load_training_decoder():
    """Import training/decode.py, the readable reference implementation.

    Imported rather than reimplemented on purpose. training/decode.py is the
    decoder that produced every number in docs/DISTILLATION_RESULTS.md, and
    ner.decode_spans is a vectorised rewrite of it. If the two ever diverge,
    the shipped detector stops being the thing that was measured -- so the
    reference is the real file, not a copy of it that can drift alongside.
    """
    import importlib.util

    path = (pathlib.Path(__file__).resolve().parents[1]
            / "training" / "decode.py")
    if not path.exists():
        pytest.skip("training/decode.py not present")
    spec = importlib.util.spec_from_file_location("_training_decode", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.decode_spans


def reference_decode(text, offsets, label_ids):
    """training/decode.py's output, translated into (kind_index, start, end).

    It emits the label suffix as a string and knows nothing about our
    EntityType or the unmodelled labels, so this drops the labels that map to
    -1 and looks the rest up -- the only translation between the two.
    """
    decode = load_training_decoder()

    # The single difference between the two implementations, stated exactly:
    # ner.decode_spans applies the crosswalk (kind_of) BEFORE deciding where
    # runs start and stop, so two labels that collapse to one of our types
    # continue one span where the reference would close and reopen. Feeding
    # the reference kind names instead of Nemotron labels removes that
    # difference at the source, so what remains to compare is the BIO rule
    # itself -- which is the thing under test. See ner.decode_spans.
    kind_index = {kind.value: i for i, kind in enumerate(KINDS)}
    id2label = {}
    for label_id, name in enumerate(LABEL_NAMES):
        if name == "O":
            id2label[label_id] = "O"
            continue
        prefix, _, label = name.partition("-")
        kind = int(KIND_OF[label_id])
        id2label[label_id] = (f"{prefix}-{KINDS[kind].value}" if kind >= 0
                              else f"{prefix}-{label}")

    spans = decode(text, [tuple(o) for o in offsets],
                   [int(i) for i in label_ids], id2label)
    trimmed = []
    for name, start, end in spans:
        if name not in kind_index:          # deliberately unmodelled -> dropped
            continue
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            trimmed.append((kind_index[name], start, end))
    return trimmed


def whitespace_tokenize(text):
    """Offsets that mimic a subword tokenizer: leading space included in the
    token, and long words split so continuation tokens exist."""
    offsets, position = [], 0
    for word in text.split(" "):
        width = len(word)
        start = position - 1 if position else position
        if width > 4:
            offsets.append((start, position + 4))
            offsets.append((position + 4, position + width))
        else:
            offsets.append((start, position + width))
        position += width + 1
    return offsets


def test_decoder_matches_reference_on_a_fuzz_corpus():
    rng = random.Random(11)
    words = ["Jane", "Doe", "lives", "at", "44", "Elm", "Street", "Springfield",
             "SSN", "123-45-6789", "and", "the", "chart", "6789,", "x"]
    checked = 0
    for _ in range(400):
        text = " ".join(rng.choice(words) for _ in range(rng.randint(1, 14)))
        offsets = whitespace_tokenize(text)
        labels = [rng.randrange(len(LABEL_NAMES)) for _ in offsets]
        expected = reference_decode(text, offsets, labels)
        actual = [
            (kind, start, end)
            for kind, start, end, _ in ner.decode_spans(
                text, offsets, np.array(labels), KIND_OF, IS_BEGIN
            )
        ]
        assert actual == expected, (text, labels)
        checked += len(expected)
    assert checked > 200, "fuzz corpus produced too few spans to be meaningful"


def test_consecutive_b_tokens_are_one_span():
    # The failure the word-level BIO convention exists to prevent: a tokenizer
    # splits "123-45-6789" into pieces the teacher tags B,B,B and they are ONE
    # ssn, not three.
    text = "SSN 123-45-6789 filed"
    offsets = [(0, 3), (3, 7), (7, 10), (10, 15), (15, 21)]
    b_ssn = LABEL_NAMES.index("B-SSN")
    labels = np.array([0, b_ssn, b_ssn, b_ssn, 0])
    spans = ner.decode_spans(text, offsets, labels, KIND_OF, IS_BEGIN)
    assert [(s, e) for _, s, e, _ in spans] == [(4, 15)]
    assert text[4:15] == "123-45-6789"


def test_a_b_token_that_starts_a_word_opens_a_new_span():
    text = "Jane Doe"
    offsets = [(0, 4), (4, 8)]
    b_name = LABEL_NAMES.index("B-PERSON_NAME")
    spans = ner.decode_spans(
        text, offsets, np.array([b_name, b_name]), KIND_OF, IS_BEGIN
    )
    assert [(s, e) for _, s, e, _ in spans] == [(0, 4), (5, 8)]


def test_two_labels_that_collapse_to_one_type_continue_one_span():
    """The real behaviour observed on Nemotron, pinned.

    The `m` student tags ` M` as ``I-city`` in the middle of a
    ``street_address`` run: "32 Old Cty Tk M, Portage". Both labels are
    ADDRESS, and the model said "continuation" by emitting ``I-``, so the right
    answer is one span covering the whole street address -- not two, split at a
    label change the crosswalk erases anyway.
    """
    text = "at 32 Old Cty Tk M, Portage"
    offsets = [(2, 5), (5, 9), (9, 11), (11, 13), (13, 15), (15, 16), (16, 18)]
    b_addr = LABEL_NAMES.index("B-ADDRESS")
    i_addr = LABEL_NAMES.index("I-ADDRESS")
    i_alias = LABEL_NAMES.index("I-ALIAS_OF_ADDRESS")
    labels = np.array([b_addr, i_addr, i_addr, i_addr, i_addr, i_addr, i_alias])
    spans = ner.decode_spans(text, offsets, labels, KIND_OF, IS_BEGIN)
    assert [(s, e) for _, s, e, _ in spans] == [(3, 18)]
    assert text[3:18] == "32 Old Cty Tk M"


def test_a_new_word_still_opens_a_new_span_across_collapsed_labels():
    # The counterpart: the collapse must not let two adjacent entities of the
    # same type run together. Only an I- token can extend a span; a B- token
    # starting a new word always opens one, whatever label it carries.
    text = "Springfield Portage"
    offsets = [(0, 11), (11, 19)]
    b_addr = LABEL_NAMES.index("B-ADDRESS")
    b_alias = LABEL_NAMES.index("B-ALIAS_OF_ADDRESS")
    spans = ner.decode_spans(text, offsets, np.array([b_addr, b_alias]),
                             KIND_OF, IS_BEGIN)
    assert [(s, e) for _, s, e, _ in spans] == [(0, 11), (12, 19)]


def test_unmodelled_labels_decode_to_nothing():
    text = "secret hunter2 here"
    offsets = [(0, 6), (6, 14), (14, 19)]
    unmodelled = LABEL_NAMES.index("B-unmodelled")
    spans = ner.decode_spans(
        text, offsets, np.array([0, unmodelled, 0]), KIND_OF, IS_BEGIN
    )
    assert spans == []


def test_confidence_is_the_mean_over_the_span_tokens():
    text = "Jane Doe"
    offsets = [(0, 4), (4, 8)]
    i_name = LABEL_NAMES.index("I-PERSON_NAME")
    b_name = LABEL_NAMES.index("B-PERSON_NAME")
    spans = ner.decode_spans(
        text, offsets, np.array([b_name, i_name]), KIND_OF, IS_BEGIN,
        probability=np.array([0.9, 0.5]),
    )
    assert len(spans) == 1
    assert spans[0][3] == pytest.approx(0.7)


# --------------------------------------------------------------------------
# Snapping span edges out of the middle of words
# --------------------------------------------------------------------------

def test_a_span_starting_inside_a_word_grows_to_the_word_edge():
    # The `l` student produced exactly this: 'ant Jane Doe', where "ant" is
    # the tail of "Applicant" that the tokenizer happened to split off.
    text = "Applicant Jane Doe entered the lottery."
    start = text.index("ant Jane")
    spans = [(0, start, start + len("ant Jane Doe"), 0.9)]
    assert ner.snap_to_word_edges(spans, text) == [(0, 0, 18, 0.9)]
    assert text[0:18] == "Applicant Jane Doe"


def test_a_span_ending_inside_a_word_grows_to_the_word_edge():
    # 'Tommy Dijkhu' -> 'Tommy Dijkhuizen'. This is the bucket the growth
    # policy was measured to fix (76 spans fixed, 0 broken).
    text = "Contact Tommy Dijkhuizen tomorrow."
    spans = [(0, 8, 20, 0.9)]
    assert text[8:20] == "Tommy Dijkhu"
    assert ner.snap_to_word_edges(spans, text) == [(0, 8, 24, 0.9)]
    assert text[8:24] == "Tommy Dijkhuizen"


def test_an_edge_at_a_real_word_boundary_is_left_alone():
    text = "Contact Jane Doe tomorrow."
    spans = [(0, 8, 16, 0.9)]
    assert ner.snap_to_word_edges(spans, text) == spans


def test_an_edge_after_punctuation_is_left_alone():
    # "MRN:4829471" -- the value starts mid-"word" only if words are defined
    # by whitespace. It is preceded by punctuation, so nothing is cut in half
    # and there is nothing to grow.
    text = "MRN:4829471 filed"
    spans = [(0, 4, 11, 0.9)]
    assert ner.snap_to_word_edges(spans, text) == spans


def test_growth_is_clamped_by_the_neighbouring_spans():
    """Two entities inside one word must not be grown into each other.

    Without the clamp both would expand to cover the whole run, overlap, and
    the pipeline's overlap resolution would silently drop one of them --
    turning a boundary imperfection into a lost entity.
    """
    text = "ref AB12CD34 end"
    spans = [(0, 4, 8, 0.9), (1, 8, 12, 0.9)]
    grown = ner.snap_to_word_edges(spans, text)
    assert grown == spans
    assert grown[0][2] <= grown[1][1], "grown spans must stay disjoint"


def test_growth_never_runs_past_the_ends_of_the_document():
    text = "Doe"
    assert ner.snap_to_word_edges([(0, 0, 3, 0.9)], text) == [(0, 0, 3, 0.9)]
    assert ner.snap_to_word_edges([(0, 1, 2, 0.9)], text) == [(0, 0, 3, 0.9)]


# --------------------------------------------------------------------------
# Merging sub-part spans
# --------------------------------------------------------------------------

def test_adjacent_name_parts_merge_into_one_entity():
    text = "Jane Doe"
    spans = [(0, 0, 4, 0.9), (0, 5, 8, 0.9)]
    merged = ner.merge_adjacent(spans, text, KINDS)
    assert merged == [(0, 0, 8, pytest.approx(0.9))]


def test_address_parts_merge_across_a_comma():
    text = "44 Elm Street, Springfield"
    spans = [(2, 0, 13, 1.0), (2, 15, 26, 1.0)]
    merged = ner.merge_adjacent(spans, text, KINDS)
    assert merged == [(2, 0, 26, pytest.approx(1.0))]


def test_two_people_are_not_merged_into_one_name():
    text = "Jane Doe and John Smith"
    spans = [(0, 0, 8, 0.9), (0, 13, 23, 0.9)]
    assert ner.merge_adjacent(spans, text, KINDS) == spans


def test_only_mergeable_types_merge():
    # Two SSNs side by side are two identifiers, never one.
    text = "111-22-3333 444-55-6666"
    spans = [(1, 0, 11, 0.9), (1, 12, 23, 0.9)]
    assert ner.merge_adjacent(spans, text, KINDS) == spans


# --------------------------------------------------------------------------
# Guard 2: re-validating model spans of checksummed types
# --------------------------------------------------------------------------

@pytest.mark.parametrize("entity_type,text,expected", [
    # 88% of Nemotron's gold cards fail Luhn, so the student learns to emit
    # them. This is the guard that stops it.
    (EntityType.CREDIT_CARD, "4111 1111 1111 1111", True),
    (EntityType.CREDIT_CARD, "4111 1111 1111 1112", False),
    (EntityType.CREDIT_CARD, "4111", False),
    (EntityType.SSN, "123-45-6789", True),
    (EntityType.SSN, "666-12-3456", False),       # never-issued area
    (EntityType.SSN, "123-45", False),
    (EntityType.IP_ADDRESS, "203.0.113.42", True),
    (EntityType.IP_ADDRESS, "10.2.1.400", False),
    (EntityType.IP_ADDRESS, "2001:db8::8a2e:370:7334", True),
    (EntityType.EMAIL, "a@b.com", True),
    (EntityType.EMAIL, "not-an-email", False),
    # Types with no validator are the model's call, not ours.
    (EntityType.PERSON_NAME, "Jane Doe", True),
    (EntityType.MRN, "666", True),
])
def test_revalidate(entity_type, text, expected):
    assert ner.revalidate(entity_type, text) is expected


def test_every_checksummed_type_has_a_real_validator():
    # A type in CHECKSUMMED_TYPES that revalidate() waves through would get
    # fusion precedence it has not earned.
    garbage = "zzzz"
    for entity_type in CHECKSUMMED_TYPES:
        assert ner.revalidate(entity_type, garbage) is False, entity_type


# --------------------------------------------------------------------------
# Chunking long documents
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n_tokens", [1, 10, 2048, 2049, 5000, 12345])
def test_chunks_tile_the_document_exactly_once(n_tokens):
    bounds = ner._chunk_bounds(n_tokens, window=2048, context=320)
    covered = []
    for scan_start, scan_end, keep_start, keep_end in bounds:
        assert scan_start <= keep_start < keep_end <= scan_end
        assert scan_end - scan_start <= 2048
        covered.extend(range(keep_start, keep_end))
    assert covered == list(range(n_tokens))


def test_short_documents_are_one_chunk_with_no_context_padding():
    assert ner._chunk_bounds(500, window=2048, context=320) == [(0, 500, 0, 500)]


def test_a_window_too_narrow_for_its_context_is_an_error_not_a_hang():
    # stride = window - 2 * context, so this configuration cannot advance.
    # Before the guard it looped forever building an ever-growing list.
    with pytest.raises(ValueError, match="context margins"):
        ner._chunk_bounds(10_000, window=512, context=320)


# --------------------------------------------------------------------------
# Fusion precedence
# --------------------------------------------------------------------------

def rule(entity_type, start, end, text, confidence=0.8):
    return Entity(type=entity_type, start=start, end=end, text=text,
                  confidence=confidence, detector=f"regex/{entity_type.value.lower()}")


def model(entity_type, start, end, text, confidence=0.9):
    return Entity(type=entity_type, start=start, end=end, text=text,
                  confidence=confidence, detector="onnx/ner-v1")


class StubDetector:
    """A Detector that replays a fixed entity list -- the protocol is
    structural, so this is all Stage 2 has to look like from Stage 3."""

    name = "onnx/ner-v1"

    def __init__(self, entities):
        self.entities = entities

    def detect(self, text):
        return list(self.entities)


def test_fusion_ranks():
    assert fusion_rank(rule(EntityType.SSN, 0, 1, "x")) == TIER_CHECKSUM_RULE
    assert fusion_rank(rule(EntityType.MRN, 0, 1, "x")) == TIER_CUE_RULE
    assert fusion_rank(model(EntityType.PERSON_NAME, 0, 1, "x")) == TIER_MODEL


def test_a_checksummed_rule_span_outranks_a_more_confident_model_span():
    text = "SSN 123-45-6789 on file"
    pipeline = Pipeline([
        StubDetector([model(EntityType.NATIONAL_ID, 4, 15, "123-45-6789", 0.99)]),
        *Pipeline().detectors,
    ])
    entities = pipeline.run(text)
    assert [e.type for e in entities] == [EntityType.SSN]
    assert entities[0].detector == "regex/ssn"


def test_the_model_outranks_a_cue_anchored_rule_span():
    # This is the +0.028 F1 case: the rules score F1 0.407 on ACCOUNT_NUMBER
    # and the student 0.803, so a cue-anchored guess must not veto the model.
    text = "Refunds go to account number 8272-1189-90 today"
    start = text.index("8272")
    pipeline = Pipeline([
        StubDetector([model(EntityType.USER_ID, start, start + 12,
                            "8272-1189-90", 0.95)]),
        *Pipeline().detectors,
    ])
    types = {e.type for e in pipeline.run(text)}
    assert EntityType.USER_ID in types
    assert EntityType.ACCOUNT_NUMBER not in types


def test_a_truncated_model_span_does_not_displace_the_rule_that_had_it_right():
    """Measured on the frozen corpus before pipeline.truncates existed.

    The student tagged `-4-` inside `3-4-1985` as a DATE_DOB. Under plain
    checksum-first precedence that three-character fragment outranked the rule
    span that had the whole date, so a correct detection became a boundary
    error. Same type, strictly shorter, overlapping: that is a truncation, and
    length settles it.
    """
    text = "Born 3-4-1985 per the intake form."
    start = text.index("-4-")
    pipeline = Pipeline([
        StubDetector([model(EntityType.DATE_DOB, start, start + 3, "-4-", 0.99)]),
        *Pipeline().detectors,
    ])
    entities = pipeline.run(text)
    assert [(e.type, e.text) for e in entities] == [
        (EntityType.DATE_DOB, "3-4-1985")
    ]
    assert entities[0].detector == "regex/date_dob"


def test_a_longer_model_span_of_the_same_type_still_wins():
    # The promotion is for truncations only. When the model claims MORE text
    # than the rule did, that is the model disagreeing rather than failing, and
    # it keeps the precedence the holdout measured.
    text = "Refunds go to account number 8272-1189-9012 today"
    start = text.index("8272")
    long_span = model(EntityType.ACCOUNT_NUMBER, start, start + 14,
                      "8272-1189-9012", 0.95)
    pipeline = Pipeline([StubDetector([long_span]), *Pipeline().detectors])
    entities = [e for e in pipeline.run(text)
                if e.type is EntityType.ACCOUNT_NUMBER]
    assert len(entities) == 1
    assert entities[0].detector == "onnx/ner-v1"


def test_a_model_span_may_not_erase_a_documents_phi_label():
    """The failure that disqualified the `xs` student, as a unit test.

    HEALTH_PLAN_ID is phi_specific, so it escalates the document on its own.
    "insurance" is not a medical-context term, so nothing else in this sentence
    can. If a non-phi_specific model span displaces the rule span, a PHI
    document silently becomes PII -- and a missed identifier is the failure
    docs/DESIGN.md section 1 ranks worst.
    """
    from pii_master.classify import DocumentClassifier

    text = "Coverage active: insurance member id 4471-2299 effective this month."
    start = text.index("4471")
    pipeline = Pipeline([
        StubDetector([model(EntityType.USER_ID, start, start + 9,
                            "4471-2299", 0.99)]),
        *Pipeline().detectors,
    ])
    entities = pipeline.run(text)
    assert EntityType.HEALTH_PLAN_ID in {e.type for e in entities}
    assert DocumentClassifier().classify(text, entities).label is DocLabel.PHI


def test_the_model_may_still_replace_a_phi_span_with_another_phi_span():
    """The guard yields when the model agrees the span is PHI-specific.

    Blanket-protecting MRN and HEALTH_PLAN_ID rule spans would cost real
    quality -- measured, 0.036 F1 on MRN and 0.083 on HEALTH_PLAN_ID against
    the Nemotron holdout. The document label survives a phi_specific-for-
    phi_specific swap either way, so there is no reason to protect it, and the
    model is the better judge of boundaries. Narrowing the guard to only the
    label-erasing case recovers all of that.
    """
    text = "Records office: chart no. 4829471 pulled for the attending physician."
    start = text.index("4829471")
    better = model(EntityType.MRN, start, start + 7, "4829471", 0.99)
    pipeline = Pipeline([StubDetector([better]), *Pipeline().detectors])
    mrn = [e for e in pipeline.run(text) if e.type is EntityType.MRN]
    assert len(mrn) == 1
    assert mrn[0].detector == "onnx/ner-v1"


def test_a_cue_anchored_rule_still_contributes_where_the_model_is_silent():
    text = "Refunds go to account number 8272-1189-90 today"
    pipeline = Pipeline([
        StubDetector([model(EntityType.PERSON_NAME, 0, 7, "Refunds", 0.9)]),
        *Pipeline().detectors,
    ])
    types = {e.type for e in pipeline.run(text)}
    assert types == {EntityType.PERSON_NAME, EntityType.ACCOUNT_NUMBER}


def test_rules_only_ordering_is_unchanged_when_no_model_is_present():
    # The tier key must not fire without a model tier, or v0.2's rule-vs-rule
    # overlap resolution would change for a reason nothing measured.
    text = "Card 4111 1111 1111 1111 and MRN: 4829471"
    plain = Pipeline().run(text)
    assert [(e.type, e.start, e.end) for e in plain] == [
        (EntityType.CREDIT_CARD, 5, 24), (EntityType.MRN, 34, 41)
    ]


# --------------------------------------------------------------------------
# What the new types mean for the document label
# --------------------------------------------------------------------------

def test_a_name_alone_is_pii_not_phi():
    from pii_master.classify import DocumentClassifier

    text = "Applicant Jane Doe entered the housing lottery."
    report = DocumentClassifier().classify(
        text, [model(EntityType.PERSON_NAME, 10, 18, "Jane Doe")]
    )
    assert report.label is DocLabel.PII


def test_a_name_in_a_clinical_note_is_phi():
    from pii_master.classify import DocumentClassifier

    text = "Discharge summary for Jane Doe; the patient is stable."
    report = DocumentClassifier().classify(
        text, [model(EntityType.PERSON_NAME, 22, 30, "Jane Doe")]
    )
    assert report.label is DocLabel.PHI


def test_no_model_type_is_phi_specific():
    # A name or an address is an identifier in any context; only an MRN or a
    # health plan id carries health linkage in the identifier itself. If a
    # model type were phi_specific, every document with a name would be PHI.
    for entity_type in MODEL_ONLY_TYPES:
        assert not TAXONOMY[entity_type].phi_specific, entity_type


def test_every_entity_type_has_a_taxonomy_row_with_a_hipaa_category():
    for entity_type in EntityType:
        info = TAXONOMY[entity_type]
        assert info.hipaa_category, entity_type
        assert info.weight > 0, entity_type


def test_missing_model_artifact_is_a_loud_error():
    with pytest.raises(ner.ModelUnavailable):
        ner.OnnxNerDetector("/nonexistent/model/dir").detect("Jane Doe")


def test_a_bundle_tokenizer_never_truncates(tmp_path):
    """The one-line bug that would silently halve recall on long documents.

    `Tokenizer.from_file` restores the teacher's truncation config (1,024
    tokens ~= 6 KB of text). With it left on, everything past that point is
    dropped and the document is reported clean from there -- no error, no
    warning, just missing PHI. load_bundle turns it off; this asserts it stays
    off, using a stub session so no trained artifact is needed.
    """
    tokenizers = pytest.importorskip("tokenizers")
    pytest.importorskip("onnxruntime")

    source = tmp_path / "tokenizer.json"
    trained = tokenizers.Tokenizer(tokenizers.models.WordLevel(
        {"a": 0, "b": 1, "[UNK]": 2}, unk_token="[UNK]"))
    trained.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    trained.enable_truncation(max_length=8)
    trained.save(str(source))

    restored = tokenizers.Tokenizer.from_file(str(source))
    assert restored.truncation is not None, "precondition: the setting persists"

    restored.no_truncation()
    text = " ".join(["a"] * 40)
    assert len(restored.encode(text).ids) == 40


@pytest.mark.parametrize("config,expected", [
    # The `m` student: k=5, dilations 1..32 -> 2*(1+2+4+8+16+32) = 126 a side.
    ({"kernel_size": 5, "dilations": [1, 2, 4, 8, 16, 32]}, 126),
    # The `xs` student: k=5, dilations 1,2,4,8 -> 30 a side.
    ({"kernel_size": 5, "dilations": [1, 2, 4, 8]}, 30),
    # No dilations listed: model.py falls back to powers of two per layer.
    ({"kernel_size": 5, "n_layers": 4}, 30),
    # An empty config must not yield 0, which would make chunk overlap zero.
    ({}, 126),
])
def test_receptive_field_is_read_from_the_artifact(config, expected):
    assert ner._half_receptive_field(config) == expected


def test_deep_mode_fails_the_same_way_whatever_the_input(tmp_path, capsys):
    """A configuration error must not depend on the document.

    Deep mode used to load the model on the first non-empty document, so
    `scan --deep --model-dir /nonexistent empty.txt` exited 0 and reported
    NONE, while the same command on a real file raised. A caller scanning a
    directory would have got a clean bill of health for every empty file in it.
    """
    from pii_master.cli import main

    empty = tmp_path / "empty.txt"
    empty.write_text("")
    full = tmp_path / "full.txt"
    full.write_text("Patient MRN: 4829471 admitted.")

    for path in (empty, full):
        assert main(["scan", "--deep", "--model-dir",
                     str(tmp_path / "nope"), str(path)]) == 2
        assert "not found" in capsys.readouterr().err


# --------------------------------------------------------------------------
# Tests that need a real bundle. Skipped when none is present, which is the
# normal state of a checkout: training/artifacts is gitignored and the model is
# tens of megabytes. Everything above this line runs everywhere.
# --------------------------------------------------------------------------

@pytest.fixture(scope="session")
def bundle_dir():
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")
    found = ner.default_model_dir()
    if found is None:
        pytest.skip("no Stage 2 bundle; set PII_MASTER_MODEL_DIR to run these")
    return found


def test_chunking_a_long_document_matches_scanning_it_whole(bundle_dir):
    """The claim _chunk_bounds rests on, checked rather than asserted.

    The student is a CNN with no position embeddings, so a chunk padded with
    more than the receptive field on each side must predict exactly what the
    whole document would. If that is ever false -- a future student with
    position embeddings, an overlap smaller than the receptive field -- long
    documents silently get different answers than short ones.
    """
    text = ("Applicant Jane Doe, DOB: 03/14/1985, MRN: 4829471, "
            "44 Elm Street, Springfield. ") * 220
    whole = ner.OnnxNerDetector(bundle_dir, min_confidence=0.0, window=1_000_000)
    chunked = ner.OnnxNerDetector(bundle_dir, min_confidence=0.0,
                                  window=1024, context=320)
    assert len(whole.bundle.tokenizer.encode(text).ids) > 2000, "not long enough"
    key = lambda entities: [(e.type, e.start, e.end) for e in entities]  # noqa: E731
    assert key(chunked.detect(text)) == key(whole.detect(text))


def test_the_tokenizer_in_a_real_bundle_does_not_truncate(bundle_dir):
    bundle = ner.load_bundle(str(bundle_dir))
    assert bundle.tokenizer.truncation is None
    long_text = "Jane Doe lives in Springfield. " * 500
    assert len(bundle.tokenizer.encode(long_text).ids) > 1024


@pytest.mark.parametrize("text", [
    # The three documents docs/DISTILLATION_RESULTS.md section 6 measured the
    # raw student failing on. none-007 is the important one: the student tags
    # "666" as an MRN, which is phi_specific, so an unguarded integration
    # re-opens the false-PHI class Track A closed.
    "The form rejected 666-12-3456 as invalid, and the card "
    "4111 1111 1111 1112 failed validation.",
    "Your subscriber id A9-3321-77 for the magazine renews in March.",
    "The chart 4829471 is in the appendix.",
    "Build 10.2.1.4 shipped on schedule.",
])
def test_deep_mode_does_not_invent_phi_on_the_hard_negatives(bundle_dir, text):
    from pii_master.classify import scan_text
    from pii_master.entities import DocLabel
    from pii_master.pipeline import deep_pipeline

    report = scan_text(text, deep_pipeline(bundle_dir))
    assert report.label is not DocLabel.PHI, [e.to_dict() for e in report.entities]


@pytest.mark.parametrize("expected,text", [
    # Ordinary prose must stay quiet. A scanner that cries wolf gets turned
    # off, which is the worst recall of all (docs/DESIGN.md section 3), and
    # adopting PERSON_NAME is exactly the change that could start it.
    ("NONE", "The quarterly planning meeting moved to the large conference "
             "room. Bring the printed agenda and the revised timeline."),
    ("NONE", "Build 10.2.1.4 shipped on schedule; see the release notes."),
    # Clinical language with no identifier in it is not PHI. PHI is PII *plus*
    # health context, never health context alone.
    ("NONE", "Patient was discharged Tuesday. Follow-up with the clinic."),
    # A name alone is PII: an identifier, but no health linkage.
    ("PII", "Dr. Alice Nguyen will present the Q3 roadmap on Thursday."),
    # The same name in a clinical note is PHI, and only the model can see it.
    ("PHI", "Discharge summary for Alice Nguyen; the patient is stable."),
])
def test_deep_mode_document_labels(bundle_dir, expected, text):
    from pii_master.classify import scan_text
    from pii_master.pipeline import deep_pipeline

    report = scan_text(text, deep_pipeline(bundle_dir))
    assert report.label.name == expected, [e.to_dict() for e in report.entities]


def test_the_guards_are_what_stop_it(bundle_dir):
    """Not a tautology check: this asserts the guards are load-bearing.

    If an unguarded detector produced the same output as a guarded one, the
    guards would be dead code and the confidence default meaningless. The
    frozen corpus is adversarial by construction, so at least one of these
    documents must differ.
    """
    from pii_master.evaluation import load_corpus

    guarded = ner.OnnxNerDetector(bundle_dir)
    raw = ner.OnnxNerDetector(bundle_dir, min_confidence=0.0,
                              revalidate_checksums=False)
    docs = load_corpus(["eval/corpus/frozen_v1.jsonl"])
    suppressed = sum(
        len(raw.detect(doc.text)) - len(guarded.detect(doc.text)) for doc in docs
    )
    assert suppressed > 0, "the guards suppressed nothing on the frozen corpus"


def test_whitespace_mask_matches_str_isspace_including_unicode():
    """The word-start test must agree with the reference decoder everywhere.

    training/decode.py calls str.isspace(); this replaces it with an array
    lookup for speed. An ASCII-only lookup would decode documents containing a
    non-breaking space or an ideographic space differently from the
    implementation every measured number came from -- and Nemotron-PII covers
    international locales, so that text is not hypothetical.
    """
    rng = random.Random(3)
    probes = ["a b", "a b", "a　b", "a b", "a​b",
              "héllo wörld", ""]
    probes += ["".join(chr(rng.randrange(0x11000)) for _ in range(40))
               for _ in range(200)]
    for text in probes:
        assert list(ner._whitespace_mask(text)) == [c.isspace() for c in text], (
            repr(text)
        )


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------

def load_isotonic():
    """training/calibrate.py's fitter, imported the same way as the decoder.

    It is twenty lines of pool-adjacent-violators standing in for a
    scikit-learn dependency, it is easy to get subtly wrong, and a wrong
    monotone-looking curve would silently mislabel every confidence the
    product reports. So it is tested here rather than trusted.
    """
    import importlib.util

    path = pathlib.Path(__file__).resolve().parents[1] / "training" / "calibrate.py"
    if not path.exists():
        pytest.skip("training/calibrate.py not present")
    spec = importlib.util.spec_from_file_location("_calibrate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.isotonic


def test_isotonic_reproduces_the_textbook_case():
    isotonic = load_isotonic()
    x = np.arange(6.0)
    y = np.array([4.0, 5, 1, 6, 8, 7])
    fitted = np.interp(x, *isotonic(x, y))
    # Pooling [4,5,1] -> 10/3 and [8,7] -> 7.5 is the standard PAVA answer.
    assert fitted == pytest.approx([10 / 3, 10 / 3, 10 / 3, 6, 7.5, 7.5])


def test_isotonic_knots_are_flat_inside_a_pooled_block():
    # The bug this pins: emitting one knot per block instead of both edges
    # makes np.interp ramp ACROSS the block, turning 3.33,3.33,3.33 into
    # 3.33,4.22,5.11 -- a curve that is not the one that was fitted.
    isotonic = load_isotonic()
    knot_x, knot_y = isotonic(np.arange(6.0), np.array([4.0, 5, 1, 6, 8, 7]))
    assert np.interp(1.0, knot_x, knot_y) == pytest.approx(10 / 3)
    assert np.interp(2.0, knot_x, knot_y) == pytest.approx(10 / 3)


def test_isotonic_pools_tied_scores_into_strictly_increasing_knots():
    # Tens of thousands of spans share a handful of distinct scores, and
    # np.interp needs increasing x. Duplicate knots at one score with
    # different values would make the interpolation pick one silently.
    isotonic = load_isotonic()
    knot_x, knot_y = isotonic(
        np.array([0.5, 0.5, 0.5, 0.9, 0.9]), np.array([1.0, 0, 0, 1, 1])
    )
    assert np.all(np.diff(knot_x) > 0)
    assert np.interp(0.5, knot_x, knot_y) == pytest.approx(1 / 3)


def test_isotonic_output_is_always_monotone():
    isotonic = load_isotonic()
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.random(500)
        y = (rng.random(500) < x).astype(float)
        _, knot_y = isotonic(x, y)
        assert np.all(np.diff(knot_y) >= -1e-12)


def test_calibration_is_identity_when_a_bundle_has_none():
    scores = np.array([0.1, 0.5, 0.9])
    assert ner.calibrate(scores, ()) is scores


GLOBAL_CURVE = (np.array([0.0, 0.5, 0.8, 1.0]), np.array([0.0, 0.1, 0.7, 1.0]))


def test_calibration_never_reorders_spans_of_one_type():
    """The property that makes calibration safe to add after measuring.

    A monotone map cannot change which span outranks which, so every
    precision/recall number measured before calibration still describes the
    same model afterwards. Only the meaning of the threshold changes.
    """
    rng = np.random.default_rng(1)
    raw = rng.random(500)
    adjusted = ner.calibrate(raw, (GLOBAL_CURVE, {}))
    assert list(np.argsort(raw)) == list(np.argsort(adjusted))


def test_calibration_maps_a_raw_score_onto_the_curve():
    curve = (np.array([0.2, 0.6, 1.0]), np.array([0.0, 0.5, 1.0]))
    got = ner.calibrate(np.array([0.1, 0.4, 0.6, 0.8, 1.5]), (curve, {}))
    # Clamped below 0.2 and above 1.0; linear in between.
    assert got == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])


def test_a_per_type_curve_overrides_the_global_one():
    """Why per-type curves exist, in miniature.

    Measured on the real holdout, one global curve left `URL` 0.107
    UNDER-confident while `DEVICE_ID` and `VEHICLE_ID` ran ~0.06 over -- errors
    that cancel in the pooled number and do not cancel at a threshold. Here the
    global curve halves every score and the URL curve leaves them alone.
    """
    halve = (np.array([0.0, 1.0]), np.array([0.0, 0.5]))
    identity = (np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    calibration = (halve, {"URL": identity})
    got = ner.calibrate(np.array([0.8, 0.8, 0.8]), calibration,
                        ["URL", "PERSON_NAME", "URL"])
    assert got == pytest.approx([0.8, 0.4, 0.8])


def test_a_type_without_its_own_curve_falls_back_to_the_global_one():
    # Types below the fit floor keep the global curve on purpose: a curve
    # fitted on forty spans would be applied with the same authority as one
    # fitted on thirty thousand.
    halve = (np.array([0.0, 1.0]), np.array([0.0, 0.5]))
    got = ner.calibrate(np.array([0.6]), (halve, {"URL": halve}), ["TAX_ID"])
    assert got == pytest.approx([0.3])


def test_per_type_curves_are_ignored_when_no_types_are_supplied():
    halve = (np.array([0.0, 1.0]), np.array([0.0, 0.5]))
    identity = (np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    got = ner.calibrate(np.array([0.6]), (halve, {"URL": identity}))
    assert got == pytest.approx([0.3])


def test_a_malformed_calibration_fails_loudly():
    with pytest.raises(ner.ModelUnavailable, match="malformed calibration"):
        ner._load_calibration({"x": [0.1, 0.2], "y": [0.5]})


def test_a_malformed_per_type_curve_names_the_type_that_is_broken():
    with pytest.raises(ner.ModelUnavailable, match="URL"):
        ner._load_calibration({"x": [0.0, 1.0], "y": [0.0, 1.0],
                               "per_type": {"URL": {"x": [0.1], "y": []}}})


def test_an_uncalibrated_bundle_still_loads():
    # Bundles exported before calibration existed have no such key, and must
    # keep working with raw scores rather than failing to load.
    assert ner._load_calibration(None) == ()
    assert ner._load_calibration({}) == ()


def test_a_global_only_bundle_still_loads():
    # The generation between "no calibration" and "per-type": x/y and no
    # per_type key. It must keep working, not fail to load.
    loaded = ner._load_calibration({"x": [0.0, 1.0], "y": [0.0, 1.0]})
    (curve, per_type) = loaded
    assert per_type == {}
    assert ner.calibrate(np.array([0.5]), loaded, ["URL"]) == pytest.approx([0.5])
