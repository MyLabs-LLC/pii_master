"""Unit tests for the Stage 2 training helpers.

These skip unless the training extras are installed (`pip install -e ".[train]"`),
so CI's zero-dependency install still runs the Stage 1 suite unchanged. What is
covered here is the part of distillation that fails *silently* when it is wrong:
label alignment against the teacher, the BIO convention, and the decoder that
has to invert it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

from data import (  # noqa: E402
    ID2LABEL,
    LABEL2ID,
    NEMOTRON_LABELS,
    NUM_LABELS,
    first_word_end,
    word_start_index,
    word_start_mask,
)
from decode import decode_spans  # noqa: E402
from train import (  # noqa: E402
    ABSENT_LOGIT,
    broadcast_to_words,
    build_label_permutation,
    distillation_loss,
    remap_teacher_logits,
)

TEACHER = {0: "O", **{i + 1: f"{p}-{t}" for i, (t, p) in enumerate(
    [(t, p) for t in NEMOTRON_LABELS for p in ("B", "I")])}}


def test_label_space_is_55_types_plus_o():
    assert len(NEMOTRON_LABELS) == 55
    assert NUM_LABELS == 111 == len(set(ID2LABEL.values()))
    assert LABEL2ID["O"] == 0
    assert NEMOTRON_LABELS == sorted(NEMOTRON_LABELS)


@pytest.mark.parametrize("text,start,end,expected", [
    ("SSN 123-45-6789 here", 4, 15, 15),            # one word: no split
    ("at 4821 Maple Avenue,", 3, 20, 7),            # split after "4821"
    ("x  spaced out", 1, 13, 9),                    # leading space is skipped
    ("trailing ", 0, 9, 8),
])
def test_first_word_end(text, start, end, expected):
    assert first_word_end(text, start, end) == expected


def test_word_start_index_runs_to_the_previous_word_start():
    mask = np.array([False, True, False, False, True, True, False])
    assert list(word_start_index(mask)) == [0, 1, 1, 1, 4, 5, 5]


def test_word_start_mask_marks_the_first_subword_of_each_word():
    text = "Patient Jane Doe"
    offsets = np.array([[0, 0], [0, 7], [7, 12], [12, 16], [0, 0]])
    real = np.array([False, True, True, True, False])
    assert list(word_start_mask(text, offsets, real)) == [False, True, True, True, False]


def _labels(names):
    return [LABEL2ID[n] for n in names]


def test_decode_merges_consecutive_b_inside_one_word():
    # "123-45-6789" is five B-ssn tokens and exactly one span.
    text = "SSN 123-45-6789 ok"
    offsets = np.array([[0, 3], [3, 7], [7, 8], [8, 10], [10, 11], [11, 15], [15, 18]])
    labels = _labels(["O", "B-ssn", "B-ssn", "B-ssn", "B-ssn", "B-ssn", "O"])
    assert decode_spans(text, offsets, labels, ID2LABEL) == [("ssn", 4, 15)]


def test_decode_splits_two_same_type_spans_at_a_word_boundary():
    text = "a@x.com b@y.com"
    offsets = np.array([[0, 7], [7, 15]])
    labels = _labels(["B-email", "B-email"])
    assert decode_spans(text, offsets, labels, ID2LABEL) == [
        ("email", 0, 7), ("email", 8, 15)]


def test_decode_continues_a_multiword_span_through_i_labels():
    text = "at 4821 Maple Ave"
    offsets = np.array([[0, 2], [2, 7], [7, 13], [13, 17]])
    labels = _labels(["O", "B-street_address", "I-street_address", "I-street_address"])
    assert decode_spans(text, offsets, labels, ID2LABEL) == [("street_address", 3, 17)]


def test_permutation_is_identity_when_orders_agree():
    perm, n_teacher = build_label_permutation(dict(enumerate(
        [ID2LABEL[i] for i in range(NUM_LABELS)])))
    assert n_teacher == NUM_LABELS
    assert torch.equal(perm, torch.arange(NUM_LABELS))


def test_permutation_follows_a_shuffled_teacher_by_name():
    shuffled = {i: name for i, name in enumerate(
        reversed([ID2LABEL[i] for i in range(NUM_LABELS)]))}
    perm, n_teacher = build_label_permutation(shuffled)
    for ours in range(NUM_LABELS):
        assert shuffled[int(perm[ours])] == ID2LABEL[ours]


def test_permutation_pads_a_prefix_variant_the_teacher_lacks():
    # ettin-68m is exactly this shape: B-ssn but no I-ssn, because it tags at
    # word level and an SSN is never two words.
    narrow = {i: name for i, name in enumerate(
        n for n in (ID2LABEL[i] for i in range(NUM_LABELS)) if n != "I-ssn")}
    perm, n_teacher = build_label_permutation(narrow)
    assert int(perm[LABEL2ID["I-ssn"]]) == n_teacher      # the appended column
    remapped = remap_teacher_logits(torch.zeros(1, 1, n_teacher), perm)
    assert remapped.shape[-1] == NUM_LABELS
    assert remapped[0, 0, LABEL2ID["I-ssn"]].item() == ABSENT_LOGIT
    assert remapped[0, 0, LABEL2ID["B-ssn"]].item() == 0.0


def test_permutation_hard_fails_when_a_type_is_missing_entirely():
    without_ssn = {i: name for i, name in enumerate(
        n for n in (ID2LABEL[i] for i in range(NUM_LABELS))
        if not n.endswith("-ssn"))}
    with pytest.raises(SystemExit):
        build_label_permutation(without_ssn)


def test_broadcast_copies_word_start_logits_across_the_word():
    logits = torch.arange(24.0).reshape(1, 4, 6)
    word_src = torch.tensor([[0, 0, 2, 2]])
    out = broadcast_to_words(logits, word_src)
    assert torch.equal(out[0, 1], logits[0, 0])
    assert torch.equal(out[0, 3], logits[0, 2])


def test_absent_column_does_not_produce_nan_loss():
    student = torch.randn(2, 4, NUM_LABELS, requires_grad=True)
    teacher = torch.randn(2, 4, NUM_LABELS)
    teacher[..., LABEL2ID["I-ssn"]] = ABSENT_LOGIT
    labels = torch.full((2, 4), LABEL2ID["B-ssn"])
    labels[0, 0] = -100
    loss, soft, hard = distillation_loss(student, teacher, labels, 0.7, 3.0)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(student.grad).all()


def test_masked_scope_falls_back_to_cross_entropy_when_nothing_is_trusted():
    student = torch.randn(1, 3, NUM_LABELS)
    teacher = torch.randn(1, 3, NUM_LABELS)
    labels = torch.full((1, 3), LABEL2ID["O"])
    loss, soft, hard = distillation_loss(
        student, teacher, labels, 0.7, 3.0, torch.zeros(1, 3, dtype=torch.bool))
    assert soft == 0.0
    assert torch.isfinite(loss)
