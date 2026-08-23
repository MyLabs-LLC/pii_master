"""BIO span decode — the inverse of the word-level labelling in training/data.py.

Kept here so the serving path does not import ``training/``. The rule is the
one measured in docs/DISTILLATION_RESULTS.md section 2: a token opens a new
span only when it is B- *and* it starts a new word, or when the type changes.
"""

from __future__ import annotations


def starts_word(text: str, start: int) -> bool:
    return start == 0 or text[start].isspace() or text[start - 1].isspace()


def decode_spans(text, offsets, label_ids, id2label):
    """Return ``[(label, start, end)]`` in document order."""
    spans: list[tuple[str, int, int]] = []
    open_type: str | None = None
    open_start = open_end = 0

    def close() -> None:
        nonlocal open_type
        if open_type is not None:
            spans.append((open_type, open_start, open_end))
            open_type = None

    for index, (a, b) in enumerate(offsets):
        a, b = int(a), int(b)
        if b <= a:
            continue
        name = id2label[int(label_ids[index])]
        if name == "O":
            close()
            continue
        prefix, _, kind = name.partition("-")
        fresh = open_type != kind or (prefix == "B" and starts_word(text, a))
        if fresh:
            close()
            start = a
            while start < b and text[start].isspace():
                start += 1
            open_type, open_start, open_end = kind, start, b
        else:
            open_end = b
    close()
    return spans
