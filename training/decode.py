"""Token labels -> character spans, the exact inverse of data.py's labelling.

data.py tags BIO at word level: every subword of a span's first word is B-,
every subword of a later word is I-. Decoding therefore cannot treat each B-
as a new span -- "123-45-6789" is five consecutive B-ssn tokens and one span.
The rule that inverts it exactly:

    a token opens a new span only when it is B- AND it starts a new word,
    or when the type changes; otherwise it extends the open span.

Boundaries come from the tokenizer offsets, with leading whitespace stripped
(offsets include the space: " Jane" is [7,12] while the gold span is [8,12]).
Decoding from every token rather than word starts is what keeps boundaries
exact for identifiers glued to punctuation ("6789," splits into two tokens,
and the comma is O).
"""

from __future__ import annotations


def starts_word(text: str, start: int) -> bool:
    return start == 0 or text[start].isspace() or text[start - 1].isspace()


def decode_spans(text, offsets, label_ids, id2label):
    """-> [(label, start, end)] in document order."""
    spans: list[tuple[str, int, int]] = []
    open_type: str | None = None
    open_start = open_end = 0

    def close():
        nonlocal open_type
        if open_type is not None:
            spans.append((open_type, open_start, open_end))
            open_type = None

    for index, (a, b) in enumerate(offsets):
        a, b = int(a), int(b)
        if b <= a:                      # special token / padding
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
