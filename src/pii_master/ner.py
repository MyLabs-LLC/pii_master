"""Stage 2: the distilled CNN tagger, wrapped as a ``Detector``.

This is the serving half of docs/DISTILLATION_PLAN.md. Training lives in
``training/``; nothing in that directory is imported here, and nothing here is
imported by the default rules-only path -- ``onnxruntime`` and ``tokenizers``
are an optional extra (``pip install pii-master[ml]``), which is what keeps the
zero-dependency ``fast`` mode honest.

**What the model adds.** The rules tier cannot find names, addresses, or
cue-free identifiers, and on Nemotron's ``certificate_license_number`` it scores
F1 0.001. The student covers those: measured F1 0.901 fused against the rules'
0.788 on a 100k-document holdout (docs/DISTILLATION_RESULTS.md section 5).

**What the model gets wrong, and the three guards that ship because of it.**
Run raw over the frozen corpus, the student tags ``666`` as an MRN inside "the
form rejected 666-12-3456 as invalid" -- a false PHI, the exact failure class
Track A of docs/IMPROVEMENT_PLAN.md closed in the rules. It also truncates:
``MRN: 4829471`` decodes as ``4829`` plus a stray ``national_id`` for ``471``.
Those are not hypotheticals, they were measured (DISTILLATION_RESULTS section
6, gate 3), and integration is only safe because of:

1. ``min_confidence`` -- a span whose mean per-token probability is below the
   threshold is dropped. Truncated and hallucinated spans are the low-confidence
   tail; this is the knob docs/DESIGN.md section 12 left as an open question,
   answered here as a library default that policy can override.
2. ``revalidate`` -- a model span of a checksummed type must pass the same
   validator the rules use. 88% of Nemotron's gold cards fail Luhn, so a
   student trained on it *learns to emit invalid cards*; without this the model
   re-opens a false-positive class the rules were built to close.
3. Fusion precedence, which lives in ``pipeline.py`` rather than here: a
   checksum-validated rule span outranks the model, and the model outranks a
   cue-anchored rule guess. Measured worth: +0.028 F1 over blanket rule
   precedence.

**Why a CNN and not a transformer.** A transformer body costs ~12*L*d^2 MACs
per token; on one core, even the smallest model in the Ettin ladder reads about
five tokens of a two-thousand-token document. The dilated depthwise-separable
convolution used here costs ~d^2 + k*d, about 100x less, and reads the whole
document. Full arithmetic in docs/DISTILLATION_PLAN.md section 1.

That architecture has a second consequence this module depends on: **the model
has no position embeddings**, so its output at a token depends only on the ~253
tokens around it. Long documents can therefore be chunked with overlap and the
result is identical to running the whole document at once, which is what
:func:`_chunk_bounds` exploits.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .crosswalk import to_entity_type
from .entities import CHECKSUMMED_TYPES, EntityType
from .models import Entity
from .validators import ipv4_ok, ipv6_ok, luhn_ok, ssn_ok

MODEL_DIR_ENV = "PII_MASTER_MODEL_DIR"

#: Types whose Nemotron labels are sub-parts of one real-world identifier:
#: "Jane"+"Doe" and "44 Elm Street"+"Springfield" arrive as adjacent spans and
#: are reported as one entity. Merging only ever joins spans separated by
#: whitespace or a comma, so "Jane Doe and John Smith" stays two names.
MERGEABLE_TYPES: frozenset[EntityType] = frozenset({
    EntityType.PERSON_NAME,
    EntityType.ADDRESS,
})

_MERGE_GAP = re.compile(r"^[\s,]{0,2}$")


class ModelUnavailable(RuntimeError):
    """The optional ML extra or the model artifact is missing."""


def revalidate(entity_type: EntityType, text: str) -> bool:
    """Re-run our own validator on a model-proposed span.

    Guard 2 in the module docstring. A type with no validator passes through --
    this rejects wrong *facts*, it does not second-guess the model's judgement.
    """
    if entity_type is EntityType.CREDIT_CARD:
        digits = "".join(c for c in text if c.isdigit())
        return 13 <= len(digits) <= 19 and luhn_ok(digits)
    if entity_type is EntityType.SSN:
        digits = "".join(c for c in text if c.isdigit())
        return len(digits) == 9 and ssn_ok(digits[:3], digits[3:5], digits[5:])
    if entity_type is EntityType.IP_ADDRESS:
        stripped = text.strip()
        return ipv4_ok(stripped) or ipv6_ok(stripped)
    if entity_type is EntityType.EMAIL:
        return "@" in text and "." in text.rsplit("@", 1)[-1]
    if entity_type is EntityType.URL:
        return "." in text or ":" in text
    return True


# --------------------------------------------------------------------------
# Span decoding
# --------------------------------------------------------------------------

# Every codepoint for which str.isspace() is True. There are 29 and the
# highest is U+3000, so a 0x3001-entry lookup table covers the whole set and
# anything above it is definitively not whitespace.
_SPACE_CODEPOINTS = (
    0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x85, 0xA0,
    0x1680, 0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007,
    0x2008, 0x2009, 0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000,
)
_SPACE_TABLE = None


def _space_table():
    global _SPACE_TABLE
    if _SPACE_TABLE is None:
        import numpy as np

        table = np.zeros(0x3001, dtype=bool)
        table[list(_SPACE_CODEPOINTS)] = True
        _SPACE_TABLE = table
    return _SPACE_TABLE


def _whitespace_mask(text: str):
    """Bool array over the document, True where ``str.isspace()`` is True.

    Built once per document so the word-start test in :func:`decode_spans` is
    an array index rather than a Python method call per token.

    It matches ``str.isspace()`` exactly, not just ASCII space. The reference
    decoder in ``training/decode.py`` calls ``str.isspace()``, and Nemotron-PII
    covers international locales -- a document using a non-breaking space or an
    ideographic space would otherwise decode differently here than in the
    implementation every measured number came from, on text where nobody would
    think to look.
    """
    import numpy as np

    # errors="replace" keeps a lone surrogate -- which a surrogateescape file
    # read can produce -- from raising here. It substitutes one codepoint for
    # one codepoint, so offsets still line up, and neither a surrogate nor its
    # replacement is whitespace, so the mask is unchanged.
    raw = np.frombuffer(text.encode("utf-32-le", "replace"), dtype=np.uint32)
    mask = np.zeros(raw.shape[0], dtype=bool)
    low = raw <= 0x3000
    mask[low] = _space_table()[raw[low]]
    return mask


def decode_spans(text, offsets, label_ids, kind_of, is_begin, probability=None):
    """Token labels -> ``[(kind_index, start, end, confidence)]``.

    The vectorised inverse of ``training/data.py``'s labelling, which tags BIO
    at *word* level: every subword of a span's first whitespace-delimited word
    gets ``B-``, later words get ``I-``. Decoding therefore cannot treat every
    ``B-`` as a new span -- "123-45-6789" is five consecutive ``B-ssn`` tokens
    and one entity. The rule that inverts it exactly:

        a token opens a new span only when its type changes, when the previous
        token was not part of a span, or when it is ``B-`` AND it starts a new
        whitespace-delimited word; otherwise it extends the open span.

    ``training/decode.py`` is the readable reference implementation of the same
    rule; ``tests/test_ner.py`` asserts the two agree token for token. This one
    is vectorised because the loop version measured 1.08 ms on a 10 KB
    document, which was more than the model's own forward pass
    (docs/DISTILLATION_RESULTS.md section 4).

    **One intended difference from the reference.** It runs on Nemotron labels;
    this runs on ``kind_of[label]``, i.e. after the crosswalk. So where two
    different labels collapse to one of our types, the reference closes a span
    and opens another and this one continues. Measured on 400 real documents
    it happens twice in 1,985 spans, always the same shape -- the model tags a
    token mid-address as ``I-city`` inside a ``street_address`` run:

        ' T'  I-street_address
        'k'   I-street_address
        ' M'  I-city              <- reference splits here, we do not

    Continuing is the better answer: both labels are ADDRESS, and the model
    itself said "continuation" by emitting ``I-``. It cannot run away, because
    only an ``I-`` token can extend a span this way -- a ``B-`` token that
    starts a new word always opens a new one, so two addresses side by side
    stay two spans. This is a *different mechanism* from
    :func:`merge_adjacent`, which joins whole spans across a separator and is
    restricted to :data:`MERGEABLE_TYPES`; this one is inside a single BIO run
    and applies to any type.

    ``kind_of[label_id]`` is the entity index for that label or -1 (``O``, or a
    label we deliberately do not model). ``probability[i]`` is the model's
    max-class probability at token i; the returned confidence is its mean over
    the span's tokens.
    """
    import numpy as np

    offsets = np.asarray(offsets)
    if offsets.size == 0:
        return []
    starts, ends = offsets[:, 0], offsets[:, 1]
    real = ends > starts                      # drops specials and padding
    if not real.any():
        return []

    label_ids = np.asarray(label_ids)[real]
    starts, ends = starts[real], ends[real]
    kind = kind_of[label_ids]
    inside = kind >= 0
    if not inside.any():
        return []

    ws = _whitespace_mask(text)
    # A token starts a word when it is at offset 0, or the character at or
    # before its start is whitespace. Tokenizer offsets include the leading
    # space (" Jane" -> [7, 12]), hence the two-sided test.
    at_zero = starts == 0
    prev = np.clip(starts - 1, 0, max(len(ws) - 1, 0))
    here = np.clip(starts, 0, max(len(ws) - 1, 0))
    starts_word = (at_zero | ws[here] | ws[prev]) if len(ws) else at_zero

    is_b = is_begin[label_ids]
    # Break BEFORE position i when the type changes (which includes leaving or
    # entering `O`, since O is -1) or when a B- token starts a fresh word.
    changed = np.empty(kind.shape, dtype=bool)
    changed[0] = True
    changed[1:] = kind[1:] != kind[:-1]
    opens = inside & (changed | (is_b & starts_word))

    index = np.nonzero(inside)[0]
    group = np.cumsum(opens[index]) - 1
    n_groups = int(group[-1]) + 1

    first = np.zeros(n_groups, dtype=np.int64)
    last = np.zeros(n_groups, dtype=np.int64)
    # index is ascending, so the last write per group wins for `last` and the
    # first-seen assignment wins for `first`.
    first[group[::-1]] = index[::-1]
    last[group] = index

    if probability is not None:
        probability = np.asarray(probability)[real][index]
        totals = np.bincount(group, weights=probability, minlength=n_groups)
        counts = np.bincount(group, minlength=n_groups)
        means = totals / np.maximum(counts, 1)
    else:
        means = np.ones(n_groups)

    spans = []
    for g in range(n_groups):
        begin, stop = int(starts[first[g]]), int(ends[last[g]])
        # Offsets include the leading space; the gold span does not.
        while begin < stop and text[begin].isspace():
            begin += 1
        while stop > begin and text[stop - 1].isspace():
            stop -= 1
        if stop > begin:
            spans.append((int(kind[first[g]]), begin, stop, float(means[g])))
    return spans


def _word_char(character: str) -> bool:
    return character.isalnum() or character == "_"


def snap_to_word_edges(spans, text):
    """Grow span edges that fall inside a word out to the word boundary.

    A subword tokenizer can put a span edge in the middle of a word, and the
    model's per-token decisions then produce spans like ``'ant Jane Doe'``
    (inside "Applicant"), ``'Tommy Dijkhu'`` or ``'amela Navas'``. Those are
    artefacts of where the tokenizer happened to split, not judgements the
    model is making about the text.

    The justification is a property of the data, not a hunch: of 103,681
    crosswalked gold spans across 20,000 Nemotron documents, **8 begin and 49
    end inside an alphanumeric run** -- 0.055%, and concentrated in a couple of
    labels that look like annotation noise. An identifier span essentially
    never cuts a word in half.

    Grow rather than shrink, which is also measured. On 2,000 holdout
    documents growing fixed 76 spans and broke **0**; shrinking fixed 0 and
    broke 0, because a truncated span that loses its first characters is
    already wrong and dropping more of it does not help.

    Edges are clamped against the neighbouring spans so growing can never make
    two spans overlap -- two entities inside one word would otherwise collide
    and the pipeline would silently drop one of them.
    """
    grown: list[tuple[int, int, int, float]] = []
    limit = len(text)
    for index, (kind, start, end, confidence) in enumerate(spans):
        floor = grown[-1][2] if grown else 0
        ceiling = spans[index + 1][1] if index + 1 < len(spans) else limit
        while start > floor and _word_char(text[start - 1]) and _word_char(text[start]):
            start -= 1
        while end < ceiling and _word_char(text[end - 1]) and _word_char(text[end]):
            end += 1
        grown.append((kind, start, end, confidence))
    return grown


def merge_adjacent(spans, text, kinds):
    """Join same-type spans separated only by whitespace or a comma.

    Nemotron tags "Jane Doe" as first_name + last_name and "44 Elm Street,
    Springfield" as street_address + city; both map onto one of our types, and
    one entity is what a report should show. The gap test is deliberately tight
    -- "Jane Doe and John Smith" has " and " between the names and stays two
    entities.
    """
    merged: list[tuple[int, int, int, float]] = []
    for span in spans:
        kind, start, end, conf = span
        if merged:
            p_kind, p_start, p_end, p_conf = merged[-1]
            if (p_kind == kind
                    and kinds[kind] in MERGEABLE_TYPES
                    and _MERGE_GAP.match(text[p_end:start])):
                width, p_width = end - start, p_end - p_start
                blended = (conf * width + p_conf * p_width) / (width + p_width)
                merged[-1] = (kind, p_start, end, blended)
                continue
        merged.append(span)
    return merged


# --------------------------------------------------------------------------
# The artifact
# --------------------------------------------------------------------------

def default_model_dir() -> Path | None:
    """First existing model directory, or None.

    Search order: ``$PII_MASTER_MODEL_DIR``, then the user cache, then the
    in-repo training output so a developer who has just trained a student can
    use it with no configuration.
    """
    candidates = []
    if os.environ.get(MODEL_DIR_ENV):
        candidates.append(Path(os.environ[MODEL_DIR_ENV]))
    candidates.append(Path.home() / ".cache" / "pii_master" / "model")
    repo = Path(__file__).resolve().parents[2] / "training" / "artifacts"
    candidates.extend([repo / "bundle", repo])
    for path in candidates:
        if (path / "model.onnx").exists() or (path / "tokenizer.json").exists():
            return path
    return None


@dataclass(frozen=True)
class ModelBundle:
    """An ONNX session, its tokenizer, and the label tables derived from both."""

    session: object
    tokenizer: object
    kinds: tuple[EntityType, ...]     # kind index -> our type
    kind_of: object                   # np.ndarray[label_id] -> kind index or -1
    is_begin: object                  # np.ndarray[label_id] -> bool
    input_names: tuple[str, ...]
    half_field: int                   # half the model's receptive field, tokens
    calibration: tuple                # (x, y) isotonic knots, or () if raw
    path: Path


@lru_cache(maxsize=4)
def load_bundle(model_dir: str, threads: int = 1) -> ModelBundle:
    """Load (and cache) the ONNX session and tokenizer for a model directory.

    Cached because a session costs tens of milliseconds to create -- thousands
    of documents' worth of budget -- and the pipeline is constructed per scan
    in the current API (docs/IMPROVEMENT_PLAN.md section 3.5).

    ``intra_op_num_threads=1`` is not a default worth overriding lightly: the
    production container is a 1-core cgroup, where thread contention makes
    ONNX Runtime slower, not faster.
    """
    try:
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer
    except ImportError as exc:                              # pragma: no cover
        raise ModelUnavailable(
            "the Stage 2 detector needs the optional ML extra: "
            "pip install 'pii-master[ml]'"
        ) from exc

    path = Path(model_dir)
    onnx_path = path / "model.onnx"
    tokenizer_path = path / "tokenizer.json"
    meta_path = path / "model.json"
    for required in (onnx_path, tokenizer_path, meta_path):
        if not required.exists():
            raise ModelUnavailable(
                f"{required} not found. Train and export a student "
                f"(training/export.py --bundle) or set ${MODEL_DIR_ENV}."
            )

    meta = json.loads(meta_path.read_text())
    label_names = meta["label_names"]
    half_field = _half_receptive_field(meta.get("config", {}))
    calibration = _load_calibration(meta.get("calibration"))

    kinds: list[EntityType] = []
    seen: dict[EntityType, int] = {}
    kind_of = np.full(len(label_names), -1, dtype=np.int64)
    is_begin = np.zeros(len(label_names), dtype=bool)
    for label_id, name in enumerate(label_names):
        if name == "O":
            continue
        prefix, _, nemotron_label = name.partition("-")
        is_begin[label_id] = prefix == "B"
        entity = to_entity_type(nemotron_label)
        if entity is None:                  # deliberately unmodelled -> O
            continue
        if entity not in seen:
            seen[entity] = len(kinds)
            kinds.append(entity)
        kind_of[label_id] = seen[entity]

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(onnx_path), options, providers=["CPUExecutionProvider"]
    )
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    # A saved tokenizer carries the teacher's truncation setting -- 1,024
    # tokens here. Left in place it silently drops everything past roughly the
    # first 6 KB of a document, so a long file would be reported as clean from
    # that point on with no error anywhere. Chunking (_chunk_bounds) is how
    # long documents are handled; the tokenizer must hand us all of it.
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return ModelBundle(
        session=session,
        tokenizer=tokenizer,
        kinds=tuple(kinds),
        kind_of=kind_of,
        is_begin=is_begin,
        input_names=tuple(i.name for i in session.get_inputs()),
        half_field=half_field,
        calibration=calibration,
        path=path,
    )


def _load_calibration(spec):
    """Calibration from the artifact, or () for an uncalibrated bundle.

    Returns ``(global_knots, {type_name: knots})``. Three artifact generations
    all load: no ``calibration`` key at all (raw scores), a global-only curve
    with ``x``/``y``, and one that also carries ``per_type``.
    """
    if not spec:
        return ()
    import numpy as np

    def knots(entry, label):
        x = np.asarray(entry["x"], dtype=np.float64)
        y = np.asarray(entry["y"], dtype=np.float64)
        if x.shape != y.shape or x.size == 0:
            raise ModelUnavailable(f"malformed calibration ({label}): {entry!r}")
        return (x, y)

    per_type = {name: knots(entry, name)
                for name, entry in (spec.get("per_type") or {}).items()}
    return (knots(spec, "global"), per_type)


def calibrate(scores, calibration, types=None):
    """Map raw max-softmax scores to calibrated precision estimates.

    A token classifier's max softmax is a *ranking* signal, not a probability:
    it is systematically overconfident, because the model is trained to put all
    the mass on one class. So ``min_confidence=0.70`` on a raw score means "in
    the top band of the model's self-assessment", which is a knob without units
    -- it cannot be compared across students, and it says nothing about how
    often a span at that score is actually right.

    The knots come from isotonic regression fitted on a held-out slice
    (``training/calibrate.py``): monotone, non-parametric, and fitted against
    the strict target -- exact ``(type, start, end)`` agreement with gold. After
    it, a confidence of 0.70 means **this span has about a 70% chance of being
    exactly right**. Monotonicity matters: it guarantees calibration can never
    re-order two spans of one type, so it changes what the number means without
    changing which spans outrank which.

    **Per type where the data supports it.** One global curve is nearly perfect
    in aggregate and wrong in detail, because the per-type errors cancel.
    Measured on a 10,000-document holdout with a single curve, the pooled gap
    between claimed and actual was 0.014 while ``URL`` ran **0.107
    under**-confident and ``DEVICE_ID``, ``VEHICLE_ID`` and ``USER_ID`` each
    ran 0.05-0.07 over. A global threshold then cuts every type in a different
    place, which is how ``URL`` lost 20 points of recall to a threshold that
    was correct on average. Per-type curves take the pooled error to 0.004 and
    the worst per-type gap from 0.107 to 0.026.

    Types with too few spans to fit keep the global curve -- a curve fitted on
    forty spans is noise wearing a probability's clothes, and it would be
    applied with the same authority as one fitted on thirty thousand.

    Linear interpolation between knots, clamped at the ends.
    """
    import numpy as np

    if not calibration:
        return scores
    global_knots, per_type = calibration
    scores = np.asarray(scores, dtype=np.float64)
    adjusted = np.interp(scores, *global_knots)
    if types is None or not per_type:
        return adjusted
    for index, name in enumerate(types):
        knots = per_type.get(name)
        if knots is not None:
            adjusted[index] = np.interp(scores[index], *knots)
    return adjusted


def _half_receptive_field(config: dict) -> int:
    """How far a token's prediction can depend on, in tokens, one side.

    Each dilated conv layer of kernel k reaches ``dilation * (k - 1) / 2``
    tokens either way, and the layers stack additively. Read from the artifact
    rather than hard-coded, so a wider student trained later gets a correctly
    sized chunk overlap instead of quietly truncated context at every boundary.
    Falls back to the `m` student's 126 when the config is absent.
    """
    kernel = int(config.get("kernel_size", 5))
    dilations = config.get("dilations")
    if not dilations:
        layers = int(config.get("n_layers", 6))
        dilations = [2 ** i for i in range(layers)]
    return sum(int(d) * (kernel - 1) // 2 for d in dilations) or 126


def _chunk_bounds(n_tokens: int, window: int, context: int):
    """-> [(scan_start, scan_end, keep_start, keep_end)] covering all tokens.

    The student is a CNN with no position embeddings, so a token's output
    depends only on its receptive field. Feeding each chunk `context` tokens of
    padding on both sides and keeping only the middle therefore yields exactly
    the whole-document result, as long as `context` exceeds the receptive field
    (253 tokens for the `m` student: 1 + 4 * sum(dilations)).
    """
    if n_tokens <= window:
        return [(0, n_tokens, 0, n_tokens)]
    stride = window - 2 * context
    if stride <= 0:
        # Silently clamping here would be worse than failing: every chunk
        # would carry less context than the receptive field needs, so long
        # documents would quietly get different answers than short ones --
        # the exact property _chunk_bounds exists to preserve.
        raise ValueError(
            f"window={window} leaves no room for two {context}-token context "
            f"margins; use a window above {2 * context}"
        )
    bounds = []
    keep = 0
    while keep < n_tokens:
        keep_end = min(keep + stride, n_tokens)
        scan_start = max(0, keep - context)
        scan_end = min(n_tokens, keep_end + context)
        bounds.append((scan_start, scan_end, keep, keep_end))
        keep = keep_end
    return bounds


class OnnxNerDetector:
    """The Stage 2 student as a ``Detector``.

    Implements the same structural protocol as every regex detector, so it
    plugs into ``Pipeline`` with no change to Stage 3 -- which is the whole
    point of the ``Entity`` contract in docs/DESIGN.md section 4.

    Args:
        model_dir: directory holding ``model.onnx``, ``tokenizer.json`` and
            ``model.json``. Defaults to :func:`default_model_dir`.
        min_confidence: drop spans whose mean per-token probability is below
            this. Guard 1 in the module docstring. 0.0 disables it.

            The 0.50 default is **swept, not guessed**, and after per-type
            calibration it has units: a span at 0.50 has roughly a 50% chance
            of being exactly right, and that means the same thing for every
            type. Measured on 3,000 holdout documents (`l` student):

                threshold  rule F1  rule F2  model F1  model F2  frozen acc
                     0.30    0.939    0.927     0.933     0.926        0.92
                     0.40    0.940    0.927     0.932     0.922        0.95
                     **0.50** 0.940   0.927     0.930     0.918        **1.00**
                     0.70    0.938    0.923     0.913     0.893        1.00

            0.30 is the F1 and F2 optimum -- they agree, which they did not
            before per-type calibration -- and 0.50 ships, because the frozen
            corpus is the other half of the evidence. Its four adversarial
            false positives (an order number, a chart number, a magazine
            subscriber id, a confirmation number: the reference-number class
            Track A of the improvement plan hardened the rules against) now
            calibrate to 0.36, 0.41 and 0.49, with the fourth gone entirely.
            0.50 clears all of them for 0.003 model-tier F1.

            This default moved down from 0.70, and per-type calibration is why.
            `USER_ID` was 0.05 over-confident under one global curve, so its
            false positives scored high enough to need a 0.70 bar; scored
            honestly they sit below 0.50, and every other type gets to keep the
            recall that bar was costing it.
        revalidate: re-run our validators on model spans of checksummed types.
            Guard 2. Leave this on unless you are measuring its cost.
        merge_adjacent_spans: report "Jane Doe" as one PERSON_NAME rather than
            the two spans Nemotron labels. Turn it OFF when scoring against
            Nemotron gold, which counts them separately.
        snap_word_edges: grow span edges that land inside a word out to the
            word boundary. See :func:`snap_to_word_edges` -- measured at 76
            spans fixed and 0 broken on 2,000 holdout documents.
        window / context: chunking for documents longer than `window` tokens.
    """

    name = "onnx/ner-v1"

    def __init__(
        self,
        model_dir: str | Path | None = None,
        *,
        min_confidence: float = 0.50,
        revalidate_checksums: bool = True,
        merge_adjacent_spans: bool = True,
        snap_word_edges: bool = True,
        window: int = 2048,
        context: int = 320,
        threads: int = 1,
    ):
        resolved = Path(model_dir) if model_dir is not None else default_model_dir()
        if resolved is None:
            raise ModelUnavailable(
                f"no Stage 2 model found. Set ${MODEL_DIR_ENV} to a directory "
                "holding model.onnx / tokenizer.json / model.json, or run "
                "training/export.py --bundle after training a student."
            )
        self.model_dir = resolved
        self.min_confidence = min_confidence
        self.revalidate_checksums = revalidate_checksums
        self.merge_adjacent_spans = merge_adjacent_spans
        self.snap_word_edges = snap_word_edges
        self.window = window
        self.context = context
        self.threads = threads
        self._bundle: ModelBundle | None = None

    @property
    def bundle(self) -> ModelBundle:
        if self._bundle is None:
            self._bundle = load_bundle(str(self.model_dir), self.threads)
        return self._bundle

    def detect(self, text: str) -> list[Entity]:
        import numpy as np

        if not text:
            return []
        bundle = self.bundle
        encoding = bundle.tokenizer.encode(text, add_special_tokens=True)
        ids = np.asarray(encoding.ids, dtype=np.int64)
        offsets = np.asarray(encoding.offsets, dtype=np.int64).reshape(-1, 2)
        if ids.size == 0:
            return []

        n = ids.shape[0]
        predictions = np.zeros(n, dtype=np.int64)
        probability = np.zeros(n, dtype=np.float64)
        # Never overlap by less than the receptive field, or a token near a
        # boundary would be predicted from truncated context and the chunked
        # result would silently differ from the whole-document one.
        context = max(self.context, bundle.half_field + 1)
        for scan_start, scan_end, keep_start, keep_end in _chunk_bounds(
            n, self.window, context
        ):
            chunk = ids[scan_start:scan_end][None, :]
            feed = {"input_ids": chunk}
            if "attention_mask" in bundle.input_names:
                feed["attention_mask"] = np.ones_like(chunk)
            logits = bundle.session.run(None, feed)[0][0]
            logits = logits[keep_start - scan_start:keep_end - scan_start]
            chunk_predictions = logits.argmax(-1)
            predictions[keep_start:keep_end] = chunk_predictions
            # Softmax only on the tokens that landed inside a span. On a 10 KB
            # document that is a few dozen rows out of ~1,700, so the
            # confidence signal costs microseconds instead of the ~1 ms a full
            # 111-class softmax over every token would.
            rows = np.nonzero(bundle.kind_of[chunk_predictions] >= 0)[0]
            if rows.size:
                scores = logits[rows]
                shifted = scores - scores.max(-1)[:, None]
                np.exp(shifted, out=shifted)
                probability[keep_start + rows] = 1.0 / shifted.sum(-1)

        # Decoded once over the whole document, not per chunk: a span that
        # straddles a chunk boundary must not come out as two entities.
        spans = decode_spans(text, offsets, predictions, bundle.kind_of,
                             bundle.is_begin, probability)

        if self.snap_word_edges:
            spans = snap_to_word_edges(spans, text)
        if self.merge_adjacent_spans:
            spans = merge_adjacent(spans, text, bundle.kinds)

        # Calibrated last, on the finished span. The fit target is "is this
        # exact (type, start, end) correct?", so the number calibration is
        # applied to has to be the same number the fit was conditioned on --
        # the mean raw token probability over the span as finally emitted,
        # merges included. Calibrating per token and then averaging would
        # target something else, because np.interp is not linear.
        if bundle.calibration and spans:
            raw = np.fromiter((c for _, _, _, c in spans), dtype=np.float64,
                              count=len(spans))
            names = [bundle.kinds[k].value for k, _, _, _ in spans]
            adjusted = calibrate(raw, bundle.calibration, names)
            spans = [(k, s, e, float(p))
                     for (k, s, e, _), p in zip(spans, adjusted)]

        entities: list[Entity] = []
        for kind, start, end, confidence in spans:
            if confidence < self.min_confidence:
                continue
            entity_type = bundle.kinds[kind]
            fragment = text[start:end]
            if (self.revalidate_checksums
                    and entity_type in CHECKSUMMED_TYPES
                    and not revalidate(entity_type, fragment)):
                continue
            entities.append(Entity(
                type=entity_type,
                start=start,
                end=end,
                text=fragment,
                confidence=confidence,
                detector=self.name,
            ))
        return entities
