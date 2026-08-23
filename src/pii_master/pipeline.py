"""Run detectors and resolve overlapping spans into a flat entity list.

This is Stage 3's front half, and since v0.3 it is also where the **fusion
policy** lives -- named here in code, as docs/DISTILLATION_PLAN.md section 7
requires, rather than left as a comment.
"""

from __future__ import annotations

from typing import Sequence

from .detectors import Detector, default_detectors
from .entities import CHECKSUMMED_TYPES, TAXONOMY
from .models import Entity

#: Provenance prefix every Stage 1 detector uses (``regex/ssn``, ...). The
#: Stage 2 detector reports ``onnx/ner-v1``, so the tier of a span is readable
#: from the span itself and no out-of-band bookkeeping is needed.
RULE_PREFIX = "regex/"

# Fusion tiers, lowest number wins an overlap. See fusion_rank.
TIER_CHECKSUM_RULE = 0
TIER_MODEL = 1
TIER_CUE_RULE = 2


def fusion_rank(entity: Entity) -> int:
    """Precedence tier for one span: **checksum rules > model > cue rules**.

    The plan's fusion clause reads "checksum-validated rule spans outrank model
    spans", and it has a narrow and a broad reading. The broad one -- *every*
    rule span outranks the model -- is the obvious implementation and it is
    wrong: measured on a 100k-document holdout it costs 0.028 F1 against the
    narrow one (docs/DISTILLATION_RESULTS.md section 5). The damage is
    concentrated exactly where you would predict, on the types where the rules
    were never strong:

        type                rules F1   model F1   rules-first   checksum-first
        US_DRIVER_LICENSE      0.001      0.829         0.820            0.831
        ACCOUNT_NUMBER         0.407      0.803         0.678            0.806
        HEALTH_PLAN_ID         0.448      0.897         0.797            0.896
        PHONE_US               0.471      0.889         0.651            0.719
        MRN                    0.615      0.891         0.851            0.887

    and nowhere on the checksummed types, which the model cannot displace under
    either reading (EMAIL 0.980, URL 0.952 in both columns).

    So: a Luhn-valid card number or a parsed IP address is a fact and outranks
    the model. A cue-anchored MRN guess is not, and does not. Cue-anchored rule
    spans still contribute recall wherever the model is silent -- they lose
    overlaps, they are not discarded.
    """
    if not entity.detector.startswith(RULE_PREFIX):
        return TIER_MODEL
    if entity.type in CHECKSUMMED_TYPES:
        return TIER_CHECKSUM_RULE
    return TIER_CUE_RULE


def erases_phi(rule: Entity, model_spans: list[Entity]) -> bool:
    """Would letting the model win this overlap cost the document its PHI label?

    MRN and HEALTH_PLAN_ID escalate a document to PHI on their own
    (``classify.py`` rule 2, reading ``phi_specific`` from the taxonomy). Every
    other type only contributes evidence. So when the model displaces one of
    these two and proposes something that is *not* phi_specific in its place, a
    PHI document silently becomes PII -- and a missed MRN is the failure this
    project ranks worst (docs/DESIGN.md section 1: a missed medical record
    number is a reportable incident; a false alarm costs a reviewer minutes).

    Not hypothetical: it is what disqualified the `xs` student. On the frozen
    corpus's `phi-011` -- "Coverage active: insurance member id 4471-2299" --
    it displaced the HEALTH_PLAN_ID rule span, and with no medical-context term
    anywhere else in the sentence the document fell to PII. Frozen-corpus PHI
    recall 0.92 against the gate's 1.00.

    The guard is deliberately narrow in two ways. It covers two types, and only
    where a rule already fired on the strict cues Track A of the improvement
    plan left behind. And it yields whenever the model *agrees* the span is
    PHI-specific: a model MRN span may still refine or replace a rule MRN span,
    because the document label survives either way and the model is the better
    judge of boundaries. Only a model span that would erase PHI-ness loses.
    """
    if not TAXONOMY[rule.type].phi_specific:
        return False
    return not any(
        span.overlaps(rule) and TAXONOMY[span.type].phi_specific
        for span in model_spans
    )


def truncates(rule: Entity, model_spans: list[Entity]) -> bool:
    """Is some model span the SAME type as this rule span, overlapping it, and
    strictly shorter?

    That is the truncation signature, and it is the *only* case in which a
    cue-anchored rule span is promoted over the model. Deliberately narrow:

      * same type -- the two tiers agree on what this is and disagree only on
        where it ends, so length settles it with no appeal to which tier is
        generally better.
      * strictly shorter -- an equal-length model span is not a truncation, and
        a longer one is the model making a real claim the rule missed.
      * anything cross-type is left alone, which keeps the measured
        checksum-first behaviour exactly as it was: the model beating a
        cue-anchored rule with a *different* type is where its +0.137 F1 comes
        from, and blanket-promoting rules would give that back.
    """
    width = rule.end - rule.start
    return any(
        span.type is rule.type
        and span.overlaps(rule)
        and (span.end - span.start) < width
        for span in model_spans
    )


class Pipeline:
    """Collects candidates from every detector and resolves overlaps.

    Resolution is greedy weighted-interval selection: candidates sorted by
    (fusion tier, -confidence, -length, start, detector name), each accepted
    iff it overlaps no already-accepted span. Deterministic and explainable;
    the sort dominates at O(n log n) for typical span counts.

    **The tier key applies only when a model tier is actually present.** With
    rules alone every candidate would sit in tier 0 or tier 2, and adding that
    key would silently re-order rule-vs-rule overlaps that v0.2 resolved by
    confidence -- changing shipped behaviour for a reason the measurement never
    supported. The finding in :func:`fusion_rank` is about rule-vs-*model*
    precedence, so that is the only case where it fires. Rules-only output is
    byte-identical to v0.2.

    **The promotion rule.** A cue-anchored rule span is promoted to the top
    tier when an overlapping model span *of the same type* is strictly shorter
    (:func:`truncates`). The model's characteristic failure on short documents
    is a truncated span, and a truncated span of the right type is worse than
    the rule it displaced -- measured on the frozen corpus before this rule existed, the
    student replaced `DATE_DOB "3-4-1985"` with `"-4-"` and
    `PHONE_US "(415) 555-2671"` with `"415) 555-2671"`. Both are strictly worse
    than what Stage 1 already had, and both are the same type, so length
    settles it without needing to know which tier is generally better. Where
    the model genuinely disagrees -- a longer span, or a different type
    entirely -- it still wins, which is where its +0.137 F1 comes from.

    v2 replacement point: allowing nested spans of *different* types (e.g.
    a phone number inside an email's display name) means swapping only this
    resolution step -- detectors and the Entity contract are unaffected.
    """

    def __init__(self, detectors: Sequence[Detector] | None = None):
        self.detectors: list[Detector] = (
            list(detectors) if detectors is not None else default_detectors()
        )

    def run(self, text: str) -> list[Entity]:
        candidates: list[Entity] = []
        for detector in self.detectors:
            candidates.extend(detector.detect(text))
        model_spans = [
            e for e in candidates if not e.detector.startswith(RULE_PREFIX)
        ]
        if model_spans:
            ranks = [
                TIER_CHECKSUM_RULE
                if fusion_rank(e) == TIER_CUE_RULE
                and (truncates(e, model_spans) or erases_phi(e, model_spans))
                else fusion_rank(e)
                for e in candidates
            ]
        else:
            ranks = [0] * len(candidates)

        ordered = sorted(
            zip(ranks, candidates),
            key=lambda pair: (
                pair[0],
                -pair[1].confidence,
                -(pair[1].end - pair[1].start),
                pair[1].start,
                pair[1].detector,
            ),
        )
        accepted: list[Entity] = []
        for _, candidate in ordered:
            if any(candidate.overlaps(kept) for kept in accepted):
                continue
            accepted.append(candidate)
        accepted.sort(key=lambda e: (e.start, e.end))
        return accepted


def deep_pipeline(model_dir=None, **kwargs) -> Pipeline:
    """Rules + the Stage 2 student -- the ``deep`` serving mode.

    Raises :class:`pii_master.ner.ModelUnavailable` if the optional ML extra or
    the model artifact is missing, rather than silently degrading to rules: a
    caller who asked for deep mode and got rules-only would read the missing
    names as "this document has no names".

    Keyword arguments are forwarded to :class:`~pii_master.ner.OnnxNerDetector`
    (``min_confidence``, ``revalidate_checksums``, ...).
    """
    from .ner import OnnxNerDetector

    detector = OnnxNerDetector(model_dir, **kwargs)
    # Load the session now rather than on the first non-empty document. The
    # detector itself is lazy, which is right for a library; a *pipeline* is
    # not, because the failure would otherwise depend on the input: scanning an
    # empty file with a misconfigured model dir would exit 0 and report NONE,
    # and the same command on a real file would raise. Configuration errors
    # must not be content-dependent. It also moves ONNX session creation --
    # tens of milliseconds -- out of the first document's latency.
    _ = detector.bundle
    return Pipeline([*default_detectors(), detector])
