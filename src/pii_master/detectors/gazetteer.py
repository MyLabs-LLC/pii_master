"""Context-gated gazetteer detector for names and street addresses.

Papadopoulou, Lison, Anderson, Øvrelid & Pilán, "Neural Text Sanitization
with Privacy Risk Indicators" (arXiv 2310.14312; LRE 2026) build a
privacy-oriented entity recognizer by combining a standard NER model with
a person-term gazetteer. PRIOR_ART.md §7 independently named the same
missing tier: an Aho-Corasick / hash-lookup gazetteer before any neural
pass, because it is O(document) and independent of dictionary size.

This is that tier. It is *not* a dump of every capitalised token:

  * a PERSON_NAME fires only on first+last (both in the list), or
    first + middle-initial + capitalised last, or a title + gazetteer name;
  * an ADDRESS fires only on house-number + street + suffix, optionally
    followed by a capitalised city.

Single gazetteer hits are silent. That is the precision bargain that lets
this run in the zero-dependency ``fast`` path without crying wolf on
"Young", "May", or "Park".

Detector names stay under the ``regex/`` prefix so the fusion policy
treats these as cue-tier rules, not as a model: checksum facts still
outrank them, and in ``--deep`` the student still outranks them on
overlap. Rules-only fusion ranking stays off, so v0.2 overlap order is
unchanged for the original detectors.
"""

from __future__ import annotations

import re

from ..entities import EntityType
from ..gazetteers import (
    FIRST_NAMES,
    LAST_NAMES,
    LAST_STOPWORDS,
    STREET_SUFFIXES,
    TITLES,
)
from ..models import Entity

# Middle initial (`Q.`) must win over a bare letter, so the dotted form
# is first in the alternation.
_TOKEN = re.compile(
    r"[A-Z]\.|[A-Za-z]+(?:'[A-Za-z]+)?|\d+"
)

# House number + street words + suffix, optional ", City".
_STREET = re.compile(
    r"\b(\d{1,6})\s+"
    r"((?:[A-Z][A-Za-z.]+(?:\s+|$)){1,3})"
    r"(Street|St\.?|Avenue|Ave\.?|Boulevard|Blvd\.?|Road|Rd\.?|"
    r"Drive|Dr\.?|Lane|Ln\.?|Court|Ct\.?|Way|Parkway|Pkwy\.?|"
    r"Place|Pl\.?|Circle|Cir\.?|Terrace|Ter\.?|Highway|Hwy\.?|"
    r"Trail|Trl\.?)"
    r"(?:,\s+([A-Z][A-Za-z]+))?",
)


def _tokens(text: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group(0)) for m in _TOKEN.finditer(text)]


class GazetteerDetector:
    """First+last names and numbered streets, from public-domain lists."""

    name = "regex/gazetteer"
    name_confidence = 0.80
    address_confidence = 0.85

    def detect(self, text: str) -> list[Entity]:
        entities: list[Entity] = []
        entities.extend(self._names(text))
        entities.extend(self._addresses(text))
        return entities

    def _names(self, text: str) -> list[Entity]:
        toks = _tokens(text)
        found: list[Entity] = []
        i = 0
        n = len(toks)
        while i < n:
            start, end, raw = toks[i]
            lower = raw.lower().rstrip(".")
            # Title + gazetteer name, optionally + last.
            if lower in TITLES:
                if i + 1 < n:
                    n_start, n_end, n_raw = toks[i + 1]
                    n_low = n_raw.lower()
                    if n_raw[0].isupper() and (
                        n_low in FIRST_NAMES or n_low in LAST_NAMES
                    ):
                        span_end = n_end
                        consume = 2
                        if i + 2 < n:
                            l_start, l_end, l_raw = toks[i + 2]
                            if (
                                l_raw[0].isupper()
                                and l_raw.lower() in LAST_NAMES
                                and text[n_end:l_start].isspace()
                            ):
                                span_end = l_end
                                consume = 3
                        # Emit the name, not the title: gold is "Jane Doe",
                        # not "Dr. Jane Doe".
                        found.append(Entity(
                            type=EntityType.PERSON_NAME,
                            start=n_start,
                            end=span_end,
                            text=text[n_start:span_end],
                            confidence=self.name_confidence,
                            detector=self.name,
                        ))
                        i += consume
                        continue
                i += 1
                continue

            if raw[0].isupper() and lower in FIRST_NAMES:
                # First + middle initial + last.
                if i + 2 < n:
                    mid_s, mid_e, mid = toks[i + 1]
                    last_s, last_e, last = toks[i + 2]
                    if (
                        re.fullmatch(r"[A-Z]\.", mid)
                        and last[0].isupper()
                        and last.lower() not in LAST_STOPWORDS
                        and last.lower() not in TITLES
                        and text[end:mid_s].isspace()
                        and text[mid_e:last_s].isspace()
                    ):
                        found.append(Entity(
                            type=EntityType.PERSON_NAME,
                            start=start,
                            end=last_e,
                            text=text[start:last_e],
                            confidence=self.name_confidence,
                            detector=self.name,
                        ))
                        i += 3
                        continue
                # First + last, both in the lists.
                if i + 1 < n:
                    last_s, last_e, last = toks[i + 1]
                    if (
                        last[0].isupper()
                        and last.lower() in LAST_NAMES
                        and text[end:last_s].isspace()
                    ):
                        found.append(Entity(
                            type=EntityType.PERSON_NAME,
                            start=start,
                            end=last_e,
                            text=text[start:last_e],
                            confidence=self.name_confidence,
                            detector=self.name,
                        ))
                        i += 2
                        continue
            i += 1
        return found

    def _addresses(self, text: str) -> list[Entity]:
        found: list[Entity] = []
        for match in _STREET.finditer(text):
            # The street-name capture is greedy over capitalised words;
            # reject if the "street name" is itself a suffix (``44 Street``).
            name = match.group(2).strip()
            suffix = match.group(3)
            if name.lower().rstrip(".") in STREET_SUFFIXES:
                continue
            if suffix.lower().rstrip(".") not in STREET_SUFFIXES:
                continue
            start, end = match.span()
            found.append(Entity(
                type=EntityType.ADDRESS,
                start=start,
                end=end,
                text=text[start:end],
                confidence=self.address_confidence,
                detector=self.name,
            ))
        return found
