"""Generate adversarial near-miss negatives: documents that read as sensitive and are not.

The gate diagnosis found that the sealed real-world half is hard because its
*negatives* look like positives -- class separation is 36% narrower there than on
the training-side half, and a gate that never saw those documents reproduces the
gap exactly. So the gap is not a modelling error to regularise away; the training
data simply contains no documents of that kind.

This makes some. Every class here is a mundane business document that is dense in
personal-data vocabulary -- field labels, schema columns, policy language, masked
records -- and contains no personal data at all. That is precisely the region
where the gate currently fires and should not.

Follows the module's generator contract: one real file per document at a
realistic path, a stamped output directory, `manifest.json` beside the corpus, a
named-class registry, round-robin class assignment, and an LLM mode that falls
back per-document rather than failing the run.

**Two rules matter more here than anywhere else, and both are enforced downstream
by `h2h_datagen_verify.py` rather than trusted:**

* **The documents must actually be negative.** An LLM asked for a blank intake
  form will sometimes helpfully fill in "John Smith". A generated negative that
  contains a real identifier is mislabelled training data, and mislabelled
  negatives teach the gate to stay shut on documents that do carry PII -- the
  exact failure this project exists to prevent. Every document is scanned with
  the same shape detectors the model itself uses, and any hit is rejected.

* **The documents must not share a tell.** If every generated negative carries
  "This is a sample document containing no real data", the gate learns that
  sentence and nothing else, and the sealed score will not move. The prompts
  forbid disclaimers, and the verifier scans for them.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROJECT = Path("/home/lence/workspace/pii_master/projects/pii-head-to-head-v1")
OUT_ROOT = PROJECT / "datagen"
DELIM = "---DOC---"
MODEL_SLUG = "claude"

# Randomised context, so 2,000 documents are not 2,000 rewrites of one document.
INDUSTRIES = ("a regional insurance carrier", "a county health department",
              "a mid-sized credit union", "a state university registrar",
              "a logistics contractor", "a municipal utility",
              "a staffing agency", "a hospital billing office",
              "a school district", "a retail pharmacy chain",
              "a title and escrow company", "a public transit authority")
ERAS = ("the late 1990s", "around 2004", "around 2011", "around 2016", "around 2021")
REGISTERS = ("terse and bureaucratic", "verbose and legalistic",
             "plain and procedural", "dry and technical",
             "formal with numbered clauses")
SHAPES = ("a plain-text memo", "a numbered-clause document", "a fixed-width table",
          "a bulleted checklist", "a form with labelled sections",
          "an indented outline", "a paragraph-heavy narrative")


@dataclass(frozen=True)
class DocTemplate:
    """One declarative class: what it is, where it lives, what the oracle says."""

    name: str
    description: str
    path_pattern: str
    llm_prompt: Callable[[random.Random], str]
    faker: Callable[[random.Random], str]
    #: The oracle. Every class here is a negative: no sensitive tag, no PII.
    expected_tags: tuple[str, ...] = ()


def _ctx(rng: random.Random) -> dict[str, str]:
    return {"industry": rng.choice(INDUSTRIES), "era": rng.choice(ERAS),
            "register": rng.choice(REGISTERS), "shape": rng.choice(SHAPES)}


#: Appended to every prompt. The hard rules are the whole point of the corpus.
RULES = """
HARD RULES — these define the document's correctness:
- Contain NO personal data whatsoever: no person names (use role titles such as
  "the Records Custodian"), no social security or tax numbers, no account or card
  numbers, no phone numbers, no email addresses, no street addresses, no dates of
  birth, no patient or member identifiers. Field LABELS and column NAMES are the
  point and must appear; values must not.
- Do NOT state that the document is a sample, a template, a test, redacted for
  illustration, or free of real data. No disclaimers of any kind. A reader must
  not be able to tell this document apart from an ordinary internal file.
- No preamble, no trailing commentary, no markdown code fences, no title like
  "Document 1". Output only the documents themselves.
"""


def _p(body: str, ctx: dict[str, str], n: int) -> str:
    return (f"Produce {n} plain-text business documents, separated by a line "
            f"containing exactly {DELIM}.\n\n{body.format(**ctx)}\n"
            f"Register: {ctx['register']}. Present it as {ctx['shape']}.\n"
            f"Vary the length of each between 150 and 600 words, and vary the "
            f"formatting between them.\n{RULES}")


# --------------------------------------------------------------- faker fallback
def _fields(rng: random.Random, n: int) -> list[str]:
    pool = ["Full Name", "Date of Birth", "Social Security Number", "Home Address",
            "Primary Phone", "Email Address", "Member ID", "Account Number",
            "Emergency Contact", "Marital Status", "Gender", "Nationality",
            "Driver's License Number", "Passport Number", "Employer",
            "Medical Record Number", "Policy Number", "Taxpayer Identification"]
    return rng.sample(pool, min(n, len(pool)))


def _faker_form(rng: random.Random) -> str:
    head = rng.choice(["SECTION A — APPLICANT DETAILS", "PART 1: IDENTIFYING INFORMATION",
                       "ENROLMENT RECORD", "INTAKE WORKSHEET"])
    lines = [head, ""]
    for f in _fields(rng, rng.randint(8, 14)):
        lines.append(f"{f}: " + "_" * rng.randint(18, 40))
    lines += ["", "Reviewed by: ______________________  Date: ____ / ____ / ______"]
    return "\n".join(lines)


def _faker_schema(rng: random.Random) -> str:
    cols = [("cust_id", "NUMBER(12)"), ("ssn", "CHAR(11)"), ("date_of_birth", "DATE"),
            ("home_address", "VARCHAR2(120)"), ("primary_phone", "VARCHAR2(15)"),
            ("email_addr", "VARCHAR2(80)"), ("member_no", "VARCHAR2(20)"),
            ("acct_number", "VARCHAR2(24)"), ("dl_number", "VARCHAR2(18)")]
    rng.shuffle(cols)
    out = ["TABLE: CUST_MASTER", ""]
    for c, t in cols[:rng.randint(6, 9)]:
        out.append(f"  {c:<20} {t:<16} restricted={rng.choice(['Y', 'N'])}")
    return "\n".join(out)


def _faker_policy(rng: random.Random) -> str:
    return "\n".join([
        "RECORDS HANDLING STANDARD", "",
        "1. Scope. This standard governs categories of personal information held",
        "   by the organisation, including identifiers, contact details, dates of",
        "   birth, government-issued numbers and financial account references.",
        "2. Access. Access to restricted identifier columns is granted by role and",
        "   reviewed each quarter by the Records Custodian.",
        "3. Retention. Records are retained seven years past account closure.",
        "4. Disposal. Physical media is destroyed under witnessed procedure.",
    ])


def _faker_redacted(rng: random.Random) -> str:
    return "\n".join([
        "CLAIM SUMMARY (RESTRICTED FIELDS SUPPRESSED)", "",
        "Claimant Name: [SUPPRESSED]", "Date of Birth: [SUPPRESSED]",
        "Member Identifier: XXXXXXXXX", "Contact Telephone: XXX-XXX-XXXX",
        "Service Address: [SUPPRESSED]", "",
        "Adjudication notes follow in section 4.",
    ])


CLASSES: tuple[DocTemplate, ...] = (
    DocTemplate(
        "blank_intake_form", "unfilled form: personal-data labels, no values",
        "hr/onboarding/forms/intake_{uid}.txt",
        lambda r: _p("An UNFILLED internal form used by {industry} in {era}. It has "
                     "labelled fields for applicant identity and contact details, with "
                     "blank underscores or empty boxes where entries would go. Include "
                     "section headings, instructions to the person completing it, and "
                     "an office-use-only block.", _ctx(r), 4),
        _faker_form),
    DocTemplate(
        "data_dictionary", "schema/column documentation naming personal-data fields",
        "engineering/schemas/{uid}_data_dictionary.txt",
        lambda r: _p("A data dictionary excerpt for a customer or member database at "
                     "{industry} in {era}. List column names, types, nullability and "
                     "operational notes. Column names should include personal-data "
                     "fields as SCHEMA NAMES ONLY, with notes on access restriction, "
                     "retention and downstream extracts.", _ctx(r), 4),
        _faker_schema),
    DocTemplate(
        "privacy_policy", "policy/notice text about handling personal data",
        "legal/privacy/notice_{year}_{uid}.txt",
        lambda r: _p("An internal privacy or records-handling standard issued by "
                     "{industry} in {era}. It describes the CATEGORIES of personal "
                     "information the organisation holds, who may access them, "
                     "retention periods, and disposal procedure. It names categories "
                     "abstractly and never gives an example value.", _ctx(r), 4),
        _faker_policy),
    DocTemplate(
        "redacted_record", "a real-shaped record with every identifier suppressed",
        "compliance/disclosures/redacted/{uid}.txt",
        lambda r: _p("A case, claim or personnel record from {industry} in {era} that "
                     "has been prepared for disclosure with every identifying field "
                     "suppressed. Identifier fields appear as [SUPPRESSED], solid "
                     "blocks, or runs of X. The surrounding narrative, dates of "
                     "process steps, reference codes and adjudication notes remain "
                     "intact and read naturally.", _ctx(r), 4),
        _faker_redacted),
    DocTemplate(
        "aggregate_statistics", "demographic tables reporting counts, never individuals",
        "analytics/reports/{uid}_demographics.txt",
        lambda r: _p("A statistical report from {industry} in {era} tabulating "
                     "demographics of a served population: counts and percentages "
                     "broken down by age band, sex, ethnicity, language, income "
                     "bracket and geography at postal-district level or coarser. "
                     "Aggregates only, with a note on small-cell suppression.",
                     _ctx(r), 4),
        lambda r: "AGE BAND    COUNT   PCT\n18-24        1,204  11.3\n25-34        2,881  27.0"),
    DocTemplate(
        "regulatory_text", "statute or regulation excerpts about personal information",
        "legal/regulatory/{uid}_excerpt.txt",
        lambda r: _p("An excerpt of a statute, regulation or administrative rule "
                     "governing the protection of personal information, of the kind "
                     "{industry} would keep on file in {era}. Numbered sections, "
                     "definitions of terms such as identifier categories, obligations "
                     "on holders of records, and penalties.", _ctx(r), 4),
        _faker_policy),
    DocTemplate(
        "security_guidance", "guidance about protecting identifiers",
        "security/awareness/{uid}_handout.txt",
        lambda r: _p("A staff or customer guidance handout issued by {industry} in "
                     "{era} explaining how to protect personal identifiers: what to "
                     "shred, what never to send by email, how to recognise a "
                     "pretexting call, and who to contact internally by role title.",
                     _ctx(r), 4),
        _faker_policy),
    DocTemplate(
        "system_log_config", "logs and configuration naming personal-data fields",
        "ops/logs/{uid}.txt",
        lambda r: _p("An application log extract or configuration file from {industry} "
                     "in {era}. It references personal-data field names in mapping "
                     "rules, masking configuration, validation errors and ETL job "
                     "output. Values are absent, tokenised, or shown as masked "
                     "placeholders. Include timestamps, job identifiers and row counts.",
                     _ctx(r), 4),
        _faker_schema),
    DocTemplate(
        "vendor_questionnaire", "security assessment asking how personal data is handled",
        "procurement/assessments/{uid}.txt",
        lambda r: _p("A vendor security assessment questionnaire sent by {industry} in "
                     "{era}, with numbered questions about how a supplier stores, "
                     "encrypts, transmits and disposes of personal information, plus "
                     "blank response boxes and an evidence-required column.", _ctx(r), 4),
        _faker_form),
    DocTemplate(
        "training_material", "compliance training handout about personal data",
        "training/compliance/{uid}_module.txt",
        lambda r: _p("A compliance training handout or slide script used by {industry} "
                     "in {era}, teaching staff which categories of information are "
                     "restricted, the consequences of mishandling them, and a short "
                     "scenario-based quiz whose scenarios describe situations without "
                     "naming any individual.", _ctx(r), 4),
        _faker_policy),
)


# ------------------------------------------------------------------- generation
def _stamped_dir(count: int) -> Path:
    stamp = datetime.now().strftime("%y%m%d%H")
    base = OUT_ROOT / f"{count}_{stamp}_{MODEL_SLUG}"
    n = 1
    out = base
    while out.exists():
        out = OUT_ROOT / f"{count}_{stamp}_{n}_{MODEL_SLUG}"
        n += 1
    return out


def _call_claude(prompt: str, timeout: int) -> str | None:
    try:
        r = subprocess.run(["claude", "-p", "--output-format", "text"],
                           input=prompt, text=True, capture_output=True,
                           timeout=timeout, cwd="/tmp")
        return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None
    except (subprocess.TimeoutExpired, OSError):
        return None


_FENCE = re.compile(r"^```[a-zA-Z]*\s*$|^```\s*$", re.M)


def _split(raw: str) -> list[str]:
    parts = [p.strip() for p in raw.split(DELIM)]
    return [_FENCE.sub("", p).strip() for p in parts if len(p.strip()) >= 200]


def _batch(job: tuple[int, DocTemplate, int, int]) -> tuple[int, str, list[str]]:
    idx, tpl, seed, timeout = job
    rng = random.Random(seed)
    raw = _call_claude(tpl.llm_prompt(rng), timeout)
    return idx, tpl.name, (_split(raw) if raw else [])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=2000)
    ap.add_argument("--per-call", type=int, default=4)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--dry-run", action="store_true", help="faker only, no network")
    args = ap.parse_args()

    out_dir = _stamped_dir(args.count)
    (out_dir).mkdir(parents=True, exist_ok=True)
    print(f"corpus -> {out_dir}", flush=True)

    n_calls = (args.count + args.per_call - 1) // args.per_call
    # Round-robin so `--count` equal to the class count yields one of each.
    jobs = [(i, CLASSES[i % len(CLASSES)], args.seed + i, args.timeout)
            for i in range(n_calls)]

    produced: list[tuple[str, str, str]] = []   # (class, mode, text)
    if args.dry_run:
        for i, tpl, seed, _ in jobs:
            rng = random.Random(seed)
            for _ in range(args.per_call):
                produced.append((tpl.name, "faker", tpl.faker(rng)))
    else:
        started = time.perf_counter()
        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_batch, j): j for j in jobs}
            for fut in as_completed(futs):
                idx, name, docs = fut.result()
                tpl = next(t for t in CLASSES if t.name == name)
                if docs:
                    produced += [(name, "llm", d) for d in docs]
                else:
                    # A per-document LLM failure never fails the run.
                    rng = random.Random(args.seed + idx)
                    produced += [(name, "faker-fallback", tpl.faker(rng))
                                 for _ in range(args.per_call)]
                done += 1
                if done % 20 == 0 or done == len(jobs):
                    el = time.perf_counter() - started
                    print(f"  {done}/{len(jobs)} calls, {len(produced):,} docs, "
                          f"{el:.0f}s ({done / max(el, 1e-9) * 60:.1f} calls/min)",
                          flush=True)

    rng = random.Random(args.seed)
    manifest: list[dict[str, Any]] = []
    for i, (cls, mode, text) in enumerate(produced[:args.count]):
        tpl = next(t for t in CLASSES if t.name == cls)
        uid = f"{rng.getrandbits(48):012x}"
        rel = tpl.path_pattern.format(uid=uid, year=rng.randint(1998, 2022),
                                      date=f"{rng.randint(1998, 2022)}-{rng.randint(1, 12):02d}",
                                      q=rng.randint(1, 4))
        path = out_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        manifest.append({"index": i, "class": cls, "path": rel, "mode": mode,
                         "expected_tags": list(tpl.expected_tags),
                         "doc_has_pii": False, "bytes": len(text.encode("utf-8"))})
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n",
                                           encoding="utf-8")
    from collections import Counter
    print(f"\nwrote {len(manifest):,} documents")
    print("  by class:", dict(Counter(m['class'] for m in manifest)))
    print("  by mode :", dict(Counter(m['mode'] for m in manifest)))
    print(f"  bytes   : mean {sum(m['bytes'] for m in manifest) / max(len(manifest), 1):,.0f}")
    print(f"  manifest: {out_dir / 'manifest.json'}")
    print("\nNOT YET USABLE: run h2h_datagen_verify.py before any document of this "
          "corpus reaches training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
