# Prior art — who has already solved this, and what it cost them

Survey date **2026-08-22**. Purpose: decide what to adopt, what to avoid, and what
accuracy is actually reachable inside our **5 ms / 1 CPU core** budget (docs/DESIGN.md §5).

Verification: ✅ primary source read · 📄 secondary/summary of a primary source ·
🏷️ vendor self-report · ⚠️ no hard number found.

---

## 1. The finding that matters most

**Nobody in the clinical de-identification literature reports CPU inference latency.**
Not the i2b2/n2c2 2014 challenge, not NeuroNER, not the BERT-era papers, not the 2025–26
LLM papers. The one exception is a JMIR 2024 multi-system study that reports
notes/second — and the *fastest* system in it runs at ~41 ms/note, the slowest at 1 s/note
(📄 JMIR 2024;26:e55676).

So the honest framing of this project is not "we will beat i2b2 F1". It is:

> **document-level PHI classification at 100–170× the throughput of any published
> alternative, with explainable evidence spans.**

That position is unoccupied precisely because the field does not measure latency.

## 2. Measured CPU latency of everything we could otherwise adopt

| System | Params | CPU latency | Hardware | ✓ |
|---|--:|--:|---|:--:|
| **pii_master (this repo)** | — | **0.84 ms** p95 / 10 KB | 1 core, Xeon 2.8 GHz | ✅ |
| DataFog regex engine | — | 1.20 ms / 10 KB | Apple M5 Pro | ✅ |
| Presidio (spaCy) | — | 15.1 ms median; **174 ms / 10 KB** | M1 Pro / M5 Pro | ✅ |
| spaCy `en_core_web_sm` | 13 MB | 171 ms / 10 KB | Apple M5 Pro | ✅ |
| Piiranha (mDeBERTa) | 86M / 278M | 118.5 ms | Apple M1 Pro | ✅ |
| GLiNER v1 / GLiNER2 | 209M / 205M | 161 ms / 198 ms | Apple M1 Pro | ✅ |
| `openai/privacy-filter` | 1.5B (50M active) | ~357 ms | CPU | ✅ |
| DistilBERT ONNX **static int8** | 66M | 26.75 ms p95 @ seq 128 | **4 vCPU** | ✅ |
| TinyBERT-4L | 14.3M | 18 ms | Colab CPU (~2 vCPU) | ✅ |

**Our rules tier is already at parity with the fastest published PII engine.** Every
learned system on this list is **20–400× over our entire budget**.

**Quantization will not rescue an oversized model.** Published int8 gains are 1.9–2.9× on
typical Xeons (4.2–5.8× only with Granite Rapids AMX). You cannot quantize 120 ms into
5 ms. DeBERTa specifically degrades most under quantization — so Piiranha is doubly wrong
for us.

## 3. The accuracy ceiling, and one model that defines it

✅ **`kalyan-ks/ettin-68m-nemotron-pii`** (MIT, 68.5M params, ModernBERT/`ettin-encoder-68m`
base) reports **F1 0.9627 / P 0.9635 / R 0.9619** across all 55 entity types on the
`nvidia/Nemotron-PII` test split — the *same dataset we hold out on* in
docs/BASELINE_NEMOTRON.md. Verified from the model card's `model-index`.

That is the reference point for Stage 2, and the obvious **distillation teacher**: it is
small enough to run cheaply on a training GPU, MIT-licensed, and already fluent in our
evaluation dataset. It is **not** a candidate for serving — 68M ModernBERT over a whole
document will not fit 5 ms on one core.

Reachable quality by entity class, given the constraint:

| Class | Realistic | Basis |
|---|---|---|
| Checksummable (email, phone, SSN, card, IBAN) | **F1 0.93–1.00** — rules already reach it | ✅ our own baseline; Presidio EMAIL 0.93–0.996 |
| Person names, in-domain, gazetteer + gated tiny model | 0.65–0.80 | ✅ Piiranha 0.787, GLiNER 0.45–0.76 — all at 100–200 ms |
| Person names, **out-of-domain** | **0.15–0.45 — plan for this** | ✅ Piiranha 0.780 → 0.169 on financial text |
| Broad 48-type cross-domain | 0.14 off-the-shelf → ~0.65 fine-tuned | ✅ PIIBench |
| i2b2-level clinical PHI (0.94–0.98) | **not reachable in 5 ms** | 📄 those are 110–355M models at 100–1000 ms/note |

## 4. Cross-domain generalization is the field's actual unsolved problem

✅ **PIIBench** (arXiv 2604.15776): 8 published systems — Presidio, spaCy, BERT, XLM-R,
SpanMarker ×2, Piiranha, XtremeDistil — evaluated on 2.37M unified sequences.
**Every one scored span-level F1 below 0.14**, best 0.1385, with "zero recall on most
entity types". A plain DeBERTa fine-tuned on the same diverse data reaches 0.6476
(✅ arXiv 2605.25816), beating fancier hierarchical and curriculum variants — the authors'
conclusion is that **data diversity beats architecture**.

Corroborating drops: Piiranha 0.780 → 0.169 moving to financial text; `openai/privacy-filter`
0.855 → 0.464 (medical) → 0.038 (Arabic); Philter's 99.9% i2b2 recall → F1 0.49 on UK
neurosurgical notes, where it over-redacted.

**This validates our decision to hold out on Nemotron rather than trust the frozen
corpus** — and warns that even our external number will not survive a domain change.

## 5. Lessons the field has already paid for

1. **Context is not optional.** ✅ Microsoft's own transparency note: "without context, a
   ten-digit number is just a number"; their health de-id model "relies on both pre- and
   post-text", which is why they **advise against real-time use**. → *Our pre-scan windows
   trade away exactly this signal. Any Stage 2 window must carry left/right context, and
   we must measure the recall cost of the window width.*
2. **Recall-first is the field consensus** (✅ Azure Health targets **>95% recall** as the
   *primary* release criterion) — but recall-first *without validators* degenerates into
   over-redaction, which is how Philter lost to Presidio out-of-domain. Our
   checksum-validator design is the documented correct answer.
3. **95% recall is nowhere near enough at scale.** 📄 UCSF: at 130M notes, 5% miss ≈ tens
   of millions of leaked identifiers.
4. **Aggregate F1 hides the operational failure.** The field now recommends
   **document-level leakage rate** — the fraction of documents with ≥1 missed identifier —
   because one miss makes a whole document unsafe. *This is our natural headline metric:
   our product output is already a document label.*
5. **Entity-level, not mention-level.** 📄 TAB (Pilán et al., CL 48(4) 2022): an entity is
   only protected if **every** mention is masked, and direct identifiers (ERdi) should be
   scored separately from quasi-identifiers (ERqi). Our taxonomy already separates them.
6. **PII detectors carry demographic bias.** 📄 "Behind the Mask" (LTEDI@ACL 2022): three
   off-the-shelf maskers all showed significantly higher error rates on names associated
   with Black and Asian/Pacific Islander individuals, worst for Black women. → *A
   name-demographic eval slice must exist before Stage 2 ships names.*
7. **Never claim HIPAA de-identification.** 📄 Re-identification attacks recover ~25% of
   records from Safe-Harbor-compliant data; ✅ even Microsoft's certified health service
   disclaims Safe Harbor compliance. → *We say "detection and risk scoring", which
   docs/DESIGN.md already does.*
8. **Adding regex on top of a neural model is nearly worthless** (✅ measured +0.001 to
   +0.012 F1) **unless the tiers cover disjoint entity space** — which is why RECAP's
   regex+LLM hybrid *did* win by covering 300+ structured types the LLM was bad at.
   → *Measure Stage 2's marginal contribution per entity type, never in aggregate.*
9. **Vendor accuracy claims are unfalsifiable by construction.** Google, AWS and Azure
   publish taxonomies and disclaimers, no numbers. Nightfall publishes precision without
   recall. The only substantive cloud figure found is Azure Health's ">95% recall" goal.

## 6. Housekeeping fact

✅ **Presidio has left Microsoft.** It is now community-governed at
`github.com/data-privacy-stack/presidio`, docs at `presidio.dataprivacystack.org`, images
at `ghcr.io/data-privacy-stack/*` (the old MCR images are no longer updated). Still MIT.
Its *architecture* — recognizer registry, per-recognizer confidence, and a
`ContextAwareEnhancer` that boosts scores on nearby context words — is the pattern our
Stage 1 independently arrived at and is worth continuing to mirror. Its *runtime* is not
adoptable at our budget.

## 7. Decisions this survey forces

**Adopt**
- Keep the rules tier as primary; it is at the published state of the art for speed.
- Add a **gazetteer tier before any neural tier**: Aho-Corasick/FlashText is O(document)
  and independent of dictionary size, so a name/city/hospital gazetteer plus a context
  lexicon is sub-millisecond. This is the documented mechanism by which >50% of inputs
  bypass the model entirely in tiered production systems.
- Build Stage 2 as a **gated span classifier over 32–64 token candidate windows**, int8
  ONNX, `intra_op_num_threads=1` — not a document tagger. This is the only shape with any
  chance of fitting the residual budget.
- Use `kalyan-ks/ettin-68m-nemotron-pii` as accuracy ceiling **and** distillation teacher.
- Add **document-level leakage rate** and TAB-style entity-level recall to `eval`.
- Add a **name-demographic slice** to the corpus before Stage 2.

**Avoid**
- Presidio, GLiNER, Piiranha, `openai/privacy-filter` as *runtimes* (20–400× over budget).
  GLiNER additionally has a known-broken int8 ONNX path (urchade/GLiNER#218, open since
  2024-12).
- spaCy `en_core_web_sm` as a "fast fallback" — 171 ms/10 KB and F1 0.845 on OntoNotes.
- Trusting any in-domain F1, ours included, as a forecast.
- Narrowing pre-scan windows so far that the model loses the context it needs.

**Open question worth chasing:** 📄 "Fast DistilBERT on CPUs" (arXiv 2211.07715) reports
**1.27 ms/sample** at seq len 32 with 70–90% sparsity + int8 — the only sub-5 ms
transformer figure that exists. It was measured on a multi-socket Xeon, so it probably
does not transfer to 1 core; but if it holds at seq 64 on one core, it materially changes
Stage 2 feasibility. Read the PDF directly before designing the student.
