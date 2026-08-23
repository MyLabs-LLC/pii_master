# Draft: licence request to AI4Privacy

`ai4privacy/pii-masking-300k` is licensed for academic research and
non-commercial use, and separately requires an explicit written licence for
"redistribution, uploading to databases, sharing through any medium, or the
creation and dissemination of derivative works". A model trained on it is a
derivative work, so **`pii-master-ner-l-mixed` cannot be distributed until
AI4Privacy grants one** — see docs/STAGE2_INTEGRATION.md, Acknowledgements.

The licence asks for "a detailed description of the intended use". Send to
**licensing@ai4privacy.com**. Fill in the bracketed fields before sending.

---

**Subject:** Licence request — derivative model trained on pii-masking-300k

Hello AI4Privacy team,

I am writing to request a licence covering a derivative work of
`ai4privacy/pii-masking-300k`, per the terms in your LICENSE.md.

**Who we are.** [Legal entity name], [jurisdiction]. [State plainly whether the
work is academic/non-commercial, whether the entity is a company, and what it
does — the licence distinguishes these and says commercial licensing requires
prior discussion.]

**What we built.** `pii_master` is an open-source (MIT) PII/PHI detection
library that runs on one CPU core. Its Stage 2 component is a 10M-parameter
dilated-CNN token tagger distilled from `kalyan-ks/ettin-68m-nemotron-pii` and
trained on `nvidia/Nemotron-PII` (CC BY 4.0). We trained a second variant on a
mixture of Nemotron-PII and the **English rows of pii-masking-300k**, with your
28 labels mapped into the Nemotron label space.

**What your data changed.** Our Nemotron-only model scored micro F1 0.934 on
its own holdout but only **0.385 strict span recall on your validation split** —
your corpus is structured JSON/XML key-value data and UK/EU-localised, and ours
had learned span boundaries from US narrative prose. Adding your English rows
raised that to **0.580 strict / 0.747 located**, and document-level recall from
0.870 to 0.924, with no regression on the original corpus. Your dataset is the
reason we know the model does not generalise, and the reason it now generalises
better.

**What we are asking for.** Permission to [distribute the trained weights
publicly under an open licence / distribute to named customers / use
commercially — pick and state precisely]. We are not asking to redistribute the
dataset itself, and none of your data is recoverable from the model artifact:
what ships is ONNX weights, a tokenizer and a label table.

**What we will do either way.** AI4Privacy is acknowledged in our published
evaluation write-up, as your licence stipulates. If a licence is not available,
the mixed model stays an internal research result and only our Nemotron-only
models are published — which is the current state.

Happy to share the evaluation harness, the label mapping, or the full results.

[Name, role, contact]
