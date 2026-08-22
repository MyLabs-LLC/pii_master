# Stage 2 distillation plan — ettin-68m → a CNN student that fits one CPU core

**Goal.** Distil `kalyan-ks/ettin-68m-nemotron-pii` (MIT, 68.5M params, reported
F1 0.963 on the Nemotron-PII test split) into a student small enough to tag a whole
document on a single CPU core, covering the **full 55-label Nemotron taxonomy**.

**Serving modes** (as chosen): a default **fast** path and an opt-in **deep** path.

| mode | tiers | budget |
|---|---|---|
| `fast` (default) | rules only | ≤ 5 ms / 10 KB — currently **0.93 ms** |
| `deep` (`--deep`) | rules + student | ≤ 25 ms / 10 KB |

Everything below is **measured on a deliberately slow reference box** (shared virtualised
Xeon @ 2.80 GHz, 1 core, ONNX Runtime 1.29, `intra_op_num_threads=1`). Your 4080 laptop's
CPU is single-core faster, so treat these as an upper bound on latency.

---

## 1. Why the student is not a small transformer

The budget arithmetic kills the obvious idea before any training happens. A transformer
body costs ≈ `12 · L · d²` MACs per token. A 10 KB English document is ≈ 2,000 tokens.
At a measured ~7 G MAC/s on one core of the reference box:

| candidate | MACs/token | tokens affordable in 4 ms |
|---|--:|--:|
| ettin-17m (L≈7, d≈256) | 5,505 k | **≈ 5** |
| 2-layer d=256 | 1,573 k | ≈ 18 |
| 2-layer d=128 | 393 k | ≈ 71 |
| **depthwise-separable CNN, d=64, 4 layers** | **25 k** | **≈ 1,100** |

Even ettin-17m — the smallest model in the Ettin ladder — can read about five tokens of a
two-thousand-token document. **No transformer of any size reads a whole document in this
budget.** A depthwise-separable convolution costs ≈ `d² + k·d` per token instead of
`12·L·d²`, which is the ~100× that makes the problem tractable.

Receptive field comes from **dilation**, not attention: kernel 5 with dilations 1,2,4,8
gives a 61-token field, comfortably more than the cue→value distances PII needs (the rules
tier works in a 40-character window, and 96.4% of Nemotron DOBs have their cue inside it).

## 2. Three measured findings that shaped the design

These came out of profiling, and two of them contradict the standard advice.

**a) int8 quantization makes this model 5–6× SLOWER. Do not ship it.**

| size | fp32 | dynamic int8 |
|---|--:|--:|
| xs (d=64) | **2.82 ms** | 37.45 ms |
| s (d=96) | **5.62 ms** | 57.75 ms |
| m (d=128, 6L) | **9.53 ms** | 116.53 ms |

Quantizing only MatMul/Gemm avoids the penalty but buys nothing (6.79 ms vs 6.89 ms). The
reason is (c) below. Ship **fp32**; revisit int8 only if a future student becomes
matmul-bound.

**b) The bottleneck is normalisation and activation, not arithmetic.** ONNX op profile of
the first design (d=64, 2,000 tokens, 7.6 ms total):

| op | share |
|---|--:|
| Conv | 27.6% |
| SkipLayerNormalization | 17.3% |
| LayerNormalization | 13.1% |
| **Erf (GELU)** | **10.8%** |
| Transpose | 7.4% |
| Mul / Add / Div | 15.6% |
| **MatMul** | **4.7%** |

MatMul is 4.7% of runtime — which is precisely why quantizing weights does nothing.

**c) Rebuilding the block around that profile gave a 2.4× speedup at identical FLOPs**
(6.69 ms → 2.76 ms for 2,000 tokens):

- **BatchNorm instead of LayerNorm** — BN folds into the preceding convolution at
  inference and costs nothing; LayerNorm + its SkipLayerNorm fusion was 30% of runtime.
- **ReLU instead of GELU** — `Erf` alone was 10.8%, with a Mul/Div tail behind it.
- **Channel-first `(B, D, T)` end to end**, head as a 1×1 convolution — one transpose in,
  one out, instead of two per block.

## 3. The size ladder (measured, 2,000 tokens, 1 core, fp32)

| size | d | layers | params | MACs/token | mean | p95 | fits |
|---|--:|--:|--:|--:|--:|--:|---|
| **xs** | 64 | 4 | 3.25 M | 24.8 k | **2.82 ms** | **3.08 ms** | `fast` **and** `deep` |
| **s** | 96 | 4 | 4.89 M | 49.4 k | 5.62 ms | 7.77 ms | `deep` |
| **m** | 128 | 6 | 6.57 M | 116.4 k | 9.53 ms | 10.58 ms | `deep` |

Parameters are dominated by the embedding table (50,368 × d), which is a **lookup** — it
costs memory, not MACs. The conv body is only 0.03–0.12 M parameters.

`xs` already fits the strict 5 ms end-to-end contract on the *slow* reference box
(0.93 ms rules + 3.08 ms student = 4.01 ms). On your hardware it should have room to
spare — which means the fast/deep split may end up being `xs` vs `m` rather than
rules-only vs model.

## 4. Method

**Teacher.** `kalyan-ks/ettin-68m-nemotron-pii`. **The student reuses the teacher's
tokenizer**, so teacher and student logits align 1:1 per token and distillation needs no
alignment logic at all.

**Loss.** `α · KL(student ‖ teacher, T) + (1−α) · CE(student, gold BIO)`, defaults
α = 0.7, T = 3.0. Nemotron ships gold spans, so this is distillation *plus* supervision:
the teacher contributes dark knowledge (which confusions are plausible), the gold labels
keep it anchored.

**Label space.** 55 types × BI + O = **111 classes**, frozen in sorted order in
`training/data.py` and asserted to match the dataset exactly.

> **The one bug that would silently ruin the run:** the teacher's label ids are not ours.
> `build_label_permutation` remaps teacher logits into our order from the teacher's own
> `config.id2label`, and **hard-fails** if the teacher does not cover our label space.
> It is unit-tested against identity, shuffled, and missing-label cases. Never train
> through a warning here — scrambled targets look like slow convergence, not an error.

## 5. Runbook for your workstation

```bash
# 0. Environment (CUDA build of torch for the 4080)
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers pyarrow onnx onnxruntime onnxscript

# 1. Data (~300 MB, do not commit it)
mkdir -p ~/nemotron && cd ~/nemotron
for s in train test; do
  curl -L -o $s-00000-of-00001.parquet \
    "https://huggingface.co/datasets/nvidia/Nemotron-PII/resolve/main/data/$s-00000-of-00001.parquet"
done

# 2. Smoke run first — 2,000 rows, one epoch. Confirms the label permutation,
#    tokenisation alignment and VRAM headroom before you commit hours.
cd /path/to/pii_master/training
python train.py --data-dir ~/nemotron --size xs --limit 2000 --epochs 1

# 3. Full run. Batch 16 x accum 4 = effective 64 at seq 512; ~9 GB of your 12 GB.
python train.py --data-dir ~/nemotron --size xs --epochs 3
python train.py --data-dir ~/nemotron --size m  --epochs 3   # the deep-mode student

# 4. Export + measure on YOUR cpu (this is the number that decides the modes)
python export.py --size xs --checkpoint artifacts/student_xs.pt --tokens 512 2000
python export.py --size m  --checkpoint artifacts/student_m.pt  --tokens 512 2000
```

Expected wall-clock on a 12 GB 4080 laptop: the teacher is only 68M and runs under
`no_grad`, so a full pass over 100k documents at seq 512 should be roughly **25–45 min per
epoch**, i.e. ~1.5–2.5 h for three epochs. If VRAM is tight, drop `--batch-size` to 8 and
raise `--accum` to 8 (same effective batch). ⚠️ These are estimates — I could not measure
GPU time here.

## 6. Acceptance gates

A student ships only if **all** of these hold. They reuse the harnesses already in the
repo, so none of this is new tooling.

1. **Latency.** `deep` mode p95 ≤ 25 ms per 10 KB on your CPU, `intra_op_num_threads=1`.
   If a student also lands under 4 ms it is promoted into `fast` mode.
2. **It must beat the rules.** Span F1 on the Nemotron holdout must exceed the committed
   rules-only baseline (`docs/BASELINE_NEMOTRON.md`: micro P 0.854 / R 0.732 / F1 0.788 on
   the 12 mapped types) **on those same 12 types**. The 43 new types are upside, not an
   excuse for regressing the ones that already work.
3. **No regression on the frozen corpus.** `pii-master eval --fail-under` stays green —
   in particular PHI recall 1.00 and the zero false-PHI probes from Track A.
4. **Checksum types must not degrade.** Fusion keeps rule spans authoritative wherever a
   validator applies. Watch `CREDIT_CARD` specifically: 88% of Nemotron's gold cards fail
   Luhn, so the student *will* learn to emit invalid card spans. Rules must win there, or
   external precision drops.
5. **A name-demographic slice exists** before names ship. Published PII maskers show
   significantly higher error rates on names associated with Black and Asian/Pacific
   Islander individuals (docs/PRIOR_ART.md §5f). Ship the eval slice with the feature.

If the student cannot clear gate 2, **do not ship it**. The cascade is additive: rules
alone are already a defensible product.

## 7. Integration (after training, separate change)

- `OnnxNerDetector` implementing the existing `Detector` protocol — it plugs into
  `Pipeline` with no changes to Stage 3.
- **Fusion policy**, named in `pipeline.py` rather than left as a comment: checksum-validated
  rule spans outrank model spans on overlap; model spans of checksummed types are
  re-validated before they may carry high confidence.
- `onnxruntime` becomes an **optional extra** (`pip install pii-master[ml]`). The default
  install stays zero-dependency and rules-only, which is what keeps `fast` mode honest.
- **Taxonomy decision required:** the student emits 55 labels, we model 11. The other 44
  need either new `EntityType` members with HIPAA rows, or a profile-gated channel.
  Recommended split, matching `crosswalk.py`'s existing grouping: adopt the HIPAA-mapped
  ones (`fax_number`, `bank_routing_number`, `street_address`, `city`, `postcode`, names,
  …); keep `secrets` (password, api_key, cvv, pin, cookie) and `gdpr_special`
  (race_ethnicity, religious_belief, political_view, sexuality) behind M3 policy profiles
  so they never enter the default HIPAA output.

## 8. What is already done vs what you run

| | status |
|---|---|
| `training/model.py` — student, profiled design, size ladder | ✅ written, forward pass verified |
| `training/data.py` — BIO alignment, 55 labels | ✅ written; word-level BIO + word-start index added during the run |
| `training/train.py` — distillation loop | ✅ written; label permutation now pads the teacher's 4 absent columns |
| `training/export.py` — ONNX + int8 + 1-core benchmark | ✅ written and run |
| GPU training run | ✅ **done** — `xs` and `m`, 3 epochs each, ~5 min/epoch on a 4080 |
| `training/eval_student.py` — holdout scoring + fusion policies | ✅ added; gate 2 |
| `training/eval_names.py` — name-demographic slice | ✅ added; gate 5 |
| `training/bench_e2e.py` — whole-cascade latency on one core | ✅ added; gate 1 |
| All five acceptance gates | ✅ **pass** — see [DISTILLATION_RESULTS.md](DISTILLATION_RESULTS.md) |
| `OnnxNerDetector` + fusion | ⬜ next change; results section 8 lists what it must carry |

**Results:** [DISTILLATION_RESULTS.md](DISTILLATION_RESULTS.md). Headline: the `m`
student plus checksum-first fusion scores **F1 0.901** on the Nemotron holdout against
the rules-only baseline's 0.788, at **8.0 ms p95** per 10 KB document on one core
against a 25 ms `deep` budget.

⚠️ The limitation this section used to carry — "no run has touched real data" — is
closed. What replaced it is smaller and specific: the learning rate was never swept
(3e-3 with OneCycle converged on the first attempt), α and T were never varied, and
three epochs was the plan's number rather than a measured optimum. The training loss was
still falling when the schedule ended, so a longer run is the cheapest untried
improvement.
