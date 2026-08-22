"""End-to-end deep-mode latency on ONE core: rules + tokenizer + student + decode.

docs/DISTILLATION_PLAN.md sections 3 and 6 budget the student's ONNX forward
pass, which is the big term but not the whole bill. Serving `deep` mode also
pays for tokenization and span decoding, and those are real milliseconds on a
10 KB document. This measures the whole cascade on the documents the committed
rules benchmark already uses (pii_master.bench.generate_docs, same seed), so
the rules column here and `pii-master bench` agree.

Run it pinned to a single core, which is what the production container gets:

    RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \\
      taskset -c 0 python bench_e2e.py --onnx artifacts/student_xs.onnx
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAYON_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import ID2LABEL  # noqa: E402
from decode import decode_spans  # noqa: E402

from pii_master.bench import generate_docs  # noqa: E402
from pii_master.classify import scan_text  # noqa: E402

TEACHER_ID = "kalyan-ks/ettin-68m-nemotron-pii"


def percentile(values, q):
    values = sorted(values)
    return values[max(0, round(q * len(values)) - 1)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--bytes", type=int, default=10_000)
    ap.add_argument("--docs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args(argv)

    import onnxruntime as ort
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TEACHER_ID)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(args.onnx, opts, providers=["CPUExecutionProvider"])

    docs = generate_docs(args.seed, [args.bytes], args.docs)[args.bytes]
    stages = {"rules": [], "tokenize": [], "student": [], "decode": [], "total": []}
    tokens_seen = []

    for round_index in range(args.warmup + 1):
        measuring = round_index == args.warmup
        for text in docs:
            t0 = time.perf_counter()
            report = scan_text(text)
            t1 = time.perf_counter()
            enc = tokenizer(text, return_offsets_mapping=True, return_tensors="np")
            ids = enc["input_ids"].astype(np.int64)
            mask = enc["attention_mask"].astype(np.int64)
            t2 = time.perf_counter()
            logits = sess.run(None, {"input_ids": ids, "attention_mask": mask})[0]
            t3 = time.perf_counter()
            spans = decode_spans(text, enc["offset_mapping"][0],
                                 logits[0].argmax(-1), ID2LABEL)
            t4 = time.perf_counter()
            if measuring:
                stages["rules"].append((t1 - t0) * 1000)
                stages["tokenize"].append((t2 - t1) * 1000)
                stages["student"].append((t3 - t2) * 1000)
                stages["decode"].append((t4 - t3) * 1000)
                stages["total"].append((t4 - t0) * 1000)
                tokens_seen.append(ids.shape[1])
                del report, spans

    print(f"{args.docs} documents x {args.bytes:,} bytes "
          f"(~{int(statistics.fmean(tokens_seen)):,} tokens), 1 core, "
          f"{Path(args.onnx).name}")
    print(f"{'stage':>10} {'mean':>9} {'p50':>9} {'p95':>9} {'max':>9}")
    for name in ("rules", "tokenize", "student", "decode", "total"):
        v = stages[name]
        print(f"{name:>10} {statistics.fmean(v):>7.2f}ms {percentile(v, .50):>7.2f}ms "
              f"{percentile(v, .95):>7.2f}ms {max(v):>7.2f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
