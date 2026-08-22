"""Export a trained student to ONNX and dynamic-int8, then time it on one core.

Serving config is deliberate: intra_op_num_threads=1 and inter_op_num_threads=1,
because the production container has a single core and thread contention on a
1-core cgroup makes things slower, not faster.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from model import LADDER, StudentConfig, StudentTagger


def export_onnx(model: StudentTagger, path: Path, seq_len: int = 512) -> Path:
    model.eval()
    dummy_ids = torch.randint(0, 1000, (1, seq_len), dtype=torch.long)
    dummy_mask = torch.ones((1, seq_len), dtype=torch.long)
    torch.onnx.export(
        model, (dummy_ids, dummy_mask), str(path),
        input_names=["input_ids", "attention_mask"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                      "attention_mask": {0: "batch", 1: "seq"},
                      "logits": {0: "batch", 1: "seq"}},
        opset_version=17,
    )
    return path


def quantize(src: Path, dst: Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
    return dst


def bench(path: Path, tokens: int, runs: int = 30, warmup: int = 5) -> dict:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])

    ids = np.random.randint(0, 1000, (1, tokens)).astype(np.int64)
    mask = np.ones((1, tokens), dtype=np.int64)
    feed = {"input_ids": ids, "attention_mask": mask}

    for _ in range(warmup):
        sess.run(None, feed)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, feed)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {
        "tokens": tokens,
        "mean_ms": statistics.fmean(times),
        "p50_ms": statistics.median(times),
        "p95_ms": times[max(0, round(0.95 * len(times)) - 1)],
        "size_mb": path.stat().st_size / 1e6,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", default="all", choices=[*LADDER, "all"])
    ap.add_argument("--checkpoint", help="trained .pt state_dict; random weights if omitted")
    ap.add_argument("--out-dir", default="artifacts")
    ap.add_argument("--tokens", type=int, nargs="+", default=[512, 2000])
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sizes = list(LADDER) if args.size == "all" else [args.size]

    print(f"{'size':>5} {'variant':>6} {'MB':>7} {'tokens':>7} {'mean':>9} {'p95':>9}")
    for name in sizes:
        cfg = LADDER[name]
        model = StudentTagger(cfg)
        if args.checkpoint:
            model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        fp32 = export_onnx(model, out / f"student_{name}.onnx")
        int8 = quantize(fp32, out / f"student_{name}.int8.onnx")
        for variant, path in (("fp32", fp32), ("int8", int8)):
            for tokens in args.tokens:
                r = bench(path, tokens)
                print(f"{name:>5} {variant:>6} {r['size_mb']:>6.1f}M {tokens:>7} "
                      f"{r['mean_ms']:>7.2f}ms {r['p95_ms']:>7.2f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
