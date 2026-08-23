"""Export a trained student to ONNX and dynamic-int8, then time it on one core.

Serving config is deliberate: intra_op_num_threads=1 and inter_op_num_threads=1,
because the production container has a single core and thread contention on a
1-core cgroup makes things slower, not faster.
"""

from __future__ import annotations

import argparse
import json
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


def verify_export(model: StudentTagger, path: Path, tokens: int = 777) -> float:
    """Assert the ONNX graph agrees with the PyTorch model it came from.

    An export can go wrong quietly: a dynamic axis that is not actually
    dynamic, a BatchNorm exported in training mode, an opset that lowers an op
    differently. All three produce a graph that loads, runs, and returns
    plausible-looking logits -- and a model that has silently changed. The
    length is deliberately not one of the shapes used for the export dummy, so
    a frozen sequence axis is caught here rather than in production.
    """
    import numpy as np
    import onnxruntime as ort

    ids = torch.randint(0, 1000, (1, tokens), dtype=torch.long)
    mask = torch.ones((1, tokens), dtype=torch.long)
    model.eval()
    with torch.no_grad():
        expected = model(ids, mask).numpy()

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    actual = sess.run(None, {"input_ids": ids.numpy(),
                             "attention_mask": mask.numpy()})[0]

    if actual.shape != expected.shape:
        raise SystemExit(f"ONNX shape {actual.shape} != torch {expected.shape}")
    drift = float(np.abs(actual - expected).max())
    agree = float((actual.argmax(-1) == expected.argmax(-1)).mean())
    if drift > 1e-3 or agree < 1.0:
        raise SystemExit(
            f"ONNX export disagrees with the checkpoint: max |delta| {drift:.2e}, "
            f"argmax agreement {agree:.4f}. Do not ship this bundle."
        )
    print(f"  export verified: max |delta| {drift:.2e}, argmax agreement 1.0000 "
          f"at {tokens} tokens")
    return drift


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
        "size_mb": on_disk_mb(path),
    }


def on_disk_mb(path: Path) -> float:
    """Graph plus any external weight file.

    torch's exporter writes tensors past a size threshold to a sibling
    `.onnx.data`, so `path.stat().st_size` alone reports a 26 MB model as
    0.0 MB -- which is how a bundle gets shipped missing its weights.
    """
    total = path.stat().st_size
    external = path.with_suffix(path.suffix + ".data")
    if external.exists():
        total += external.stat().st_size
    return total / 1e6


def write_bundle(model: StudentTagger, meta: dict, dest: Path,
                 teacher_id: str) -> Path:
    """Write the three files pii_master.ner.load_bundle expects.

    A *bundle* is what ships: the fp32 ONNX graph, the tokenizer, and the label
    table that maps logit columns back to Nemotron labels. All three have to
    travel together -- a model.onnx with the wrong tokenizer produces confident
    garbage, silently -- so exporting them as a unit is the point.

    fp32, not int8, and that is measured rather than conventional: dynamic
    int8 quantization makes this model **12x slower** (1.14 ms -> 13.48 ms p95
    for `xs`), because it is normalisation- and activation-bound, not
    matmul-bound -- MatMul is 4.7% of its ONNX op profile.
    docs/DISTILLATION_PLAN.md section 2, reproduced in DISTILLATION_RESULTS
    section 4.
    """
    from transformers import AutoTokenizer

    dest.mkdir(parents=True, exist_ok=True)
    export_onnx(model, dest / "model.onnx")
    verify_export(model, dest / "model.onnx")
    tokenizer = AutoTokenizer.from_pretrained(teacher_id)
    # The serving path loads this with `tokenizers.Tokenizer.from_file`, which
    # is the same Rust tokenizer without the transformers dependency -- that is
    # what keeps the ML extra to onnxruntime + tokenizers.
    tokenizer.backend_tokenizer.save(str(dest / "tokenizer.json"))
    (dest / "model.json").write_text(json.dumps(meta, indent=2))
    return dest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", default="all", choices=[*LADDER, "all"])
    ap.add_argument("--checkpoint", help="trained .pt state_dict; random weights if omitted")
    ap.add_argument("--out-dir", default="artifacts")
    ap.add_argument("--tokens", type=int, nargs="+", default=[512, 2000])
    ap.add_argument("--bundle", metavar="DIR",
                    help="also write a serving bundle (model.onnx, "
                         "tokenizer.json, model.json) for pii_master.ner")
    ap.add_argument("--no-int8", action="store_true",
                    help="skip the int8 export and its benchmark row")
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sizes = list(LADDER) if args.size == "all" else [args.size]

    print(f"{'size':>5} {'variant':>6} {'MB':>7} {'tokens':>7} {'mean':>9} {'p95':>9}")
    for name in sizes:
        cfg = LADDER[name]
        meta = None
        if args.checkpoint:
            meta_path = Path(args.checkpoint).with_suffix(".json")
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                cfg = StudentConfig(**meta["config"])
        model = StudentTagger(cfg)
        if args.checkpoint:
            model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        fp32 = export_onnx(model, out / f"student_{name}.onnx")
        if args.bundle:
            if meta is None:
                raise SystemExit(
                    "--bundle needs --checkpoint and its sidecar .json "
                    "(the label table); random weights have no label space."
                )
            dest = write_bundle(model, meta, Path(args.bundle), meta["teacher"])
            print(f"  bundle -> {dest} "
                  f"({on_disk_mb(dest / 'model.onnx'):.1f} MB model + "
                  f"{(dest / 'tokenizer.json').stat().st_size / 1e6:.1f} MB tokenizer)")
        variants = [("fp32", fp32)]
        if not args.no_int8:
            variants.append(("int8", quantize(fp32, out / f"student_{name}.int8.onnx")))
        for variant, path in variants:
            for tokens in args.tokens:
                r = bench(path, tokens)
                print(f"{name:>5} {variant:>6} {r['size_mb']:>6.1f}M {tokens:>7} "
                      f"{r['mean_ms']:>7.2f}ms {r['p95_ms']:>7.2f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
