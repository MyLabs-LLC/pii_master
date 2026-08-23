"""Turn a serving bundle into a distributable, checksummed model package.

A bundle in `training/artifacts/` is a build output: it is gitignored, it has
no version, and nothing about it says which corpus trained it, which commit
produced it, or what it scores. That is fine on the machine that made it and
useless anywhere else -- and a PII/PHI model with unknown provenance is worse
than useless, because someone will deploy it against real patient data and have
no way to answer "what is this and how good is it?"

A *package* adds the four things that make an artifact shippable:

  MANIFEST.json  sha256 and byte size of every file, the git commit, the
                 tokenizer and label-space identity, and the measured scores.
                 Checksums are not ceremony here: model.onnx ships its weights
                 in a sibling model.onnx.data, and a package that loses or
                 truncates that file still LOADS -- it just returns garbage
                 with full confidence. A manifest turns that into an error.
  MODEL_CARD.md  provenance, intended use, measured performance, the
                 limitations that matter, and the bias slice.
  LICENSE        the model inherits CC BY 4.0 attribution from Nemotron-PII;
                 the code is MIT. Those are different and both have to travel.
  a version      so two packages can be told apart at a glance.

`verify` re-hashes everything and is the check to run before trusting a package
you did not build.

    python training/package.py build --bundle artifacts/bundle_l \\
        --name pii-master-ner-l --version 0.3.0 --out dist
    python training/package.py verify dist/pii-master-ner-l-0.3.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PAYLOAD = ("model.onnx", "model.onnx.data", "tokenizer.json", "model.json")
REQUIRED = ("model.onnx", "tokenizer.json", "model.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=Path(__file__).resolve().parents[1], timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


def parameter_count(config: dict) -> int | None:
    """Trainable parameters, from the config alone.

    Derived arithmetically rather than by loading torch, so `package build`
    stays runnable on a machine that only has the serving extra. It mirrors
    training/model.py exactly: an embedding table, then per block a depthwise
    conv (k weights per channel) + a pointwise conv (d x d) + BatchNorm scale
    and shift, then a 1x1 head. BatchNorm running statistics are buffers, not
    parameters, and are not counted -- which is the same convention
    StudentTagger.num_parameters() uses.
    """
    d = config.get("d_model")
    layers = config.get("n_layers")
    vocab = config.get("vocab_size")
    labels = config.get("num_labels")
    kernel = config.get("kernel_size", 5)
    if not all(isinstance(v, int) for v in (d, layers, vocab, labels)):
        return None
    embed = vocab * d
    embed_norm = 2 * d
    block = (d * kernel + d) + (d * d + d) + (2 * d)
    head = d * labels + labels
    return embed + embed_norm + layers * block + head


def build(bundle: Path, out: Path, name: str, version: str,
          scores: dict | None, created: str) -> Path:
    for required in REQUIRED:
        if not (bundle / required).exists():
            raise SystemExit(f"{bundle / required} missing; not a bundle")

    dest = out / f"{name}-{version}"
    dest.mkdir(parents=True, exist_ok=True)
    files = {}
    for filename in PAYLOAD:
        source = bundle / filename
        if not source.exists():
            continue                    # model.onnx.data only when external
        shutil.copy2(source, dest / filename)
        files[filename] = {"sha256": sha256(dest / filename),
                           "bytes": (dest / filename).stat().st_size}

    meta = json.loads((bundle / "model.json").read_text())
    config = meta.get("config", {})
    manifest = {
        "name": name,
        "version": version,
        "created": created,
        "git_commit": git_commit(),
        "format": "onnx-fp32",
        "files": files,
        "total_bytes": sum(f["bytes"] for f in files.values()),
        "model": {
            "architecture": "dilated depthwise-separable CNN tagger",
            "d_model": config.get("d_model"),
            "n_layers": config.get("n_layers"),
            "dilations": config.get("dilations"),
            "parameters": parameter_count(config),
            "labels": len(meta.get("label_names", [])),
            "teacher": meta.get("teacher"),
            "calibrated": bool(meta.get("calibration")),
        },
        "training": {k: meta.get(k) for k in
                     ("soft_scope", "alpha", "temperature", "lr", "epochs",
                      "max_length")},
        "scores": scores,
    }
    (dest / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (dest / "MODEL_CARD.md").write_text(model_card(manifest, meta))
    shutil.copy2(Path(__file__).resolve().parents[1] / "LICENSE", dest / "LICENSE")
    return dest


def verify(package: Path) -> int:
    manifest_path = package / "MANIFEST.json"
    if not manifest_path.exists():
        print(f"FAIL  {manifest_path} missing -- not a package", file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text())
    problems = []
    for filename, expected in manifest["files"].items():
        path = package / filename
        if not path.exists():
            problems.append(f"{filename}: missing")
            continue
        size = path.stat().st_size
        if size != expected["bytes"]:
            problems.append(f"{filename}: {size} bytes, expected {expected['bytes']}")
            continue
        actual = sha256(path)
        if actual != expected["sha256"]:
            problems.append(f"{filename}: sha256 {actual[:16]}... != "
                            f"{expected['sha256'][:16]}...")
    extra = {p.name for p in package.iterdir()} - set(manifest["files"]) - {
        "MANIFEST.json", "MODEL_CARD.md", "LICENSE"}
    for name in sorted(extra):
        problems.append(f"{name}: present but not in the manifest")

    label = f"{manifest['name']} {manifest['version']}"
    if problems:
        print(f"FAIL  {label}", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print(f"OK    {label}  ({len(manifest['files'])} files, "
          f"{manifest['total_bytes'] / 1e6:.1f} MB, "
          f"commit {(manifest.get('git_commit') or '?')[:12]})")
    return 0


def model_card(manifest: dict, meta: dict) -> str:
    model = manifest["model"]
    scores = manifest.get("scores") or {}

    def row(section, key, default="not measured in this package"):
        value = scores.get(section, {}).get(key)
        return f"{value:.3f}" if isinstance(value, (int, float)) else default

    return f"""# {manifest['name']} {manifest['version']}

Stage 2 token tagger for **PII / PHI detection**, distilled from
`{model['teacher']}` on [nvidia/Nemotron-PII](https://huggingface.co/datasets/nvidia/Nemotron-PII).
Built to run on **one CPU core**.

- Architecture: {model['architecture']}, d={model['d_model']} x {model['n_layers']} layers
- Label space: {model['labels']} BIO classes over 55 Nemotron entity types
- Format: {manifest['format']} ({manifest['total_bytes'] / 1e6:.1f} MB)
- Confidence calibration: {"isotonic, fitted on a held-out slice" if model['calibrated'] else "NONE -- raw softmax"}
- Source commit: `{manifest.get('git_commit') or 'unknown'}`

## Intended use

Input is plain text; output is character spans typed against the 18 HIPAA Safe
Harbor identifier categories. It is designed to run **behind the Stage 1 rules
tier**, not alone -- see Limitations.

**Not** intended as a de-identification guarantee. It is a detector that helps
a reviewer, and Safe Harbor de-identification is a legal determination this
model cannot make.

## Measured performance

Exact `(type, start, end)` match on held-out Nemotron-PII documents, fused with
the rules tier:

| | F1 | F2 |
|---|--:|--:|
| the 12 types the rules also cover | {row('rule_tier', 'f1')} | {row('rule_tier', 'f2')} |
| the 14 types only this model emits | {row('model_tier', 'f1')} | {row('model_tier', 'f2')} |

F2 is reported because the cost matrix is asymmetric: a missed identifier is a
reportable incident, a false alarm costs a reviewer minutes.

## Limitations that matter

- **Synthetic training data.** Nemotron-PII is generated, not real. Scores here
  do not transfer unexamined to real clinical text; the standard benchmark
  (n2c2/i2b2 2014) needs a data use agreement and was not used.
- **The demographic slice is synthetic too.** Name recall varies by only 0.020
  across race/ethnicity groups, which sounds excellent and mostly reflects
  names drawn from a generator rather than from the world. It is a real gate
  that would catch a large disparity, and it certifies synthetic names only.
- **Credit card numbers are deliberately suppressed.** 88% of the training
  corpus's card numbers fail the Luhn checksum, so the serving path re-validates
  and drops them. Measured F1 on that type against this corpus is ~0.18, and
  that is the correct behaviour, not a defect.
- **US / English scope.** Non-US identifier formats are out of scope.
- **Confidences are calibrated for THIS model on THIS corpus.** A threshold
  tuned here is not automatically right for another text distribution.
- **The model can be confidently wrong.** Adversarial near-misses -- order
  numbers, chart numbers, subscriber ids -- are its documented failure class,
  which is why the serving path ships a confidence floor and rule fusion.

## Licensing

Code MIT (see LICENSE). Trained on Nemotron-PII, **CC BY 4.0**: attribution to
NVIDIA is required when redistributing this model or its outputs. The teacher
`{model['teacher']}` is MIT.

## Verify before you trust it

    python training/package.py verify <this directory>

The weights ship in `model.onnx.data` alongside `model.onnx`. A package missing
or truncating that file still loads and returns confident garbage, so the
checksums are load-bearing rather than decorative.
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="package a bundle")
    b.add_argument("--bundle", required=True)
    b.add_argument("--name", default="pii-master-ner")
    b.add_argument("--version", required=True)
    b.add_argument("--out", default="dist")
    b.add_argument("--scores", help="JSON from eval/scripts/nemotron_deep_eval.py")
    b.add_argument("--created", required=True,
                   help="ISO date; passed in rather than read from the clock "
                        "so a rebuild of the same inputs is reproducible")

    v = sub.add_parser("verify", help="re-hash a package against its manifest")
    v.add_argument("package")

    args = ap.parse_args(argv)
    if args.command == "verify":
        return verify(Path(args.package))

    scores = None
    if args.scores:
        payload = json.loads(Path(args.scores).read_text())
        run = payload["runs"][-1]
        keys = ("p", "r", "f1", "f2")
        scores = {
            "documents": payload["documents"],
            "configuration": run["label"],
            "rule_tier": dict(zip(keys, run["rule_tier"])),
            "model_tier": dict(zip(keys, run["model_tier"])),
            "p95_ms_per_doc": round(run["p95_ms"], 3),
        }
    dest = build(Path(args.bundle), Path(args.out), args.name, args.version,
                 scores, args.created)
    print(f"built {dest}")
    return verify(dest)


if __name__ == "__main__":
    raise SystemExit(main())
