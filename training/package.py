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
          scores: dict | None, created: str,
          document_scores: list | None = None) -> Path:
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
        "document_scores": document_scores,
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
    """The card, generated from the manifest so it cannot drift from the model.

    Ordering is deliberate. **Recall and F2 lead; F1 follows.** For a PII/PHI
    scanner the two errors do not cost the same -- a missed medical record
    number is a reportable incident, a false alarm is a reviewer-minute -- and
    a card whose headline is F1 quietly tells the reader those are equivalent.
    F1 is still shown, because precision is what keeps a scanner switched on.
    """
    model = manifest["model"]
    scores = manifest.get("scores") or {}
    doc = manifest.get("document_scores") or []

    def cell(section, key):
        value = scores.get(section, {}).get(key)
        return f"{value:.3f}" if isinstance(value, (int, float)) else "n/a"

    lines = [
        f"# {manifest['name']} {manifest['version']}",
        "",
        "Token tagger for **PII / PHI detection**, distilled from",
        f"`{model['teacher']}` on "
        "[nvidia/Nemotron-PII](https://huggingface.co/datasets/nvidia/Nemotron-PII).",
        "Runs on **one CPU core**.",
        "",
        f"- {model['architecture']}, d={model['d_model']} x {model['n_layers']} "
        f"layers, {(model['parameters'] or 0) / 1e6:.2f}M parameters",
        f"- {model['labels']} BIO classes over 55 Nemotron entity types, "
        "crosswalked to 25 HIPAA-mapped types",
        f"- {manifest['format']}, {manifest['total_bytes'] / 1e6:.1f} MB",
        "- Confidence calibration: " + ("**isotonic, per entity type**"
                                        if model["calibrated"] else
                                        "**NONE** -- raw softmax, do not "
                                        "threshold these scores"),
        f"- Source commit: `{manifest.get('git_commit') or 'unknown'}`",
        "",
        "## Intended use",
        "",
        "Input is plain text; output is character spans typed against the 18",
        "HIPAA Safe Harbor identifier categories. It is designed to run "
        "**behind the",
        "Stage 1 rules tier**, which supplies checksum-validated spans it "
        "cannot beat,",
        "and which suppresses failure classes it re-introduces on its own "
        "(see Limitations).",
        "",
        "**Not a de-identification guarantee.** It is a detector that helps a "
        "reviewer.",
        "Safe Harbor de-identification is a legal determination this model "
        "cannot make.",
        "",
        "## How good is it?",
        "",
        "### At the level you act on: does this document contain PII?",
        "",
    ]
    if doc:
        lines += [
            "| configuration | **recall** | documents missed | false alarms |",
            "|---|--:|--:|--:|",
        ]
        for row in doc:
            alarm = ("n/a" if row.get("false_alarm_rate") is None else
                     f"{row['false_alarm_rate']:.3f}")
            star = "**" if row.get("shipped") else ""
            lines.append(
                f"| {star}{row['mode']}{star} | {star}{row['recall']:.4f}{star} "
                f"| {row['missed']:,} of {row['sensitive']:,} | {alarm} |")
        lines += [
            "",
            f"Measured on {doc[0]['documents']:,} held-out Nemotron documents; "
            "a document counts as",
            "sensitive if it carries a gold span of a type we model. False "
            "alarms are measured",
            f"on {doc[0].get('negatives', 0)} **adversarial** negatives -- order "
            "numbers, chart numbers,",
            "subscriber ids -- not on easy ones.",
            "",
        ]
    else:
        lines += ["Not measured in this package.", ""]

    lines += [
        "### At the span level: is the tag right?",
        "",
        "Exact `(type, start, end)` match, fused with the rules tier.",
        "**F2 weights recall four times as heavily as precision**, which is "
        "closer to this",
        "system's cost matrix than F1; both are shown.",
        "",
        "| | recall | **F2** | F1 | precision |",
        "|---|--:|--:|--:|--:|",
        f"| the 12 types the rules also cover | {cell('rule_tier', 'r')} "
        f"| **{cell('rule_tier', 'f2')}** | {cell('rule_tier', 'f1')} "
        f"| {cell('rule_tier', 'p')} |",
        f"| the 14 types only this model emits | {cell('model_tier', 'r')} "
        f"| **{cell('model_tier', 'f2')}** | {cell('model_tier', 'f1')} "
        f"| {cell('model_tier', 'p')} |",
        "",
    ]

    per_type = scores.get("per_type") or {}
    if per_type:
        lines += ["<details><summary>Per type</summary>", "",
                  "| type | gold | recall | F2 | F1 | precision |",
                  "|---|--:|--:|--:|--:|--:|"]
        for name in sorted(per_type, key=lambda k: per_type[k].get("f2", 0)):
            row = per_type[name]
            lines.append(
                f"| `{name}` | {row.get('gold', 0):,} | {row.get('recall', 0):.3f} "
                f"| {row.get('f2', 0):.3f} | {row.get('f1', 0):.3f} "
                f"| {row.get('precision', 0):.3f} |")
        lines += ["", "</details>", ""]

    lines += [
        "## Limitations that matter",
        "",
        "- **These scores are for Nemotron-PII, and they do NOT generalise.** "
        "Measured on",
        "  ai4privacy/pii-masking-300k -- a different corpus, label space, "
        "document style",
        "  and locale -- in-scope strict span recall is **0.385** against "
        "0.914 here, and",
        "  document-level recall **0.870** against 0.998. Format-anchored "
        "types transfer",
        "  intact (EMAIL 0.943, IP 0.988); learned semantic types collapse "
        "(names and",
        "  addresses ~0.30, mostly boundary errors on structured JSON text). "
        "Deep mode",
        "  still roughly doubles the rules on that corpus, so the cascade "
        "earns its place --",
        "  but budget for the lower number on text unlike the training set.",
        "- **Synthetic training data.** Nemotron-PII is generated, not real, "
        "and the",
        "  standard clinical benchmark (n2c2/i2b2 2014) requires a data use "
        "agreement and",
        "  was not used. No number here describes real clinical text.",
        "- **The demographic slice is synthetic too.** Name recall varies by "
        "only 0.020",
        "  across race/ethnicity groups, which sounds excellent and mostly "
        "reflects names",
        "  drawn from a generator rather than from the world. It is a real gate "
        "that would",
        "  catch a large disparity; it certifies synthetic names only.",
        "- **The PII-vs-PHI split has no external gold.** Nemotron has no "
        "document labels",
        "  and no medical-context annotation, so that boundary is only scored "
        "on a",
        "  39-document authored corpus. It is the weakest link in the "
        "evaluation.",
        "- **Credit card numbers are deliberately suppressed.** 88% of the "
        "training",
        "  corpus's cards fail the Luhn checksum, so the serving path "
        "re-validates and",
        "  drops them. Measured F1 on that type is ~0.18 against this corpus, "
        "and that is",
        "  correct behaviour rather than a defect.",
        "- **US / English scope.** Non-US identifier formats are out of scope.",
        "- **Calibrated for THIS model on THIS corpus.** A threshold tuned here "
        "is not",
        "  automatically right for another text distribution.",
        "- **It can be confidently wrong.** Adversarial near-misses are its "
        "documented",
        "  failure class, which is why the serving path ships a confidence "
        "floor, checksum",
        "  re-validation and rule fusion. Do not run it bare.",
        "",
        "## Licensing",
        "",
        "Code MIT (see LICENSE). Trained on Nemotron-PII, **CC BY 4.0**: "
        "attribution to",
        "NVIDIA is required when redistributing this model or its outputs. The "
        "teacher",
        f"`{model['teacher']}` is MIT.",
        "",
        "## Verify before you trust it",
        "",
        "    python training/package.py verify <this directory>",
        "",
        "The weights ship in `model.onnx.data` beside `model.onnx`. A package "
        "whose weights",
        "are corrupted **without changing their size** still loads and still "
        "answers: on the",
        "real artifact a flipped kilobyte turned an `MRN` into a `USER_ID` at "
        "0.87",
        "confidence, which is a silent PHI miss. Only the checksum catches "
        "that.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="package a bundle")
    b.add_argument("--bundle", required=True)
    b.add_argument("--name", default="pii-master-ner")
    b.add_argument("--version", required=True)
    b.add_argument("--out", default="dist")
    b.add_argument("--scores", help="JSON from eval/scripts/nemotron_deep_eval.py")
    b.add_argument("--doc-scores", help="JSON from eval/scripts/document_eval.py")
    b.add_argument("--shipped-threshold", type=float,
                   help="mark this row of --doc-scores as the shipped default")
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
            "per_type": run.get("per_type"),
            "p95_ms_per_doc": round(run["p95_ms"], 3),
        }
    document_scores = None
    if args.doc_scores:
        document_scores = json.loads(Path(args.doc_scores).read_text())
        for row in document_scores:
            row.pop("missed_examples", None)
            row.pop("labels", None)
            if (args.shipped_threshold is not None
                    and row.get("threshold") == args.shipped_threshold):
                row["shipped"] = True

    dest = build(Path(args.bundle), Path(args.out), args.name, args.version,
                 scores, args.created, document_scores)
    print(f"built {dest}")
    return verify(dest)


if __name__ == "__main__":
    raise SystemExit(main())
