"""The model packager: manifest, checksums, and tamper detection.

A PII/PHI model with unknown provenance is worse than useless -- someone will
point it at real patient data and have no way to answer "what is this and how
good is it?". These tests cover the parts of that answer that can go wrong
silently.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest


def load_packager():
    path = pathlib.Path(__file__).resolve().parents[1] / "training" / "package.py"
    if not path.exists():
        pytest.skip("training/package.py not present")
    spec = importlib.util.spec_from_file_location("_package", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_bundle(root: pathlib.Path) -> pathlib.Path:
    bundle = root / "bundle"
    bundle.mkdir()
    (bundle / "model.onnx").write_bytes(b"onnx graph")
    (bundle / "model.onnx.data").write_bytes(b"weights" * 1000)
    (bundle / "tokenizer.json").write_text('{"version": "1.0"}')
    (bundle / "model.json").write_text(json.dumps({
        "config": {"d_model": 192, "n_layers": 8, "vocab_size": 50368,
                   "num_labels": 111, "kernel_size": 5},
        "label_names": ["O"] + [f"B-t{i}" for i in range(110)],
        "teacher": "kalyan-ks/ettin-68m-nemotron-pii",
        "calibration": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "alpha": 0.7, "temperature": 3.0, "lr": 3e-3, "epochs": 4,
        "soft_scope": "word_homogeneous", "max_length": 512,
    }))
    return bundle


def test_a_freshly_built_package_verifies(tmp_path, capsys):
    pkg = load_packager()
    dest = pkg.build(fake_bundle(tmp_path), tmp_path / "dist",
                     "test-model", "1.2.3", None, "2026-01-01")
    assert pkg.verify(dest) == 0
    assert "test-model 1.2.3" in capsys.readouterr().out


def test_the_manifest_records_every_shipped_file(tmp_path):
    pkg = load_packager()
    dest = pkg.build(fake_bundle(tmp_path), tmp_path / "dist",
                     "test-model", "1.2.3", None, "2026-01-01")
    manifest = json.loads((dest / "MANIFEST.json").read_text())
    assert set(manifest["files"]) == {
        "model.onnx", "model.onnx.data", "tokenizer.json", "model.json"}
    assert all(len(f["sha256"]) == 64 for f in manifest["files"].values())
    assert manifest["model"]["calibrated"] is True
    assert manifest["model"]["parameters"] == 10_001_199   # the `l` geometry


def test_same_size_corruption_is_caught(tmp_path, capsys):
    """The failure mode the checksums exist for.

    Weights ship in a sibling model.onnx.data. Corrupt it *without changing
    its size* and the model still loads and still answers -- confidently, and
    differently. Measured on the real artifact, a flipped kilobyte turned an
    MRN into a USER_ID at 0.87 confidence, which is a silent PHI miss. Size
    checks alone would not see it.
    """
    pkg = load_packager()
    dest = pkg.build(fake_bundle(tmp_path), tmp_path / "dist",
                     "test-model", "1.2.3", None, "2026-01-01")
    target = dest / "model.onnx.data"
    before = target.stat().st_size
    data = bytearray(target.read_bytes())
    data[100:200] = bytes(b ^ 0xFF for b in data[100:200])
    target.write_bytes(bytes(data))
    assert target.stat().st_size == before, "the point is that size is unchanged"

    assert pkg.verify(dest) == 1
    assert "sha256" in capsys.readouterr().err


def test_a_missing_weight_file_is_caught(tmp_path, capsys):
    pkg = load_packager()
    dest = pkg.build(fake_bundle(tmp_path), tmp_path / "dist",
                     "test-model", "1.2.3", None, "2026-01-01")
    (dest / "model.onnx.data").unlink()
    assert pkg.verify(dest) == 1
    assert "missing" in capsys.readouterr().err


def test_an_unlisted_extra_file_is_caught(tmp_path, capsys):
    # A package is a closed set. An unlisted file is either tampering or a
    # build that shipped something nobody described.
    pkg = load_packager()
    dest = pkg.build(fake_bundle(tmp_path), tmp_path / "dist",
                     "test-model", "1.2.3", None, "2026-01-01")
    (dest / "surprise.onnx").write_bytes(b"?")
    assert pkg.verify(dest) == 1
    assert "not in the manifest" in capsys.readouterr().err


def test_verifying_something_that_is_not_a_package_fails_clearly(tmp_path, capsys):
    pkg = load_packager()
    assert pkg.verify(tmp_path) == 1
    assert "not a package" in capsys.readouterr().err


def test_building_from_an_incomplete_bundle_refuses(tmp_path):
    pkg = load_packager()
    bundle = fake_bundle(tmp_path)
    (bundle / "tokenizer.json").unlink()
    with pytest.raises(SystemExit, match="not a bundle"):
        pkg.build(bundle, tmp_path / "dist", "test-model", "1.2.3", None,
                  "2026-01-01")


def test_the_model_card_carries_the_limitations_that_matter(tmp_path):
    pkg = load_packager()
    dest = pkg.build(fake_bundle(tmp_path), tmp_path / "dist",
                     "test-model", "1.2.3", None, "2026-01-01")
    card = (dest / "MODEL_CARD.md").read_text().lower()
    for claim in ("synthetic training data", "demographic slice",
                  "credit card numbers are deliberately suppressed",
                  "cc by 4.0", "de-identification guarantee",
                  # The PII/PHI split has no external gold; a card that omits
                  # that is hiding the weakest link in the evaluation.
                  "no external gold"):
        assert claim in card, claim


def test_the_model_card_leads_with_recall_not_f1():
    """Ordering is a claim about what matters.

    For a PII/PHI scanner a missed identifier is a reportable incident and a
    false alarm is a reviewer-minute. A card whose headline metric is F1 tells
    the reader those are equivalent, which is wrong -- so recall and F2 come
    first and F1 follows.
    """
    pkg = load_packager()
    import tempfile, pathlib as pl
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pl.Path(tmp)
        dest = pkg.build(fake_bundle(tmp), tmp / "dist", "test-model", "1.2.3",
                         {"rule_tier": {"p": 0.9, "r": 0.9, "f1": 0.9,
                                        "f2": 0.9},
                          "model_tier": {"p": 0.9, "r": 0.9, "f1": 0.9,
                                         "f2": 0.9}},
                         "2026-01-01")
        card = (dest / "MODEL_CARD.md").read_text()
    header = next(line for line in card.splitlines()
                  if line.startswith("| |") and "recall" in line)
    assert header.index("recall") < header.index("F2") < header.index("F1")


@pytest.mark.parametrize("size,expected", [
    ({"d_model": 64, "n_layers": 4, "vocab_size": 50368, "num_labels": 111,
      "kernel_size": 5}, 3_249_583),
    ({"d_model": 128, "n_layers": 6, "vocab_size": 50368, "num_labels": 111,
      "kernel_size": 5}, 6_566_895),
])
def test_parameter_count_matches_the_torch_model(size, expected):
    # Derived arithmetically so packaging does not need torch; pinned against
    # the numbers StudentTagger.num_parameters() reports for the same configs.
    assert load_packager().parameter_count(size) == expected


def test_parameter_count_declines_to_guess_from_a_partial_config():
    assert load_packager().parameter_count({"d_model": 192}) is None
