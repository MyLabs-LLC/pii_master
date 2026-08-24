"""The ai4privacy -> Nemotron label mapping used to build the training mixture.

A wrong entry here is silent: training just gets worse, or a type quietly
learns the wrong concept. The SOCIALNUMBER case in particular has already
caused one real bug on the evaluation side, where collapsing US SSNs and UK
national numbers into one type scored 854 correct detections as errors.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest


def load():
    path = pathlib.Path(__file__).resolve().parents[1] / "training" / "ai4privacy.py"
    if not path.exists():
        pytest.skip("training/ai4privacy.py not present")
    spec = importlib.util.spec_from_file_location("_ai4privacy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def nemotron_labels():
    pytest.importorskip("torch")
    path = pathlib.Path(__file__).resolve().parents[1] / "training" / "data.py"
    spec = importlib.util.spec_from_file_location("_data", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return set(module.NEMOTRON_LABELS)


def test_every_target_exists_in_the_nemotron_label_space():
    """The mixture keeps the 111-class head, so every target must be a class.

    Extending the head instead would leave new classes unsupervised by the
    teacher -- it emits columns over Nemotron's inventory and knows nothing
    else -- so the soft-target half of the loss would go silent exactly where
    the new data is meant to teach something.
    """
    module = load()
    valid = nemotron_labels()
    for source, target in module.AI4P_TO_NEMOTRON.items():
        if target is not None:
            assert target in valid, f"{source} -> {target} is not a Nemotron label"


def test_social_number_is_split_by_format_not_collapsed():
    # ai4privacy uses one label for US SSNs and non-US national numbers.
    # Nemotron distinguishes them and so do our detectors, so the format
    # decides. Collapsing them would teach the student to call UK numbers SSNs.
    module = load()
    assert module.to_nemotron_label("SOCIALNUMBER", "473-54-7641") == "ssn"
    assert module.to_nemotron_label("SOCIALNUMBER", "231 45 6789") == "ssn"
    assert module.to_nemotron_label("SOCIALNUMBER", "669 398 5477") == "national_id"
    assert module.to_nemotron_label("SOCIALNUMBER", "231.576.7658") == "national_id"


def test_ip_version_is_decided_by_format():
    module = load()
    assert module.to_nemotron_label("IP", "192.168.1.1") == "ipv4"
    assert module.to_nemotron_label("IP", "2001:db8::8a2e:370:7334") == "ipv6"


def test_a_label_with_no_equivalent_becomes_background_explicitly():
    # TITLE ("Sir") is a salutation, not an identifier. None means background
    # by decision; an unknown label must raise instead.
    module = load()
    assert module.to_nemotron_label("TITLE", "Sir") is None


def test_an_unknown_label_raises_rather_than_becoming_background():
    """A dataset revision that adds a category must fail loudly.

    Silently mapping it to O would teach the student that a whole new class of
    identifier is background -- the worst possible failure for a PII scanner,
    and invisible.
    """
    module = load()
    with pytest.raises(KeyError, match="unknown ai4privacy label"):
        module.to_nemotron_label("SOME_NEW_LABEL_V2", "x")


def test_passport_maps_to_national_id_rather_than_being_dropped():
    # Nemotron has no passport label. Dropping to background would actively
    # teach the student that passport numbers are not identifiers; a passport
    # is a national identity document, so national_id is the honest target.
    module = load()
    assert module.to_nemotron_label("PASSPORT", "X1234567") == "national_id"


def test_the_mapping_covers_every_label_the_corpus_actually_uses():
    module = load()
    observed = {
        "EMAIL", "GIVENNAME1", "GIVENNAME2", "LASTNAME1", "LASTNAME2",
        "LASTNAME3", "STREET", "BUILDING", "SECADDRESS", "CITY", "POSTCODE",
        "STATE", "COUNTRY", "GEOCOORD", "USERNAME", "TEL", "BOD", "DATE",
        "TIME", "SEX", "PASS", "IDCARD", "PASSPORT", "DRIVERLICENSE",
        "CARDISSUER", "TITLE", "SOCIALNUMBER", "IP",
    }
    for label in observed:
        module.to_nemotron_label(label, "sample-value")   # must not raise
