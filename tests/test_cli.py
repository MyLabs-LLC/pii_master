import io
import json
from pathlib import Path

from pii_master.cli import main


def run_cli(args, capsys):
    code = main(args)
    return code, json.loads(capsys.readouterr().out)


def test_scan_file_with_ssn(tmp_path, capsys):
    p = tmp_path / "doc.txt"
    p.write_text("Payroll SSN 123-45-6789 on file.", encoding="utf-8")
    code, payload = run_cli(["scan", str(p)], capsys)
    assert code == 0
    (entry,) = payload["files"]
    assert entry["path"] == str(p)
    assert entry["label"] == "PII"
    assert entry["counts"] == {"SSN": 1}


def test_fail_on_detect_exits_1(tmp_path, capsys):
    p = tmp_path / "doc.txt"
    p.write_text("SSN 123-45-6789", encoding="utf-8")
    code, _ = run_cli(["scan", str(p), "--fail-on-detect"], capsys)
    assert code == 1


def test_fail_on_detect_clean_file_exits_0(tmp_path, capsys):
    p = tmp_path / "doc.txt"
    p.write_text("nothing sensitive here", encoding="utf-8")
    code, payload = run_cli(["scan", str(p), "--fail-on-detect"], capsys)
    assert code == 0
    assert payload["files"][0]["label"] == "NONE"


def test_stdin_dash(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("Patient MRN: 4829471"))
    code, payload = run_cli(["scan", "-"], capsys)
    assert code == 0
    assert payload["files"][0]["label"] == "PHI"


def test_multiple_files_and_pretty(tmp_path, capsys):
    clean = tmp_path / "clean.txt"
    clean.write_text("hello", encoding="utf-8")
    dirty = tmp_path / "dirty.txt"
    dirty.write_text("mail jane@example.com", encoding="utf-8")
    code, payload = run_cli(["scan", str(clean), str(dirty), "--pretty"], capsys)
    assert code == 0
    labels = [f["label"] for f in payload["files"]]
    assert labels == ["NONE", "PII"]


def test_eval_subcommand(tmp_path, capsys):
    corpus = tmp_path / "mini.jsonl"
    corpus.write_text(
        '{"id": "d1", "label": "PII", "text": "SSN [[SSN:123-45-6789]] here."}\n',
        encoding="utf-8",
    )
    code, payload = run_cli(["eval", str(corpus), "--json"], capsys)
    assert code == 0
    assert payload["documents"] == 1
    assert payload["span_exact"]["SSN"]["tp"] == 1
    assert payload["doc_accuracy"] == 1.0


def test_eval_subcommand_text_output(tmp_path, capsys):
    corpus = tmp_path / "mini.jsonl"
    corpus.write_text(
        '{"id": "d1", "label": "NONE", "text": "clean memo"}\n', encoding="utf-8"
    )
    code = main(["eval", str(corpus)])
    out = capsys.readouterr().out
    assert code == 0
    assert "PHI recall" in out


def test_bench_subcommand(capsys):
    code, payload = run_cli(
        ["bench", "--sizes", "500", "--docs-per-size", "2", "--json"], capsys
    )
    assert code == 0
    assert len(payload["buckets"]) == 1
    assert payload["buckets"][0]["docs"] == 2


def test_eval_fail_under_detects_regression(tmp_path, capsys):
    corpus = tmp_path / "mini.jsonl"
    corpus.write_text(
        '{"id": "d1", "label": "PII", "text": "SSN [[SSN:123-45-6789]] here."}\n',
        encoding="utf-8",
    )
    scores = tmp_path / "scores.json"
    # An unreachable baseline stands in for "the pipeline got worse".
    scores.write_text(json.dumps({
        "doc_accuracy": 1.0, "phi_recall": 1.0,
        "span_exact": {"SSN": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
                       "PERSON_NAME": {"precision": 1.0, "recall": 1.0, "f1": 1.0}},
    }), encoding="utf-8")
    assert main(["eval", str(corpus), "--fail-under", str(scores)]) == 1
    assert "QUALITY REGRESSION" in capsys.readouterr().err


def test_eval_save_scores_then_gate_passes(tmp_path, capsys):
    corpus = tmp_path / "mini.jsonl"
    corpus.write_text(
        '{"id": "d1", "label": "PII", "text": "SSN [[SSN:123-45-6789]] here."}\n',
        encoding="utf-8",
    )
    scores = tmp_path / "scores.json"
    assert main(["eval", str(corpus), "--save-scores", str(scores)]) == 0
    capsys.readouterr()
    assert main(["eval", str(corpus), "--fail-under", str(scores)]) == 0


def test_eval_report_shows_error_taxonomy(capsys):
    corpus = Path(__file__).resolve().parent.parent / "eval" / "corpus" / "frozen_v1.jsonl"
    assert main(["eval", str(corpus)]) == 0
    assert "Error taxonomy" in capsys.readouterr().out
