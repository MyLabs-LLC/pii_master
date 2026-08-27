"""Render the report's tables from the recorded arms and decisions.

The conventions are explicit that a number reaches a report from a recorded
decision and never by being retyped, so every table the report carries is
generated here from `evaluations/arm_*.json` and `decision/*.json`. Nothing in
the Markdown is hand-entered, which is also why the report cannot disagree with
the gate: both read the same objects.

`NOT MEASURABLE` is printed as itself. A corpus whose tag gold is positive-only
cannot report precision, and rendering that as `0.0000` -- or as a blank a reader
fills in with a zero -- is the specific mistake this whole run is arranged to
avoid.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.h2h_assemble import MODEL_NAMES, PROJECT  # noqa: E402
from training.h2h_score import ARMS  # noqa: E402
from training.quiet_data import PRIORITY_TAGS  # noqa: E402

NM = "NOT MEASURABLE"
ORDER = ("A", "B", "C")


def _f(v: Any, places: int = 4) -> str:
    if v is None:
        return NM
    if isinstance(v, dict):
        v = v.get("value")
    return NM if v is None else f"{v:.{places}f}"


def _val(arm: dict[str, Any], metric: str) -> Any:
    m = arm["metrics"].get(metric)
    return m.get("value") if isinstance(m, dict) else m


def _ci(arm: dict[str, Any], metric: str) -> str:
    m = arm["metrics"].get(metric) or {}
    lo, hi = m.get("ci_low"), m.get("ci_high")
    return "" if lo is None or hi is None else f"[{lo:.4f}, {hi:.4f}]"


def _short(corpus: str) -> str:
    return corpus.split("_", 1)[1] if "_" in corpus else corpus


# ------------------------------------------------------------------ scoreboard
def scoreboard(arms: dict[str, dict]) -> str:
    rows = [
        ("Read window (chars)", lambda a, k: f"{ARMS[k]['window']:,}"),
        ("**macro F2** (headline ranker)", lambda a, k: _f(_val(a, "macro_f2"))),
        ("macro F1", lambda a, k: _f(_val(a, "f1_macro_catalogue"))),
        ("macro F0.5", lambda a, k: _f(_val(a, "macro_f05"))),
        ("macro F3", lambda a, k: _f(_val(a, "f3_macro_catalogue"))),
        ("macro precision", lambda a, k: _f(_val(a, "precision_macro_catalogue"))),
        ("macro recall", lambda a, k: _f(_val(a, "recall_macro_catalogue"))),
        ("micro F1", lambda a, k: _f(_val(a, "micro_f1"))),
        ("micro F2", lambda a, k: _f(_val(a, "f2_micro"))),
        ("micro F0.5", lambda a, k: _f(_val(a, "f05_micro"))),
        ("micro precision", lambda a, k: _f(_val(a, "precision_micro"))),
        ("micro recall", lambda a, k: _f(_val(a, "recall_micro"))),
        ("priority macro F0.5 (contra ranker)", lambda a, k: _f(_val(a, "priority_macro_f05"))),
        ("priority macro precision", lambda a, k: _f(_val(a, "priority_macro_precision"))),
        ("priority macro recall", lambda a, k: _f(_val(a, "priority_macro_recall"))),
        ("worst priority-tag recall", lambda a, k: _f(_val(a, "severity_recall_min"))),
        ("document precision", lambda a, k: _f(_val(a, "equal_corpus_doc_precision"))),
        ("document recall", lambda a, k: _f(_val(a, "equal_corpus_doc_recall"))),
        ("document specificity", lambda a, k: _f(_val(a, "equal_corpus_doc_specificity"))),
        ("prediction rate", lambda a, k: _f(_val(a, "prediction_rate"))),
        # Summed over the five corpora whose gold can measure F2 at all, so this
        # counts dead tag x corpus PAIRS, not distinct tags -- a tag dead on
        # three corpora contributes three.
        ("dead tag×corpus pairs (F2 = 0)",
         lambda a, k: str(int(sum(c.get("n_tags_f2_zero") or 0
                                  for c in a["per_corpus"].values())))),
        ("median tag F2", lambda a, k: _f(_val(a, "f2_median"))),
        ("one-core p95 (ms)", lambda a, k: _f(_val(a, "p95_latency_ms"), 3)),
        ("one-core docs/s", lambda a, k: _f(_val(a, "docs_per_s"), 0)),
    ]
    head = "| Measure | " + " | ".join(
        f"{MODEL_NAMES[k]} (arm {k})" for k in ORDER) + " |"
    sep = "| --- | " + " | ".join("---:" for _ in ORDER) + " |"
    body = [f"| {name} | " + " | ".join(fn(arms[k], k) for k in ORDER) + " |"
            for name, fn in rows]
    return "\n".join([head, sep, *body])


# ------------------------------------------------------------------ per corpus
def per_corpus(arms: dict[str, dict], metric: str, title: str) -> str:
    corpora = list(arms["A"]["per_corpus"])
    head = f"| Corpus | n | " + " | ".join(f"{MODEL_NAMES[k]}" for k in ORDER) + " |"
    sep = "| --- | ---: | " + " | ".join("---:" for _ in ORDER) + " |"
    out = [f"**{title}**", "", head, sep]
    for c in corpora:
        n = arms["A"]["per_corpus"][c]["n_rows"]
        cells = [_f(arms[k]["per_corpus"][c].get(metric)) for k in ORDER]
        out.append(f"| {_short(c)} | {n:,} | " + " | ".join(cells) + " |")
    equal = [_f(_val(arms[k], {"f2_macro_catalogue": "macro_f2",
                               "f1_micro": "micro_f1"}.get(metric, metric)))
             for k in ORDER]
    out.append("| **equal-corpus mean** | | " + " | ".join(f"**{v}**" for v in equal) + " |")
    return "\n".join(out)


# --------------------------------------------------------------------- per tag
def per_tag_worst(arms: dict[str, dict], corpus: str, limit: int = 20) -> str:
    """Worst-first, precision beside recall -- silent and unlearned look alike in F2."""
    out = [f"**{_short(corpus)} — worst {limit} tags by F2, per arm**", ""]
    for k in ORDER:
        tags = arms[k]["per_tag"][corpus]
        scored = [(t, v) for t, v in tags.items()
                  if v["support"] > 0 and v.get("f2") is not None]
        if not scored:
            out += [f"*{MODEL_NAMES[k]} (arm {k}): {NM} on this corpus "
                    f"(positive-only tag gold).*", ""]
            continue
        scored.sort(key=lambda kv: (kv[1]["f2"], -kv[1]["support"]))
        out += [f"*{MODEL_NAMES[k]} (arm {k})*", "",
                "| tag | n | F2 | precision | recall | predicted | state |",
                "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
        for tag, v in scored[:limit]:
            if v["predicted"] == 0:
                state = "**unlearned**"
            elif (v["precision"] or 0) >= 0.7 and (v["recall"] or 0) < 0.3:
                state = "silent"
            else:
                state = ""
            star = " ★" if tag in PRIORITY_TAGS else ""
            out.append(f"| `{tag}`{star} | {v['support']:,} | {_f(v['f2'])} | "
                       f"{_f(v['precision'])} | {_f(v['recall'])} | "
                       f"{v['predicted']:,} | {state} |")
        out.append("")
    out.append("★ = priority tag. **unlearned** = never predicted once; "
               "*silent* = recognised but almost never emitted.")
    return "\n".join(out)


# -------------------------------------------------------------------- decision
def _hard(arm: dict[str, Any]) -> list[dict[str, Any]]:
    return [c for c in arm.get("constraints", [])
            if (c.get("constraint") or {}).get("severity") == "hard"]


def gates(decision: dict[str, Any]) -> str:
    """Per-arm verdicts. `n_pass/n_measurable of n_scopes` is the honest triple:
    a scoped constraint can pass, fail, or have nothing to judge it on."""
    out = [f"Selected: **{decision.get('selected') or 'none — no feasible arm'}**", "",
           "| Arm | Verdict | Hard constraints failed | Scoped detail |",
           "| --- | --- | ---: | --- |"]
    for a in decision.get("arms", []):
        hard = _hard(a)
        verdict = "**FEASIBLE**" if a.get("feasible") else "blocked"
        if a.get("arm") == decision.get("selected"):
            verdict = "**WINNER**"
        detail = "; ".join(
            f"{c['reason']}" for c in hard
            if c.get("verdict") == "fail" and c.get("n_measurable"))
        out.append(f"| {a['arm']} | {verdict} | {a.get('n_hard_failed', 0)} of "
                   f"{len(hard)} | {detail or '—'} |")
    return "\n".join(out)


def gate_detail(decision: dict[str, Any], limit: int = 10) -> str:
    """Which scopes failed, worst margin first, with support so a thin scope is
    recognisable as one."""
    out = []
    for a in decision.get("arms", []):
        fails = [c for c in _hard(a) if c.get("verdict") == "fail"]
        if not fails:
            out.append(f"- **{a['arm']}** — cleared every hard constraint.")
            continue
        out.append(f"- **{a['arm']}** — {len(fails)} hard constraint(s) failed:")
        for c in fails:
            con = c.get("constraint") or {}
            out.append(f"    - *{con.get('name', '?')}* — {c.get('reason', '')}")
            scoped = [s for s in c.get("scopes", []) if s.get("verdict") == "fail"]
            threshold = con.get("threshold")

            def margin(s):
                v = s.get("compared")
                return (threshold - v) if (v is not None and threshold is not None) else 0.0

            for s in sorted(scoped, key=margin, reverse=True)[:limit]:
                out.append(f"        - `{s['scope']}` — {_f(s.get('compared'))} "
                           f"(point {_f(s.get('value'))}, n={s.get('support')})")
            if len(scoped) > limit:
                out.append(f"        - …and {len(scoped) - limit} more")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=PROJECT / "reports" / "_tables.md")
    args = ap.parse_args()
    arms = {k: json.loads((PROJECT / "evaluations" / f"arm_{k}.json").read_text(encoding="utf-8"))
            for k in ORDER}
    blocks = ["## Model scoreboard", "", scoreboard(arms), "",
              "## Per corpus", "",
              per_corpus(arms, "f2_macro_catalogue", "macro F2 (the headline ranker)"), "",
              per_corpus(arms, "f1_micro", "micro F1"), "",
              per_corpus(arms, "priority_macro_f05", "priority macro F0.5 (contra-view ranker)"), "",
              per_corpus(arms, "recall_macro_catalogue", "macro recall"), "",
              per_corpus(arms, "precision_macro_catalogue", "macro precision"), "",
              per_corpus(arms, "prediction_rate", "prediction rate"), ""]
    for corpus in arms["A"]["per_corpus"]:
        if arms["A"]["per_corpus"][corpus].get("can_measure_precision"):
            blocks += [per_tag_worst(arms, corpus), ""]
    for label in ("headline", "precision_view"):
        p = PROJECT / "decision" / f"{label}.json"
        if p.is_file():
            d = json.loads(p.read_text(encoding="utf-8"))
            blocks += [f"## Decision — {label}", "", gates(d), "", gate_detail(d), ""]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(blocks) + "\n", encoding="utf-8")
    print(f"tables -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
