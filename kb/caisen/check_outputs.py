"""Progress + convergence report for the 蔡森 multi-model debate (see kb/caisen/README.md).

    uv run python kb/caisen/check_outputs.py                       # everything
    uv run python kb/caisen/check_outputs.py --phase read
    uv run python kb/caisen/check_outputs.py --phase challenge --round 2

Exit code 1 when anything requested is missing or malformed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

KB = Path(__file__).resolve().parent
MODELS = ("opus-5-5", "gpt-5-5", "gemini-3-1-pro", "grok-4-7")
# Optional extra readers (e.g. a cloud Devin run whose model is not pinned):
# reported when their notes exist, never required for convergence.
EXTRA_MODELS = ("devin-cloud",)
NOTE_KEYS = ("unit_id", "model", "pages_read", "summary", "patterns", "rules")
FINDING_KEYS = ("id", "about", "severity", "type", "claim", "correction", "evidence_pages")
MAX_ROUND = 4


def _load(path: Path) -> tuple[dict | None, str | None]:
    if not path.exists():
        return None, "missing"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except json.JSONDecodeError as e:
        return None, f"invalid JSON: {e}"


def check_read(units: list[dict]) -> int:
    problems = 0
    extras = [m for m in EXTRA_MODELS if (KB / "runs" / "notes" / m).is_dir()]
    for model in (*MODELS, *extras):
        done, issues = 0, []
        for unit in units:
            data, err = _load(KB / "runs" / "notes" / model / f"{unit['id']}.json")
            if err is None:
                missing = [k for k in NOTE_KEYS if k not in data]
                expected = set(range(unit["first_page"], unit["last_page"] + 1))
                unread = sorted(expected - set(data.get("pages_read", [])))
                if missing:
                    err = f"missing keys {missing}"
                elif unread:
                    err = f"pages not read {unread}"
            if err:
                issues.append(f"{unit['id']}: {err}")
            else:
                done += 1
        print(f"[read] {model:15s} {done}/{len(units)} units")
        for issue in issues:
            print(f"    - {issue}")
        problems += len(issues)
    return problems


def check_round(n: int) -> int:
    problems, majors = 0, 0
    for model in MODELS:
        data, err = _load(KB / "runs" / "critiques" / f"r{n}" / f"{model}.json")
        if err is None:
            findings = data.get("findings")
            if not isinstance(findings, list):
                err = "no findings list"
            else:
                bad = [f.get("id", "?") for f in findings if any(k not in f for k in FINDING_KEYS)]
                if bad:
                    err = f"findings missing required keys: {bad[:5]}"
        if err:
            print(f"[r{n}]   {model:15s} {err}")
            problems += 1
            continue
        m = sum(1 for f in findings if f.get("severity") == "major")
        majors += m
        print(f"[r{n}]   {model:15s} {len(findings)} findings ({m} major)")
    if problems == 0:
        if n >= 2 and majors == 0:
            print(f"[r{n}]   CONVERGED: zero major findings — synthesis is final.")
        elif n >= MAX_ROUND:
            print(f"[r{n}]   round cap reached: run a final synthesis; leftovers go to §7.")
        else:
            print(f"[r{n}]   {majors} major findings -> run 03_synthesize.md with N={n}.")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("read", "challenge", "all"), default="all")
    ap.add_argument("--round", type=int)
    args = ap.parse_args()

    units_path = KB / "work" / "units.json"
    if not units_path.exists():
        raise SystemExit("kb/caisen/work/units.json missing: run kb/caisen/render_pages.py first")
    units = json.loads(units_path.read_text(encoding="utf-8"))

    problems = 0
    if args.phase in ("read", "all"):
        problems += check_read(units)
    if args.phase in ("challenge", "all"):
        rounds = (
            [args.round]
            if args.round
            else sorted(
                int(p.name[1:]) for p in (KB / "runs" / "critiques").glob("r*") if p.is_dir()
            )
        )
        for n in rounds:
            problems += check_round(n)
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
