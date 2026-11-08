# Phase 2 — Challenge the other models' notes (round 1; once per model, in parallel)

Replace `<MODEL>` with your slug. Requires all four `kb/caisen/runs/notes/<model>/` folders.

---

You are an adversarial reviewer of other models' study notes on 蔡森《多空转折一手抓》.
Read `kb/caisen/schema.md` (Challenge section).

For each unit in `kb/caisen/work/units.json`:
1. Read the notes of the OTHER three models: `kb/caisen/runs/notes/<other>/<unit_id>.json`.
2. Where they differ from each other or from your own notes, or a claim looks wrong, open the cited
   page images in the unit's `page_dir` and check the book itself. The book is the only authority —
   not majority vote and not your own earlier notes (you may also flag errors in your own notes).
3. Record a finding for every factual error, misread number, omitted rule/pattern/example,
   contradiction, or overreach (interpretation presented as the book's rule).

Write `kb/caisen/runs/critiques/r1/<MODEL>.json`. Be specific: `about` must point at the file and
field; `evidence_pages` + `quote_zh` must let anyone verify the finding in seconds.
Do not pad with stylistic nits; a missing stop rule is `major`, a wording preference is not a finding.

Then run `uv run python kb/caisen/check_outputs.py --phase challenge --round 1` and commit only
your critique file.
