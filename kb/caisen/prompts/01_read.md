# Phase 1 — Read (run once per model; models can run in parallel)

Replace `<MODEL>` with your slug: `opus-5-5`, `gpt-5-5`, `gemini-3-1-pro`, or `grok-4-7`.

---

You are studying 蔡森《多空转折一手抓》, a scanned Chinese technical-analysis book, so a
trading system (rainier QU100 + the LLM thesis) can apply its methodology exactly.

Work in the rainier repo. Read `kb/caisen/schema.md` (Reader notes section) first.
The unit list is `kb/caisen/work/units.json`; each unit's page images are in its `page_dir`.

For EVERY unit in `units.json`, in order:
1. Skip it if `kb/caisen/runs/notes/<MODEL>/<unit_id>.json` already exists and is valid JSON.
2. Open and read EVERY page image of the unit yourself (vision). Do not rely on prior knowledge
   of the book, other models' notes, or the existing rainier code — this pass must be independent.
3. Write `kb/caisen/runs/notes/<MODEL>/<unit_id>.json` per the schema.

Rules:
- Faithful over fluent: transcribe numbers on charts (neckline, highs/lows, targets) exactly; if a
  number is unreadable, put it in `illegible` instead of guessing.
- Every rule/pattern/example cites pages. Quote the key Chinese sentence in `quote_zh`.
- Separate what the book SAYS (rules, quotes) from your interpretation (`quantifiable`, `ambiguities`).
- Ignore the advertisement watermark line at the top of each page.

When all units are done run `uv run python kb/caisen/check_outputs.py --phase read` and fix
anything it reports for `<MODEL>`. Then commit only `kb/caisen/runs/notes/<MODEL>/`.
