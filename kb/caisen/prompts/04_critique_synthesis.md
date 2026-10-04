# Phase 4 — Critique the synthesis (round N ≥ 2; once per model, in parallel)

Replace `<MODEL>` with your slug and `<N>` with the next round number (first time: 2).

---

You are an adversarial reviewer of `docs/caisen_methodology.md`, the canonical knowledge base
distilled from 蔡森《多空转折一手抓》. Read `kb/caisen/schema.md` (Challenge section, target=synthesis)
and the previous round's `kb/caisen/runs/synthesis/r<N-1>/changelog.json`.

1. Check every section against the page images in `kb/caisen/work/pages/` (cited pages first, then
   spot-check uncited chapters for omissions). Pay most attention to target formulas, stop rules,
   and worked numbers — these drive trades.
2. Re-raise a rejected finding only with NEW page evidence.
3. Check §6 detector conditions are faithful to §3 and actually testable on daily OHLCV.

Write `kb/caisen/runs/critiques/r<N>/<MODEL>.json`. An empty `findings` list is a valid, useful
answer when the doc is right. Run `uv run python kb/caisen/check_outputs.py --phase challenge
--round <N>` and commit only your file.

Loop: Opus runs Phase 3 with this round's N, then Phase 4 with N+1, until a round has zero `major`
findings across all models or N reaches 4 (check_outputs.py reports convergence).
