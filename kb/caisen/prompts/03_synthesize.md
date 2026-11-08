# Phase 3 — Synthesize (Claude Opus 5.5 only)

Set `<N>` to the round whose critiques you are resolving (first synthesis: N=1).

---

You are the editor of rainier's canonical knowledge base for 蔡森《多空转折一手抓》.
Inputs: all notes in `kb/caisen/runs/notes/*/`, all critiques in `kb/caisen/runs/critiques/r<N>/`,
the page images in `kb/caisen/work/pages/`, and (if N>1) the current `docs/caisen_methodology.md`.
Use `docs/price_action_methodology.md` as the format reference (machine-readable, rule-numbered).

1. Resolve EVERY finding in `critiques/r<N>/`: verify it against the cited pages, then accept,
   reject (with reason), or mark disputed. Log each in
   `kb/caisen/runs/synthesis/r<N>/changelog.json` (schema.md, Synthesis section).
2. Write/update `docs/caisen_methodology.md` with this structure:
   - §0 Source, scope, how this doc was produced (models, rounds), and how to cite it.
   - §1 Core philosophy & risk principles (大赚小赔, stops, position sizing, time).
   - §2 Shared primitives: neckline, breakout/retest, 破底翻, 假突破, 满足点 (target) arithmetic,
     时间波, volume — each with precise, codeable definitions.
   - §3 One subsection per pattern (the 12 chapters): structure, entry, stop, target formula with the
     book's own worked numbers, confirmation, invalidation, failure handling, pages.
   - §4 Market-level vs individual-stock application (进阶篇), with the case studies as a table.
   - §5 Rules index: every rule as `C-<section>-<n>` with one-line text + pages, so code and the
     thesis prompt can reference rule ids.
   - §6 Detector spec: for each pattern, testable conditions mapped to rainier
     (`src/rainier/analysis/stock_patterns.py`, `target_calculator.py`), flagging where the current
     code deviates from the book (read the code for this section only).
   - §7 Disputed / unclear points: each with the competing readings, which models hold them, pages.
3. Never silently drop a disagreement: if the pages don't settle it, it goes in §7.

Commit `docs/caisen_methodology.md` and `kb/caisen/runs/synthesis/r<N>/`.
