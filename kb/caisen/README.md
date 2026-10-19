# 蔡森《多空转折一手抓》 — multi-model knowledge base

Four vision models read the scanned book independently, challenge each other against the page
images, and Claude Opus 5.5 synthesizes the result into `docs/caisen_methodology.md` — the
canonical methodology for QU100 pattern detection and the LLM-QU100 thesis prompt. Rounds repeat
until reviewers find no major errors.

| Slug | Model (pick in the Devin Desktop model picker) |
|---|---|
| `opus-5-5` | Claude Opus 5.5 (also the synthesizer) |
| `gpt-5-5` | GPT-5.5 |
| `gemini-3-1-pro` | Gemini 3.1 Pro Preview |
| `grok-4-7` | Grok 4.7 |

## Setup (once, on the machine running Devin Desktop)

```bash
uv run --with pymupdf python kb/caisen/render_pages.py \
    --pdf ~/Downloads/多空转折一手抓\(高清\).pdf --out kb/caisen/work
```

Writes `kb/caisen/work/units.json` (32 reading units of ≤ 12 pages, split along the book's TOC) and
`kb/caisen/work/pages/<unit>/pNNN.png`. `work/` is gitignored — the scan is copyrighted; only our
notes are committed.

## Run

In Devin Desktop open the rainier repo, choose the model, and paste the prompt (replace `<MODEL>`
/ `<N>`). Same-phase runs for different models are independent and can run in parallel sessions.

| Step | Prompt | Who | Output |
|---|---|---|---|
| 1 Read | `prompts/01_read.md` | all 4 | `runs/notes/<model>/<unit>.json` |
| 2 Challenge notes | `prompts/02_challenge.md` | all 4 | `runs/critiques/r1/<model>.json` |
| 3 Synthesize (N=1) | `prompts/03_synthesize.md` | Opus 5.5 | `docs/caisen_methodology.md`, `runs/synthesis/r1/` |
| 4 Critique synthesis (N=2) | `prompts/04_critique_synthesis.md` | all 4 | `runs/critiques/r2/<model>.json` |
| 5 … | repeat 3 (N) → 4 (N+1) | | until converged or N = 4 |

Tips: pilot step 1 on `u01`–`u03` with one model first and eyeball the JSON before running the
whole book. A model that runs out of context can be restarted — finished units are skipped.

## Check progress

```bash
uv run python kb/caisen/check_outputs.py                 # all phases
uv run python kb/caisen/check_outputs.py --phase challenge --round 2
```

Reports missing/malformed files per model, unread pages, finding counts, and whether the latest
round converged (zero `major` findings).

## After convergence

`docs/caisen_methodology.md` §6 (detector spec, with code deviations) and §5 (rule ids) feed the
follow-up work: aligning `src/rainier/analysis/stock_patterns.py` / `target_calculator.py` with the
book and citing rule ids in the thesis `SYSTEM_PROMPT` (`src/rainier/llm_thesis/prompt.py`).
