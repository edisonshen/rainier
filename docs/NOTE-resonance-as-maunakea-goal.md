# Note — Resonance Gate-Search as the Flagship Maunakea Trading Goal

**Status:** forward-pointer (not v1 work) · **Owner of the eventual build:** the maunakea *trading wrapper* (downstream, Phase 4) · **Blocked on:** maunakea `Engine.run()` (Phase 2) + `recurse()` (Phase 3), both `NotImplementedError` today

---

## Why this note exists

The whole TQQQ resonance investigation (10+ exploratory studies in `docs/REPORT-*.html`, scripts in `scripts/`) was me **hand-cranking** a research loop: propose a signal/gate combo → backtest → A/B vs the SMA22/44 gate → check OOS → guard against overfit → propose the next combo. **That loop is exactly what maunakea automates.** This note captures it as a ready-made goal so the trading wrapper has a real first target the day maunakea's iterate/recurse machinery exists.

## Architecture reminder (so this lands in the right place)

Per `~/projects/maunakea/CLAUDE.md`, maunakea core is **domain-agnostic** — CI enforces zero `rainier` references in `src/maunakea/`. Trading lives in a **downstream wrapper** (Phase 4 in maunakea's roadmap), never in core.

```
maunakea core (engine: suggest/run/recurse)  ──used by──►  trading wrapper (THIS goal)
                                                                  │ data from / strategy to
                                                                  ▼
                                                              rainier (data backfill, execution, live signals)
```

- **rainier** ships the data (per `DESIGN-rainier-data-backfill-for-maunakea`) and runs the vetted gate as a live `WeightStrategy` (the v1 we're building now in `src/rainier/signals/`).
- **The trading wrapper** points maunakea at that data with the goal below and lets the engine search recursively.

## The goal spec (for the wrapper)

```
dataset: QQQ, TQQQ, SPY, VIX, IRX daily (rainier backfill) — real adjusted prices,
         2010→now real TQQQ + synthetic 3×QQQ pre-2010 for OOS.

goal:    "Find a daily entry/exit gate for TQQQ that beats the SMA22/44 gate AND
          buy-and-hold on Calmar AND max-drawdown, out-of-sample, net of cost."

hard guardrails (the wrapper must hand maunakea these as constraints, because the
hand-search proved they're where naive results die):
  - no moving-average / lookback > 66 bars (composed/nested span counts)
  - no-lookahead: signal on close[t], effective t→t+1
  - real TQQQ already embeds 3× financing — charge only turnover + cash-leg rate;
    synthetic 3× (3·r − 2·rate − fee) ONLY pre-2010 where no real ETF exists
  - validate on a re-derived ≤2022 train / 2023→now test AND a genuinely independent
    slice (pre-2010 synthetic or SOXL — NOT SPXL/UPRO near-clones)
  - de-levering is not edge: must beat a matched-exposure control
  - deflated Sharpe over #configs tried; pre-register the final config

known baseline to beat (from the hand-search):
  asymmetric SMA22/44 gate → TQQQ/cash : in-window Calmar ≈ 1.37, full-cycle Calmar ≈ 0.34,
  full-cycle (1999–2026) max drawdown ≈ −76%. Hard to beat. If maunakea can't, the
  honest answer is "ship the SMA gate" — and that's a valid recursive-loop result.
```

## What "recursive" buys us here that the hand-search couldn't

- **Breadth of search** — the hand-search tried ~270 SMA pairs + ~80 signal add-ons. Maunakea can sweep far wider without me getting bored or sloppy.
- **Self-improvement** — its `meta_suggestion` learns *how to search this space better* (e.g. "stop proposing slow-MA exits, they overfit the one 2022 bear"), which is the lesson it took me ten messages to learn by hand.
- **Honesty by construction** — the guardrails above become the engine's scoring constraints, so it can't reward a leaky/overfit result.

## Status / next

- **Now:** rainier builds the resonance gate v1 (the hand-found candidate) as a live strategy. See `DESIGN-multi-signal-resonance.md`.
- **When maunakea Phase 2/3 land:** stand up the trading wrapper, hand it this goal + guardrails, and let it try to beat the v1 gate. The v1 gate becomes maunakea's first concrete "can you do better than the human?" benchmark.
