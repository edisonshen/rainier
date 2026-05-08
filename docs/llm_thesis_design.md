# LLM thesis layer for QU100 daily picks (v1)

## Context

Rainier already runs a 3-layer rule-based screener (`analysis/stock_screener.py:screen_stocks`) on every QU100 scrape — money flow + sector + 蔡森 chart pattern → composite score → ranked top-N candidates → Discord. The user wants an LLM layer on top of this pipeline that takes the top candidates, **looks at multi-day history (rank trajectory, capital flow streaks, sector momentum), checks fundamentals (P/E, earnings dates), and produces a structured per-ticker confidence + a short narrative thesis** for the two daily scans where the decision matters most: the pre-close scan at **12:45 PT** (15 min before US market close, "what to buy today") and the post-close scan at **15:00 PT** (end-of-day summary).

The eng-cleared draft at `docs/llm_thesis_plan.md` covers ~70% of this. This plan extends it with three deltas the user explicitly requested:

1. **Multi-day trends in the evidence pack** — rank trajectory, capital flow streak, sector net_sentiment delta over 5–10 days. Pure DB queries on existing `MoneyFlowSnapshot` and `StockCapitalFlow` rows.
2. **yfinance fundamentals** — P/E, market cap, next earnings date, recent earnings surprise. Free, ~1s/ticker.
3. **Run on `afternoon` + `close` sessions only** (not morning/midday). Other sessions keep existing behavior — screener output → Discord, no LLM.

Shadow mode confirmed: Discord ordering stays rule-based; LLM confidence is stored in `ScreenedStockRecord` for 30-day validation but does NOT change displayed ranks. Re-rank decision is gated on the validation data at day 30.

X/Twitter and any external news API are explicitly out of scope for v1.

## Implementation sequencing (stacked PRs, locked in eng review D1)

The design ships in four sequential PRs, each landing on top of the prior:

| PR | Scope | Tables touched | Ships when |
|---|---|---|---|
| **PR1** | Signal framework + 4 signals + scheduler hook + Discord renderer extension + ScreenedStockRecord + idempotency cache + per-scan settings reload | `LLMAnalysisRecord` (alter), `ScreenedStockRecord` (new) | LLM thesis appears in afternoon/close Discord |
| **PR2** | Daily eval job + outcome backfill + ThesisEvaluation table + eval Discord report | `ThesisEvaluation` (new) | Daily Discord eval message renders |
| **PR3** | Auto-research weekly job + ResearchInsight + ACTION_EXECUTORS + accept/reject CLI + ruamel.yaml | `ResearchInsight` (new) | First insights produced; user can accept/reject from CLI |
| **PR4** | Streamlit dashboard (4 tabs) | none | Dashboard reachable at localhost:8501 |

Each PR is self-contained, individually reviewable, and leaves the system in a working state.

## What's reused (no new code)

| Existing | Path | Reuse |
|---|---|---|
| Screener | `src/rainier/analysis/stock_screener.py:39` | Take `screen_stocks(settings)[:5]` as LLM input set |
| Screener OHLCV | `stock_screener.py:67` (`_fetch_stock_data`) | Refactor return signature to bubble OHLCV up (10-line change) |
| `StockCandidate` | `src/rainier/core/types.py:164-189` | Frozen, unmodified — passed into evidence pack |
| `MoneyFlowSnapshot` | `src/rainier/core/models.py:122-176` | Multi-day rank query (10-day lookback by symbol) |
| `StockCapitalFlow` | `src/rainier/core/models.py` | Capital flow streak query (already has `flow_date` + `period_type`) |
| Sector analyzer | `src/rainier/analysis/sector_analyzer.py:22-132` | Add date-range method; existing point-in-time stays unchanged |
| `LLMAnalysisRecord` | `src/rainier/core/models.py` | Per-ticker storage + cost tracking. Add `input_hash` column for idempotency |
| `viz/charts.py` | `src/rainier/viz/charts.py` | Add `create_static_stock_chart()` (new function in this file) |
| Discord renderer | `src/rainier/alerts/discord.py:send_stock_candidates` | Accept optional `theses` dict; render rich messages when present |
| Scheduler hook | `src/rainier/scheduler/service.py:run_qu_scrape` | Add LLM block conditional on `session_name in ("afternoon", "close")` |
| LiteLLM | `pyproject.toml` line 31 | LLM transport, multimodal, prompt caching |
| Kaleido | `pyproject.toml` line 24 | Plotly → PNG export |
| yfinance | `pyproject.toml` | Already used by screener; reuse for fundamentals |

## What's net new

### A. ORM + config

1. **`core/models.py:LLMAnalysisRecord`** — add `input_hash: Mapped[str | None] = mapped_column(String(64), index=True)` + unique partial index `idx_llm_analysis_idempotent` on `(date_trunc('day', created_at), target_symbols, llm_model, prompt_template, input_hash)`. The existing `llm_model` column (line 329) becomes the multi-model discriminator — including it in the unique index means v2 A/B (Sonnet + GPT-5 in parallel on the same ticker) drops in with **zero further schema migration**. Manual `ALTER TABLE` for existing DBs (script in plan).

2. **`core/models.py:ScreenedStockRecord`** — new table. Captures ALL screened candidates per scan (20 rows × every scan, ~80/day) plus LLM augmentation on the 5 top picks of `afternoon`/`close` scans, plus manual outcome tracking. Schema:
   - `scan_date`, `session_name`, `symbol`, `rule_rank`, `composite_score`, `money_flow_score`, `sector`, `pattern_type`, `pattern_confidence` (always populated)
   - `llm_confidence`, `shadow_combined_score`, `would_be_combined_rank`, `thesis_id` FK → `LLMAnalysisRecord`, `patterns_in_chart_not_in_indicators_count` (only on LLM-augmented scans)
   - `action_taken`, `outcome_pct`, `outcome_recorded_at`, `notes` (manual via `rainier thesis log`)
   - `forward_return_5d`, `forward_return_10d`, `outcome_backfilled_at` (auto via outcome backfill job)
   - `UNIQUE(scan_date, session_name, symbol)`. Plain Postgres, NOT a hypertable (~28K rows/year is small).

3. **`core/config.py:LLMThesisConfig`** + matching `config/settings.yaml` keys. **Signal-set is config-driven** — new signals plug in via YAML toggle, no scheduler/prompt code changes:
   ```python
   class SignalConfig(BaseModel):
       enabled: bool = True
       params: dict[str, Any] = {}
       weight: float = 1.0  # used by performance dashboard, not by LLM directly

   class LLMThesisConfig(BaseModel):
       enabled: bool = True
       model: str = "claude-sonnet-4-6"
       max_usd_per_scan: float = 1.0
       prompt_version: str = "v1"
       enabled_sessions: list[str] = ["afternoon", "close"]
       fallback_to_anthropic_sdk: bool = False
       signals: dict[str, SignalConfig] = {
           "rank_trajectory":     SignalConfig(params={"days": 10}),
           "capital_flow_streak": SignalConfig(params={"days": 10}),
           "sector_momentum":     SignalConfig(params={"days": 10}),
           "fundamentals":        SignalConfig(params={}),
           # add new signals here, default-disabled until validated:
           # "market_breadth":    SignalConfig(enabled=False, params={"universe": "SP500"}),
       }
   ```
   Toggling `enabled: false` for any signal in `config/settings.yaml` removes it from the next scan's evidence pack with **zero code change** and **no scheduler restart**.

   **Settings reload per scan (eng review D2).** The current `core/config.py:get_settings()` uses a Pydantic Settings singleton cached for the process lifetime. To make YAML toggles actually take effect mid-day, replace the singleton with a per-scan reload: at the top of `scheduler/service.py:run_qu_scrape`, call `settings = load_settings_fresh()` to read `config/settings.yaml` into a new Settings instance, then pass it down through the call chain. Cost: ~5 lines in config.py + scheduler.py. Effect: YAML edit takes effect on the NEXT scan invocation (max 30-min wait), no daemon restart required.

4. **Add `signals_used: Mapped[list[str]] = mapped_column(ARRAY(String(50)))` to `LLMAnalysisRecord`** so per-signal performance can be queried in SQL. Keeps the dashboard performance page fast. Migration: `ALTER TABLE analysis_results ADD COLUMN signals_used VARCHAR(50)[];`

5. **Add Tier-1 lookup index (eng review)** — `CREATE INDEX ix_llm_analysis_date_target_template ON analysis_results (date(created_at), target_symbols, prompt_template);` so the cache hit query (`WHERE date(created_at)=? AND target_symbols=[symbol] AND prompt_template=?`) uses an index seek instead of a partial-index probe.

### B. New module `src/rainier/llm_thesis/`

#### B.0 Signal framework (extensibility)

Each piece of evidence is a **Signal** — a small, self-contained class with a uniform interface. The evidence pack is built by iterating the registry and calling `compute()` on every enabled signal. **Adding a new signal is one new file + one YAML toggle.**

**Protocol** — `src/rainier/llm_thesis/signals/base.py`:
```python
@runtime_checkable
class ThesisSignal(Protocol):
    name: str            # registry key, must match settings.yaml entry
    version: str         # bump to invalidate cached values when logic changes
    cost_estimate_ms: int  # rough latency budget hint, used by dashboard

    def compute(self, ctx: SignalContext) -> SignalValue | None:
        """Pure computation — returns JSON-serializable value or None to skip this ticker."""

    def render_for_prompt(self, value: SignalValue) -> str:
        """Convert value to a short text snippet the LLM sees in the evidence pack."""
```

`SignalContext` carries `symbol`, `scan_date`, `session_name`, `ohlcv_df`, `candidate: StockCandidate`, `params: dict` (from YAML), and a DB session. `SignalValue` is `dict[str, Any]` (JSONB-storable).

**Registry** — `src/rainier/llm_thesis/signals/__init__.py`:
```python
from .rank_trajectory import RankTrajectorySignal
from .capital_flow_streak import CapitalFlowStreakSignal
from .sector_momentum import SectorMomentumSignal
from .fundamentals import FundamentalsSignal

REGISTRY: dict[str, type[ThesisSignal]] = {
    "rank_trajectory":     RankTrajectorySignal,
    "capital_flow_streak": CapitalFlowStreakSignal,
    "sector_momentum":     SectorMomentumSignal,
    "fundamentals":        FundamentalsSignal,
}
```

**Evidence assembly** — `service.py:assemble_evidence` iterates `settings.llm_thesis.signals.items()`, instantiates enabled signals from the registry, and runs them **in parallel** via `asyncio.gather` + `asyncio.to_thread` (eng review D4). Each signal's `compute()` is sync but blocking I/O (yfinance, DB), so threadpool execution keeps the event loop free. `return_exceptions=True` isolates failures: one bad signal does not kill the thesis. Per-ticker latency drops from ~15s sequential to ~3s parallel (bottleneck = slowest signal, not sum). `render_for_prompt()` produces a labeled section the LLM template stitches together.

```python
# service.py:assemble_evidence (parallel signals)
signal_tasks = [
    asyncio.to_thread(signal.compute, ctx)
    for signal in enabled_signals
]
results = await asyncio.gather(*signal_tasks, return_exceptions=True)
for signal, value in zip(enabled_signals, results):
    if isinstance(value, Exception):
        log.warning("signal_failed", signal=signal.name, error=str(value))
        continue
    if value is not None:
        pack.signals[signal.name] = value
```

**Adding a new signal — full workflow:**
1. Drop a new file at `src/rainier/llm_thesis/signals/market_breadth.py` implementing `ThesisSignal`.
2. Add it to `REGISTRY` in `signals/__init__.py` (one line).
3. Add YAML entry under `llm_thesis.signals.market_breadth` with `enabled: false` initially.
4. Test it with `uv run rainier thesis signals test market_breadth --symbol NVDA` (new CLI command).
5. Flip `enabled: true` in YAML when ready. Next scan picks it up automatically.
6. After 7–14 days of shadow data, check the dashboard's per-signal performance page; if it doesn't lift forward returns, flip `enabled: false`. **Zero code change to disable.**

**Initial signal set (the four already in this design):**

| Signal `name` | Source | What it returns | Cost |
|---|---|---|---|
| `rank_trajectory` | `MoneyFlowSnapshot` query | `{"points": [(date, rank)], "delta_10d": int, "trend": "rising"\|"falling"\|"flat"}` | ~5ms DB |
| `capital_flow_streak` | `StockCapitalFlow` query | `{"streak_days": int, "direction": "+"\|"-"\|"N", "history": [...]}` | ~5ms DB |
| `sector_momentum` | New `sector_analyzer.analyze_sectors_at(date)` over last N days | `{"sentiment_today": float, "sentiment_5d_ago": float, "delta": float, "shifted_at": date or None}` | ~20ms DB |
| `fundamentals` | yfinance `Ticker(symbol)` | `{"pe_trailing": float, "pe_forward": float, "market_cap": int, "next_earnings_date": date, "last_surprise_pct": float}` | ~1s network |

These four ship in v1. The framework exists so a fifth (e.g., `market_breadth`) can be added by the user later with no scheduler/prompt rewrites.

#### B.1 Schema and assembly

5. **`schemas.py`** — `TradeThesis` Pydantic model: `verdict` (Literal `setup_long`/`watch`/`no_setup` — NO `setup_short` since screener is long-only), `setup_quality` (1–10), `llm_confidence` (1–10), `paragraph_radar` (≤400ch), `paragraph_evidence` (≤600ch), `paragraph_invalidation` (≤400ch), `risks` (list[str], max 5), `watch_items` (list[str], max 5), `evidence_used` (list[str]), `signals_used` (list[str] — names from the registry that produced non-None values), `patterns_in_chart_not_in_indicators` (list[str] max 5 OR Literal `"none"`). Plus `EvidencePack` (carries `signals: dict[name, value]`) and `compute_input_hash(pack, image_bytes)`.

7. **`chart_export.py`** — kaleido wrapper:
   ```python
   def render_chart_png(symbol, df, pattern=None) -> tuple[bytes, str]:
       fig = create_static_stock_chart(symbol, df, pattern)
       png = fig.to_image(format="png", width=1280, height=800, engine="kaleido")
       return png, hashlib.sha256(png).hexdigest()
   ```

8. **`prompt.py`** — static prompt prefix as module constant. System prompt + JSON output schema + 1 few-shot example + style guide. Includes:
   - "You are reading a daily QU100 candidate. Use the chart image as a discovery channel BEYOND engineered features. The `patterns_in_chart_not_in_indicators` field is required and falsifiable."
   - Multi-day trend section in evidence template (rank trajectory, flow streak, sector momentum).
   - Fundamentals section in evidence template (P/E, earnings date proximity, surprise).
   - `PROMPT_VERSION = "v1"`.

9. **`service.py`** — two functions:
   - `assemble_evidence(candidate, ohlcv_df, trend_data, fundamentals) -> EvidencePack`. Pulls together the technical core, multi-day trends from `trends.py`, fundamentals from `fundamentals.py`. Defers `render_chart_png` until cache miss.
   - `generate_thesis(symbol, scan_date, session_name, evidence_provider, prompt_version, max_usd_remaining) -> tuple[TradeThesis | None, float]`. Two-tier cache:
     - **Tier 1 (cheap):** `SELECT id, structured_output FROM analysis_results WHERE date(created_at)=today AND target_symbols=[symbol] AND prompt_template=? LIMIT 1`. If hit → return cached, no LLM, no chart, no yfinance.
     - **Tier 2 (miss):** call `evidence_provider()` (renders chart, fetches fundamentals, computes trends), call LiteLLM with Anthropic `cache_control` markers on static prefix, 3 retries on Pydantic validation failure with reprompt context. Hard kill-switch on `max_usd_remaining`. Persist with `INSERT ... ON CONFLICT DO NOTHING`.

10. **`persistence.py`** — two functions:
    - `persist_screened_stocks(candidates, scan_date, session_name)` — bulk INSERT 20 rows on EVERY scan (every session, not just LLM-augmented ones). Single round-trip via `session.execute(insert(...).on_conflict_do_nothing(...))`. Idempotent.
    - `update_with_thesis(symbol, scan_date, session_name, llm_confidence, shadow_combined_score, would_be_combined_rank, thesis_id, patterns_count)` — UPDATE matching row after thesis is generated. Logs warning if zero rows affected.

11. **`__init__.py`** — public surface.

### C. Surgical modifications to existing files

12. **`analysis/stock_screener.py:screen_stocks()`** — change return from `list[StockCandidate]` to `tuple[list[StockCandidate], dict[str, pd.DataFrame]]`. Update one call site in `scheduler/service.py`.

13. **`analysis/sector_analyzer.py`** — add `analyze_sectors_at(captured_at: datetime)` taking a date parameter (existing `analyze_sectors()` continues to use `max(captured_at)`). Used by `trends.get_sector_momentum`.

14. **`viz/charts.py`** — add `create_static_stock_chart(symbol, df, pattern=None) -> go.Figure` rendering 60 daily candles + horizontal lines for entry/SL/target if pattern fired + pattern label annotation. No tabs, no interactivity. Stock-specific (the existing `create_chart()` is futures-only).

15. **`scheduler/service.py:run_qu_scrape`** — extend the post-scrape hook. All blocking work (yfinance, kaleido, LiteLLM, DB) runs via `asyncio.to_thread` so the event loop stays responsive. **First line: reload settings fresh from YAML** (eng review D2) so toggles take effect on the next scan without daemon restart:
    ```python
    settings = await asyncio.to_thread(load_settings_fresh)  # NEW — read settings.yaml each scan
    candidates, ohlcv_by_symbol = await asyncio.to_thread(screen_stocks, settings)
    candidates = candidates[:20]

    # ALWAYS persist screener output for every scan (afternoon, close, morning, midday).
    try:
        await asyncio.to_thread(persist_screened_stocks, candidates,
                                scan_date=today, session_name=session_name)
    except Exception as exc:
        log.error("persist_screened_stocks_failed", error=str(exc))
        await asyncio.to_thread(_increment_db_failure_counter)

    # LLM thesis ONLY on configured sessions (afternoon + close per user choice).
    theses: dict[str, TradeThesis] = {}
    if session_name in settings.llm_thesis.enabled_sessions:
        try:
            theses = await asyncio.to_thread(
                compute_theses_and_persist,
                candidates[:5], ohlcv_by_symbol,
                settings.llm_thesis.max_usd_per_scan,
                scan_date=today, session_name=session_name,
            )
        except Exception as exc:
            log.error("compute_theses_unexpected_failure", error=str(exc))
            theses = {}

    # Discord always fires. Empty theses dict = existing behavior, no thesis paragraphs.
    await asyncio.to_thread(send_stock_candidates, candidates,
                            settings.alerts.discord, theses=theses)
    ```

16. **`alerts/discord.py:send_stock_candidates`** — add optional `theses: dict[str, dict] | None = None`. When provided, after the existing top-20 summary, iterate `candidates[:5]` and emit a rich per-ticker message (verdict, quality/confidence, 3 paragraphs, risks, watch_items, surprise patterns). Each message ≤1800 chars (Discord 2000-char limit).

17. **`scheduler/jobs.py` outcome backfill** — new scheduled entry: `run_outcome_backfill()` runs daily ~1h after market close (15:00 PT scan finishes ~15:05). For every `ScreenedStockRecord` row from N-5 days ago lacking `forward_return_5d`, look up close from `StockPrice` and compute `(close_today - close_at_scan) / close_at_scan`. Same for 10-day at N-10. **Unbiased outcome data for ALL screened names**, not just user-traded ones — needed for the day-30 correlation analysis.

### D. CLI

18. **`cli.py`** — subcommands under `rainier thesis`:
    - `rainier thesis daily --session afternoon --top-n 5 --discord [--dry-run] [--max-usd 1.0]` — manual scheduler-hook trigger.
    - `rainier thesis ticker SYMBOL` — single-ticker debug pipeline.
    - `rainier thesis log --ticker NVDA --date 2026-05-07 --action took --outcome +2.3% [--notes "..."]` — UPDATE outcome on `ScreenedStockRecord`.
    - `rainier thesis signals list` — print all signals from the registry, their `enabled` status from settings.yaml, last-run hit count, last-run latency.
    - `rainier thesis signals enable NAME` / `disable NAME` — flip `enabled` in settings.yaml without hand-editing.
    - `rainier thesis signals test NAME --symbol NVDA` — dry-run a single signal's `compute()` against the latest data. For validating new signals before flipping them on.
    - `rainier thesis eval [--date YYYY-MM-DD] [--horizon 1d|5d|10d]` — run the evaluation job manually (also runs nightly via scheduler).
    - `rainier thesis research run` — manually trigger the weekly research job (also runs on cron Sunday 18:00 PT).
    - `rainier thesis research insights list [--status pending|accepted|rejected|all]` — browse `ResearchInsight` rows.
    - `rainier thesis research insights accept ID` — apply the suggested config change and mark accepted.
    - `rainier thesis research insights reject ID --reason "..."` — dismiss with a note (kept for audit).
    - `rainier thesis research signals [--signal NAME] [--days 30]` — ad-hoc per-signal contribution SQL.
    - `rainier thesis research verdicts [--days 30]` — ad-hoc per-verdict hit rate.

### E. Daily evaluation (day-1, NOT deferred)

Evaluation is **first-class from day 1**, not a 30-day-later check. Every morning after the prior trading day closes, the system grades yesterday's theses and posts a Discord eval report. This drives the auto-research loop: when a signal or verdict underperforms over a rolling window, the dashboard flags it and the user toggles it off in YAML.

19. **New `core/models.py:ThesisEvaluation` table** — one row per (thesis_id, horizon). Schema:
    - `thesis_id` FK → `LLMAnalysisRecord`, `screened_record_id` FK → `ScreenedStockRecord`
    - `evaluated_at` (timestamp), `horizon` (Literal `"1d"`/`"5d"`/`"10d"`), `scan_date`, `symbol`, `verdict`, `llm_confidence`
    - `entry_price` (close on scan_date), `exit_price` (close at horizon), `return_pct`, `hit` (bool: did thesis direction match return sign?)
    - `signals_used` (denormalized from `LLMAnalysisRecord` for fast joins)
    - `notes` (auto: e.g., "earnings released day after scan — exclude from base rate")
    - `UNIQUE(thesis_id, horizon)`

20. **`src/rainier/llm_thesis/eval.py`** — three functions:
    - `evaluate_horizon(scan_date, horizon)` — for every thesis from `scan_date - horizon` days ago lacking a `ThesisEvaluation` row at this horizon, compute return from `StockPrice` and INSERT. Idempotent.
    - `compute_signal_contribution(days=30) -> list[SignalContribution]` — for each signal in the registry, partitions theses into "signal was used" vs "signal was absent", computes mean forward_return_5d in each partition, returns delta + sample size + p-value (Mann-Whitney U). This is the falsifiable per-signal value-add metric.
    - `compute_verdict_hit_rate(days=30) -> dict[verdict, HitRate]` — for each verdict, computes hit rate (% positive returns) and avg return at 1d/5d/10d horizons.

21. **Scheduler entry `run_daily_eval()`** — fires nightly at 17:00 PT (Mon–Fri, ~2h after market close). Calls `evaluate_horizon` for `1d`, `5d`, `10d` horizons (idempotent — only fills missing rows). Then posts an eval Discord summary:
    ```
    Eval report — 2026-05-07
    Yesterday's afternoon scan (2026-05-06):
      ✓ NVDA  setup_long (8/10) → +2.3%  HIT
      ✓ TSLA  setup_long (7/10) → +0.8%  HIT
      ✗ AAPL  watch     (5/10) → -1.2%  miss (was watch, not buy)
      ✓ MSFT  setup_long (6/10) → +1.1%  HIT
      ✗ GOOG  no_setup  (3/10) → +2.5%  miss (no_setup, but rallied)

    30-day base rates (rolling):
      setup_long verdict:  win-rate 68%   avg +1.4%
      watch verdict:       win-rate 51%   avg +0.2%
      no_setup verdict:    win-rate 47%   avg -0.1%

    Signal contribution (rolling 30d, p<0.05 only):
      ✓ rank_trajectory   +0.6% lift   n=78   p=0.012
      ✓ sector_momentum   +0.4% lift   n=78   p=0.038
      ⚠ fundamentals      +0.1% lift   n=78   p=0.41   (no significant lift — consider disabling)
    ```

### F. Auto-research loop (day-1)

Auto-research is the **automated half of "look at eval data → improve the system"**. It runs on top of the daily eval data and produces concrete, actionable recommendations the user can accept or reject from a CLI or dashboard. v1 is **recommend-only** (human-in-the-loop accept/reject); v2 promotes accepted insight types to auto-apply on threshold breach.

22. **New `core/models.py:ResearchInsight` table** — one row per finding produced by the research job. Schema (eng review D3 + D6):
    - `id`, `created_at`, `updated_at`, `kind` (Literal: `signal_underperform`/`signal_overperform`/`verdict_drift`/`calibration_off`/`prompt_regression`/`new_pattern_discovered`)
    - `severity` (Literal `info`/`warn`/`critical`), `subject` (e.g., signal name, verdict name, prompt_version)
    - `evidence: JSONB` — the statistical facts (sample size, lift, p-value, chart data)
    - **`action: JSONB`** — STRUCTURED action object so the accept handler can dispatch directly. Shape: `{"kind": "disable_signal" | "bump_prompt_version" | "raise_signal_weight" | "lower_signal_weight" | "noop", "target": "<signal_name | verdict | prompt_version>", "params": {...}}`. NOT free-text; mapped 1:1 to executors (D3).
    - `rationale: TEXT` — human-readable narrative for Discord/UI rendering, separate from the executable action.
    - `recurrence_count: int = 1` — incremented when the same (kind, target) fires again while still pending (D6 dedupe).
    - `status` (Literal `pending`/`accepted`/`rejected`/`auto_applied`/`stale`), `decided_at`, `decided_by`
    - `applied_change: JSONB | None` — once accepted/auto-applied, what config diff was written
    - `UNIQUE(kind, target, status)` partial index on `status='pending'` so UPSERT is a single statement.

23. **`src/rainier/llm_thesis/research.py`** — research job runs weekly **Friday 09:00 PT** (eng review D5; analyzes `scan_date in [today-37, today-7]` so every thesis has at least 7 days of forward-return backfill). Also manually via CLI. Produces `ResearchInsight` rows from `ThesisEvaluation` data via UPSERT (D6). Initial check set:
    - **`signal_underperform`** — for each signal, Mann-Whitney U on used vs absent forward returns over 30d. If p<0.05 AND lift<0.1%, emit `warn` + suggested_action `"disable signal X"`.
    - **`signal_overperform`** — symmetric, when a signal shows p<0.05 lift >0.5%, emit `info` + `"consider raising weight on signal X in scoring"`.
    - **`verdict_drift`** — chi-square on verdict × hit-rate over rolling 14d vs 30d. If discrimination collapses, emit `critical` + `"prompt revision needed; consider bumping prompt_version"`.
    - **`calibration_off`** — bin theses by `llm_confidence` (1–10), compute realized hit-rate per bin, regress against expected. If slope deviates from 1.0 by >0.3, emit `warn` + `"LLM is over/under-confident — recalibrate prompt instruction"`.
    - **`new_pattern_discovered`** — `patterns_in_chart_not_in_indicators` field hit-rate analysis: which discovered patterns correlate with positive forward returns? Emit `info` + `"consider adding deterministic detector for pattern X"`.
    - **`prompt_regression`** (active when prompt_version changes mid-window) — A/B compare hit-rates across prompt_version values. Flag if new version is statistically worse.

24. **Insight workflow** — fully day-1:
    - Friday research job emits findings via UPSERT: same `(kind, target)` while still pending UPDATEs the existing row (refreshes evidence + bumps `recurrence_count`); else INSERTs new `pending`.
    - Discord eval report (Friday afternoon) appends a "New research insights this week" block listing `pending` + `warn`/`critical` findings with their human `rationale` and structured `action.kind`.
    - User runs `rainier thesis research insights list` to browse, `accept ID` to mark accepted, or `reject ID --reason "..."` to dismiss.
    - **Accept dispatch** (D3): `accept` reads `insight.action.kind`, looks up `ACTION_EXECUTORS[kind]`, calls the executor with `insight.action.target` + `params`. Executors mutate `config/settings.yaml` via **ruamel.yaml** (preserves comments + key order) with atomic temp-file-rename. Initial executor map:
      ```python
      ACTION_EXECUTORS = {
          "disable_signal":     _disable_signal,        # flips signals.<target>.enabled = false
          "bump_prompt_version": _bump_prompt_version,  # increments llm_thesis.prompt_version
          "raise_signal_weight": _raise_signal_weight,  # multiplies signals.<target>.weight by 1.2
          "lower_signal_weight": _lower_signal_weight,  # multiplies signals.<target>.weight by 0.8
          "noop":               _noop,                  # explicit no-op for info-only insights
      }
      ```
    - Reject just sets `status='rejected'`, stores `decided_by`, `decided_at`, free-text reason. Settings.yaml unchanged.
    - Dashboard tab 4 (below) gives a click-to-accept UI alternative.
    - Insight rows go `stale` automatically after 30 days if still pending — prevents cruft.

25. **What's NOT in v1** (deferred to v2):
    - **Auto-apply** without user approval. v2 will let the user mark certain insight kinds (e.g., `signal_underperform` at `critical` severity) as `auto_apply: true` per kind, and the research job will then write the config change directly. v1 is recommend-only because 30d × 5 picks/day = 150 datapoints is statistically thin and the human ack guards against overfitting to noise.
    - **Auto-prompt-tuning** (the system writes new prompt variants, runs A/B, picks winner). Substantial new system; defer until v1 produces enough data to justify.
    - **Auto-signal-discovery** (the system mines correlations in raw QU100 data and proposes brand-new signal classes). Same — defer.

### G. Performance + config dashboard

23. **`src/rainier/dashboard/`** — Streamlit app launched via `uv run streamlit run src/rainier/dashboard/app.py`. Single source-of-truth for human inspection. Three tabs:

    **Tab 1: Signals (config + live status)**
    - Table: signal name, enabled, version, last 7d hit count, last-run latency, last-run error (if any).
    - Toggle: each signal has an enable/disable checkbox that writes back to `config/settings.yaml` via the `rainier thesis signals enable/disable` CLI under the hood.
    - "Test signal" button per row: runs `compute()` for a sample symbol, shows the value + render_for_prompt output.

    **Tab 2: Performance (per-signal contribution)**
    - Time-series: rolling 30d signal contribution lift (line chart per signal).
    - Per-signal scatterplot: signal numeric value (e.g., `rank_trajectory.delta_10d`) vs `forward_return_5d`. Quick visual on whether the signal is informative.
    - Per-verdict hit rate over time.
    - Falsification flags: red row for any signal with p>0.20 and lift <0.1% over 30+ days.

    **Tab 3: Recent theses (browse)**
    - Last 50 theses with verdict, llm_confidence, signals_used, evaluated outcomes at 1d/5d/10d.
    - Click through to the chart PNG that was sent to the LLM (stored in `chart_image_ids` on `LLMAnalysisRecord`).
    - Filterable by date range, verdict, symbol.

    **Tab 4: Research insights (accept/reject UI)**
    - List of `ResearchInsight` rows with severity, kind, subject, evidence summary, suggested action.
    - Buttons: Accept (applies the YAML change), Reject (with reason), View evidence (expands the JSONB).
    - Shows historical accepted/rejected insights so the user sees the audit trail of what they have and haven't acted on.

    Streamlit is the lowest-friction choice (already a placeholder in the module map per CLAUDE.md). All data comes from existing tables — no new ETL.

## Cost estimate

Sonnet 4.6 multimodal at $3/$15 per MTok input/output, $0.30/MTok cached input. Per-ticker prompt: ~10K static prefix (cacheable), ~3.5K volatile evidence (was 3K in v1; +500 for trends + fundamentals), ~1.5K image, ~2K output. 5 tickers × 2 scans = 10 calls/day with prompt cache warm within each scan.

| Component | Per-scan tokens | Cost |
|---|---|---|
| Static prefix, ticker 1 (uncached) | 10K | $0.030 |
| Static prefix, tickers 2–5 (cached) | 40K | $0.012 |
| Volatile evidence × 5 | 17.5K | $0.053 |
| Image tokens × 5 | 7.5K | $0.023 |
| Output × 5 | 10K | $0.150 |
| **Per scan** | | **~$0.27** |
| **Per day (afternoon + close)** | | **~$0.54** |

Monthly (weekdays only): ~$11 on Sonnet. Re-runs hit DB cache and cost zero.

## Failure handling (changes/additions vs. v1 plan)

| Codepath | Failure | Handling |
|---|---|---|
| `trends.get_rank_trajectory` | DB query fails / no rows | Pass empty trajectory in evidence; thesis still proceeds |
| `trends.get_sector_momentum` | Sector data missing for date | Pass `None` momentum; thesis still proceeds |
| `fundamentals.get_fundamentals` | yfinance error / missing fields | Return all-None Fundamentals; thesis still proceeds |
| All v1-plan failure modes | (chart export, LLM call, DB persist, etc.) | Per existing `docs/llm_thesis_plan.md` |

Per-ticker try/except around full thesis assembly. One ticker failing does not kill the scan; Discord posts a "Skipped TICKER: <reason>" notice.

## Verification

After implementation:

1. **DB schema** — `uv run rainier db init` creates `screened_stocks`. `psql -c "\d analysis_results"` shows `input_hash` column. Manual ALTER if upgrading existing DB.

2. **Unit tests** — full coverage from `docs/llm_thesis_plan.md` test plan plus the following additions from the eng review:
   - `tests/test_llm_thesis/signals/test_protocol.py` — verify all REGISTRY signals conform to `ThesisSignal` Protocol via `isinstance(s, ThesisSignal)`.
   - `tests/test_llm_thesis/signals/test_rank_trajectory.py`, `test_capital_flow_streak.py`, `test_sector_momentum.py`, `test_fundamentals.py` — happy + empty + error paths per signal.
   - `tests/test_llm_thesis/test_assemble_evidence_parallel.py` — verify `asyncio.gather` correctness with a slow mocked signal (one signal sleeps 2s, others return immediately, total wall time ≤ 2.5s, all values populated).
   - `tests/test_llm_thesis/test_eval.py` — `evaluate_horizon` idempotent fill, `compute_signal_contribution` Mann-Whitney U on synthetic data, verdict hit-rate.
   - `tests/test_llm_thesis/test_research.py` — UPSERT recurrence_count on repeat fire, all 6 insight kinds emit correctly, ACTION_EXECUTORS dispatch (one test per kind).
   - `tests/test_llm_thesis/test_yaml_mutation.py` — ruamel.yaml round-trip preserves comments/order; only the targeted line changes.
   - `tests/integration/test_thesis_e2e.py` (gated `pytest -m llm_integration`, requires `RAINIER_LLM_INTEGRATION_TEST=1`) — actual Sonnet 4.6 call against a fixture symbol; asserts TradeThesis schema validity. Run on demand, not in CI.
   - **REGRESSION: `tests/test_stock_screener.py`** — `screen_stocks()` returns `tuple[list, dict]`; existing callers receive both elements correctly.
   - **REGRESSION: `tests/test_alerts/test_discord.py`** — `send_stock_candidates(theses=None)` preserves existing top-20-only behavior (no thesis messages).
   - **REGRESSION: `tests/test_scheduler/test_service.py`** — verify `afternoon`/`close` sessions fire LLM block, `morning`/`midday` skip it; verify settings reload per scan (mid-process YAML edit takes effect on next scan invocation).
   - **PR4 only**: `tests/test_dashboard/test_data.py` — unit-test the data-loading helpers used by Streamlit. The Streamlit pages themselves get manual smoke tests, not unit tests.

3. **Manual end-to-end smoke test**:
   ```bash
   # one-off afternoon scan (uses latest scraped data, real LiteLLM call)
   uv run rainier thesis daily --session afternoon --top-n 5 --max-usd 1.0
   # check Discord webhook posted theses + DB has rows
   psql -c "SELECT scan_date, session_name, symbol, rule_rank, llm_confidence, shadow_combined_score FROM screened_stocks WHERE scan_date = CURRENT_DATE ORDER BY session_name, rule_rank;"
   ```

4. **Idempotency check** — re-run the same `rainier thesis daily` immediately. Tier-1 cache hits, zero new LiteLLM calls (verify via `analysis_results.total_cost_usd` unchanged).

5. **30-day validation queries** — at day 30, run the SQL from `docs/llm_thesis_plan.md` (Step 12 sample queries) plus a new one to evaluate fundamentals contribution:
   ```sql
   -- Did proximity to earnings move the LLM verdict in a useful direction?
   -- (requires logging fundamental fields used in the thesis call;
   --  cheaper alternative: spot-check `evidence_used` field in TradeThesis JSONB)
   SELECT verdict, COUNT(*), AVG(forward_return_5d)
   FROM screened_stocks s
   JOIN analysis_results a ON a.id = s.thesis_id
   WHERE s.scan_date >= CURRENT_DATE - INTERVAL '30 days'
   GROUP BY verdict;
   ```

6. **Day-30 go/no-go decision** — based on:
   - Spearman rank correlation between `forward_return_5d` and (a) `rule_rank` vs (b) `would_be_combined_rank`, computed across ALL screened names (unbiased)
   - Hit rate of `patterns_in_chart_not_in_indicators_count > 0`
   - Whether the user opens Discord theses daily

## Out of scope (deferred)

| Item | Why | When |
|---|---|---|
| Live combined-score re-rank in Discord | Shadow data first, decision at day 30 | v2 if 30-day data supports |
| Multi-model A/B (GPT-5, DeepSeek alongside Sonnet) | Schema is already future-proofed (unique index keys on `llm_model`); flip on by setting `llm_thesis.models = [...]` and gathering N parallel calls per ticker | v2, no further migration required |
| News context (X/Twitter, news API) | User explicitly excluded; X is rate-limited; news API is net new infra | v2 if technical+fundamentals theses prove thin |
| Historical replay (`--replay YYYY-MM-DD`) | `screen_stocks`, `analyze_sectors` hard-code `max(captured_at)` in 3 places — separate project | Later |
| Case-based analog engine (Codex's "coolest version") | Gates on whether `patterns_in_chart_not_in_indicators` is substantive | v3 |
| Auto-execution from LLM thesis | Out of scope by design (human-in-the-loop) | Never |
| LLM tool-use / agentic mode | Settled in /office-hours; fixed pipeline beats agentic for this domain | Revisit only if pipeline insufficient |
| Promote `ScreenedStockRecord` to hypertable | ~28K rows/year is small for plain Postgres | Only if rowcount grows 10× |

## Critical files to be modified or created

**Created:**
- `src/rainier/llm_thesis/__init__.py`
- `src/rainier/llm_thesis/schemas.py`
- `src/rainier/llm_thesis/signals/base.py` — `ThesisSignal` protocol + `SignalContext` + `SignalValue`
- `src/rainier/llm_thesis/signals/__init__.py` — `REGISTRY`
- `src/rainier/llm_thesis/signals/rank_trajectory.py`
- `src/rainier/llm_thesis/signals/capital_flow_streak.py`
- `src/rainier/llm_thesis/signals/sector_momentum.py`
- `src/rainier/llm_thesis/signals/fundamentals.py`
- `src/rainier/llm_thesis/chart_export.py`
- `src/rainier/llm_thesis/prompt.py`
- `src/rainier/llm_thesis/service.py`
- `src/rainier/llm_thesis/persistence.py`
- `src/rainier/llm_thesis/eval.py` — daily evaluation job + signal/verdict analyses
- `src/rainier/llm_thesis/research.py` — weekly auto-research job + insight generation
- `src/rainier/dashboard/app.py` — Streamlit entrypoint
- `src/rainier/dashboard/pages/` — signals, performance, recent_theses, insights
- `tests/test_llm_thesis/` (multiple test files including signal protocol tests + eval/research)

**Modified:**
- `src/rainier/core/models.py` — `LLMAnalysisRecord` add `input_hash` + `signals_used`; new `ScreenedStockRecord`, `ThesisEvaluation`, `ResearchInsight` tables
- `src/rainier/core/config.py` — add `LLMThesisConfig` + `SignalConfig`
- `src/rainier/analysis/stock_screener.py` — return signature `tuple[list[StockCandidate], dict[str, pd.DataFrame]]`
- `src/rainier/analysis/sector_analyzer.py` — add `analyze_sectors_at(captured_at)`
- `src/rainier/viz/charts.py` — add `create_static_stock_chart()`
- `src/rainier/scheduler/service.py` — extend `run_qu_scrape` with persist + conditional LLM block
- `src/rainier/scheduler/jobs.py` — add daily eval + weekly research + outcome backfill scheduled jobs
- `src/rainier/alerts/discord.py` — `send_stock_candidates` accepts optional `theses` dict; eval-report and insight-report renderers
- `src/rainier/cli.py` — add `rainier thesis` subcommands (daily/ticker/log/signals/eval/research)
- `config/settings.yaml` — add `llm_thesis` block including `signals` map

**Estimated effort:** 2.5–3 weekends solo, **split across 4 stacked PRs** (eng review D1) — see "Implementation sequencing" section near the top. PR1 ships first, gives you a working LLM thesis in Discord within ~1 weekend; subsequent PRs add eval, research, dashboard.

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 0 | — | — |
| Codex Review | `/codex review` | Independent 2nd opinion | 1 | SKIPPED — rate-limited | Codex usage limit hit (resets ~8 PM PT). Outside voice deferred; can re-run after limit clears. |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | CLEAR (PLAN) | 11 issues, 0 critical gaps, mode FULL_REVIEW. 5 architectural decisions locked (D1 stacked PRs, D2 settings reload, D3 structured action, D4 asyncio.gather signals, D5 Friday research). Test plan additions: ~10 new test files including 3 regressions. Test plan artifact written to `~/.gstack/projects/edisonshen-rainier/...eng-review-test-plan-20260507-192552.md`. |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | — | — |
| DX Review | `/plan-devex-review` | Developer experience gaps | 0 | — | — |

**CODEX:** SKIPPED — rate-limited at 2026-05-07 19:17 PT, retry after 20:00 PT. /review skill is the load-bearing reviewer per global CLAUDE.md.

**UNRESOLVED:** 0 — all 5 architectural decisions answered (D1–D5, plus D6 dedupe, D7 doc-update). Test additions applied inline to design doc.

**VERDICT:** ENG CLEARED — ready to implement PR1. Codex outside-voice can be re-run after rate limit clears; not blocking. The plan went through a full scope-reduction (mega-PR → 4 stacked PRs) and architectural review that caught settings-singleton drift, free-text action parsing risk, and sequential signal latency — all addressed in the design before any code lands.
