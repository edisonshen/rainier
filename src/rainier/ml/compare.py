"""Compare signal emitters by running backtests side-by-side.

Answers the question: "Does the ML model produce better trading results
than the rule-based BookScorer on the same historical data?"

Two modes:
- ``compare_emitters`` — single full-sample backtest per emitter (fast,
  good for sanity checks).
- ``run_walkforward_compare`` — slides a (train, test) window through the
  data, runs backtest on each test window, and aggregates per-emitter
  metrics. Prevents lookahead leakage: emitter factories are given only
  the train slice, and the engine sees only the test slice.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from rainier.backtest.engine import run_backtest
from rainier.core.config import BacktestConfig
from rainier.core.protocols import BacktestMetrics, SignalEmitter, TradeRecord
from rainier.core.types import Timeframe

EmitterFactory = Callable[[pd.DataFrame], SignalEmitter]


@dataclass
class ComparisonResult:
    """Side-by-side backtest results for multiple emitters."""

    rows: list[tuple[str, BacktestMetrics]] = field(default_factory=list)

    @property
    def labels(self) -> list[str]:
        return [label for label, _ in self.rows]

    def get_metrics(self, label: str) -> BacktestMetrics | None:
        for lbl, metrics in self.rows:
            if lbl == label:
                return metrics
        return None


def compare_emitters(
    df: pd.DataFrame,
    symbol: str,
    timeframe: Timeframe,
    emitters: dict[str, SignalEmitter],
    config: BacktestConfig | None = None,
) -> ComparisonResult:
    """Run backtest with each emitter and collect results.

    Args:
        df: OHLCV DataFrame
        symbol: Instrument symbol
        timeframe: Bar timeframe
        emitters: Mapping of label → SignalEmitter (e.g. "BookScorer" → emitter)
        config: Shared backtest configuration

    Returns:
        ComparisonResult with one (label, BacktestMetrics) per emitter
    """
    if config is None:
        config = BacktestConfig()

    result = ComparisonResult()

    for label, emitter in emitters.items():
        metrics = run_backtest(df, symbol, timeframe, emitter, config)
        result.rows.append((label, metrics))

    return result


def format_comparison_table(result: ComparisonResult) -> str:
    """Format comparison results as a side-by-side ASCII table."""
    if not result.rows:
        return "No results to compare."

    # Metrics to display
    metric_defs = [
        ("Total trades", lambda m: f"{m.total_trades}"),
        ("Winners", lambda m: f"{m.winners}"),
        ("Losers", lambda m: f"{m.losers}"),
        ("Win rate", lambda m: f"{m.win_rate:.1%}"),
        ("Profit factor", lambda m: f"{m.profit_factor:.2f}"),
        ("Net P&L", lambda m: f"{m.total_net_pnl:+,.2f}"),
        ("Gross P&L", lambda m: f"{m.total_gross_pnl:+,.2f}"),
        ("Commission", lambda m: f"{m.total_commission:,.2f}"),
        ("Slippage", lambda m: f"{m.total_slippage:,.2f}"),
        ("Max drawdown", lambda m: f"{m.max_drawdown_pct:.1%}"),
        ("Sharpe ratio", lambda m: f"{m.sharpe_ratio:.2f}"),
        ("Avg win", lambda m: f"{m.avg_win:+,.2f}"),
        ("Avg loss", lambda m: f"{m.avg_loss:+,.2f}"),
        ("Avg hold bars", lambda m: f"{m.avg_hold_bars:.1f}"),
        ("Largest win", lambda m: f"{m.largest_win:+,.2f}"),
        ("Largest loss", lambda m: f"{m.largest_loss:+,.2f}"),
        ("Final equity", lambda m: f"{m.final_equity:,.2f}"),
    ]

    labels = result.labels
    col_width = max(14, *(len(label) + 2 for label in labels))
    label_col_width = max(len(name) for name, _ in metric_defs) + 2

    # Header
    lines: list[str] = []
    sep = "=" * (label_col_width + col_width * len(labels) + 4)
    lines.append(sep)
    lines.append("SCORER COMPARISON")
    lines.append(sep)

    # Column headers
    header = f"{'':>{label_col_width}}"
    for label in labels:
        header += f"  {label:>{col_width}}"
    lines.append(header)
    lines.append("-" * len(header))

    # Rows
    for metric_name, fmt_fn in metric_defs:
        row = f"{metric_name:>{label_col_width}}"
        for _, metrics in result.rows:
            row += f"  {fmt_fn(metrics):>{col_width}}"
        lines.append(row)

    lines.append(sep)

    # Delta summary (if exactly 2 emitters)
    if len(result.rows) == 2:
        _, m1 = result.rows[0]
        _, m2 = result.rows[1]
        lines.append("")
        lines.append(f"Delta ({labels[1]} vs {labels[0]}):")

        pf_delta = m2.profit_factor - m1.profit_factor
        wr_delta = m2.win_rate - m1.win_rate
        pnl_delta = m2.total_net_pnl - m1.total_net_pnl
        dd_delta = m2.max_drawdown_pct - m1.max_drawdown_pct
        sharpe_delta = m2.sharpe_ratio - m1.sharpe_ratio

        lines.append(f"  Profit factor: {pf_delta:+.2f}")
        lines.append(f"  Win rate:      {wr_delta:+.1%}")
        lines.append(f"  Net P&L:       {pnl_delta:+,.2f}")
        lines.append(f"  Max drawdown:  {dd_delta:+.1%}")
        lines.append(f"  Sharpe ratio:  {sharpe_delta:+.2f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Walk-forward comparison
# ---------------------------------------------------------------------------


@dataclass
class WalkForwardWindow:
    """Per-window record from a walk-forward backtest."""

    fold: int
    train_start: int
    train_end: int  # exclusive
    test_start: int
    test_end: int  # exclusive
    metrics_by_label: dict[str, BacktestMetrics] = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward comparison results."""

    windows: list[WalkForwardWindow] = field(default_factory=list)
    aggregated: list[tuple[str, BacktestMetrics]] = field(default_factory=list)

    @property
    def labels(self) -> list[str]:
        return [label for label, _ in self.aggregated]

    def get_aggregate(self, label: str) -> BacktestMetrics | None:
        for lbl, metrics in self.aggregated:
            if lbl == label:
                return metrics
        return None


def run_walkforward_compare(
    df: pd.DataFrame,
    symbol: str,
    timeframe: Timeframe,
    emitter_factories: dict[str, EmitterFactory],
    train_bars: int,
    test_bars: int,
    step_bars: int | None = None,
    config: BacktestConfig | None = None,
) -> WalkForwardResult:
    """Run a walk-forward comparison: slide a (train, test) window forward.

    For each window:
      - Slice df[train_start:train_end] as training data
      - Build an emitter via emitter_factories[label](train_df) for each label
      - Run backtest on df[test_start:test_end] with that emitter
      - Record per-window metrics

    Aggregates across windows by concatenating trade lists and recomputing
    portfolio metrics (so profit factor / win rate reflect the full
    out-of-sample series, not an average of fold-level summaries).

    Args:
        df: OHLCV DataFrame (must have 'timestamp', 'open', 'high', 'low',
            'close' columns; sorted ascending by timestamp).
        symbol: Instrument symbol.
        timeframe: Bar timeframe.
        emitter_factories: Mapping label → factory(train_df) → emitter.
            BookScorer factories may ignore train_df; ML factories
            typically refit / reload a model anchored on train_df.
        train_bars: Length of training window in bars.
        test_bars: Length of test (out-of-sample) window in bars.
        step_bars: How far to advance between folds. Defaults to ``test_bars``
            (non-overlapping test windows, i.e. classic walk-forward).
        config: Shared backtest config across windows.

    Returns:
        WalkForwardResult with per-window records + aggregated metrics.

    Raises:
        ValueError: if train_bars/test_bars are non-positive, if the
            dataset is shorter than one train+test span, or if
            emitter_factories is empty.
    """
    if not emitter_factories:
        raise ValueError("emitter_factories must be non-empty")
    if train_bars <= 0 or test_bars <= 0:
        raise ValueError("train_bars and test_bars must be positive")
    if step_bars is None:
        step_bars = test_bars
    if step_bars <= 0:
        raise ValueError("step_bars must be positive")

    n = len(df)
    if n < train_bars + test_bars:
        raise ValueError(
            f"DataFrame has {n} rows, need at least train_bars + test_bars "
            f"= {train_bars + test_bars}"
        )

    if config is None:
        config = BacktestConfig()

    result = WalkForwardResult()
    fold = 0
    train_start = 0

    while train_start + train_bars + test_bars <= n:
        train_end = train_start + train_bars
        test_start = train_end
        test_end = test_start + test_bars

        train_df = df.iloc[train_start:train_end].reset_index(drop=True)
        test_df = df.iloc[test_start:test_end].reset_index(drop=True)

        window = WalkForwardWindow(
            fold=fold,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )

        for label, factory in emitter_factories.items():
            emitter = factory(train_df)
            metrics = run_backtest(test_df, symbol, timeframe, emitter, config)
            window.metrics_by_label[label] = metrics

        result.windows.append(window)
        fold += 1
        train_start += step_bars

    # Aggregate: concatenate per-window trades + recompute portfolio metrics
    from rainier.backtest.engine import compute_metrics

    for label in emitter_factories:
        all_trades: list[TradeRecord] = []
        equity_curve: list[float] = [config.initial_capital]
        running = config.initial_capital
        for window in result.windows:
            metrics = window.metrics_by_label.get(label)
            if metrics is None:
                continue
            for trade in metrics.trades:
                all_trades.append(trade)
                running += trade.net_pnl
                equity_curve.append(running)
        agg = compute_metrics(all_trades, equity_curve, config)
        result.aggregated.append((label, agg))

    return result


def format_walkforward_table(result: WalkForwardResult) -> str:
    """Format walk-forward results: per-window summary + aggregate row."""
    if not result.windows or not result.aggregated:
        return "No walk-forward results to display."

    labels = result.labels
    lines: list[str] = []

    # Header
    width = max(14, *(len(label) + 2 for label in labels))
    fold_col = 8
    sep_len = fold_col + 4 + width * len(labels) + 4
    sep = "=" * sep_len

    lines.append(sep)
    lines.append("WALK-FORWARD COMPARISON (per-window net P&L)")
    lines.append(sep)

    header = f"{'Fold':>{fold_col}}"
    for label in labels:
        header += f"  {label:>{width}}"
    lines.append(header)
    lines.append("-" * len(header))

    for window in result.windows:
        row = f"{window.fold:>{fold_col}}"
        for label in labels:
            metrics = window.metrics_by_label.get(label)
            pnl = metrics.total_net_pnl if metrics is not None else 0.0
            row += f"  {pnl:>+{width},.2f}"
        lines.append(row)

    lines.append("-" * len(header))

    # Aggregate row
    agg_header = f"{'Aggregate':>{fold_col}}"
    for label in labels:
        agg = result.get_aggregate(label)
        pnl = agg.total_net_pnl if agg is not None else 0.0
        agg_header += f"  {pnl:>+{width},.2f}"
    lines.append(agg_header)

    lines.append(sep)
    lines.append("")
    lines.append("Aggregate metrics (concatenated out-of-sample trades):")
    lines.append("")

    # Per-label aggregate detail
    metric_defs = [
        ("Total trades", lambda m: f"{m.total_trades}"),
        ("Win rate", lambda m: f"{m.win_rate:.1%}"),
        ("Profit factor", lambda m: f"{m.profit_factor:.2f}"),
        ("Net P&L", lambda m: f"{m.total_net_pnl:+,.2f}"),
        ("Max drawdown", lambda m: f"{m.max_drawdown_pct:.1%}"),
        ("Sharpe ratio", lambda m: f"{m.sharpe_ratio:.2f}"),
    ]
    label_col = max(len(name) for name, _ in metric_defs) + 2
    detail_header = f"{'':>{label_col}}"
    for label in labels:
        detail_header += f"  {label:>{width}}"
    lines.append(detail_header)
    lines.append("-" * len(detail_header))

    for metric_name, fmt_fn in metric_defs:
        row = f"{metric_name:>{label_col}}"
        for label in labels:
            agg = result.get_aggregate(label)
            row += f"  {(fmt_fn(agg) if agg else 'n/a'):>{width}}"
        lines.append(row)

    lines.append(sep)
    return "\n".join(lines)
