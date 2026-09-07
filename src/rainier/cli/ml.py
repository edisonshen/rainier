"""Feature export and ML commands."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import click
import numpy as np
import pandas as pd

from rainier.cli import (
    cli,
)
from rainier.core.types import Timeframe


@cli.group()
def features():
    """ML feature store commands."""


@features.command(name="export")
@click.option("--start", "start_date", default=None, help="Start date (YYYY-MM-DD)")
@click.option("--end", "end_date", default=None, help="End date (YYYY-MM-DD)")
@click.option("--symbols", default=None, help="Comma-separated symbols (default: all with prices)")
@click.option("--output-dir", default="data/features", help="Output directory for Parquet files")
@click.option("--min-bars", default=100, type=int, help="Minimum bars per symbol")
@click.option("--dry-run", is_flag=True, help="Show what would be exported")
def features_export(start_date, end_date, symbols, output_dir, min_bars, dry_run):
    """Export ML features + labels to Parquet for model training."""
    from rainier.ml.feature_store import (
        export_training_data,
        get_symbols_with_prices,
    )

    end = (
        datetime.strptime(end_date, "%Y-%m-%d").date()
        if end_date
        else datetime.now().date()
    )
    start = (
        datetime.strptime(start_date, "%Y-%m-%d").date()
        if start_date
        else date(end.year - 5, end.month, end.day)
    )

    if symbols:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
    else:
        symbol_list = get_symbols_with_prices(start, end, min_bars=min_bars)

    click.echo(f"Symbols: {len(symbol_list)}")
    click.echo(f"Date range: {start} to {end}")
    click.echo(f"Output: {output_dir}/")

    if dry_run:
        click.echo(f"\nWould export: {symbol_list[:30]}{'...' if len(symbol_list) > 30 else ''}")
        return

    if not symbol_list:
        click.echo("No symbols with sufficient price data. Run `rainier db backfill-prices` first.")
        return

    output_path = export_training_data(
        symbols=symbol_list,
        start=start,
        end=end,
        output_dir=Path(output_dir),
    )
    click.echo(f"\nExported to: {output_path}")

    # Validate
    from rainier.ml.feature_store import validate_parquet

    stats = validate_parquet(output_path)
    click.echo("\nValidation:")
    click.echo(f"  Rows: {stats['rows']:,}")
    click.echo(f"  Symbols: {stats['symbols']}")
    click.echo(f"  Features: {stats['features']}")
    click.echo(f"  Date range: {stats['date_range']}")
    click.echo(f"  Feature NaN count: {stats['feature_nan_total']}")
    for col, rate in stats["label_positive_rate"].items():
        click.echo(f"  {col} positive rate: {rate}")


@features.command(name="validate")
@click.argument("path", type=click.Path(exists=True))
def features_validate(path):
    """Validate an exported Parquet file."""
    from rainier.ml.feature_store import validate_parquet

    stats = validate_parquet(Path(path))
    for key, val in stats.items():
        if isinstance(val, dict):
            click.echo(f"{key}:")
            for k, v in val.items():
                click.echo(f"  {k}: {v}")
        else:
            click.echo(f"{key}: {val}")


# ---------------------------------------------------------------------------
# ML commands
# ---------------------------------------------------------------------------


@cli.group()
def ml():
    """Machine learning model commands."""


@ml.command(name="train")
@click.argument("features_path", type=click.Path(exists=True))
@click.option("--output-dir", default="models", help="Directory to save model")
@click.option("--label", default="label_5d", help="Label column to train on")
@click.option("--folds", default=3, type=int, help="Walk-forward CV folds")
def ml_train(features_path, output_dir, label, folds):
    """Train XGBoost pattern scorer on feature store data."""
    from rainier.ml.pattern_scorer import TrainConfig, train_model

    config = TrainConfig(label_col=label, n_folds=folds)
    click.echo(f"Training on: {features_path}")
    click.echo(f"Label: {label}, Folds: {folds}")

    model, result = train_model(
        parquet_path=Path(features_path),
        config=config,
        output_dir=Path(output_dir),
    )

    click.echo("\n--- Evaluation ---")
    click.echo(f"Accuracy:      {result.accuracy:.3f}")
    click.echo(f"Precision:     {result.precision:.3f}")
    click.echo(f"Recall:        {result.recall:.3f}")
    click.echo(f"F1:            {result.f1:.3f}")
    click.echo(f"Profit Factor: {result.profit_factor:.2f}")
    click.echo(f"Test samples:  {result.n_test} ({result.n_positive} positive)")
    click.echo(f"Fold scores:   {[f'{s:.3f}' for s in result.fold_scores]}")

    click.echo("\n--- Top 10 Features ---")
    for i, (feat, imp) in enumerate(list(result.feature_importance.items())[:10], 1):
        click.echo(f"  {i:2d}. {feat:30s} {imp:.4f}")

    click.echo(f"\nModel saved to: {output_dir}/pattern_scorer.json")


@ml.command(name="evaluate")
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("features_path", type=click.Path(exists=True))
@click.option("--label", default="label_5d", help="Label column")
def ml_evaluate(model_path, features_path, label):
    """Evaluate a trained model on feature store data."""
    import xgboost as xgb_lib
    from sklearn.metrics import accuracy_score as acc_score
    from sklearn.metrics import classification_report as cls_report

    from rainier.ml.pattern_scorer import get_feature_columns

    model = xgb_lib.XGBClassifier()
    model.load_model(model_path)

    df = pd.read_parquet(features_path)
    df = df.dropna(subset=[label])
    feature_cols = get_feature_columns(df)

    X = df[feature_cols].values
    y = df[label].values.astype(int)
    y_pred = model.predict(X)

    click.echo(cls_report(y, y_pred, target_names=["bearish", "bullish"]))
    click.echo(f"Accuracy: {acc_score(y, y_pred):.3f}")


@ml.command(name="explain")
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("features_path", type=click.Path(exists=True))
@click.option("--output", default=None, help="Save SHAP results to JSON")
def ml_explain(model_path, features_path, output):
    """Generate SHAP explanations for a trained model."""
    import xgboost as xgb_lib

    from rainier.ml.pattern_scorer import explain_model

    model = xgb_lib.XGBClassifier()
    model.load_model(model_path)

    output_path = Path(output) if output else None
    result = explain_model(model, Path(features_path), output_path)

    click.echo("--- SHAP Feature Importance (Top 10) ---")
    for feat in result["top_10"]:
        click.echo(f"  {feat:30s} {result['mean_shap'][feat]:.4f}")

    if output_path:
        click.echo(f"\nSaved to: {output_path}")


@ml.command(name="regime")
@click.argument("features_path", type=click.Path(exists=True))
@click.option("--symbols", default=None, help="Comma-separated symbols to analyze")
@click.option("--save-model", default=None, help="Save HMM model to path")
def ml_regime(features_path, symbols, save_model):
    """Fit HMM regime detector and show regime distribution."""
    from rainier.ml.regime import HMMRegimeDetector

    df = pd.read_parquet(features_path)

    if symbols:
        sym_list = [s.strip().upper() for s in symbols.split(",")]
        df = df[df["symbol"].isin(sym_list)]

    if "close" not in df.columns:
        click.echo("Error: Parquet file doesn't contain 'close' column")
        return

    # Use first symbol for fitting
    sym = df["symbol"].iloc[0]
    sym_df = df[df["symbol"] == sym].sort_values("date").reset_index(drop=True)

    # Reconstruct minimal OHLCV from features
    ohlcv = pd.DataFrame({
        "open": sym_df["close"].shift(1).fillna(sym_df["close"]),
        "high": sym_df["close"] * (1 + sym_df.get("range", 0.01).clip(lower=0.001)),
        "low": sym_df["close"] * (1 - sym_df.get("range", 0.01).clip(lower=0.001)),
        "close": sym_df["close"],
        "volume": sym_df.get("volume", 1_000_000),
    })

    detector = HMMRegimeDetector(n_states=3)
    regimes = detector.fit_predict(ohlcv)
    summary = detector.regime_summary(regimes)

    click.echo(f"\n--- HMM Regime Detection ({sym}, {len(ohlcv)} bars) ---")
    click.echo("Distribution:")
    for regime, pct in summary["pct"].items():
        count = summary["distribution"][regime]
        dur = summary["avg_duration"].get(regime, "N/A")
        click.echo(f"  {regime:20s} {pct:>6s} ({count:>4d} bars, avg duration: {dur})")

    if save_model:
        detector.save(Path(save_model))
        click.echo(f"\nModel saved to: {save_model}")


@ml.command(name="select-features")
@click.argument("features_path", type=click.Path(exists=True))
@click.option("--label", default="label_5d", help="Label column to optimize for")
@click.option("--folds", default=5, type=int, help="Walk-forward CV folds")
@click.option("--min-features", default=5, type=int, help="Minimum features to test")
@click.option("--max-features", default=40, type=int, help="Maximum features to test")
@click.option(
    "--stability", default=0.6, type=float,
    help="Stability threshold (fraction of folds a feature must appear in)",
)
@click.option(
    "--methods", default="mdi,mda",
    help="Importance methods: mdi,mda,shap (comma-separated)",
)
@click.option("--output", default=None, help="Save results to JSON")
def ml_select_features(
    features_path, label, folds, min_features, max_features, stability, methods, output,
):
    """Select optimal feature set using nested walk-forward CV.

    Runs feature importance inside each training fold (no data leakage),
    then ranks features by stability across folds. Sweeps feature counts
    to find the set that maximizes out-of-sample profit factor.
    """
    from rainier.ml.feature_selector import SelectionConfig, select_features

    method_list = [m.strip() for m in methods.split(",")]
    config = SelectionConfig(
        label_col=label,
        n_folds=folds,
        min_features=min_features,
        max_features=max_features,
        stability_threshold=stability,
        methods=method_list,
    )
    output_path = Path(output) if output else None

    click.echo(f"Feature selection: {features_path}")
    click.echo(f"Label: {label}, Folds: {folds}, Methods: {method_list}")
    click.echo(f"Stability threshold: {stability:.0%}")
    click.echo()

    result = select_features(
        parquet_path=Path(features_path),
        config=config,
        output_path=output_path,
    )

    # --- Sweep results table ---
    click.echo("--- Feature Count Sweep ---")
    click.echo(f"{'Features':>10s}  {'Accuracy':>10s}  {'Acc Std':>10s}  {'Profit Factor':>14s}")
    for s in result.sweep_results:
        pf_str = f"{s['profit_factor']:.2f}" if np.isfinite(s["profit_factor"]) else "inf"
        click.echo(
            f"{s['n_features']:>10d}  {s['accuracy']:>10.3f}  "
            f"{s['accuracy_std']:>10.3f}  {pf_str:>14s}"
        )

    # --- Stability rankings ---
    click.echo("\n--- Feature Stability (top 20) ---")
    click.echo(f"{'Feature':30s}  {'Stability':>10s}  {'Mean Rank':>10s}")
    for r in result.rankings[:20]:
        click.echo(f"{r.name:30s}  {r.stability:>10.0%}  {r.mean_rank:>10.1f}")

    # --- Selected features ---
    click.echo(f"\n--- Optimal Feature Set ({result.optimal_n} features) ---")
    for i, feat in enumerate(result.selected_features, 1):
        click.echo(f"  {i:2d}. {feat}")

    click.echo(
        f"\nTotal: {result.total_features} -> {result.optimal_n} features "
        f"({result.total_features - result.optimal_n} dropped)"
    )

    if output_path:
        click.echo(f"Results saved to: {output_path}")


@ml.command(name="compare")
@click.option("--csv", "csv_path", required=True, type=click.Path(exists=True),
              help="OHLCV CSV file")
@click.option("--symbol", required=True, help="Instrument symbol (e.g. MES)")
@click.option("--timeframe", "tf", default="1H",
              type=click.Choice([t.value for t in Timeframe]),
              help="Bar timeframe")
@click.option("--model", "model_path", default=None, type=click.Path(exists=True),
              help="Trained ML model path (XGBoost JSON). When omitted, only "
                   "the BookScorer baseline is evaluated.")
@click.option("--start", default=None, help="Start date (YYYY-MM-DD)")
@click.option("--end", default=None, help="End date (YYYY-MM-DD)")
@click.option("--walk-forward", "walk_forward", is_flag=True, default=False,
              help="Run walk-forward backtest comparison instead of full-sample.")
@click.option("--wf-train-bars", default=500, type=int,
              help="Walk-forward training window size (bars).")
@click.option("--wf-test-bars", default=100, type=int,
              help="Walk-forward test window size (bars).")
@click.option("--wf-step-bars", default=None, type=int,
              help="Walk-forward step between folds (default: --wf-test-bars).")
@click.pass_context
def ml_compare(ctx, csv_path, symbol, tf, model_path, start, end,
               walk_forward, wf_train_bars, wf_test_bars, wf_step_bars):
    """Compare BookScorer vs MLScorer side-by-side on the same data.

    Closes the ML feedback loop: takes a trained XGBoost model and runs it
    through the backtest engine alongside the rule-based BookScorer so the
    operator can answer "would this model make more money?".

    Use --walk-forward to slide a (train, test) window through the data and
    aggregate per-fold out-of-sample metrics. NOTE: this CLI does NOT refit
    the ML model per fold; the model from --model is loaded once and
    evaluated on each test window. Model retraining is offline via
    ``rainier ml train``. Lookahead leakage is therefore only as good as the
    training cut-off used to produce the model file — pass a model trained
    strictly on data before the CSV's earliest test bar.
    """
    from rainier.core.config import ScorerConfig, SignalConfig
    from rainier.data.csv_provider import CSVProvider
    from rainier.features.extractor import FeatureExtractor
    from rainier.ml.compare import (
        compare_emitters,
        format_comparison_table,
        format_walkforward_table,
        run_walkforward_compare,
    )
    from rainier.ml.scorers import MLScorer
    from rainier.signals.emitter import PinBarSignalEmitter
    from rainier.signals.ml_emitter import MLSignalEmitter

    settings = ctx.obj["settings"]
    timeframe = Timeframe(tf)
    start_dt = datetime.strptime(start, "%Y-%m-%d") if start else None
    end_dt = datetime.strptime(end, "%Y-%m-%d") if end else None

    provider = CSVProvider(Path(csv_path).parent)
    df = provider._read_csv(Path(csv_path), start_dt, end_dt)

    if df is None or df.empty:
        click.echo(f"No data loaded from {csv_path}")
        return

    bt_config = settings.backtest
    sig_config = SignalConfig(
        scorer=ScorerConfig(min_confidence=settings.signal.scorer.min_confidence),
        min_rr_ratio=settings.signal.min_rr_ratio,
    )
    extractor = FeatureExtractor()

    # Always include BookScorer baseline; ML scorer only if --model given.
    def make_book_emitter(_train_df=None) -> PinBarSignalEmitter:
        return PinBarSignalEmitter(settings.analysis, sig_config)

    ml_factory = None
    if model_path is not None:
        ml_scorer = MLScorer(model_path=Path(model_path))

        def _make_ml(_train_df=None) -> MLSignalEmitter:
            # MLScorer is loaded once and reused across folds (the model
            # itself doesn't refit; the caller refits offline via `ml train`).
            return MLSignalEmitter(
                scorer=ml_scorer,
                feature_extractor=extractor,
                analysis_config=settings.analysis,
                signal_config=sig_config,
            )

        ml_factory = _make_ml

    if walk_forward:
        factories: dict = {"BookScorer": make_book_emitter}
        if ml_factory is not None:
            factories["MLScorer"] = ml_factory

        click.echo(
            f"Walk-forward compare: train={wf_train_bars}, test={wf_test_bars}, "
            f"step={wf_step_bars or wf_test_bars}, n_bars={len(df)}"
        )
        wf_result = run_walkforward_compare(
            df=df,
            symbol=symbol,
            timeframe=timeframe,
            emitter_factories=factories,
            train_bars=wf_train_bars,
            test_bars=wf_test_bars,
            step_bars=wf_step_bars,
            config=bt_config,
        )
        click.echo(format_walkforward_table(wf_result))
        return

    emitters: dict = {"BookScorer": make_book_emitter()}
    if ml_factory is not None:
        emitters["MLScorer"] = ml_factory()

    click.echo(f"Single-window compare: n_bars={len(df)}, emitters={list(emitters)}")
    cmp_result = compare_emitters(
        df=df,
        symbol=symbol,
        timeframe=timeframe,
        emitters=emitters,
        config=bt_config,
    )
    click.echo(format_comparison_table(cmp_result))


# ---------------------------------------------------------------------------
# `rainier thesis` — LLM thesis layer (PR1)
# ---------------------------------------------------------------------------

