"""Tests for `rainier ml compare` CLI command.

Composition-root verification: the CLI must wire BookScorer baseline (and
optional MLScorer if --model is given) through compare_emitters or
run_walkforward_compare without us needing to invoke a real XGBoost model.
For the ML branch we substitute a fake MLScorer so the test stays fast and
deterministic — model-loading fidelity is covered separately by
test_ml_emitter / test_ml_compare unit tests.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from click.testing import CliRunner

from rainier.cli import cli


def _write_csv(path: Path, n: int = 400) -> None:
    """Write a synthetic OHLCV CSV the CSVProvider accepts."""
    rows = []
    price = 100.0
    base = datetime(2025, 1, 1)
    for i in range(n):
        cycle = i % 40
        move = 1.0 if cycle < 20 else -1.0
        rows.append({
            "timestamp": (base + timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S"),
            "open": price,
            "high": price + abs(move) + 1.0,
            "low": price - abs(move) - 0.5,
            "close": price + move,
            "volume": 1000 + (i % 10) * 100,
        })
        price = price + move
    pd.DataFrame(rows).to_csv(path, index=False)


class _FakeMLScorer:
    """Stand-in for ml.scorers.MLScorer that needs no XGBoost model."""

    def __init__(self, model_path=None):
        self.model_path = model_path

    def score(self, pattern, features) -> float:
        return 0.5


class TestMLCompareCLI:
    """The composition-root wiring of `rainier ml compare`."""

    def test_baseline_only_runs_without_model(self, tmp_path):
        csv_path = tmp_path / "MES_1H.csv"
        _write_csv(csv_path, n=300)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "ml", "compare",
                "--csv", str(csv_path),
                "--symbol", "MES",
                "--timeframe", "1H",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "BookScorer" in result.output
        assert "SCORER COMPARISON" in result.output

    def test_with_model_runs_both_scorers(self, tmp_path):
        csv_path = tmp_path / "MES_1H.csv"
        _write_csv(csv_path, n=300)
        # We don't actually need a real model file (MLScorer is faked) but
        # click validates --model with exists=True, so write a placeholder.
        model_path = tmp_path / "model.json"
        model_path.write_text("{}")

        runner = CliRunner()
        with patch("rainier.ml.scorers.MLScorer", _FakeMLScorer):
            result = runner.invoke(
                cli,
                [
                    "ml", "compare",
                    "--csv", str(csv_path),
                    "--symbol", "MES",
                    "--timeframe", "1H",
                    "--model", str(model_path),
                ],
            )
        assert result.exit_code == 0, result.output
        assert "BookScorer" in result.output
        assert "MLScorer" in result.output

    def test_walk_forward_baseline_only(self, tmp_path):
        csv_path = tmp_path / "MES_1H.csv"
        _write_csv(csv_path, n=600)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "ml", "compare",
                "--csv", str(csv_path),
                "--symbol", "MES",
                "--timeframe", "1H",
                "--walk-forward",
                "--wf-train-bars", "200",
                "--wf-test-bars", "100",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "WALK-FORWARD COMPARISON" in result.output
        assert "Aggregate" in result.output

    def test_walk_forward_with_model(self, tmp_path):
        csv_path = tmp_path / "MES_1H.csv"
        _write_csv(csv_path, n=600)
        model_path = tmp_path / "model.json"
        model_path.write_text("{}")

        runner = CliRunner()
        with patch("rainier.ml.scorers.MLScorer", _FakeMLScorer):
            result = runner.invoke(
                cli,
                [
                    "ml", "compare",
                    "--csv", str(csv_path),
                    "--symbol", "MES",
                    "--timeframe", "1H",
                    "--model", str(model_path),
                    "--walk-forward",
                    "--wf-train-bars", "200",
                    "--wf-test-bars", "100",
                    "--wf-step-bars", "100",
                ],
            )
        assert result.exit_code == 0, result.output
        assert "WALK-FORWARD COMPARISON" in result.output
        assert "BookScorer" in result.output
        assert "MLScorer" in result.output

    def test_missing_csv_fails_cleanly(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "ml", "compare",
                "--csv", str(tmp_path / "does_not_exist.csv"),
                "--symbol", "MES",
                "--timeframe", "1H",
            ],
        )
        assert result.exit_code != 0
        assert "does_not_exist" in result.output or "Path" in result.output
