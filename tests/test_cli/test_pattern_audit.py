"""`rainier pattern-audit` CLI — wiring smoke test.

The audit logic (corpus build / aggregation / forward returns) is covered in
tests/test_paper/test_pattern_audit.py. Here we only assert the CLI wires
settings.stock_screener into run_pattern_audit and writes the markdown report
to the requested path.
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
from click.testing import CliRunner

from rainier.cli import cli
from rainier.paper.pattern_audit import CORPUS_COLUMNS


def test_pattern_audit_writes_report(tmp_path):
    corpus = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "as_of": pd.Timestamp("2025-06-01").date(),
                "pattern_type": "false_breakdown",
                "direction": "bullish",
                "status": "confirmed",
                "confidence": 0.7,
                "pattern_contribution": 0.45,
                "regime": "bull",
                "close_at_t": 100.0,
                "fwd_return_5d": 0.04,
                "fwd_return_10d": 0.06,
                "fwd_return_20d": 0.09,
            }
        ],
        columns=list(CORPUS_COLUMNS),
    )
    from rainier.paper.pattern_audit import aggregate_to_frame

    agg = aggregate_to_frame(corpus)
    report = tmp_path / "report.md"

    with patch(
        "rainier.paper.pattern_audit.run_pattern_audit",
        return_value=(corpus, agg, tmp_path / "corpus.parquet"),
    ) as mock_run:
        runner = CliRunner()
        result = runner.invoke(
            cli, ["pattern-audit", "--report", str(report), "--symbols", "AAA,BBB"]
        )

    assert result.exit_code == 0, result.output
    mock_run.assert_called_once()
    # symbols parsed into a list and passed through
    assert mock_run.call_args.kwargs["symbols"] == ["AAA", "BBB"]
    assert report.exists()
    body = report.read_text()
    assert "false_breakdown" in body
    assert "pattern forward-return audit" in body.lower()
