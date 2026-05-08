"""Tests for chart_export.render_chart_png — happy path + kaleido failure."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from rainier.llm_thesis.chart_export import render_chart_png


def _df():
    rows = []
    base = datetime(2026, 3, 1)
    for i in range(60):
        rows.append({
            "open": 100 + i * 0.1,
            "high": 100 + i * 0.1 + 1.0,
            "low": 100 + i * 0.1 - 1.0,
            "close": 100 + i * 0.1 + 0.5,
            "volume": 1_000_000,
        })
    df = pd.DataFrame(rows, index=[base + timedelta(days=i) for i in range(60)])
    df.index.name = "date"
    return df


def test_happy_path_returns_bytes_and_digest():
    df = _df()
    fig = MagicMock()
    fig.to_image.return_value = b"PNG-PAYLOAD"
    with patch("rainier.llm_thesis.chart_export.create_static_stock_chart", return_value=fig):
        png, digest = render_chart_png("NVDA", df)
    assert png == b"PNG-PAYLOAD"
    assert len(digest) == 64
    fig.to_image.assert_called_once()


def test_digest_deterministic_for_fixed_input():
    fig = MagicMock()
    fig.to_image.return_value = b"deterministic-payload"
    with patch("rainier.llm_thesis.chart_export.create_static_stock_chart", return_value=fig):
        _, d1 = render_chart_png("NVDA", _df())
        _, d2 = render_chart_png("NVDA", _df())
    assert d1 == d2


def test_kaleido_failure_propagates():
    fig = MagicMock()
    fig.to_image.side_effect = RuntimeError("kaleido subprocess crashed")
    with patch("rainier.llm_thesis.chart_export.create_static_stock_chart", return_value=fig):
        with pytest.raises(RuntimeError):
            render_chart_png("NVDA", _df())
