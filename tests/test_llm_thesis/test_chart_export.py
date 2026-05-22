"""Tests for chart_export.render_chart_png — happy path + kaleido failure + determinism guards."""

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


# ---------------------------------------------------------------------------
# Determinism guards — these tests pin every non-determinism source called out
# in docs/TASK-PLAN-chart-export-determinism-3c8e.md §4. The two-process
# byte-identical test lives in test_chart_determinism.py; these are the cheap
# in-process guards that fire at PR time before the slow subprocess test runs.
# ---------------------------------------------------------------------------


def _parse_png_chunks(png: bytes) -> list[tuple[str, int]]:
    """Return [(chunk_type, length), ...] for a PNG byte string."""
    assert png[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    out = []
    i = 8
    while i < len(png):
        length = int.from_bytes(png[i:i + 4], "big")
        chunk_type = png[i + 4:i + 8].decode("ascii", errors="replace")
        out.append((chunk_type, length))
        i += 12 + length
        if chunk_type == "IEND":
            break
    return out


def test_render_strips_known_metadata_chunks():
    """`render_chart_png` must strip `tIME`, `tEXt`, `iTXt`, `zTXt` chunks.

    Belt-and-suspenders: kaleido 0.2.1 today emits only IHDR/IDAT/IEND, but a
    future kaleido upgrade could re-introduce timestamps. The post-render scrub
    in render_chart_png guarantees the LLM cache key never depends on
    wall-clock state injected by an upstream library.
    """
    # Forge a Plotly figure whose `to_image` returns a PNG with a tIME chunk.
    # PNG layout: 8-byte signature + IHDR + tIME + IDAT + IEND. We hand-build a
    # minimal PNG so the test doesn't depend on Pillow.
    import struct
    import zlib

    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        return (
            len(data).to_bytes(4, "big")
            + chunk_type
            + data
            + zlib.crc32(chunk_type + data).to_bytes(4, "big")
        )

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)  # 1x1 RGB
    ihdr = chunk(b"IHDR", ihdr_data)
    # tIME — wall-clock leak. The scrub MUST remove this.
    time_data = struct.pack(">HBBBBB", 2026, 5, 22, 12, 0, 0)
    time_chunk = chunk(b"tIME", time_data)
    text_chunk = chunk(b"tEXt", b"Software\x00Kaleido 0.2.1")
    # Single-pixel IDAT.
    raw = b"\x00\xff\xff\xff"  # filter byte + RGB
    idat = chunk(b"IDAT", zlib.compress(raw))
    iend = chunk(b"IEND", b"")
    forged_png = sig + ihdr + time_chunk + text_chunk + idat + iend

    fig = MagicMock()
    fig.to_image.return_value = forged_png
    with patch("rainier.llm_thesis.chart_export.create_static_stock_chart", return_value=fig):
        png, _digest = render_chart_png("NVDA", _df())

    chunks = _parse_png_chunks(png)
    chunk_types = [c[0] for c in chunks]
    assert "tIME" not in chunk_types, f"tIME chunk leaked through: {chunks}"
    assert "tEXt" not in chunk_types, f"tEXt chunk leaked through: {chunks}"
    # The PNG must still be valid: IHDR first, IEND last, at least one IDAT.
    assert chunk_types[0] == "IHDR"
    assert chunk_types[-1] == "IEND"
    assert "IDAT" in chunk_types


def test_render_passes_explicit_scale_and_engine():
    """`fig.to_image` must be called with engine='kaleido' and scale=1 explicitly.

    Relying on defaults is a determinism hazard — a kaleido version that changes
    its default scale would silently shift PNG bytes.
    """
    fig = MagicMock()
    fig.to_image.return_value = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
    with patch("rainier.llm_thesis.chart_export.create_static_stock_chart", return_value=fig):
        render_chart_png("AAPL", _df())

    fig.to_image.assert_called_once()
    kwargs = fig.to_image.call_args.kwargs
    assert kwargs.get("engine") == "kaleido", f"engine not pinned: {kwargs}"
    assert kwargs.get("scale") == 1, f"scale not pinned: {kwargs}"
    assert kwargs.get("format") == "png"
    # Width/height come from the call defaults — confirm they are 1280x800.
    assert kwargs.get("width") == 1280
    assert kwargs.get("height") == 800


def test_static_stock_chart_pins_font_family():
    """`create_static_stock_chart` must set an explicit font family in its layout.

    A bare Plotly default would leave room for future Plotly upgrades to ship
    a different default font, which would shift PNG bytes silently. We pin a
    family that kaleido's bundled Chromium can resolve everywhere.
    """
    from rainier.viz.charts import create_static_stock_chart

    fig = create_static_stock_chart("AAPL", _df())
    family = fig.layout.font.family
    assert family is not None and family != "", (
        "create_static_stock_chart left font.family unpinned"
    )


def test_static_stock_chart_annotation_order_deterministic_same_process():
    """Rendering the same fixture twice in-process produces identical fig JSON.

    Defends against accidental set-iteration / dict-insertion-order bugs at the
    Python level (before kaleido even runs). Cheap to run; fails loudly the
    instant someone reintroduces non-determinism into the figure assembly.
    """
    from rainier.viz.charts import create_static_stock_chart

    df = _df()
    fig_a = create_static_stock_chart("AAPL", df)
    fig_b = create_static_stock_chart("AAPL", df)
    # Compare canonical JSON — order-sensitive.
    assert fig_a.to_json() == fig_b.to_json()


def test_static_stock_chart_uses_no_wallclock():
    """Static-source check — render path imports must not pull in wall-clock symbols.

    Walks the AST of charts.py + chart_export.py and asserts no usage of
    `datetime.now`, `time.time`, `time.perf_counter`, `pd.Timestamp.now`, or
    `uuid.uuid4`. A future contributor that reintroduces any of these fails
    this test at PR time.
    """
    import ast
    from pathlib import Path

    import rainier.llm_thesis.chart_export as ce
    import rainier.viz.charts as vc

    banned_attrs = {
        ("datetime", "now"),
        ("datetime", "utcnow"),
        ("Timestamp", "now"),
        ("time", "time"),
        ("time", "perf_counter"),
        ("uuid", "uuid4"),
    }
    for module in (vc, ce):
        src = Path(module.__file__).read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                key = (node.value.id, node.attr)
                assert key not in banned_attrs, (
                    f"{module.__name__} references {node.value.id}.{node.attr} "
                    f"— wall-clock leak forbidden in render path"
                )
