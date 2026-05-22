"""P0 — chart render is byte-identical across processes for the same (ticker, date).

Failing assertion blocks merge. Per docs/DESIGN-qu100-llm-backtest.md [D-025]
and docs/TASK-PLAN-chart-export-determinism-3c8e.md §6.

The load-bearing artifact: two subprocesses each render the same fixture and
must produce identical PNG bytes + identical SHA-256 + identical
CHART_RENDER_VERSION. This is the cache-key contract for the LLM-research
engine — if the chart bytes drift, every cached LLM call re-pays full API cost.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import textwrap
from datetime import datetime, timedelta

import pandas as pd
import pytest

# Render script executed in subprocess. Reads a JSON-serialized OHLCV slice from
# stdin, renders the chart, writes "<digest> <len>\n" then raw PNG bytes to
# stdout. Kept tiny so import latency dominates rather than test orchestration.
_RENDER_SCRIPT = textwrap.dedent("""
    import json
    import sys

    import pandas as pd

    from rainier.llm_thesis.chart_export import render_chart_png

    payload = json.loads(sys.stdin.read())
    df = pd.DataFrame(payload["rows"])
    df.index = pd.to_datetime(payload["index"])
    df.index.name = "date"
    png, digest = render_chart_png(payload["symbol"], df)
    header = f"{digest} {len(png)}\\n".encode()
    sys.stdout.buffer.write(header)
    sys.stdout.buffer.write(png)
""").strip()


def _fixture_payload(symbol: str = "AAPL") -> str:
    """Build the synthetic 60-bar OHLCV slice as a JSON payload.

    Synthetic on purpose: a checked-in pickle would couple the test to pandas
    serialization format. JSON keeps the fixture human-readable and version-
    independent — only the render code path is what we are pinning here.
    """
    import json

    rows = []
    base = datetime(2026, 3, 1)
    for i in range(60):
        rows.append({
            "open":   100.0 + i * 0.1,
            "high":   100.0 + i * 0.1 + 1.0,
            "low":    100.0 + i * 0.1 - 1.0,
            "close":  100.0 + i * 0.1 + 0.5,
            "volume": 1_000_000,
        })
    index = [(base + timedelta(days=i)).isoformat() for i in range(60)]
    return json.dumps({"symbol": symbol, "rows": rows, "index": index})


def _spawn_and_render(payload: str) -> tuple[str, int, bytes]:
    """Spawn a fresh Python process, render, return (digest, png_len, png_bytes)."""
    env = {
        "PYTHONHASHSEED": "0",
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        # Preserve uv / venv resolution so the subprocess can import the package.
        "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV", ""),
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        # Some kaleido / Chromium builds want HOME for cache dirs.
        "HOME": os.environ.get("HOME", "/tmp"),
    }
    proc = subprocess.run(
        [sys.executable, "-c", _RENDER_SCRIPT],
        input=payload.encode(),
        capture_output=True,
        check=True,
        env=env,
        timeout=180,
    )
    header, _, png = proc.stdout.partition(b"\n")
    digest, _, length_str = header.decode().strip().partition(" ")
    return digest, int(length_str), png


@pytest.mark.slow
def test_chart_render_byte_identical_across_processes():
    """P0 — same (symbol, df) renders to identical PNG bytes in two cold subprocesses."""
    payload = _fixture_payload("AAPL")
    digest_a, len_a, png_a = _spawn_and_render(payload)
    digest_b, len_b, png_b = _spawn_and_render(payload)

    assert digest_a == digest_b, (
        f"SHA-256 mismatch across processes: {digest_a} vs {digest_b}"
    )
    assert len_a == len_b, f"PNG byte length differs: {len_a} vs {len_b}"
    assert png_a == png_b, "PNG bytes differ across processes (length matched)"
    assert digest_a == hashlib.sha256(png_a).hexdigest()
    # PNG magic header — guard against the digest collapsing to empty bytes.
    assert png_a[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.slow
def test_chart_render_version_stable_across_processes():
    """CHART_RENDER_VERSION must be identical across two cold imports."""
    code = (
        "from rainier.llm_thesis.chart_render_version import CHART_RENDER_VERSION; "
        "print(CHART_RENDER_VERSION)"
    )
    env = {
        "PYTHONHASHSEED": "0",
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV", ""),
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        "HOME": os.environ.get("HOME", "/tmp"),
    }
    a = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, check=True, env=env, timeout=60
    ).stdout.decode().strip()
    b = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, check=True, env=env, timeout=60
    ).stdout.decode().strip()
    assert a == b, f"CHART_RENDER_VERSION drifted across processes: {a} vs {b}"
    assert len(a) == 64, f"CHART_RENDER_VERSION must be a SHA-256 hex string: {a!r}"
    int(a, 16)  # raises if not hex


def test_chart_render_version_is_module_attribute():
    """Cheap in-process check — `CHART_RENDER_VERSION` is reachable from llm_thesis.

    This protects callers (e.g., compute_input_hash) from a refactor that hides
    the constant or renames the module.
    """
    from rainier.llm_thesis.chart_render_version import CHART_RENDER_VERSION

    assert isinstance(CHART_RENDER_VERSION, str)
    assert len(CHART_RENDER_VERSION) == 64
    int(CHART_RENDER_VERSION, 16)


def test_chart_render_version_re_exported_from_chart_export():
    """The chart_export module also re-exports CHART_RENDER_VERSION for convenience."""
    from rainier.llm_thesis.chart_export import CHART_RENDER_VERSION as exported
    from rainier.llm_thesis.chart_render_version import (
        CHART_RENDER_VERSION as canonical,
    )

    assert exported == canonical


def test_chart_render_version_changes_when_render_file_changes(tmp_path, monkeypatch):
    """Mutating any tracked render-path file must shift the SHA.

    Implementation: import the module's `_compute` directly, swap its tracked
    files list with a tmpdir copy, mutate one file, recompute, assert mismatch.
    Guards against the SHA accidentally being computed over a fixed string.
    """
    from rainier.llm_thesis import chart_render_version as crv

    # Snapshot the real files into tmp_path so we don't touch the source tree.
    copies = []
    for p in crv._RENDER_FILES:
        dst = tmp_path / p.name
        dst.write_bytes(p.read_bytes())
        copies.append(dst)

    monkeypatch.setattr(crv, "_RENDER_FILES", copies)
    before = crv._compute()
    assert before == crv._compute(), "compute is non-deterministic for same inputs"

    # Append a comment byte to one file — recompute should differ.
    copies[0].write_bytes(copies[0].read_bytes() + b"\n# determinism-test marker\n")
    after = crv._compute()
    assert after != before, "CHART_RENDER_VERSION did not change when source bytes mutated"
