"""Fixtures for tests/research/.

In-memory SQLite session for survivorship tests + deterministic
cost-pilot fixture path lookup.
"""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def cost_pilot_10call_fixture() -> Path:
    return FIXTURES_DIR / "cost_pilot_10call.json"


@pytest.fixture
def survivorship_seed_fixture() -> Path:
    return FIXTURES_DIR / "survivorship_seed.json"
