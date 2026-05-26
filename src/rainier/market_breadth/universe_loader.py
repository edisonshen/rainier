"""Read ``config/sp500_universe.yaml`` → list[(symbol, sector)].

The S&P 500 universe YAML uses a flat list-of-dicts shape (different from
``thematic_universe.yaml`` which is sector-keyed). One entry per S&P 500
constituent::

    version: 1
    schema: sp500_universe.v1
    universe:
      - {symbol: 'AAPL',  sector: information_technology}
      - {symbol: 'BRK.B', sector: financials}
      ...

Why a separate loader from ``rainier.breadth.universe_loader``? Different
input schema (list-of-dicts vs sector-keyed map) and different downstream
join semantics. The two modules share no code on purpose — the naming
overlap is a known wart; renaming is out of scope per the task brief.

Design refs:
    docs/DESIGN-market-breadth-webpage.md §3.2
    docs/TASK-PLAN-sp500-ohlcv-backfill-47b8.md §Expected files / surfaces
"""

from __future__ import annotations

from pathlib import Path

import yaml


def load_sp500_universe(yaml_path: str | Path) -> list[tuple[str, str]]:
    """Parse the universe YAML into ``[(symbol, sector), ...]``.

    YAML iteration order is preserved so callers that care about display
    sorting (typically by symbol, since the seed script writes sorted) get
    a stable list.

    Raises
    ------
    ValueError
        If the YAML lacks a ``universe`` key or it isn't a list.
    """
    path = Path(yaml_path)
    parsed = yaml.safe_load(path.read_bytes())

    universe = (parsed or {}).get("universe")
    if not isinstance(universe, list):
        raise ValueError(
            f"{path}: expected top-level `universe:` to be a list of "
            f"{{symbol, sector}} dicts; got {type(universe).__name__}."
        )

    out: list[tuple[str, str]] = []
    for i, entry in enumerate(universe):
        if not isinstance(entry, dict):
            raise ValueError(
                f"{path}: universe[{i}] must be a dict, got "
                f"{type(entry).__name__}: {entry!r}"
            )
        sym = entry.get("symbol")
        sec = entry.get("sector")
        if not isinstance(sym, str) or not sym:
            raise ValueError(
                f"{path}: universe[{i}] missing/empty `symbol`: {entry!r}"
            )
        if not isinstance(sec, str) or not sec:
            raise ValueError(
                f"{path}: universe[{i}] missing/empty `sector`: {entry!r}"
            )
        out.append((sym, sec))
    return out
