"""One-shot seed for ``config/sp500_universe.yaml``.

Fetches the S&P 500 constituents table from Wikipedia and writes a YAML
universe file in the same shape as ``config/thematic_universe.yaml`` —
just ``symbol`` + ``sector`` per entry, lowercase snake_case sectors.

Operator runs it once to seed the file, and any time they want to
refresh membership. The YAML is the artifact committed to the repo; the
script is the regenerator. Membership drifts slowly (a few names per
year), so this stays a manual operator action — not a cron.

Flow:

    +-----------+   requests    +-----------+   bs4    +---------------+
    | Wikipedia | ------------> |   HTML    | -------> | rows of cells |
    +-----------+               +-----------+          +-------+-------+
                                                               |
                                  per row: pick Symbol + Sector|
                                  normalize sector name        |
                                  skip rows missing Symbol     |
                                                               v
                                                       +---------------+
                                          ruamel.yaml  |  universe[]   |
                                          atomic write |   {sym,sec}   |
                                                       +-------+-------+
                                                               |
                                                               v
                                                  config/sp500_universe.yaml

Design refs:
    docs/DESIGN-market-breadth-webpage.md
    docs/TASK-PLAN-sp500-universe-yaml-94a6.md §Acceptance
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup
from ruamel.yaml import YAML

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "config" / "sp500_universe.yaml"
DEFAULT_SOURCE = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
DEFAULT_TABLE_ID = "constituents"

# Header of the doc — top-of-file comment block. Kept stable across runs so
# the YAML is byte-identical when the table content hasn't changed.
_HEADER_COMMENT = (
    "# S&P 500 universe — seeded from Wikipedia.\n"
    "#\n"
    "# Source-of-truth for the breadth-webpage child series. Operator\n"
    "# re-runs scripts/seed_sp500_universe.py to refresh; the diff against\n"
    "# the previous commit IS the membership delta.\n"
    "#\n"
    "# Entries are sorted by symbol. Sectors are GICS sector (level-1)\n"
    "# normalized to lowercase snake_case (e.g. ``Information Technology``\n"
    "# -> ``information_technology``).\n"
    "#\n"
    "# Design refs:\n"
    "#   docs/DESIGN-market-breadth-webpage.md\n"
    "#   docs/TASK-PLAN-sp500-universe-yaml-94a6.md\n"
)


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sector normalization
# ---------------------------------------------------------------------------


_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_sector(raw: str) -> str:
    """Return GICS sector as lowercase snake_case.

    ``Information Technology`` -> ``information_technology``
    ``Health Care``            -> ``health_care``
    ``Consumer Staples``       -> ``consumer_staples``

    Collapses any run of non-alphanumeric characters (whitespace, &, /, etc.)
    into a single underscore. Strips leading / trailing underscores.
    """
    s = raw.strip().lower()
    s = _NON_ALNUM.sub("_", s)
    return s.strip("_")


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------


def parse_constituents(html: str, table_id: str = DEFAULT_TABLE_ID) -> list[dict[str, str]]:
    """Parse the constituents table out of the Wikipedia HTML.

    Returns a list of ``{"symbol": ..., "sector": ...}`` dicts in the order
    they appear in the page. Rows whose Symbol cell is empty (or missing)
    are skipped with a WARNING log so the operator can audit.

    The Wikipedia table layout (May 2026 snapshot):

        col 0  Symbol
        col 1  Security
        col 2  GICS Sector       <-- normalized to snake_case
        col 3  GICS Sub-Industry
        col 4  Headquarters Location
        col 5  Date added
        col 6  CIK
        col 7  Founded
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id=table_id)
    if table is None:
        raise ValueError(
            f"could not find <table id='{table_id}'> in the supplied HTML. "
            f"Did the Wikipedia layout change?"
        )

    out: list[dict[str, str]] = []
    rows = table.find_all("tr")
    skipped = 0
    for tr in rows:
        cells = tr.find_all("td")
        if len(cells) < 3:
            # Header row (uses <th>) or a non-data row — silent skip.
            continue
        symbol = cells[0].get_text(strip=True)
        sector = cells[2].get_text(strip=True)
        if not symbol:
            log.warning(
                "skip: row has empty Symbol cell (sector=%r)", sector or "<empty>"
            )
            skipped += 1
            continue
        # Wikipedia uses "." for BRK.B, BF.B etc. yfinance prefers "-".
        # Leave as-is here; downstream consumers normalize per provider.
        out.append({"symbol": symbol, "sector": normalize_sector(sector)})

    if skipped:
        log.info(
            "parsed %d S&P 500 rows (%d skipped for missing Symbol)",
            len(out), skipped,
        )
    return out


# ---------------------------------------------------------------------------
# Render + atomic write
# ---------------------------------------------------------------------------


def _build_yaml_doc(
    entries: list[dict[str, str]],
    asof: str,
    source_url: str,
) -> str:
    """Build the YAML text. ruamel preserves comment + key order.

    Entries are sorted by symbol for deterministic output across runs (the
    Wikipedia table is roughly alphabetical but contains occasional
    out-of-order rows after edits; sorting locks down idempotency).
    """
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    # Default flow style for the entry mapping keeps each entry on one line,
    # matching the spec example:
    #   - { symbol: AAPL, sector: information_technology }
    yaml.default_flow_style = False

    sorted_entries = sorted(entries, key=lambda e: e["symbol"])

    from ruamel.yaml.comments import CommentedMap, CommentedSeq

    universe = CommentedSeq()
    for entry in sorted_entries:
        m = CommentedMap()
        m["symbol"] = entry["symbol"]
        m["sector"] = entry["sector"]
        m.fa.set_flow_style()
        universe.append(m)

    doc = CommentedMap()
    doc["version"] = 1
    doc["schema"] = "sp500_universe.v1"
    doc["asof_seeded"] = asof
    doc["source"] = source_url
    doc["source_table_row_count"] = len(sorted_entries)
    doc["universe"] = universe

    buf = io.StringIO()
    buf.write(_HEADER_COMMENT)
    buf.write("\n")
    yaml.dump(doc, buf)
    return buf.getvalue()


def _atomic_write(path: Path, text: str) -> None:
    """Write to ``path.tmp`` then ``os.replace`` so readers never see a
    half-written file. Mirrors backfill_thematic_universe.py house style.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(text)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def seed_universe(
    html: str,
    out_path: Path,
    asof: str,
    source_url: str = DEFAULT_SOURCE,
    table_id: str = DEFAULT_TABLE_ID,
) -> int:
    """Render YAML for the constituents table in ``html``; return entry count.

    Parameters
    ----------
    html : str
        Wikipedia ``List_of_S&P_500_companies`` HTML — full page or just the
        ``#constituents`` table. The CLI hits the network; tests pass a
        frozen fixture string.
    out_path : Path
        Destination YAML path. Parent directories are created. Write is
        atomic (``.tmp`` then ``os.replace``).
    asof : str
        ISO date string for ``asof_seeded``. The CLI uses today's date.
    source_url : str
        Recorded in the YAML for provenance.
    table_id : str
        HTML id of the constituents table. Wikipedia uses ``constituents``.

    Returns
    -------
    int
        Number of universe entries written.
    """
    entries = parse_constituents(html, table_id=table_id)
    text = _build_yaml_doc(entries, asof=asof, source_url=source_url)
    _atomic_write(Path(out_path), text)
    return len(entries)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _fetch_wikipedia(url: str = DEFAULT_SOURCE, timeout: float = 30.0) -> str:
    """Production fetcher. Imported lazily to keep test collection offline-safe."""
    import requests

    headers = {
        # Wikipedia 403s on the default requests UA.
        "User-Agent": (
            "rainier-seed-sp500-universe/1.0 "
            "(https://github.com/edisonshen/rainier; ops@example.com)"
        ),
    }
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.text


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Seed config/sp500_universe.yaml from the Wikipedia "
            "List_of_S&P_500_companies table. Operator re-runs to refresh."
        )
    )
    p.add_argument("--out", default=str(DEFAULT_OUT), help="Output YAML path.")
    p.add_argument(
        "--source-url",
        default=DEFAULT_SOURCE,
        help="Wikipedia URL (overridable for testing alternate mirrors).",
    )
    p.add_argument(
        "--asof",
        default=None,
        help=(
            "ISO date for ``asof_seeded`` field. Defaults to today's UTC "
            "date in YYYY-MM-DD form."
        ),
    )
    p.add_argument(
        "--from-file",
        default=None,
        help="Read HTML from this file instead of fetching the URL.",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true", help="Log per-row skip details."
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    ns = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if ns.verbose else logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
    )

    if ns.from_file:
        html = Path(ns.from_file).read_text()
    else:
        html = _fetch_wikipedia(ns.source_url)

    if ns.asof is None:
        from datetime import datetime, timezone

        asof = datetime.now(tz=timezone.utc).date().isoformat()
    else:
        asof = ns.asof

    out_path = Path(ns.out)
    n = seed_universe(
        html=html,
        out_path=out_path,
        asof=asof,
        source_url=ns.source_url,
    )
    print(f"wrote {n} entries -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
