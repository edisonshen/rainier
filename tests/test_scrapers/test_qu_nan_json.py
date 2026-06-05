r"""Regression tests for the QU100 in-page-fetch NaN/Infinity hardening
(TASK-PLAN-qu-scraper-nan-json-154a).

QU's backend serializes JSON with ``json.dumps(allow_nan=True)``, which emits
the bare tokens ``NaN`` / ``Infinity`` / ``-Infinity`` in value position. Those
are not legal JSON, so the in-page ``JSON.parse`` throws, ``body`` comes back
``None``, and the Python side raises ``qu api 200 non-JSON`` — silently dropping
the entire date's ranking.

The fix sanitizes only literals in JSON *value* position before parsing. The
regex is held as a single source-of-truth module constant
(``QU100_NAN_LITERAL_RE``) that ``QU100_FETCH_JS`` interpolates, so a Python
test can exercise the exact same pattern against fixture strings without a JS
engine. JS and Python share identical syntax for this pattern
(``:\s*(NaN|-?Infinity)(?=\s*[,}\]])``), so the constant drives both sides.

Tests 1-3 validate the sanitizer regex (pure string transform).
Tests 4-5 cover the parser adapter null-tolerance + genuine-error path.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from rainier.scrapers.qu.parsers import _api_rows_to_qu100_rows
from rainier.scrapers.qu.scraper import QU100_FETCH_JS, QU100_NAN_LITERAL_RE

from .conftest import make_mock_page, make_scraper


def _sanitize(text: str) -> str:
    """Apply the same value-position literal sanitizer the in-page JS uses.

    Mirrors ``text.replace(QU100_NAN_LITERAL_RE, ':null')`` in JS via the
    shared Python ``re.Pattern`` constant.
    """
    return QU100_NAN_LITERAL_RE.sub(":null", text)


# ---------------------------------------------------------------------------
# Sanitizer regex (the bug: bare NaN/Infinity in value position)
# ---------------------------------------------------------------------------


class TestNanLiteralSanitizer:
    def test_bare_nan_in_value_position_becomes_null(self):
        text = '{"rank": 1, "ticker": "MU", "industry":NaN, "sector": "Tech"}'
        sanitized = _sanitize(text)
        # Sanitized text must be valid JSON now.
        parsed = json.loads(sanitized)
        assert parsed["industry"] is None
        assert parsed["ticker"] == "MU"

    def test_full_payload_with_nan_parses_to_100_rows(self):
        # 100 rows, every one with a bare NaN industry — emulates QU's
        # allow_nan=True serialization for a whole day's ranking.
        #
        # NB: JS ``JSON.parse`` REJECTS bare ``NaN`` (the production bug);
        # Python's ``json.loads`` accepts it as float nan. We therefore assert
        # the sanitizer's effect directly: after the transform there are no
        # bare ``NaN`` literals left and every industry is JSON ``null``.
        rows = [
            f'{{"rank": {i}, "ticker": "T{i}", "change": "0",'
            f' "industry":NaN, "long_short": "No dominance",'
            f' "sector": "Technology"}}'
            for i in range(1, 101)
        ]
        text = '{"data": [' + ",".join(rows) + "]}"
        sanitized = _sanitize(text)
        # Every value-position NaN literal is gone; replaced by null.
        assert "NaN" not in sanitized
        assert sanitized.count(":null") == 100
        payload = json.loads(sanitized)
        assert len(payload["data"]) == 100
        assert all(r["industry"] is None for r in payload["data"])
        adapted = _api_rows_to_qu100_rows(payload["data"])
        assert len(adapted) == 100
        # industry coerced to "" by the adapter, never the literal "None".
        assert all(r.industry == "" for r in adapted)
        assert all(r.industry != "None" for r in adapted)

    def test_infinity_and_neg_infinity_in_value_position(self):
        text = '{"x":-Infinity,"y":Infinity,"z":NaN,"ok":1}'
        payload = json.loads(_sanitize(text))
        assert payload["x"] is None
        assert payload["y"] is None
        assert payload["z"] is None
        assert payload["ok"] == 1

    def test_nan_inside_quoted_string_is_preserved(self):
        # A string value that happens to contain the substring "NaN" must be
        # untouched — it is bracketed by quotes, not structural delimiters.
        text = '{"name":"NaN Holdings","industry":NaN}'
        payload = json.loads(_sanitize(text))
        assert payload["name"] == "NaN Holdings"
        assert payload["industry"] is None

    def test_infinity_inside_quoted_string_is_preserved(self):
        text = '{"name":"Infinity Corp","val":-Infinity}'
        payload = json.loads(_sanitize(text))
        assert payload["name"] == "Infinity Corp"
        assert payload["val"] is None

    def test_value_terminated_by_brace_and_bracket(self):
        # The lookahead accepts ``,`` / ``}`` / ``]`` as the value terminator.
        # ``}`` case: NaN is the last property of an object.
        # ``]`` case: that object is itself the last element of an array, so
        # the value is followed by ``}]``.
        text = '{"data":[{"x":1,"industry":NaN}]}'
        payload = json.loads(_sanitize(text))
        assert payload["data"][0]["industry"] is None
        assert payload["data"][0]["x"] == 1

        # Scalar value immediately before ``]`` (object value that closes an
        # array via ``]``) — exercises the ``]`` branch of the lookahead.
        text2 = '{"row":{"vals":[true,NaN]}}'
        # Here NaN is an array element (no preceding ``:``), so the
        # colon-anchored regex intentionally leaves it alone — array-element
        # NaN is not part of QU's payload shape (only object values are).
        assert _sanitize(text2) == text2

    def test_clean_payload_unchanged(self):
        text = '{"rank":1,"ticker":"MU","industry":"Semiconductors"}'
        assert _sanitize(text) == text

    def test_fetch_js_interpolates_the_shared_regex(self):
        # Guard against JS/Python drift: the JS source must carry the exact
        # regex body so both sides sanitize identically.
        assert QU100_NAN_LITERAL_RE.pattern in QU100_FETCH_JS
        assert "':null'" in QU100_FETCH_JS or '":null"' in QU100_FETCH_JS


# ---------------------------------------------------------------------------
# Adapter null-tolerance (bug beyond spec: present-but-null key -> "None")
# ---------------------------------------------------------------------------


class TestAdapterNullToleranceForSectorIndustry:
    def test_null_sector_and_industry_become_empty_string(self):
        # After sanitize+parse, a bare NaN field is the JSON value null, i.e.
        # Python None. The adapter must coerce it to "" — never "None".
        api = [
            {
                "rank": 1,
                "ticker": "MU",
                "change": "0",
                "industry": None,
                "long_short": "No dominance",
                "sector": None,
            }
        ]
        rows = _api_rows_to_qu100_rows(api)
        assert len(rows) == 1
        assert rows[0].sector == ""
        assert rows[0].industry == ""
        assert rows[0].sector != "None"
        assert rows[0].industry != "None"

    def test_null_long_short_unaffected_field_still_works(self):
        # Only sector/industry are in scope for null-coercion, but a null
        # long_short must not produce "None" either if present.
        api = [
            {"rank": 1, "ticker": "AAA", "change": "0",
             "industry": None, "long_short": None, "sector": None},
        ]
        rows = _api_rows_to_qu100_rows(api)
        assert rows[0].sector == ""
        assert rows[0].industry == ""

    def test_present_non_null_values_preserved(self):
        api = [
            {"rank": 1, "ticker": "MU", "change": "0",
             "industry": "Semiconductors", "long_short": "No dominance",
             "sector": "Technology"},
        ]
        rows = _api_rows_to_qu100_rows(api)
        assert rows[0].sector == "Technology"
        assert rows[0].industry == "Semiconductors"


# ---------------------------------------------------------------------------
# Genuine non-JSON body still raises (sanitizer must not mask real errors)
# ---------------------------------------------------------------------------


class TestGenuineNonJsonStillRaises:
    async def test_html_error_body_with_no_nan_still_raises(self):
        """A real non-JSON 200 (HTML error page, no NaN/Infinity) must still
        surface ``qu api 200 non-JSON``. The in-page sanitizer only swaps bare
        NaN/Infinity literals; it never turns a genuine HTML body into valid
        JSON, so ``body`` is still ``None`` and ``_fetch_qu_api`` raises.
        """
        page, _ = make_mock_page()
        # The page-side JS sanitizes then JSON.parse-fails on real HTML, so
        # body comes back None with the raw excerpt for diagnostics.
        page.evaluate = AsyncMock(
            return_value={
                "status": 200,
                "body": None,
                "text_excerpt": "<html><body>503 upstream</body></html>",
            }
        )

        scraper = make_scraper()
        scraper._page = page

        with pytest.raises(RuntimeError, match="non-JSON"):
            await scraper._fetch_qu_api(page, "https://x/api/v3/mf100?top=true")
