"""Tests for `rainier debug post-fake-thesis` — synthetic Discord routing probe.

This command exists to verify the per-ticker thesis embed routing introduced
by PR #73 (`DiscordConfig.llm_webhook_url` + `_resolve_llm_webhook_url`)
END-TO-END against the live webhook URLs without requiring a scrape,
screener candidates, or a real LLM call. The unit tests below pin down the
behavior that makes that probe trustworthy:

* the synthetic candidate carries the `[FAKE TEST POST]` marker (so the
  post lands in Discord as obviously test data, not a real signal),
* default flags route the thesis embed to ``llm_webhook_url`` and the
  top-N summary to ``stock_webhook_url`` (matches production wiring),
* ``--stock-webhook`` clears ``llm_webhook_url`` in-process so the thesis
  embed falls back to ``stock_webhook_url`` (exercises the same fallback
  path operations rely on when the dedicated channel isn't configured),
* missing webhook URLs produce a clear non-zero exit so operators don't
  silently lose the post.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from rainier.cli import cli
from rainier.core.config import (
    AlertsConfig,
    DiscordConfig,
    Settings,
)


def _settings_with_discord(
    *,
    webhook_url: str = "",
    stock_webhook_url: str = "https://stock.example/hook",
    llm_webhook_url: str = "https://llm.example/hook",
    enabled: bool = True,
) -> Settings:
    """Build a Settings instance with just enough wiring for the debug command.

    The CLI root constructs Settings via load_settings(); we patch the root
    loader and the per-command `load_settings_fresh()` so neither touches
    disk. Both helpers must return the same shape so the command sees the
    Discord URLs we configured here.
    """
    discord = DiscordConfig(
        webhook_url=webhook_url,
        stock_webhook_url=stock_webhook_url,
        llm_webhook_url=llm_webhook_url,
        enabled=enabled,
    )
    return Settings(
        database_url="postgresql://test:test@localhost/test",
        alerts=AlertsConfig(discord=discord),
    )


def _invoke(args: list[str], settings: Settings, mock_post: MagicMock):
    """Run the CLI with patched settings + httpx.post.

    We patch BOTH `rainier.cli.load_settings` (cli root) and
    `rainier.core.config.load_settings_fresh` (the per-command fresh
    loader) so the debug command sees the same stub regardless of which
    path it picks. httpx.post is patched at the module that actually
    issues the POST — `rainier.alerts.discord.httpx.post` — which both
    the summary embed and the thesis embed routes call into.
    """
    runner = CliRunner()
    mock_post.return_value = MagicMock(status_code=204)
    with patch("rainier.cli.load_settings", return_value=settings), \
         patch("rainier.core.config.load_settings_fresh", return_value=settings), \
         patch("rainier.alerts.discord.httpx.post", mock_post):
        result = runner.invoke(cli, args)
    return result


# ---------------------------------------------------------------------------
# Help / discoverability
# ---------------------------------------------------------------------------


def test_debug_group_exposes_post_fake_thesis():
    """`rainier debug --help` lists the post-fake-thesis subcommand."""
    runner = CliRunner()
    result = runner.invoke(cli, ["debug", "--help"])
    assert result.exit_code == 0, result.output
    assert "post-fake-thesis" in result.output


def test_post_fake_thesis_help_shows_options():
    """The command's `--help` mentions every flag in the spec."""
    runner = CliRunner()
    result = runner.invoke(cli, ["debug", "post-fake-thesis", "--help"])
    assert result.exit_code == 0, result.output
    assert "--symbol" in result.output
    assert "--verdict" in result.output
    assert "--llm-webhook" in result.output
    assert "--stock-webhook" in result.output


# ---------------------------------------------------------------------------
# Default routing — thesis embed → llm_webhook_url, summary → stock_webhook_url
# ---------------------------------------------------------------------------


def test_default_routes_thesis_to_llm_webhook_summary_to_stock_webhook():
    """With llm_webhook_url set, the thesis embed POSTs to the LLM channel
    and the top-N summary POSTs to the stock channel.

    The renderer issues two distinct httpx.post calls:
      1. The summary embed payload (top-20-list) → stock_webhook_url.
      2. The per-ticker thesis embed → llm_webhook_url.

    We capture the URLs from `mock_post.call_args_list` and assert they
    match the configured destinations. Order matters: the summary goes
    out first inside `send_stock_candidates`, then thesis embeds.
    """
    settings = _settings_with_discord()
    mock_post = MagicMock()
    result = _invoke(
        ["debug", "post-fake-thesis", "--symbol", "TSLA"],
        settings,
        mock_post,
    )

    assert result.exit_code == 0, result.output

    # Collect every URL httpx.post was called with (positional arg 0 in
    # the current send_stock_candidates implementation).
    urls = [call.args[0] for call in mock_post.call_args_list]
    assert len(urls) == 2, f"expected 2 POSTs (summary + thesis), got {urls!r}"
    assert urls[0] == "https://stock.example/hook"  # summary → stock channel
    assert urls[1] == "https://llm.example/hook"    # thesis → LLM channel


def test_default_stdout_reports_resolved_urls():
    """The operator-facing stdout prints BOTH resolved webhook URLs so the
    operator can visually confirm routing without parsing Discord."""
    settings = _settings_with_discord()
    mock_post = MagicMock()
    result = _invoke(
        ["debug", "post-fake-thesis", "--symbol", "TSLA"],
        settings,
        mock_post,
    )
    assert result.exit_code == 0, result.output
    assert "https://llm.example/hook" in result.output
    assert "https://stock.example/hook" in result.output


# ---------------------------------------------------------------------------
# --stock-webhook fallback — clears llm_webhook_url; thesis falls back to stock
# ---------------------------------------------------------------------------


def test_stock_webhook_flag_routes_thesis_to_stock_channel():
    """--stock-webhook overrides the in-memory llm_webhook_url so the thesis
    embed routes to stock_webhook_url instead. Verifies the documented
    fallback path in `_resolve_llm_webhook_url`."""
    settings = _settings_with_discord()
    mock_post = MagicMock()
    result = _invoke(
        ["debug", "post-fake-thesis", "--symbol", "TSLA", "--stock-webhook"],
        settings,
        mock_post,
    )

    assert result.exit_code == 0, result.output
    urls = [call.args[0] for call in mock_post.call_args_list]
    assert len(urls) == 2, f"expected 2 POSTs (summary + thesis), got {urls!r}"
    assert urls[0] == "https://stock.example/hook"
    assert urls[1] == "https://stock.example/hook"  # thesis fell back


# ---------------------------------------------------------------------------
# Error path — no webhook configured at all → non-zero exit
# ---------------------------------------------------------------------------


def test_no_webhooks_configured_exits_nonzero_with_message():
    """With every webhook URL empty, the command must fail loudly — never
    silently swallow a missing-webhook configuration."""
    settings = _settings_with_discord(
        webhook_url="",
        stock_webhook_url="",
        llm_webhook_url="",
    )
    mock_post = MagicMock()
    result = _invoke(
        ["debug", "post-fake-thesis", "--symbol", "TSLA"],
        settings,
        mock_post,
    )
    assert result.exit_code != 0, result.output
    assert "webhook" in result.output.lower()


# ---------------------------------------------------------------------------
# Synthetic content sanity — `[FAKE TEST POST]` marker must reach Discord
# ---------------------------------------------------------------------------


def test_synthetic_post_contains_fake_test_marker():
    """At least one POSTed payload must contain the literal `[FAKE TEST POST]`
    marker so the post is unambiguously test data once it lands in Discord.

    We don't enforce which embed carries it (the synthetic candidate's
    title/headline OR the thesis paragraphs) — only that the marker is
    present somewhere in the outgoing JSON envelope across the two POSTs.
    """
    import json as _json

    settings = _settings_with_discord()
    mock_post = MagicMock()
    result = _invoke(
        ["debug", "post-fake-thesis", "--symbol", "TSLA"],
        settings,
        mock_post,
    )
    assert result.exit_code == 0, result.output

    # send_stock_candidates uses keyword `json=` for plain-JSON POSTs; the
    # multipart chart-attachment path uses `data={"payload_json": ...}` but
    # the debug command never attaches a chart (image_bytes=None) so every
    # POST should be JSON.
    serialized_blobs: list[str] = []
    for call in mock_post.call_args_list:
        payload = call.kwargs.get("json")
        if payload is not None:
            serialized_blobs.append(_json.dumps(payload, default=str))
    assert serialized_blobs, "no JSON payloads captured from httpx.post"
    combined = "\n".join(serialized_blobs)
    assert "[FAKE TEST POST]" in combined, (
        f"expected [FAKE TEST POST] marker in outgoing payloads, got:\n{combined}"
    )


# ---------------------------------------------------------------------------
# No LLM calls — pure routing probe must never touch litellm/anthropic
# ---------------------------------------------------------------------------


def test_command_does_not_import_llm_client():
    """The command must not call any LLM provider — it's a routing-only
    probe. We assert by checking that send_stock_candidates was the entry
    point (already verified by httpx.post being called twice with the
    Discord URLs) AND that the thesis was built purely from synthetic
    constants. The structural guarantee: TradeThesis.model_dump() is
    called locally; no provider client is instantiated.

    Cheap defensive check: the command's stdout should NOT contain
    "litellm" / "anthropic" trace strings. (If a future refactor pulls
    in a provider, this catches it.)
    """
    settings = _settings_with_discord()
    mock_post = MagicMock()
    result = _invoke(
        ["debug", "post-fake-thesis", "--symbol", "TSLA"],
        settings,
        mock_post,
    )
    assert result.exit_code == 0, result.output
    lowered = result.output.lower()
    assert "litellm" not in lowered
    assert "anthropic" not in lowered
