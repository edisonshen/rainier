"""CLI surface: the consolidated `rainier backtest` group.

The six loose top-level backtest commands were folded into one group, and
the low-value wrappers (`scan`, top-level `report`, `db backfill-from-parquet`,
`db backfill-screened-levels`, `debug post-fake-thesis`) were removed.
"""

from __future__ import annotations

from click.testing import CliRunner

from rainier.cli import cli

BACKTEST_SUBCOMMANDS = ("futures", "pattern", "portfolio", "qu100", "audit", "sma-sweep")

REMOVED_TOP_LEVEL = ("scan", "report", "debug", "backtest-pattern",
                     "backtest-portfolio", "backtest-qu100", "pattern-audit",
                     "sma-sweep")


def test_backtest_group_lists_all_subcommands():
    res = CliRunner().invoke(cli, ["backtest", "--help"])
    assert res.exit_code == 0, res.output
    for sub in BACKTEST_SUBCOMMANDS:
        assert sub in res.output, f"`rainier backtest --help` missing `{sub}`"


def test_backtest_subcommands_resolve():
    runner = CliRunner()
    for sub in BACKTEST_SUBCOMMANDS:
        res = runner.invoke(cli, ["backtest", sub, "--help"])
        assert res.exit_code == 0, f"`backtest {sub} --help` failed:\n{res.output}"


def test_removed_top_level_commands_are_gone():
    root_help = CliRunner().invoke(cli, ["--help"])
    assert root_help.exit_code == 0
    listed = {
        line.strip().split()[0]
        for line in root_help.output.splitlines()
        if line.startswith("  ") and line.strip()
    }
    for removed in REMOVED_TOP_LEVEL:
        assert removed not in listed, f"removed command `{removed}` still registered"


def test_futures_sweep_workers_option_exists():
    res = CliRunner().invoke(cli, ["backtest", "futures", "--help"])
    assert res.exit_code == 0, res.output
    assert "--sweep-workers" in res.output
