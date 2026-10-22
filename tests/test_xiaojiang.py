"""Spec tests for the Xiaojiang line signal on real MES bars (fixture: Sep-Oct 2026)."""

from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from rainier.core.types import Timeframe
from rainier.research.xiaojiang.lines import LineBook, Lines
from rainier.research.xiaojiang.sim import Market, XJParams, plan_orders, simulate, targets
from rainier.research.xiaojiang.tune import evaluate

FIXTURE = Path(__file__).parent / "fixtures" / "mes_sr_small.parquet"
LOOSE = XJParams(min_rank=1, min_conf=1, max_touches=99, exit="next")


def _frames(df: pd.DataFrame) -> dict[Timeframe, pd.DataFrame]:
    return {Timeframe(tf): g.drop(columns="timeframe").sort_values("timestamp").reset_index(drop=True)
            for tf, g in df.groupby("timeframe")}


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return pd.read_parquet(FIXTURE)


@pytest.fixture(scope="module")
def mes(raw) -> dict[Timeframe, pd.DataFrame]:
    df = raw.copy()  # same naive-PT timestamps the CSV loader produces
    df["timestamp"] = df.timestamp.dt.tz_convert("America/Los_Angeles").dt.tz_localize(None)
    return _frames(df)


@pytest.fixture(scope="module")
def mkt(mes) -> Market:
    return Market(mes, LineBook(mes))


def test_lines_are_point_in_time(mes, mkt) -> None:
    """The lines an hour starts with are identical whether or not later bars exist."""
    i = int(len(mkt.c) * 0.7)
    t = mkt.t[i].floor("h")
    i = int(mkt.t.searchsorted(t))
    cut = {tf: df[df.timestamp < t] for tf, df in mes.items()}
    cut[Timeframe.M5] = mes[Timeframe.M5].iloc[: i + 1]
    assert LineBook(cut).at(t.value, i) == mkt.book.for_bar(i)
    assert len(mkt.book.for_bar(i).price) > 4


def test_plan_buys_support_sells_resistance() -> None:
    lines = Lines([100.0, 110.0, 120.0, 130.0, 140.0, 150.0], [4, 4, 1, 4, 4, 4],
                  [2, 2, 2, 2, 2, 1], [1, 1, 1, 1, 9, 1])
    p = XJParams(k_near=3, stop_pts=12.0)
    got = {(o.direction, o.line, o.stop) for o in plan_orders(lines, 125.0, p)}
    # 120 is a thin (15m) line, 140 was tested 9 times, 150 has no confluence.
    assert got == {(1, 110.0, 98.0), (1, 100.0, 88.0), (-1, 130.0, 142.0)}
    assert {o.direction for o in plan_orders(lines, 125.0, p, bias=-1)} == {-1}
    assert [o.line for o in plan_orders(lines, 125.0, p, used=[110.5])] == [100.0, 130.0]
    # Exits: the next lines at least 1R away, half at each; or a fixed R multiple.
    assert targets(lines, 1, 110.0, 98.0, p) == [130.0, 140.0]
    assert targets(lines, -1, 130.0, 142.0, XJParams(exit="rr2")) == [106.0]


def test_backtest_trades_follow_the_rules(mkt) -> None:
    tr = simulate(mkt, LOOSE)
    assert len(tr) > 30 and set(tr.exit) <= {"stop", "target", "session_end"}
    assert (tr.t_in.iloc[1:].to_numpy() >= tr.t_out.iloc[:-1].to_numpy()).all()  # one at a time
    # Left side: filled at the line or better, never risking more than the fixed stop.
    assert ((tr.dir * (tr.line - tr.entry)) >= 0).all() and (tr.risk <= 12.0 + 1e-9).all()
    assert (tr.pts[tr.exit == "target"] > 0).all() and (tr.pts[tr.exit == "stop"] < 0).all()
    gaps = mkt.t[mkt.last_before_gap]
    assert tr.t_out[tr.exit == "session_end"].isin(gaps).all()


def test_random_lines_control_trades_elsewhere(mkt) -> None:
    periods = {"all": (mkt.t.iloc[0], mkt.t.iloc[-1] + pd.Timedelta("1min"))}
    real, rand = evaluate(mkt, LOOSE, periods), evaluate(mkt, LOOSE, periods, seed=1)
    assert real["all"]["trades"] > 0 and rand["all"]["trades"] > 0
    assert real != rand


def test_cli_signals_and_backtest(raw, tmp_path) -> None:
    from rainier.cli import cli

    for tf, g in raw.groupby("timeframe"):
        g.drop(columns="timeframe").to_csv(tmp_path / f"MES_{tf}.csv", index=False)
    loose = [a for k in ("min_rank=1", "min_conf=1", "max_touches=99") for a in ("--param", k)]
    runner = CliRunner()
    r = runner.invoke(cli, ["xj-signals", "--data-dir", str(tmp_path), *loose])
    assert r.exit_code == 0, r.output
    assert "BUY  LIMIT" in r.output and "SELL LIMIT" in r.output and "stop" in r.output
    r = runner.invoke(cli, ["xj-backtest", "--data-dir", str(tmp_path), "--train", "2026-09-08",
                            "2026-10-01", "--holdout", "2026-10-01", "2026-10-21", "--seeds", "3",
                            *loose])
    assert r.exit_code == 0, r.output
    assert "holdout" in r.output and "random median" in r.output
    r = runner.invoke(cli, ["xj-signals", "--data-dir", str(tmp_path), "--param", "bogus=1"])
    assert r.exit_code != 0
