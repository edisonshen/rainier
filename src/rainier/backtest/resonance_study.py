"""Resonance-gate A/B study — the §6 falsifiable evaluation, built to *reject*.

Runs the design ``docs/DESIGN-multi-signal-resonance.md`` §6 plan:

  §6.1  Thesis CI    — resonance-bucket × forward-20d win-rate, block-bootstrap
                       CIs + effective non-overlapping N. The win-rate trend's
                       CI must exclude the null or the premise fails.
  §6.2  No snooping  — (a) re-derived ≤2022 train / 2023→now test split
                       (mandatory); (b) True-OOS on synthetic 3×QQQ pre-2010
                       (dot-com + GFC) with pre-registered cost params.
  §6.3  A/B          — resonance-gate vs SMA22/44 gate vs AND/OR combos vs
                       buy-hold TQQQ/QQQ, with the anti-gaming criteria.
  §6.4  Overfit      — deflated metric over #configs, cost/turnover survival.

**Honest verdict:** if the resonance gate does NOT beat the SMA22/44 gate AND
buy-hold out-of-sample, the report says "ship the SMA gate" — a valid outcome.

This module lives in ``backtest/`` and imports only from ``core/`` and
``signals/`` *boundary* code (panel/scorer/gate are signal builders; the gate
is consumed via the ``WeightStrategy`` shape). Costs follow §5.5: real adjusted
TQQQ already embeds 3× financing → charge only turnover + T-bill; the synthetic
``3·r − 2·rate − fee`` is used ONLY for the pre-2010 OOS slice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from rainier.backtest.daily_mtm import (
    ANNUAL,
    PortfolioResult,
    max_drawdown,
    run_portfolio,
    shift_decision,
)
from rainier.signals.panel import PanelInputs, SignalPanel
from rainier.signals.resonance import ResonanceScorer
from rainier.signals.resonance_gate import ResonanceGate

EXPENSE = 0.0095        # TQQQ-class expense ratio (synthetic-3× path only)
DEFAULT_FEE = EXPENSE
WIN_START = "2020-10-01"
TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"

# Threshold grids swept on the train slice (capped per §6.4).
BUY_GRID = (0.55, 0.60, 0.65, 0.70)
SELL_GRID = (0.35, 0.40, 0.45)
WEIGHT_MODES = ("equal", "category_balanced")

# SMA gate (the bar to beat): enter QQQ>SMA_E, exit QQQ<SMA_X.
SMA_ENTRY = 22
SMA_EXIT = 44


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _csv_dir() -> Path:
    """Default CSV directory, resolved from the INVOCATION context, not the
    package install path.

    ``__file__``-relative paths break for a wheel/non-editable install (they
    point inside ``site-packages``). The data lives in the user's working
    project, so prefer ``<cwd>/data/csv`` (and walk up to the repo root if the
    CLI is run from a subdirectory). Fall back to the source-tree location only
    when neither exists, so editable installs still work without a cwd match.
    """
    cwd = Path.cwd()
    for base in (cwd, *cwd.parents):
        cand = base / "data" / "csv"
        if cand.is_dir():
            return cand
    return Path(__file__).resolve().parents[3] / "data" / "csv"


def _load(sym: str, csv_dir: Path) -> pd.DataFrame:
    return pd.read_csv(csv_dir / f"{sym}_1D.csv", parse_dates=["timestamp"])


@dataclass
class World:
    """Aligned market frame for one study window."""
    ts: pd.Series          # timestamps
    df: pd.DataFrame       # QQQ open/high/low/close + vix + spy columns
    tqqq_ret: np.ndarray   # leveraged asset daily simple return (real or synthetic)
    qqq_ret: np.ndarray
    rate_daily: np.ndarray  # daily T-bill rate (cash sleeve)
    synthetic: bool


def build_world(
    start: str = "1999-01-01",
    end: str | None = None,
    real_tqqq: bool = True,
    csv_dir: Path | None = None,
) -> World:
    """Aligned QQQ/SPY/VIX/IRX frame + the leveraged return series.

    real_tqqq=True uses real adjusted TQQQ daily returns (2010+); False builds
    the validated synthetic 3× (``3·r − 2·rate − fee``) for the pre-2010 slice.
    """
    csv_dir = csv_dir or _csv_dir()
    qqq = _load("QQQ_long", csv_dir)
    spy = _load("SPY_long", csv_dir)[["timestamp", "close"]].rename(columns={"close": "spy"})
    vix = _load("VIX_long", csv_dir)[["timestamp", "close"]].rename(columns={"close": "vix"})
    irx = _load("IRX_long", csv_dir)[["timestamp", "close"]].rename(columns={"close": "irx"})
    df = qqq.merge(spy, on="timestamp").merge(vix, on="timestamp").merge(irx, on="timestamp")
    if real_tqqq:
        tq = _load("TQQQ_long", csv_dir)[["timestamp", "close"]].rename(columns={"close": "tqqq"})
        df = df.merge(tq, on="timestamp", how="left")
        # Drop pre-inception rows (TQQQ launched 2010-02): a left-join leaves the
        # tqqq column NaN before then, and pct_change().fillna(0) would otherwise
        # fabricate years of flat 0% "real TQQQ" history. Start at real inception
        # so any caller (not just the internal 2019-06 window) gets honest data.
        df = df[df["tqqq"].notna()].reset_index(drop=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["irx"] = df["irx"].ffill().fillna(2.0) / 100.0
    rate_daily = (df["irx"] / ANNUAL).to_numpy()
    qret = df["close"].pct_change().fillna(0).to_numpy()
    if real_tqqq:
        tqqq_ret = df["tqqq"].pct_change().fillna(0).to_numpy()
    else:
        tqqq_ret = 3.0 * qret - 2.0 * rate_daily - EXPENSE / ANNUAL
    lo = pd.Timestamp(start, tz="UTC")
    mask = df["timestamp"] >= lo
    if end is not None:
        mask &= df["timestamp"] <= pd.Timestamp(end, tz="UTC")
    df = df[mask].reset_index(drop=True)
    sl = mask[mask].index
    return World(
        ts=df["timestamp"],
        df=df,
        tqqq_ret=tqqq_ret[sl][: len(df)],
        qqq_ret=qret[sl][: len(df)],
        rate_daily=rate_daily[sl][: len(df)],
        synthetic=not real_tqqq,
    )


def panel_inputs(world: World) -> PanelInputs:
    return PanelInputs(
        qqq=world.df[["open", "high", "low", "close"]],
        vix=world.df["vix"],
        spy=world.df["spy"],
        breadth=None,  # frozen-universe breadth excluded from OOS claims (§6.2)
    )


# ---------------------------------------------------------------------------
# Window helpers + metrics measured over a sub-window
# ---------------------------------------------------------------------------

def _window_idx(ts: pd.Series, start: str | None, end: str | None) -> np.ndarray:
    m = np.ones(len(ts), dtype=bool)
    if start:
        m &= (ts >= pd.Timestamp(start, tz="UTC")).to_numpy()
    if end:
        m &= (ts <= pd.Timestamp(end, tz="UTC")).to_numpy()
    return np.where(m)[0]


def metrics_over(
    decision: np.ndarray,
    world: World,
    asset_ret: np.ndarray,
    start: str | None,
    end: str | None,
    name: str,
    *,
    synthetic: bool = False,
) -> PortfolioResult:
    """Sim over the full series, then renormalize equity to a sub-window.

    The gate runs on the full buffered history (warmup honored); metrics are
    measured only over [start, end]. Costs: real TQQQ → turnover + T-bill only;
    synthetic path already nets financing into the return series.
    """
    idx = _window_idx(world.ts, start, end)
    # cash_apy per-day series so pre-2010 uses the historical T-bill (§6.2).
    cash_apy = world.rate_daily * ANNUAL
    full = run_portfolio(name, {"A": decision}, {"A": asset_ret}, n_years=1.0,
                         cash_apy=cash_apy, one_way_cost=0.0003)
    seg = full.equity[idx] / full.equity[idx[0]]
    yrs = max((world.ts.iloc[idx[-1]] - world.ts.iloc[idx[0]]).days / 365.25, 0.1)
    res = PortfolioResult(name, seg)
    res.total_return = seg[-1] - 1.0
    res.cagr = seg[-1] ** (1.0 / yrs) - 1.0
    res.max_dd = max_drawdown(seg)
    res.calmar = res.cagr / res.max_dd if res.max_dd > 0 else float("inf")
    # Switches over the window only. Seed the diff with the position carried
    # INTO the window (the shifted weight just before idx[0]), not 0 — otherwise
    # a strategy already invested when the window opens (e.g. buy-hold over a
    # later sub-window) is miscounted as a fresh entry on the first bar.
    shifted = shift_decision(decision)
    sw = shifted[idx]
    prior = shifted[idx[0] - 1] if idx[0] > 0 else 0.0
    res.switches = int(np.sum(np.abs(np.diff(np.r_[prior, sw])) > 1e-9))
    res.exposure = float(np.mean(sw))
    return res


# ---------------------------------------------------------------------------
# Gate constructors → decision series (un-shifted)
# ---------------------------------------------------------------------------

def resonance_decision(world: World, buy: float, sell: float, mode: str) -> np.ndarray:
    gate = ResonanceGate(ResonanceScorer(SignalPanel(), mode=mode), buy=buy, sell=sell)
    return gate.decide(panel_inputs(world))


def sma_gate_decision(world: World, entry: int = SMA_ENTRY, exit_: int = SMA_EXIT) -> np.ndarray:
    """The bar-to-beat: enter QQQ>SMA_entry, exit QQQ<SMA_exit (state machine)."""
    c = world.df["close"]
    sma_e = c.rolling(entry).mean()
    sma_x = c.rolling(exit_).mean()
    warm = (sma_e.notna() & sma_x.notna()).to_numpy()
    enter = ((c > sma_e).to_numpy() & warm)
    exit_cond = ((c < sma_x).to_numpy() & warm)
    n = len(c)
    w = np.zeros(n)
    s = 0.0
    for t in range(n):
        if s == 0.0 and enter[t]:
            s = 1.0
        elif s == 1.0 and exit_cond[t]:
            s = 0.0
        w[t] = s
    return w


def combine(a: np.ndarray, b: np.ndarray, mode: str) -> np.ndarray:
    if mode == "AND":
        return ((a > 0.5) & (b > 0.5)).astype(float)
    return ((a > 0.5) | (b > 0.5)).astype(float)


# ---------------------------------------------------------------------------
# §6.1 — thesis CI (bucketed forward-20d win-rate with block bootstrap)
# ---------------------------------------------------------------------------

@dataclass
class ThesisResult:
    buckets: list[tuple[str, float, int]]   # (label, win_rate, effective_N)
    slope_point: float
    slope_ci: tuple[float, float]
    excludes_null: bool


def _forward_winrate(score: np.ndarray, fwd_ret: np.ndarray, edges) -> list:
    out = []
    for lo, hi in edges:
        m = (score >= lo) & (score < hi) & np.isfinite(fwd_ret)
        wins = fwd_ret[m] > 0
        wr = float(wins.mean()) if m.sum() else float("nan")
        eff = int(max(1, m.sum() // 20))  # ~N/20 non-overlapping (20d windows)
        out.append((f"{int(lo*100)}-{int(hi*100)}%", wr, eff))
    return out


def thesis_test(world: World, score: np.ndarray, fwd: int = 20, n_boot: int = 2000,
                block: int = 20, seed: int = 0) -> ThesisResult:
    """Win-rate by score bucket + block-bootstrap CI on the bucket→win-rate slope.

    The slope of bucket-index vs win-rate must have a CI excluding 0 (the null
    "more agreement does not raise win-rate"). Block bootstrap (20-day blocks)
    respects the forward-window overlap.
    """
    cum = np.cumprod(1 + world.tqqq_ret)
    n = len(score)
    fwd_ret = np.full(n, np.nan)
    for t in range(n - fwd):
        fwd_ret[t] = cum[t + fwd] / cum[t] - 1.0
    edges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0001)]
    buckets = _forward_winrate(score, fwd_ret, edges)

    def slope(sc, fr):
        rows = _forward_winrate(sc, fr, edges)
        xs = np.arange(len(rows))
        ys = np.array([r[1] for r in rows])
        ok = np.isfinite(ys)
        if ok.sum() < 2:
            return np.nan
        return float(np.polyfit(xs[ok], ys[ok], 1)[0])

    point = slope(score, fwd_ret)
    rng = np.random.default_rng(seed)
    valid = np.where(np.isfinite(fwd_ret))[0]
    slopes = []
    nblocks = max(1, len(valid) // block)
    for _ in range(n_boot):
        starts = rng.integers(0, max(1, len(valid) - block), size=nblocks)
        idx = np.concatenate([valid[s:s + block] for s in starts])
        s = slope(score[idx], fwd_ret[idx])
        if np.isfinite(s):
            slopes.append(s)
    lo, hi = np.percentile(slopes, [2.5, 97.5]) if slopes else (np.nan, np.nan)
    return ThesisResult(buckets, point, (float(lo), float(hi)),
                        excludes_null=bool(np.isfinite(lo) and lo > 0))


# ---------------------------------------------------------------------------
# §6.4 — deflated Calmar over #configs tried (López de Prado spirit)
# ---------------------------------------------------------------------------

def deflate(best: float, n_configs: int) -> float:
    """Crude haircut: shrink the best metric by the multiple-testing factor.

    Not a full DSR (we lack per-config Sharpe variance), but a transparent,
    monotone penalty: divide by (1 + ln(n_configs)). Reported alongside the raw
    metric so the reader sees the deflation, not just the flattering number.
    """
    if n_configs <= 1 or best <= 0:
        # Deflation only ever HURTS a metric. Dividing a negative Calmar by the
        # >1 factor moves it toward zero (flattering a losing config), so the
        # haircut applies only to positive metrics; non-positive pass through.
        return best
    return best / (1.0 + np.log(n_configs))


# ---------------------------------------------------------------------------
# §6.2/§6.3 — train/test selection + A/B
# ---------------------------------------------------------------------------

@dataclass
class GateConfig:
    buy: float
    sell: float
    mode: str

    def label(self) -> str:
        return f"BUY{self.buy:.2f}/SELL{self.sell:.2f}/{self.mode}"


def select_on_train(world: World, train_end: str = TRAIN_END) -> tuple[GateConfig, int, float]:
    """Re-derive the resonance config on ≤train_end ONLY → pick best train Calmar.

    Returns (best_config, n_configs_tried, best_train_calmar). This is the §6.2
    mandatory re-derived split: nothing past train_end informs the pick.
    """
    best, best_calmar, n_tried = None, -np.inf, 0
    nan_fallback: GateConfig | None = None  # used only if NOTHING finite/inf ran
    for mode in WEIGHT_MODES:
        for buy in BUY_GRID:
            for sell in SELL_GRID:
                if not buy > sell:
                    continue
                n_tried += 1
                dec = resonance_decision(world, buy, sell, mode)
                m = metrics_over(dec, world, world.tqqq_ret, WIN_START, train_end,
                                 "train")
                cfg = GateConfig(buy, sell, mode)
                # A zero-drawdown candidate has Calmar = +inf and IS the best —
                # do NOT reject inf (the old `np.isfinite` filter dropped it and
                # could leave best=None → crash). Skip NaN (never comparable): a
                # NaN must NOT seed best_calmar, or `m.calmar > NaN` is forever
                # False and finite candidates can never win (codex P2). Stash it
                # only as a last-resort fallback when no comparable config exists.
                if np.isnan(m.calmar):
                    if nan_fallback is None:
                        nan_fallback = cfg
                    continue
                if m.calmar > best_calmar:
                    best_calmar, best = m.calmar, cfg
    if best is None:  # every candidate was NaN
        return nan_fallback, n_tried, best_calmar
    return best, n_tried, best_calmar


@dataclass
class ABRow:
    name: str
    ret: float
    dd: float
    calmar: float
    switches: int
    beats_sma_and_bh: bool = False


def beats_baselines(r: PortfolioResult, sma_r: PortfolioResult,
                    bh_r: PortfolioResult) -> bool:
    """§6.3 pass: beat BOTH the SMA gate and buy-hold on Calmar AND drawdown.

    Drawdown must undercut both baselines (codex P2) — a higher-Calmar row with
    a deeper drawdown than the SMA gate is NOT a win.
    """
    return (r.calmar > sma_r.calmar and r.calmar > bh_r.calmar
            and r.max_dd < sma_r.max_dd and r.max_dd < bh_r.max_dd)


def ab_table(world: World, cfg: GateConfig, start: str | None, end: str | None,
             label: str) -> tuple[list[ABRow], dict]:
    """The §6.3 A/B over a window: all comparators + anti-gaming flags."""
    res_dec = resonance_decision(world, cfg.buy, cfg.sell, cfg.mode)
    sma_dec = sma_gate_decision(world)
    and_dec = combine(res_dec, sma_dec, "AND")
    or_dec = combine(res_dec, sma_dec, "OR")
    ones = np.ones(len(world.tqqq_ret))

    def m(dec, nm, ret=None):
        return metrics_over(dec, world, ret if ret is not None else world.tqqq_ret,
                            start, end, nm)

    rows = {
        "resonance": m(res_dec, "resonance-gate"),
        "sma": m(sma_dec, "SMA22/44 gate"),
        "and": m(and_dec, "resonance AND SMA"),
        "or": m(or_dec, "resonance OR SMA"),
        "bh_tqqq": m(ones, "buy-hold TQQQ"),
        "bh_qqq": m(ones, "buy-hold QQQ", ret=world.qqq_ret),
    }
    sma_r, bh_r = rows["sma"], rows["bh_tqqq"]

    def beats(r):
        return beats_baselines(r, sma_r, bh_r)

    # anti-gaming: a combo winner must beat its SMA component AND resonance-only
    # must beat buy-hold on this slice.
    res_beats_bh = rows["resonance"].calmar > bh_r.calmar
    out = []
    for key in ("resonance", "sma", "and", "or", "bh_tqqq", "bh_qqq"):
        r = rows[key]
        flag = beats(r)
        if key in ("and", "or"):
            flag = flag and r.calmar > sma_r.calmar and res_beats_bh
        out.append(ABRow(r.name, r.total_return, r.max_dd, r.calmar, r.switches, bool(flag)))
    anti = {
        "res_beats_bh": bool(res_beats_bh),
        "resonance_beats_sma_and_bh": bool(beats(rows["resonance"])),
    }
    return out, anti


# ---------------------------------------------------------------------------
# Top-level study
# ---------------------------------------------------------------------------

@dataclass
class StudyResult:
    window: str
    thesis: ThesisResult
    train_cfg: GateConfig
    n_configs: int
    train_calmar: float
    deflated_train_calmar: float
    test_ab: list[ABRow]
    test_anti: dict
    oos_ab: list[ABRow] | None
    oos_anti: dict | None
    verdict: str

    extra: dict = field(default_factory=dict)


def run_study(csv_dir: Path | None = None, n_boot: int = 2000) -> StudyResult:
    real = build_world(start="2019-06-01", real_tqqq=True, csv_dir=csv_dir)
    window = f"{str(real.ts.iat[0])[:10]} → {str(real.ts.iat[-1])[:10]}"

    # §6.1 thesis CI (category-balanced score over the measurement window)
    score = ResonanceScorer(SignalPanel(), mode="category_balanced").score(
        SignalPanel().compute(panel_inputs(real)))
    win_idx = _window_idx(real.ts, WIN_START, None)
    thesis = thesis_test(
        World(real.ts.iloc[win_idx].reset_index(drop=True),
              real.df.iloc[win_idx].reset_index(drop=True),
              real.tqqq_ret[win_idx], real.qqq_ret[win_idx],
              real.rate_daily[win_idx], real.synthetic),
        score[win_idx], n_boot=n_boot)

    # §6.2(a) re-derived split: select on ≤2022, report once on 2023→now
    cfg, n_cfg, train_calmar = select_on_train(real)
    test_ab, test_anti = ab_table(real, cfg, TEST_START, None, "test 2023→now")

    # §6.2(b) True-OOS: frozen cfg on synthetic 3×QQQ pre-2010 (dot-com + GFC).
    # Skip ONLY when the pre-2010 data is genuinely unavailable (missing CSV or
    # too few rows for the 2000-2009 window). Real bugs in build_world/ab_table
    # MUST surface — this OOS slice is load-bearing for the §6.2 verdict, so a
    # blanket except would let a broken evaluation masquerade as a clean run.
    oos_ab = oos_anti = None
    try:
        synth = build_world(start="1999-06-01", end="2009-12-31",
                            real_tqqq=False, csv_dir=csv_dir)
        if len(_window_idx(synth.ts, "2000-01-01", "2009-12-31")) < 60:
            raise FileNotFoundError("pre-2010 OOS window has too few rows")
        oos_ab, oos_anti = ab_table(synth, cfg, "2000-01-01", "2009-12-31",
                                    "pre-2010 synthetic 3×QQQ")
    except FileNotFoundError:
        # genuine data-unavailable: leave the OOS slice unreported (not a failure)
        oos_ab = oos_anti = None

    # Verdict (§6.3 anti-gaming). The combination mode (resonance-only / AND /
    # OR) was NOT selected on train (select_on_train sweeps weights+thresholds
    # only), so letting a combo win on the held-out test slice would let the
    # test window pick the mode after the fact — data snooping (codex P2). Per
    # the anti-gaming rule, the load-bearing claim is that the **resonance-only**
    # gate itself beats SMA+buy-hold; a combo that wins only because resonance-
    # only does not is "the SMA gate in disguise". So the ship verdict requires
    # resonance-only to win — combos are reported descriptively but cannot flip
    # the verdict. Resonance/combo must also survive the pre-2010 OOS slice.
    res_row = next(r for r in test_ab if r.name == "resonance-gate")
    test_pass = res_row.beats_sma_and_bh and test_anti["res_beats_bh"]
    res_oos = next((r for r in (oos_ab or []) if r.name == "resonance-gate"), None)
    oos_pass = bool(res_oos and res_oos.beats_sma_and_bh)
    if not thesis.excludes_null:
        verdict = ("PREMISE FAILS — §6.1 win-rate slope CI includes 0; the "
                   "more-agreement→higher-win-rate lead is not significant. "
                   "Ship the SMA gate.")
    elif test_pass and oos_pass:
        verdict = ("SHIP THE RESONANCE GATE — it beats the SMA22/44 gate and "
                   "buy-hold on the re-derived 2023→now split AND survives the "
                   "pre-2010 synthetic OOS, passing the anti-gaming criteria.")
    else:
        verdict = ("SHIP THE SMA GATE — the resonance gate does not beat the "
                   "SMA22/44 gate + buy-hold across the required OOS slices "
                   "(§6.3). Simpler wins; this is a valid, expected outcome.")

    return StudyResult(
        window=window, thesis=thesis, train_cfg=cfg, n_configs=n_cfg,
        train_calmar=train_calmar,
        deflated_train_calmar=deflate(train_calmar, n_cfg),
        test_ab=test_ab, test_anti=test_anti, oos_ab=oos_ab, oos_anti=oos_anti,
        verdict=verdict,
    )
