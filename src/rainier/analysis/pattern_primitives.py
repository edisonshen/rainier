"""Pattern detection primitives — swing points, necklines, breakouts, volume-price analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class SwingPoint:
    index: int  # Bar index in DataFrame
    price: float  # High for swing high, Low for swing low
    type: str  # "high" or "low"
    strength: int  # Number of bars on each side that are lower/higher


@dataclass(frozen=True, slots=True)
class Neckline:
    price: float  # Neckline price level (mean for horizontal)
    slope: float  # 0 for horizontal, positive for ascending
    touch_points: list[int]  # Bar indices where price touched neckline
    type: str  # "support" or "resistance"


@dataclass(frozen=True, slots=True)
class Breakout:
    bar_index: int
    direction: str  # "up" or "down"
    level: float  # The neckline/level that was broken
    with_volume: bool  # volume breakout = higher confidence
    false_breakout: bool  # Reverses back within 3 bars


@dataclass(frozen=True, slots=True)
class VolumePriceSignal:
    type: str  # "price_up_vol_up", "price_up_vol_down", etc.
    divergence: bool  # True if price and volume disagree (bearish signal)
    vol_ratio: float  # Current volume / average volume


def find_swing_points(df: pd.DataFrame, lookback: int = 5) -> list[SwingPoint]:
    """Find local swing highs and lows using a rolling window approach.

    A swing high at index i requires high[i] to be the strict unique maximum
    of highs[i-lookback : i+lookback+1]. Similarly for swing lows with the minimum.

    For the trailing edge (last `lookback` bars), provisional swing points are
    detected using only the left side of the window. These are marked with
    reduced strength so pattern detectors can use them for live screening.
    """
    highs = df["high"].to_numpy(dtype=np.float64)
    lows = df["low"].to_numpy(dtype=np.float64)
    n = len(highs)
    points: list[SwingPoint] = []

    # Confirmed swing points (full window on both sides)
    for i in range(lookback, n - lookback):
        window_start = i - lookback
        window_end = i + lookback + 1

        # Swing high: high[i] must be strictly greater than all others in window
        high_window = highs[window_start:window_end]
        if highs[i] == np.max(high_window) and np.sum(high_window == highs[i]) == 1:
            points.append(SwingPoint(
                index=i, price=float(highs[i]), type="high", strength=lookback
            ))

        # Swing low: low[i] must be strictly less than all others in window
        low_window = lows[window_start:window_end]
        if lows[i] == np.min(low_window) and np.sum(low_window == lows[i]) == 1:
            points.append(SwingPoint(
                index=i, price=float(lows[i]), type="low", strength=lookback
            ))

    # Provisional swing points at trailing edge (left-side only)
    # Requires at least `lookback` bars to the left, uses whatever bars exist to the right
    min_right = 2  # need at least 2 bars after to avoid noise
    for i in range(max(lookback, n - lookback), n - min_right):
        window_start = i - lookback
        window_end = min(i + lookback + 1, n)
        window_highs = highs[window_start:window_end]
        window_lows = lows[window_start:window_end]

        if highs[i] == np.max(window_highs) and np.sum(window_highs == highs[i]) == 1:
            points.append(SwingPoint(
                index=i, price=float(highs[i]), type="high",
                strength=lookback - 1,  # reduced strength = provisional
            ))

        if lows[i] == np.min(window_lows) and np.sum(window_lows == lows[i]) == 1:
            points.append(SwingPoint(
                index=i, price=float(lows[i]), type="low",
                strength=lookback - 1,  # reduced strength = provisional
            ))

    points.sort(key=lambda sp: sp.index)
    return points


def find_neckline(
    swing_points: list[SwingPoint],
    sp_type: str,
    df: pd.DataFrame,
    tolerance_pct: float = 0.03,
) -> Neckline | None:
    """Fit a neckline through swing points using iterative linear regression.

    Args:
        swing_points: List of detected swing points.
        sp_type: "high" for resistance neckline, "low" for support neckline.
        df: Price DataFrame (used for reference only).
        tolerance_pct: Maximum residual as fraction of mean price for inclusion.

    Returns:
        Neckline if at least 2 touch points found, else None.
    """
    filtered = [sp for sp in swing_points if sp.type == sp_type]
    if len(filtered) < 2:
        return None

    indices = np.array([sp.index for sp in filtered], dtype=np.float64)
    prices = np.array([sp.price for sp in filtered], dtype=np.float64)
    mean_price = float(np.mean(prices))
    threshold = tolerance_pct * mean_price

    # First fit: least squares through all points
    coeffs = np.polyfit(indices, prices, 1)  # [slope, intercept]
    fitted = np.polyval(coeffs, indices)
    residuals = np.abs(prices - fitted)

    # Remove outliers and refit
    inlier_mask = residuals <= threshold
    if np.sum(inlier_mask) < 2:
        return None

    indices_clean = indices[inlier_mask]
    prices_clean = prices[inlier_mask]
    coeffs = np.polyfit(indices_clean, prices_clean, 1)
    slope = float(coeffs[0])
    mean_price_clean = float(np.mean(prices_clean))

    # Determine if slope is effectively zero
    if abs(slope) < 0.001 * mean_price_clean:
        slope = 0.0
        neckline_price = mean_price_clean
    else:
        neckline_price = mean_price_clean

    # Find touch points: swing points within tolerance of the fitted line
    touch_indices: list[int] = []
    for sp in filtered:
        if slope == 0.0:
            fitted_val = neckline_price
        else:
            fitted_val = float(np.polyval(coeffs, sp.index))
        if abs(sp.price - fitted_val) <= threshold:
            touch_indices.append(sp.index)

    if len(touch_indices) < 2:
        return None

    neckline_type = "support" if sp_type == "low" else "resistance"
    return Neckline(
        price=neckline_price,
        slope=slope,
        touch_points=touch_indices,
        type=neckline_type,
    )


def detect_breakout(
    df: pd.DataFrame,
    level: float,
    direction: str,
    start_idx: int = 0,
    vol_multiplier: float = 1.5,
    vol_window: int = 20,
) -> Breakout | None:
    """Detect the first breakout of a price level.

    Args:
        df: Price DataFrame with columns: open, high, low, close, volume.
        level: The price level to check for breakout.
        direction: "up" or "down".
        start_idx: Bar index to start scanning from.
        vol_multiplier: Volume must exceed this multiple of average for volume breakout.
        vol_window: Number of bars for computing average volume.

    Returns:
        Breakout if found, else None.
    """
    closes = df["close"].to_numpy(dtype=np.float64)
    volumes = df["volume"].to_numpy(dtype=np.float64)
    n = len(closes)

    for i in range(start_idx, n):
        broke = (
            (direction == "up" and closes[i] > level)
            or (direction == "down" and closes[i] < level)
        )
        if not broke:
            continue

        # Check volume confirmation
        vol_start = max(0, i - vol_window)
        if vol_start < i:
            avg_vol = float(np.mean(volumes[vol_start:i]))
        else:
            avg_vol = float(volumes[i])
        with_volume = bool(volumes[i] > vol_multiplier * avg_vol) if avg_vol > 0 else False

        # Check for false breakout: price reverses back within 3 bars
        false_breakout = False
        check_end = min(i + 4, n)  # next 3 bars = i+1, i+2, i+3
        for j in range(i + 1, check_end):
            if direction == "up" and closes[j] < level:
                false_breakout = True
                break
            elif direction == "down" and closes[j] > level:
                false_breakout = True
                break

        return Breakout(
            bar_index=i,
            direction=direction,
            level=level,
            with_volume=with_volume,
            false_breakout=false_breakout,
        )

    return None


def analyze_volume_price(df: pd.DataFrame, window: int = 20) -> VolumePriceSignal:
    """Analyze volume-price relationship for the most recent bar.

    Compares the last bar's close and volume against values from `window` bars ago
    to classify the volume-price signal and detect divergences.

    Args:
        df: Price DataFrame with columns: open, high, low, close, volume.
        window: Lookback period for comparison.

    Returns:
        VolumePriceSignal describing the current volume-price dynamics.
    """
    closes = df["close"].to_numpy(dtype=np.float64)
    volumes = df["volume"].to_numpy(dtype=np.float64)
    n = len(closes)

    # Price direction: compare last close vs close `window` bars ago
    ref_idx = max(0, n - 1 - window)
    price_up = closes[-1] >= closes[ref_idx]

    # Volume direction: compare last volume vs average over window
    vol_start = max(0, n - 1 - window)
    avg_vol = float(np.mean(volumes[vol_start: n - 1])) if vol_start < n - 1 else float(
        volumes[-1]
    )
    vol_ratio = float(volumes[-1] / avg_vol) if avg_vol > 0 else 1.0
    vol_up = vol_ratio >= 1.0

    # Classify signal type and divergence
    if price_up and vol_up:
        signal_type = "price_up_vol_up"
        divergence = False
    elif price_up and not vol_up:
        signal_type = "price_up_vol_down"
        divergence = True
    elif not price_up and vol_up:
        signal_type = "price_down_vol_up"
        divergence = False
    else:
        signal_type = "price_down_vol_down"
        divergence = True

    return VolumePriceSignal(
        type=signal_type,
        divergence=divergence,
        vol_ratio=vol_ratio,
    )
