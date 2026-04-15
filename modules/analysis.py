"""
modules/analysis.py
Technical indicator computation + pattern detection.
All indicators are calculated manually (no pandas_ta / ta-lib dependency).
"""

import numpy as np
import pandas as pd
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Low-level indicator helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (macd_line, signal_line, histogram)."""
    ema_fast = compute_ema(close, fast)
    ema_slow = compute_ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def compute_bollinger(
    close: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Returns (upper, mid, lower, width) where width = (upper-lower)/mid."""
    mid = compute_sma(close, period)
    std = close.rolling(period, min_periods=period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    width = (upper - lower) / mid.replace(0, np.nan)
    return upper, mid, lower, width


def _compute_indicators(df: pd.DataFrame) -> dict:
    """Compute all indicators for a daily/weekly OHLCV DataFrame."""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    sma20 = compute_sma(close, 20)
    sma50 = compute_sma(close, 50)
    sma200 = compute_sma(close, 200)

    rsi = compute_rsi(close, 14)
    macd_line, signal_line, histogram = compute_macd(close)
    atr = compute_atr(high, low, close, 14)
    bb_upper, bb_mid, bb_lower, bb_width = compute_bollinger(close, 20)
    vol_avg20 = compute_sma(volume, 20)

    return {
        "close": close,
        "high": high,
        "low": low,
        "volume": volume,
        "sma20": sma20,
        "sma50": sma50,
        "sma200": sma200,
        "rsi": rsi,
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram,
        "atr": atr,
        "bb_upper": bb_upper,
        "bb_mid": bb_mid,
        "bb_lower": bb_lower,
        "bb_width": bb_width,
        "vol_avg20": vol_avg20,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pattern detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_patterns(ind: dict) -> list[tuple[str, str, str]]:
    """
    Detect technical patterns from pre-computed indicators.

    Returns
    -------
    list of (name, emoji, strength) — strength: 'high' | 'medium' | 'low'
    """
    patterns: list[tuple[str, str, str]] = []
    close = ind["close"]
    high = ind["high"]
    low = ind["low"]
    volume = ind["volume"]

    n = len(close)
    if n < 60:
        return patterns

    current = close.iloc[-1]
    sma20 = ind["sma20"]
    sma50 = ind["sma50"]
    sma200 = ind["sma200"]
    rsi = ind["rsi"]
    macd = ind["macd"]
    signal = ind["signal"]
    histogram = ind["histogram"]
    bb_width = ind["bb_width"]
    vol_avg20 = ind["vol_avg20"]

    # ── 1. Breakout / Near 52-Week High ──────────────────────────────────────
    lookback = min(252, n)
    week52_high = high.iloc[-lookback:].max()
    dist = (week52_high - current) / week52_high if week52_high > 0 else 1.0

    if dist <= 0.01:
        patterns.append(("52W Breakout", "🚀", "high"))
    elif dist <= 0.05:
        patterns.append(("Near 52W High", "📈", "high"))

    # ── 2. Bull Flag ─────────────────────────────────────────────────────────
    if n >= 25:
        # Strong move in the prior 15 trading days (days -21 to -6)
        prior_gain = (close.iloc[-6] - close.iloc[-21]) / close.iloc[-21] if close.iloc[-21] > 0 else 0
        # Tight range over the last 5 days (flag / consolidation)
        recent_slice = close.iloc[-5:]
        recent_range = (recent_slice.max() - recent_slice.min()) / recent_slice.mean() if recent_slice.mean() > 0 else 1
        if prior_gain > 0.07 and recent_range < 0.04:
            patterns.append(("Bull Flag", "🚩", "high"))
        elif prior_gain > 0.04 and recent_range < 0.06:
            patterns.append(("Consolidation", "🔄", "medium"))

    # ── 3. Bollinger Band Squeeze ─────────────────────────────────────────────
    bw_window = bb_width.iloc[-60:].dropna()
    if len(bw_window) >= 20:
        current_bw = bb_width.iloc[-1]
        if not np.isnan(current_bw):
            pctile = (current_bw < bw_window).mean()
            if pctile < 0.15:
                patterns.append(("BB Squeeze", "⚡", "high"))
            elif pctile < 0.35:
                patterns.append(("Tightening", "🔃", "medium"))

    # ── 4. Golden Cross ───────────────────────────────────────────────────────
    if n >= 210:
        s50_now = sma50.iloc[-1]
        s200_now = sma200.iloc[-1]
        s50_20d = sma50.iloc[-20]
        s200_20d = sma200.iloc[-20]
        if (
            not any(np.isnan(v) for v in [s50_now, s200_now, s50_20d, s200_20d])
            and s50_now > s200_now
            and s50_20d <= s200_20d
        ):
            patterns.append(("Golden Cross", "🌟", "high"))

    # ── 5. MACD Bullish Crossover (last 5 bars) ───────────────────────────────
    macd_v = macd.dropna()
    sig_v = signal.dropna()
    if len(macd_v) >= 10 and len(sig_v) >= 10:
        for i in range(-5, -1):
            try:
                if (
                    macd_v.iloc[i] > sig_v.iloc[i]
                    and macd_v.iloc[i - 1] <= sig_v.iloc[i - 1]
                ):
                    patterns.append(("MACD Crossover", "📊", "medium"))
                    break
            except IndexError:
                pass

    # ── 6. Oversold Bounce ────────────────────────────────────────────────────
    rsi_clean = rsi.dropna()
    if len(rsi_clean) >= 10:
        rsi_last = rsi_clean.iloc[-1]
        rsi_recent_min = rsi_clean.iloc[-10:].min()
        if rsi_last > 40 and rsi_recent_min < 30:
            patterns.append(("Oversold Bounce", "↗️", "medium"))

    # ── 7. Volume Surge ───────────────────────────────────────────────────────
    vol_avg = vol_avg20.iloc[-1]
    if vol_avg > 0:
        vol_ratio = volume.iloc[-1] / vol_avg
        if vol_ratio >= 2.5:
            patterns.append(("Volume Surge", "🔊", "high"))
        elif vol_ratio >= 1.5:
            patterns.append(("Volume Pickup", "📢", "low"))

    # ── 8. Ascending Triangle ─────────────────────────────────────────────────
    if n >= 25:
        highs_20 = high.iloc[-20:].values.astype(float)
        lows_20 = low.iloc[-20:].values.astype(float)
        x = np.arange(len(highs_20), dtype=float)
        if np.std(highs_20) > 0 and np.std(lows_20) > 0:
            high_slope = np.polyfit(x, highs_20, 1)[0]
            low_slope = np.polyfit(x, lows_20, 1)[0]
            flat_thresh = 0.001 * np.mean(highs_20)
            if abs(high_slope) < flat_thresh and low_slope > 0:
                patterns.append(("Ascending Triangle", "△", "medium"))

    return patterns


# ─────────────────────────────────────────────────────────────────────────────
# Main per-stock analysis function
# ─────────────────────────────────────────────────────────────────────────────

def analyze_stock(
    ticker: str,
    df_daily: pd.DataFrame,
    df_weekly: Optional[pd.DataFrame] = None,
) -> Optional[dict]:
    """
    Full technical analysis for a single stock.

    Parameters
    ----------
    ticker : str
    df_daily : pd.DataFrame
        Daily OHLCV (auto-adjusted). Minimum 60 rows required.
    df_weekly : pd.DataFrame, optional
        Weekly OHLCV for additional trend confirmation.

    Returns
    -------
    dict with all metrics, indicators, and pattern list — or None if insufficient data.
    """
    if df_daily is None or len(df_daily) < 60:
        return None

    ind = _compute_indicators(df_daily)
    close = ind["close"]
    high = ind["high"]
    low = ind["low"]
    volume = ind["volume"]
    current = close.iloc[-1]

    if np.isnan(current) or current <= 0:
        return None

    # ── Derived values ────────────────────────────────────────────────────────
    lookback = min(252, len(close))
    week52_high = high.iloc[-lookback:].max()
    week52_low = low.iloc[-lookback:].min()
    dist_52w = (week52_high - current) / week52_high if week52_high > 0 else 1.0

    vol_avg = ind["vol_avg20"].iloc[-1]
    vol_ratio = volume.iloc[-1] / vol_avg if vol_avg > 0 else 1.0

    rsi_val = ind["rsi"].dropna()
    current_rsi = float(rsi_val.iloc[-1]) if len(rsi_val) > 0 else 50.0

    sma20_val = ind["sma20"].iloc[-1]
    sma50_val = ind["sma50"].iloc[-1]
    sma200_val = ind["sma200"].iloc[-1]

    macd_val = ind["macd"].iloc[-1]
    sig_val = ind["signal"].iloc[-1]
    hist_now = ind["histogram"].iloc[-1]
    hist_prev = ind["histogram"].iloc[-2] if len(ind["histogram"]) >= 2 else 0.0

    # ── Weekly trend confirmation ──────────────────────────────────────────────
    weekly_bullish = False
    if df_weekly is not None and len(df_weekly) >= 20:
        w_ind = _compute_indicators(df_weekly)
        w_close = w_ind["close"].iloc[-1]
        w_sma50 = w_ind["sma50"].iloc[-1]
        if not np.isnan(w_sma50) and w_close > w_sma50:
            weekly_bullish = True

    # ── Pattern detection ─────────────────────────────────────────────────────
    patterns = detect_patterns(ind)

    # ── Historical returns ─────────────────────────────────────────────────────
    returns = {}
    for days, label in [(5, "1W"), (21, "1M"), (63, "3M")]:
        if len(close) > days:
            base = close.iloc[-(days + 1)]
            if base > 0:
                returns[label] = round((current - base) / base * 100, 2)

    return {
        # Identifiers
        "ticker": ticker,
        # Price data
        "price": round(float(current), 2),
        "week52_high": round(float(week52_high), 2),
        "week52_low": round(float(week52_low), 2),
        "dist_52w": round(float(dist_52w) * 100, 2),   # % below 52W high
        # Indicator values
        "rsi": round(current_rsi, 1),
        "vol_ratio": round(float(vol_ratio), 2),
        "macd_bullish": bool(not np.isnan(macd_val) and not np.isnan(sig_val) and macd_val > sig_val),
        "hist_expanding": bool(not np.isnan(hist_now) and not np.isnan(hist_prev) and hist_now > hist_prev),
        "above_sma20": bool(not np.isnan(sma20_val) and current > sma20_val),
        "above_sma50": bool(not np.isnan(sma50_val) and current > sma50_val),
        "above_sma200": bool(not np.isnan(sma200_val) and current > sma200_val),
        "sma50_gt_sma200": bool(not np.isnan(sma50_val) and not np.isnan(sma200_val) and sma50_val > sma200_val),
        "weekly_bullish": weekly_bullish,
        # Patterns
        "patterns": patterns,
        # Returns
        "returns": returns,
        # Raw data for charting (only kept for selected stock — not in bulk table)
        "_df_daily": df_daily,
        "_df_weekly": df_weekly,
        "_ind": ind,
    }
