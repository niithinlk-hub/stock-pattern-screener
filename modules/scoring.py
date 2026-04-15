"""
modules/scoring.py
Composite scoring, signal labelling, breakout probability estimate, and ranking.
"""

from __future__ import annotations
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Scoring constants (max points per component)
# ─────────────────────────────────────────────────────────────────────────────
_TREND_MAX     = 30
_MOMENTUM_MAX  = 25
_VOLUME_MAX    = 20
_SETUP_MAX     = 25
_WEEKLY_BONUS  = 5
_TOTAL_MAX     = _TREND_MAX + _MOMENTUM_MAX + _VOLUME_MAX + _SETUP_MAX + _WEEKLY_BONUS  # 105 → capped at 100


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def calculate_score(analysis: dict) -> dict:
    """
    Calculate the composite technical score for a single stock analysis dict.

    Scoring breakdown (max 100):
      Trend     (30): SMA alignment
      Momentum  (25): RSI zone + MACD status
      Volume    (20): Recent volume vs 20-day average
      Setup     (25): Proximity to 52W high + BB squeeze
      Weekly bonus (+5, capped at 100): weekly trend confirmed

    Returns
    -------
    dict with keys:
        trend_score, momentum_score, volume_score, setup_score,
        total_score, breakout_prob, signal, signal_emoji
    """
    # ── 1. Trend Score (0–30) ─────────────────────────────────────────────────
    trend = 0
    if analysis["above_sma200"]:      trend += 12
    if analysis["above_sma50"]:       trend += 9
    if analysis["above_sma20"]:       trend += 5
    if analysis["sma50_gt_sma200"]:   trend += 4

    # ── 2. Momentum Score (0–25) ──────────────────────────────────────────────
    momentum = 0
    rsi = analysis["rsi"]
    if 55 <= rsi <= 70:          momentum += 15   # sweet spot
    elif 70 < rsi <= 80:         momentum += 10   # strong but stretching
    elif 45 <= rsi < 55:         momentum += 7    # neutral / recovery
    elif rsi > 80:               momentum += 4    # overbought
    else:                        momentum += 2    # bearish / weak

    if analysis["macd_bullish"]:       momentum += 7
    if analysis["hist_expanding"]:     momentum += 3

    # ── 3. Volume Score (0–20) ────────────────────────────────────────────────
    vr = analysis["vol_ratio"]
    if vr >= 2.0:      volume_score = 20
    elif vr >= 1.5:    volume_score = 15
    elif vr >= 1.2:    volume_score = 10
    elif vr >= 1.0:    volume_score = 5
    else:              volume_score = 2

    # ── 4. Setup / Breakout Score (0–25) ──────────────────────────────────────
    setup = 0

    # Proximity to 52-week high
    dist = analysis["dist_52w"]   # % below 52W high (lower = better)
    if dist <= 2.0:      setup += 15
    elif dist <= 5.0:    setup += 10
    elif dist <= 10.0:   setup += 5

    # BB squeeze percentile — extracted from patterns list
    # We re-read bb_width from the stored indicator dict
    bb_setup = _bb_squeeze_score(analysis)
    setup += bb_setup

    # ── 5. Weekly Bonus (0–5) ─────────────────────────────────────────────────
    weekly_bonus = _WEEKLY_BONUS if analysis.get("weekly_bullish", False) else 0

    # ── Total ─────────────────────────────────────────────────────────────────
    raw_total = trend + momentum + volume_score + setup + weekly_bonus
    total = min(100, round(raw_total, 1))

    # ── Breakout Probability (heuristic 0–100 %) ──────────────────────────────
    prob = (
        0.40 * (trend / _TREND_MAX)
        + 0.30 * (setup / _SETUP_MAX)
        + 0.20 * (momentum / _MOMENTUM_MAX)
        + 0.10 * (volume_score / _VOLUME_MAX)
    )
    breakout_prob = round(min(100.0, prob * 100), 1)

    # ── Signal Label ──────────────────────────────────────────────────────────
    if total >= 75:
        signal, emoji = "Strong Buy", "🟢"
    elif total >= 60:
        signal, emoji = "Buy", "🟡"
    elif total >= 45:
        signal, emoji = "Watch", "🔵"
    else:
        signal, emoji = "Neutral", "⚪"

    return {
        "trend_score":    trend,
        "momentum_score": momentum,
        "volume_score":   volume_score,
        "setup_score":    setup,
        "weekly_bonus":   weekly_bonus,
        "total_score":    total,
        "breakout_prob":  breakout_prob,
        "signal":         signal,
        "signal_emoji":   emoji,
    }


def rank_results(
    results: list[dict],
    min_score: float = 0,
    rsi_range: tuple[int, int] = (0, 100),
    pattern_filter: list[str] | None = None,
) -> list[dict]:
    """
    Filter and rank a list of scored stock dicts.

    Parameters
    ----------
    results : list of dicts that each contain a 'scores' sub-dict.
    min_score : float
        Minimum total_score to include.
    rsi_range : (low, high)
        Inclusive RSI range filter.
    pattern_filter : list[str] or None
        If non-empty, only include stocks with at least one matching pattern name.

    Returns
    -------
    Sorted list (descending by total_score), filtered.
    """
    filtered = []
    for r in results:
        score = r["scores"]["total_score"]
        rsi = r["rsi"]

        if score < min_score:
            continue
        if not (rsi_range[0] <= rsi <= rsi_range[1]):
            continue
        if pattern_filter:
            pattern_names = [p[0] for p in r["patterns"]]
            if not any(pf in pattern_names for pf in pattern_filter):
                continue
        filtered.append(r)

    filtered.sort(key=lambda x: x["scores"]["total_score"], reverse=True)
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bb_squeeze_score(analysis: dict) -> int:
    """
    Return setup points (0, 6, or 10) based on the BB squeeze percentile
    derived from the stored indicator dict.
    """
    ind = analysis.get("_ind")
    if ind is None:
        return 2   # default if no indicator data

    bb_width = ind.get("bb_width")
    if bb_width is None:
        return 2

    window = bb_width.iloc[-60:].dropna()
    if len(window) < 20:
        return 2

    current_bw = bb_width.iloc[-1]
    if np.isnan(current_bw):
        return 2

    pctile = float((current_bw < window).mean())
    if pctile < 0.15:    return 10
    elif pctile < 0.35:  return 6
    else:                return 2
