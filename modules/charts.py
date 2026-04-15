"""
modules/charts.py
Plotly chart builders for the stock screener.
All charts use the dark 'plotly_dark' template.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_BG = "#0E1117"
_GRID = "#1E2530"
_GREEN = "#26A69A"
_RED = "#EF5350"
_ORANGE = "#FF9800"
_BLUE = "#2196F3"
_PURPLE = "#9C27B0"
_TEAL = "#00D4AA"


# ─────────────────────────────────────────────────────────────────────────────
# Price chart (candlestick + volume + RSI + MACD)
# ─────────────────────────────────────────────────────────────────────────────

def create_price_chart(analysis: dict, timeframe: str = "daily") -> go.Figure:
    """
    Build a 4-panel Plotly chart:
      Row 1 (50%): Candlestick + SMA20/50/200 + Bollinger Bands
      Row 2 (15%): Volume bars (coloured by candle direction)
      Row 3 (17.5%): RSI with overbought/oversold bands
      Row 4 (17.5%): MACD histogram + lines

    Parameters
    ----------
    analysis : dict returned by analyze_stock()
    timeframe : 'daily' | 'weekly'
    """
    if timeframe == "weekly" and analysis.get("_df_weekly") is not None:
        df = analysis["_df_weekly"]
        from modules.analysis import _compute_indicators
        ind = _compute_indicators(df)
        title_suffix = "Weekly"
    else:
        df = analysis["_df_daily"]
        ind = analysis["_ind"]
        title_suffix = "Daily"

    display_days = min(252, len(df))
    df = df.iloc[-display_days:]
    close = ind["close"].iloc[-display_days:]
    high = ind["high"].iloc[-display_days:]
    low = ind["low"].iloc[-display_days:]
    volume = ind["volume"].iloc[-display_days:]
    sma20 = ind["sma20"].iloc[-display_days:]
    sma50 = ind["sma50"].iloc[-display_days:]
    sma200 = ind["sma200"].iloc[-display_days:]
    bb_upper = ind["bb_upper"].iloc[-display_days:]
    bb_lower = ind["bb_lower"].iloc[-display_days:]
    rsi = ind["rsi"].iloc[-display_days:]
    macd = ind["macd"].iloc[-display_days:]
    signal_line = ind["signal"].iloc[-display_days:]
    histogram = ind["histogram"].iloc[-display_days:]

    # Open column might not exist after auto_adjust on weekly resample;
    # fall back to close.shift(1) for candle colouring
    if "Open" in df.columns:
        opens = df["Open"].iloc[-display_days:]
    else:
        opens = close.shift(1).fillna(close)

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.50, 0.15, 0.175, 0.175],
    )

    # ── Row 1: Candlestick ────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=opens,
        high=high,
        low=low,
        close=close,
        name="Price",
        increasing_line_color=_GREEN,
        decreasing_line_color=_RED,
        increasing_fillcolor=_GREEN,
        decreasing_fillcolor=_RED,
        line=dict(width=1),
    ), row=1, col=1)

    # Bollinger Bands (filled area)
    fig.add_trace(go.Scatter(
        x=df.index, y=bb_upper,
        line=dict(color="rgba(173,216,230,0.4)", width=1),
        name="BB Upper", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=bb_lower,
        line=dict(color="rgba(173,216,230,0.4)", width=1),
        fill="tonexty", fillcolor="rgba(173,216,230,0.06)",
        name="BB Lower", showlegend=False,
    ), row=1, col=1)

    # Moving averages
    fig.add_trace(go.Scatter(
        x=df.index, y=sma20,
        line=dict(color=_ORANGE, width=1.5), name="SMA 20",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=sma50,
        line=dict(color=_BLUE, width=1.5), name="SMA 50",
    ), row=1, col=1)
    if sma200.dropna().shape[0] > 0:
        fig.add_trace(go.Scatter(
            x=df.index, y=sma200,
            line=dict(color=_PURPLE, width=1.5), name="SMA 200",
        ), row=1, col=1)

    # ── Row 2: Volume ─────────────────────────────────────────────────────────
    vol_colors = [
        _GREEN if c >= o else _RED
        for c, o in zip(close, opens)
    ]
    fig.add_trace(go.Bar(
        x=df.index, y=volume,
        name="Volume",
        marker_color=vol_colors,
        marker_line_width=0,
        opacity=0.7,
        showlegend=False,
    ), row=2, col=1)

    # ── Row 3: RSI ────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df.index, y=rsi,
        line=dict(color=_TEAL, width=2), name="RSI",
    ), row=3, col=1)
    for level, colour in [(70, "rgba(239,83,80,0.5)"), (50, "rgba(255,255,255,0.2)"), (30, "rgba(38,166,154,0.5)")]:
        fig.add_hline(y=level, line_dash="dash", line_color=colour, row=3, col=1)

    # RSI overbought / oversold shading
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,83,80,0.07)", line_width=0, row=3, col=1)
    fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(38,166,154,0.07)", line_width=0, row=3, col=1)

    # ── Row 4: MACD ───────────────────────────────────────────────────────────
    hist_colors = [_GREEN if v >= 0 else _RED for v in histogram.fillna(0)]
    fig.add_trace(go.Bar(
        x=df.index, y=histogram,
        name="MACD Hist",
        marker_color=hist_colors,
        marker_line_width=0,
        opacity=0.7,
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=macd,
        line=dict(color=_BLUE, width=1.5), name="MACD",
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=signal_line,
        line=dict(color=_ORANGE, width=1.5), name="Signal",
    ), row=4, col=1)

    # ── Layout ────────────────────────────────────────────────────────────────
    ticker = analysis["ticker"]
    fig.update_layout(
        title=dict(
            text=f"{ticker} — {title_suffix} Chart",
            font=dict(size=15, color="#FAFAFA"),
        ),
        xaxis_rangeslider_visible=False,
        height=700,
        template="plotly_dark",
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        showlegend=True,
        legend=dict(
            orientation="h", y=1.02, x=0,
            font=dict(size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=50, r=20, t=60, b=30),
    )

    # Axis styling
    axis_style = dict(
        gridcolor=_GRID,
        zerolinecolor=_GRID,
        tickfont=dict(size=11),
    )
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    fig.update_yaxes(title_text="Price",  row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI",    row=3, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD",   row=4, col=1)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Score radar chart
# ─────────────────────────────────────────────────────────────────────────────

def create_score_radar(scores: dict) -> go.Figure:
    """
    Spider / radar chart showing the 4 scoring components as a percentage
    of their respective maximums.

    Parameters
    ----------
    scores : dict returned by calculate_score()
    """
    categories = ["Trend", "Momentum", "Volume", "Setup"]
    maxima = [30, 25, 20, 25]
    values = [
        scores["trend_score"],
        scores["momentum_score"],
        scores["volume_score"],
        scores["setup_score"],
    ]
    # Normalise to 0–100 %
    normalised = [round(v / m * 100, 1) for v, m in zip(values, maxima)]
    # Close the polygon
    normalised_closed = normalised + [normalised[0]]
    categories_closed = categories + [categories[0]]

    fig = go.Figure()

    # Fill area
    fig.add_trace(go.Scatterpolar(
        r=normalised_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor=f"rgba(0,212,170,0.20)",
        line=dict(color=_TEAL, width=2),
        marker=dict(color=_TEAL, size=8),
        name="Score",
    ))

    # Axis labels with raw points
    fig.add_trace(go.Scatterpolar(
        r=[0] * len(categories),
        theta=categories,
        mode="text",
        text=[f"{v}/{m}" for v, m in zip(values, maxima)],
        textposition="top center",
        textfont=dict(size=11, color="#FAFAFA"),
        showlegend=False,
    ))

    fig.update_layout(
        polar=dict(
            bgcolor=_BG,
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                ticksuffix="%",
                tickfont=dict(size=9, color="#888"),
                gridcolor=_GRID,
                linecolor=_GRID,
            ),
            angularaxis=dict(
                tickfont=dict(size=13, color="#FAFAFA"),
                gridcolor=_GRID,
                linecolor=_GRID,
            ),
        ),
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        template="plotly_dark",
        showlegend=False,
        height=320,
        margin=dict(l=40, r=40, t=30, b=30),
    )
    return fig
