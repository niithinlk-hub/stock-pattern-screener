"""
app.py — Stock Pattern Screener
Main Streamlit application entry point.
Deploy: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from modules.data import get_sp500_tickers, get_nifty500_tickers, download_batch
from modules.analysis import analyze_stock
from modules.scoring import calculate_score, rank_results
from modules.charts import create_price_chart, create_score_radar

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Pattern Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Header ── */
.app-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00D4AA, #2196F3);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0.2rem;
}
.app-subtitle {
    font-size: 0.95rem;
    color: #888;
    text-align: center;
    margin-bottom: 1.5rem;
}

/* ── Score bar ── */
.score-bar-wrap {
    background: #2A2D3A;
    border-radius: 5px;
    height: 10px;
    width: 100%;
    margin: 4px 0 10px 0;
}
.score-bar-fill {
    border-radius: 5px;
    height: 10px;
}

/* ── Pattern tag ── */
.pattern-tag {
    display: inline-block;
    background: rgba(0,212,170,0.15);
    border: 1px solid rgba(0,212,170,0.4);
    color: #00D4AA;
    padding: 3px 10px;
    border-radius: 14px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 3px 2px;
    white-space: nowrap;
}
.pattern-tag.high   { background: rgba(0,212,170,0.18); border-color: #00D4AA; color: #00D4AA; }
.pattern-tag.medium { background: rgba(255,152,0,0.15);  border-color: #FF9800; color: #FF9800; }
.pattern-tag.low    { background: rgba(100,100,100,0.15); border-color: #666;   color: #aaa;    }

/* ── Metric card ── */
div[data-testid="metric-container"] {
    background: #161B22;
    border: 1px solid #2A2D3A;
    border-radius: 10px;
    padding: 0.6rem 1rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
ALL_PATTERN_NAMES = [
    "52W Breakout", "Near 52W High",
    "Bull Flag", "Consolidation",
    "BB Squeeze", "Tightening",
    "Golden Cross",
    "MACD Crossover",
    "Oversold Bounce",
    "Volume Surge", "Volume Pickup",
    "Ascending Triangle",
]

_SCORE_COLOR = {
    "Strong Buy": "#00C853",
    "Buy":        "#FFB300",
    "Watch":      "#2196F3",
    "Neutral":    "#9E9E9E",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper rendering functions
# ─────────────────────────────────────────────────────────────────────────────

def _score_bar(value: float, max_val: float, color: str) -> str:
    pct = min(100, int(value / max_val * 100))
    return (
        f'<div class="score-bar-wrap">'
        f'<div class="score-bar-fill" style="width:{pct}%;background:{color};"></div>'
        f'</div>'
    )


def _pattern_tags_html(patterns: list) -> str:
    if not patterns:
        return "<span style='color:#666;font-size:0.85rem;'>No patterns detected</span>"
    return " ".join(
        f'<span class="pattern-tag {strength}">{emoji} {name}</span>'
        for name, emoji, strength in patterns
    )


def _signal_badge(signal: str, emoji: str) -> str:
    color = _SCORE_COLOR.get(signal, "#9E9E9E")
    return (
        f'<span style="background:rgba(0,0,0,0.3);border:1px solid {color};'
        f'color:{color};padding:4px 12px;border-radius:16px;font-weight:700;'
        f'font-size:0.9rem;">{emoji} {signal}</span>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def _render_sidebar() -> dict:
    with st.sidebar:
        st.markdown("## ⚙️ Screener Settings")

        market = st.selectbox(
            "Market",
            ["🇮🇳 India (NSE Nifty 500)", "🇺🇸 US (S&P 500)"],
            index=0,
        )
        st.divider()

        n_stocks = st.slider(
            "Max stocks to analyse",
            min_value=50, max_value=500, value=100, step=50,
            help="More stocks = more comprehensive but slower. 100 takes ~30 s on first run.",
        )
        period = st.selectbox("Data period", ["6mo", "1y", "2y"], index=1)

        st.divider()
        st.markdown("### 🔍 Filters")

        min_score = st.slider("Minimum score", 0, 100, 40)
        rsi_lo, rsi_hi = st.select_slider(
            "RSI range",
            options=list(range(0, 105, 5)),
            value=(30, 80),
        )
        pattern_filter = st.multiselect(
            "Must have pattern(s)",
            ALL_PATTERN_NAMES,
            default=[],
            help="Leave empty to show all patterns.",
        )

        st.divider()
        run = st.button("🚀 Run Screener", type="primary", use_container_width=True)

        st.divider()
        with st.expander("📖 Score guide", expanded=False):
            st.markdown("""
| Score | Signal |
|-------|--------|
| 75–100 | 🟢 Strong Buy |
| 60–74  | 🟡 Buy        |
| 45–59  | 🔵 Watch      |
| < 45   | ⚪ Neutral    |

**Components**
- **Trend** (30 pts): SMA alignment
- **Momentum** (25 pts): RSI + MACD
- **Volume** (20 pts): vs 20-day avg
- **Setup** (25 pts): 52W proximity + BB squeeze
- **Weekly bonus** (+5): weekly trend confirmed
""")

        with st.expander("🔎 Patterns detected", expanded=False):
            st.markdown("""
- 🚀 **52W Breakout** — within 1% of yearly high
- 📈 **Near 52W High** — within 5%
- 🚩 **Bull Flag** — strong rally + tight consolidation
- ⚡ **BB Squeeze** — Bollinger Band compression
- 🌟 **Golden Cross** — 50 SMA crossed above 200 SMA
- 📊 **MACD Crossover** — bullish signal crossover (last 5 bars)
- ↗️ **Oversold Bounce** — RSI recovering from < 30
- 🔊 **Volume Surge** — 2.5× average volume
- △ **Ascending Triangle** — rising lows, flat resistance
""")

    return {
        "market": market,
        "n_stocks": n_stocks,
        "period": period,
        "min_score": min_score,
        "rsi_range": (rsi_lo, rsi_hi),
        "pattern_filter": pattern_filter,
        "run": run,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Welcome screen
# ─────────────────────────────────────────────────────────────────────────────

def _render_welcome():
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Step 1** — Select market & settings in the sidebar")
    with col2:
        st.info("**Step 2** — Click **🚀 Run Screener** to download and analyse")
    with col3:
        st.info("**Step 3** — Review ranked results, click any stock for a detailed chart")

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Scoring System (0–100)")
        st.markdown("""
| Component | Max | What it measures |
|-----------|-----|-----------------|
| Trend | 30 | Price vs 20/50/200 SMA |
| Momentum | 25 | RSI zone + MACD direction |
| Volume | 20 | Recent volume vs 20-day avg |
| Setup | 25 | 52W high proximity + BB squeeze |
| Weekly bonus | +5 | Weekly trend confirmation |
""")

    with c2:
        st.markdown("### Patterns Detected")
        st.markdown("""
| Pattern | Signal |
|---------|--------|
| 🚀 52W Breakout | Price ≤ 1% from yearly high |
| 📈 Near 52W High | Price ≤ 5% from yearly high |
| 🚩 Bull Flag | Strong rally + tight consolidation |
| ⚡ BB Squeeze | Bollinger Bands compressed (volatility coiling) |
| 🌟 Golden Cross | 50 SMA recently crossed above 200 SMA |
| 📊 MACD Crossover | Bullish MACD crossover in last 5 bars |
| ↗️ Oversold Bounce | RSI recovering from oversold |
| 🔊 Volume Surge | Volume ≥ 2.5× 20-day average |
| △ Ascending Triangle | Rising lows + flat resistance |
""")


# ─────────────────────────────────────────────────────────────────────────────
# Core screening logic
# ─────────────────────────────────────────────────────────────────────────────

def _run_screener(settings: dict) -> list[dict] | None:
    """Download data, analyse, score, and return a list of result dicts."""
    is_india = "India" in settings["market"]
    currency = "₹" if is_india else "$"
    period = settings["period"]

    with st.status("Loading stock list…", expanded=True) as status:
        if is_india:
            st.write("Fetching Nifty 500 from NSE…")
            tickers, info_df = get_nifty500_tickers()
        else:
            st.write("Fetching S&P 500 from Wikipedia…")
            tickers, info_df = get_sp500_tickers()

        if not tickers:
            st.error("Failed to load ticker list. Check your internet connection.")
            return None

        tickers = tickers[: settings["n_stocks"]]
        st.write(f"Found **{len(tickers)}** tickers. Downloading daily data…")

        daily_data = download_batch(tuple(tickers), period=period, interval="1d")

        if not daily_data:
            st.error("No data downloaded. Please try again.")
            return None

        if period in ("1y", "2y"):
            st.write("Downloading weekly data for trend confirmation…")
            weekly_data = download_batch(tuple(list(daily_data.keys())), period=period, interval="1wk")
        else:
            weekly_data = {}

        st.write(f"Analysing **{len(daily_data)}** stocks…")
        status.update(label="Analysing stocks…", state="running")

    # Build a fast info lookup
    if not info_df.empty and "ticker" in info_df.columns:
        info_map = {row["ticker"]: row.to_dict() for _, row in info_df.iterrows()}
    else:
        info_map = {}

    results = []
    progress = st.progress(0, text="Analysing stocks…")
    total = len(daily_data)

    for i, (ticker, df_daily) in enumerate(daily_data.items()):
        df_weekly = weekly_data.get(ticker)
        analysis = analyze_stock(ticker, df_daily, df_weekly)

        if analysis is not None:
            scores = calculate_score(analysis)
            info = info_map.get(ticker, {})
            result = {
                **analysis,
                "scores": scores,
                "name": info.get("name", ticker.replace(".NS", "").replace("-", " ")),
                "sector": info.get("sector", "N/A"),
                "currency": currency,
            }
            results.append(result)

        progress.progress(
            (i + 1) / total,
            text=f"Analysing {ticker} ({i + 1}/{total})",
        )

    progress.empty()
    st.success(f"✅ Analysis complete — {len(results)} stocks processed.")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Results rendering
# ─────────────────────────────────────────────────────────────────────────────

def _render_summary(filtered: list[dict], total_analysed: int):
    strong_buys = sum(1 for r in filtered if r["scores"]["total_score"] >= 75)
    avg_score = np.mean([r["scores"]["total_score"] for r in filtered]) if filtered else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stocks Analysed", total_analysed)
    c2.metric("Matching Filters", len(filtered))
    c3.metric("Strong Buy Setups", strong_buys)
    c4.metric("Avg Score (filtered)", f"{avg_score:.1f}")


def _build_table(filtered: list[dict]) -> pd.DataFrame:
    rows = []
    for rank, r in enumerate(filtered[:100], start=1):
        sc = r["scores"]
        pattern_str = "  ".join(
            f"{p[1]} {p[0]}" for p in r["patterns"]
        ) if r["patterns"] else "—"
        rows.append({
            "Rank": rank,
            "Ticker": r["ticker"],
            "Name": r["name"][:28],
            "Sector": r["sector"],
            "Price": f"{r['currency']}{r['price']:,.2f}",
            "Score": sc["total_score"],
            "Signal": f"{sc['signal_emoji']} {sc['signal']}",
            "RSI": r["rsi"],
            "52W High %": f"-{r['dist_52w']:.1f}%",
            "Vol Ratio": f"{r['vol_ratio']:.1f}x",
            "Breakout Prob": f"{sc['breakout_prob']:.0f}%",
            "Patterns": pattern_str,
        })
    return pd.DataFrame(rows)


def _render_results_table(filtered: list[dict]):
    st.markdown("### 🏆 Top Ranked Stocks")
    if not filtered:
        st.warning("No stocks match the current filters. Try lowering the minimum score or widening the RSI range.")
        return

    df_table = _build_table(filtered)
    st.dataframe(
        df_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score": st.column_config.ProgressColumn(
                "Score", min_value=0, max_value=100, format="%.0f",
            ),
            "Breakout Prob": st.column_config.TextColumn("Breakout Prob"),
        },
    )


def _render_individual(filtered: list[dict]):
    st.markdown("---")
    st.markdown("### 🔎 Individual Stock Analysis")

    options = [r["ticker"] for r in filtered[:100]]
    if not options:
        return

    selected = st.selectbox(
        "Select a stock",
        options,
        format_func=lambda t: (
            f"{t}  —  Score: {next(r['scores']['total_score'] for r in filtered if r['ticker'] == t):.0f}"
            f"  |  {next(r['scores']['signal_emoji'] + ' ' + r['scores']['signal'] for r in filtered if r['ticker'] == t)}"
        ),
    )
    r = next(r for r in filtered if r["ticker"] == selected)
    sc = r["scores"]
    currency = r["currency"]

    # ── Top metrics ───────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Price", f"{currency}{r['price']:,.2f}")
    m2.metric("Score", f"{sc['total_score']}/100")
    m3.metric("RSI (14)", r["rsi"])
    m4.metric("Volume Ratio", f"{r['vol_ratio']:.1f}x")
    m5.metric("Breakout Prob", f"{sc['breakout_prob']}%")

    # ── Signal + patterns ─────────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 3])
    with col_left:
        st.markdown(
            _signal_badge(sc["signal"], sc["signal_emoji"]),
            unsafe_allow_html=True,
        )
    with col_right:
        st.markdown(_pattern_tags_html(r["patterns"]), unsafe_allow_html=True)

    st.markdown("---")

    # ── Score breakdown ───────────────────────────────────────────────────────
    st.markdown("#### Score Breakdown")
    sb1, sb2, sb3, sb4 = st.columns(4)
    components = [
        (sb1, "Trend",    sc["trend_score"],    30,  "#00D4AA"),
        (sb2, "Momentum", sc["momentum_score"], 25,  "#2196F3"),
        (sb3, "Volume",   sc["volume_score"],   20,  "#FF9800"),
        (sb4, "Setup",    sc["setup_score"],    25,  "#9C27B0"),
    ]
    for col, label, val, mx, color in components:
        with col:
            st.markdown(f"**{label}** &nbsp; `{val}/{mx}`")
            st.markdown(_score_bar(val, mx, color), unsafe_allow_html=True)

    if sc["weekly_bonus"] > 0:
        st.markdown(
            f"<small>🌐 Weekly trend confirmed → +{sc['weekly_bonus']} bonus pts</small>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Returns + key levels ──────────────────────────────────────────────────
    col_ret, col_lvl = st.columns(2)

    with col_ret:
        st.markdown("#### Historical Returns")
        if r["returns"]:
            rc = st.columns(len(r["returns"]))
            for col, (period_lbl, ret) in zip(rc, r["returns"].items()):
                col.metric(period_lbl, f"{ret:+.1f}%")
        else:
            st.caption("Insufficient history")

    with col_lvl:
        st.markdown("#### Key Price Levels")
        kc1, kc2, kc3 = st.columns(3)
        kc1.metric("52W High", f"{currency}{r['week52_high']:,.2f}")
        kc2.metric("52W Low",  f"{currency}{r['week52_low']:,.2f}")
        kc3.metric("From 52W High", f"-{r['dist_52w']:.1f}%")

    st.markdown("---")

    # ── Chart section ─────────────────────────────────────────────────────────
    chart_col, radar_col = st.columns([3, 1])

    with chart_col:
        st.markdown("#### Price Chart")
        timeframe = st.radio(
            "Timeframe", ["daily", "weekly"],
            horizontal=True,
            key=f"tf_{selected}",
        )
        if timeframe == "weekly" and r.get("_df_weekly") is None:
            st.info("Weekly data not available for this period selection.")
            timeframe = "daily"
        fig_price = create_price_chart(r, timeframe=timeframe)
        st.plotly_chart(fig_price, use_container_width=True)

    with radar_col:
        st.markdown("#### Score Radar")
        fig_radar = create_score_radar(sc)
        st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("**MA Alignment**")
        for label, flag in [
            ("Above 200 SMA", r["above_sma200"]),
            ("Above 50 SMA",  r["above_sma50"]),
            ("Above 20 SMA",  r["above_sma20"]),
            ("50 > 200 SMA",  r["sma50_gt_sma200"]),
            ("MACD Bullish",  r["macd_bullish"]),
        ]:
            icon = "✅" if flag else "❌"
            st.markdown(f"{icon} {label}")


def _render_export(filtered: list[dict]):
    st.markdown("---")
    export_keys = [
        "ticker", "name", "sector", "price", "rsi",
        "vol_ratio", "dist_52w", "week52_high", "week52_low",
        "above_sma200", "above_sma50", "macd_bullish",
    ]
    rows = []
    for r in filtered:
        row = {k: r[k] for k in export_keys if k in r}
        row["score"]         = r["scores"]["total_score"]
        row["signal"]        = r["scores"]["signal"]
        row["breakout_prob"] = r["scores"]["breakout_prob"]
        row["patterns"]      = " | ".join(p[0] for p in r["patterns"])
        rows.append(row)

    csv = pd.DataFrame(rows).to_csv(index=False)
    st.download_button(
        label="⬇️ Export Results (CSV)",
        data=csv,
        file_name=f"screener_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )

    st.markdown("""
> ⚠️ **Disclaimer:** This tool is for educational and informational purposes only.
> It does not constitute financial advice. Always conduct your own due diligence
> before making investment decisions.
""")


# ─────────────────────────────────────────────────────────────────────────────
# Main entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown('<p class="app-title">📈 Stock Pattern Screener</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="app-subtitle">Identify breakout setups in Indian & US markets using technical analysis</p>',
        unsafe_allow_html=True,
    )

    settings = _render_sidebar()

    if not settings["run"]:
        _render_welcome()
        return

    # Run the screener
    all_results = _run_screener(settings)
    if not all_results:
        return

    # Apply filters and rank
    filtered = rank_results(
        all_results,
        min_score=settings["min_score"],
        rsi_range=settings["rsi_range"],
        pattern_filter=settings["pattern_filter"] or None,
    )

    st.markdown("---")
    st.markdown("## 📊 Screening Results")
    _render_summary(filtered, len(all_results))
    st.markdown("---")
    _render_results_table(filtered)

    if filtered:
        _render_individual(filtered)
        _render_export(filtered)


if __name__ == "__main__":
    main()
