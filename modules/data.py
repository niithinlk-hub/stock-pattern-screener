"""
modules/data.py
Data fetching layer: stock lists + OHLCV downloads via yfinance.
All heavy network calls are cached with st.cache_data.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from io import StringIO
import warnings
warnings.filterwarnings("ignore")

# ── NSE request headers to reduce blocking probability ────────────────────────
_NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/",
}

# ── Nifty 50 fallback symbols (no .NS suffix — added later) ───────────────────
_NIFTY50_FALLBACK = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR",
    "ICICIBANK", "KOTAKBANK", "BHARTIARTL", "ITC", "BAJFINANCE",
    "SBIN", "LT", "AXISBANK", "ASIANPAINT", "DMART",
    "MARUTI", "TITAN", "WIPRO", "ULTRACEMCO", "POWERGRID",
    "NTPC", "SUNPHARMA", "HCLTECH", "ONGC", "TECHM",
    "ADANIENT", "ADANIPORTS", "COALINDIA", "JSWSTEEL", "TATAMOTORS",
    "INDUSINDBK", "TATASTEEL", "BPCL", "GRASIM", "M&M",
    "CIPLA", "EICHERMOT", "HEROMOTOCO", "DRREDDY", "NESTLEIND",
    "APOLLOHOSP", "BAJAJ-AUTO", "DIVISLAB", "BRITANNIA", "BAJAJFINSV",
    "SBILIFE", "HDFCLIFE", "UPL", "TATACONSUM", "HINDALCO",
]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_sp500_tickers() -> tuple[list[str], pd.DataFrame]:
    """
    Fetch the S&P 500 constituent list from Wikipedia.

    Returns
    -------
    tickers : list[str]
        yfinance-compatible ticker symbols (BRK.B → BRK-B).
    info_df : pd.DataFrame
        Columns: ticker, name, sector.
    """
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url, flavor=["lxml", "html5lib"])
        df = tables[0]

        # Wikipedia column names can vary slightly
        sym_col = next(c for c in df.columns if "symbol" in c.lower() or "ticker" in c.lower())
        name_col = next((c for c in df.columns if "security" in c.lower() or "name" in c.lower()), sym_col)
        sector_col = next((c for c in df.columns if "sector" in c.lower()), None)

        symbols = df[sym_col].str.replace(".", "-", regex=False).tolist()
        names = df[name_col].tolist()
        sectors = df[sector_col].tolist() if sector_col else ["N/A"] * len(symbols)

        info_df = pd.DataFrame({"ticker": symbols, "name": names, "sector": sectors})
        return symbols, info_df

    except Exception as exc:
        st.warning(f"Could not fetch S&P 500 list from Wikipedia: {exc}")
        return [], pd.DataFrame(columns=["ticker", "name", "sector"])


@st.cache_data(ttl=3600, show_spinner=False)
def get_nifty500_tickers() -> tuple[list[str], pd.DataFrame]:
    """
    Fetch the Nifty 500 constituent list from NSE archives.
    Falls back to a hardcoded Nifty 50 list if NSE blocks the request.

    Returns
    -------
    tickers : list[str]
        yfinance-compatible NSE symbols with '.NS' suffix.
    info_df : pd.DataFrame
        Columns: ticker, name, sector.
    """
    nse_url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    try:
        resp = requests.get(nse_url, headers=_NSE_HEADERS, timeout=12)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))

        # NSE CSV columns: Company Name, Industry, Symbol, Series, ISIN Code
        sym_col = next(c for c in df.columns if "symbol" in c.lower())
        name_col = next((c for c in df.columns if "company" in c.lower() or "name" in c.lower()), sym_col)
        sector_col = next((c for c in df.columns if "industry" in c.lower() or "sector" in c.lower()), None)

        symbols = df[sym_col].str.strip().tolist()
        names = df[name_col].tolist()
        sectors = df[sector_col].tolist() if sector_col else ["N/A"] * len(symbols)
        tickers = [f"{s}.NS" for s in symbols]

        info_df = pd.DataFrame({"ticker": tickers, "name": names, "sector": sectors})
        return tickers, info_df

    except Exception as exc:
        st.warning(
            f"Could not fetch Nifty 500 from NSE ({exc}). "
            "Using Nifty 50 fallback list instead."
        )
        tickers = [f"{s}.NS" for s in _NIFTY50_FALLBACK]
        info_df = pd.DataFrame({
            "ticker": tickers,
            "name": _NIFTY50_FALLBACK,
            "sector": ["N/A"] * len(tickers),
        })
        return tickers, info_df


@st.cache_data(ttl=14400, show_spinner=False)
def download_batch(
    tickers: tuple,   # tuple for cache hashability
    period: str = "1y",
    interval: str = "1d",
) -> dict[str, pd.DataFrame]:
    """
    Batch-download OHLCV data for a list of tickers using yfinance.

    Parameters
    ----------
    tickers : tuple[str]
        Must be a tuple so Streamlit can hash it for caching.
    period : str
        yfinance period string, e.g. '6mo', '1y', '2y'.
    interval : str
        yfinance interval string, e.g. '1d', '1wk'.

    Returns
    -------
    dict mapping ticker → pd.DataFrame (OHLCV, auto-adjusted).
    Tickers with < 60 valid rows are excluded.
    """
    tickers_list = list(tickers)
    if not tickers_list:
        return {}

    try:
        raw = yf.download(
            tickers=tickers_list,
            period=period,
            interval=interval,
            group_by="ticker",
            auto_adjust=True,
            threads=True,
            progress=False,
        )
    except Exception as exc:
        st.error(f"yfinance download error: {exc}")
        return {}

    result: dict[str, pd.DataFrame] = {}

    if len(tickers_list) == 1:
        # Single ticker → flat DataFrame (no MultiIndex)
        df = raw.dropna(how="all")
        if len(df) >= 60:
            result[tickers_list[0]] = df
        return result

    # Multiple tickers → MultiIndex columns
    for ticker in tickers_list:
        try:
            df = raw[ticker].dropna(how="all")
            if len(df) >= 60:
                result[ticker] = df
        except (KeyError, TypeError):
            pass

    return result
